import logging
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from llm.gemini import structured_model, safe_stream_answer
from retrieval.vector_search import vector_search
from retrieval.graph_traversal import expand_one_hop
from retrieval.context_builder import build_context

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    question: str
    collection_name: str
    context_chunks: list[dict]
    search_history: list[str]
    iterations: int
    next_action: Literal["retrieve", "end"]

class SearchQuery(BaseModel):
    query: str = Field(description="The next search query to run against the codebase")

class CriticDecision(BaseModel):
    is_sufficient: bool = Field(description="True if the context is sufficient to answer the question completely")

planner_model = structured_model.with_structured_output(SearchQuery)
critic_model = structured_model.with_structured_output(CriticDecision)

def planner_node(state: AgentState):
    history = state.get("search_history", [])
    prompt = f"""You are a codebase search planner.
User Question: {state['question']}
Previous searches you tried: {history}

Provide a new, highly specific search query to execute against the codebase vector database to find the missing information."""
    
    try:
        res = planner_model.invoke([HumanMessage(content=prompt)])
        query = res.query
    except Exception as e:
        logger.warning(f"Planner LLM call failed, using question as fallback query: {e}")
        query = state['question']
    
    logger.info(f"[Planner] Generated search query: {query}")
    return {"search_history": history + [query]}

def retriever_node(state: AgentState):
    query = state["search_history"][-1]
    logger.info(f"[Retriever] Searching for: {query}")
    
    seed = vector_search(query, state["collection_name"])
    expanded = expand_one_hop(seed, state["collection_name"])
    
    current_chunks = state.get("context_chunks", [])
    existing_ids = {c["id"] for c in current_chunks}
    
    new_chunks = []
    for c in (seed + expanded):
        if c["id"] not in existing_ids:
            new_chunks.append(c)
            existing_ids.add(c["id"])
            
    logger.info(f"[Retriever] Added {len(new_chunks)} new unique chunks to memory.")
    return {"context_chunks": current_chunks + new_chunks}

def critic_node(state: AgentState):
    iterations = state.get("iterations", 0) + 1
    
    if iterations >= 3:
        logger.info(f"[Critic] Max iterations reached. Proceeding to generation.")
        return {"iterations": iterations, "next_action": "end"}
        
    context_str = build_context(state.get("context_chunks", []), [])
    
    prompt = f"""Evaluate if the following context contains enough information to fully answer the user's question.
Question: {state['question']}
Context: {context_str}

Is this context sufficient? Reply True if sufficient, False if more searching is needed."""
    
    try:
        res = critic_model.invoke([HumanMessage(content=prompt)])
        is_sufficient = res.is_sufficient
    except Exception as e:
        logger.warning(f"Critic LLM call failed, defaulting to end: {e}")
        is_sufficient = True
    
    if is_sufficient:
        logger.info(f"[Critic] Context is sufficient. Proceeding to generation.")
        action = "end"
    else:
        logger.info(f"[Critic] Context insufficient. Requesting more searches. (Iteration {iterations}/3)")
        action = "retrieve"
        
    return {"iterations": iterations, "next_action": action}

def should_continue(state: AgentState):
    return state["next_action"]

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("critic", critic_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "critic")
workflow.add_conditional_edges("critic", should_continue, {"retrieve": "planner", "end": END})

agent_app = workflow.compile()

def run_agentic_retrieval(question: str, collection_name: str) -> str:
    initial_state = {
        "question": question,
        "collection_name": collection_name,
        "context_chunks": [],
        "search_history": [],
        "iterations": 0
    }
    
    final_state = agent_app.invoke(initial_state)
    
    # Build the final massive context string
    return build_context(final_state["context_chunks"], [])

def run_agent_with_chunks(question: str, collection_name: str) -> tuple[str, list[dict]]:
    """Run the agentic retrieval and return both the answer and raw chunks.
    
    Used by the benchmark to evaluate context_precision on individual chunks
    instead of a single concatenated string.
    """
    initial_state = {
        "question": question,
        "collection_name": collection_name,
        "context_chunks": [],
        "search_history": [],
        "iterations": 0
    }
    
    final_state = agent_app.invoke(initial_state)
    chunks = final_state["context_chunks"]
    context = build_context(chunks, [])
    answer = safe_stream_answer(question, context)
    return answer, chunks