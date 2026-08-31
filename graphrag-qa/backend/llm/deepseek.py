from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage
import os
import time
from dotenv import load_dotenv

# Ensure .env is loaded from backend directory regardless of process working directory
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BACKEND_DIR, ".env"))
load_dotenv()

DEEPSEEK_MODEL = "deepseek-v4-flash"
# DeepSeek current model names (deepseek-chat / deepseek-reasoner were retired 2026-07-24).
# deepseek-v4-flash — non-thinking mode, fast + cheap, use for generation and structured calls.
# deepseek-v4-pro   — higher quality at higher cost, swap in if flash underperforms.
# See https://platform.deepseek.com/docs for rate limits and model names.

# Used for final answer generation — some creative variation in phrasing is fine here.
model = ChatDeepSeek(
    model=DEEPSEEK_MODEL,
    temperature=0.3,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# Used for structured/classification calls (agent planner + critic in llm/agent.py)
# where we want deterministic, repeatable decisions rather than creative variation.
# temperature=0.3 on a True/False "is context sufficient?" classifier would make the
# agent's iteration count — and therefore the benchmark numbers — vary run to run.
structured_model = ChatDeepSeek(
    model=DEEPSEEK_MODEL,
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

SYSTEM_PROMPT = """You are an expert code analysis assistant. Answer questions using ONLY the provided code chunks.

Rules:
1. If the question asks about a sequence, lifecycle, or "what happens when," answer with steps in execution order.
2. Be precise — cite specific function names, parameters, and return values from the code.
3. If the code chunks don't contain enough information, say so explicitly.
4. End every response with a ## Citations section.
5. Format citations as: `filepath:startline-endline`
"""

def stream_answer(question: str, context: str):
    """Streams the answer chunk by chunk — used by the live /query endpoint."""
    prompt = f"{SYSTEM_PROMPT}\n\n## Code Context\n{context}\n\n## Question\n{question}\n"
    for chunk in model.stream([HumanMessage(content=prompt)]):
        content = chunk.content
        if content:
            if isinstance(content, list):
                yield "".join(c.get("text", "") for c in content if isinstance(c, dict))
            else:
                yield str(content)

def safe_stream_answer(question: str, context: str, retries: int = 3) -> str:
    """stream_answer wrapped with exponential backoff, collected into one string.

    Shared by every pipeline in the benchmark harness (naive, graph, AND agentic)
    so a single transient error doesn't silently poison one row's answer with
    "Error: ..." while the other two pipelines retry and succeed — which would
    otherwise make the pipeline comparison unfair.
    """
    for attempt in range(retries):
        try:
            return "".join(stream_answer(question, context))
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(2 ** attempt)
