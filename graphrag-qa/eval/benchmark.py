import json
import sys
import os
from dotenv import load_dotenv
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, "..", "backend", ".env"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "backend"))

from retrieval.vector_search import vector_search
from retrieval.graph_traversal import expand_one_hop
from retrieval.context_builder import build_context
from llm.gemini import stream_answer
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0))
COLLECTION_NAME = "pallets_flask"

def safe_stream(question: str, context: str):
    # Wrapped with exponential backoff
    for attempt in range(3):
        try:
            return "".join(stream_answer(question, context))
        except Exception as e:
            if attempt == 2: raise e
            time.sleep(2 ** attempt)

def run_graphrag(question: str) -> dict:
    seed = vector_search(question, COLLECTION_NAME)
    expanded = expand_one_hop(seed, COLLECTION_NAME)
    context = build_context(seed, expanded)
    return {"answer": safe_stream(question, context), "contexts": [c["source_code"] for c in seed + expanded]}

def run_naive_rag(question: str) -> dict:
    chunks = vector_search(question, COLLECTION_NAME)
    context = build_context(chunks, [])
    return {"answer": safe_stream(question, context), "contexts": [c["source_code"] for c in chunks]}

def run_agentic_rag(question: str) -> dict:
    """Run the full multi-agent LangGraph retrieval pipeline."""
    try:
        from llm.agent import run_agentic_retrieval
        context = run_agentic_retrieval(question, COLLECTION_NAME)
        answer = safe_stream(question, context)
        # We don't have the raw chunk list from the agent, so use the context string
        return {"answer": answer, "contexts": [context]}
    except Exception as e:
        print(f"Agentic RAG error: {e}")
        return {"answer": f"Error: {e}", "contexts": [""]}

def build_dataset(questions: list[dict], pipeline_fn) -> Dataset:
    rows = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['question'][:70]}...")
        result = pipeline_fn(q["question"])
        rows["user_input"].append(q["question"])
        rows["response"].append(result["answer"])
        rows["retrieved_contexts"].append(result["contexts"])
        rows["reference"].append(q["reference"])
    return Dataset.from_dict(rows)

if __name__ == "__main__":
    with open(os.path.join(SCRIPT_DIR, "questions.json")) as f:
        questions = json.load(f)

    metrics = [context_precision, answer_relevancy]

    print("\nEvaluating Naive RAG...")
    naive_ds = build_dataset(questions, run_naive_rag)
    naive_scores = evaluate(naive_ds, metrics=metrics, llm=evaluator_llm)

    print("\nEvaluating GraphRAG...")
    graph_ds = build_dataset(questions, run_graphrag)
    graph_scores = evaluate(graph_ds, metrics=metrics, llm=evaluator_llm)

    print("\nEvaluating Agentic RAG...")
    agentic_ds = build_dataset(questions, run_agentic_rag)
    agentic_scores = evaluate(agentic_ds, metrics=metrics, llm=evaluator_llm)

    print("\n" + "=" * 80)
    print(f"{'Metric':<28} {'Naive RAG':>12} {'GraphRAG':>12} {'Agentic':>12} {'Best Delta':>12}")
    print("-" * 80)
    
    naive_df = naive_scores.to_pandas()
    graph_df = graph_scores.to_pandas()
    agentic_df = agentic_scores.to_pandas()

    for metric in ["context_precision", "answer_relevancy"]:
        n_val = naive_df[metric].mean()
        g_val = graph_df[metric].mean()
        a_val = agentic_df[metric].mean()
        best = max(g_val, a_val)
        delta = best - n_val
        pct = (delta / max(n_val, 0.001)) * 100
        print(f"{metric:<28} {n_val:>12.3f} {g_val:>12.3f} {a_val:>12.3f} {delta:>+10.3f} ({pct:+.0f}%)")
