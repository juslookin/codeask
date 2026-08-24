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
from llm.gemini import safe_stream_answer
from llm.agent import run_agent_with_chunks
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, answer_relevancy, answer_correctness
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
import pandas as pd

# Delay between benchmark rows (build_dataset), on top of Ragas's own max_retries/
# max_wait backoff. Tune via env var: raise it if you're still on the Gemini free
# tier (15 RPM / 500 RPD for gemini-3.5-flash-lite); once billing is enabled
# (Tier 1) this can safely drop toward 0. See graphrag-qa/README.md.
EVAL_ROW_DELAY = float(os.getenv("EVAL_RATE_LIMIT_DELAY", "4.0"))

evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
)
COLLECTION_NAME = "pallets_flask"
# NOTE: "models/embedding-001" is deprecated/retired — use gemini-embedding-001.
evaluator_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")))

def run_naive_rag(question: str) -> dict:
    chunks = vector_search(question, COLLECTION_NAME)
    context = build_context(chunks, [])
    return {"answer": safe_stream_answer(question, context), "contexts": [c["source_code"] for c in chunks]}

def run_graphrag(question: str) -> dict:
    seed = vector_search(question, COLLECTION_NAME)
    expanded = expand_one_hop(seed, COLLECTION_NAME)
    context = build_context(seed, expanded)
    return {"answer": safe_stream_answer(question, context), "contexts": [c["source_code"] for c in seed + expanded]}

def run_agentic_rag(question: str) -> dict:
    """Run the full multi-agent LangGraph retrieval pipeline.
    
    Uses run_agent_with_chunks to get individual chunks for proper
    context_precision scoring (not a single concatenated string).
    """
    try:
        answer, chunks = run_agent_with_chunks(question, COLLECTION_NAME)
        return {"answer": answer, "contexts": [c["source_code"] for c in chunks]}
    except Exception as e:
        print(f"Agentic RAG error: {e}")
        return {"answer": f"Error: {e}", "contexts": [""]}

def build_dataset(questions: list[dict], pipeline_fn, name: str) -> Dataset:
    checkpoint_file = os.path.join(SCRIPT_DIR, f"checkpoint_{name}.json")
    rows = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                rows = json.load(f)
        except Exception:
            pass # fallback to empty if corrupted
            
    start_idx = len(rows.get("user_input", []))
    if start_idx >= len(questions):
        print(f"  [+] Loaded completed dataset '{name}' from checkpoint.")
        return Dataset.from_dict(rows)
        
    for i in range(start_idx, len(questions)):
        q = questions[i]
        print(f"  [{i+1}/{len(questions)}] {q['question'][:70]}...")
        
        # Robust retry loop in case internet drops during generation
        for attempt in range(5):
            try:
                result = pipeline_fn(q["question"])
                break
            except Exception as e:
                print(f"    Network Error: {e}. Retrying in 10s...")
                time.sleep(10)
        else:
            result = pipeline_fn(q["question"]) # final attempt, will crash if fails
            
        rows["user_input"].append(q["question"])
        rows["response"].append(result["answer"])
        rows["retrieved_contexts"].append(result["contexts"])
        rows["reference"].append(q["reference"])
        
        # Save progress row-by-row
        with open(checkpoint_file, "w") as f:
            json.dump(rows, f, indent=2)
            
        time.sleep(EVAL_ROW_DELAY)
        
    return Dataset.from_dict(rows)

if __name__ == "__main__":
    with open(os.path.join(SCRIPT_DIR, "questions.json")) as f:
        questions = json.load(f)

    from ragas.metrics import faithfulness
    # answer_correctness scores the answer against `reference` (factual
    # correctness + semantic similarity) — faithfulness alone only checks
    # the answer doesn't contradict retrieved context, not that it's right.
    metrics = [context_precision, faithfulness, answer_correctness]

    # max_retries increased to 10 to survive Gemini's 15 RPM free tier limit
    run_config = RunConfig(max_workers=1, max_retries=10, max_wait=120)
    
    print("\nEvaluating Naive RAG...")
    naive_ds = build_dataset(questions, run_naive_rag, "naive")
    naive_scores = evaluate(naive_ds, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings, run_config=run_config)

    print("\nEvaluating GraphRAG...")
    graph_ds = build_dataset(questions, run_graphrag, "graph")
    graph_scores = evaluate(graph_ds, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings, run_config=run_config)

    print("\nEvaluating Agentic RAG...")
    agentic_ds = build_dataset(questions, run_agentic_rag, "agentic")
    agentic_scores = evaluate(agentic_ds, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings, run_config=run_config)

    print("\n" + "=" * 80)
    print(f"{'Metric':<28} {'Naive RAG':>12} {'GraphRAG':>12} {'Agentic':>12} {'Best Delta':>12}")
    print("-" * 80)
    
    naive_df = naive_scores.to_pandas()
    graph_df = graph_scores.to_pandas()
    agentic_df = agentic_scores.to_pandas()

    for metric in ["context_precision", "faithfulness", "answer_correctness"]:
        n_val = naive_df[metric].mean()
        g_val = graph_df[metric].mean()
        a_val = agentic_df[metric].mean()
        best = max(g_val, a_val)
        delta = best - n_val
        pct = (delta / max(n_val, 0.001)) * 100
        print(f"{metric:<28} {n_val:>12.3f} {g_val:>12.3f} {a_val:>12.3f} {delta:>+10.3f} ({pct:+.0f}%)")