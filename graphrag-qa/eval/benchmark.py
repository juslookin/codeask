import json
import sys
import os
import asyncio
from dotenv import load_dotenv
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, "..", "backend", ".env"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "backend"))

from retrieval.vector_search import vector_search
from retrieval.graph_traversal import expand_one_hop
from retrieval.context_builder import build_context
from retrieval.selector import select_top_k
from llm.deepseek import safe_stream_answer
from llm.agent import run_agent_with_chunks
from datasets import Dataset
from ragas import evaluate
from openai import AsyncOpenAI
from ragas.metrics import ContextPrecision, AnswerCorrectness, Faithfulness
from ragas.llms import llm_factory
from google import genai
from ragas.embeddings import GoogleEmbeddings
from ragas.run_config import RunConfig
import pandas as pd

# Delay between benchmark rows (build_dataset), on top of Ragas's own max_retries/
# max_wait backoff. The old 4.0 s default was tuned for Gemini's free-tier 15 RPM
# ceiling — not needed on DeepSeek's paid tier. Tune via env var if you hit limits.
EVAL_ROW_DELAY = float(os.getenv("EVAL_RATE_LIMIT_DELAY", "0"))

_deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=60.0,
    max_retries=3
)

evaluator_llm = llm_factory(
    model="deepseek-v4-flash",
    provider="openai",
    client=_deepseek_client,
    temperature=0,
    max_tokens=8192,
    extra_body={"thinking": {"type": "disabled"}}
)

COLLECTION_NAME = "pallets_flask"

# Ragas 0.4.3 native GoogleEmbeddings handles retries natively
_gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
evaluator_embeddings = GoogleEmbeddings(
    client=_gemini_client,
    model="gemini-embedding-2"
)

def run_naive_rag(question: str) -> dict:
    chunks, _ = vector_search(question, COLLECTION_NAME)
    context = build_context(chunks, [])
    return {"answer": safe_stream_answer(question, context), "contexts": [c["source_code"] for c in chunks]}

def run_graphrag(question: str) -> dict:
    seed, query_emb = vector_search(question, COLLECTION_NAME)
    expanded = expand_one_hop(seed, COLLECTION_NAME)
    chunks = select_top_k(question, seed, expanded,
                          query_embedding=query_emb,
                          collection_name=COLLECTION_NAME)
    context = build_context(chunks, [])
    return {"answer": safe_stream_answer(question, context), "contexts": [c["source_code"] for c in chunks]}

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
    rows = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": [],
            "latency_s": [], "chunk_count": []}
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                rows = json.load(f)
            # Back-fill keys missing from older checkpoints so we don't crash on resume
            rows.setdefault("latency_s", [None] * len(rows.get("user_input", [])))
            rows.setdefault("chunk_count", [None] * len(rows.get("user_input", [])))
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
        t0 = time.time()
        for attempt in range(5):
            try:
                result = pipeline_fn(q["question"])
                break
            except Exception as e:
                print(f"    Network Error: {e}. Retrying in 10s...")
                time.sleep(10)
        else:
            result = pipeline_fn(q["question"]) # final attempt, will crash if fails
        latency = round(time.time() - t0, 2)
            
        rows["user_input"].append(q["question"])
        rows["response"].append(result["answer"])
        rows["retrieved_contexts"].append(result["contexts"])
        rows["reference"].append(q["reference"])
        rows["latency_s"].append(latency)
        rows["chunk_count"].append(len(result["contexts"]))
        
        # Save progress row-by-row
        with open(checkpoint_file, "w") as f:
            json.dump(rows, f, indent=2)
            
        time.sleep(EVAL_ROW_DELAY)
        
    return Dataset.from_dict(rows)

if __name__ == "__main__":
    with open(os.path.join(SCRIPT_DIR, "questions.json")) as f:
        questions = json.load(f)

    # answer_correctness scores the answer against `reference` (factual
    # correctness + semantic similarity) — faithfulness alone only checks
    # the answer doesn't contradict retrieved context, not that it's right.
    metrics = [
        ContextPrecision(),
        Faithfulness(),
        AnswerCorrectness()
    ]

    # max_workers=4: non-blocking async coroutines run cleanly with zero thread locks.
    # timeout=240: gives 8-chunk Agentic Faithfulness jobs full time (~2-3 mins) to verify all statements.
    run_config = RunConfig(max_workers=4, max_retries=3, max_wait=120, timeout=240)
    
    def get_evaluation_df(dataset, name):
        checkpoint_scores_file = os.path.join(SCRIPT_DIR, f"checkpoint_scores_{name}.csv")
        if os.path.exists(checkpoint_scores_file):
            print(f"  [+] Loaded completed evaluation scores for '{name}' from checkpoint.")
            return pd.read_csv(checkpoint_scores_file)
        
        scores = evaluate(dataset, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings, run_config=run_config)
        df = scores.to_pandas()
        df.to_csv(checkpoint_scores_file, index=False)
        print(f"  [+] Saved evaluation scores for '{name}' to checkpoint.")
        return df

    print("\nEvaluating Naive RAG...")
    naive_ds = build_dataset(questions, run_naive_rag, "naive")
    naive_df = get_evaluation_df(naive_ds, "naive")

    print("\nEvaluating GraphRAG...")
    graph_ds = build_dataset(questions, run_graphrag, "graph")
    graph_df = get_evaluation_df(graph_ds, "graph")

    print("\nEvaluating Agentic RAG...")
    agentic_ds = build_dataset(questions, run_agentic_rag, "agentic")
    agentic_df = get_evaluation_df(agentic_ds, "agentic")

    print("\n" + "=" * 95)
    print(f"{'Metric':<35} {'Naive RAG':>12} {'GraphRAG':>12} {'Agentic':>12} {'Best Delta':>12}")
    print("-" * 95)

    # Merge latency/chunk columns from build_dataset into score DataFrames
    for pipeline_name, score_df, dataset in [
        ("naive", naive_df, naive_ds),
        ("graph", graph_df, graph_ds),
        ("agentic", agentic_df, agentic_ds),
    ]:
        ds_df = dataset.to_pandas()
        for col in ["latency_s", "chunk_count"]:
            if col in ds_df.columns:
                score_df[col] = ds_df[col].values

    for metric in ["context_precision", "faithfulness", "answer_correctness"]:
        n_val = naive_df[metric].mean()
        g_val = graph_df[metric].mean()
        a_val = agentic_df[metric].mean()
        best = max(g_val, a_val)
        delta = best - n_val
        pct = (delta / max(n_val, 0.001)) * 100
        print(f"{metric:<35} {n_val:>12.3f} {g_val:>12.3f} {a_val:>12.3f} {delta:>+10.3f} ({pct:+.0f}%)")

    # Print latency and chunk-count summary
    print("-" * 95)
    for label, df in [("Median latency (s)", "latency_s"), ("Avg chunks retrieved", "chunk_count")]:
        row_vals = []
        for score_df in [naive_df, graph_df, agentic_df]:
            if df in score_df.columns:
                val = score_df[df].dropna().median() if "latency" in df else score_df[df].dropna().mean()
                row_vals.append(f"{val:>12.2f}")
            else:
                row_vals.append(f"{'N/A':>12}")
        print(f"{label:<35} {''.join(row_vals)}")

    # --- Eval Logging ---
    from datetime import datetime, timezone
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save per-question scores for each pipeline
    for name, df in [("naive", naive_df), ("graph", graph_df), ("agentic", agentic_df)]:
        csv_path = os.path.join(results_dir, f"scores_{name}_{run_ts}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved {name} per-question scores: {csv_path}")

    def _pipeline_summary(df):
        s = {m: float(df[m].mean()) for m in ["context_precision", "faithfulness", "answer_correctness"]}
        if "latency_s" in df.columns:
            s["median_latency_s"] = float(df["latency_s"].dropna().median())
        if "chunk_count" in df.columns:
            s["avg_chunk_count"] = float(df["chunk_count"].dropna().mean())
        return s

    # Save run metadata
    run_info = {
        "run_timestamp": run_ts,
        "question_count": len(questions),
        "collection": COLLECTION_NAME,
        "eval_model": "deepseek-v4-flash",
        "eval_embeddings": "gemini-embedding-2",
        "generation_model": "deepseek-v4-flash",
        "retrieval_embedding": "gemini-embedding-2",
        "max_context_chunks": 8,
        "summary": {
            "naive": _pipeline_summary(naive_df),
            "graph": _pipeline_summary(graph_df),
            "agentic": _pipeline_summary(agentic_df),
        }
    }
    run_info_path = os.path.join(results_dir, f"run_info_{run_ts}.json")
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"  Saved run metadata: {run_info_path}")