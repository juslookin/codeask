import os
import json
import asyncio
import time
import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, "..", "backend", ".env")
EVAL_DIR = SCRIPT_DIR
load_dotenv(ENV_PATH)
load_dotenv()

from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=60.0
)

JUDGE_PROMPT = """You are an expert code evaluation judge. Evaluate the RAG system output across 3 metrics:
1. context_precision (0.0 to 1.0): Did the retrieved code chunks contain the necessary definitions/methods to answer the question, with minimal noise?
2. faithfulness (0.0 to 1.0): Are all statements and code citations in the response strictly grounded in the retrieved code chunks (no hallucinated functions)?
3. answer_correctness (0.0 to 1.0): Does the response accurately convey the same technical facts, lifecycle order, and logic as the Reference ground truth?

You MUST respond strictly with a valid JSON object in this exact format:
{
  "context_precision": 0.85,
  "faithfulness": 0.95,
  "answer_correctness": 0.90,
  "reasoning": "brief reason"
}"""

semaphore = asyncio.Semaphore(6)

async def score_sample(q, resp, contexts, ref):
    async with semaphore:
        ctx_str = "\n\n---\n\n".join(contexts)
        user_msg = f"## Question\n{q}\n\n## Retrieved Code Context\n{ctx_str}\n\n## System Response\n{resp}\n\n## Reference Ground Truth\n{ref}"

        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[
                        {"role": "system", "content": JUDGE_PROMPT},
                        {"role": "user", "content": user_msg}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    extra_body={"thinking": {"type": "disabled"}}
                )
                res = json.loads(response.choices[0].message.content)
                res["question"] = q
                return res
            except Exception as e:
                if attempt == 2:
                    raise e
                await asyncio.sleep(1.5)

async def evaluate_pipeline(name):
    json_path = os.path.join(EVAL_DIR, f"checkpoint_{name}.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    tasks = [
        score_sample(
            data["user_input"][i],
            data["response"][i],
            data["retrieved_contexts"][i],
            data["reference"][i]
        )
        for i in range(len(data["user_input"]))
    ]
    scores = await asyncio.gather(*tasks)
    df = pd.DataFrame(scores)
    
    out_csv = os.path.join(EVAL_DIR, f"checkpoint_scores_geval_{name}.csv")
    df.to_csv(out_csv, index=False)
    return df, data

async def main():
    t0 = time.time()
    print("Starting Unified G-Eval for ALL 3 Pipelines (Naive, Graph, Agentic)...")
    
    df_n, data_n = await evaluate_pipeline("naive")
    print("  [+] Evaluated 20 questions for Naive RAG")
    
    df_g, data_g = await evaluate_pipeline("graph")
    print("  [+] Evaluated 20 questions for GraphRAG")
    
    df_a, data_a = await evaluate_pipeline("agentic")
    print("  [+] Evaluated 20 questions for Agentic RAG")
    
    elapsed = round(time.time() - t0, 2)
    print(f"\nALL 60 EVALUATIONS COMPLETED IN {elapsed}s!\n")
    
    print("=" * 95)
    print(f"{'Metric':<30} {'Naive RAG':>15} {'GraphRAG':>15} {'Agentic RAG':>15} {'Best Delta':>15}")
    print("-" * 95)
    
    for m in ["context_precision", "faithfulness", "answer_correctness"]:
        nv = df_n[m].mean()
        gv = df_g[m].mean()
        av = df_a[m].mean()
        best = max(gv, av)
        delta = best - nv
        pct = (delta / max(nv, 0.001)) * 100
        print(f"{m:<30} {nv:>15.4f} {gv:>15.4f} {av:>15.4f} {delta:>+9.4f} ({pct:+.1f}%)")
        
    print("-" * 95)
    n_lat = pd.Series(data_n.get("latency_s", [])).median()
    g_lat = pd.Series(data_g.get("latency_s", [])).median()
    a_lat = pd.Series(data_a.get("latency_s", [])).median()
    print(f"{'Median Latency (s)':<30} {n_lat:>14.2f}s {g_lat:>14.2f}s {a_lat:>14.2f}s")

    n_chk = pd.Series([len(c) for c in data_n.get("retrieved_contexts", [])]).mean()
    g_chk = pd.Series([len(c) for c in data_g.get("retrieved_contexts", [])]).mean()
    a_chk = pd.Series([len(c) for c in data_a.get("retrieved_contexts", [])]).mean()
    print(f"{'Avg Chunks Retrieved':<30} {n_chk:>15.1f} {g_chk:>15.1f} {a_chk:>15.1f}")
    print("=" * 95)

if __name__ == "__main__":
    asyncio.run(main())
