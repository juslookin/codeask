# CodeAsk — Scoped Implementation Plan

**Status:** Planning only. Nothing in this plan has been implemented yet.
**Supersedes:** `retrieval_and_evaluation_implementation_guide.md`, narrowed to what's actually worth doing before a placement deadline.
**Goal:** Fix the retrieval-quality gap driving low context precision, make eval results reproducible/reportable, and migrate the generation model to DeepSeek — without turning this into a production rewrite.

---

## 0. What this plan deliberately does NOT include

Carried over from the original doc but cut, and why:

| Cut | Reason |
|---|---|
| Typed graph edges w/ confidence enum, resolution types | `MAX_FANOUT_CANDIDATES` is already a documented stopgap in `graph_builder.py`. Not the bottleneck the eval numbers point to. |
| SSE migration, structured retrieval diagnostic API, frontend rewrite | Sentinel-text streaming (`__GRAPH_START__`) works. Not eval-relevant. |
| Full critic/planner evidence-inventory agent redesign | The one-line fix in §1.2 (reuse the same selector on merge) removes the actual failure mode (unbounded accumulation) at near-zero cost. |
| Embedding/reranker experiment matrix | Different project; do only if there's time left after everything else. |
| Feature flags, rollback aliasing, structured logging, unit/integration/frontend test suites | Team/production concerns, not placement-project concerns. |
| Full `manifest.json` / schema-versioned collections | Replaced with a lightweight run-metadata header (§4). |

---

## 1. Retrieval quality: bounded, ranked evidence

**Problem:** `graph_traversal.expand_one_hop` adds *every* resolvable direct callee with no ranking or cap. `agent.py`'s `retriever_node` appends new chunks every iteration with no cap either. This is almost certainly why context precision is your weakest metric on both the graph and agent arms.

### 1.1 New shared selector

`graphrag-qa/backend/retrieval/selector.py` (new file, ~25 lines):

```python
import numpy as np
from ingestion.embedder import model

def select_top_k(query: str, seed_chunks: list[dict], candidate_chunks: list[dict], k: int = 8) -> list[dict]:
    """Keep all seeds, rank expansion/candidate chunks by cosine similarity
    to the query, and keep only enough of them to fill the budget."""
    if not candidate_chunks:
        return seed_chunks

    query_vec = model.encode([query])[0]
    cand_vecs = model.encode([c["source_code"] for c in candidate_chunks])
    sims = cand_vecs @ query_vec / (
        np.linalg.norm(cand_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8
    )
    ranked = [c for c, _ in sorted(zip(candidate_chunks, sims), key=lambda x: -x[1])]

    budget = max(k - len(seed_chunks), 0)
    return seed_chunks + ranked[:budget]
```

Config addition (`.env` / defaults): `MAX_CONTEXT_CHUNKS=8`

### 1.2 Wire it into graph mode

In `api/query.py::get_context_and_graph` and `eval/benchmark.py::run_graphrag`, replace:

```python
seed = vector_search(question, collection)
expanded = expand_one_hop(seed, collection)
chunks = seed + expanded
```

with:

```python
seed = vector_search(question, collection)
expanded = expand_one_hop(seed, collection)
chunks = select_top_k(question, seed, expanded, k=MAX_CONTEXT_CHUNKS)
```

No LLM calls added — still the deterministic fast path.

### 1.3 Wire it into agent mode (fixes the accumulation bug)

In `llm/agent.py::retriever_node`, replace the raw `current_chunks + new_chunks` append with a re-rank-and-trim over the *combined* pool:

```python
def retriever_node(state: AgentState):
    query = state["search_history"][-1]
    seed = vector_search(query, state["collection_name"])
    expanded = expand_one_hop(seed, state["collection_name"])

    current_chunks = state.get("context_chunks", [])
    existing_ids = {c["id"] for c in current_chunks}
    new_chunks = [c for c in (seed + expanded) if c["id"] not in existing_ids]

    merged = select_top_k(state["question"], [], current_chunks + new_chunks, k=MAX_CONTEXT_CHUNKS)
    return {"context_chunks": merged}
```

This is the single change most likely to fix agentic RAG's worst-of-three-arms result — new evidence now has to *earn* its place by similarity score instead of just piling on.

**Acceptance check (manual, not a test suite):** rerun the benchmark; graph and agent arms should show equal-or-fewer average chunks with equal-or-better context precision / answer correctness than the current numbers.

---

## 2. Fix the qualified-name / graph collision bug

**Problem:** `ast_parser.py` builds `module_name` from just the file's basename, not its path:

```python
module_name = os.path.splitext(os.path.basename(file_path))[0]
```

Two files with the same basename in different directories (or two classes with the same name in different modules) silently collide in the graph and in ChromaDB metadata's `qualified_name`.

**Fix (literally a 1-line change — `relative_path` is already computed one line above this):**

```python
module_name = os.path.splitext(relative_path)[0].replace(os.sep, ".").replace("/", ".")
```

**Follow-up:** delete the local `chroma_db/` folder and re-ingest the Flask benchmark corpus once after this change — the old qualified names are now stale. No need for the original doc's versioned-collection system for a single personal benchmark corpus; just re-ingest.

---

## 3. Prompt tweak for sequence/lifecycle questions

Half your eval questions ask for ordered multi-step behavior ("sequence of internal functions," "role of X in the routing system," "teardown process"). Cheapest possible lever on answer correctness — add one line to `llm/gemini.py`'s `SYSTEM_PROMPT`:

```python
SYSTEM_PROMPT = """You are an expert code assistant. Answer questions using ONLY the provided code chunks.
1. If the question asks about a sequence, lifecycle, or "what happens when," answer with the steps in the order they occur.
2. End every response with a ## Citations section.
3. Format citations strictly as: `filepath:startline-endline`
"""
```

---

## 4. Eval reproducibility (cut-down version)

No `manifest.json` schema, no immutable timestamped result directories. Just enough to reproduce your report table and support a real resume claim.

### 4.1 Persist raw per-question scores

In `eval/benchmark.py`, after each `evaluate(...)` call:

```python
RUN_TS = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
...
naive_scores.to_pandas().to_csv(os.path.join(SCRIPT_DIR, f"scores_naive_{RUN_TS}.csv"), index=False)
```

(same for graph/agentic scores)

### 4.2 Run metadata header

One small JSON alongside the CSVs — not a full manifest, just enough that a future you (or an interviewer) knows what produced a given number:

```python
run_info = {
    "run_ts": RUN_TS,
    "question_count": len(questions),
    "generation_model": "deepseek-v4-flash",  # after §6 migration
    "evaluator_llm": "deepseek-v4-flash",     # after §6.4 correction — was gemini-3.5-flash-lite
    "evaluator_embedding": "models/gemini-embedding-001",  # or -2, see §6.5 — this one stays on Gemini
    "retrieval_embedding": "all-MiniLM-L6-v2",
    "max_context_chunks": MAX_CONTEXT_CHUNKS,
}
json.dump(run_info, open(os.path.join(SCRIPT_DIR, f"run_info_{RUN_TS}.json"), "w"), indent=2)
```

### 4.3 Track latency and chunk count per pipeline (new, cheap, high resume value)

You currently have zero latency numbers. Wrap each `pipeline_fn(q["question"])` call with `time.time()` before/after and store `latency_s` per row, plus `len(result["contexts"])`. This gives you the "accuracy vs cost vs latency" table that makes the agentic-vs-graph story concrete instead of asserted — e.g. *"agentic mode used up to 7 LLM calls and Xs median latency per question for a lower answer-correctness score than the zero-LLM-call graph path."*

### 4.4 Update the README

`graphrag-qa/README.md`'s results table is currently stale (2-metric run, older numbers, no `answer_correctness` column). Replace it with your latest 3-metric table and keep the existing "10 questions, one repo, one language — treat as directional" caveat paragraph. This file is the artifact an interviewer is most likely to actually open — make it match what you can defend.

---

## 5. Deferred, not in this pass

Explicitly not doing these now — listed so they don't get silently lost:

- Stale re-ingestion bug (`embedder.py` reuses an existing non-empty collection even if the upstream repo changed) — real bug, low priority, do only if time remains.
- Reverse "who calls X" caller index — none of the current 10 eval questions need it.

---

## 6. Model migration: Gemini → DeepSeek for generation, Gemini kept for embeddings

### 6.1 Scope of the swap

Only the **chat/completion** usages move to DeepSeek. Embeddings are unaffected by this migration:

| Current usage | File | Model today | After migration |
|---|---|---|---|
| Answer generation | `llm/gemini.py` → `model` | `gemini-3.5-flash-lite` | DeepSeek |
| Agent planner + critic (structured output) | `llm/agent.py` via `structured_model` | `gemini-3.5-flash-lite` | DeepSeek |
| Ragas judge (`evaluator_llm`) | `eval/benchmark.py` | `gemini-3.5-flash-lite` | **DeepSeek — reversed from the original recommendation, see 6.4** |
| Retrieval embeddings | `ingestion/embedder.py` | `all-MiniLM-L6-v2` (local) | unchanged, unrelated to either provider |
| Ragas judge embeddings (`evaluator_embeddings`) | `eval/benchmark.py` | `models/gemini-embedding-001` | unchanged / optionally upgraded — see 6.5 |

### 6.2 Use `langchain-deepseek`, not a bare `ChatOpenAI` wrapper

DeepSeek's API is OpenAI-format-compatible, but there's a known LangChain issue where a generic `ChatOpenAI(base_url="https://api.deepseek.com", ...)` does **not** reliably support `.with_structured_output()` — which `llm/agent.py`'s planner and critic both depend on. Use the dedicated `langchain-deepseek` package's `ChatDeepSeek` class instead, which handles structured output correctly (including a strict-schema beta mode).

```
pip install langchain-deepseek
```

```python
from langchain_deepseek import ChatDeepSeek

model = ChatDeepSeek(model="deepseek-v4-flash", temperature=0.3, api_key=os.getenv("DEEPSEEK_API_KEY"))
structured_model = ChatDeepSeek(model="deepseek-v4-flash", temperature=0, api_key=os.getenv("DEEPSEEK_API_KEY"))
```

### 6.3 Model name — current as of Aug 2026

DeepSeek's legacy names `deepseek-chat` / `deepseek-reasoner` were scheduled for retirement on **2026-07-24** (already past) — they now exist only as compatibility aliases mapping to the non-thinking and thinking modes of `deepseek-v4-flash` respectively. Use the current names directly:

- **`deepseek-v4-flash`** (non-thinking mode) — use this for both answer generation and the planner/critic structured-output calls.
- **Do not use `deepseek-v4-flash` in thinking mode / `deepseek-reasoner`-equivalent for the planner or critic** — DeepSeek's reasoning/thinking mode does not support tool calling or structured output, which `with_structured_output()` relies on.
- `deepseek-v4-pro` exists for higher quality at higher cost if `flash` underperforms on your benchmark — treat as a variant to test, not a default swap.

### 6.4 Correction: move the Ragas judge to DeepSeek too — this is now the point of the migration

**Reversed from the original recommendation.** The stated reason for this migration is that Gemini's free-tier rate limits are blocking the eval from running at all — and `evaluator_llm` is very likely the *dominant* source of Gemini calls during a benchmark run, not the generation model. Ragas' `context_precision`, `faithfulness`, and `answer_correctness` each issue multiple LLM calls per row (a verdict call per retrieved chunk, claim extraction + verification, etc.), all against the same `gemini-3.5-flash-lite` 15 RPM / 500 RPD free-tier bucket that answer generation also shares. Leaving `evaluator_llm` on Gemini after this migration would leave the actual bottleneck in place and likely still fail to run.

So: move `evaluator_llm` to DeepSeek as well.

```python
from langchain_deepseek import ChatDeepSeek
from ragas.llms import LangchainLLMWrapper

evaluator_llm = LangchainLLMWrapper(
    ChatDeepSeek(model="deepseek-v4-flash", temperature=0, api_key=os.getenv("DEEPSEEK_API_KEY"))
)
```

The self-preference-bias point from the original recommendation (a model family tends to rate its own phrasing more favorably) is still technically true, but it's a minor methodological footnote, not a reason to keep an eval you can't run. If it matters for how you present this, one sentence in your README/report is enough: *"the same model family is used for generation and judging; absolute scores should be read with that in mind."* Don't let it block unblocking the eval.

### 6.5 Embeddings: `gemini-embedding-001` vs `gemini-embedding-2`

You said you're continuing with Gemini embeddings — current code uses `models/gemini-embedding-001`. Google has since shipped `gemini-embedding-2` (their newest, multimodal embedding model). Important caveat before switching: **the embedding spaces of `gemini-embedding-001` and `gemini-embedding-2` are not compatible** — vectors from one can't be meaningfully compared to vectors from the other. This matters here because Ragas' `answer_correctness` metric uses embedding-based semantic similarity under the hood: if you switch evaluator-embedding versions mid-project, any new benchmark run is not directly comparable to earlier runs computed under `-001`. Practically:
- If you switch, treat it as a new baseline, not a continuation — record the exact model string in `run_info.json` (§4.2) so it's traceable.
- Verify the exact model string against Google's current docs when you implement this (embedding model IDs have changed before in this codebase — see the existing comment in `benchmark.py` about `models/embedding-001` being retired).

### 6.6 Other places this migration touches

- **`.env.example`**: add `DEEPSEEK_API_KEY`; keep `GEMINI_API_KEY` (still needed for both embedding usages — retrieval stays local/MiniLM regardless, but `evaluator_embeddings` in §6.5 stays on Gemini).
- **`requirements.txt`**: add `langchain-deepseek`; keep `langchain-google-genai` (needed for the Gemini embedding calls only, now that chat/judge calls move off Gemini entirely).
- **Rate-limit / retry config in `benchmark.py`**: `EVAL_ROW_DELAY=4.0`, `max_retries=10`, `max_wait=120` in `RunConfig`, and `max_workers=1` were all tuned specifically around Gemini's free-tier 15 RPM. None of that is needed on a paid DeepSeek tier and, left as-is, it will just make your run slower than it needs to be. See §6.7.
- **Naming hygiene (optional)**: `llm/gemini.py` will contain a DeepSeek client after this change, which is confusing. Consider renaming to something provider-neutral like `llm/generation.py` — cosmetic, skip if short on time.

### 6.7 Concrete steps for the $2 budget

1. **Use `deepseek-v4-flash` everywhere** (generation, agent planner/critic, and the Ragas judge) — not `deepseek-v4-pro`. Flash is roughly 3x cheaper per token and this is a benchmarking task, not a quality-critical production path.
2. **Loosen the Gemini-era throttling in `benchmark.py`** now that you're not fighting a 15 RPM free-tier ceiling:
   ```python
   EVAL_ROW_DELAY = float(os.getenv("EVAL_RATE_LIMIT_DELAY", "0"))   # was 4.0
   run_config = RunConfig(max_workers=4, max_retries=3, max_wait=30)  # was max_workers=1, max_retries=10, max_wait=120
   ```
   Start with `max_workers=4` rather than something more aggressive — you don't yet know DeepSeek's actual concurrent-request limit for your account tier, and this is cheap to raise later if runs are fast and clean.
3. **Run a 1-2 question pilot on one pipeline first**, check DeepSeek's dashboard for actual token spend on that pilot, and extrapolate to the full 10-question × 3-pipeline × 3-metric run before committing the whole $2. This is the single best way to avoid an unpleasant surprise given how many small Ragas judge calls happen per row.
4. **Mind peak-hour pricing.** As of the current DeepSeek pricing (checked against their live pricing page — verify again before you actually run this, it's changed more than once in the last month), Flash is billed at roughly $0.22/M input and $0.66/M output tokens off-peak, and about double that (~$0.44/$1.32) during peak hours **01:00-04:00 and 06:00-10:00 UTC, Monday-Friday**. All other times — including the entire weekend — are off-peak. Running your full benchmark outside those windows roughly halves the cost.
5. **Check whether you already have signup credit.** New DeepSeek accounts have reportedly received several million free tokens on signup (no card required) in some recent reporting — check your own dashboard balance before assuming you need to spend the $2 at all.
6. Given typical token volume for this workload (30 generation rows + up to a few agent iterations each + a Ragas judge that issues several short calls per row per metric), total spend is very likely well under $1 at Flash off-peak rates even before any free credit — the $2 you've budgeted should have real margin. Treat step 3's pilot as the actual source of truth over this estimate.

---

## 7. Suggested order of work

Reordered from the original draft — the DeepSeek migration is no longer a parallel/optional track, it's the prerequisite for running any eval at all, so it goes first.

1. **§6 (DeepSeek migration, including §6.4's corrected evaluator_llm move)** — do this first. Nothing downstream matters until you can actually complete a benchmark run.
2. **§6.7 step 3** — run the 1-2 question pilot, sanity-check cost and that structured output (planner/critic) actually works against `ChatDeepSeek`, before committing to a full run.
3. §2 (module collision fix) — 1 line, do next since it invalidates the local Chroma DB anyway and everything else benefits from a clean re-ingest.
4. §1 (top-k selector, both call sites) — highest-leverage fix on your worst metric.
5. §3 (prompt tweak) — trivial, do alongside §1.
6. §4 (eval logging + latency/chunk-count tracking) — do last, right before your final benchmark run, so the numbers you log reflect the post-fix pipeline running on DeepSeek.
7. Re-run the full benchmark once, update the README table (§4.4), pull your resume metrics from that run.

Note for the numbers you get: this run is not directly comparable to your original Gemini-based baseline — both the generation model and the judge model changed. Treat it as a new baseline, not a continuation, and say so in the README caveat paragraph.
