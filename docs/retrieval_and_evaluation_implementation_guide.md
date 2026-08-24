# CodeAsk Retrieval and Evaluation Implementation Guide

**Status:** Proposed implementation plan  
**Audience:** CodeAsk maintainers and contributors  
**Scope:** Retrieval quality, graph accuracy, agentic retrieval, evaluation reproducibility, API/frontend transparency, and regression testing

## 1. Purpose

This document converts the findings from the CodeAsk evaluation report into an implementation-ready plan. It is intentionally more specific than a roadmap: it identifies the affected modules, the target contracts, migration strategy, tests, rollout order, and measurable acceptance criteria.

The goal is not simply to make the graph or agent more complex. The goal is to improve answer correctness by selecting a small, well-grounded evidence set, while retaining a deterministic fast path that is easy to measure and debug.

## 2. Current System Summary

The current request path is:

```text
Repository URL
  -> shallow Git clone
  -> tree-sitter function/method chunks
  -> heuristic call graph
  -> MiniLM embeddings in ChromaDB
  -> vector-search seeds
  -> forward one-hop callee expansion
  -> bounded textual context
  -> Gemini answer stream
```

The default `graph` mode retrieves five vector-search seed chunks and then adds every resolvable direct callee. The optional `agent` mode repeats this operation up to three times and appends all newly found chunks to its memory.

The relevant implementation modules are:

| Concern | Current location |
|---|---|
| Repository clone and file selection | `graphrag-qa/backend/ingestion/github.py` |
| AST chunks | `graphrag-qa/backend/ingestion/ast_parser.py` |
| Graph construction | `graphrag-qa/backend/ingestion/graph_builder.py` |
| Persistence and embeddings | `graphrag-qa/backend/ingestion/embedder.py` |
| Vector retrieval | `graphrag-qa/backend/retrieval/vector_search.py` |
| Graph expansion | `graphrag-qa/backend/retrieval/graph_traversal.py` |
| Context assembly | `graphrag-qa/backend/retrieval/context_builder.py` |
| Agent loop | `graphrag-qa/backend/llm/agent.py` |
| Query API and graph response | `graphrag-qa/backend/api/query.py` |
| Chat stream parsing | `graphrag-qa/frontend/src/components/ChatWindow.jsx` |
| Benchmark harness | `graphrag-qa/eval/benchmark.py` |

## 3. Problem Statement and Design Principles

The evaluation checkpoints show a meaningful retrieval-size gap: naïve retrieval returns five chunks, while graph retrieval returns roughly 7–20 and agentic retrieval roughly 8–33. More retrieved context is not automatically better. Extra chunks compete for the generator's attention and may reduce answer correctness even when they are structurally adjacent.

The implementation should follow these principles:

1. **Measure before replacing.** Treat new embedding models and rerankers as evaluated variants, not unconditional upgrades.
2. **Prefer exact identity over clever heuristics.** A graph edge that points to the wrong `save` or `run` function is worse than no graph edge.
3. **Separate candidate generation from evidence selection.** Vector search and graph traversal may propose many chunks; a scoring/budget stage chooses what reaches the LLM.
4. **Keep the fast path deterministic.** The graph mode must remain inspectable, low-latency, and free of mandatory extra LLM calls.
5. **Version every evaluation input.** A metric without repository revision, corpus configuration, model configuration, and raw per-question results is not reproducible evidence.
6. **Expose enough retrieval evidence to debug production answers.** A developer should be able to determine why each chunk was selected without reading server logs.

## 4. Target Architecture

```text
                         +--------------------------+
Git URL -> Ingestion ->  | Versioned repository      |
                         | manifest + collection     |
                         +------------+-------------+
                                      |
                    +-----------------+-----------------+
                    |                                   |
             AST chunks and symbols              Typed graph edges
                    |                                   |
                    +---------------+-------------------+
                                    |
                                 ChromaDB
                                    |
Query -> vector candidates -> graph candidates -> rank / dedupe / budget
                                                      |
                                 +--------------------+-------------------+
                                 |                                        |
                           Graph fast path                         Agent evidence loop
                                 |                                        |
                       structured retrieval events                  final selected evidence
                                 +--------------------+-------------------+
                                                      |
                                            answer generation
                                                      |
                                SSE answer and retrieval diagnostics
```

The candidate set is deliberately broader than the evidence set. Expansion may inspect up to a configured number of neighboring nodes, but the final generator context should normally contain no more than 8–12 chunks and a defined character/token budget.

## 5. Phase 0 - Reproducible Evaluation Baseline

### 5.1 Objectives

Before changing retrieval behavior, make every benchmark result auditable. The existing checkpoint files preserve generated answers and contexts, but do not persist the Ragas result tables or the settings used to produce them. This phase closes that gap.

### 5.2 Required changes

Create `graphrag-qa/eval/results/` and write one immutable result directory per benchmark run:

```text
eval/results/<UTC timestamp>-<short git SHA>/
  manifest.json
  inputs.json
  naive_dataset.json
  graph_dataset.json
  agentic_dataset.json
  naive_scores.json
  graph_scores.json
  agentic_scores.json
  summary.json
  report.md
```

Do not commit API keys, complete cloned repositories, raw Chroma database files, or provider request/response bodies to this directory.

### 5.3 Benchmark manifest contract

`manifest.json` must contain at least:

```json
{
  "schema_version": 1,
  "run_id": "2026-08-24T12-00-00Z-abc1234",
  "codeask_git_sha": "abc1234...",
  "started_at": "2026-08-24T12:00:00Z",
  "finished_at": "2026-08-24T12:15:00Z",
  "question_set": {
    "path": "eval/questions.json",
    "sha256": "...",
    "count": 10
  },
  "corpus": {
    "collection_name": "pallets_flask__<source-sha>",
    "source_url": "https://github.com/pallets/flask",
    "source_revision": "...",
    "ingestion_config_sha256": "...",
    "chunk_count": 0,
    "edge_count": 0
  },
  "pipelines": {
    "naive": {"seed_count": 5},
    "graph": {"seed_count": 5, "neighbor_limit": 2, "context_chunk_limit": 10},
    "agentic": {"max_iterations": 3, "context_chunk_limit": 10}
  },
  "models": {
    "retrieval_embedding": "...",
    "generator": "...",
    "evaluator_llm": "...",
    "evaluator_embedding": "..."
  },
  "packages": {
    "python": "...",
    "ragas": "...",
    "chromadb": "...",
    "tree_sitter": "..."
  }
}
```

`inputs.json` should include the question set and normalized pipeline configuration. The per-pipeline score JSON should include every metric for every question, not just means.

### 5.4 Benchmark behavior changes

1. Replace checkpoint-only execution with an explicit `--resume` option. A normal run creates a new result directory; `--resume <run-id>` is the only path that reuses partial data.
2. Save generated datasets before calling Ragas and save Ragas outputs immediately after each pipeline completes.
3. Record any row that fails generation or evaluation as a structured error. Do not substitute an `Error: ...` answer into a dataset that is then averaged without clearly marking it.
4. Keep `max_workers=1` for free-tier stability, but record it in the manifest.
5. Add a `--dry-run` mode that verifies corpus availability, question count, manifest generation, and configuration without provider calls.
6. Generate the Markdown benchmark report from `summary.json`; do not manually transcribe the table.

### 5.5 Evaluation interpretation rules

The generated report must state:

- Results are specific to the recorded repository revision, question set, models, and date.
- Ten questions are not sufficient to claim a general ranking across repositories or languages.
- Faithfulness is an evaluator-produced grounding score, not a proof that hallucination never occurs.
- A metric change must be accompanied by absolute delta, relative delta, per-question deltas, and latency/cost change.

### 5.6 Acceptance criteria

- A completed run has all files listed in section 5.2.
- The result folder can be used to recreate the report table without calling a model.
- A reviewer can identify the corpus revision and every model involved from the manifest alone.
- A failed row cannot silently alter a mean score.

## 6. Phase 1 - Versioned Ingestion and Canonical Symbol Identity

### 6.1 Objectives

Eliminate graph collisions caused by non-unique qualified names and make re-ingestion deterministic. Today, class names do not incorporate a module path, and re-ingestion can silently reuse an existing non-empty Chroma collection even if its upstream repository changed.

### 6.2 Repository manifest

During cloning, capture:

- canonical GitHub owner/repository;
- normalized source URL;
- checked-out commit SHA;
- collection schema version;
- parser configuration version;
- accepted file list and file SHA-256 values;
- configured size/file limits.

Write the manifest adjacent to the persistent collection or persist it in Chroma collection metadata. Prefer a small metadata collection or JSON sidecar keyed by the collection name; Chroma collection metadata alone is insufficient for large file manifests.

Use a collection name including a stable revision discriminator, for example:

```text
<owner>_<repo>__<first 12 source commit characters>__v2
```

Maintain an optional human-readable alias only after it has been resolved to one immutable collection revision. Never overwrite an old revision in place during ingestion.

### 6.3 Chunk identity contract

Introduce a canonical symbol name and a stable chunk identifier. The name must be unambiguous within a repository revision.

```python
canonical_module = relative_path_without_extension.replace("/", ".").replace("\\", ".")
symbol = f"{canonical_module}.{class_path}.{function_name}"  # class path is optional
chunk_id = sha256(
    f"{repository_revision}\0{relative_path}\0{start_line}\0{end_line}\0{symbol}\0{chunk_kind}"
).hexdigest()
```

Examples:

```text
src.flask.app.Flask.handle_exception
src.flask.testing.FlaskClient.__enter__
apps.server.auctionEngine.resolveLot
```

The stored metadata should include:

```json
{
  "schema_version": 2,
  "repository_revision": "...",
  "file_path": "src/flask/app.py",
  "language": "python",
  "symbol": "src.flask.app.Flask.handle_exception",
  "base_symbol": "src.flask.app.Flask.handle_exception",
  "parent_symbol": "src.flask.app.Flask",
  "start_line": 868,
  "end_line": 922,
  "kind": "function",
  "source_sha256": "..."
}
```

Use forward slashes in persisted paths on every platform. Convert only at file-system boundaries.

### 6.4 AST extraction changes

Refactor `ast_parser.py` to produce a richer intermediate representation instead of directly returning only text chunks. At minimum, capture:

- module path;
- containing class/function path;
- definitions and locations;
- imports and aliases;
- call-expression receiver and member components;
- language and parser version;
- decorators/exports when they influence visibility;
- source spans for whole functions and generated subchunks.

For oversized functions, each subchunk must retain the same `base_symbol` and a distinct `subchunk_index`. Graph edges should target the base function symbol and be mapped to the best overlapping/full chunk later; do not make every subchunk appear to be an independent callable function.

### 6.5 Migration strategy

Do not try to mutate v1 collections. Add a schema version to the collection name and create v2 collections afresh. Query endpoints should reject unknown/incompatible collection schemas with an actionable message rather than returning partially malformed results.

### 6.6 Acceptance criteria

- Two functions named `run` in distinct files have different canonical symbols and separate graph nodes.
- Two classes sharing a name in different modules cannot resolve one another's methods solely because their class names match.
- Re-ingesting a repository at a new source commit produces a new collection revision.
- A query response identifies the source revision used to answer it.

## 7. Phase 2 - Typed, Conservative Graph Construction

### 7.1 Objectives

Improve edge precision before increasing graph traversal. The graph should represent confirmed or clearly labeled heuristic relationships, not infer a precise call edge from a bare method name with no scope information.

### 7.2 Graph edge contract

Replace the `dict[symbol, list[symbol]]` graph representation with typed edge records:

```json
{
  "source_symbol": "src.flask.app.Flask.full_dispatch_request",
  "target_symbol": "src.flask.app.Flask.handle_user_exception",
  "relationship": "calls",
  "resolution": "same_class",
  "confidence": 1.0,
  "source_span": {"start_line": 1009, "end_line": 1009}
}
```

Supported initial resolution values:

| Resolution | Description | Default confidence |
|---|---|---:|
| `same_class` | `self.method()` or equivalent resolved in the enclosing class | 1.00 |
| `same_module` | A direct function reference resolves in the same module | 0.95 |
| `direct_import` | Imported name resolves to a parsed local symbol | 0.95 |
| `module_alias` | An imported module alias resolves to a local parsed symbol | 0.90 |
| `unambiguous_global` | Exactly one allowed fallback candidate exists | 0.60 |
| `unresolved` | Preserve diagnostic metadata but do not traverse by default | 0.00 |

Maintain both adjacency directions:

```python
outgoing_edges_by_symbol: dict[str, list[GraphEdge]]
incoming_edges_by_symbol: dict[str, list[GraphEdge]]
```

### 7.3 Resolution order

For each call expression, resolve once in this order:

1. If the receiver is the current instance/class and the method exists on the enclosing class, create a `same_class` edge.
2. If the call is a direct local function name defined in the current module, create a `same_module` edge.
3. If an imported name maps to an indexed local module and symbol, create a `direct_import` edge.
4. If `alias.function()` maps to an imported local module and indexed symbol, create a `module_alias` edge.
5. Use an unqualified global-name fallback only if exactly one indexed candidate remains after language/module filtering.
6. Otherwise record an `unresolved` diagnostic and create no traversable edge.

Never resolve an arbitrary `object.method()` to a method merely because the method name is present in the repository. Full type inference is out of scope for the first release; incorrect edges must not be presented as fact.

### 7.4 Language rollout

Implement and test the resolver in Python first, because the evaluation corpus is Flask/Python. Preserve generic intermediate models so JavaScript/TypeScript resolvers can be added without changing retrieval contracts.

For JS/TS/JSX/TSX, begin with same-module function declarations, imports/exports, and direct named imports. Mark dynamic imports, computed property names, and runtime property dispatch unresolved.

### 7.5 Tests and fixtures

Create fixture repositories under `graphrag-qa/backend/tests/fixtures/`:

- Python: same module functions, sibling classes with a shared method name, imports, aliases, inheritance, and unresolved external calls.
- JS/TS: named exports/imports, aliases, arrow functions, object methods, and dynamic calls.
- Collision fixture: identical class/function names in multiple modules.

Test edge sets exactly. Tests should assert both edges that must exist and ambiguous edges that must not exist.

### 7.6 Acceptance criteria

- The graph contains caller and callee indexes.
- Exact fixture graphs have no false cross-module collision edges.
- All traversable edges include a resolution type and confidence.
- The UI can identify whether an edge is confirmed or heuristic.

## 8. Phase 3 - Evidence Selection for Graph Retrieval

### 8.1 Objectives

Introduce a deterministic selection stage between candidate generation and context building. Graph traversal should add useful supporting evidence, not blindly add every direct callee.

### 8.2 Retrieval data model

Use an internal record such as:

```python
@dataclass
class RetrievalCandidate:
    chunk: Chunk
    origin: Literal["vector_seed", "outgoing_neighbor", "incoming_neighbor", "agent_search"]
    source_seed_id: str | None
    vector_distance: float | None
    lexical_score: float
    graph_confidence: float | None
    graph_relationship: str | None
    final_score: float
```

Do not overload Chroma metadata with transient scores. Scores belong to the request-level retrieval result.

### 8.3 Query-intent routing

Derive a lightweight, deterministic intent classification from the question. This can initially be keyword/rule based and must be logged.

| Intent | Examples | Preferred expansion |
|---|---|---|
| Implementation/detail | "How does X serialize?" | outgoing callees |
| Control flow | "What happens when X runs?" | outgoing and incoming |
| Responsibility | "What calls X?" | incoming callers |
| Architecture | "How do A and B connect?" | both directions, tighter scoring |
| Definition | "What is X?" | vector seeds, little/no expansion |

The classifier should fail safely to a balanced default, not a multi-hop traversal.

### 8.4 Candidate generation and budgets

Initial configuration (environment-overridable):

```text
VECTOR_SEED_COUNT=5
MAX_NEIGHBORS_PER_SEED=2
MAX_GRAPH_CANDIDATES=20
FINAL_CONTEXT_CHUNK_LIMIT=10
FINAL_CONTEXT_MAX_CHARS=60000
MIN_GRAPH_EDGE_CONFIDENCE=0.75
```

Use edges with confidence below the threshold only in a debug view, not in default context selection. The initial settings deliberately favor precision. Later evaluations may tune them.

### 8.5 Deterministic scoring

Use normalized values and a versioned score function. An initial score can be:

```text
seed candidate:
  0.75 * vector_relevance + 0.25 * lexical_overlap

graph neighbor:
  0.45 * source_seed_relevance
  + 0.25 * graph_edge_confidence
  + 0.20 * lexical_overlap
  + 0.10 * structural_locality
```

`structural_locality` may reward candidates in the same module or class as the seed. Keep score weights in a named configuration object and record its version in benchmark manifests.

Lexical overlap should be identifier-aware: tokenize `full_dispatch_request`, `full-dispatch-request`, and `full dispatch request` into comparable terms. It must not simply reward punctuation or common language keywords.

### 8.6 Deduplication and diversity

Before final selection:

1. Deduplicate exact chunk IDs.
2. Prefer a full function chunk over a subchunk if the function fits within the context budget.
3. Remove chunks with the same base symbol unless their spans cover distinct relevant parts of an oversized definition.
4. Penalize redundant chunks from the same file/class once a higher-scoring sibling is retained.
5. Retain at least one high-quality seed whenever graph neighbors are selected.

The selection result must preserve the original rank, origin, and scores for diagnostics.

### 8.7 Context builder changes

Change `build_context` to accept already-selected evidence rather than concatenate arbitrary seed and expansion lists. Give every context section a stable evidence label:

```text
[#1 | vector seed | score 0.91]
src/flask/app.py:995-1022
<source code>
```

The labels improve post-answer debugging without telling the answer model to cite internal scores. The answer prompt must still instruct citations to use source file paths and line ranges only.

### 8.8 Acceptance criteria

- Graph mode has a hard final chunk and character budget.
- Each selected chunk has an origin and score trace.
- The default graph retrieval path uses no LLM calls.
- Benchmark comparison shows no answer-correctness regression from the current baseline, with lower or equal median context size.

## 9. Phase 4 - Agentic Retrieval With Selective Evidence Memory

### 9.1 Objectives

Make agentic mode deliberate rather than accumulative. The agent should search to fill identified evidence gaps, retain only the best evidence, and stop when more retrieval is unlikely to help.

### 9.2 State contract

Replace the loosely typed context list with a state containing selected evidence, candidate traces, and a compact summary:

```python
class CriticDecision(BaseModel):
    is_sufficient: bool
    missing_knowledge: str
    suggested_query: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

class EvidenceItem(TypedDict):
    chunk_id: str
    symbol: str
    file_path: str
    line_range: tuple[int, int]
    origin: str
    score: float
    rationale: str

class AgentState(TypedDict):
    question: str
    collection_name: str
    selected_evidence: list[EvidenceItem]
    evidence_summary: str
    search_history: list[str]
    iteration_traces: list[dict]
    critic_feedback: CriticDecision | None
    iterations: int
    next_action: Literal["retrieve", "end"]
```

`evidence_summary` should be generated deterministically from file paths, symbols, and concise extracted signatures/docstrings where possible. Do not ask an LLM to summarize all code before the planner makes its next query; doing so adds cost and can introduce hallucinated state.

### 9.3 Proposed workflow

```text
question
  -> initial retrieval using question
  -> rank/filter/budget evidence
  -> critic evaluates evidence sufficiency and names gap
  -> if gap is actionable and within iteration budget:
       planner receives question + current evidence inventory + gap
       planner creates one focused query
       retrieve + rank against entire evidence set
       retain only improved evidence set
  -> otherwise final answer from selected evidence
```

The critic may use an LLM structured-output call. The rest of the evidence selection should remain deterministic.

### 9.4 Agent stopping rules

Stop when any of the following are true:

- the critic says the evidence is sufficient with configured confidence;
- the maximum iteration count is reached;
- the proposed query duplicates a normalized prior query;
- a search contributes no new candidate above `MIN_AGENT_NOVELTY_SCORE`;
- the evidence context budget is already full and no candidate can displace lower-scoring evidence.

Keep a default maximum of three searches, including the initial retrieval. This caps latency and cost.

### 9.5 Evidence replacement policy

After every search, combine current selected evidence with newly ranked candidates and run the same selector used by graph mode. New evidence must displace weaker redundant evidence; it must not merely append to it.

Each retained item requires a request-level rationale, for example:

```text
Retained because it is the highest-scoring implementation of the exception path
and is directly called by an existing seed.
```

### 9.6 Failure behavior

- If the planner call fails, use the critic's suggested query or the original question once.
- If the critic call fails, stop rather than continue speculative searches.
- If retrieval fails, return a structured terminal retrieval error and do not fabricate an answer.
- If structured output fails schema validation, retry once with an explicit schema-repair prompt; then stop safely.

### 9.7 Acceptance criteria

- Agentic retrieval never sends more than `FINAL_CONTEXT_CHUNK_LIMIT` chunks to answer generation.
- Every extra search includes a recorded knowledge gap and produces a retrieval trace.
- Duplicate or low-novelty searches terminate early.
- Evaluation compares agentic cost, latency, selected-context size, and accuracy with graph mode.

## 10. Phase 5 - Retrieval Model Experiments

### 10.1 Objectives

Evaluate whether embedding and reranking changes improve CodeAsk on code queries. Do not alter the production default until the experiment has a documented result.

### 10.2 Experiment matrix

Run the same indexed repository revision and question set across:

| Variant | Embedding | Reranker | Purpose |
|---|---|---|---|
| Baseline | `all-MiniLM-L6-v2` | none | Current comparable baseline |
| Rerank-only | `all-MiniLM-L6-v2` | local cross-encoder | Isolate evidence-selection value |
| Code embedding | candidate code model | none | Isolate embedding value |
| Combined | candidate code model | selected reranker | Test combined outcome |

Choose models according to licensing, hardware, latency, privacy, and cost constraints. Record the exact published model IDs and revision hashes where supported.

### 10.3 Evaluation requirements

Each variant needs:

- the reproducibility artifacts from Phase 0;
- index build time and embedding storage size;
- median/p95 retrieval latency;
- mean selected chunks and characters;
- answer generation latency and estimated provider cost;
- retrieval and answer metrics;
- per-question comparison with the baseline.

Promote a variant only if it improves the chosen primary metric (initially answer correctness) without exceeding agreed latency/cost limits. A small context-precision gain alone is not sufficient if correctness or latency regresses.

## 11. Phase 6 - API, Streaming, and Frontend Transparency

### 11.1 Query validation

Replace free-form `mode: str` with a Pydantic enum:

```python
class RetrievalMode(str, Enum):
    GRAPH = "graph"
    AGENT = "agent"

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    collection_name: str
    mode: RetrievalMode = RetrievalMode.GRAPH
```

Validate that `collection_name` exists and is compatible with the expected collection schema. Return `404` for a missing collection and `409` for a legacy/incompatible collection.

### 11.2 Replace sentinel text with SSE

The current stream prepends graph JSON with private text sentinels. Replace it with Server-Sent Events:

```text
event: retrieval
data: {"collection_revision":"...","mode":"graph","selected_chunks":[...],"graph":{...}}

event: answer_delta
data: {"text":"Flask first..."}

event: complete
data: {"answer_chars":1234,"latency_ms":456}

event: error
data: {"code":"RETRIEVAL_FAILED","message":"..."}
```

Set `media_type="text/event-stream"`, use a safe JSON serializer, and include an event identifier if future reconnect support is needed. The frontend must parse events incrementally and preserve text correctly when a network chunk splits a JSON line.

### 11.3 Retrieval diagnostic payload

The `retrieval` event should include only what is safe and useful:

```json
{
  "mode": "graph",
  "collection_revision": "pallets_flask__abc123__v2",
  "selected_chunks": [
    {
      "id": "...",
      "symbol": "src.flask.app.Flask.full_dispatch_request",
      "file_path": "src/flask/app.py",
      "start_line": 995,
      "end_line": 1022,
      "origin": "vector_seed",
      "score": 0.91,
      "reason": "high semantic and identifier overlap"
    }
  ],
  "graph": {"nodes": [], "edges": []}
}
```

Do not expose raw source text, provider credentials, complete prompts, or hidden chain-of-thought. The `reason` is a deterministic, brief retrieval rationale, not model reasoning.

### 11.4 Frontend changes

Update `ChatWindow.jsx` to:

1. parse SSE using a robust streaming parser;
2. populate the graph when the `retrieval` event arrives;
3. display selected chunk count and retrieval mode while the answer streams;
4. show a collapsible evidence panel with source location, symbol, origin, and short rationale;
5. render error events in the in-progress assistant message rather than adding a confusing duplicate message.

Update `GraphVisualizer.jsx` to color/style edges by resolution confidence and direction. Tooltips should identify caller/callee relationship and confidence category. Low-confidence and unresolved relationships should be hidden by default.

### 11.5 Ingestion refresh UX

After an ingestion request, return:

- repository source revision;
- collection revision;
- whether the source revision was already indexed;
- indexed chunk/edge counts;
- parser schema version.

The UI should clearly offer reuse of an existing immutable index or creation of a refreshed index. It must not imply that a previously indexed alias reflects the current GitHub default branch without checking revision.

### 11.6 Acceptance criteria

- Invalid retrieval modes receive a `422` validation response.
- A streamed answer works even if the retrieval event or answer data crosses arbitrary network chunk boundaries.
- Users can see exactly which source snippets were selected without reading backend logs.
- No hidden sentinel syntax appears in answer text.

## 12. Phase 7 - Observability and Operational Controls

### 12.1 Structured logs and metrics

Emit structured logs for:

- ingestion: source revision, file count, chunk count, edge count, rejected file count, duration;
- retrieval: collection revision, mode, candidate count, selected count, graph direction, score configuration version, duration;
- agent: iteration count, query hashes, novelty score, stop condition, provider-call duration;
- generation: context characters, response characters, duration, error class;
- evaluation: run ID and artifact output path.

Avoid logging API keys, full repository source, private prompts, or full answer context by default. Use hashes and bounded identifiers for correlation.

### 12.2 Configuration

Centralize configuration in a typed settings module, documented in `.env.example`:

```text
COLLECTION_SCHEMA_VERSION=2
VECTOR_SEED_COUNT=5
MAX_NEIGHBORS_PER_SEED=2
MAX_GRAPH_CANDIDATES=20
FINAL_CONTEXT_CHUNK_LIMIT=10
FINAL_CONTEXT_MAX_CHARS=60000
MIN_GRAPH_EDGE_CONFIDENCE=0.75
AGENT_MAX_SEARCHES=3
MIN_AGENT_NOVELTY_SCORE=0.05
RERANKER_ENABLED=false
```

Make benchmark settings explicit rather than inheriting accidental production defaults. A benchmark manifest must show effective values after all environment resolution.

## 13. Testing Plan

### 13.1 Unit tests

Add pytest tests for:

- URL/repository revision parsing and immutable collection-name creation;
- path normalization and canonical symbol construction;
- AST extraction across supported languages;
- graph resolution priority and ambiguity rejection;
- incoming/outgoing index construction;
- query intent routing;
- candidate scoring, deterministic tie-breaking, deduplication, and budget enforcement;
- agent stopping rules and evidence displacement;
- context builder ordering and character limit;
- Pydantic request validation and event serialization.

### 13.2 Integration tests

Use a tiny checked-in fixture repository and deterministic fake embedding/LLM adapters. The integration suite should:

- ingest the fixture into a disposable test collection;
- query graph mode and assert selected evidence and relationships;
- query agent mode and assert no unbounded accumulation;
- verify SSE event order: `retrieval`, zero or more `answer_delta`, then `complete`;
- test missing, legacy, and valid collection revisions;
- confirm re-ingestion of a changed fixture produces a distinct collection.

### 13.3 Frontend tests

Use the existing frontend toolchain or add an appropriate test runner. Cover:

- SSE parser behavior for arbitrary byte chunk boundaries;
- rendering retrieval diagnostics;
- graph reset on new question;
- errors during retrieval and answer generation;
- mode switch request payload;
- accessibility labels for retrieval controls and graph metadata.

### 13.4 Benchmark regression suite

Maintain a fast, no-provider regression fixture with expected chunks/edges. The full Ragas run remains a scheduled/manual quality gate because it is slower and uses external model APIs.

## 14. Rollout and Migration Sequence

### Milestone A - Baseline and contracts

1. Implement Phase 0 artifacts and generated report.
2. Add tests around the current retrieval behavior.
3. Establish an archived baseline run before changing any algorithm.

### Milestone B - Ingestion v2 and graph precision

1. Add repository manifests, schema versioning, canonical symbols, and new collection names.
2. Implement Python resolver and typed graph edges.
3. Add fresh v2 ingestion for the Flask benchmark source revision.
4. Verify graph fixture tests and inspect unresolved-edge rates.

### Milestone C - Deterministic selector

1. Add candidate models, intent routing, scores, deduplication, and budgets.
2. Keep the old expansion behind a short-lived comparison feature flag.
3. Run the benchmark and compare selected-context size, latency, and metrics.
4. Promote the selector to default only if it meets the acceptance criteria.

### Milestone D - Agent redesign

1. Implement structured critic feedback and evidence inventory.
2. Reuse the deterministic selector for every agent iteration.
3. Log and expose stop conditions.
4. Benchmark agentic mode separately from graph mode; do not assume it should be the default.

### Milestone E - Streaming and UI

1. Ship SSE server support alongside the current stream format only if a temporary compatibility window is needed.
2. Update the frontend parser and diagnostic views.
3. Remove sentinel parsing after the frontend migration is verified.

### Milestone F - Model experiments and operational hardening

1. Run the embedding/reranker matrix.
2. Select models based on measured quality, latency, cost, and deployment constraints.
3. Complete observability, documentation, and production configuration review.

## 15. Feature Flags and Rollback

Use server-side settings to control behavior during rollout:

```text
RETRIEVAL_SELECTOR_VERSION=v2
GRAPH_RESOLUTION_VERSION=v2
QUERY_STREAM_FORMAT=sse
AGENT_EVIDENCE_SELECTION_ENABLED=true
RERANKER_ENABLED=false
```

Do not delete v1 collections as part of rollout. New code should query the version declared in the collection manifest. Rollback consists of switching the alias/default collection and feature flags, not destructive data operations.

Every response should record effective retrieval versions in logs and evaluation manifests. This makes regressions traceable.

## 16. Definition of Done

The implementation is complete when all of the following are true:

- Evaluation results are persisted with enough provenance to reproduce the report without rerunning external evaluators.
- Each indexed collection is tied to a known repository source revision and schema version.
- Symbols and graph edges are module-aware, typed, bidirectionally indexed, and conservatively resolved.
- Graph mode selects a bounded, ranked evidence set instead of forwarding all direct neighbors.
- Agent mode retains a bounded, replaceable evidence set and can explain why it searched again or stopped.
- API streams use structured events and the UI presents retrieval provenance clearly.
- Unit, integration, and frontend tests cover the new contracts.
- The updated benchmark is run on the same corpus/question configuration as the baseline, and the generated report presents complete raw artifacts, caveats, and quality/latency tradeoffs.

## 17. Immediate Backlog

The following tickets are implementation-ready and should be completed in order:

1. **EVAL-1:** Add result-directory writer, manifest generator, per-question score persistence, and report generation to `eval/benchmark.py`.
2. **TEST-1:** Add a deterministic fixture suite that captures current vector/graph selection behavior.
3. **INGEST-1:** Capture source commit SHA and create immutable versioned collection names plus repository manifests.
4. **AST-1:** Add canonical module-aware symbol identities and source-span metadata.
5. **GRAPH-1:** Replace bare-name graph values with typed Python graph edges and reverse indexes.
6. **RETRIEVAL-1:** Add candidate scoring, deduplication, intent-aware neighbor selection, and configured context budgets.
7. **API-1:** Add typed retrieval mode validation and structured retrieval diagnostics.
8. **AGENT-1:** Add critic knowledge gaps, evidence inventory, replacement selection, and deterministic stopping rules.
9. **STREAM-1:** Migrate from sentinel-prefixed text to SSE and update `ChatWindow.jsx`.
10. **EVAL-2:** Run the baseline-vs-v2 benchmark, publish the generated report, and decide whether to test reranker/embedding variants.

