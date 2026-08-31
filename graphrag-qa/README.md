# CodeAsk: Agentic GraphRAG Engine

CodeAsk is a full-stack, enterprise-grade AI application designed to ingest massive GitHub repositories and answer highly technical architectural questions. 

Unlike standard Retrieval-Augmented Generation (RAG) tools that blindly slice text based on character counts, CodeAsk uses **compiler-level Abstract Syntax Tree (AST) parsing** to build a call-graph over the codebase, then pairs dense vector search with topological graph traversal and an autonomous multi-step reasoning agent.

![CodeAsk UI](https://img.shields.io/badge/UI-3--Panel_React-blue) ![Language Support](https://img.shields.io/badge/Languages-Python_|_JS_|_TS-yellow) ![Eval](https://img.shields.io/badge/Evaluations-G--Eval-green)

---

## The Problem with Standard RAG for Code

Standard RAG pipelines use naive text splitters (e.g., recursive character splitting) to chunk documents. When applied to codebases, this approach fundamentally fails:
1. **Broken Semantics:** Functions and classes are arbitrarily sliced in half, destroying logic.
2. **Missing Dependencies:** A chunk containing a function call lacks the context of what that function actually does.
3. **Enterprise Scale Issues:** Dumping an entire monorepo into an LLM context window (even 2M+ tokens) results in extreme latency, exorbitant API costs, and the "Lost in the Middle" degradation phenomenon.

## The Solution: CodeAsk Architecture

CodeAsk solves these problems by treating code as data structures rather than strings.

### 1. Ingestion Pipeline (AST & Graph DB)
* **Multi-Language AST Chunking:** Uses `tree-sitter` and Python AST to parse repositories. It intelligently extracts exact `function_definition` and `class_declaration` blocks, preserving absolute structural boundaries.
* **Deterministic Graph Engine:** Scans the AST for `call_expression` nodes to build a directed NetworkX call graph — mapping caller/callee execution relationships across modules.
* **Vector Storage:** Embeds these semantic chunks into a **ChromaDB** vector database using **Gemini Embedding 2** (3,072-dim) with rich metadata (filepaths, start/end lines, parent classes).

### 2. Retrieval Pipelines
* **Naive Vector Search (`mode: "naive"`):** Semantic search over dense embeddings. Fast (14.75s latency), ideal for direct single-file function lookups.
* **Deterministic GraphRAG (`mode: "graph"`):** Vector search finds seed chunks, then a deterministic 1-hop expansion across the AST call-graph pulls in caller/callee context, boosting multi-file lifecycle flow accuracy.
* **Agentic RAG (`mode: "agent"`):** An autonomous multi-step reasoning agent powered by **DeepSeek-V4** that iteratively inspects symbol hierarchies, searches for missing dependencies, and verifies code evidence before synthesizing answers (achieving the highest overall accuracy: **0.8825**).

---

## Tech Stack

**Backend & AI Engine**
* **LangGraph & LangChain:** Multi-agent orchestration and LLM chain management.
* **Tree-sitter:** High-performance, language-agnostic AST parsing.
* **ChromaDB:** Local vector database for extremely fast semantic similarity search.
* **FastAPI:** High-throughput asynchronous REST API for the frontend and streaming text generation.
* **DeepSeek-V4 Flash (deepseek-v4-flash):** LLM reasoning model, via `langchain-deepseek`.
* **Gemini Embedding 2 (gemini-embedding-2):** Embeddings model.

**Frontend UI**
* **React 18 & Vite:** Lightning-fast frontend build tooling.
* **@xyflow/react (ReactFlow):** Physics-based interactive node graph rendering.
* **Tailwind CSS:** Utility-first styling for the IDE-like 3-panel dark mode interface.

**LLMOps & Evaluation**
* **Ragas:** Automated statistical evaluation framework for benchmarking RAG pipelines.

---

## Interactive 3-Panel IDE Interface

CodeAsk isn't just a chatbot; it's a dedicated workspace.
1. **Project Explorer (Left):** Recursively renders the ingested repository's file structure directly from ChromaDB metadata.
2. **Chat (Center):** Streams the answer in real time with precise, clickable code citations.
3. **Graph Visualizer (Right):** Uses algorithmic circular layouts to render the exact AST dependency network retrieved for your question — one-hop call/callee edges by default, or the agent's accumulated retrieval graph in `mode: "agent"`.

---

## Setup & Running

### 1. Backend Server
```bash
# From the project root
venv\Scripts\activate
cd backend
python -m uvicorn main:app --reload
```
*(Requires a `.env` file in the `backend/` directory with `DEEPSEEK_API_KEY` (for LLM), `GEMINI_API_KEY` (for embeddings), and `ALLOWED_ORIGINS=http://localhost:5173`)*

### 2. Frontend Application
```bash
# In a new terminal, from the project root
cd frontend
npm run dev
```

---

## Recommended Testing

1. Start both servers and open `http://localhost:5173`. 
2. Enter a valid GitHub repository URL (e.g., `https://github.com/pallets/flask` or `https://github.com/tiangolo/fastapi`).
3. Click **Ingest** and wait for the files to be cloned, AST-parsed, embedded, and stored in ChromaDB (1-3 minutes).
4. Ask a complex architectural question (e.g., *"How does Flask handle unhandled exceptions during request processing?"*) and watch the Graph Visualizer map the file dependencies!

---

## LLMOps & Benchmark Results

CodeAsk was rigorously evaluated against a 20-question benchmark on **Pallets/Flask** comparing three retrieval paradigms across two distinct industry evaluation frameworks:
1. **Ragas 0.4 Suite:** Statistical sentence-level decomposition with embedding distance.
2. **Unified G-Eval Suite:** 1-shot chain-of-thought LLM-as-a-judge rubric.

---

### 📊 Suite 1: Ragas Evaluation Framework

| Metric | Naive RAG | GraphRAG | Agentic RAG | Best Pipeline Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Context Precision** | 0.7938 | 0.7352 | **0.8800** | **+0.0862 (+11%)** 🚀 |
| **Faithfulness** | **0.9839** | 0.9671 | 0.9500 | -0.0170 (-2%) |
| **Answer Correctness** | 0.6636 | 0.6095 | **0.8775** | **+0.2139 (+32%)** 🚀 |
| **Median Latency (s)** | **14.75s** | 16.88s | 24.96s | +10.21s |
| **Avg Chunks Retrieved** | **5.0** | 8.0 | 8.0 | +3.0 chunks |

```bash
# Run Ragas evaluation
venv\Scripts\python.exe eval/benchmark.py
```

---

### 📊 Suite 2: Unified G-Eval (LLM-as-a-Judge)

| Metric | Naive RAG | GraphRAG | Agentic RAG | Best Pipeline Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Context Precision** | 0.8300 | 0.8300 | **0.8825** | **+0.0525 (+6.3%)** 🚀 |
| **Faithfulness** | 0.9450 | 0.9450 | **0.9550** | **+0.0100 (+1.1%)** 🚀 |
| **Answer Correctness** | 0.8325 | 0.8525 | **0.8825** | **+0.0500 (+6.0%)** 🚀 |
| **Median Latency (s)** | **14.75s** | 16.88s | 24.96s | +10.21s |
| **Avg Chunks Retrieved** | **5.0** | 8.0 | 8.0 | +3.0 chunks |

```bash
# Run G-Eval evaluation
venv\Scripts\python.exe eval/run_geval.py
```

---

### 🧠 Key Engineering Takeaways

* **Agentic RAG achieves highest overall accuracy across both suites:** By dynamically verifying evidence, citing exact file paths/line numbers, and self-correcting intermediate reasoning steps, Agentic RAG achieves **0.8775 – 0.8825** answer quality (+6% to +32% over Naive RAG).
* **GraphRAG dominates multi-file execution flows:** While Naive RAG is sufficient for isolated class lookups (e.g. `_AppCtxGlobals`, `config.py`), GraphRAG significantly outperforms Naive RAG on multi-file lifecycle flows:
  * **Exception Handling Lifecycle (`wsgi_app` → `handle_user_exception`):** GraphRAG **0.87 – 0.90** vs. Naive 0.40 – 0.85
  * **Blueprint Routing (`BlueprintSetupState` → `add_url_rule`):** GraphRAG **0.90** vs. Naive 0.80
  * **Request Lifecycle (`full_dispatch_request`):** GraphRAG **0.90** vs. Naive 0.85
* **Latency vs. Accuracy Trade-Off:**
  * **Naive RAG:** Lowest latency (**14.75s**), ideal for quick single-file lookups.
  * **GraphRAG:** Fast (**16.88s**), adds multi-hop caller/callee context with only +2.1s overhead.
  * **Agentic RAG:** Highest accuracy (**24.96s**), trades latency for deep reasoning and multi-step verification.