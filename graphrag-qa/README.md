# CodeAsk: Agentic GraphRAG Engine

CodeAsk is a full-stack, enterprise-grade AI application designed to ingest massive GitHub repositories and answer highly technical architectural questions. 

Unlike standard Retrieval-Augmented Generation (RAG) tools that blindly slice text based on character counts, CodeAsk uses **compiler-level Abstract Syntax Tree (AST) parsing** to build a call-graph over the codebase, then retrieves by default with a deterministic one-hop graph traversal. An optional **LangGraph multi-agent (planner/critic) mode** is also available for harder multi-hop questions — though our own Ragas eval (below) found the deterministic traversal outperforms it on this benchmark, which is why it's the default rather than the other way around.

![CodeAsk UI](https://img.shields.io/badge/UI-3--Panel_React-blue) ![Language Support](https://img.shields.io/badge/Languages-Python_|_JS_|_TS-yellow) ![Eval](https://img.shields.io/badge/Evaluations-Ragas-green)

---

## The Problem with Standard RAG for Code

Standard RAG pipelines use naive text splitters (e.g., recursive character splitting) to chunk documents. When applied to codebases, this approach fundamentally fails:
1. **Broken Semantics:** Functions and classes are arbitrarily sliced in half, destroying logic.
2. **Missing Dependencies:** A chunk containing a function call lacks the context of what that function actually does.
3. **Enterprise Scale Issues:** Dumping an entire monorepo into an LLM context window (even 2M+ tokens) results in extreme latency, exorbitant API costs, and the "Lost in the Middle" degradation phenomenon.

## The Solution: CodeAsk Architecture

CodeAsk solves these problems by treating code as data structures rather than strings.

### 1. Ingestion Pipeline (AST & Graph DB)
* **Multi-Language AST Chunking:** Uses `tree-sitter` to parse Python, JavaScript, TypeScript, JSX, and TSX repositories. It intelligently extracts exact `function_definition` and `class_declaration` blocks, preserving absolute structural boundaries.
* **Deterministic Graph Engine:** Scans the AST for `call_expression` nodes to build a bidirectional dependency graph (e.g., mapping exactly which functions call `db.commit()`).
* **Vector Storage:** Embeds these semantic chunks into a **ChromaDB** vector database with rich metadata (filepaths, start/end lines, parent classes).

### 2. Retrieval Pipeline
* **Default (`mode: "graph"`):** Vector search finds seed chunks, then a deterministic 1-hop expansion across the AST call-graph pulls in caller/callee context a vector search alone would miss. No extra LLM calls, lowest latency, and the best-performing mode in our eval below.
* **Optional (`mode: "agent"`):** A LangGraph Planner/Critic loop that iterates up to 3 rounds, deciding whether to search again before answering. More thorough on paper, but slower and costlier (up to ~7 LLM calls per question) — and it underperformed the deterministic mode on our benchmark, so it's opt-in rather than default.

---

## Tech Stack

**Backend & AI Engine**
* **LangGraph & LangChain:** Multi-agent orchestration and LLM chain management.
* **Tree-sitter:** High-performance, language-agnostic AST parsing.
* **ChromaDB:** Local vector database for extremely fast semantic similarity search.
* **FastAPI:** High-throughput asynchronous REST API for the frontend and streaming text generation.
* **Google Gemini 3.5 Flash-Lite:** LLM reasoning model, via `langchain-google-genai`.

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
*(Requires a `.env` file in the `backend/` directory with `GEMINI_API_KEY` and `ALLOWED_ORIGINS=http://localhost:5173`)*

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

## LLMOps & Evaluation Results

CodeAsk is rigorously, statistically evaluated against a dataset of highly technical questions using the **Ragas** framework. The benchmark suite measures two key metrics across Naive RAG, GraphRAG, and Agentic RAG:
* **Context Precision:** Did the engine retrieve the most highly relevant structural codebase files?
* **Faithfulness:** Is the final LLM answer strictly grounded in the retrieved code, with absolutely zero hallucinations?

| Architecture | Context Precision | Faithfulness |
| :--- | :--- | :--- |
| **Naive RAG** (Standard Vector Search) | 0.250 | **0.989** |
| **GraphRAG** (AST 1-Hop Expansion) | **0.263** | 0.947 |
| **Agentic RAG** (LangGraph Iterative) | 0.189 | 0.917 |

**Conclusion:** The deterministic one-hop expansion using Abstract Syntax Tree (AST) parsing (GraphRAG) yielded the highest **Context Precision** (0.263), successfully routing the most relevant codebase files to the top of the context window. Naive RAG achieved the highest **Faithfulness** (0.989), meaning 99% of its generated claims were strictly grounded in the code. Agentic RAG underperformed on this dataset, likely due to over-retrieval polluting the context window during its recursive loops.

Because of this, the `/query` endpoint defaults to the deterministic GraphRAG pipeline (`mode: "graph"`). The LangGraph agent is still implemented and available via `mode: "agent"` for cases that need iterative multi-hop search, but it's opt-in rather than default — the eval, not intuition, decided which mode ships as the default.

Caveat worth being upfront about: this benchmark is 10 questions against one repository (Flask), one language (Python). Treat the ranking as directional, not conclusive — a larger, multi-repo, multi-language question set is the obvious next step before leaning on these numbers too hard.
