# CodeAsk: Agentic GraphRAG Engine

CodeAsk is a full-stack, enterprise-grade AI application designed to ingest massive GitHub repositories and answer highly technical architectural questions. 

Unlike standard Retrieval-Augmented Generation (RAG) tools that blindly slice text based on character counts, CodeAsk uses **compiler-level Abstract Syntax Tree (AST) parsing** combined with an autonomous **LangGraph multi-agent state machine** to build, traverse, and reason over a semantic graph of the codebase.

![CodeAsk UI](https://img.shields.io/badge/UI-3--Panel_React-blue) ![Language Support](https://img.shields.io/badge/Languages-Python_|_JS_|_TS-yellow) ![Eval](https://img.shields.io/badge/Evaluations-Ragas-green)

---

## ?? The Problem with Standard RAG for Code

Standard RAG pipelines use naive text splitters (e.g., recursive character splitting) to chunk documents. When applied to codebases, this approach fundamentally fails:
1. **Broken Semantics:** Functions and classes are arbitrarily sliced in half, destroying logic.
2. **Missing Dependencies:** A chunk containing a function call lacks the context of what that function actually does.
3. **Enterprise Scale Issues:** Dumping an entire monorepo into an LLM context window (even 2M+ tokens) results in extreme latency, exorbitant API costs, and the "Lost in the Middle" degradation phenomenon.

## ?? The Solution: CodeAsk Architecture

CodeAsk solves these problems by treating code as data structures rather than strings.

### 1. Ingestion Pipeline (AST & Graph DB)
* **Multi-Language AST Chunking:** Uses `tree-sitter` to parse Python, JavaScript, TypeScript, JSX, and TSX repositories. It intelligently extracts exact `function_definition` and `class_declaration` blocks, preserving absolute structural boundaries.
* **Deterministic Graph Engine:** Scans the AST for `call_expression` nodes to build a bidirectional dependency graph (e.g., mapping exactly which functions call `db.commit()`).
* **Vector Storage:** Embeds these semantic chunks into a **ChromaDB** vector database with rich metadata (filepaths, start/end lines, parent classes).

### 2. Retrieval Pipeline (LangGraph Agent)
* **Planner Agent:** Formulates a search strategy based on the user's architectural question.
* **Critic Agent:** An autonomous node that evaluates the retrieved chunks. If the retrieved context is insufficient, the Critic triggers a recursive feedback loop to perform deeper multi-hop traversal along the graph edges before synthesizing a final answer.
* **Graph Traversal:** Executes 1-hop expansions across the AST dependency graph to pull in caller/callee context that a vector search alone would miss.

---

## ?? Tech Stack

**Backend & AI Engine**
* **LangGraph & LangChain:** Multi-agent orchestration and LLM chain management.
* **Tree-sitter:** High-performance, language-agnostic AST parsing.
* **ChromaDB:** Local vector database for extremely fast semantic similarity search.
* **FastAPI:** High-throughput asynchronous REST API for the frontend and streaming text generation.
* **Google Gemini 1.5 Flash:** LLM reasoning and embedding models.

**Frontend UI**
* **React 18 & Vite:** Lightning-fast frontend build tooling.
* **@xyflow/react (ReactFlow):** Physics-based interactive node graph rendering.
* **Tailwind CSS:** Utility-first styling for the IDE-like 3-panel dark mode interface.

**LLMOps & Evaluation**
* **Ragas:** Automated statistical evaluation framework for benchmarking RAG pipelines.

---

## ?? Interactive 3-Panel IDE Interface

CodeAsk isn't just a chatbot; it's a dedicated workspace.
1. **Project Explorer (Left):** Recursively renders the ingested repository's file structure directly from ChromaDB metadata.
2. **Agentic Chat (Center):** Intercepts streaming JSON payloads from the LangGraph agent to provide real-time markdown answers with precise code citations.
3. **Graph Visualizer (Right):** Uses algorithmic circular layouts to render the exact AST dependency network the LangGraph agent traversed to find your answer.

---

## ??? Setup & Running

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

## ?? Recommended Testing

1. Start both servers and open `http://localhost:5173`. 
2. Enter a valid GitHub repository URL (e.g., `https://github.com/pallets/flask` or `https://github.com/tiangolo/fastapi`).
3. Click **Ingest** and wait for the files to be cloned, AST-parsed, embedded, and stored in ChromaDB (1-3 minutes).
4. Ask a complex architectural question (e.g., *"How does Flask handle unhandled exceptions during request processing?"*) and watch the Graph Visualizer map the file dependencies!

---

## ?? LLMOps & Evaluation Results

CodeAsk is rigorously, statistically evaluated against a dataset of highly technical questions using the **Ragas** framework. The benchmark suite measures two key metrics across Naive RAG, GraphRAG, and Agentic RAG:
* **Context Precision:** Did the engine retrieve the most highly relevant structural codebase files?
* **Faithfulness:** Is the final LLM answer strictly grounded in the retrieved code, with absolutely zero hallucinations?

| Architecture | Context Precision | Faithfulness |
| :--- | :--- | :--- |
| **Naive RAG** (Standard Vector Search) | 0.250 | **0.989** |
| **GraphRAG** (AST 1-Hop Expansion) | **0.263** | 0.947 |
| **Agentic RAG** (LangGraph Iterative) | 0.189 | 0.917 |

**Conclusion:** The deterministic one-hop expansion using Abstract Syntax Tree (AST) parsing (GraphRAG) yielded the highest **Context Precision** (0.263), successfully routing the most relevant codebase files to the top of the context window. Naive RAG achieved the highest **Faithfulness** (0.989), meaning 99% of its generated claims were strictly grounded in the code. Agentic RAG underperformed on this dataset, likely due to over-retrieval polluting the context window during its recursive loops.
