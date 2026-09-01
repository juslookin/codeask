# CodeAsk

An AI codebase Q&A tool: point it at a public GitHub repo and it ingests the code with
tree-sitter AST parsing, builds a call graph alongside vector embeddings, and answers
questions about the codebase with cited, line-accurate context. Includes an offline Ragas
evaluation harness comparing a naive vector-search baseline against a graph-augmented
retriever and a LangGraph planner/critic agent.

**Full documentation, architecture, and setup instructions live in
[`graphrag-qa/README.md`](graphrag-qa/README.md).**

## Quick facts

- **Backend:** FastAPI, tree-sitter (AST parsing + call-graph construction), ChromaDB, Gemini
- **Frontend:** React + Vite
- **Languages supported:** Python, JavaScript, TypeScript, JSX, TSX
- **Retrieval modes:** naive vector search, graph-augmented ("GraphRAG"), and an agentic
  LangGraph planner/critic pipeline
- **Evaluation:** offline benchmark (`graphrag-qa/eval/`) scoring all three modes with Ragas
  metrics against a hand-written reference question set

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for production deployment instructions (Docker, Vercel, Render, Railway, VPS).
See [`graphrag-qa/README.md`](graphrag-qa/README.md) for local setup, architecture writeup, and evaluation results.

