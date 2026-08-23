# GraphRAG Codebase Q&A — Integrated Build Plan (v5)

This plan reflects the **actual current implementation** as of August 2026. It supersedes v4, which documented a one-generation-older retrieval architecture and contained several bugs that caused hard failures on a fresh install.

### What changed from v4

| Area | v4 (plan) | v5 (actual + fixed) |
|---|---|---|
| `tree-sitter` pin | `==0.21.3` (old 2-arg API) | `>=0.22.0` (new single-arg API) |
| `chroma_db` path | CWD-relative `./chroma_db` | File-relative, always resolves correctly |
| CORS | Hardcoded `localhost:5173` | Env-driven `ALLOWED_ORIGINS` |
| Ingest UI | Missing — app hardcoded to Flask | `IngestForm.jsx` + dynamic `App.jsx` |
| Retrieval architecture | Vector search + one-hop expand | LangGraph agent (planner → retriever → critic) |
| Model | `gemini-2.0-flash` | `gemini-3.6-flash` |
| Benchmark arms | Naive RAG, GraphRAG | Naive RAG, GraphRAG, Agentic RAG |
| `.env.example` | Missing | Added for both `backend/` and `frontend/` |

---

## One-Time Setup

### 1. Project scaffold

```bash
git clone https://github.com/juslookin/codeask
cd codeask/graphrag-qa
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

cd backend
```

### 2. `backend/requirements.txt`

```
fastapi
uvicorn[standard]
tree-sitter>=0.22.0
tree-sitter-python>=0.22.0
chromadb
sentence-transformers
langchain-google-genai
langchain-core
langgraph
gitpython
python-dotenv
ragas
datasets
pandas
```

```bash
pip install -r requirements.txt
```

> **tree-sitter API note:** 0.22 removed the two-argument `Language(grammar, "name")` form and `parser.set_language()`. All files in this repo use the new API. Do not mix old and new forms.

### 3. Environment variables

Copy `backend/.env.example` → `backend/.env` and fill in:

```
GEMINI_API_KEY=your_key_here
ALLOWED_ORIGINS=http://localhost:5173
```

Copy `frontend/.env.example` → `frontend/.env`:

```
VITE_API_URL=http://localhost:8000
```

### 4. Frontend scaffold

```bash
cd ../frontend
npm create vite@latest . -- --template react
npm install
npm install tailwindcss @tailwindcss/vite react-markdown
```

`vite.config.js`:

```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
})
```

`src/index.css`:

```css
@import "tailwindcss";
```

---

## Milestone 1 — Repo Ingestion

### `backend/ingestion/github.py`

> No changes from v4. The file filter is a `.py`-only allowlist (not the blocklist described in v4 prose). `IGNORE_EXTENSIONS` was dead code and has been removed.

```python
import os, shutil, tempfile, git, re, stat

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

IGNORE_DIRS = {
    "node_modules", "__pycache__", ".git", "venv", "env",
    "dist", "build", ".next", "vendor", "target", ".venv"
}
MAX_FILES = 500
MAX_SIZE_MB = 50

def parse_owner_repo(github_url: str) -> str:
    match = re.match(r"^https://github\.com/([\w.-]+)/([\w.-]+)/?$", github_url)
    if not match:
        raise ValueError("Invalid GitHub URL format")
    repo = match.group(2).removesuffix('.git')
    return f"{match.group(1)}_{repo}"

def clone_repo(github_url: str) -> dict:
    try:
        owner_repo = parse_owner_repo(github_url)
    except ValueError as e:
        return {"repo_path": None, "files": [], "error": str(e), "owner_repo": None}

    tmp_dir = tempfile.mkdtemp()
    try:
        git.Repo.clone_from(github_url, tmp_dir, depth=1)
    except Exception as e:
        shutil.rmtree(tmp_dir, onerror=remove_readonly)
        return {"repo_path": None, "files": [], "error": f"Clone failed: {e}", "owner_repo": None}

    total_mb = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(tmp_dir)
        for f in files
        if not os.path.islink(os.path.join(r, f))
    ) / (1024 * 1024)

    if total_mb > MAX_SIZE_MB:
        shutil.rmtree(tmp_dir, onerror=remove_readonly)
        return {"repo_path": None, "files": [], "error": f"Repo exceeds {MAX_SIZE_MB}MB", "owner_repo": None}

    accepted = []
    for root, dirs, files in os.walk(tmp_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if os.path.splitext(f)[1].lower() == '.py':   # allowlist, not blocklist
                accepted.append(os.path.join(root, f))
                if len(accepted) >= MAX_FILES:
                    break
        if len(accepted) >= MAX_FILES:
            break

    return {"repo_path": tmp_dir, "files": accepted, "error": None, "owner_repo": owner_repo}
```

---

## Milestone 2 — AST Parsing

### `backend/ingestion/ast_parser.py`

**Fix from v4:** `Language` now takes one argument; `Parser` takes the language directly. The old `parser.set_language()` method does not exist in tree-sitter ≥0.22.

```python
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
import os

# ✅ tree-sitter >=0.22 API
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

GOD_FUNCTION_LINE_LIMIT = 500
SUB_CHUNK_WINDOW = 30
SUB_CHUNK_OVERLAP = 10

def extract_chunks_from_file(file_path: str, repo_root: str) -> list[dict]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except Exception:
        return []

    tree = parser.parse(bytes(source, "utf8"))
    lines = source.splitlines()
    relative_path = os.path.relpath(file_path, repo_root)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    chunks = []

    def get_name(node: Node) -> str:
        name_node = node.child_by_field_name("name")
        return name_node.text.decode("utf8") if name_node else "unknown"

    def make_chunks(node: Node, class_name: str | None = None):
        if node.type == "class_definition":
            raw_name = get_name(node)
            current_class = f"{class_name}.{raw_name}" if class_name else raw_name
            for child in node.children:
                make_chunks(child, class_name=current_class)
            return

        if node.type == "function_definition":
            raw_name = get_name(node)
            base_qualified_name = (
                f"{class_name}.{raw_name}" if class_name else f"{module_name}.{raw_name}"
            )

            span_node = node
            if node.parent and node.parent.type == "decorated_definition":
                span_node = node.parent

            start_line = span_node.start_point[0]
            end_line = span_node.end_point[0]
            chunk_lines = lines[start_line:end_line + 1]
            length = len(chunk_lines)

            if length > GOD_FUNCTION_LINE_LIMIT:
                step = SUB_CHUNK_WINDOW - SUB_CHUNK_OVERLAP
                for i in range(0, length, step):
                    window = chunk_lines[i:i + SUB_CHUNK_WINDOW]
                    abs_start = start_line + i
                    chunks.append({
                        "name": raw_name,
                        "base_qualified_name": base_qualified_name,
                        "qualified_name": f"{base_qualified_name}[{i}:{i+SUB_CHUNK_WINDOW}]",
                        "class_name": class_name,
                        "file_path": relative_path,
                        "start_line": abs_start + 1,
                        "end_line": abs_start + len(window),
                        "source_code": "\n".join(window),
                        "type": "function_subchunk",
                    })
            else:
                chunks.append({
                    "name": raw_name,
                    "base_qualified_name": base_qualified_name,
                    "qualified_name": base_qualified_name,
                    "class_name": class_name,
                    "file_path": relative_path,
                    "start_line": start_line + 1,
                    "end_line": end_line + 1,
                    "source_code": "\n".join(chunk_lines),
                    "type": "function",
                })
            return

        for child in node.children:
            make_chunks(child, class_name=class_name)

    make_chunks(tree.root_node)
    return chunks

def parse_repo(files: list[str], repo_root: str) -> list[dict]:
    all_chunks = []
    for f in files:
        all_chunks.extend(extract_chunks_from_file(f, repo_root))
    return all_chunks
```

---

## Milestone 3 — Call Graph

### `backend/ingestion/graph_builder.py`

**Fix from v4:** Same tree-sitter API fix as `ast_parser.py`.

```python
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython

# ✅ tree-sitter >=0.22 API
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

def build_graph(chunks: list[dict]) -> dict[str, list[str]]:
    raw_to_base: dict[str, list[str]] = {}
    base_to_qualified: dict[str, list[str]] = {}

    for c in chunks:
        if c["type"] in ("function", "function_subchunk"):
            raw_to_base.setdefault(c["name"], []).append(c["base_qualified_name"])
            base_to_qualified.setdefault(c["base_qualified_name"], []).append(c["qualified_name"])

    graph: dict[str, list[str]] = {
        c["qualified_name"]: []
        for c in chunks if "function" in c["type"]
    }

    for chunk in chunks:
        if "function" not in chunk["type"]: continue

        tree = parser.parse(bytes(chunk["source_code"], "utf8"))
        callees: list[str] = []

        def find_calls(node: Node):
            if node.type == "call":
                func_node = node.child_by_field_name("function")
                if func_node:
                    call_text = func_node.text.decode("utf8")
                    if "." in call_text:
                        raw_method = call_text.split(".")[-1]
                        caller_class = chunk.get("class_name")
                        preferred_base = f"{caller_class}.{raw_method}"
                        if caller_class and preferred_base in base_to_qualified:
                            for qname in base_to_qualified[preferred_base]:
                                if qname != chunk["qualified_name"]: callees.append(qname)
                            for child in node.children: find_calls(child)
                            return
                        for base in raw_to_base.get(raw_method, []):
                            for qname in base_to_qualified.get(base, []):
                                if qname != chunk["qualified_name"]: callees.append(qname)
                    else:
                        for base in raw_to_base.get(call_text, []):
                            for qname in base_to_qualified.get(base, []):
                                if qname != chunk["qualified_name"]: callees.append(qname)
            for child in node.children: find_calls(child)

        find_calls(tree.root_node)
        graph[chunk["qualified_name"]] = list(set(callees))

    return graph
```

---

## Milestone 4 — Embeddings

### `backend/ingestion/embedder.py`

**Fix from v4:** `chroma_db` path is now anchored to the module file, not CWD. Also uses `upsert` (not `add`) and index-suffixed IDs to handle re-ingestion and node collisions.

```python
import os, chromadb, json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ File-relative path — resolves correctly regardless of launch directory
_HERE = os.path.dirname(os.path.abspath(__file__))
chroma_client = chromadb.PersistentClient(path=os.path.join(_HERE, "chroma_db"))

def embed_and_store(chunks: list[dict], graph: dict, owner_repo: str) -> str:
    collection_name = owner_repo.lower().replace(".", "_")[:60]

    try:
        existing = chroma_client.get_collection(collection_name)
        if existing.count() > 0:
            print(f"Collection '{collection_name}' already exists — skipping.")
            return collection_name
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(collection_name)
    if not chunks:
        return collection_name

    enriched_texts = [
        f"File: {c['file_path']}\nFunction: {c['base_qualified_name']}\n\n{c['source_code']}"
        for c in chunks
    ]
    embeddings = model.encode(enriched_texts, show_progress_bar=True).tolist()
    ids = [f"{c['file_path']}:{c['qualified_name']}:{i}" for i, c in enumerate(chunks)]
    metadatas = [{
        "file_path": c["file_path"],
        "start_line": c["start_line"],
        "end_line": c["end_line"],
        "qualified_name": c["qualified_name"],
        "base_qualified_name": c["base_qualified_name"],
        "callees": json.dumps(graph.get(c["qualified_name"], [])),
    } for c in chunks]

    for i in range(0, len(chunks), 100):
        collection.upsert(
            documents=enriched_texts[i:i+100],
            embeddings=embeddings[i:i+100],
            ids=ids[i:i+100],
            metadatas=metadatas[i:i+100],
        )
    return collection_name

def get_collection(collection_name: str):
    return chroma_client.get_collection(collection_name)
```

---

## Milestone 5 — Retrieval Layer

No changes from v4. These files are correct as written.

### `backend/retrieval/vector_search.py`

```python
from ingestion.embedder import model, get_collection
import json

def vector_search(query: str, collection_name: str, n: int = 5) -> list[dict]:
    collection = get_collection(collection_name)
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n)
    return [{
        "id": results["ids"][0][i],
        "source_code": results["documents"][0][i],
        "file_path": meta["file_path"],
        "start_line": meta["start_line"],
        "end_line": meta["end_line"],
        "qualified_name": meta["qualified_name"],
        "callees": json.loads(meta.get("callees", "[]"))
    } for i, meta in enumerate(results["metadatas"][0])]
```

### `backend/retrieval/graph_traversal.py`

```python
from ingestion.embedder import get_collection
import json

def expand_one_hop(seed_chunks: list[dict], collection_name: str) -> list[dict]:
    collection = get_collection(collection_name)
    seed_ids = {c["id"] for c in seed_chunks}
    callee_names = set()
    for chunk in seed_chunks:
        callee_names.update(chunk.get("callees", []))
    if not callee_names: return []

    expanded = []
    callee_list = list(callee_names)
    for batch_idx in range(0, len(callee_list), 500):
        batch = callee_list[batch_idx:batch_idx+500]
        try:
            results = collection.get(where={"qualified_name": {"$in": batch}})
            for i, chunk_id in enumerate(results["ids"]):
                if chunk_id not in seed_ids:
                    meta = results["metadatas"][i]
                    expanded.append({
                        "id": chunk_id,
                        "source_code": results["documents"][i],
                        "file_path": meta["file_path"],
                        "start_line": meta["start_line"],
                        "end_line": meta["end_line"],
                        "qualified_name": meta["qualified_name"],
                        "callees": json.loads(meta.get("callees", "[]"))
                    })
        except Exception:
            pass
    return expanded
```

### `backend/retrieval/context_builder.py`

```python
def build_context(seed_chunks: list[dict], expanded_chunks: list[dict]) -> str:
    all_chunks = seed_chunks + expanded_chunks
    seen, unique = set(), []
    for c in all_chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)
    parts = [
        f"# {c['file_path']} (lines {c['start_line']}-{c['end_line']})\n{c['source_code']}"
        for c in unique[:15]
    ]
    return "\n\n---\n\n".join(parts)
```

---

## Milestone 6 — LangGraph Agent + FastAPI

### Actual retrieval architecture

The `/query` endpoint no longer calls vector search directly. It runs a LangGraph loop:

```
POST /query
    └── llm/agent.py
            ├── [planner node]    LLM decides retrieval strategy / sub-questions
            ├── [retriever node]  vector_search + expand_one_hop
            ├── [critic node]     LLM scores context quality
            └── (up to 3 iterations)
            └── [answer node]     stream_answer(question, final_context)
```

Each `/query` call can fire up to ~7 LLM calls (1 planner + up to 3× critic + 1 answer). Factor this into rate-limit and cost estimates.

### `backend/llm/gemini.py`

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",   # updated from gemini-2.0-flash
    temperature=0.3,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

SYSTEM_PROMPT = """You are an expert code assistant. Answer questions using ONLY the provided code chunks.
1. End every response with a ## Citations section.
2. Format citations strictly as: `filepath:startline-endline`
"""

def stream_answer(question: str, context: str):
    prompt = f"{SYSTEM_PROMPT}\n\n## Code Context\n{context}\n\n## Question\n{question}\n"
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        if chunk.content:
            yield chunk.content
```

### `backend/api/ingest.py`

No changes from v4.

```python
from fastapi import APIRouter
from pydantic import BaseModel
import asyncio, shutil
from ingestion.github import clone_repo, remove_readonly
from ingestion.ast_parser import parse_repo
from ingestion.graph_builder import build_graph
from ingestion.embedder import embed_and_store

router = APIRouter()

class IngestRequest(BaseModel):
    github_url: str

def blocking_ingest(github_url: str):
    res = clone_repo(github_url)
    if res["error"]:
        return {"success": False, "error": res["error"]}
    try:
        chunks = parse_repo(res["files"], res["repo_path"])
        graph = build_graph(chunks)
        collection_name = embed_and_store(chunks, graph, res["owner_repo"])
        return {"success": True, "collection_name": collection_name}
    finally:
        shutil.rmtree(res["repo_path"], onerror=remove_readonly)

@router.post("/ingest")
async def ingest(req: IngestRequest):
    return await asyncio.to_thread(blocking_ingest, req.github_url)
```

### `backend/api/query.py`

```python
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from llm.agent import run_agent   # LangGraph loop, not raw retrieval

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    collection_name: str

@router.post("/query")
async def query(req: QueryRequest):
    answer_stream = await asyncio.to_thread(
        run_agent, req.question, req.collection_name
    )
    return StreamingResponse(answer_stream, media_type="text/plain")
```

### `backend/main.py`

**Fix from v4:** CORS origins read from env, not hardcoded to `localhost:5173`.

```python
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.ingest import router as ingest_router
from api.query import router as query_router

app = FastAPI()

# ✅ Env-driven CORS — works locally and in production without code changes
raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(query_router)
```

---

## Milestone 7 — React Frontend

**Fix from v4:** `App.jsx` is now dynamic — it shows `IngestForm` first, then passes the returned `collection_name` to `ChatWindow`. The hardcoded `collectionName="pallets_flask"` has been removed. All fetch calls read `VITE_API_URL` from env.

### `frontend/src/App.jsx`

```jsx
import { useState } from "react"
import IngestForm from "./components/IngestForm"
import ChatWindow from "./components/ChatWindow"

export default function App() {
  const [collectionName, setCollectionName] = useState(null)

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center p-8">
      <h1 className="text-3xl font-bold mb-2">CodeAsk</h1>
      <p className="text-gray-400 mb-8 text-sm">
        Paste a GitHub repo URL and ask questions about the code.
      </p>
      {collectionName ? (
        <>
          <div className="mb-4 flex items-center gap-3">
            <span className="text-xs text-gray-500">
              Indexed: <code className="text-green-400">{collectionName}</code>
            </span>
            <button
              onClick={() => setCollectionName(null)}
              className="text-xs text-gray-500 underline hover:text-gray-300"
            >
              ← index another repo
            </button>
          </div>
          <ChatWindow collectionName={collectionName} />
        </>
      ) : (
        <IngestForm onIngested={setCollectionName} />
      )}
    </div>
  )
}
```

### `frontend/src/components/IngestForm.jsx` *(new file)*

```jsx
import { useState } from "react"

const API = import.meta.env.VITE_API_URL || "http://localhost:8000"

export default function IngestForm({ onIngested }) {
  const [url, setUrl] = useState("")
  const [status, setStatus] = useState(null)   // null | "loading" | "error"
  const [message, setMessage] = useState("")

  const isValid = (s) =>
    /^https:\/\/github\.com\/[\w.-]+\/[\w.-]+\/?$/.test(s.trim())

  async function handleSubmit() {
    const trimmed = url.trim()
    if (!trimmed) return
    if (!isValid(trimmed)) {
      setStatus("error")
      setMessage("Please enter a valid GitHub repo URL (https://github.com/owner/repo)")
      return
    }

    setStatus("loading")
    setMessage("Cloning and indexing — this takes 1–3 minutes for a typical repo…")

    try {
      const res = await fetch(`${API}/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_url: trimmed }),
      })
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      const data = await res.json()
      if (data.success) {
        onIngested(data.collection_name)
      } else {
        setStatus("error")
        setMessage(data.error || "Ingestion failed — check the backend logs.")
      }
    } catch (e) {
      setStatus("error")
      setMessage(e.message.startsWith("Server") ? e.message : "Could not reach the backend.")
    }
  }

  return (
    <div className="w-full max-w-xl">
      <div className="flex gap-2">
        <input
          value={url}
          onChange={(e) => { setUrl(e.target.value); if (status === "error") setStatus(null) }}
          onKeyDown={(e) => e.key === "Enter" && status !== "loading" && handleSubmit()}
          placeholder="https://github.com/owner/repo"
          disabled={status === "loading"}
          className="flex-1 p-3 rounded bg-gray-800 text-white placeholder-gray-500 disabled:opacity-50"
        />
        <button
          onClick={handleSubmit}
          disabled={status === "loading" || !url.trim()}
          className="bg-blue-600 px-5 py-3 rounded text-white font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {status === "loading" ? "Indexing…" : "Index"}
        </button>
      </div>
      {message && (
        <p className={`mt-3 text-sm ${status === "error" ? "text-red-400" : "text-gray-400"}`}>
          {message}
        </p>
      )}
      <p className="mt-4 text-xs text-gray-600">
        Public repos only · Python files · max 500 files / 50 MB
      </p>
    </div>
  )
}
```

### `frontend/src/components/ChatWindow.jsx`

**Fix from v4:** Reads API URL from `VITE_API_URL` env var.

```jsx
import { useState, useRef, useEffect } from "react"
import ReactMarkdown from "react-markdown"
import CitationTag from "./CitationTag"

const API = import.meta.env.VITE_API_URL || "http://localhost:8000"

const renderers = {
  code({ node, className, children, ...props }) {
    const match = /[\w/.-]+:\d+-\d+/.exec(String(children).trim())
    if (!className && match) return <CitationTag citation={match[0]} />
    return <code className={className} {...props}>{children}</code>
  }
}

export default function ChatWindow({ collectionName }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [streaming, setStreaming] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }) }, [messages])

  async function handleSend() {
    if (!input.trim() || streaming) return
    const question = input.trim()
    setInput("")
    setMessages(prev => [...prev, { role: "user", text: question }])
    setStreaming(true)

    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, collection_name: collectionName })
      })
      if (!res.ok) throw new Error(`Backend error ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let assistantText = ""
      setMessages(prev => [...prev, { role: "assistant", text: "" }])

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        assistantText += decoder.decode(value, { stream: true })
        setMessages(prev => {
          const updated = [...prev]
          updated[updated.length - 1] = { role: "assistant", text: assistantText }
          return updated
        })
      }
    } catch {
      setMessages(prev => [...prev, { role: "assistant", text: "Error connecting to backend." }])
    } finally {
      setStreaming(false)
    }
  }

  return (
    <div className="flex flex-col h-[600px] w-full max-w-2xl bg-gray-900 rounded-xl p-4">
      <div className="flex-1 overflow-y-auto mb-4 space-y-2">
        {messages.length === 0 && (
          <p className="text-gray-600 text-sm text-center mt-8">Ask anything about the indexed codebase.</p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`p-3 rounded-lg text-white text-sm ${
            m.role === "user" ? "bg-blue-600 ml-auto max-w-[75%]" : "bg-gray-800 max-w-full"
          }`}>
            {m.role === "assistant"
              ? <ReactMarkdown components={renderers}>{m.text}</ReactMarkdown>
              : m.text}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <div className="flex gap-2">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleSend()}
          disabled={streaming}
          placeholder={streaming ? "Thinking…" : "Ask a question about the code…"}
          className="flex-1 p-2 rounded bg-gray-800 text-white placeholder-gray-500 disabled:opacity-50"
        />
        <button
          onClick={handleSend}
          disabled={streaming || !input.trim()}
          className="bg-blue-600 px-4 py-2 rounded text-white hover:bg-blue-500 disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  )
}
```

### `frontend/src/components/CitationTag.jsx`

No changes from v4.

```jsx
export default function CitationTag({ citation }) {
  return (
    <span className="bg-blue-800 text-xs px-2 py-1 rounded ml-1 cursor-pointer hover:bg-blue-700">
      {citation}
    </span>
  )
}
```

---

## Milestone 8 — RAGAS Evaluation (3-arm)

**Fix from v4:** The agentic arm now returns a list of chunks (not a single joined string), so `context_precision` is scored on the same K as the other arms.

### `eval/benchmark.py`

```python
import json, sys, os, time
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, "..", "backend", ".env"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "backend"))

from retrieval.vector_search import vector_search
from retrieval.graph_traversal import expand_one_hop
from retrieval.context_builder import build_context
from llm.gemini import stream_answer
from llm.agent import run_agent_with_chunks  # returns (answer_str, chunk_list)
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
)
COLLECTION_NAME = "pallets_flask"

def safe_stream(question: str, context: str) -> str:
    for attempt in range(3):
        try:
            return "".join(stream_answer(question, context))
        except Exception as e:
            if attempt == 2: raise e
            time.sleep(2 ** attempt)

def run_naive_rag(question: str) -> dict:
    chunks = vector_search(question, COLLECTION_NAME)
    return {"answer": safe_stream(question, build_context(chunks, [])),
            "contexts": [c["source_code"] for c in chunks]}

def run_graphrag(question: str) -> dict:
    seed = vector_search(question, COLLECTION_NAME)
    expanded = expand_one_hop(seed, COLLECTION_NAME)
    return {"answer": safe_stream(question, build_context(seed, expanded)),
            "contexts": [c["source_code"] for c in seed + expanded]}

def run_agentic_rag(question: str) -> dict:
    # ✅ Fixed: run_agent_with_chunks returns (answer, list[chunk])
    # so context_precision is scored on K=N, same as naive and graphrag
    answer, chunks = run_agent_with_chunks(question, COLLECTION_NAME)
    return {"answer": answer, "contexts": [c["source_code"] for c in chunks]}

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
    pipelines = [
        ("Naive RAG",   run_naive_rag),
        ("GraphRAG",    run_graphrag),
        ("Agentic RAG", run_agentic_rag),
    ]

    results = {}
    for name, fn in pipelines:
        print(f"\nEvaluating {name}...")
        ds = build_dataset(questions, fn)
        results[name] = evaluate(ds, metrics=metrics, llm=evaluator_llm).to_pandas()

    print("\n" + "=" * 75)
    print(f"{'Metric':<28}", end="")
    for name, _ in pipelines:
        print(f" {name:>14}", end="")
    print()
    print("-" * 75)
    for metric in ["context_precision", "answer_relevancy"]:
        print(f"{metric:<28}", end="")
        for name, _ in pipelines:
            print(f" {results[name][metric].mean():>14.3f}", end="")
        print()
```

---

## Running Locally

```bash
# Terminal 1 — backend
cd graphrag-qa/backend
uvicorn main:app --reload

# Terminal 2 — frontend
cd graphrag-qa/frontend
npm run dev
```

Open `http://localhost:5173`, paste a public GitHub repo URL, wait for indexing (~1–3 min), then ask questions.

---

## Deployment

**Backend** (e.g. Railway):

```
GEMINI_API_KEY=your_key_here
ALLOWED_ORIGINS=https://your-frontend.vercel.app
```

**Frontend** (e.g. Vercel):

```
VITE_API_URL=https://your-backend.railway.app
```

---

## Known Limitations

- Only indexes `.py` files. Multi-language support requires additional tree-sitter grammar packages and per-extension dispatch in `ast_parser.py`.
- The LangGraph agent adds ~3–8 s latency and consumes significantly more Gemini quota per query than the v4 retrieval path.
- ChromaDB is embedded and single-process — not suitable for concurrent multi-user load without a hosted vector DB.
- Repos > 50 MB or > 500 Python files are rejected at the ingestion endpoint.