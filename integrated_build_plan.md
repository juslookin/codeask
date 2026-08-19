# GraphRAG Codebase Q&A — Integrated Build Plan (v4 Final)

This is the fully integrated, corrected build plan incorporating all architectural enhancements: AST decorator fixes, `langchain-google-genai` migration (`gemini-2.0-flash`), non-blocking endpoints, robust RAGAS evaluation, and `owner/repo` namespacing.

---

## One-Time Setup (~1 hour)

### 1. Project scaffold
```bash
mkdir graphrag-qa && cd graphrag-qa
python -m venv venv
# Activate — Windows:
venv\Scripts\activate

mkdir -p backend/ingestion backend/retrieval backend/llm backend/api frontend eval
cd backend
```

### 2. requirements.txt
```text
fastapi
uvicorn[standard]
tree-sitter==0.21.3
tree-sitter-python==0.21.0
chromadb
sentence-transformers
langchain-google-genai
gitpython
python-dotenv
ragas
datasets
pandas
```
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
In `backend/.env`:
```text
GEMINI_API_KEY=your_key_here
```

### 4. Frontend scaffold
```bash
cd ../frontend
npm create vite@latest . -- --template react
npm install
npm install tailwindcss @tailwindcss/vite react-markdown
```
Update `vite.config.js` to include the Tailwind Vite plugin:
```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
})
```
Replace the contents of `src/index.css` with:
```css
@import "tailwindcss";
```

---

## Milestone 1 — Repo Ingestion (with URL validation)

### `backend/ingestion/github.py`
```python
import os
import shutil
import tempfile
import git
import re
import stat

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

IGNORE_DIRS = {
    "node_modules", "__pycache__", ".git", "venv", "env",
    "dist", "build", ".next", "vendor", "target", ".venv"
}
IGNORE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".pdf", ".zip", ".exe", ".bin", ".whl",
    ".pyc", ".so", ".dll", ".mp3", ".mp4", ".db", ".sqlite"
}
MAX_FILES = 500
MAX_SIZE_MB = 50

def parse_owner_repo(github_url: str) -> str:
    match = re.match(r"^https://github\.com/([\w.-]+)/([\w.-]+)/?$", github_url)
    if not match:
        raise ValueError("Invalid GitHub URL format")
    repo = match.group(2)
    if repo.endswith('.git'):
        repo = repo[:-4]
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
        return {"repo_path": None, "files": [], "error": f"Clone failed: {str(e)}", "owner_repo": None}

    total_size_mb = 0
    for root, _, files in os.walk(tmp_dir):
        for f in files:
            path = os.path.join(root, f)
            if not os.path.islink(path):
                total_size_mb += os.path.getsize(path)
    total_size_mb /= (1024 * 1024)

    if total_size_mb > MAX_SIZE_MB:
        shutil.rmtree(tmp_dir, onerror=remove_readonly)
        return {"repo_path": None, "files": [], "error": f"Repo exceeds {MAX_SIZE_MB}MB limit", "owner_repo": None}

    accepted = []
    for root, dirs, files in os.walk(tmp_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if os.path.splitext(f)[1].lower() not in IGNORE_EXTENSIONS:
                accepted.append(os.path.join(root, f))
                if len(accepted) >= MAX_FILES:
                    break
        if len(accepted) >= MAX_FILES:
            break

    return {"repo_path": tmp_dir, "files": accepted, "error": None, "owner_repo": owner_repo}
```

---

## Milestone 2 — AST Parsing with Decorator Support

### `backend/ingestion/ast_parser.py`
```python
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
import os

PY_LANGUAGE = Language(tspython.language(), "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)

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
            base_qualified_name = f"{class_name}.{raw_name}" if class_name else f"{module_name}.{raw_name}"
            
            # Walk up to capture decorators
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
                        "type": "function_subchunk"
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
                    "type": "function"
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

## Milestone 3 — Namespaced Graph (Base Name Resolution)

### `backend/ingestion/graph_builder.py`
```python
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language(), "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)

def build_graph(chunks: list[dict]) -> dict[str, list[str]]:
    raw_to_base: dict[str, list[str]] = {}
    base_to_qualified: dict[str, list[str]] = {}
    
    for c in chunks:
        if c["type"] in ("function", "function_subchunk"):
            raw_to_base.setdefault(c["name"], []).append(c["base_qualified_name"])
            base_to_qualified.setdefault(c["base_qualified_name"], []).append(c["qualified_name"])

    graph: dict[str, list[str]] = {c["qualified_name"]: [] for c in chunks if "function" in c["type"]}

    for chunk in chunks:
        if "function" not in chunk["type"]: continue
        
        tree = parser.parse(bytes(chunk["source_code"], "utf8"))
        callees = []

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

## Milestone 4 — Enriched Embeddings

### `backend/ingestion/embedder.py`
```python
import chromadb
from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def embed_and_store(chunks: list[dict], graph: dict, owner_repo: str) -> str:
    collection_name = owner_repo.lower().replace(".", "_")[:60]
    
    # Skip re-ingestion if collection already exists and is populated
    try:
        existing = chroma_client.get_collection(collection_name)
        if existing.count() > 0:
            print(f"Collection {collection_name} exists, skipping ingestion.")
            return collection_name
    except Exception:
        pass

    collection = chroma_client.create_collection(collection_name)
    if not chunks:
        return collection_name

    enriched_texts = [
        f"File: {c['file_path']}\nFunction: {c['base_qualified_name']}\n\n{c['source_code']}"
        for c in chunks
    ]
    embeddings = model.encode(enriched_texts, show_progress_bar=True).tolist()
    ids = [f"{c['file_path']}:{c['qualified_name']}" for c in chunks]

    metadatas = [{
        "file_path": c["file_path"],
        "start_line": c["start_line"],
        "end_line": c["end_line"],
        "qualified_name": c["qualified_name"],
        "base_qualified_name": c["base_qualified_name"],
        "callees": json.dumps(graph.get(c["qualified_name"], []))
    } for c in chunks]

    for i in range(0, len(chunks), 100):
        collection.add(
            documents=enriched_texts[i:i + 100],
            embeddings=embeddings[i:i + 100],
            ids=ids[i:i + 100],
            metadatas=metadatas[i:i + 100]
        )
    return collection_name

def get_collection(collection_name: str):
    return chroma_client.get_collection(collection_name)
```

---

## Milestone 5 — O(1) Traversal & Retrieval

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
    
    try:
        # Chunk into batches of 500 to avoid SQLite limit (999 variables)
        for batch_idx in range(0, len(callee_list), 500):
            batch = callee_list[batch_idx:batch_idx + 500]
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

    parts = []
    for c in unique[:15]:
        parts.append(f"# {c['file_path']} (lines {c['start_line']}-{c['end_line']})\n{c['source_code']}")
    return "\n\n---\n\n".join(parts)
```

---

## Milestone 6 — Gemini API & Async FastAPI

### `backend/llm/gemini.py`
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

SYSTEM_PROMPT = """You are an expert code assistant. Answer questions using ONLY the provided code chunks.
1. End every response with a ## Citations section.
2. Format citations strictly as: `filepath:startline-endline`
"""

def stream_answer(question: str, context: str):
    prompt = f"{SYSTEM_PROMPT}\n\n## Code Context\n{context}\n\n## Question\n{question}\n"
    for chunk in model.stream([HumanMessage(content=prompt)]):
        if chunk.content:
            yield chunk.content
```

### `backend/api/ingest.py`
```python
from fastapi import APIRouter
from pydantic import BaseModel
import asyncio
from ingestion.github import clone_repo
from ingestion.ast_parser import parse_repo
from ingestion.graph_builder import build_graph
from ingestion.embedder import embed_and_store
import shutil
from ingestion.github import remove_readonly

router = APIRouter()

class IngestRequest(BaseModel):
    github_url: str

def blocking_ingest(github_url: str):
    res = clone_repo(github_url)
    if res["error"]: return {"success": False, "error": res["error"]}

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
from retrieval.vector_search import vector_search
from retrieval.graph_traversal import expand_one_hop
from retrieval.context_builder import build_context
from llm.gemini import stream_answer

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    collection_name: str

def get_context(question: str, collection: str):
    seed = vector_search(question, collection)
    expanded = expand_one_hop(seed, collection)
    return build_context(seed, expanded)

@router.post("/query")
async def query(req: QueryRequest):
    context = await asyncio.to_thread(get_context, req.question, req.collection_name)
    
    return StreamingResponse(stream_answer(req.question, context), media_type="text/plain")
```

### `backend/main.py`
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from api.ingest import router as ingest_router
from api.query import router as query_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production replace with env array
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(query_router)
```

---

## Milestone 7 — React Markdown Frontend

### `frontend/src/components/ChatWindow.jsx`
```jsx
import { useState, useRef, useEffect } from "react"
import ReactMarkdown from 'react-markdown'
import CitationTag from "./CitationTag"

// ReactMarkdown allows custom renderers. We intercept inline code blocks for citations.
const renderers = {
  code({node, className, children, ...props}) {
    // Regex updated to support extensionless files like Dockerfile
    const match = /[\w/.-]+:\d+-\d+/.exec(String(children).trim())
    if (!className && match) {
      return <CitationTag citation={match[0]} />
    }
    return <code className={className} {...props}>{children}</code>
  }
}

export default function ChatWindow({ collectionName }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [streaming, setStreaming] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), [messages])

  async function handleSend() {
    if (!input.trim() || streaming) return
    const question = input.trim()
    setInput("")
    setMessages(prev => [...prev, { role: "user", text: question }])
    setStreaming(true)

    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, collection_name: collectionName })
      })
      if (!res.ok) throw new Error("Backend error")

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
    <div className="flex flex-col h-[600px] bg-gray-900 rounded-xl p-4">
      <div className="flex-1 overflow-y-auto mb-4">
         {messages.map((m, i) => (
           <div key={i} className={`p-3 rounded-lg my-2 text-white ${m.role === 'user' ? 'bg-blue-600 ml-auto w-3/4' : 'bg-gray-800'}`}>
             {m.role === "assistant" ? <ReactMarkdown components={renderers}>{m.text}</ReactMarkdown> : m.text}
           </div>
         ))}
         <div ref={bottomRef} />
      </div>
      <div className="flex gap-2">
        <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key==='Enter' && handleSend()} className="flex-1 p-2 rounded bg-gray-800 text-white" />
        <button onClick={handleSend} className="bg-blue-600 px-4 py-2 rounded text-white">Send</button>
      </div>
    </div>
  )
}
```

### `frontend/src/components/CitationTag.jsx`
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

## Milestone 8 — Robust RAGAS Eval

### `eval/benchmark.py`
```python
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
from llm.gemini import stream_answer
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0))
COLLECTION_NAME = "pallets_flask"

def safe_stream(question: str, context: str):
    # Wrapped with exponential backoff
    for attempt in range(3):
        try:
            return "".join(stream_answer(question, context))
        except Exception as e:
            if attempt == 2: raise e
            time.sleep(2 ** attempt)

def run_graphrag(question: str) -> dict:
    seed = vector_search(question, COLLECTION_NAME)
    expanded = expand_one_hop(seed, COLLECTION_NAME)
    context = build_context(seed, expanded)
    return {"answer": safe_stream(question, context), "contexts": [c["source_code"] for c in seed + expanded]}

def run_naive_rag(question: str) -> dict:
    chunks = vector_search(question, COLLECTION_NAME)
    context = build_context(chunks, [])
    return {"answer": safe_stream(question, context), "contexts": [c["source_code"] for c in chunks]}

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

    print("\nEvaluating Naive RAG...")
    naive_ds = build_dataset(questions, run_naive_rag)
    naive_scores = evaluate(naive_ds, metrics=metrics, llm=evaluator_llm)

    print("\nEvaluating GraphRAG...")
    graph_ds = build_dataset(questions, run_graphrag)
    graph_scores = evaluate(graph_ds, metrics=metrics, llm=evaluator_llm)

    print("\n" + "=" * 65)
    print(f"{'Metric':<28} {'Naive RAG':>12} {'GraphRAG':>12} {'Delta':>10}")
    print("-" * 65)
    
    naive_df, graph_df = naive_scores.to_pandas(), graph_scores.to_pandas()
    for metric in ["context_precision", "answer_relevancy"]:
        n_val = naive_df[metric].mean()
        g_val = graph_df[metric].mean()
        delta, pct = g_val - n_val, ((g_val - n_val) / max(n_val, 0.001)) * 100
        print(f"{metric:<28} {n_val:>12.3f} {g_val:>12.3f} {delta:>+8.3f} ({pct:+.0f}%)")
```
