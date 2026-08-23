import logging
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import chromadb
from llm.gemini import stream_answer
from llm.agent import agent_app
from retrieval.vector_search import vector_search
from retrieval.graph_traversal import expand_one_hop
from retrieval.context_builder import build_context

logger = logging.getLogger(__name__)

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    collection_name: str
    # "graph": plain vector search + one-hop AST traversal (no extra LLM calls).
    # "agent": LangGraph planner/critic loop (more LLM calls, more latency).
    # Our own eval (see eval/benchmark.py, README) found "graph" beats "agent"
    # on both context_precision and faithfulness, so it's the default.
    mode: str = "graph"

def _build_graph_payload(chunks: list[dict]) -> dict:
    """Build the {nodes, edges} payload the frontend's GraphVisualizer expects.

    Chunk dicts (from vector_search / graph_traversal) carry `id`,
    `qualified_name`, `file_path` and `callees` (a list of *qualified names*,
    not ids). Edges have to be reconstructed by resolving each chunk's
    callees against the qualified_name -> id map of the chunks we actually
    retrieved — there is no separate "edges" field on a chunk.
    """
    nodes, edges = [], []
    existing_nodes = set()
    qname_to_ids: dict[str, list[str]] = {}
    for c in chunks:
        qname_to_ids.setdefault(c["qualified_name"], []).append(c["id"])

    for c in chunks:
        node_id = c["id"]
        if node_id not in existing_nodes:
            nodes.append({"id": node_id, "label": c.get("qualified_name", node_id), "is_active": True})
            existing_nodes.add(node_id)

        for callee_qname in c.get("callees", []):
            for target_id in qname_to_ids.get(callee_qname, []):
                if target_id != node_id:
                    edges.append({"source": node_id, "target": target_id})

    return {"nodes": nodes, "edges": edges}

def get_context_and_graph(question: str, collection: str, mode: str = "graph"):
    if mode == "agent":
        initial_state = {
            "question": question,
            "collection_name": collection,
            "context_chunks": [],
            "search_history": [],
            "iterations": 0
        }
        final_state = agent_app.invoke(initial_state)
        chunks = final_state.get("context_chunks", [])
    else:
        seed = vector_search(question, collection)
        expanded = expand_one_hop(seed, collection)
        chunks = seed + expanded

    graph_data = _build_graph_payload(chunks)
    context = build_context(chunks, [])
    return context, graph_data

@router.post("/query")
async def query(req: QueryRequest):
    try:
        context, graph_data = await asyncio.to_thread(
            get_context_and_graph, req.question, req.collection_name, req.mode
        )
    except chromadb.errors.NotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Collection not found. Please ingest a repository first."
        )
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    def event_stream():
        # First yield the graph data
        yield f"__GRAPH_START__\n{json.dumps(graph_data)}\n__GRAPH_END__\n"
        # Then stream the answer
        for chunk in stream_answer(req.question, context):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/plain")

@router.get('/api/files')
async def get_files(collection: str):
    try:
        from ingestion.embedder import chroma_client
        col = chroma_client.get_collection(collection)
        res = col.get(include=['metadatas'])
        paths = set()
        for m in res['metadatas']:
            if 'file_path' in m: paths.add(m['file_path'])
                
        def insert_node(tree, parts):
            if not parts: return
            part = parts[0]
            if len(parts) == 1:
                if not any(n['name'] == part for n in tree): tree.append({'name': part, 'type': 'file'})
            else:
                dir_node = next((n for n in tree if n['name'] == part and n['type'] == 'directory'), None)
                if not dir_node:
                    dir_node = {'name': part, 'type': 'directory', 'children': []}
                    tree.append(dir_node)
                insert_node(dir_node['children'], parts[1:])
                
        file_tree = []
        for path in sorted(paths):
            parts = path.replace('\\', '/').split('/')
            insert_node(file_tree, parts)
            
        return {'files': file_tree}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
