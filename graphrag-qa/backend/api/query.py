import logging
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import chromadb
from llm.gemini import stream_answer
from llm.agent import agent_app
from retrieval.context_builder import build_context

logger = logging.getLogger(__name__)

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    collection_name: str

def get_context_and_graph(question: str, collection: str):
    initial_state = {
        "question": question,
        "collection_name": collection,
        "context_chunks": [],
        "search_history": [],
        "iterations": 0
    }
    final_state = agent_app.invoke(initial_state)
    chunks = final_state.get("context_chunks", [])
    
    # Build graph payload
    nodes = []
    edges = []
    existing_nodes = set()
    for c in chunks:
        node_id = c["id"]
        if node_id not in existing_nodes:
            nodes.append({"id": node_id, "label": c.get("filename", node_id), "is_active": True})
            existing_nodes.add(node_id)
            
        for e in c.get("edges", []):
            edges.append({"source": e["source"], "target": e["target"]})
            if e["source"] not in existing_nodes:
                nodes.append({"id": e["source"], "label": e["source"].split(":")[-1], "is_active": False})
                existing_nodes.add(e["source"])
            if e["target"] not in existing_nodes:
                nodes.append({"id": e["target"], "label": e["target"].split(":")[-1], "is_active": False})
                existing_nodes.add(e["target"])

    context = build_context(chunks, [])
    return context, {"nodes": nodes, "edges": edges}

@router.post("/query")
async def query(req: QueryRequest):
    try:
        context, graph_data = await asyncio.to_thread(get_context_and_graph, req.question, req.collection_name)
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
