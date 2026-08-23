from fastapi import APIRouter, HTTPException
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

def blocking_ingest(github_url: str) -> dict:
    res = clone_repo(github_url)
    if res["error"]:
        # Return error as plain dict — HTTPException cannot cross thread boundaries
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
    result = await asyncio.to_thread(blocking_ingest, req.github_url)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
