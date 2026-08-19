import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import chromadb
from llm.gemini import stream_answer
from llm.agent import run_agentic_retrieval

logger = logging.getLogger(__name__)

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    collection_name: str

def get_context(question: str, collection: str):
    # This invokes the LangGraph iterative multi-agent workflow
    return run_agentic_retrieval(question, collection)

@router.post("/query")
async def query(req: QueryRequest):
    try:
        context = await asyncio.to_thread(get_context, req.question, req.collection_name)
    except chromadb.errors.NotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Collection not found. Please ingest a repository first."
        )
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    return StreamingResponse(stream_answer(req.question, context), media_type="text/plain")
