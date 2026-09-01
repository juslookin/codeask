"""Vector search with query embedding caching.

Uses the centralized GeminiEmbedder from ingestion.embedder and adds an
LRU cache on query embeddings to avoid redundant Gemini API calls for
repeated or follow-up questions.
"""

from functools import lru_cache
from ingestion.embedder import model, get_collection
import json
import numpy as np

# Cache the last 64 unique query embeddings.  Covers repeat questions,
# follow-ups, and the agentic loop re-ranking against the original question.
@lru_cache(maxsize=64)
def _embed_query_cached(query: str) -> tuple:
    """Embed a query and return as a tuple (hashable for lru_cache)."""
    vec = model.encode([query], task_type="RETRIEVAL_QUERY")[0]
    return tuple(vec.tolist())


def embed_query(query: str) -> np.ndarray:
    """Public helper: returns the cached query embedding as a numpy array."""
    return np.array(_embed_query_cached(query), dtype=np.float32)


def vector_search(query: str, collection_name: str, n: int = 5) -> tuple[list[dict], np.ndarray]:
    """Run vector search and return (chunks, query_embedding).

    The query embedding is returned so downstream consumers (selector.py,
    agent.py) can reuse it without making a redundant API call.
    """
    collection = get_collection(collection_name)
    query_vec = embed_query(query)
    results = collection.query(query_embeddings=[query_vec.tolist()], n_results=n)

    chunks = [{
        "id": results["ids"][0][i],
        "source_code": results["documents"][0][i],
        "file_path": meta["file_path"],
        "start_line": meta["start_line"],
        "end_line": meta["end_line"],
        "qualified_name": meta["qualified_name"],
        "callees": json.loads(meta.get("callees", "[]"))
    } for i, meta in enumerate(results["metadatas"][0])]

    return chunks, query_vec
