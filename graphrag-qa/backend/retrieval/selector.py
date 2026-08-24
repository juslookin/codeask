"""Evidence selector: ranks graph-expanded candidate chunks by query relevance
and caps total context to avoid context bloat.

Used by both the deterministic graph pipeline and the agentic retriever to
ensure only the most query-relevant expansion chunks make it into the LLM
context window — fixing the unbounded expand_one_hop accumulation bug.
"""

import numpy as np
from ingestion.embedder import get_collection

MAX_CONTEXT_CHUNKS = 8


def _fetch_embeddings_from_chromadb(
    chunk_ids: list[str], collection_name: str
) -> dict[str, np.ndarray]:
    """Fetch stored embeddings from ChromaDB instead of re-embedding via API.

    ChromaDB stores embeddings at ingestion time, so we can retrieve them
    for free — no Gemini API call needed.
    """
    if not chunk_ids:
        return {}
    collection = get_collection(collection_name)
    results = collection.get(ids=chunk_ids, include=["embeddings"])
    return {
        cid: np.array(emb, dtype=np.float32)
        for cid, emb in zip(results["ids"], results["embeddings"])
    }


def select_top_k(
    query: str,
    seed_chunks: list[dict],
    candidate_chunks: list[dict],
    k: int = MAX_CONTEXT_CHUNKS,
    query_embedding: np.ndarray | None = None,
    collection_name: str | None = None,
) -> list[dict]:
    """Keep all seed chunks, rank candidates by cosine similarity to query, cap at k total.

    Args:
        query: The user's original question text (used as fallback if query_embedding is None).
        seed_chunks: Chunks from the initial vector search (highest relevance).
        candidate_chunks: Chunks from graph expansion or agent iterations.
        k: Maximum total chunks to return.
        query_embedding: Pre-computed query embedding from vector_search.
            Avoids a redundant Gemini API call.
        collection_name: Collection to fetch stored candidate embeddings from.
            When provided, fetches embeddings from ChromaDB instead of
            re-embedding candidate source code via the Gemini API.

    Returns:
        seeds + top-ranked candidates, capped at k chunks total.
    """
    if not candidate_chunks:
        return seed_chunks[:k]

    # Get query embedding — reuse if provided, otherwise compute (with cache)
    if query_embedding is None:
        from retrieval.vector_search import embed_query
        query_vec = embed_query(query)
    else:
        query_vec = query_embedding

    # Get candidate embeddings — from ChromaDB if possible, else re-embed
    if collection_name:
        cand_ids = [c["id"] for c in candidate_chunks]
        id_to_emb = _fetch_embeddings_from_chromadb(cand_ids, collection_name)
        cand_vecs = np.array(
            [id_to_emb.get(c["id"], np.zeros_like(query_vec)) for c in candidate_chunks],
            dtype=np.float32,
        )
    else:
        # Fallback: re-embed (costs API calls, but ensures correctness)
        from ingestion.embedder import model
        cand_vecs = model.encode([c["source_code"] for c in candidate_chunks])

    # Cosine similarity
    norms = np.linalg.norm(cand_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8
    sims = cand_vecs @ query_vec / norms

    # Rank candidates by descending similarity
    ranked = [c for c, _ in sorted(zip(candidate_chunks, sims), key=lambda x: -x[1])]

    # Budget: how many expansion slots remain after seeds
    budget = max(k - len(seed_chunks), 0)
    return seed_chunks + ranked[:budget]
