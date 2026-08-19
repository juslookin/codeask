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
