import logging
from ingestion.embedder import get_collection
import json

logger = logging.getLogger(__name__)

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
    except Exception as e:
        logger.warning(f'Graph expansion error: {e}')

    return expanded
