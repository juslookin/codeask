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
        "callees": json.dumps(graph.get(c["qualified_name"], []))
    } for c in chunks]

    for i in range(0, len(chunks), 100):
        collection.upsert(
            documents=enriched_texts[i:i + 100],
            embeddings=embeddings[i:i + 100],
            ids=ids[i:i + 100],
            metadatas=metadatas[i:i + 100]
        )
    return collection_name

def get_collection(collection_name: str):
    return chroma_client.get_collection(collection_name)
