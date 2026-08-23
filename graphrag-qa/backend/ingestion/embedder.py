import os
import chromadb
from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

# Anchor path to this file's directory so it resolves correctly regardless of
# which directory the process was launched from.  The old "./chroma_db" was
# CWD-relative, which meant the API server and eval/benchmark.py silently
# opened two different databases when run from different directories.
_HERE = os.path.dirname(os.path.abspath(__file__))
chroma_client = chromadb.PersistentClient(path=os.path.join(_HERE, "chroma_db"))


def embed_and_store(chunks: list[dict], graph: dict, owner_repo: str) -> str:
    collection_name = owner_repo.lower().replace(".", "_")[:60]

    try:
        existing = chroma_client.get_collection(collection_name)
        if existing.count() > 0:
            print(f"Collection '{collection_name}' already exists — skipping ingestion.")
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

    # Index-suffix the ID so graph nodes whose qualified_name collides (e.g.
    # subchunks of the same function) get distinct ChromaDB IDs.
    ids = [f"{c['file_path']}:{c['qualified_name']}:{i}" for i, c in enumerate(chunks)]

    metadatas = [
        {
            "file_path": c["file_path"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "qualified_name": c["qualified_name"],
            "base_qualified_name": c["base_qualified_name"],
            "callees": json.dumps(graph.get(c["qualified_name"], [])),
        }
        for c in chunks
    ]

    for i in range(0, len(chunks), 100):
        # upsert instead of add so re-ingesting the same repo doesn't raise
        # duplicate-ID errors.
        collection.upsert(
            documents=enriched_texts[i : i + 100],
            embeddings=embeddings[i : i + 100],
            ids=ids[i : i + 100],
            metadatas=metadatas[i : i + 100],
        )

    return collection_name


def get_collection(collection_name: str):
    return chroma_client.get_collection(collection_name)