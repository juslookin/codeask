import os
import chromadb
from google import genai
from google.genai import types
import json
import numpy as np
from dotenv import load_dotenv

# Ensure .env is loaded from backend directory regardless of process working directory
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BACKEND_DIR, ".env"))
load_dotenv()

GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
# Gemini Embedding 2 — 3072-dim, state-of-the-art code/text embeddings.
# https://ai.google.dev/gemini-api/docs/models#text-embedding


EMBED_BATCH_SIZE = 100  # Max texts per Gemini embed_content call


class GeminiEmbedder:
    """Thin wrapper around the Gemini Embedding API that exposes the same
    `.encode(texts)` interface as SentenceTransformer so the rest of the
    codebase (vector_search.py, embedder.py) requires zero changes."""

    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL, batch_size: int = 100):
        # The new standard model (Gemini Embedding 2)
        self.model_name = model_name
        self.batch_size = batch_size
        self._client = None

    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set.")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> np.ndarray:
        """Embed a list of texts and return an (N, D) float32 numpy array.

        Batches texts into groups of EMBED_BATCH_SIZE to minimize API calls.
        For example, 200 texts → 2 API calls instead of 200.
        """
        all_embeddings: list[list[float]] = []
        total = len(texts)

        import time
        for batch_start in range(0, total, EMBED_BATCH_SIZE):
            batch = texts[batch_start : batch_start + EMBED_BATCH_SIZE]
            if show_progress_bar:
                batch_end = min(batch_start + EMBED_BATCH_SIZE, total)
                print(f"\rEmbedding {batch_end}/{total}...", end="", flush=True)

            # Retry loop for 429 Rate Limits (RPM or TPM)
            for attempt in range(5):
                try:
                    result = self.client.models.embed_content(
                        model=self.model_name,
                        contents=[[t] for t in batch],
                        config=types.EmbedContentConfig(task_type=task_type),
                    )
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        if attempt == 4: raise e
                        if show_progress_bar: print(f"\nTPM/RPM rate limit hit, sleeping for 60s to reset quota...")
                        time.sleep(60) # Full minute reset for Tokens-Per-Minute
                    else:
                        raise e

            all_embeddings.extend(e.values for e in result.embeddings)
            
            # Base delay to pace out the chunks
            time.sleep(2)

        if show_progress_bar and total > 0:
            print()  # newline after progress
        return np.array(all_embeddings, dtype=np.float32)


model = GeminiEmbedder()

# Anchor path to this file's directory so it resolves correctly regardless of
# which directory the process was launched from.  The old "./chroma_db" was
# CWD-relative, which meant the API server and eval/benchmark.py silently
# opened two different databases when run from different directories.
from chromadb.config import Settings

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CHROMA_PATH = os.path.join(_HERE, "chroma_db")
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", _DEFAULT_CHROMA_PATH)
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False, is_persistent=True)
)


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