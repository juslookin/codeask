import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.ingest import router as ingest_router
from api.query import router as query_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    port = os.getenv("PORT", "8000")
    print(f"=== CodeAsk API server is READY on port {port} ===")
    yield

app = FastAPI(
    title="CodeAsk API",
    description="Agentic GraphRAG API for codebase Q&A",
    version="1.0.0",
    lifespan=lifespan,
)

raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
if not origins:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True if "*" not in origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "service": "CodeAsk API", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

app.include_router(ingest_router)
app.include_router(query_router)

if __name__ == "__main__":
    import uvicorn
    raw_port = os.getenv("PORT", "8000")
    if not str(raw_port).isdigit():
        raw_port = 8000
    port = int(raw_port)
    print(f"Starting Uvicorn on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)