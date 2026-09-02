import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.ingest import router as ingest_router
from api.query import router as query_router

app = FastAPI(
    title="CodeAsk API",
    description="Agentic GraphRAG API for codebase Q&A",
    version="1.0.0",
)

# Read allowed origins from the environment so the same binary works both
# locally (ALLOWED_ORIGINS=http://localhost:5173) and in production
# (ALLOWED_ORIGINS=https://your-app.vercel.app).
# Falls back to "*" if the variable is not set so a fresh dev install just works.
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

@app.on_event("startup")
def on_startup():
    print(f"=== CodeAsk API server is READY on port {os.getenv('PORT', 8000)} ===")

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
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)