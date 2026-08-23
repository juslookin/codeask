import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.ingest import router as ingest_router
from api.query import router as query_router

app = FastAPI()

# Read allowed origins from the environment so the same binary works both
# locally (ALLOWED_ORIGINS=http://localhost:5173) and in production
# (ALLOWED_ORIGINS=https://your-app.vercel.app).
# Falls back to "*" if the variable is not set so a fresh dev install just works.
raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(query_router)