import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.routers import documents, chat

app = FastAPI(title="AMANLM", version="1.0.0")

# ALLOWED_ORIGINS: comma-separated list of allowed frontend origins.
# Set to your Vercel URL in Railway, e.g. "https://amanlm.vercel.app".
# Defaults to "*" for local development.
_raw = os.environ.get("ALLOWED_ORIGINS", "*")
_origins = [o.strip() for o in _raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])

STATIC_DIR = Path(__file__).parent.parent / "frontend" / "dist"

if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        index = STATIC_DIR / "index.html"
        return FileResponse(str(index))

    @app.get("/", include_in_schema=False)
    async def serve_root():
        return FileResponse(str(STATIC_DIR / "index.html"))
