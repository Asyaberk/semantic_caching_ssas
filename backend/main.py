from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from backend.orchestrator import Orchestrator
from backend.demo_router import router as demo_router

app = FastAPI(
    title="SSAS Semantic Cache Seeder",
    description="Scans an SSAS Cube, generates question/MDX pairs using an LLM, and uploads them to Qdrant.",
    version="0.1.0",
)

# Allow requests from the frontend dev server (e.g. localhost:3001)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One orchestrator instance shared across all requests
_orchestrator = Orchestrator()

# Demo router (semantic cache query)
app.include_router(demo_router)

# Serve the frontend UI (must be mounted after defining API routes)
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the Admin UI at the root path."""
    return FileResponse(os.path.join(_frontend_dir, "index.html"))


@app.get("/demo", include_in_schema=False)
async def demo_ui():
    """Serve the Semantic Cache Demo UI."""
    return FileResponse(os.path.join(_frontend_dir, "demo.html"))


# ── Health ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Simple liveness check. The UI calls this to confirm the server is up."""
    return {"status": "ok", "message": "SSAS Cache Seeder is running."}


# ── Pipeline control ──────────────────────────────────────────────────────

@app.post("/pipeline/start")
async def start_pipeline():
    """
    Start the seeding pipeline in the background.
    Returns immediately; poll /pipeline/status for progress.
    """
    try:
        _orchestrator.start()
        return {"message": "Pipeline started."}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/pipeline/stop")
async def stop_pipeline():
    """
    Request a clean stop. The current batch will finish before halting.
    """
    _orchestrator.stop()
    return {"message": "Stop requested. Pipeline will halt after the current batch."}


@app.get("/pipeline/status")
async def pipeline_status():
    """
    Return the current pipeline state.
    The UI polls this endpoint to display live progress.
    """
    state = _orchestrator.get_state()
    return state.model_dump()
