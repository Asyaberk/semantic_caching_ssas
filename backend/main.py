from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SSAS Semantic Cache Seeder",
    description="Scans an SSAS Cube, generates question/MDX pairs using an LLM, and uploads them to Qdrant.",
    version="0.1.0",
)

# Allow requests from the frontend dev server (e.g. localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Simple liveness check. The UI calls this to confirm the server is up."""
    return {"status": "ok", "message": "SSAS Cache Seeder is running."}
