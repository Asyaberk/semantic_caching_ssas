from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum


class ComplexityLevel(str, Enum):
    simple = "simple"
    medium = "medium"
    complex = "complex"


class PipelineStatus(str, Enum):
    idle = "idle"
    running = "running"
    stopped = "stopped"
    completed = "completed"
    error = "error"


class QAPair(BaseModel):
    """
    A question + MDX query pair.
    This is the unit of data uploaded to Qdrant.
    """
    id: Optional[str] = None
    question: str
    mdx: str
    cube_name: str = "Sales"
    dimensions_used: list[str] = []
    measures_used: list[str] = []
    complexity: ComplexityLevel = ComplexityLevel.medium
    language: str = "tr"
    generated_at: datetime = datetime.utcnow()
    langfuse_trace_id: Optional[str] = None
    # Tracks Qdrant upload result: "uploaded" | "failed" | "pending"
    upload_status: Optional[str] = None


class PipelineState(BaseModel):
    """
    Represents the current state of the generation pipeline.
    The UI polls this to display live progress.
    """
    status: PipelineStatus = PipelineStatus.idle
    questions_generated: int = 0
    mdx_generated: int = 0
    uploaded_to_qdrant: int = 0
    errors: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    current_task: Optional[str] = None
    logs: list[str] = []
