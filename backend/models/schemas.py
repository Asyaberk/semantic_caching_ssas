from pydantic import BaseModel
from typing import Optional
from enum import Enum


class ComplexityLevel(str, Enum):
    simple  = "simple"
    medium  = "medium"
    complex = "complex"


class PipelineStatus(str, Enum):
    idle      = "idle"
    running   = "running"
    stopped   = "stopped"
    completed = "completed"
    error     = "error"


class QAPair(BaseModel):
    """
    A question + MDX query pair.
    This is the unit of data that gets embedded and uploaded to Qdrant.
    """
    id:               Optional[str]   = None
    question:         str
    mdx:              str
    cube_name:        str             = "Sales"
    dimensions_used:  list[str]       = []
    measures_used:    list[str]       = []
    complexity:       ComplexityLevel = ComplexityLevel.medium
    language:         str             = "en"
    langfuse_trace_id: Optional[str]  = None
    upload_status:    Optional[str]   = "pending"
    # Template fields — parameterised MDX for entity-independent caching
    mdx_template:     Optional[str]   = None   # MDX with {{YEAR}}, {{COUNTRY}} etc.
    entity_map:       Optional[dict]  = None   # {"YEAR": "2025", "COUNTRY": "Turkey"}


class PipelineState(BaseModel):
    """
    Live state of the seeding pipeline.
    Updated in real-time by the Orchestrator; served to the UI via FastAPI.
    """
    status:               PipelineStatus = PipelineStatus.idle
    current_cube:         Optional[str]  = None
    total_cubes:          int            = 0

    # Cumulative counters (grand total across all cubes)
    questions_generated:  int            = 0
    mdx_generated:        int            = 0
    uploaded_count:       int            = 0   # grand total in Qdrant

    # Per-cube progress
    cube_uploaded_count:  int            = 0   # uploaded for the current cube only
    cube_target:          int            = 0   # target per cube (= settings.questions_per_cube)
    cube_progress:        dict           = {}  # {cube_name: uploaded_count}

    # Error tracking
    last_error:           Optional[str]  = None

    # Timestamps (ISO-8601 strings for easy JSON serialisation)
    started_at:           Optional[str]  = None
    last_updated:         Optional[str]  = None
