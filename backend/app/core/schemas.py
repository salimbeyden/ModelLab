from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- Enums ---

class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# --- Plugin / Registry Schemas ---

class PluginMeta(BaseModel):
    id: str
    name: str
    supported_tasks: List[TaskType]
    description: str = ""
    param_schema: Dict[str, Any]  # JSON Schema for the parameters
    task_schemas: Optional[Dict[TaskType, Dict[str, Any]]] = None # Task-specific overrides
    ui_schema: Optional[Dict[str, Any]] = None # UI hints
    docs_url: Optional[str] = None # Link to documentation

# --- Data Schemas ---

class DatasetMetadata(BaseModel):
    id: str
    parent_id: Optional[str] = None  # Link to parent
    name: Optional[str] = None  # User-provided name
    filename: str
    columns: List[str]
    row_count: int
    created_at: str

class DatasetProfile(BaseModel):
    dataset_id: str
    column_stats: Dict[str, Any] # simplified for MVP
    null_counts: Dict[str, int]
    dtypes: Dict[str, str] = {}  # Column data types
    # Add more as needed

# --- Run Configuration & State ---

class RunConfig(BaseModel):
    dataset_id: str
    target: str
    features: Optional[List[str]] = None # None means all except target
    task: TaskType
    model_id: str
    model_params: Dict[str, Any] = {}
    test_size: float = 0.2  # Train/test split ratio (0.0 to 1.0)
    output_dir: Optional[str] = None
    
    # Extras for future extensibility
    evaluation_options: Dict[str, Any] = {}
    explain_options: Dict[str, Any] = {}
    
class RunState(BaseModel):
    id: str
    config: RunConfig
    status: RunStatus
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    artifacts: List[str] = []
