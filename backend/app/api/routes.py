"""
Core API Routes: Dataset and Run management endpoints.
"""
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional

from app.core.schemas import DatasetMetadata, DatasetProfile, RunConfig, RunState
from app.core.preprocessing_schemas import (
    PreprocessingConfig, PreprocessedPreview,
    DataQualityReport, DataCleaningConfig
)
from app.core.exceptions import handle_service_errors
from app.services.data_service import data_service
from app.services.run_service import run_service
from app.services.preprocessing_service import preprocessing_service

router = APIRouter()


# ==================== DATASETS ====================

@router.get("/datasets", response_model=List[DatasetMetadata], tags=["Datasets"])
def list_datasets():
    """List all available datasets."""
    return data_service.list_datasets()


@router.post("/datasets", response_model=DatasetMetadata, tags=["Datasets"])
@handle_service_errors
async def upload_dataset(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    """Upload a new dataset (CSV or Parquet)."""
    return await data_service.save_dataset(file, file.filename, name)


@router.post("/datasets/{dataset_id}/profile", response_model=DatasetProfile, tags=["Datasets"])
@handle_service_errors
def profile_dataset(dataset_id: str):
    """Generate profile statistics for a dataset."""
    return data_service.profile_dataset(dataset_id)


@router.delete("/datasets/{dataset_id}", status_code=204, tags=["Datasets"])
@handle_service_errors
def delete_dataset(dataset_id: str):
    """Delete a dataset and its metadata."""
    data_service.delete_dataset(dataset_id)


# ==================== RUNS ====================

@router.post("/runs", response_model=RunState, tags=["Runs"])
@handle_service_errors
def create_run(config: RunConfig):
    """Create and start a new training run."""
    state = run_service.create_run(config)
    run_service.start_run(state.id)
    return state


@router.get("/runs", response_model=List[RunState], tags=["Runs"])
def list_runs(status: Optional[str] = None):
    """List all runs, optionally filtered by status."""
    return run_service.list_runs(status)


@router.get("/runs/{run_id}", response_model=RunState, tags=["Runs"])
@handle_service_errors
def get_run(run_id: str):
    """Get details of a specific run."""
    state = run_service.get_run(run_id)
    if not state:
        raise FileNotFoundError(f"Run {run_id} not found")
    return state


@router.delete("/runs/{run_id}", status_code=204, tags=["Runs"])
@handle_service_errors
def delete_run(run_id: str):
    """Delete a run and all its artifacts."""
    run_service.delete_run(run_id)


# ==================== PREPROCESSING ====================

@router.post("/datasets/{dataset_id}/preprocess/preview", response_model=PreprocessedPreview, tags=["Preprocessing"])
@handle_service_errors
def preview_preprocessing(dataset_id: str, config: PreprocessingConfig):
    """Preview the result of applying preprocessing configuration."""
    return preprocessing_service.get_preview(dataset_id, config)


@router.post("/datasets/{dataset_id}/preprocess/save", tags=["Preprocessing"])
@handle_service_errors
def save_preprocessing_config(dataset_id: str, config: PreprocessingConfig):
    """Save preprocessing configuration for a dataset."""
    data_service.save_preprocessing_config(dataset_id, config)
    return {"status": "saved", "dataset_id": dataset_id}


@router.get("/datasets/{dataset_id}/preprocess/config", response_model=PreprocessingConfig, tags=["Preprocessing"])
@handle_service_errors
def get_preprocessing_config(dataset_id: str):
    """Get saved preprocessing configuration for a dataset."""
    return data_service.get_preprocessing_config(dataset_id)


# ==================== DATA PREPARATION ====================

@router.get("/datasets/{dataset_id}/quality", response_model=DataQualityReport, tags=["Data Preparation"])
@handle_service_errors
def get_data_quality(dataset_id: str):
    """Get data quality report with nulls, duplicates, outliers per column."""
    return preprocessing_service.get_data_quality_report(dataset_id)


@router.post("/datasets/{dataset_id}/prepare/preview", response_model=PreprocessedPreview, tags=["Data Preparation"])
@handle_service_errors
def preview_data_preparation(dataset_id: str, config: DataCleaningConfig):
    """Preview enhanced data preparation results."""
    return preprocessing_service.get_enhanced_preview(dataset_id, config)


@router.post("/datasets/{dataset_id}/prepare/save", tags=["Data Preparation"])
@handle_service_errors
def save_data_preparation_config(dataset_id: str, config: DataCleaningConfig):
    """Save data preparation configuration."""
    data_service.save_cleaning_config(dataset_id, config)
    return {"status": "saved", "dataset_id": dataset_id}


class SaveVersionRequest(BaseModel):
    """Request model for saving a dataset version."""
    config: DataCleaningConfig
    name: str


@router.post("/datasets/{dataset_id}/prepare/save_version", response_model=DatasetMetadata, tags=["Data Preparation"])
@handle_service_errors
async def save_dataset_version(dataset_id: str, request: SaveVersionRequest):
    """Save the transformed data as a new dataset version."""
    return await preprocessing_service.create_derived_dataset(
        dataset_id, request.config, request.name
    )
