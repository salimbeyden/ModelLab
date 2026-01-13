from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from pydantic import BaseModel
from typing import List, Optional
from app.core.schemas import DatasetMetadata, DatasetProfile, RunConfig, RunState
from app.services.data_service import data_service
from app.services.run_service import run_service

router = APIRouter()

# --- Datasets ---

@router.get("/datasets", response_model=List[DatasetMetadata])
def list_datasets():
    return data_service.list_datasets()

@router.post("/datasets", response_model=DatasetMetadata)
async def upload_dataset(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    return await data_service.save_dataset(file, file.filename, name)

@router.post("/datasets/{id}/profile", response_model=DatasetProfile)
def profile_dataset(id: str):
    try:
        return data_service.profile_dataset(id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

@router.delete("/datasets/{id}", status_code=204)
def delete_dataset(id: str):
    try:
        data_service.delete_dataset(id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

# --- Runs ---

@router.post("/runs", response_model=RunState)
def create_run(config: RunConfig, background_tasks: BackgroundTasks):
    # 1. Create Run Record
    state = run_service.create_run(config)
    
    # 2. Submit to background execution
    # We use our own thread management, but we could trigger it here.
    # To ensure it runs *after* response, we can use background_tasks or just fire and forget in run_service.
    # run_service.start_run uses a thread, so it's non-blocking.
    run_service.start_run(state.id)
    
    return state

@router.get("/runs", response_model=List[RunState])
def list_runs(status: Optional[str] = None):
    """List all runs, optionally filtered by status."""
    runs = run_service.list_runs(status)
    return runs

@router.get("/runs/{id}", response_model=RunState)
def get_run(id: str):
    state = run_service.get_run(id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    return state

@router.delete("/runs/{id}", status_code=204)
def delete_run(id: str):
    """Delete a run and all its artifacts."""
    try:
        run_service.delete_run(id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- Preprocessing ---

from app.core.preprocessing_schemas import PreprocessingConfig, PreprocessedPreview
from app.services.preprocessing_service import preprocessing_service

@router.post("/datasets/{id}/preprocess/preview", response_model=PreprocessedPreview)
def preview_preprocessing(id: str, config: PreprocessingConfig):
    """Preview the result of applying preprocessing configuration."""
    try:
        return preprocessing_service.get_preview(id, config)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

@router.post("/datasets/{id}/preprocess/save")
def save_preprocessing_config(id: str, config: PreprocessingConfig):
    """Save preprocessing configuration for a dataset."""
    try:
        data_service.save_preprocessing_config(id, config)
        return {"status": "saved", "dataset_id": id}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

@router.get("/datasets/{id}/preprocess/config", response_model=PreprocessingConfig)
def get_preprocessing_config(id: str):
    """Get saved preprocessing configuration for a dataset."""
    try:
        return data_service.get_preprocessing_config(id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

# --- Enhanced Data Preparation ---

from app.core.preprocessing_schemas import DataQualityReport, DataCleaningConfig

@router.get("/datasets/{id}/quality", response_model=DataQualityReport)
def get_data_quality(id: str):
    """Get data quality report with nulls, duplicates, outliers per column."""
    try:
        return preprocessing_service.get_data_quality_report(id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Quality analysis error: {str(e)}")

@router.post("/datasets/{id}/prepare/preview", response_model=PreprocessedPreview)
def preview_data_preparation(id: str, config: DataCleaningConfig):
    """Preview enhanced data preparation results."""
    try:
        return preprocessing_service.get_enhanced_preview(id, config)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preparation error: {str(e)}")

@router.post("/datasets/{id}/prepare/save")
def save_data_preparation_config(id: str, config: DataCleaningConfig):
    """Save data preparation configuration."""
    try:
        data_service.save_cleaning_config(id, config)
        return {"status": "saved", "dataset_id": id}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

class SaveVersionRequest(BaseModel):
    config: DataCleaningConfig
    name: str

@router.post("/datasets/{id}/prepare/save_version", response_model=DatasetMetadata)
async def save_dataset_version(id: str, request: SaveVersionRequest):
    """Save the transformed data as a new dataset version."""
    try:
        return await preprocessing_service.create_derived_dataset(id, request.config, request.name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Version creation failed: {str(e)}")
