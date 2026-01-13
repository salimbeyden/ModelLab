import os
import json
import time
import threading
import uuid
from datetime import datetime
from typing import List, Optional
from app.core.schemas import RunConfig, RunState, RunStatus
from app.core.registry import registry
from app.plugins.base import RunContext

# Use absolute path relative to this file's location
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(_BACKEND_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

class RunService:
    def __init__(self):
        self._active_runs = {} # In-memory cache of active runs

    def create_run(self, config: RunConfig) -> RunState:
        run_id = str(uuid.uuid4())
        run_dir = os.path.join(RUNS_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Initialize state
        state = RunState(
            id=run_id,
            config=config,
            status=RunStatus.QUEUED,
            created_at=datetime.isoformat(datetime.now())
        )
        
        # Save initial state
        self._save_run_state(run_dir, state)
        
        return state

    def start_run(self, run_id: str):
        """
        Start the run in a background thread.
        """
        thread = threading.Thread(target=self._execute_run, args=(run_id,))
        thread.start()
        
    def list_runs(self, status_filter: Optional[str] = None) -> List[RunState]:
        """List all runs, optionally filtered by status."""
        runs = []
        if not os.path.exists(RUNS_DIR):
            return runs
        
        for run_id in os.listdir(RUNS_DIR):
            run_dir = os.path.join(RUNS_DIR, run_id)
            if os.path.isdir(run_dir):
                state = self._load_run_state(run_dir)
                if state:
                    if status_filter is None or state.status == status_filter:
                        runs.append(state)
        
        # Sort by created_at descending
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs
    
    def get_run(self, run_id: str) -> Optional[RunState]:
        # Try to read from disk
        run_dir = os.path.join(RUNS_DIR, run_id)
        if not os.path.exists(run_dir):
            return None
            
        return self._load_run_state(run_dir)
    
    def delete_run(self, run_id: str) -> bool:
        """Delete a run and all its artifacts."""
        import shutil
        import stat
        
        run_dir = os.path.join(RUNS_DIR, run_id)
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"Run {run_id} not found")
        
        # Check if run is currently active
        state = self._load_run_state(run_dir)
        if state and state.status == RunStatus.RUNNING:
            raise ValueError("Cannot delete a running job")
        
        # Helper to handle read-only files on Windows
        def remove_readonly(func, path, excinfo):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        
        # Delete the entire run directory
        shutil.rmtree(run_dir, onerror=remove_readonly)
        return True

    def _execute_run(self, run_id: str):
        run_dir = os.path.join(RUNS_DIR, run_id)
        state = self._load_run_state(run_dir)
        
        try:
            # Update status to RUNNING
            state.status = RunStatus.RUNNING
            self._save_run_state(run_dir, state)
            
            # 1. Load Config & Context
            config = state.config
            
            # Resolving dataset path requires data_service
            # We import here to avoid circular dependency at top level if possible, 
            # or move data_service to a shared module. 
            # ideally data_service should be simpler or injected.
            from app.services.data_service import data_service
            dataset_path = data_service.get_dataset_path(config.dataset_id)
            
            # --- PREPROCESSING INJECTION ---
            try:
                # Check for cleaning config
                cleaning_config = data_service.get_cleaning_config(config.dataset_id)
                
                # If config exists and has active rules (simple check, or just always apply if exists)
                # The get_cleaning_config returns a DataCleaningConfig with defaults if not found 
                # but let's check if it was actually saved.
                # Actually data_service.get_cleaning_config returns a config object. 
                # If the user never saved one, it returns default empty config.
                
                # Let's perform a check if we should preprocess.
                # Ideally, we should check if the config is not default.
                # But for now, let's just attempt to apply it.
                # If it's a fresh/empty config, apply_enhanced_preprocessing won't do much (duplicates=False, no columns).
                # But to save time on large datasets, maybe check if we have entries.
                
                has_rules = (
                    cleaning_config.remove_duplicates or 
                    len(cleaning_config.numeric_columns) > 0 or 
                    len(cleaning_config.categorical_columns) > 0
                )
                
                if has_rules:
                    print(f"[{run_id}] Found preprocessing configuration. Applying...")
                    from app.services.preprocessing_service import preprocessing_service
                    import pandas as pd
                    
                    # Load original data
                    df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_parquet(dataset_path)
                    
                    # Apply preprocessing
                    processed_df, transformations = preprocessing_service.apply_enhanced_preprocessing(df, cleaning_config)
                    
                    if transformations:
                        print(f"[{run_id}] Applied {len(transformations)} transformations.")
                        
                        # Save processed dataset to run directory
                        processed_filename = "processed_train.csv"
                        processed_path = os.path.join(run_dir, processed_filename)
                        processed_df.to_csv(processed_path, index=False)
                        
                        # UPDATE DATASET PATH FOR RUN
                        print(f"[{run_id}] using processed dataset: {processed_path}")
                        dataset_path = processed_path
                    else:
                         print(f"[{run_id}] No transformations applied.")
            except Exception as e:
                print(f"[{run_id}] Preprocessing failed (continuing with raw data): {e}")
                # We do not fail the run, just log and continue with raw data
            # -------------------------------
            
            # 2. Get Training Service and Data Loading
            from app.services.training_service import training_service
            import pandas as pd
            
            # Load the data (raw or preprocessed)
            df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_parquet(dataset_path)
            
            # 3. Train using TrainingService
            print(f"[{run_id}] Delegating to TrainingService for model {config.model_id}...")
            metrics, explanations = training_service.train_model(config, df, run_dir)
            
            # Update state metrics
            state.metrics = metrics
            
            # 4. Complete
            state.status = RunStatus.COMPLETED
            state.completed_at = datetime.isoformat(datetime.now())
            state.artifacts = os.listdir(run_dir)
            
            self._save_run_state(run_dir, state)
            
        except Exception as e:
            import traceback
            state.status = RunStatus.FAILED
            state.error = str(e)
            state.completed_at = datetime.isoformat(datetime.now())
            self._save_run_state(run_dir, state)
            print(f"Run {run_id} failed: {e}")
            print(traceback.format_exc())

    def _save_run_state(self, run_dir: str, state: RunState):
        with open(os.path.join(run_dir, "status.json"), "w") as f:
            f.write(state.model_dump_json(indent=2))
            
    def _load_run_state(self, run_dir: str) -> RunState:
        try:
            with open(os.path.join(run_dir, "status.json"), "r") as f:
                data = json.load(f)
                return RunState(**data)
        except Exception:
            return None

run_service = RunService()
