"""
Run Service: Training run management and execution.
"""
import json
import shutil
import stat
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.core.config import settings
from app.core.schemas import RunConfig, RunState, RunStatus


class RunService:
    """Service for managing training runs."""
    
    @property
    def runs_dir(self) -> Path:
        return settings.RUNS_DIR

    def create_run(self, config: RunConfig) -> RunState:
        """Create a new run record."""
        run_id = str(uuid.uuid4())
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        state = RunState(
            id=run_id,
            config=config,
            status=RunStatus.QUEUED,
            created_at=datetime.now().isoformat()
        )
        
        self._save_run_state(run_dir, state)
        return state

    def start_run(self, run_id: str):
        """Start a run in a background thread."""
        thread = threading.Thread(target=self._execute_run, args=(run_id,))
        thread.start()
        
    def list_runs(self, status_filter: Optional[str] = None) -> List[RunState]:
        """List all runs, optionally filtered by status."""
        runs = []
        
        if not self.runs_dir.exists():
            return runs
        
        for run_dir in self.runs_dir.iterdir():
            if run_dir.is_dir():
                state = self._load_run_state(run_dir)
                if state and (status_filter is None or state.status == status_filter):
                    runs.append(state)
        
        # Sort by created_at descending
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs
    
    def get_run(self, run_id: str) -> Optional[RunState]:
        """Get a specific run by ID."""
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            return None
        return self._load_run_state(run_dir)
    
    def delete_run(self, run_id: str) -> bool:
        """Delete a run and all its artifacts."""
        run_dir = self.runs_dir / run_id
        
        if not run_dir.exists():
            raise FileNotFoundError(f"Run {run_id} not found")
        
        state = self._load_run_state(run_dir)
        if state and state.status == RunStatus.RUNNING:
            raise ValueError("Cannot delete a running job")
        
        # Handle read-only files on Windows
        def remove_readonly(func, path, excinfo):
            Path(path).chmod(stat.S_IWRITE)
            func(path)
        
        shutil.rmtree(run_dir, onerror=remove_readonly)
        return True

    def _execute_run(self, run_id: str):
        """Execute a training run (background thread)."""
        import pandas as pd
        import traceback
        
        run_dir = self.runs_dir / run_id
        state = self._load_run_state(run_dir)
        
        try:
            # Update status to RUNNING
            state.status = RunStatus.RUNNING
            self._save_run_state(run_dir, state)
            
            config = state.config
            
            # Get dataset path
            from app.services.data_service import data_service
            dataset_path = data_service.get_dataset_path(config.dataset_id)
            
            # Apply preprocessing if configured
            dataset_path = self._apply_preprocessing(run_id, run_dir, config, dataset_path)
            
            # Load data
            df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_parquet(dataset_path)
            
            # Train model
            from app.services.training_service import training_service
            print(f"[{run_id}] Training {config.model_id} on {len(df)} samples...")
            metrics, explanations = training_service.train_model(config, df, str(run_dir))
            
            # Update state
            state.metrics = metrics
            state.status = RunStatus.COMPLETED
            state.completed_at = datetime.now().isoformat()
            state.artifacts = [f.name for f in run_dir.iterdir()]
            
            self._save_run_state(run_dir, state)
            
        except Exception as e:
            state.status = RunStatus.FAILED
            state.error = str(e)
            state.completed_at = datetime.now().isoformat()
            self._save_run_state(run_dir, state)
            print(f"Run {run_id} failed: {e}")
            traceback.print_exc()

    def _apply_preprocessing(self, run_id: str, run_dir: Path, config: RunConfig, dataset_path: str) -> str:
        """Apply preprocessing if configured. Returns (possibly new) dataset path."""
        try:
            from app.services.data_service import data_service
            cleaning_config = data_service.get_cleaning_config(config.dataset_id)
            
            has_rules = (
                cleaning_config.remove_duplicates or 
                len(cleaning_config.numeric_columns) > 0 or 
                len(cleaning_config.categorical_columns) > 0
            )
            
            if has_rules:
                print(f"[{run_id}] Applying preprocessing...")
                import pandas as pd
                from app.services.preprocessing_service import preprocessing_service
                
                df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_parquet(dataset_path)
                processed_df, transformations = preprocessing_service.apply_enhanced_preprocessing(df, cleaning_config)
                
                if transformations:
                    processed_path = run_dir / "processed_train.csv"
                    processed_df.to_csv(processed_path, index=False)
                    print(f"[{run_id}] Applied {len(transformations)} transformations")
                    return str(processed_path)
                    
        except Exception as e:
            print(f"[{run_id}] Preprocessing failed (using raw data): {e}")
        
        return dataset_path

    def _save_run_state(self, run_dir: Path, state: RunState):
        """Save run state to disk."""
        with open(run_dir / "status.json", "w") as f:
            f.write(state.model_dump_json(indent=2))
            
    def _load_run_state(self, run_dir: Path) -> Optional[RunState]:
        """Load run state from disk."""
        try:
            with open(run_dir / "status.json", "r") as f:
                return RunState(**json.load(f))
        except Exception:
            return None


run_service = RunService()
