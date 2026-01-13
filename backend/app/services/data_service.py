import os
import shutil
import json
import pandas as pd
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from app.core.schemas import DatasetMetadata, DatasetProfile

# Use absolute path relative to backend directory
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_STORAGE_DIR = os.path.join(_BACKEND_DIR, "data")
METADATA_FILE = os.path.join(DATA_STORAGE_DIR, "metadata.json")
os.makedirs(DATA_STORAGE_DIR, exist_ok=True)

class DataService:
    def _load_metadata(self) -> Dict[str, Any]:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

    def list_datasets(self):
        # Scan data dir for data files
        files = [f for f in os.listdir(DATA_STORAGE_DIR) if f.endswith(".parquet") or f.endswith(".csv")]
        metadata_registry = self._load_metadata()
        
        datasets = []
        for f in files:
            dataset_id = f.split('.')[0]
            meta = metadata_registry.get(dataset_id, {})
            datasets.append(DatasetMetadata(
                id=dataset_id,
                parent_id=meta.get("parent_id"),
                name=meta.get("name"),
                filename=meta.get("original_filename", f),
                columns=meta.get("columns", []),
                row_count=meta.get("row_count", 0),
                created_at=meta.get("created_at", "unknown")
            ))
        return datasets

    async def save_dataset(self, file, filename: str, name: Optional[str] = None, parent_id: Optional[str] = None) -> DatasetMetadata:
        dataset_id = str(uuid.uuid4())
        ext = filename.split('.')[-1]
        save_path = os.path.join(DATA_STORAGE_DIR, f"{dataset_id}.{ext}")
        
        # Save bytes asynchronously
        with open(save_path, "wb") as buffer:
            # Check if file has async read (UploadFile or our FileWrapper)
            if hasattr(file, 'read'):
                # Read in chunks to avoid memory issues
                chunk_size = 1024 * 1024  # 1MB
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    buffer.write(chunk)
            else:
                 # Fallback for sync files (unlikely in this path)
                 shutil.copyfileobj(file.file, buffer)
            
        # Optimize: Don't load full file into memory just for metadata
        columns = []
        row_count = 0
        try:
            print(f"[{datetime.now()}] Parsing metadata for {filename}...")
            if ext == 'csv':
                # Get columns
                df_head = pd.read_csv(save_path, nrows=0)
                columns = df_head.columns.tolist()
                
                # Count rows efficiently with fixed memory usage
                with open(save_path, 'rb') as f:
                    row_count = 0
                    buffer_size = 1024 * 1024
                    while True:
                        chunk = f.read(buffer_size)
                        if not chunk:
                            break
                        row_count += chunk.count(b'\n')
                    # Adjust if last line has no newline or empty file
                    # This is an approximation (count of newlines). accurate enough for progress.
                    if row_count == 0 and os.path.getsize(save_path) > 0:
                        row_count = 1
                print(f"[{datetime.now()}] CSV metadata parsed: {len(columns)} cols, {row_count} rows")
            elif ext == 'parquet':
                # Parquet metadata is cheap to read
                df = pd.read_parquet(save_path) 
                columns = df.columns.tolist()
                row_count = len(df)
                print(f"[{datetime.now()}] Parquet metadata parsed")
            else:
                df = pd.DataFrame()
                columns = []
                row_count = 0
        except Exception as e:
            print(f"Error parsing metadata: {e}")
            columns = []
            row_count = 0
            
        created_at = datetime.now().isoformat()
        
        # Store in metadata registry
        metadata_registry = self._load_metadata()
        metadata_registry[dataset_id] = {
            "name": name if name else filename,  # Default to filename if no name
            "original_filename": filename,
            "parent_id": parent_id,
            "columns": columns,
            "row_count": row_count,
            "created_at": created_at
        }
        self._save_metadata(metadata_registry)
        print(f"[{datetime.now()}] Dataset saved successfully: {dataset_id}")
        
        return DatasetMetadata(
            id=dataset_id,
            parent_id=parent_id,
            name=name if name else filename,
            filename=filename,
            columns=columns,
            row_count=row_count,
            created_at=created_at
        )
        
    def get_dataset_path(self, dataset_id: str) -> str:
        # crude lookup
        for f in os.listdir(DATA_STORAGE_DIR):
            if f.startswith(dataset_id):
                return os.path.join(DATA_STORAGE_DIR, f)
        raise FileNotFoundError(f"Dataset {dataset_id} not found")

    def profile_dataset(self, dataset_id: str) -> DatasetProfile:
        path = self.get_dataset_path(dataset_id)
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
            
        profile = {
            "columns": list(df.columns),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "nulls": df.isnull().sum().to_dict(),
            "stats": df.describe(include='all').to_dict()  # include='all' for object columns
        }
        
        return DatasetProfile(
            dataset_id=dataset_id,
            column_stats=profile["stats"],
            null_counts=profile["nulls"],
            dtypes=profile["dtypes"]
        )

    def delete_dataset(self, dataset_id: str):
        path = self.get_dataset_path(dataset_id)
        os.remove(path)
        
        # Also remove from metadata registry
        metadata_registry = self._load_metadata()
        if dataset_id in metadata_registry:
            del metadata_registry[dataset_id]
            self._save_metadata(metadata_registry)
    
    def save_preprocessing_config(self, dataset_id: str, config):
        """Save preprocessing configuration for a dataset."""
        # Verify dataset exists
        self.get_dataset_path(dataset_id)
        
        metadata_registry = self._load_metadata()
        if dataset_id not in metadata_registry:
            metadata_registry[dataset_id] = {}
        
        # Store config as dict
        metadata_registry[dataset_id]["preprocessing_config"] = config.model_dump()
        self._save_metadata(metadata_registry)
    
    def get_preprocessing_config(self, dataset_id: str):
        """Load preprocessing configuration for a dataset."""
        from app.core.preprocessing_schemas import PreprocessingConfig
        
        # Verify dataset exists
        self.get_dataset_path(dataset_id)
        
        metadata_registry = self._load_metadata()
        if dataset_id in metadata_registry:
            config_dict = metadata_registry[dataset_id].get("preprocessing_config", {})
            return PreprocessingConfig(**config_dict)
        
        return PreprocessingConfig()  # Return default config
    
    def save_cleaning_config(self, dataset_id: str, config):
        """Save data cleaning configuration for a dataset."""
        # Verify dataset exists
        self.get_dataset_path(dataset_id)
        
        metadata_registry = self._load_metadata()
        if dataset_id not in metadata_registry:
            metadata_registry[dataset_id] = {}
        
        metadata_registry[dataset_id]["cleaning_config"] = config.model_dump()
        self._save_metadata(metadata_registry)
    
    def get_cleaning_config(self, dataset_id: str):
        """Load data cleaning configuration for a dataset."""
        from app.core.preprocessing_schemas import DataCleaningConfig
        
        # Verify dataset exists
        self.get_dataset_path(dataset_id)
        
        metadata_registry = self._load_metadata()
        if dataset_id in metadata_registry:
            config_dict = metadata_registry[dataset_id].get("cleaning_config", {})
            return DataCleaningConfig(**config_dict)
        
        return DataCleaningConfig()

data_service = DataService()
