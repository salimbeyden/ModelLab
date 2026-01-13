"""
Data Service: Dataset management and storage operations.
"""
import json
import shutil
import pandas as pd
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from app.core.config import settings
from app.core.schemas import DatasetMetadata, DatasetProfile


class DataService:
    """Service for managing dataset storage and metadata."""
    
    @property
    def data_dir(self):
        return settings.DATA_DIR
    
    @property
    def metadata_file(self):
        return settings.METADATA_FILE
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata registry from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata registry to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def list_datasets(self):
        """List all available datasets."""
        files = [f for f in self.data_dir.iterdir() 
                 if f.suffix in (".parquet", ".csv")]
        metadata_registry = self._load_metadata()
        
        datasets = []
        for f in files:
            dataset_id = f.stem
            meta = metadata_registry.get(dataset_id, {})
            datasets.append(DatasetMetadata(
                id=dataset_id,
                parent_id=meta.get("parent_id"),
                name=meta.get("name"),
                filename=meta.get("original_filename", f.name),
                columns=meta.get("columns", []),
                row_count=meta.get("row_count", 0),
                created_at=meta.get("created_at", "unknown")
            ))
        return datasets

    async def save_dataset(
        self, 
        file, 
        filename: str, 
        name: Optional[str] = None, 
        parent_id: Optional[str] = None
    ) -> DatasetMetadata:
        """Save an uploaded dataset file."""
        dataset_id = str(uuid.uuid4())
        ext = filename.split('.')[-1]
        save_path = self.data_dir / f"{dataset_id}.{ext}"
        
        # Save file in chunks to avoid memory issues
        with open(save_path, "wb") as buffer:
            chunk_size = 1024 * 1024  # 1MB
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
            
        # Extract metadata efficiently
        columns, row_count = self._extract_metadata(save_path, ext)
        created_at = datetime.now().isoformat()
        
        # Store in metadata registry
        metadata_registry = self._load_metadata()
        metadata_registry[dataset_id] = {
            "name": name or filename,
            "original_filename": filename,
            "parent_id": parent_id,
            "columns": columns,
            "row_count": row_count,
            "created_at": created_at
        }
        self._save_metadata(metadata_registry)
        
        print(f"[DataService] Dataset saved: {dataset_id}")
        
        return DatasetMetadata(
            id=dataset_id,
            parent_id=parent_id,
            name=name or filename,
            filename=filename,
            columns=columns,
            row_count=row_count,
            created_at=created_at
        )
    
    def _extract_metadata(self, path, ext: str) -> tuple:
        """Extract columns and row count from a data file."""
        columns = []
        row_count = 0
        
        try:
            if ext == 'csv':
                # Get columns without loading full file
                df_head = pd.read_csv(path, nrows=0)
                columns = df_head.columns.tolist()
                
                # Count rows efficiently
                with open(path, 'rb') as f:
                    buffer_size = 1024 * 1024
                    while chunk := f.read(buffer_size):
                        row_count += chunk.count(b'\n')
                    if row_count == 0 and path.stat().st_size > 0:
                        row_count = 1
                        
            elif ext == 'parquet':
                df = pd.read_parquet(path)
                columns = df.columns.tolist()
                row_count = len(df)
                
        except Exception as e:
            print(f"[DataService] Error extracting metadata: {e}")
            
        return columns, row_count
        
    def get_dataset_path(self, dataset_id: str) -> str:
        """Get the file path for a dataset."""
        for f in self.data_dir.iterdir():
            if f.stem == dataset_id:
                return str(f)
        raise FileNotFoundError(f"Dataset {dataset_id} not found")

    def profile_dataset(self, dataset_id: str) -> DatasetProfile:
        """Generate profile statistics for a dataset."""
        path = self.get_dataset_path(dataset_id)
        
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
            
        return DatasetProfile(
            dataset_id=dataset_id,
            column_stats=df.describe(include='all').to_dict(),
            null_counts=df.isnull().sum().to_dict(),
            dtypes={k: str(v) for k, v in df.dtypes.items()}
        )

    def delete_dataset(self, dataset_id: str):
        """Delete a dataset and its metadata."""
        path = self.get_dataset_path(dataset_id)
        
        import os
        os.remove(path)
        
        # Remove from metadata registry
        metadata_registry = self._load_metadata()
        if dataset_id in metadata_registry:
            del metadata_registry[dataset_id]
            self._save_metadata(metadata_registry)
    
    def save_preprocessing_config(self, dataset_id: str, config):
        """Save preprocessing configuration for a dataset."""
        self.get_dataset_path(dataset_id)  # Verify exists
        
        metadata_registry = self._load_metadata()
        if dataset_id not in metadata_registry:
            metadata_registry[dataset_id] = {}
        
        metadata_registry[dataset_id]["preprocessing_config"] = config.model_dump()
        self._save_metadata(metadata_registry)
    
    def get_preprocessing_config(self, dataset_id: str):
        """Load preprocessing configuration for a dataset."""
        from app.core.preprocessing_schemas import PreprocessingConfig
        
        self.get_dataset_path(dataset_id)  # Verify exists
        
        metadata_registry = self._load_metadata()
        if dataset_id in metadata_registry:
            config_dict = metadata_registry[dataset_id].get("preprocessing_config", {})
            return PreprocessingConfig(**config_dict)
        
        return PreprocessingConfig()
    
    def save_cleaning_config(self, dataset_id: str, config):
        """Save data cleaning configuration for a dataset."""
        self.get_dataset_path(dataset_id)  # Verify exists
        
        metadata_registry = self._load_metadata()
        if dataset_id not in metadata_registry:
            metadata_registry[dataset_id] = {}
        
        metadata_registry[dataset_id]["cleaning_config"] = config.model_dump()
        self._save_metadata(metadata_registry)
    
    def get_cleaning_config(self, dataset_id: str):
        """Load data cleaning configuration for a dataset."""
        from app.core.preprocessing_schemas import DataCleaningConfig
        
        self.get_dataset_path(dataset_id)  # Verify exists
        
        metadata_registry = self._load_metadata()
        if dataset_id in metadata_registry:
            config_dict = metadata_registry[dataset_id].get("cleaning_config", {})
            return DataCleaningConfig(**config_dict)
        
        return DataCleaningConfig()


data_service = DataService()
