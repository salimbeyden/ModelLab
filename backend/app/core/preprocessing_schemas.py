"""
Pydantic schemas for data preprocessing configuration.
Enhanced with outlier handling, scaling methods, and encoding options.
"""
from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel


# ==================== ENUMS ====================

class ImputationStrategy(str, Enum):
    """Strategies for handling null values."""
    NONE = "none"
    MEAN = "mean"  # Numeric only
    MEDIAN = "median"  # Numeric only
    MODE = "mode"  # Works for both
    CONSTANT = "constant"
    DROP_ROWS = "drop_rows"
    UNKNOWN = "unknown"  # Categorical only


class OutlierStrategy(str, Enum):
    """Strategies for handling outliers."""
    NONE = "none"  # Keep outliers
    REMOVE = "remove"  # Delete rows with outliers
    CLIP_IQR = "clip_iqr"  # Clip to IQR bounds
    CLIP_ZSCORE = "clip_zscore"  # Clip to Z-score bounds


class ScalingMethod(str, Enum):
    """Scaling methods for numeric columns."""
    NONE = "none"
    STANDARD = "standard"  # (x - mean) / std
    MINMAX = "minmax"  # (x - min) / (max - min)
    ROBUST = "robust"  # (x - median) / IQR


class EncodingMethod(str, Enum):
    """Encoding methods for categorical columns."""
    NONE = "none"
    ONEHOT = "onehot"  # One-hot encoding
    LABEL = "label"  # Label encoding (0, 1, 2, ...)
    FREQUENCY = "frequency"  # Frequency encoding


class DataTypeOverride(str, Enum):
    """Supported data type overrides."""
    AUTO = "auto"
    INT = "int64"
    FLOAT = "float64"
    STRING = "object"
    BOOL = "bool"
    DATETIME = "datetime64"


# ==================== COLUMN CONFIGS ====================

class NumericColumnConfig(BaseModel):
    """Configuration for a numeric column."""
    include: bool = True
    dtype_override: DataTypeOverride = DataTypeOverride.AUTO
    imputation_strategy: ImputationStrategy = ImputationStrategy.NONE
    constant_value: Optional[float] = None
    outlier_strategy: OutlierStrategy = OutlierStrategy.NONE
    scaling_method: ScalingMethod = ScalingMethod.NONE


class CategoricalColumnConfig(BaseModel):
    """Configuration for a categorical column."""
    include: bool = True
    dtype_override: DataTypeOverride = DataTypeOverride.AUTO
    imputation_strategy: ImputationStrategy = ImputationStrategy.NONE
    constant_value: Optional[str] = None
    encoding_method: EncodingMethod = EncodingMethod.NONE


# ==================== MAIN CONFIG ====================

class DataCleaningConfig(BaseModel):
    """Configuration for data cleaning operations."""
    remove_duplicates: bool = False
    numeric_columns: Dict[str, NumericColumnConfig] = {}
    categorical_columns: Dict[str, CategoricalColumnConfig] = {}


class PreprocessingConfig(BaseModel):
    """Full preprocessing configuration for a dataset."""
    # Column-level configs (for backward compatibility)
    columns: Dict[str, Any] = {}
    
    # Enhanced configs
    cleaning: DataCleaningConfig = DataCleaningConfig()
    
    # Global options
    apply_one_hot_encoding: bool = True
    apply_scaling: bool = True


# ==================== DATA QUALITY ====================

class ColumnQuality(BaseModel):
    """Quality metrics for a single column."""
    name: str
    dtype: str
    is_numeric: bool
    null_count: int
    null_percentage: float
    unique_count: int
    outlier_count: int = 0  # Only for numeric
    outlier_percentage: float = 0.0


class DataQualityReport(BaseModel):
    """Overall data quality report."""
    dataset_id: str
    total_rows: int
    total_columns: int
    duplicate_rows: int
    duplicate_percentage: float
    numeric_columns: List[ColumnQuality]
    categorical_columns: List[ColumnQuality]


# ==================== PREVIEW ====================

class PreprocessedPreview(BaseModel):
    """Preview of processed data."""
    columns: List[str]
    data: List[Dict[str, Any]]
    original_data: Optional[List[Dict[str, Any]]] = None  # Original data for comparison
    row_count: int
    original_row_count: int
    applied_transformations: List[str]
    quality_report: Optional[DataQualityReport] = None  # Quality report for the processed data
