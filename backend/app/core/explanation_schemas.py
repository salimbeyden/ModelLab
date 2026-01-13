"""
Pydantic schemas for Model Explanations and Interpretability API.
Enterprise-grade type definitions for EBM and GAM model outputs.
"""
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


# ==================== ENUMS ====================

class ModelType(str, Enum):
    """Supported model types for explanations."""
    EBM = "ebm"
    MGCV = "mgcv"
    PYGAM = "pygam"


class TaskType(str, Enum):
    """ML task types."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


# ==================== SHAPE FUNCTION SCHEMAS ====================

class ShapePoint(BaseModel):
    """Single point on a shape function curve."""
    x: Union[float, str]  # float for numeric, str for categorical
    y: float  # Shape function value (log-odds for classification, contribution for regression)
    y_lower: Optional[float] = None  # Lower confidence bound
    y_upper: Optional[float] = None  # Upper confidence bound


class ShapeFunction(BaseModel):
    """Complete shape function for a single feature."""
    feature_name: str
    feature_type: str = Field(description="'numeric' or 'categorical'")
    points: List[ShapePoint]
    importance_score: float = Field(description="Mean absolute contribution")
    description: Optional[str] = None


class InteractionShape(BaseModel):
    """Shape function for feature interaction (2D heatmap data)."""
    feature_1: str
    feature_2: str
    x1_values: List[Union[float, str]]
    x2_values: List[Union[float, str]]
    z_values: List[List[float]]  # 2D matrix of interaction effects
    importance_score: float


class GlobalExplanation(BaseModel):
    """Complete global explanation for a model."""
    model_id: str
    model_type: ModelType
    task_type: TaskType
    feature_names: List[str]
    shape_functions: List[ShapeFunction]
    interactions: Optional[List[InteractionShape]] = None
    intercept: float = Field(description="Model intercept/baseline")
    feature_importance: Dict[str, float] = Field(description="Feature name to importance score")


# ==================== LOCAL EXPLANATION (WHAT-IF) SCHEMAS ====================

class FeatureContribution(BaseModel):
    """Single feature's contribution to a prediction."""
    feature_name: str
    feature_value: Union[float, str, None]
    contribution: float  # How much this feature adds/subtracts from baseline
    cumulative: float  # Running total after this feature


class LocalExplanation(BaseModel):
    """Local explanation for a single prediction (What-If result)."""
    model_id: str
    prediction: float  # Final predicted value
    prediction_probability: Optional[float] = None  # For classification
    baseline: float  # Intercept/average prediction
    contributions: List[FeatureContribution]
    total_contribution: float  # Sum of all contributions


class WhatIfRequest(BaseModel):
    """Request schema for What-If predictions."""
    run_id: str
    feature_values: Dict[str, Union[float, str, None]]


class WhatIfResponse(BaseModel):
    """Response schema for What-If predictions."""
    explanation: LocalExplanation
    feature_ranges: Dict[str, Dict[str, Any]]  # For slider bounds


# ==================== FEATURE RANGE SCHEMAS ====================

class NumericRange(BaseModel):
    """Range information for numeric features."""
    min: float
    max: float
    mean: float
    median: float
    std: float
    percentiles: Dict[str, float] = Field(default_factory=dict)  # 25, 50, 75


class CategoricalRange(BaseModel):
    """Range information for categorical features."""
    categories: List[str]
    frequencies: Dict[str, int]
    mode: str


class FeatureRange(BaseModel):
    """Combined feature range info."""
    feature_name: str
    feature_type: str  # 'numeric' or 'categorical'
    numeric_range: Optional[NumericRange] = None
    categorical_range: Optional[CategoricalRange] = None


class FeatureRangesResponse(BaseModel):
    """All feature ranges for a model."""
    run_id: str
    features: List[FeatureRange]


# ==================== PERFORMANCE METRICS SCHEMAS ====================

class RegressionMetrics(BaseModel):
    """Regression model performance metrics."""
    r2: float = Field(description="R-squared score")
    rmse: float = Field(description="Root Mean Square Error")
    mae: float = Field(description="Mean Absolute Error")
    mape: Optional[float] = Field(default=None, description="Mean Absolute Percentage Error")
    explained_variance: float
    max_error: float
    n_samples: int


class ClassificationMetrics(BaseModel):
    """Classification model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None
    log_loss: Optional[float] = None
    n_samples: int
    class_distribution: Dict[str, int]


class ConfusionMatrixData(BaseModel):
    """Confusion matrix data for classification."""
    labels: List[str]  # Class labels
    matrix: List[List[int]]  # Raw counts
    normalized_matrix: List[List[float]]  # Row-normalized (percentages)


class PerClassMetrics(BaseModel):
    """Per-class metrics for multi-class classification."""
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int  # Number of samples in this class


class ResidualPoint(BaseModel):
    """Single point for residual analysis."""
    actual: float
    predicted: float
    residual: float
    index: int


class ResidualAnalysis(BaseModel):
    """Complete residual analysis data."""
    points: List[ResidualPoint]
    mean_residual: float
    std_residual: float
    residual_histogram: Dict[str, int]  # Binned histogram


class ActualVsPredicted(BaseModel):
    """Actual vs Predicted scatter data."""
    actual: List[float]
    predicted: List[float]
    perfect_line: Dict[str, float]  # min, max for 45-degree line


class PerformanceResponse(BaseModel):
    """Complete performance suite response."""
    run_id: str
    task_type: TaskType
    regression_metrics: Optional[RegressionMetrics] = None
    classification_metrics: Optional[ClassificationMetrics] = None
    confusion_matrix: Optional[ConfusionMatrixData] = None
    per_class_metrics: Optional[List[PerClassMetrics]] = None
    residuals: Optional[ResidualAnalysis] = None
    actual_vs_predicted: Optional[ActualVsPredicted] = None
    feature_importance: Dict[str, float]


# ==================== DASHBOARD AGGREGATE SCHEMAS ====================

class DashboardSummary(BaseModel):
    """Aggregated summary for dashboard header."""
    run_id: str
    model_type: str
    task_type: str
    target_variable: str
    n_features: int
    n_samples_train: int
    n_samples_test: int
    primary_metric: float
    primary_metric_name: str
    training_date: str


class ModelDashboardData(BaseModel):
    """Complete data bundle for model dashboard."""
    summary: DashboardSummary
    global_explanation: GlobalExplanation
    performance: PerformanceResponse
    feature_ranges: List[FeatureRange]
