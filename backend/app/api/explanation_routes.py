"""
Explanation API Routes: Model interpretability and diagnostics endpoints.
Supports EBM/GAM shape functions, What-If analysis, and performance metrics.
"""
from fastapi import APIRouter

from app.core.explanation_schemas import (
    GlobalExplanation, WhatIfRequest, WhatIfResponse,
    FeatureRangesResponse, PerformanceResponse, ModelDashboardData,
    ShapeFunction
)
from app.core.exceptions import handle_service_errors, NotFoundError
from app.services.explanation_service import explanation_service

router = APIRouter(prefix="/explain", tags=["Explanations"])


# ==================== GLOBAL EXPLANATIONS ====================

@router.get("/runs/{run_id}/global", response_model=GlobalExplanation)
@handle_service_errors
def get_global_explanation(run_id: str):
    """
    Get global explanation (shape functions) for a trained model.
    
    Returns shape functions showing how each feature affects the prediction,
    feature importance scores, and interaction terms if available.
    """
    return explanation_service.get_global_explanation(run_id)


@router.get("/runs/{run_id}/shapes/{feature_name}", response_model=ShapeFunction)
@handle_service_errors
def get_feature_shape(run_id: str, feature_name: str):
    """
    Get shape function for a specific feature.
    
    Returns x/y points with confidence intervals for plotting the
    feature's partial dependence curve.
    """
    global_exp = explanation_service.get_global_explanation(run_id)
    
    for shape in global_exp.shape_functions:
        if shape.feature_name == feature_name:
            return shape
    
    raise NotFoundError("Feature", feature_name)


# ==================== WHAT-IF / LOCAL EXPLANATIONS ====================

@router.post("/runs/{run_id}/what-if", response_model=WhatIfResponse)
@handle_service_errors
def what_if_prediction(run_id: str, request: WhatIfRequest):
    """
    Generate prediction with local explanation for custom feature values.
    
    Returns:
    - Predicted value
    - Baseline (intercept)
    - Per-feature contributions (for waterfall chart)
    - Feature ranges (for slider configuration)
    """
    return explanation_service.get_what_if_prediction(
        run_id=run_id,
        feature_values=request.feature_values
    )


@router.get("/runs/{run_id}/feature-ranges", response_model=FeatureRangesResponse)
@handle_service_errors
def get_feature_ranges(run_id: str):
    """
    Get feature value ranges for slider generation.
    
    Returns min/max/mean for numeric features and categories for categorical features.
    Used to configure the What-If simulator UI.
    """
    return explanation_service.get_feature_ranges(run_id)


# ==================== PERFORMANCE METRICS ====================

@router.get("/runs/{run_id}/performance", response_model=PerformanceResponse)
@handle_service_errors
def get_performance_metrics(run_id: str):
    """
    Get comprehensive performance metrics for a model.
    
    For regression: R², RMSE, MAE, residual analysis, actual vs predicted
    For classification: Accuracy, Precision, Recall, F1, ROC-AUC
    """
    return explanation_service.get_performance_metrics(run_id)


# ==================== DASHBOARD BUNDLE ====================

@router.get("/runs/{run_id}/dashboard", response_model=ModelDashboardData)
@handle_service_errors
def get_dashboard_data(run_id: str):
    """
    Get complete dashboard data bundle in a single request.
    
    Includes: summary, global explanation, performance metrics, feature ranges.
    Optimized for initial dashboard load.
    """
    return explanation_service.get_dashboard_data(run_id)
