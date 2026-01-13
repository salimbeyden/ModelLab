/**
 * Explanation API Types and Functions
 * Enterprise-grade TypeScript interfaces for Model Interpretability Dashboard
 */
import { api } from './api';

// ==================== ENUMS ====================

export type ModelType = 'ebm' | 'mgcv' | 'pygam';
export type TaskType = 'regression' | 'classification';

// ==================== SHAPE FUNCTION TYPES ====================

export interface ShapePoint {
    x: number | string;
    y: number;
    y_lower?: number | null;
    y_upper?: number | null;
}

export interface ShapeFunction {
    feature_name: string;
    feature_type: 'numeric' | 'categorical';
    points: ShapePoint[];
    importance_score: number;
    description?: string;
}

export interface InteractionShape {
    feature_1: string;
    feature_2: string;
    x1_values: (number | string)[];
    x2_values: (number | string)[];
    z_values: number[][];
    importance_score: number;
}

export interface GlobalExplanation {
    model_id: string;
    model_type: ModelType;
    task_type: TaskType;
    feature_names: string[];
    shape_functions: ShapeFunction[];
    interactions?: InteractionShape[];
    intercept: number;
    feature_importance: Record<string, number>;
}

// ==================== LOCAL EXPLANATION (WHAT-IF) TYPES ====================

export interface FeatureContribution {
    feature_name: string;
    feature_value: number | string | null;
    contribution: number;
    cumulative: number;
}

export interface LocalExplanation {
    model_id: string;
    prediction: number;
    prediction_probability?: number;
    baseline: number;
    contributions: FeatureContribution[];
    total_contribution: number;
}

export interface WhatIfRequest {
    run_id: string;
    feature_values: Record<string, number | string | null>;
}

export interface WhatIfResponse {
    explanation: LocalExplanation;
    feature_ranges: Record<string, FeatureRangeInfo>;
}

export interface FeatureRangeInfo {
    type: 'numeric' | 'categorical';
    min?: number;
    max?: number;
    mean?: number;
    step?: number;
    categories?: string[];
}

// ==================== FEATURE RANGE TYPES ====================

export interface NumericRange {
    min: number;
    max: number;
    mean: number;
    median: number;
    std: number;
    percentiles: Record<string, number>;
}

export interface CategoricalRange {
    categories: string[];
    frequencies: Record<string, number>;
    mode: string;
}

export interface FeatureRange {
    feature_name: string;
    feature_type: 'numeric' | 'categorical';
    numeric_range?: NumericRange;
    categorical_range?: CategoricalRange;
}

export interface FeatureRangesResponse {
    run_id: string;
    features: FeatureRange[];
}

// ==================== PERFORMANCE METRICS TYPES ====================

export interface RegressionMetrics {
    r2: number;
    rmse: number;
    mae: number;
    mape?: number;
    explained_variance: number;
    max_error: number;
    n_samples: number;
}

export interface ClassificationMetrics {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    roc_auc?: number;
    log_loss?: number;
    n_samples: number;
    class_distribution: Record<string, number>;
}

export interface ConfusionMatrixData {
    labels: string[];
    matrix: number[][];
    normalized_matrix: number[][];
}

export interface PerClassMetrics {
    class_name: string;
    precision: number;
    recall: number;
    f1_score: number;
    support: number;
}

export interface ResidualPoint {
    actual: number;
    predicted: number;
    residual: number;
    index: number;
}

export interface ResidualAnalysis {
    points: ResidualPoint[];
    mean_residual: number;
    std_residual: number;
    residual_histogram: Record<string, number>;
}

export interface ActualVsPredicted {
    actual: number[];
    predicted: number[];
    perfect_line: { min: number; max: number };
}

export interface PerformanceResponse {
    run_id: string;
    task_type: TaskType;
    regression_metrics?: RegressionMetrics;
    classification_metrics?: ClassificationMetrics;
    confusion_matrix?: ConfusionMatrixData;
    per_class_metrics?: PerClassMetrics[];
    residuals?: ResidualAnalysis;
    actual_vs_predicted?: ActualVsPredicted;
    feature_importance: Record<string, number>;
}

// ==================== DASHBOARD TYPES ====================

export interface DashboardSummary {
    run_id: string;
    model_type: string;
    task_type: string;
    target_variable: string;
    n_features: number;
    n_samples_train: number;
    n_samples_test: number;
    primary_metric: number;
    primary_metric_name: string;
    training_date: string;
}

export interface ModelDashboardData {
    summary: DashboardSummary;
    global_explanation: GlobalExplanation;
    performance: PerformanceResponse;
    feature_ranges: FeatureRange[];
}

// ==================== API FUNCTIONS ====================

/**
 * Get global explanation (shape functions) for a model
 */
export const getGlobalExplanation = async (runId: string): Promise<GlobalExplanation> => {
    const response = await api.get<GlobalExplanation>(`/explain/runs/${runId}/global`);
    return response.data;
};

/**
 * Get shape function for a specific feature
 */
export const getFeatureShape = async (runId: string, featureName: string): Promise<ShapeFunction> => {
    const response = await api.get<ShapeFunction>(`/explain/runs/${runId}/shapes/${featureName}`);
    return response.data;
};

/**
 * Get What-If prediction with local explanation
 */
export const getWhatIfPrediction = async (
    runId: string,
    featureValues: Record<string, number | string | null>
): Promise<WhatIfResponse> => {
    const request: WhatIfRequest = { run_id: runId, feature_values: featureValues };
    const response = await api.post<WhatIfResponse>(`/explain/runs/${runId}/what-if`, request);
    return response.data;
};

/**
 * Get feature ranges for slider configuration
 */
export const getFeatureRanges = async (runId: string): Promise<FeatureRangesResponse> => {
    const response = await api.get<FeatureRangesResponse>(`/explain/runs/${runId}/feature-ranges`);
    return response.data;
};

/**
 * Get performance metrics
 */
export const getPerformanceMetrics = async (runId: string): Promise<PerformanceResponse> => {
    const response = await api.get<PerformanceResponse>(`/explain/runs/${runId}/performance`);
    return response.data;
};

/**
 * Get complete dashboard data bundle
 */
export const getDashboardData = async (runId: string): Promise<ModelDashboardData> => {
    const response = await api.get<ModelDashboardData>(`/explain/runs/${runId}/dashboard`);
    return response.data;
};
