import axios from 'axios';

// Determine API URL based on environment
const getApiBase = () => {
    // Check for env variable first
    if (process.env.NEXT_PUBLIC_API_URL) {
        return process.env.NEXT_PUBLIC_API_URL;
    }
    // In browser, check if we're on production
    if (typeof window !== 'undefined' && window.location.hostname !== 'localhost') {
        return 'https://modellab-production.up.railway.app';
    }
    // Default to localhost for local development
    return 'http://localhost:8000';
};

const API_BASE = getApiBase();
const API_REF = `${API_BASE}/v1`;

console.log('[API] Using base URL:', API_BASE); // Debug log

export const api = axios.create({
    baseURL: API_REF,
});

export interface Dataset {
    id: string;
    parent_id?: string;
    name?: string;
    filename: string;
    columns: string[];
    row_count: number;
    created_at: string;
}

export interface RunConfig {
    dataset_id: string;
    target: string;
    features?: string[];
    task: "regression" | "classification";
    model_id: string;
    model_params: Record<string, any>;
    test_size?: number; // Train/test split ratio (default 0.2)
}

export interface RunState {
    id: string;
    config: RunConfig;
    status: "queued" | "running" | "completed" | "failed";
    created_at: string;
    completed_at?: string;
    error?: string;
    metrics?: Record<string, any>;
    artifacts: string[];
}

export interface PluginMeta {
    id: string;
    name: string;
    description: string;
    supported_tasks: string[];
    param_schema: any;
    task_schemas?: Record<string, any>;
    ui_schema: any;
    docs_url?: string;
}

export interface DatasetProfile {
    dataset_id: string;
    column_stats: Record<string, any>; // describe() output
    null_counts: Record<string, number>;
    dtypes: Record<string, string>;
}

export const getDatasets = async () => (await api.get<Dataset[]>('/datasets')).data;
export const uploadDataset = async (file: File, name?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (name) {
        formData.append('name', name);
    }
    return (await api.post<Dataset>('/datasets', formData)).data;
};
export const getDatasetProfile = async (id: string) => (await api.post<DatasetProfile>(`/datasets/${id}/profile`)).data;

export const deleteDataset = async (id: string) => {
    await api.delete(`/datasets/${id}`);
};

export const getModels = async () => (await api.get<PluginMeta[]>('/plugins/models')).data;
export const listRuns = async (status?: string) => {
    const params = status ? { status } : {};
    return (await api.get<RunState[]>('/runs', { params })).data;
};
export const createRun = async (config: RunConfig) => (await api.post<RunState>('/runs', config)).data;
export const getRun = async (id: string) => (await api.get<RunState>(`/runs/${id}`)).data;
export const deleteRun = async (id: string) => {
    await api.delete(`/runs/${id}`);
};

// --- Preprocessing ---

export type ImputationStrategy = "none" | "mean" | "median" | "mode" | "constant" | "drop_rows" | "unknown";
export type DataTypeOverride = "auto" | "int64" | "float64" | "object" | "bool" | "datetime64";

export interface ColumnConfig {
    include: boolean;
    dtype_override: DataTypeOverride;
    imputation_strategy: ImputationStrategy;
    constant_value?: string;
}

export interface PreprocessingConfig {
    columns: Record<string, ColumnConfig>;
    apply_one_hot_encoding: boolean;
    apply_scaling: boolean;
}

export interface PreprocessedPreview {
    columns: string[];
    data: Record<string, any>[];
    row_count: number;
    applied_transformations: string[];
}

export const getPreprocessingPreview = async (id: string, config: PreprocessingConfig) =>
    (await api.post<PreprocessedPreview>(`/datasets/${id}/preprocess/preview`, config)).data;

export const savePreprocessingConfig = async (id: string, config: PreprocessingConfig) =>
    (await api.post(`/datasets/${id}/preprocess/save`, config)).data;

export const getPreprocessingConfig = async (id: string) =>
    (await api.get<PreprocessingConfig>(`/datasets/${id}/preprocess/config`)).data;

// --- Enhanced Data Preparation ---

export type OutlierStrategy = "none" | "remove" | "clip_iqr" | "clip_zscore";
export type ScalingMethod = "none" | "standard" | "minmax" | "robust";
export type EncodingMethod = "none" | "onehot" | "label" | "frequency";

export interface NumericColumnConfig {
    include: boolean;
    dtype_override?: DataTypeOverride;
    imputation_strategy: ImputationStrategy;
    constant_value?: number;
    outlier_strategy: OutlierStrategy;
    scaling_method: ScalingMethod;
}

export interface CategoricalColumnConfig {
    include: boolean;
    dtype_override?: DataTypeOverride;
    imputation_strategy: ImputationStrategy;
    constant_value?: string;
    encoding_method: EncodingMethod;
}

export interface DataCleaningConfig {
    remove_duplicates: boolean;
    numeric_columns: Record<string, NumericColumnConfig>;
    categorical_columns: Record<string, CategoricalColumnConfig>;
}

export interface ColumnQuality {
    name: string;
    dtype: string;
    is_numeric: boolean;
    null_count: number;
    null_percentage: number;
    unique_count: number;
    outlier_count: number;
    outlier_percentage: number;
}

export interface DataQualityReport {
    dataset_id: string;
    total_rows: number;
    total_columns: number;
    duplicate_rows: number;
    duplicate_percentage: number;
    numeric_columns: ColumnQuality[];
    categorical_columns: ColumnQuality[];
}

export interface EnhancedPreview {
    columns: string[];
    data: Record<string, any>[];
    original_data?: Record<string, any>[];
    row_count: number;
    original_row_count: number;
    applied_transformations: string[];
    quality_report?: DataQualityReport;
}

export const getDataQuality = async (id: string) =>
    (await api.get<DataQualityReport>(`/datasets/${id}/quality`)).data;

export const getPreparationPreview = async (id: string, config: DataCleaningConfig) =>
    (await api.post<EnhancedPreview>(`/datasets/${id}/prepare/preview`, config)).data;

export const saveCleaningConfig = async (id: string, config: DataCleaningConfig) =>
    (await api.post(`/datasets/${id}/prepare/save`, config)).data;

export const saveDatasetVersion = async (id: string, config: DataCleaningConfig, name: string) =>
    (await api.post<Dataset>(`/datasets/${id}/prepare/save_version`, { config, name })).data;
