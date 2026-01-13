"""
Data preprocessing service.
Enhanced with outlier detection, duplicate handling, scaling methods, and encoding options.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

from app.core.preprocessing_schemas import (
    PreprocessingConfig,
    DataCleaningConfig,
    NumericColumnConfig,
    CategoricalColumnConfig,
    ImputationStrategy,
    OutlierStrategy,
    ScalingMethod,
    EncodingMethod,
    DataTypeOverride,
    PreprocessedPreview,
    DataQualityReport,
    ColumnQuality
)
from app.services.data_service import data_service


class PreprocessingService:
    """Service for applying data preprocessing transformations."""
    
    # ==================== DATA QUALITY ====================
    
    def get_data_quality_report(self, dataset_id: str) -> DataQualityReport:
        """Generate a data quality report for a dataset."""
        path = data_service.get_dataset_path(dataset_id)
        df = pd.read_csv(path) if path.endswith('.csv') else pd.read_parquet(path)
        return self._generate_quality_report(df, dataset_id)

    def _generate_quality_report(self, df: pd.DataFrame, dataset_id: str) -> DataQualityReport:
        """Generate report for a dataframe."""
        total_rows = len(df)
        
        # Duplicates
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        numeric_quality = []
        for col in numeric_cols:
            null_count = df[col].isnull().sum()
            # Handle empty dataframes or all-NaN columns gracefully
            valid_data = df[col].dropna()
            outliers = self._detect_outliers_iqr(valid_data) if len(valid_data) > 0 else np.array([])
            
            numeric_quality.append(ColumnQuality(
                name=col,
                dtype=str(df[col].dtype),
                is_numeric=True,
                null_count=int(null_count),
                null_percentage=round(null_count / total_rows * 100, 2) if total_rows > 0 else 0,
                unique_count=int(df[col].nunique()),
                outlier_count=int(outliers.sum()) if len(outliers) > 0 else 0,
                outlier_percentage=round(outliers.sum() / len(valid_data) * 100, 2) if len(valid_data) > 0 else 0
            ))
        
        categorical_quality = []
        for col in categorical_cols:
            null_count = df[col].isnull().sum()
            categorical_quality.append(ColumnQuality(
                name=col,
                dtype=str(df[col].dtype),
                is_numeric=False,
                null_count=int(null_count),
                null_percentage=round(null_count / total_rows * 100, 2) if total_rows > 0 else 0,
                unique_count=int(df[col].nunique()),
                outlier_count=0,
                outlier_percentage=0
            ))
        
        return DataQualityReport(
            dataset_id=dataset_id,
            total_rows=total_rows,
            total_columns=len(df.columns),
            duplicate_rows=int(duplicate_rows),
            duplicate_percentage=round(duplicate_percentage, 2),
            numeric_columns=numeric_quality,
            categorical_columns=categorical_quality
        )
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        if len(series) == 0:
            return pd.Series([], dtype=bool)
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method."""
        if len(series) == 0 or series.std() == 0:
            return pd.Series([False] * len(series), dtype=bool)
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    def _apply_dtype_conversion(
        self,
        df: pd.DataFrame,
        col_name: str,
        dtype_override: DataTypeOverride
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Apply data type conversion to a column."""
        if dtype_override == DataTypeOverride.AUTO:
            return df, None
        
        result = df.copy()
        original_dtype = str(df[col_name].dtype)
        target_dtype = dtype_override.value
        
        # Skip if already the same type
        if original_dtype == target_dtype:
            return result, None
        
        try:
            if dtype_override == DataTypeOverride.INT:
                # Handle NaN by filling with 0 first, then converting
                if result[col_name].isnull().any():
                    result[col_name] = result[col_name].fillna(0)
                result[col_name] = result[col_name].astype('int64')
            
            elif dtype_override == DataTypeOverride.FLOAT:
                result[col_name] = pd.to_numeric(result[col_name], errors='coerce')
            
            elif dtype_override == DataTypeOverride.STRING:
                result[col_name] = result[col_name].astype(str)
                # Replace 'nan' strings with actual NaN
                result[col_name] = result[col_name].replace('nan', np.nan)
            
            elif dtype_override == DataTypeOverride.BOOL:
                # Handle various boolean representations
                bool_map = {
                    'true': True, 'false': False,
                    'yes': True, 'no': False,
                    '1': True, '0': False,
                    1: True, 0: False,
                    1.0: True, 0.0: False
                }
                result[col_name] = result[col_name].map(
                    lambda x: bool_map.get(str(x).lower() if pd.notna(x) else x, x)
                )
                result[col_name] = result[col_name].astype('bool')
            
            elif dtype_override == DataTypeOverride.DATETIME:
                result[col_name] = pd.to_datetime(result[col_name], errors='coerce')
            
            return result, f"{col_name}: converted from {original_dtype} to {target_dtype}"
        
        except Exception as e:
            # If conversion fails, return original dataframe with error message
            return df, f"{col_name}: failed to convert to {target_dtype} - {str(e)}"
    
    # ==================== ENHANCED PREPROCESSING ====================
    
    def apply_enhanced_preprocessing(
        self,
        df: pd.DataFrame,
        config: DataCleaningConfig
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Apply enhanced preprocessing with outliers, duplicates, and more."""
        transformations = []
        result = df.copy()
        original_rows = len(result)
        
        # 1. Remove duplicates
        if config.remove_duplicates:
            before = len(result)
            result = result.drop_duplicates()
            removed = before - len(result)
            if removed > 0:
                transformations.append(f"Removed {removed} duplicate rows")
        
        # 2. Apply dtype conversions first (before other processing)
        for col_name, col_config in config.numeric_columns.items():
            if col_name not in result.columns or not col_config.include:
                continue
            result, msg = self._apply_dtype_conversion(result, col_name, col_config.dtype_override)
            if msg:
                transformations.append(msg)
        
        for col_name, col_config in config.categorical_columns.items():
            if col_name not in result.columns or not col_config.include:
                continue
            result, msg = self._apply_dtype_conversion(result, col_name, col_config.dtype_override)
            if msg:
                transformations.append(msg)
        
        # 3. Process numeric columns
        for col_name, col_config in config.numeric_columns.items():
            if col_name not in result.columns:
                continue
            
            if not col_config.include:
                result = result.drop(columns=[col_name])
                transformations.append(f"Excluded numeric column: {col_name}")
                continue
            
            # Null imputation
            result, msg = self._apply_numeric_imputation(result, col_name, col_config)
            if msg:
                transformations.append(msg)
            
            # Outlier handling
            result, msg = self._apply_outlier_handling(result, col_name, col_config.outlier_strategy)
            if msg:
                transformations.append(msg)
            
            # Scaling
            result, msg = self._apply_column_scaling(result, col_name, col_config.scaling_method)
            if msg:
                transformations.append(msg)
        
        # 4. Process categorical columns
        for col_name, col_config in config.categorical_columns.items():
            if col_name not in result.columns:
                continue
            
            if not col_config.include:
                result = result.drop(columns=[col_name])
                transformations.append(f"Excluded categorical column: {col_name}")
                continue
            
            # Null imputation
            result, msg = self._apply_categorical_imputation(result, col_name, col_config)
            if msg:
                transformations.append(msg)
            
            # Encoding
            result, msg = self._apply_encoding(result, col_name, col_config.encoding_method)
            if msg:
                transformations.append(msg)
        
        return result, transformations
    
    def _apply_numeric_imputation(
        self,
        df: pd.DataFrame,
        col_name: str,
        config: NumericColumnConfig
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Apply imputation for numeric column."""
        result = df.copy()
        strategy = config.imputation_strategy
        null_count = result[col_name].isnull().sum()
        
        if null_count == 0 or strategy == ImputationStrategy.NONE:
            return result, None
        
        if strategy == ImputationStrategy.MEAN:
            fill = result[col_name].mean()
            result[col_name] = result[col_name].fillna(fill)
            return result, f"{col_name}: filled {null_count} nulls with mean ({fill:.2f})"
        
        elif strategy == ImputationStrategy.MEDIAN:
            fill = result[col_name].median()
            result[col_name] = result[col_name].fillna(fill)
            return result, f"{col_name}: filled {null_count} nulls with median ({fill:.2f})"
        
        elif strategy == ImputationStrategy.MODE:
            modes = result[col_name].mode()
            if len(modes) > 0:
                fill = modes[0]
                result[col_name] = result[col_name].fillna(fill)
                return result, f"{col_name}: filled {null_count} nulls with mode ({fill})"
        
        elif strategy == ImputationStrategy.CONSTANT:
            fill = config.constant_value if config.constant_value is not None else 0
            result[col_name] = result[col_name].fillna(fill)
            return result, f"{col_name}: filled {null_count} nulls with {fill}"
        
        elif strategy == ImputationStrategy.DROP_ROWS:
            before = len(result)
            result = result.dropna(subset=[col_name])
            return result, f"{col_name}: dropped {before - len(result)} rows with nulls"
        
        return result, None
    
    def _apply_categorical_imputation(
        self,
        df: pd.DataFrame,
        col_name: str,
        config: CategoricalColumnConfig
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Apply imputation for categorical column."""
        result = df.copy()
        strategy = config.imputation_strategy
        null_count = result[col_name].isnull().sum()
        
        if null_count == 0 or strategy == ImputationStrategy.NONE:
            return result, None
        
        if strategy == ImputationStrategy.MODE:
            modes = result[col_name].mode()
            if len(modes) > 0:
                result[col_name] = result[col_name].fillna(modes[0])
                return result, f"{col_name}: filled {null_count} nulls with mode ({modes[0]})"
        
        elif strategy == ImputationStrategy.CONSTANT:
            fill = config.constant_value if config.constant_value else "Unknown"
            result[col_name] = result[col_name].fillna(fill)
            return result, f"{col_name}: filled {null_count} nulls with '{fill}'"
        
        elif strategy == ImputationStrategy.UNKNOWN:
            result[col_name] = result[col_name].fillna("Unknown")
            return result, f"{col_name}: filled {null_count} nulls with 'Unknown'"
        
        elif strategy == ImputationStrategy.DROP_ROWS:
            before = len(result)
            result = result.dropna(subset=[col_name])
            return result, f"{col_name}: dropped {before - len(result)} rows with nulls"
        
        return result, None
    
    def _apply_outlier_handling(
        self,
        df: pd.DataFrame,
        col_name: str,
        strategy: OutlierStrategy
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Handle outliers for a numeric column."""
        result = df.copy()
        
        if strategy == OutlierStrategy.NONE:
            return result, None
        
        series = result[col_name].dropna()
        if len(series) == 0:
            return result, None
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (result[col_name] < lower_bound) | (result[col_name] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count == 0:
            return result, None
        
        if strategy == OutlierStrategy.REMOVE:
            result = result[~outlier_mask]
            return result, f"{col_name}: removed {outlier_count} outlier rows"
        
        elif strategy == OutlierStrategy.CLIP_IQR:
            result[col_name] = result[col_name].clip(lower=lower_bound, upper=upper_bound)
            return result, f"{col_name}: clipped {outlier_count} outliers to IQR bounds"
        
        elif strategy == OutlierStrategy.CLIP_ZSCORE:
            mean = series.mean()
            std = series.std()
            lower_z = mean - 3 * std
            upper_z = mean + 3 * std
            result[col_name] = result[col_name].clip(lower=lower_z, upper=upper_z)
            return result, f"{col_name}: clipped outliers to Z-score bounds"
        
        return result, None
    
    def _apply_column_scaling(
        self,
        df: pd.DataFrame,
        col_name: str,
        method: ScalingMethod
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Apply scaling to a numeric column."""
        result = df.copy()
        
        if method == ScalingMethod.NONE:
            return result, None
        
        values = result[[col_name]].values
        
        if method == ScalingMethod.STANDARD:
            scaler = StandardScaler()
            result[col_name] = scaler.fit_transform(values)
            return result, f"{col_name}: applied StandardScaler"
        
        elif method == ScalingMethod.MINMAX:
            scaler = MinMaxScaler()
            result[col_name] = scaler.fit_transform(values)
            return result, f"{col_name}: applied MinMaxScaler"
        
        elif method == ScalingMethod.ROBUST:
            scaler = RobustScaler()
            result[col_name] = scaler.fit_transform(values)
            return result, f"{col_name}: applied RobustScaler"
        
        return result, None
    
    def _apply_encoding(
        self,
        df: pd.DataFrame,
        col_name: str,
        method: EncodingMethod
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Apply encoding to a categorical column."""
        result = df.copy()
        
        if method == EncodingMethod.NONE:
            return result, None
        
        if method == EncodingMethod.ONEHOT:
            dummies = pd.get_dummies(result[col_name], prefix=col_name, drop_first=True)
            result = pd.concat([result.drop(columns=[col_name]), dummies], axis=1)
            return result, f"{col_name}: one-hot encoded into {len(dummies.columns)} columns"
        
        elif method == EncodingMethod.LABEL:
            le = LabelEncoder()
            # Handle NaN by converting to string first
            result[col_name] = result[col_name].fillna("_missing_")
            result[col_name] = le.fit_transform(result[col_name].astype(str))
            return result, f"{col_name}: label encoded"
        
        elif method == EncodingMethod.FREQUENCY:
            freq_map = result[col_name].value_counts(normalize=True).to_dict()
            result[col_name] = result[col_name].map(freq_map)
            return result, f"{col_name}: frequency encoded"
        
        return result, None
    
    # ==================== PREVIEW ====================
    
    def get_enhanced_preview(
        self,
        dataset_id: str,
        config: DataCleaningConfig,
        n_rows: int = 5
    ) -> PreprocessedPreview:
        """Get preview of enhanced preprocessing."""
        path = data_service.get_dataset_path(dataset_id)
        df = pd.read_csv(path) if path.endswith('.csv') else pd.read_parquet(path)
        
        original_rows = len(df)
        processed_df, transformations = self.apply_enhanced_preprocessing(df, config)
        
        preview_data = processed_df.head(n_rows).replace({np.nan: None}).to_dict(orient='records')
        original_preview = df.head(n_rows).replace({np.nan: None}).to_dict(orient='records')
        
        # Generate quality report for the PROCESSED data
        quality_report = self._generate_quality_report(processed_df, dataset_id)
        
        return PreprocessedPreview(
            columns=list(processed_df.columns),
            data=preview_data,
            original_data=original_preview,
            row_count=len(processed_df),
            original_row_count=original_rows,
            applied_transformations=transformations,
            quality_report=quality_report
        )
    async def create_derived_dataset(
        self,
        parent_id: str,
        config: DataCleaningConfig,
        name: str
    ) -> Any:
        """Create a new dataset version from parent dataset and configuration."""
        # 1. Load parent data
        path = data_service.get_dataset_path(parent_id)
        df = pd.read_csv(path) if path.endswith('.csv') else pd.read_parquet(path)
        
        # 2. Apply preprocessing
        processed_df, _ = self.apply_enhanced_preprocessing(df, config)
        
        # 3. Save as new dataset
        import io
        from fastapi import UploadFile
        
        buffer = io.BytesIO()
        processed_df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        class FileWrapper:
            def __init__(self, file):
                self.file = file
                self.filename = f"{name}.csv"
            
            async def read(self, size: int = -1):
                return self.file.read(size)
        
        file_wrapper = FileWrapper(buffer)
        
        # Call data service to save
        return await data_service.save_dataset(
            file=file_wrapper, 
            filename=f"{name}.csv", 
            name=name, 
            parent_id=parent_id
        )


# Singleton instance
preprocessing_service = PreprocessingService()
