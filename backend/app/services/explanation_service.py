"""
Explanation Service: Extracts shape functions and interpretability data from trained models.
Supports EBM (InterpretML) and mgcv (R GAM) models.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, 
    explained_variance_score, max_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, precision_recall_fscore_support
)

from app.core.explanation_schemas import (
    ShapePoint, ShapeFunction, InteractionShape, GlobalExplanation,
    FeatureContribution, LocalExplanation, WhatIfResponse,
    NumericRange, CategoricalRange, FeatureRange, FeatureRangesResponse,
    RegressionMetrics, ClassificationMetrics, ResidualPoint, ResidualAnalysis,
    ActualVsPredicted, PerformanceResponse, DashboardSummary, ModelDashboardData,
    ConfusionMatrixData, PerClassMetrics,
    ModelType, TaskType
)


def _to_scalar(val) -> float:
    """Safely convert a value (possibly array) to a Python float scalar."""
    if val is None:
        return 0.0
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            return float(val.item())
        elif val.size > 0:
            return float(val.flatten()[0])
        return 0.0
    return float(val)


class ExplanationService:
    """Service for extracting model explanations and interpretability data."""
    
    def __init__(self):
        # Use absolute paths relative to backend directory
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.runs_dir = os.path.join(backend_dir, "runs")
        self.data_dir = os.path.join(backend_dir, "data")
        self._rscript_cmd = None  # Cache R command path
    
    def _get_r_command(self) -> str:
        """
        Detect Rscript location. Try PATH first, then common Windows paths.
        """
        if self._rscript_cmd is not None:
            return self._rscript_cmd
            
        import subprocess
        
        # 1. Try PATH
        try:
            subprocess.run(["Rscript", "--version"], capture_output=True, check=True)
            self._rscript_cmd = "Rscript"
            return self._rscript_cmd
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # 2. Try common Windows paths
        common_paths = [
            r"C:\Program Files\R\R-4.5.2\bin\x64\Rscript.exe",
            r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe",
            r"C:\Program Files\R\R-4.5.1\bin\x64\Rscript.exe",
            r"C:\Program Files\R\R-4.5.0\bin\x64\Rscript.exe",
            r"C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe",
            r"C:\Program Files\R\R-4.4.1\bin\x64\Rscript.exe",
            r"C:\Program Files\R\R-4.3.3\bin\x64\Rscript.exe",
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                self._rscript_cmd = path
                return self._rscript_cmd
                
        # 3. Last resort - let the error bubble up if nothing found
        self._rscript_cmd = "Rscript"
        return self._rscript_cmd

    # ==================== SHAPE FUNCTIONS ====================
    
    def get_global_explanation(self, run_id: str) -> GlobalExplanation:
        """Extract global explanation (shape functions) from a trained model."""
        run_dir = os.path.join(self.runs_dir, run_id)
        status = self._load_run_status(run_dir)
        
        model_id = status["config"]["model_id"]
        task_type = TaskType(status["config"]["task"])
        
        if model_id == "ebm":
            return self._extract_ebm_shapes(run_dir, run_id, task_type)
        elif model_id == "mgcv":
            return self._extract_mgcv_shapes(run_dir, run_id, task_type)
        elif model_id == "lightgbm":
            return self._extract_lightgbm_explanation(run_dir, run_id, task_type)
        else:
            raise ValueError(f"Unsupported model type: {model_id}")
    
    def _extract_ebm_shapes(self, run_dir: str, run_id: str, task_type: TaskType) -> GlobalExplanation:
        """Extract shape functions from EBM model."""
        # Try multiple possible model file names
        model_path = None
        for fname in ["model.joblib", "model.pkl"]:
            candidate = os.path.join(run_dir, fname)
            if os.path.exists(candidate):
                model_path = candidate
                break
        
        if model_path is None:
            raise FileNotFoundError(f"No model file found in {run_dir}")
        
        model = joblib.load(model_path)
        
        # Get global explanation from EBM
        global_exp = model.explain_global()
        exp_data = global_exp.data()
        
        shape_functions = []
        feature_importance = {}
        
        # Extract individual feature shapes
        for i, feature_name in enumerate(exp_data['names']):
            # Skip interaction terms for main shapes
            if ' x ' in str(feature_name):
                continue
            
            # Get the feature's shape data
            try:
                feature_exp = global_exp.data(i)
                
                # Build shape points
                points = []
                x_vals = feature_exp.get('names', [])
                y_vals = feature_exp.get('scores', [])
                
                # Handle confidence intervals if available
                upper_bounds = feature_exp.get('upper_bounds', [None] * len(y_vals))
                lower_bounds = feature_exp.get('lower_bounds', [None] * len(y_vals))
                
                # Safely flatten y_vals if they are arrays (multi-class case)
                y_vals_flat = [_to_scalar(y) for y in y_vals]
                upper_flat = [_to_scalar(u) if u is not None else None for u in upper_bounds]
                lower_flat = [_to_scalar(l) if l is not None else None for l in lower_bounds]
                
                for j, x in enumerate(x_vals):
                    x_val = _to_scalar(x) if isinstance(x, (int, float, np.number, np.ndarray)) else str(x)
                    point = ShapePoint(
                        x=x_val,
                        y=y_vals_flat[j] if j < len(y_vals_flat) else 0.0,
                        y_upper=upper_flat[j] if j < len(upper_flat) and upper_flat[j] is not None else None,
                        y_lower=lower_flat[j] if j < len(lower_flat) and lower_flat[j] is not None else None
                    )
                    points.append(point)
                
                # Calculate importance as mean absolute contribution
                importance = float(np.mean(np.abs(y_vals_flat))) if len(y_vals_flat) > 0 else 0.0
                
                # Determine feature type
                feature_type = "categorical" if len(x_vals) > 0 and isinstance(x_vals[0], str) else "numeric"
                
                shape_func = ShapeFunction(
                    feature_name=str(feature_name),
                    feature_type=feature_type,
                    points=points,
                    importance_score=importance
                )
                shape_functions.append(shape_func)
                feature_importance[str(feature_name)] = importance
                
            except Exception as e:
                print(f"Warning: Could not extract shape for {feature_name}: {e}")
                continue
        
        # Extract interaction shapes
        interactions = self._extract_ebm_interactions(model, exp_data)
        
        # Get intercept (handle array case for multi-class)
        intercept = _to_scalar(model.intercept_) if hasattr(model, 'intercept_') else 0.0
        
        return GlobalExplanation(
            model_id=run_id,
            model_type=ModelType.EBM,
            task_type=task_type,
            feature_names=[sf.feature_name for sf in shape_functions],
            shape_functions=shape_functions,
            interactions=interactions,
            intercept=intercept,
            feature_importance=feature_importance
        )
    
    def _extract_ebm_interactions(self, model, exp_data) -> List[InteractionShape]:
        """Extract interaction terms from EBM."""
        interactions = []
        
        for i, feature_name in enumerate(exp_data['names']):
            if ' x ' in str(feature_name):
                try:
                    features = str(feature_name).split(' x ')
                    if len(features) == 2:
                        # Get interaction data
                        global_exp = model.explain_global()
                        interaction_data = global_exp.data(i)
                        
                        # This is a simplified extraction - actual 2D heatmap needs more work
                        importance = _to_scalar(exp_data['scores'][i]) if i < len(exp_data['scores']) else 0.0
                        
                        interaction = InteractionShape(
                            feature_1=features[0],
                            feature_2=features[1],
                            x1_values=[],  # Would need full extraction
                            x2_values=[],
                            z_values=[],
                            importance_score=importance
                        )
                        interactions.append(interaction)
                except Exception as e:
                    print(f"Warning: Could not extract interaction {feature_name}: {e}")
        
        return interactions
    
    def _extract_mgcv_shapes(self, run_dir: str, run_id: str, task_type: TaskType) -> GlobalExplanation:
        """Extract shape functions from mgcv model using R bridge."""
        import subprocess
        
        model_path = os.path.join(run_dir, "model.rds")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"mgcv model not found: {model_path}")
        
        # Create R script to extract shape functions using plot.gam data
        output_json = os.path.join(run_dir, "shapes_extracted.json")
        
        r_script = f'''
library(mgcv)
library(jsonlite)

model <- readRDS("{model_path.replace('\\', '/')}")

# Use plot.gam to extract smooth term data (this is the standard approach)
# plot.gam returns the data used for plotting without actually plotting
plot_data <- plot(model, pages = 0, se = TRUE, residuals = FALSE, rug = FALSE)

smooth_data <- list()
feature_names <- c()

# If plot_data is NULL (no smooth terms), handle gracefully
if (!is.null(plot_data)) {{
    for (i in seq_along(plot_data)) {{
        pd <- plot_data[[i]]
        
        term_name <- pd$xlab
        feature_names <- c(feature_names, term_name)
        
        # Extract x and y values for the smooth
        x_vals <- pd$x
        y_vals <- pd$fit
        se_vals <- pd$se
        
        # Calculate importance as mean absolute effect
        importance <- mean(abs(y_vals), na.rm = TRUE)
        
        smooth_data[[term_name]] <- list(
            feature_name = term_name,
            x_values = as.list(x_vals),
            y_values = as.list(y_vals),
            se_upper = as.list(y_vals + 2 * se_vals),
            se_lower = as.list(y_vals - 2 * se_vals),
            importance = importance
        )
    }}
}}

# Get intercept
intercept <- coef(model)[1]

# Get feature names from model summary if not from smooth terms
if (length(feature_names) == 0) {{
    feature_names <- names(model$var.summary)
}}

# Ensure feature_names is always an array (use I() to prevent auto_unbox)
result <- list(
    shapes = smooth_data,
    intercept = intercept,
    feature_names = I(feature_names)
)

write(toJSON(result, auto_unbox = TRUE), "{output_json.replace('\\', '/')}")
'''
        
        # Execute R script
        script_path = os.path.join(run_dir, "extract_shapes.R")
        with open(script_path, "w") as f:
            f.write(r_script)
        
        try:
            r_cmd = self._get_r_command()
            result = subprocess.run([r_cmd, script_path], capture_output=True, check=True)
            
            with open(output_json, "r") as f:
                r_data = json.load(f)
            
            # Convert R output to our schema
            shape_functions = []
            feature_importance = {}
            
            for term_name, term_data in r_data.get("shapes", {}).items():
                x_values = term_data.get("x_values", [])
                y_values = term_data.get("y_values", [])
                se_upper = term_data.get("se_upper", [])
                se_lower = term_data.get("se_lower", [])
                
                points = []
                for i, x in enumerate(x_values):
                    y = y_values[i] if i < len(y_values) else 0.0
                    y_up = se_upper[i] if i < len(se_upper) else None
                    y_lo = se_lower[i] if i < len(se_lower) else None
                    points.append(ShapePoint(
                        x=float(x), 
                        y=float(y),
                        y_upper=float(y_up) if y_up is not None else None,
                        y_lower=float(y_lo) if y_lo is not None else None
                    ))
                
                importance = float(term_data.get("importance", 0))
                
                shape_func = ShapeFunction(
                    feature_name=term_data.get("feature_name", term_name),
                    feature_type="numeric",
                    points=points,
                    importance_score=importance
                )
                shape_functions.append(shape_func)
                feature_importance[term_data.get("feature_name", term_name)] = importance
            
            return GlobalExplanation(
                model_id=run_id,
                model_type=ModelType.MGCV,
                task_type=task_type,
                feature_names=r_data.get("feature_names", []),
                shape_functions=shape_functions,
                intercept=float(r_data.get("intercept", 0)),
                feature_importance=feature_importance
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract mgcv shapes: {e}")
    
    def _extract_lightgbm_explanation(self, run_dir: str, run_id: str, task_type: TaskType) -> GlobalExplanation:
        """Extract SHAP-based global explanations from LightGBM model."""
        import shap
        
        # Load model
        model_path = None
        for fname in ["model.joblib", "model.pkl"]:
            candidate = os.path.join(run_dir, fname)
            if os.path.exists(candidate):
                model_path = candidate
                break
        
        if model_path is None:
            raise FileNotFoundError(f"No model file found in {run_dir}")
        
        saved_data = joblib.load(model_path)
        
        if isinstance(saved_data, dict) and 'model' in saved_data:
            model = saved_data['model']
            feature_names = saved_data.get('feature_names', [])
        else:
            model = saved_data
            feature_names = list(getattr(model, 'feature_name_', []))
        
        # Load training data for SHAP background
        status = self._load_run_status(run_dir)
        dataset_id = status["config"]["dataset_id"]
        target = status["config"]["target"]
        
        from app.services.data_service import data_service
        dataset_path = data_service.get_dataset_path(dataset_id)
        
        df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_parquet(dataset_path)
        X = df.drop(columns=[target])
        
        # Convert object columns to category for LightGBM
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype('category')
        
        # Use a sample for SHAP (for speed)
        sample_size = min(500, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        
        # Create SHAP TreeExplainer for LightGBM
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle classification (shap_values is list) vs regression (array)
        if isinstance(shap_values, list):
            # For classification, use the positive class (index 1) or last class
            shap_values = shap_values[-1] if len(shap_values) > 1 else shap_values[0]
        
        # Ensure shap_values is 2D (samples x features)
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            # Multi-output: take the last output (for classification, positive class)
            shap_values = shap_values[:, :, -1]
        elif shap_values.ndim == 1:
            # Single sample: reshape to 2D
            shap_values = shap_values.reshape(1, -1)
        
        # Calculate mean absolute SHAP values for feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Ensure mean_abs_shap is 1D
        mean_abs_shap = np.array(mean_abs_shap).flatten()
        
        feature_importance = {}
        shape_functions = []
        
        for i, fname in enumerate(feature_names):
            importance = float(mean_abs_shap[i]) if i < len(mean_abs_shap) else 0.0
            feature_importance[fname] = importance
            
            # Create SHAP-based dependence plot data (partial dependence via SHAP)
            col_data = X_sample.iloc[:, i]
            col_shap = shap_values[:, i] if i < shap_values.shape[1] else np.zeros(len(X_sample))
            
            # Ensure col_shap is 1D
            col_shap = np.array(col_shap).flatten()
            
            # Determine feature type
            is_categorical = col_data.dtype.name == 'category' or col_data.dtype == 'object'
            
            if is_categorical:
                # Aggregate by category
                unique_cats = col_data.unique()
                points = []
                for cat in sorted([str(c) for c in unique_cats]):
                    mask = col_data.astype(str) == cat
                    if mask.sum() > 0:
                        mean_shap = float(np.mean(col_shap[mask]))
                        points.append(ShapePoint(x=str(cat), y=mean_shap))
                feature_type = "categorical"
            else:
                # Create binned dependence plot
                try:
                    col_numeric = pd.to_numeric(col_data, errors='coerce')
                    valid_mask = ~col_numeric.isna()
                    if valid_mask.sum() > 10:
                        col_valid = col_numeric[valid_mask].values
                        shap_valid = col_shap[valid_mask]
                        
                        # Sort by feature value
                        sort_idx = np.argsort(col_valid)
                        col_sorted = col_valid[sort_idx]
                        shap_sorted = shap_valid[sort_idx]
                        
                        # Bin into ~50 points for smooth curve
                        n_bins = min(50, len(col_sorted))
                        bin_size = len(col_sorted) // n_bins
                        
                        points = []
                        for b in range(n_bins):
                            start = b * bin_size
                            end = start + bin_size if b < n_bins - 1 else len(col_sorted)
                            x_val = float(np.mean(col_sorted[start:end]))
                            y_val = float(np.mean(shap_sorted[start:end]))
                            # Calculate std for confidence bounds
                            y_std = float(np.std(shap_sorted[start:end]))
                            points.append(ShapePoint(
                                x=x_val, 
                                y=y_val,
                                y_upper=y_val + y_std,
                                y_lower=y_val - y_std
                            ))
                    else:
                        points = [ShapePoint(x=0, y=0)]
                except Exception:
                    points = [ShapePoint(x=0, y=importance)]
                feature_type = "numeric"
            
            shape_func = ShapeFunction(
                feature_name=fname,
                feature_type=feature_type,
                points=points,
                importance_score=importance,
                description="SHAP dependence plot"
            )
            shape_functions.append(shape_func)
        
        # Sort by importance
        shape_functions.sort(key=lambda x: x.importance_score, reverse=True)
        
        # Get expected value (baseline)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[-1] if len(expected_value) > 1 else expected_value[0]
        
        return GlobalExplanation(
            model_id=run_id,
            model_type=ModelType.EBM,  # Using as fallback
            task_type=task_type,
            feature_names=feature_names,
            shape_functions=shape_functions,
            intercept=float(expected_value),
            feature_importance=feature_importance
        )
    
    # ==================== WHAT-IF / LOCAL EXPLANATION ====================
    
    def get_what_if_prediction(
        self, 
        run_id: str, 
        feature_values: Dict[str, Union[float, str, None]]
    ) -> WhatIfResponse:
        """Generate local explanation for a what-if scenario."""
        run_dir = os.path.join(self.runs_dir, run_id)
        status = self._load_run_status(run_dir)
        
        model_id = status["config"]["model_id"]
        
        if model_id == "ebm":
            return self._ebm_what_if(run_dir, run_id, feature_values, status)
        elif model_id == "mgcv":
            return self._mgcv_what_if(run_dir, run_id, feature_values, status)
        elif model_id == "lightgbm":
            return self._lightgbm_what_if(run_dir, run_id, feature_values, status)
        else:
            raise ValueError(f"Unsupported model type: {model_id}")
    
    def _ebm_what_if(
        self, 
        run_dir: str, 
        run_id: str, 
        feature_values: Dict[str, Any],
        status: Dict
    ) -> WhatIfResponse:
        """Generate EBM local explanation."""
        model_path = os.path.join(run_dir, "model.joblib")
        model = joblib.load(model_path)
        
        # Create input DataFrame
        df = pd.DataFrame([feature_values])
        
        # Get prediction and probability
        task_type = status.get("task_type", "regression")
        is_classification = task_type == "classification" or hasattr(model, 'classes_')
        
        raw_prediction = model.predict(df)[0]
        prediction_probability = None
        
        if is_classification and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
            # For binary classification, return probability of positive class
            prediction_probability = float(proba[-1]) if len(proba) > 1 else float(proba[0])
            # Use log-odds or probability as the "prediction" for waterfall
            prediction = float(prediction_probability)
        else:
            prediction = float(raw_prediction)
        
        # Get local explanation
        local_exp = model.explain_local(df)
        local_data = local_exp.data(0)
        
        # Build contributions
        contributions = []
        baseline = _to_scalar(model.intercept_) if hasattr(model, 'intercept_') else 0.0
        cumulative = baseline
        
        feature_names = local_data.get('names', [])
        feature_scores = local_data.get('scores', [])
        feature_vals = local_data.get('values', [])
        
        for i, fname in enumerate(feature_names):
            if fname == 'Intercept':
                continue
                
            contribution = _to_scalar(feature_scores[i]) if i < len(feature_scores) else 0.0
            cumulative += contribution
            
            contributions.append(FeatureContribution(
                feature_name=str(fname),
                feature_value=feature_vals[i] if i < len(feature_vals) else None,
                contribution=contribution,
                cumulative=cumulative
            ))
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        
        # Get feature ranges
        feature_ranges = self._get_feature_ranges_dict(run_id)
        
        local_explanation = LocalExplanation(
            model_id=run_id,
            prediction=prediction,
            prediction_probability=prediction_probability,
            baseline=baseline,
            contributions=contributions,
            total_contribution=cumulative - baseline
        )
        
        return WhatIfResponse(
            explanation=local_explanation,
            feature_ranges=feature_ranges
        )
    
    def _mgcv_what_if(
        self, 
        run_dir: str, 
        run_id: str, 
        feature_values: Dict[str, Any],
        status: Dict
    ) -> WhatIfResponse:
        """Generate mgcv local explanation (simplified)."""
        import subprocess
        
        model_path = os.path.join(run_dir, "model.rds")
        
        # Write feature values to temp file
        input_json = os.path.join(run_dir, "whatif_input.json")
        output_json = os.path.join(run_dir, "whatif_output.json")
        
        with open(input_json, "w") as f:
            json.dump(feature_values, f)
        
        r_script = f'''
library(mgcv)
library(jsonlite)

model <- readRDS("{model_path.replace('\\', '/')}")
input <- fromJSON("{input_json.replace('\\', '/')}")

# Create prediction dataframe
new_data <- as.data.frame(input)
prediction <- predict(model, new_data, type = "response")

# Get term contributions (simplified)
terms <- predict(model, new_data, type = "terms")

result <- list(
    prediction = prediction[1],
    intercept = coef(model)[1],
    contributions = as.list(terms[1,])
)

write(toJSON(result, auto_unbox = TRUE), "{output_json.replace('\\', '/')}")
'''
        
        script_path = os.path.join(run_dir, "whatif_script.R")
        with open(script_path, "w") as f:
            f.write(r_script)
        
        r_cmd = self._get_r_command()
        subprocess.run([r_cmd, script_path], capture_output=True, check=True)
        
        with open(output_json, "r") as f:
            r_result = json.load(f)
        
        # Build contributions
        contributions = []
        baseline = float(r_result.get("intercept", 0))
        cumulative = baseline
        
        for fname, contribution in r_result.get("contributions", {}).items():
            contribution_val = float(contribution)
            cumulative += contribution_val
            
            contributions.append(FeatureContribution(
                feature_name=fname,
                feature_value=feature_values.get(fname),
                contribution=contribution_val,
                cumulative=cumulative
            ))
        
        feature_ranges = self._get_feature_ranges_dict(run_id)
        
        local_explanation = LocalExplanation(
            model_id=run_id,
            prediction=float(r_result.get("prediction", 0)),
            baseline=baseline,
            contributions=contributions,
            total_contribution=cumulative - baseline
        )
        
        return WhatIfResponse(
            explanation=local_explanation,
            feature_ranges=feature_ranges
        )
    
    def _lightgbm_what_if(
        self, 
        run_dir: str, 
        run_id: str, 
        feature_values: Dict[str, Any],
        status: Dict
    ) -> WhatIfResponse:
        """Generate SHAP-based local explanation for LightGBM."""
        import shap
        
        # Load model
        model_path = None
        for fname in ["model.joblib", "model.pkl"]:
            candidate = os.path.join(run_dir, fname)
            if os.path.exists(candidate):
                model_path = candidate
                break
        
        if model_path is None:
            raise FileNotFoundError(f"No model file found in {run_dir}")
        
        saved_data = joblib.load(model_path)
        
        if isinstance(saved_data, dict) and 'model' in saved_data:
            model = saved_data['model']
            feature_names = saved_data.get('feature_names', [])
        else:
            model = saved_data
            feature_names = list(getattr(model, 'feature_name_', []))
        
        # Create input DataFrame
        input_data = {}
        for fname in feature_names:
            if fname in feature_values:
                input_data[fname] = [feature_values[fname]]
            else:
                input_data[fname] = [None]
        
        X = pd.DataFrame(input_data)
        
        # Convert object columns to category
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype('category')
        
        # Determine if classification
        task_type = status.get("task_type", "regression")
        is_classification = task_type == "classification" or hasattr(model, 'classes_')
        
        # Get prediction and probability
        raw_prediction = model.predict(X)[0]
        prediction_probability = None
        
        if is_classification and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            prediction_probability = float(proba[-1]) if len(proba) > 1 else float(proba[0])
            prediction = float(prediction_probability)
        else:
            prediction = float(raw_prediction)
        
        # Use SHAP for local explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Handle classification (shap_values is list) vs regression (array)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1] if len(shap_values) > 1 else shap_values[0]
        
        # Ensure shap_values is 2D
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]
        elif shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        
        # Get expected value (baseline)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            ev = np.array(expected_value).flatten()
            expected_value = ev[-1] if len(ev) > 1 else ev[0]
        baseline = float(expected_value)
        
        # Build contributions from SHAP values
        contributions = []
        cumulative = baseline
        
        shap_row = np.array(shap_values[0]).flatten()  # First (and only) row, ensure 1D
        
        for i, fname in enumerate(feature_names):
            contribution = float(shap_row[i]) if i < len(shap_row) else 0.0
            cumulative += contribution
            
            contributions.append(FeatureContribution(
                feature_name=fname,
                feature_value=feature_values.get(fname),
                contribution=contribution,
                cumulative=cumulative
            ))
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        
        # Recalculate cumulative after sorting
        cumulative = baseline
        for c in contributions:
            cumulative += c.contribution
            c.cumulative = cumulative
        
        feature_ranges = self._get_feature_ranges_dict(run_id)
        
        local_explanation = LocalExplanation(
            model_id=run_id,
            prediction=prediction,
            prediction_probability=prediction_probability,
            baseline=baseline,
            contributions=contributions,
            total_contribution=prediction - baseline
        )
        
        return WhatIfResponse(
            explanation=local_explanation,
            feature_ranges=feature_ranges
        )
    
    # ==================== FEATURE RANGES ====================
    
    def get_feature_ranges(self, run_id: str) -> FeatureRangesResponse:
        """Get feature ranges for slider generation."""
        run_dir = os.path.join(self.runs_dir, run_id)
        status = self._load_run_status(run_dir)
        
        # Load the training data
        dataset_id = status["config"]["dataset_id"]
        target = status["config"]["target"]
        
        from app.services.data_service import data_service
        dataset_path = data_service.get_dataset_path(dataset_id)
        
        df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_parquet(dataset_path)
        
        # Remove target column
        if target in df.columns:
            df = df.drop(columns=[target])
        
        features = []
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Numeric feature
                numeric_range = NumericRange(
                    min=float(df[col].min()),
                    max=float(df[col].max()),
                    mean=float(df[col].mean()),
                    median=float(df[col].median()),
                    std=float(df[col].std()),
                    percentiles={
                        "25": float(df[col].quantile(0.25)),
                        "50": float(df[col].quantile(0.50)),
                        "75": float(df[col].quantile(0.75))
                    }
                )
                features.append(FeatureRange(
                    feature_name=col,
                    feature_type="numeric",
                    numeric_range=numeric_range
                ))
            else:
                # Categorical feature (includes booleans)
                value_counts = df[col].value_counts().to_dict()
                categorical_range = CategoricalRange(
                    categories=[str(k) for k in value_counts.keys()],
                    frequencies={str(k): int(v) for k, v in value_counts.items()},
                    mode=str(df[col].mode()[0]) if len(df[col].mode()) > 0 else ""
                )
                features.append(FeatureRange(
                    feature_name=col,
                    feature_type="categorical",
                    categorical_range=categorical_range
                ))
        
        return FeatureRangesResponse(run_id=run_id, features=features)
    
    def _get_feature_ranges_dict(self, run_id: str) -> Dict[str, Dict[str, Any]]:
        """Get feature ranges as a dictionary for WhatIfResponse."""
        ranges_response = self.get_feature_ranges(run_id)
        
        result = {}
        for fr in ranges_response.features:
            if fr.feature_type == "numeric" and fr.numeric_range:
                result[fr.feature_name] = {
                    "type": "numeric",
                    "min": fr.numeric_range.min,
                    "max": fr.numeric_range.max,
                    "mean": fr.numeric_range.mean,
                    "step": (fr.numeric_range.max - fr.numeric_range.min) / 100
                }
            elif fr.feature_type == "categorical" and fr.categorical_range:
                result[fr.feature_name] = {
                    "type": "categorical",
                    "categories": fr.categorical_range.categories
                }
        
        return result
    
    # ==================== PERFORMANCE METRICS ====================
    
    def get_performance_metrics(self, run_id: str) -> PerformanceResponse:
        """Calculate comprehensive performance metrics."""
        run_dir = os.path.join(self.runs_dir, run_id)
        status = self._load_run_status(run_dir)
        
        task_type = TaskType(status["config"]["task"])
        model_id = status["config"]["model_id"]
        
        # Load model and data
        if model_id in ["ebm", "lightgbm"]:
            # Try multiple possible model file names
            model_path = None
            for fname in ["model.joblib", "model.pkl"]:
                candidate = os.path.join(run_dir, fname)
                if os.path.exists(candidate):
                    model_path = candidate
                    break
            
            if model_path is None:
                raise FileNotFoundError(f"No model file found in {run_dir}")
            
            saved_data = joblib.load(model_path)
            # LightGBM saves a dict with 'model' key, EBM saves the model directly
            if isinstance(saved_data, dict) and 'model' in saved_data:
                model = saved_data['model']
            else:
                model = saved_data
        else:
            model = None  # mgcv handled separately
        
        # Load test predictions
        dataset_id = status["config"]["dataset_id"]
        target = status["config"]["target"]
        test_size = status["config"].get("test_size", 0.2)
        
        from app.services.data_service import data_service
        dataset_path = data_service.get_dataset_path(dataset_id)
        
        df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_parquet(dataset_path)
        
        # Recreate train/test split
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=[target])
        y = df[target]
        
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Get predictions
        if model_id in ["ebm", "lightgbm"]:
            # Handle categorical features for LightGBM
            if model_id == "lightgbm":
                X_test_prepared = X_test.copy()
                for col in X_test_prepared.columns:
                    if X_test_prepared[col].dtype == 'object':
                        X_test_prepared[col] = X_test_prepared[col].astype('category')
                y_pred = model.predict(X_test_prepared)
            else:
                y_pred = model.predict(X_test)
        else:
            y_pred = self._mgcv_predict(run_dir, X_test)
        
        # Calculate metrics based on task type
        if task_type == TaskType.REGRESSION:
            metrics = self._calculate_regression_metrics(y_test, y_pred)
            residuals = self._calculate_residuals(y_test, y_pred)
            actual_vs_pred = self._calculate_actual_vs_predicted(y_test, y_pred)
            
            return PerformanceResponse(
                run_id=run_id,
                task_type=task_type,
                regression_metrics=metrics,
                residuals=residuals,
                actual_vs_predicted=actual_vs_pred,
                feature_importance=status.get("metrics", {}).get("feature_importance", {})
            )
        else:
            metrics = self._calculate_classification_metrics(y_test, y_pred, model)
            conf_matrix = self._calculate_confusion_matrix(y_test, y_pred)
            per_class = self._calculate_per_class_metrics(y_test, y_pred)
            
            return PerformanceResponse(
                run_id=run_id,
                task_type=task_type,
                classification_metrics=metrics,
                confusion_matrix=conf_matrix,
                per_class_metrics=per_class,
                feature_importance=status.get("metrics", {}).get("feature_importance", {})
            )
    
    def _calculate_regression_metrics(self, y_true, y_pred) -> RegressionMetrics:
        """Calculate regression metrics."""
        return RegressionMetrics(
            r2=float(r2_score(y_true, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
            mae=float(mean_absolute_error(y_true, y_pred)),
            mape=float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100) if not (y_true == 0).any() else None,
            explained_variance=float(explained_variance_score(y_true, y_pred)),
            max_error=float(max_error(y_true, y_pred)),
            n_samples=len(y_true)
        )
    
    def _calculate_classification_metrics(self, y_true, y_pred, model) -> ClassificationMetrics:
        """Calculate classification metrics."""
        metrics = ClassificationMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            recall=float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            f1_score=float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            n_samples=len(y_true),
            class_distribution={str(k): int(v) for k, v in pd.Series(y_true).value_counts().items()}
        )
        
        # Try to get probability-based metrics
        if model and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(y_true.index)
                metrics.roc_auc = float(roc_auc_score(y_true, y_proba, multi_class='ovr'))
            except:
                pass
        
        return metrics
    
    def _calculate_confusion_matrix(self, y_true, y_pred) -> ConfusionMatrixData:
        """Calculate confusion matrix with normalized version."""
        labels = sorted(list(set(y_true) | set(y_pred)))
        labels_str = [str(l) for l in labels]
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Row-normalize (each row sums to 1 = percentage of actual class)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm.astype(float), row_sums, where=row_sums != 0)
        
        return ConfusionMatrixData(
            labels=labels_str,
            matrix=cm.tolist(),
            normalized_matrix=np.round(cm_normalized, 4).tolist()
        )
    
    def _calculate_per_class_metrics(self, y_true, y_pred) -> List[PerClassMetrics]:
        """Calculate per-class precision, recall, F1 for multi-class."""
        labels = sorted(list(set(y_true) | set(y_pred)))
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        
        return [
            PerClassMetrics(
                class_name=str(labels[i]),
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1_score=float(f1[i]),
                support=int(support[i])
            )
            for i in range(len(labels))
        ]
    
    def _calculate_residuals(self, y_true, y_pred) -> ResidualAnalysis:
        """Calculate residual analysis data."""
        residuals = y_true.values - y_pred
        
        points = [
            ResidualPoint(
                actual=float(y_true.iloc[i]),
                predicted=float(y_pred[i]),
                residual=float(residuals[i]),
                index=i
            )
            for i in range(len(residuals))
        ]
        
        # Create histogram bins
        hist, bin_edges = np.histogram(residuals, bins=20)
        histogram = {
            f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}": int(hist[i])
            for i in range(len(hist))
        }
        
        return ResidualAnalysis(
            points=points,
            mean_residual=float(np.mean(residuals)),
            std_residual=float(np.std(residuals)),
            residual_histogram=histogram
        )
    
    def _calculate_actual_vs_predicted(self, y_true, y_pred) -> ActualVsPredicted:
        """Calculate actual vs predicted scatter data."""
        return ActualVsPredicted(
            actual=y_true.tolist(),
            predicted=y_pred.tolist(),
            perfect_line={
                "min": float(min(y_true.min(), y_pred.min())),
                "max": float(max(y_true.max(), y_pred.max()))
            }
        )
    
    def _mgcv_predict(self, run_dir: str, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from mgcv model."""
        import subprocess
        
        model_path = os.path.join(run_dir, "model.rds")
        data_path = os.path.join(run_dir, "pred_data.csv")
        output_path = os.path.join(run_dir, "predictions.csv")
        
        X.to_csv(data_path, index=False)
        
        r_script = f'''
library(mgcv)
model <- readRDS("{model_path.replace('\\', '/')}")
new_data <- read.csv("{data_path.replace('\\', '/')}")

# Convert character columns to factors
for (col in names(new_data)) {{
    if (is.character(new_data[[col]])) {{
        new_data[[col]] <- as.factor(new_data[[col]])
    }}
}}

preds <- predict(model, new_data, type = "response")
write.csv(preds, "{output_path.replace('\\', '/')}", row.names=FALSE)
'''
        
        script_path = os.path.join(run_dir, "predict_script.R")
        with open(script_path, "w") as f:
            f.write(r_script)
        
        r_cmd = self._get_r_command()
        result = subprocess.run([r_cmd, script_path], capture_output=True, check=True)
        
        return pd.read_csv(output_path).values.flatten()
        
        return pd.read_csv(output_path).values.flatten()
    
    # ==================== DASHBOARD SUMMARY ====================
    
    def get_dashboard_data(self, run_id: str) -> ModelDashboardData:
        """Get complete dashboard data bundle."""
        run_dir = os.path.join(self.runs_dir, run_id)
        status = self._load_run_status(run_dir)
        
        # Get global explanation first to count features
        global_explanation = self.get_global_explanation(run_id)
        
        # Build summary with safe defaults for missing fields
        metrics = status.get("metrics", {})
        n_features = len(global_explanation.feature_names) if global_explanation.feature_names else 0
        
        summary = DashboardSummary(
            run_id=run_id,
            model_type=status["config"]["model_id"],
            task_type=status["config"]["task"],
            target_variable=status["config"]["target"],
            n_features=n_features,
            n_samples_train=metrics.get("train_samples", 0),
            n_samples_test=metrics.get("test_samples", 0),
            primary_metric=self._get_primary_metric(status),
            primary_metric_name="R²" if status["config"]["task"] == "regression" else "Accuracy",
            training_date=status.get("created_at", "")
        )
        
        # Get remaining components
        performance = self.get_performance_metrics(run_id)
        feature_ranges = self.get_feature_ranges(run_id)
        
        return ModelDashboardData(
            summary=summary,
            global_explanation=global_explanation,
            performance=performance,
            feature_ranges=feature_ranges.features
        )
    
    def _get_primary_metric(self, status: Dict) -> float:
        """Extract primary metric from run status."""
        metrics = status.get("metrics", {})
        task = status["config"]["task"]
        
        if task == "regression":
            return float(metrics.get("r2", 0))
        else:
            return float(metrics.get("accuracy", 0))
    
    def _load_run_status(self, run_dir: str) -> Dict:
        """Load run status from disk."""
        status_path = os.path.join(run_dir, "status.json")
        if not os.path.exists(status_path):
            raise FileNotFoundError(f"Status file not found: {status_path}")
        with open(status_path, "r") as f:
            return json.load(f)


# Singleton instance
explanation_service = ExplanationService()
