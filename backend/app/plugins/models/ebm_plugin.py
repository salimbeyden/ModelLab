import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score

from app.core.model_interface import BaseModel
from app.core.model_factory import model_factory
from app.core.schemas import PluginMeta, TaskType

class EBMModel(BaseModel):
    """
    Explainable Boosting Machine (EBM) Plugin.
    Leverages interpretml for high-performance glass-box modeling.
    """
    
    @classmethod
    def get_meta(cls) -> PluginMeta:
        # Standard multiples from docs
        interaction_multiples = [
            "0", "0.5x", "1x", "1.5x", "2x", "2.5x", "3x", "3.5x", 
            "4x", "4.5x", "5x", "6x", "7x", "8x", "9x", "10x", "15x", "20x", "25x"
        ]

        # Base schema with common structure
        base_properties = {
            "learning_rate": {"type": "number", "default": 0.015},
            "interactions": {
                "anyOf": [
                    {"type": "string", "enum": interaction_multiples, "title": "Suggested Multiple (Nx)"},
                    {"type": "integer", "minimum": 0, "title": "Exact Integer Count"}
                ],
                "default": "3x",
                "description": "Number of interaction terms. 'Nx' means N times the number of features. Integer means exact count."
            },
            "max_bins": {"type": "integer", "default": 1024},
            "max_interaction_bins": {"type": "integer", "default": 64},
            "outer_bags": {"type": "integer", "default": 14},
            "inner_bags": {"type": "integer", "default": 0},
            "early_stopping_rounds": {"type": "integer", "default": 100},
            "early_stopping_tolerance": {"type": "number", "default": 1e-5},
            "max_rounds": {"type": "integer", "default": 50000},
            "min_samples_leaf": {"type": "integer", "default": 4},
            "max_leaves": {"type": "integer", "default": 3},
            "validation_size": {"type": "number", "default": 0.15},
            "smoothing_rounds": {"type": "integer", "default": 75},
            "interaction_smoothing_rounds": {"type": "integer", "default": 75},
            "min_hessian": {"type": "number", "default": 1e-4},
            "greedy_ratio": {"type": "number", "default": 10.0},
            "cyclic_progress": {"type": "number", "default": 0.0},
            "missing": {"type": "string", "enum": ["separate", "low", "high", "gain"], "default": "separate"},
            "reg_alpha": {"type": "number", "default": 0.0},
            "reg_lambda": {"type": "number", "default": 0.0},
            "max_delta_step": {"type": "number", "default": 0.0},
            "gain_scale": {"type": "number", "default": 5.0},
            "min_cat_samples": {"type": "integer", "default": 10},
            "cat_smooth": {"type": "number", "default": 10.0}
        }

        # Regression-specific overrides
        regression_props = base_properties.copy()
        regression_props["learning_rate"] = {"type": "number", "default": 0.04}
        regression_props["interactions"] = {
            "anyOf": [
                {"type": "string", "enum": interaction_multiples, "title": "Suggested Multiple (Nx)"},
                {"type": "integer", "minimum": 0, "title": "Exact Integer Count"}
            ],
            "default": "5x",
            "description": "Number of interaction terms. Regression defaults to 5x features."
        }
        regression_props["smoothing_rounds"] = {"type": "integer", "default": 500}
        regression_props["interaction_smoothing_rounds"] = {"type": "integer", "default": 100}
        regression_props["min_hessian"] = {"type": "number", "default": 0.0}

        # Classification-specific overrides
        classification_props = base_properties.copy()
        classification_props["learning_rate"] = {"type": "number", "default": 0.015}
        classification_props["interactions"] = {
            "anyOf": [
                {"type": "string", "enum": interaction_multiples, "title": "Suggested Multiple (Nx)"},
                {"type": "integer", "minimum": 0, "title": "Exact Integer Count"}
            ],
            "default": "3x",
            "description": "Number of interaction terms. Classification defaults to 3x features."
        }
        classification_props["smoothing_rounds"] = {"type": "integer", "default": 75}
        classification_props["interaction_smoothing_rounds"] = {"type": "integer", "default": 75}
        classification_props["min_hessian"] = {"type": "number", "default": 1e-4}

        return PluginMeta(
            id="ebm",
            name="Explainable Boosting Machine",
            supported_tasks=[TaskType.CLASSIFICATION, TaskType.REGRESSION],
            description="Glassbox model from InterpretML using gradient boosting on GAMs.",
            docs_url="https://interpret.ml/docs/hyperparameters.html",
            param_schema={"type": "object", "properties": base_properties},
            task_schemas={
                TaskType.REGRESSION: {"type": "object", "properties": regression_props},
                TaskType.CLASSIFICATION: {"type": "object", "properties": classification_props}
            },
            ui_schema={
                "ui:order": [
                    "learning_rate", "interactions", "max_bins", "max_interaction_bins",
                    "outer_bags", "inner_bags", "early_stopping_rounds", "early_stopping_tolerance",
                    "max_rounds", "min_samples_leaf", "max_leaves", "validation_size",
                    "smoothing_rounds", "interaction_smoothing_rounds", "min_hessian",
                    "greedy_ratio", "cyclic_progress", "missing", "reg_alpha", "reg_lambda",
                    "max_delta_step", "gain_scale", "min_cat_samples", "cat_smooth"
                ]
            }
        )

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.task: Optional[str] = None
        self.is_classifier = False

    def train(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
        self.task = params.get("task", "regression")
        self.is_classifier = (self.task == "classification")
        
        # Hyperparameters for EBM - pulling all from params with official defaults as fallback
        # Actually, let's just use what's passed in and rely on EBM's internal defaults for anything missing,
        # but since we want to be explicit, we'll map them.
        
        is_reg = (self.task == "regression")
        
        ebm_params = {
            "learning_rate": params.get("learning_rate", 0.04 if is_reg else 0.015),
            "interactions": params.get("interactions", "5x" if is_reg else "3x"),
            "max_bins": params.get("max_bins", 1024),
            "max_interaction_bins": params.get("max_interaction_bins", 64),
            "outer_bags": params.get("outer_bags", 14),
            "inner_bags": params.get("inner_bags", 0),
            "early_stopping_rounds": params.get("early_stopping_rounds", 100),
            "early_stopping_tolerance": params.get("early_stopping_tolerance", 1e-5),
            "max_rounds": params.get("max_rounds", 50000),
            "min_samples_leaf": params.get("min_samples_leaf", 4),
            "max_leaves": params.get("max_leaves", 3),
            "validation_size": params.get("validation_size", 0.15),
            "smoothing_rounds": params.get("smoothing_rounds", 500 if is_reg else 75),
            "interaction_smoothing_rounds": params.get("interaction_smoothing_rounds", 100 if is_reg else 75),
            "min_hessian": params.get("min_hessian", 0.0 if is_reg else 1e-4),
            "greedy_ratio": params.get("greedy_ratio", 10.0),
            "cyclic_progress": params.get("cyclic_progress", 0.0),
            "missing": params.get("missing", "separate"),
            "reg_alpha": params.get("reg_alpha", 0.0),
            "reg_lambda": params.get("reg_lambda", 0.0),
            "max_delta_step": params.get("max_delta_step", 0.0),
            "gain_scale": params.get("gain_scale", 5.0),
            "min_cat_samples": params.get("min_cat_samples", 10),
            "cat_smooth": params.get("cat_smooth", 10.0),
            "n_jobs": 1 
        }
        
        if self.is_classifier:
            self.model = ExplainableBoostingClassifier(**ebm_params)
        else:
            self.model = ExplainableBoostingRegressor(**ebm_params)
            
        print(f"[EBM-TRAIN] Starting fit with params: {ebm_params}")
        self.model.fit(X, y)
        
        return {
            "model_type": "EBM",
            "task": self.task,
            "params_used": ebm_params,
            "feature_names": list(X.columns)
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        preds = self.model.predict(X_test)
        metrics = {}
        
        if self.is_classifier:
            metrics["accuracy"] = float(accuracy_score(y_test, preds))
            metrics["f1_score"] = float(f1_score(y_test, preds, average='weighted'))
            try:
                probs = self.model.predict_proba(X_test)
                if probs.shape[1] == 2: # Binary
                    metrics["roc_auc"] = float(roc_auc_score(y_test, probs[:, 1]))
                else: # Multiclass
                    metrics["roc_auc"] = float(roc_auc_score(y_test, probs, multi_class='ovr'))
            except:
                pass
        else:
            metrics["mse"] = float(mean_squared_error(y_test, preds))
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            metrics["r2"] = float(r2_score(y_test, preds))
            
        return metrics

    def save(self, path: str):
        if self.model is None:
            raise ValueError("No model to save.")
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
        # Infer task/is_classifier from loaded object
        self.is_classifier = isinstance(self.model, ExplainableBoostingClassifier)
        self.task = "classification" if self.is_classifier else "regression"

    def get_explanations(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Global explanation: Feature Importance
        Local explanation: Contribution for first 5 samples
        """
        if self.model is None:
            raise ValueError("No model for explanation.")
            
        # Global Explanation
        global_exp = self.model.explain_global()
        feature_importance = {}
        # interpret puts importance in 'scores' or similar depending on version
        # We'll use the internal data attribute for accuracy
        data = global_exp.data()
        for i, name in enumerate(data['names']):
            feature_importance[name] = float(data['scores'][i])
            
        # Local Explanation (Summary)
        local_exp = self.model.explain_local(X.head(5))
        
        return {
            "global": {
                "feature_importance": feature_importance
            },
            "local_summary": "EBM Local explanations available via interpret dashboard. Serialized summary not fully implemented yet."
        }

# Register the model
model_factory.register("ebm", EBMModel)
