import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import lightgbm as lgb
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score, log_loss
)
from sklearn.model_selection import cross_val_score

from app.core.model_interface import BaseModel
from app.core.model_factory import model_factory
from app.core.schemas import PluginMeta, TaskType


class LightGBMModel(BaseModel):
    """
    LightGBM Plugin with Optuna hyperparameter optimization.
    Supports both manual parameter tuning and auto-optimization.
    """
    
    @classmethod
    def get_meta(cls) -> PluginMeta:
        return PluginMeta(
            id="lightgbm",
            name="LightGBM",
            supported_tasks=[TaskType.CLASSIFICATION, TaskType.REGRESSION],
            description="High-performance gradient boosting with optional Optuna auto-tuning.",
            docs_url="https://lightgbm.readthedocs.io/en/latest/Parameters.html",
            param_schema={
                "type": "object",
                "properties": {
                    # Core parameters
                    "num_leaves": {
                        "type": "integer",
                        "default": 31,
                        "minimum": 2,
                        "maximum": 256,
                        "description": "Max leaves per tree. Higher = more complex model."
                    },
                    "max_depth": {
                        "type": "integer",
                        "default": -1,
                        "minimum": -1,
                        "maximum": 50,
                        "description": "Max tree depth. -1 = no limit."
                    },
                    "learning_rate": {
                        "type": "number",
                        "default": 0.1,
                        "minimum": 0.001,
                        "maximum": 1.0,
                        "description": "Boosting learning rate."
                    },
                    "n_estimators": {
                        "type": "integer",
                        "default": 100,
                        "minimum": 10,
                        "maximum": 10000,
                        "description": "Number of boosting iterations."
                    },
                    # Regularization
                    "reg_alpha": {
                        "type": "number",
                        "default": 0.0,
                        "minimum": 0.0,
                        "description": "L1 regularization term."
                    },
                    "reg_lambda": {
                        "type": "number",
                        "default": 0.0,
                        "minimum": 0.0,
                        "description": "L2 regularization term."
                    },
                    "min_child_samples": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "description": "Min samples in a leaf."
                    },
                    "min_child_weight": {
                        "type": "number",
                        "default": 0.001,
                        "minimum": 0.0,
                        "description": "Min sum of instance weight in a child."
                    },
                    # Sampling
                    "subsample": {
                        "type": "number",
                        "default": 1.0,
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "description": "Row subsample ratio per iteration."
                    },
                    "subsample_freq": {
                        "type": "integer",
                        "default": 0,
                        "minimum": 0,
                        "description": "Frequency of subsampling. 0 = disabled."
                    },
                    "colsample_bytree": {
                        "type": "number",
                        "default": 1.0,
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "description": "Feature subsample ratio per tree."
                    },
                    # Advanced
                    "min_split_gain": {
                        "type": "number",
                        "default": 0.0,
                        "minimum": 0.0,
                        "description": "Min loss reduction for split."
                    },
                    "cat_smooth": {
                        "type": "number",
                        "default": 10.0,
                        "minimum": 0.0,
                        "description": "Smoothing for categorical features."
                    },
                    "max_cat_threshold": {
                        "type": "integer",
                        "default": 32,
                        "minimum": 1,
                        "description": "Max categories for one-vs-other split."
                    },
                    # Early stopping
                    "early_stopping_rounds": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 0,
                        "description": "Early stopping patience. 0 = disabled."
                    },
                    # Auto-optimization
                    "auto_optimize": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable Optuna hyperparameter search."
                    },
                    "optuna_trials": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 10,
                        "maximum": 200,
                        "description": "Number of Optuna optimization trials."
                    },
                    "optuna_cv_folds": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 2,
                        "maximum": 10,
                        "description": "Cross-validation folds for optimization."
                    }
                }
            },
            ui_schema={
                "ui:order": [
                    "auto_optimize", "optuna_trials", "optuna_cv_folds",
                    "num_leaves", "max_depth", "learning_rate", "n_estimators",
                    "reg_alpha", "reg_lambda", "min_child_samples", "min_child_weight",
                    "subsample", "subsample_freq", "colsample_bytree",
                    "min_split_gain", "cat_smooth", "max_cat_threshold",
                    "early_stopping_rounds"
                ]
            }
        )

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.task: Optional[str] = None
        self.is_classifier = False
        self.best_params: Dict[str, Any] = {}
        self.optimization_history: List[Dict] = []
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []

    def _detect_categorical_features(self, X: pd.DataFrame) -> List[str]:
        """Detect categorical columns."""
        cat_cols = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                cat_cols.append(col)
            elif X[col].nunique() < 20 and X[col].dtype in ['int64', 'int32']:
                # Low cardinality integers could be categorical
                cat_cols.append(col)
        return cat_cols

    def _prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for LightGBM."""
        X = X.copy()
        # Convert object columns to category type for LightGBM
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype('category')
            # Also handle any remaining string-like columns
            elif str(X[col].dtype) == 'string':
                X[col] = X[col].astype('category')
        return X

    def _run_optuna_optimization(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_trials: int,
        cv_folds: int
    ) -> Dict[str, Any]:
        """Run Optuna hyperparameter optimization."""
        import optuna
        from optuna.samplers import TPESampler
        from sklearn.model_selection import KFold, StratifiedKFold
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        is_regression = not self.is_classifier
        
        # Ensure all object columns are converted to category (safety check)
        X = X.copy()
        for col in X.columns:
            if X[col].dtype == 'object' or str(X[col].dtype) == 'string':
                X[col] = X[col].astype('category')
        
        # Get categorical feature names for LightGBM
        cat_features = [col for col in X.columns if X[col].dtype.name == 'category']
        print(f"[LightGBM Optuna] Detected {len(cat_features)} categorical features: {cat_features}")
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 8, 256),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                'verbosity': -1,
                'n_jobs': 1,
                'random_state': 42
            }
            
            # Manual cross-validation to pass categorical_feature parameter
            if is_regression:
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            else:
                kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            scores = []
            try:
                for train_idx, val_idx in kf.split(X, y):
                    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Ensure categorical dtypes are preserved after slicing
                    for col in cat_features:
                        if col in X_train.columns:
                            X_train[col] = X_train[col].astype('category')
                            X_val[col] = X_val[col].astype('category')
                    
                    if is_regression:
                        model = lgb.LGBMRegressor(**params)
                    else:
                        model = lgb.LGBMClassifier(**params)
                    
                    model.fit(
                        X_train, y_train,
                        categorical_feature=cat_features if cat_features else 'auto'
                    )
                    
                    if is_regression:
                        preds = model.predict(X_val)
                        score = -np.sqrt(mean_squared_error(y_val, preds))  # neg RMSE
                    else:
                        if len(y.unique()) == 2:
                            preds = model.predict_proba(X_val)[:, 1]
                            score = roc_auc_score(y_val, preds)
                        else:
                            preds = model.predict(X_val)
                            score = accuracy_score(y_val, preds)
                    
                    scores.append(score)
                
                return np.mean(scores)
            except Exception as e:
                print(f"[LightGBM Optuna] Trial failed: {e}")
                return float('-inf') if not is_regression else float('inf')
        
        sampler = TPESampler(seed=42)
        direction = 'minimize' if is_regression else 'maximize'
        
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Store optimization history
        self.optimization_history = [
            {
                'trial': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state)
            }
            for t in study.trials
        ]
        
        best_params = study.best_trial.params
        best_params['verbosity'] = -1
        best_params['n_jobs'] = 1
        best_params['random_state'] = 42
        
        print(f"[LightGBM] Optuna found best params: {best_params}")
        print(f"[LightGBM] Best score: {study.best_value}")
        
        return best_params

    def train(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
        self.task = params.get("task", "regression")
        self.is_classifier = (self.task == "classification")
        self.feature_names = list(X.columns)
        
        # Prepare data
        X = self._prepare_data(X)
        self.categorical_features = self._detect_categorical_features(X)
        
        # Check for auto-optimization
        auto_optimize = params.get("auto_optimize", False)
        
        if auto_optimize:
            n_trials = params.get("optuna_trials", 50)
            cv_folds = params.get("optuna_cv_folds", 5)
            print(f"[LightGBM] Starting Optuna optimization with {n_trials} trials...")
            lgb_params = self._run_optuna_optimization(X, y, n_trials, cv_folds)
            self.best_params = lgb_params
        else:
            # Use provided parameters
            lgb_params = {
                'num_leaves': params.get('num_leaves', 31),
                'max_depth': params.get('max_depth', -1),
                'learning_rate': params.get('learning_rate', 0.1),
                'n_estimators': params.get('n_estimators', 100),
                'reg_alpha': params.get('reg_alpha', 0.0),
                'reg_lambda': params.get('reg_lambda', 0.0),
                'min_child_samples': params.get('min_child_samples', 20),
                'min_child_weight': params.get('min_child_weight', 0.001),
                'subsample': params.get('subsample', 1.0),
                'subsample_freq': params.get('subsample_freq', 0),
                'colsample_bytree': params.get('colsample_bytree', 1.0),
                'min_split_gain': params.get('min_split_gain', 0.0),
                'verbosity': -1,
                'n_jobs': 1,
                'random_state': 42
            }
            self.best_params = lgb_params
        
        # Train model
        if self.is_classifier:
            self.model = lgb.LGBMClassifier(**lgb_params)
        else:
            self.model = lgb.LGBMRegressor(**lgb_params)
        
        # Detect categorical features for LightGBM
        cat_features = [col for col in X.columns if X[col].dtype.name == 'category']
        
        print(f"[LightGBM] Training {'classifier' if self.is_classifier else 'regressor'}...")
        self.model.fit(
            X, y,
            categorical_feature=cat_features if cat_features else 'auto'
        )
        
        return {
            "model_type": "LightGBM",
            "task": self.task,
            "params_used": lgb_params,
            "auto_optimized": auto_optimize,
            "n_features": len(self.feature_names),
            "categorical_features": self.categorical_features,
            "optimization_trials": len(self.optimization_history) if auto_optimize else 0
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        X = self._prepare_data(X)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions for classification."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if not self.is_classifier:
            raise ValueError("predict_proba only available for classifiers.")
        X = self._prepare_data(X)
        return self.model.predict_proba(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        X_test = self._prepare_data(X_test)
        preds = self.model.predict(X_test)
        metrics = {}
        
        if self.is_classifier:
            metrics["accuracy"] = float(accuracy_score(y_test, preds))
            metrics["f1_score"] = float(f1_score(y_test, preds, average='weighted'))
            try:
                probs = self.model.predict_proba(X_test)
                metrics["log_loss"] = float(log_loss(y_test, probs))
                if probs.shape[1] == 2:  # Binary
                    metrics["roc_auc"] = float(roc_auc_score(y_test, probs[:, 1]))
                else:  # Multiclass
                    metrics["roc_auc"] = float(roc_auc_score(y_test, probs, multi_class='ovr'))
            except Exception:
                pass
        else:
            metrics["mse"] = float(mean_squared_error(y_test, preds))
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            metrics["mae"] = float(mean_absolute_error(y_test, preds))
            metrics["r2"] = float(r2_score(y_test, preds))
        
        return metrics

    def save(self, path: str):
        if self.model is None:
            raise ValueError("No model to save.")
        
        save_data = {
            'model': self.model,
            'task': self.task,
            'is_classifier': self.is_classifier,
            'best_params': self.best_params,
            'optimization_history': self.optimization_history,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features
        }
        joblib.dump(save_data, path)

    def load(self, path: str):
        save_data = joblib.load(path)
        self.model = save_data['model']
        self.task = save_data.get('task', 'regression')
        self.is_classifier = save_data.get('is_classifier', False)
        self.best_params = save_data.get('best_params', {})
        self.optimization_history = save_data.get('optimization_history', [])
        self.feature_names = save_data.get('feature_names', [])
        self.categorical_features = save_data.get('categorical_features', [])

    def get_explanations(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Feature importance explanations from LightGBM.
        """
        if self.model is None:
            raise ValueError("No model for explanation.")
        
        # Feature importance (gain-based)
        importance_gain = self.model.feature_importances_
        feature_importance = {
            name: float(imp) 
            for name, imp in zip(self.feature_names, importance_gain)
        }
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            "global": {
                "feature_importance": feature_importance,
                "importance_type": "gain"
            },
            "model_info": {
                "n_estimators": self.model.n_estimators_,
                "best_params": self.best_params,
                "auto_optimized": len(self.optimization_history) > 0
            },
            "optimization_history": self.optimization_history[:10] if self.optimization_history else None
        }


# Register the model
model_factory.register("lightgbm", LightGBMModel)
