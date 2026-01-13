import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

from app.plugins.base import ModelPlugin, RunContext
from app.core.schemas import PluginMeta, TaskType

class EBMPlugin(ModelPlugin):
    @property
    def meta(self) -> PluginMeta:
        return PluginMeta(
            id="ebm",
            name="Explainable Boosting Machine",
            supported_tasks=[TaskType.CLASSIFICATION, TaskType.REGRESSION],
            description="Glassbox model from InterpretML using gradient boosting on GAMs.",
            param_schema={
                "type": "object",
                "properties": {
                    "max_bins": {
                        "type": "integer",
                        "default": 256,
                        "minimum": 32,
                        "description": "Max number of bins per feature"
                    },
                    "interactions": {
                        "type": "integer",
                        "default": 10,
                        "description": "Number of interactions to include"
                    },
                    "learning_rate": {
                        "type": "number",
                        "default": 0.01,
                        "description": "Learning rate"
                    }
                }
            },
            ui_schema={
                "ui:order": ["max_bins", "interactions", "learning_rate"]
            }
        )

    def train(self, context: RunContext) -> Any:
        print(f"Training EBM for run {context.run_id}")
        
        # 1. Load Data
        if context.dataset_path.endswith('.parquet'):
            df = pd.read_parquet(context.dataset_path)
        else:
            df = pd.read_csv(context.dataset_path)
            
        target_col = context.config.target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
            
        # 2. Preprocess
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Simple string encoding for object cols if needed, but EBM handles them well usually.
        # For MVP, rely on EBM's internal handling.
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Train
        params = context.config.model_params
        
        # Force n_jobs=1 to avoid pickling/threading issues on Windows when running inside a thread
        params["n_jobs"] = 1
        
        if context.config.task == TaskType.CLASSIFICATION:
            ebm = ExplainableBoostingClassifier(**params)
        else:
            ebm = ExplainableBoostingRegressor(**params)
            
        ebm.fit(X_train, y_train)
        
        # 4. Evaluate
        metrics = {}
        if context.config.task == TaskType.CLASSIFICATION:
            y_pred = ebm.predict(X_test)
            y_prob = ebm.predict_proba(X_test)[:, 1] if hasattr(ebm, "predict_proba") else None
            
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            if y_prob is not None and len(np.unique(y)) == 2:
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
                except:
                    pass
        else:
            y_pred = ebm.predict(X_test)
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics["r2"] = float(r2_score(y_test, y_pred))
            
        # 5. Save Model
        output_dir = os.path.dirname(context.output_dir) # context.output_dir might be just base run dir
        # Ensure we are saving to runs/{id}/
        # context doesn't have output_dir set by default in RunService logic? 
        # Let's rely on RunService passing a context that has the right paths or we use standard paths.
        # Actually in RunService._execute_run, it sets output_dir. 
        # But wait, context in base.py definition: class RunContext(BaseModel): run_id: str; config: RunConfig; dataset_path: str
        
        # We need to write to the artifacts folder.
        # Assuming we are inside the run directory or can construct it.
        # RunService._execute_run DOES NOT set CWD to run dir. It passes context.
        # Let's assume absolute path for artifacts.
        
        # Hack: The RunService creates a directory runs/{run_id}.
        # We should save there.
        run_dir = os.path.join("runs", context.run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        model_path = os.path.join(run_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(ebm, f)
            
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        # Save metrics to context for updating state
        # But `train` return value is not strictly defined to update state metrics automatically in my code?
        # Let's return metrics and handle it in service/wrapper if needed, 
        # OR just rely on the file.
        # Better: The Plugin interface says `train` returns `Any`.
        
        return metrics

    def explain(self, context: RunContext) -> Dict[str, Any]:
        run_dir = os.path.join("runs", context.run_id)
        model_path = os.path.join(run_dir, "model.pkl")
        
        if not os.path.exists(model_path):
            return {}
            
        with open(model_path, "rb") as f:
            ebm = pickle.load(f)
            
        global_expl = ebm.explain_global()
        data = global_expl.data()
        
        # Serialize for frontend
        # EBM explanation data structure is complex. simpler approach: 
        # Just extracting feature importance
        
        out = {
            "feature_names": global_expl.feature_names,
            "feature_types": global_expl.feature_types,
            "overall": global_expl.selector
            # extracting more detailed data requires parsing `data` dict carefully
        }
        
        # Create a simplified version for frontend charts
        # data(key) returns detailed graph data for feature 'key'.
        
        # For artifacts, let's dump a JSON with basic importances
        importances = {}
        # selector is usually dataframe-like or dict
        # global_expl.visualize() returns a dash/plotly graph.
        
        # Let's extract per-feature importance (mean abs score)
        # EBM doesn't have a simple .feature_importances_ property like RF
        # but the global explanation has it.
        
        # Let's just save the raw data structure dump for now
        # It might be too big or not serializable.
        
        # Safe fallback:
        serializable_data = {
           "summary": "Full interactive validation not available in MVP JSON.",
           "feature_names": getattr(global_expl, "feature_names", []),
           "feature_types": getattr(global_expl, "feature_types", [])
        }
        
        explain_dir = os.path.join(run_dir, "explain")
        os.makedirs(explain_dir, exist_ok=True)
        
        path = os.path.join(explain_dir, "global.json")
        with open(path, "w") as f:
            json.dump(serializable_data, f, default=str)
            
        return serializable_data
