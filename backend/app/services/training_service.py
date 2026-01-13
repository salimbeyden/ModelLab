import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

from app.core.model_factory import model_factory
from app.core.schemas import RunConfig
from app.plugins.models import * # Ensure all plugins are registered

class TrainingService:
    """
    Orchestrates the training lifecycle:
    1. Data Loading & Splitting
    2. Model Selection & Instantiation
    3. Training & Evaluation
    4. Explanation Generation
    5. Artifact Persistence
    """
    
    def train_model(self, config: RunConfig, df: pd.DataFrame, output_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # 1. Prepare Data
        target = config.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe.")
            
        # Select features if specified
        if config.features:
            # Ensure target is NOT in features (usually it isn't but being safe)
            features = [f for f in config.features if f != target]
            X = df[features]
        else:
            X = df.drop(columns=[target])
            
        y = df[target]
        
        # Split with configurable test_size
        test_size = config.test_size if 0.0 < config.test_size < 1.0 else 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # 2. Get Model from Factory
        model = model_factory.get_model(config.model_id)
        
        # 3. Train
        # Combine config task into params
        train_params = config.model_params.copy()
        train_params["task"] = config.task
        
        print(f"Training {config.model_id} on {len(X_train)} samples...")
        train_summary = model.train(X_train, y_train, train_params)
        
        # 4. Evaluate
        metrics = model.evaluate(X_test, y_test)
        print(f"Evaluation complete: {metrics}")
        
        # 5. Save Model Artifact
        model_filename = "model.joblib" if config.model_id != 'mgcv' else "model.rds"
        model_path = os.path.join(output_dir, model_filename)
        model.save(model_path)
        
        # 6. Generate Explanations
        # Pass some background data for local explanations if needed
        explanations = model.get_explanations(X_test.head(10))
        
        # Save explanations as artifact
        import json
        with open(os.path.join(output_dir, "explanations.json"), "w") as f:
            json.dump(explanations, f, indent=2, default=str)
            
        # 7. Final Bundle
        result_metrics = {
            **metrics,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        
        return result_metrics, explanations

training_service = TrainingService()
