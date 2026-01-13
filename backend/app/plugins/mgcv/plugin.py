import subprocess
import os
import json
import pandas as pd
from typing import Dict, Any, List
from app.plugins.base import ModelPlugin, RunContext
from app.core.schemas import PluginMeta, TaskType

class MgcvPlugin(ModelPlugin):
    @property
    def meta(self) -> PluginMeta:
        return PluginMeta(
            id="mgcv",
            name="R mgcv (GAM)",
            supported_tasks=[TaskType.REGRESSION],
            description="Generalized Additive Models via R mgcv package.",
            param_schema={
                "type": "object",
                "properties": {
                    "formula": {
                        "type": "string", 
                        "default": "y ~ s(x1) + s(x2)",
                        "description": "R formula for the GAM"
                    },
                    "family": {
                        "type": "string",
                        "default": "gaussian",
                        "enum": ["gaussian", "poisson", "binomial", "Gamma"],
                        "description": "Family for the response variable"
                    }
                }
            },
            ui_schema={
                "formula": {"ui:widget": "textarea"},
                "ui:order": ["formula", "family"]
            }
        )

    def train(self, context: RunContext) -> Any:
        print(f"Training mgcv for run {context.run_id}")
        
        # 1. Prepare Data for R
        # We assume X_train and y_train are available in context, or we load from dataset
        # For MVP, let's assume context.data_dir has the dataset file
        
        # 2. Run Rscript
        # We'll need a wrapper script 'train_mgcv.R' to be present or written on fly.
        # For robustness, we'll write it to the run directory.
        
        r_script_path = os.path.join(context.output_dir, "train_mgcv.R")
        self._write_r_script(r_script_path, context)
        
        cmd = ["Rscript", r_script_path]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=context.output_dir)
            print("R Output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("R Error:", e.stderr)
            raise RuntimeError(f"Rscript failed: {e.stderr}")

        return "model.rds"

    def explain(self, context: RunContext) -> Dict[str, Any]:
        # check if plots exist in outputs?
        return {"type": "mgcv_plots", "files": ["plots/term_plot.png"]}

    def _write_r_script(self, path: str, context: RunContext):
        # Dynamically generate R script based on config
        params = context.config.model_params
        formula = params.get("formula", "target ~ .")
        family = params.get("family", "gaussian")
        
        # Pre-convert path for R
        data_path_r = os.path.join(context.data_dir, 'dataset.csv').replace(os.sep, '/')
        
        # This is a simplified R script template
        script = f"""
        library(mgcv)
        library(readr)
        
        # Load data (assuming parquet or csv)
        # For now, hardcoded assumption of 'train.csv' in run dir or similar
        data <- read_csv("{data_path_r}")
        
        # Train model
        model <- gam(as.formula("{formula}"), data=data, family={family})
        
        # Save model
        saveRDS(model, "model.rds")
        
        # Plot
        png("plots/term_plot.png")
        plot(model, pages=1)
        dev.off()
        
        # Write metrics (dummy for now)
        write_csv(data.frame(rmse=sqrt(mean(model$residuals^2))), "metrics.csv")
        """
        
        # Ensure plots dir exists
        os.makedirs(os.path.join(context.output_dir, "plots"), exist_ok=True)
        
        with open(path, "w") as f:
            f.write(script)
