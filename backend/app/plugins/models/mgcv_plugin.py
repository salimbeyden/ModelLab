import os
import subprocess
import pandas as pd
import numpy as np
import uuid
import json
from typing import Dict, Any, Optional, List
from app.core.model_interface import BaseModel
from app.core.model_factory import model_factory
from app.core.schemas import PluginMeta, TaskType


# Family-Link compatibility mapping
FAMILY_LINKS = {
    "gaussian": ["identity", "log", "inverse"],
    "binomial": ["logit", "probit", "cloglog", "cauchit"],
    "poisson": ["log", "identity", "sqrt"],
    "Gamma": ["inverse", "identity", "log"],
    "nb": ["log", "identity", "sqrt"],  # Negative binomial
    "inverse.gaussian": ["1/mu^2", "inverse", "identity", "log"],
    "quasi": ["identity", "log", "logit"],
    "quasibinomial": ["logit", "probit", "cloglog"],
    "quasipoisson": ["log", "identity", "sqrt"],
}

DEFAULT_LINKS = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "Gamma": "inverse",
    "nb": "log",
    "inverse.gaussian": "1/mu^2",
    "quasi": "identity",
    "quasibinomial": "logit",
    "quasipoisson": "log",
}

# Basis types for smooth terms
BASIS_TYPES = {
    "tp": "Thin Plate Regression Spline (default)",
    "cr": "Cubic Regression Spline",
    "cc": "Cyclic Cubic Spline",
    "ps": "P-Spline",
    "cp": "Cyclic P-Spline",
    "ds": "Duchon Spline",
    "re": "Random Effect",
    "mrf": "Markov Random Field",
    "gp": "Gaussian Process",
}


class MGCVModel(BaseModel):
    """
    mgcv (R) Plugin.
    Runs Generalized Additive Models using R's mgcv package.
    Supports advanced configuration for smooth terms, family/link functions,
    and tensor product interactions.
    """
    
    @classmethod
    def get_meta(cls) -> PluginMeta:
        return PluginMeta(
            id="mgcv",
            name="R mgcv (GAM)",
            supported_tasks=[TaskType.REGRESSION, TaskType.CLASSIFICATION],
            description="Generalized Additive Models via R mgcv package. Supports flexible smooth functions with automatic smoothness selection.",
            docs_url="https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gam.html",
            param_schema={
                "type": "object",
                "properties": {
                    "formula_mode": {
                        "type": "string",
                        "enum": ["auto", "manual"],
                        "default": "auto",
                        "title": "Formula Mode",
                        "description": "Auto-generate formula from features or manually specify"
                    },
                    "formula": {
                        "type": "string",
                        "default": "",
                        "title": "R Formula",
                        "description": "Manual R formula (e.g., target ~ s(x1) + x2). Only used if formula_mode is 'manual'."
                    },
                    "method": {
                        "type": "string",
                        "enum": ["REML", "GCV.Cp", "ML", "P-REML", "P-ML"],
                        "default": "REML",
                        "title": "Smoothing Method",
                        "description": "Optimization method for smoothing parameter selection. REML recommended for most cases."
                    },
                    "family": {
                        "type": "string",
                        "enum": ["gaussian", "binomial", "poisson", "Gamma", "nb", "inverse.gaussian", "quasi", "quasibinomial", "quasipoisson"],
                        "default": "gaussian",
                        "title": "Distribution Family",
                        "description": "The distribution family for the response variable"
                    },
                    "link": {
                        "type": "string",
                        "default": "",
                        "title": "Link Function",
                        "description": "Leave empty for default. Options depend on family selection."
                    },
                    "select": {
                        "type": "boolean",
                        "default": False,
                        "title": "Enable Variable Selection",
                        "description": "If TRUE, adds extra penalty to each term for variable selection"
                    },
                    "gamma": {
                        "type": "number",
                        "default": 1.0,
                        "minimum": 0.1,
                        "maximum": 10.0,
                        "title": "Smoothness Multiplier (gamma)",
                        "description": "Increase above 1 for smoother models (reduces overfitting)"
                    },
                    "smooth_terms": {
                        "type": "array",
                        "title": "Smooth Term Configuration",
                        "description": "Configure individual smooth terms for features",
                        "items": {
                            "type": "object",
                            "properties": {
                                "variable": {
                                    "type": "string",
                                    "title": "Variable Name"
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["s", "te", "ti", "linear"],
                                    "default": "s",
                                    "title": "Term Type",
                                    "description": "s=smooth, te=tensor product, ti=tensor interaction, linear=no smooth"
                                },
                                "bs": {
                                    "type": "string",
                                    "enum": ["tp", "cr", "cc", "ps", "cp", "ds", "re", "mrf", "gp"],
                                    "default": "tp",
                                    "title": "Basis Type",
                                    "description": "Type of smoothing basis"
                                },
                                "k": {
                                    "type": "integer",
                                    "default": -1,
                                    "minimum": -1,
                                    "maximum": 100,
                                    "title": "Knots (k)",
                                    "description": "Maximum basis dimension. -1 for automatic selection."
                                },
                                "by": {
                                    "type": "string",
                                    "default": "",
                                    "title": "By Variable",
                                    "description": "Optional factor variable for varying coefficients"
                                }
                            }
                        },
                        "default": []
                    },
                    "tensor_terms": {
                        "type": "array",
                        "title": "Tensor Product Smooths",
                        "description": "Configure interactions between variables using tensor products",
                        "items": {
                            "type": "object",
                            "properties": {
                                "variables": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "title": "Variables",
                                    "description": "Variables to include in tensor product"
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["te", "ti"],
                                    "default": "te",
                                    "title": "Tensor Type",
                                    "description": "te=full tensor, ti=interaction only"
                                },
                                "k": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "title": "Knots per Variable",
                                    "description": "Basis dimension for each variable"
                                }
                            }
                        },
                        "default": []
                    }
                },
                "required": []
            },
            ui_schema={
                "formula_mode": {
                    "ui:widget": "radio",
                    "ui:options": {
                        "inline": True
                    }
                },
                "formula": {
                    "ui:widget": "textarea",
                    "ui:placeholder": "e.g., target ~ s(x1, bs='cr', k=10) + s(x2) + x3",
                    "ui:options": {
                        "rows": 3
                    }
                },
                "method": {
                    "ui:help": "REML is most robust; GCV.Cp for larger datasets"
                },
                "family": {
                    "ui:help": "gaussian for continuous, binomial for binary, poisson for counts"
                },
                "gamma": {
                    "ui:widget": "range"
                },
                "smooth_terms": {
                    "ui:options": {
                        "orderable": True,
                        "addable": True,
                        "removable": True
                    }
                },
                "ui:order": ["formula_mode", "formula", "method", "family", "link", "select", "gamma", "smooth_terms", "tensor_terms"]
            },
            task_schemas={
                TaskType.CLASSIFICATION: {
                    "type": "object",
                    "properties": {
                        "formula_mode": {
                            "type": "string",
                            "enum": ["auto", "manual"],
                            "default": "auto",
                            "title": "Formula Mode"
                        },
                        "formula": {
                            "type": "string",
                            "default": "",
                            "title": "R Formula"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["REML", "GCV.Cp", "ML"],
                            "default": "REML",
                            "title": "Smoothing Method"
                        },
                        "family": {
                            "type": "string",
                            "enum": ["binomial"],
                            "default": "binomial",
                            "title": "Distribution Family"
                        },
                        "link": {
                            "type": "string",
                            "enum": ["logit", "probit", "cloglog", "cauchit"],
                            "default": "logit",
                            "title": "Link Function"
                        },
                        "select": {
                            "type": "boolean",
                            "default": False,
                            "title": "Enable Variable Selection"
                        },
                        "gamma": {
                            "type": "number",
                            "default": 1.0,
                            "minimum": 0.1,
                            "maximum": 10.0,
                            "title": "Smoothness Multiplier"
                        }
                    }
                }
            }
        )

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.tmp_dir = "tmp_r"
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.model_file = None
        self.task = "regression"
        self.is_classifier = False
        self.feature_names: List[str] = []
        self.edf_info: Dict[str, float] = {}

    def _get_r_command(self) -> str:
        """
        Detect Rscript location. Try PATH first, then common Windows paths.
        """
        # 1. Try PATH
        try:
            subprocess.run(["Rscript", "--version"], capture_output=True, check=True)
            return "Rscript"
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
                return path
                
        # 3. Last resort - let the error bubble up if nothing found
        return "Rscript"

    def _sanitize_column_name(self, col: str) -> str:
        """Make column names R-safe."""
        # Wrap in backticks if contains special chars
        if any(c in col for c in [' ', '-', '/', '(', ')', '.', '+', '*']):
            return f"`{col}`"
        return col

    def _build_family_link(self, family: str, link: str) -> str:
        """Build R family(link) expression."""
        default_link = DEFAULT_LINKS.get(family, "identity")
        
        if not link or link == default_link:
            # Use family default
            if family == "nb":
                return "nb()"  # Negative binomial needs special handling
            return family
        else:
            if family == "nb":
                return f'nb(link="{link}")'
            return f'{family}(link="{link}")'

    def _build_formula(self, X: pd.DataFrame, target_name: str, params: Dict[str, Any]) -> str:
        """Build R formula from parameters."""
        formula_mode = params.get("formula_mode", "auto")
        
        if formula_mode == "manual":
            manual_formula = params.get("formula", "").strip()
            if manual_formula:
                return manual_formula
        
        # Auto-generate formula
        smooth_terms = params.get("smooth_terms", [])
        tensor_terms = params.get("tensor_terms", [])
        
        # Create lookup for custom smooth term config
        term_config = {t.get("variable"): t for t in smooth_terms if t.get("variable")}
        
        terms = []
        safe_target = self._sanitize_column_name(target_name)
        
        for col in X.columns:
            safe_col = self._sanitize_column_name(col)
            
            if col in term_config:
                # Use custom configuration
                cfg = term_config[col]
                term_type = cfg.get("type", "s")
                
                if term_type == "linear":
                    terms.append(safe_col)
                else:
                    bs = cfg.get("bs", "tp")
                    k = cfg.get("k", -1)
                    by = cfg.get("by", "")
                    
                    parts = [safe_col]
                    if bs != "tp":
                        parts.append(f'bs="{bs}"')
                    if k > 0:
                        parts.append(f"k={k}")
                    if by:
                        parts.append(f"by={self._sanitize_column_name(by)}")
                    
                    terms.append(f"{term_type}({', '.join(parts)})")
            else:
                # Default: smooth for numeric, linear for categorical
                if X[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    terms.append(f"s({safe_col})")
                else:
                    terms.append(safe_col)
        
        # Add tensor product terms
        for tensor in tensor_terms:
            variables = tensor.get("variables", [])
            if len(variables) >= 2:
                tensor_type = tensor.get("type", "te")
                k_values = tensor.get("k", [])
                
                safe_vars = [self._sanitize_column_name(v) for v in variables]
                
                if k_values and len(k_values) == len(variables):
                    k_str = f", k=c({','.join(map(str, k_values))})"
                else:
                    k_str = ""
                
                terms.append(f"{tensor_type}({', '.join(safe_vars)}{k_str})")
        
        formula = f"{safe_target} ~ " + " + ".join(terms)
        return formula

    def train(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
        self.task = params.get("task", "regression")
        self.is_classifier = (self.task == "classification")
        self.feature_names = list(X.columns)
        
        run_id = str(uuid.uuid4())
        data_path = os.path.abspath(os.path.join(self.tmp_dir, f"train_{run_id}.csv"))
        script_path = os.path.abspath(os.path.join(self.tmp_dir, f"script_{run_id}.R"))
        output_model_path = os.path.abspath(os.path.join(self.tmp_dir, f"model_{run_id}.rds"))
        results_path = os.path.abspath(os.path.join(self.tmp_dir, f"results_{run_id}.json"))
        gam_check_path = os.path.abspath(os.path.join(self.tmp_dir, f"gam_check_{run_id}.txt"))
        
        target_name = y.name if y.name else "target"
        
        # Combine X and y for training
        full_df = X.copy()
        full_df[target_name] = y
        full_df.to_csv(data_path, index=False)
        
        # Build formula
        formula = self._build_formula(X, target_name, params)
        print(f"[R-GAM] Using formula: {formula}")
        
        # Build family/link
        family = params.get("family", "gaussian")
        link = params.get("link", "")
        family_expr = self._build_family_link(family, link)
        print(f"[R-GAM] Family: {family_expr}")
        
        # Other parameters
        method = params.get("method", "REML")
        select = "TRUE" if params.get("select", False) else "FALSE"
        gamma = params.get("gamma", 1.0)

        r_script = f'''
library(mgcv)
library(jsonlite)

# Load data
df <- read.csv("{data_path.replace('\\', '/')}")

# Convert character columns to factors
for (col in names(df)) {{
    if (is.character(df[[col]])) {{
        df[[col]] <- as.factor(df[[col]])
    }}
}}

# Train GAM model
tryCatch({{
    model <- gam(
        as.formula("{formula}"),
        data = df,
        family = {family_expr},
        method = "{method}",
        select = {select},
        gamma = {gamma}
    )
    
    # Save model
    saveRDS(model, file = "{output_model_path.replace('\\', '/')}")
    
    # Extract summary
    summ <- summary(model)
    
    # Get EDF for each smooth term
    edf_list <- list()
    if (!is.null(summ$s.table)) {{
        for (i in 1:nrow(summ$s.table)) {{
            term_name <- rownames(summ$s.table)[i]
            edf_list[[term_name]] <- summ$s.table[i, "edf"]
        }}
    }}
    
    # Prepare results
    results <- list(
        deviance_explained = summ$dev.expl,
        r_squared = summ$r.sq,
        aic = AIC(model),
        bic = BIC(model),
        gcv_score = model$gcv.ubre,
        scale_estimate = model$scale,
        n = nrow(df),
        edf = edf_list,
        formula = as.character(formula(model)),
        method = model$method,
        family = model$family$family,
        link = model$family$link
    )
    
    # Add smooth term p-values if available
    if (!is.null(summ$s.table)) {{
        p_values <- list()
        for (i in 1:nrow(summ$s.table)) {{
            term_name <- rownames(summ$s.table)[i]
            p_values[[term_name]] <- summ$s.table[i, "p-value"]
        }}
        results$smooth_pvalues <- p_values
    }}
    
    # For classification, add additional metrics
    if ("{family}" == "binomial") {{
        preds <- predict(model, type = "response")
        pred_class <- ifelse(preds > 0.5, 1, 0)
        actual <- df[["{target_name}"]]
        if (is.factor(actual)) {{
            actual <- as.numeric(actual) - 1
        }}
        accuracy <- mean(pred_class == actual)
        results$accuracy <- accuracy
    }}
    
    write(toJSON(results, auto_unbox = TRUE, pretty = TRUE), "{results_path.replace('\\', '/')}")
    
    # Run gam.check and save output
    sink("{gam_check_path.replace('\\', '/')}")
    gam.check(model)
    sink()
    
    cat("GAM training completed successfully\\n")
    
}}, error = function(e) {{
    error_result <- list(error = conditionMessage(e))
    write(toJSON(error_result, auto_unbox = TRUE), "{results_path.replace('\\', '/')}")
    stop(e)
}})
'''
        with open(script_path, "w") as f:
            f.write(r_script)
            
        # Execute R
        r_cmd = self._get_r_command()
        try:
            result = subprocess.run([r_cmd, script_path], capture_output=True, text=True, check=True)
            print("R Output:", result.stdout)
        except FileNotFoundError:
            raise RuntimeError(f"[DEBUG-R-PATH] Rscript not found at '{r_cmd}'. Please ensure R is installed.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"R execution failed: {e.stderr}")
            
        # Load results
        with open(results_path, "r") as f:
            training_results = json.load(f)
        
        if "error" in training_results:
            raise RuntimeError(f"R GAM training failed: {training_results['error']}")
            
        self.model_file = output_model_path
        self.metadata = training_results
        self.edf_info = training_results.get("edf", {})
        
        # Load gam.check output if available
        gam_check_output = ""
        if os.path.exists(gam_check_path):
            with open(gam_check_path, "r") as f:
                gam_check_output = f.read()
        
        return {
            "model_type": "mgcv",
            "formula": formula,
            "family": family_expr,
            "method": method,
            "metrics": training_results,
            "gam_check": gam_check_output
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.model_file:
            raise ValueError("Model not trained.")
            
        run_id = str(uuid.uuid4())
        data_path = os.path.abspath(os.path.join(self.tmp_dir, f"pred_X_{run_id}.csv"))
        script_path = os.path.abspath(os.path.join(self.tmp_dir, f"pred_script_{run_id}.R"))
        out_pred_path = os.path.abspath(os.path.join(self.tmp_dir, f"preds_{run_id}.csv"))
        
        X.to_csv(data_path, index=False)
        
        pred_type = "response" if self.is_classifier else "response"
        
        r_script = f'''
library(mgcv)
model <- readRDS("{self.model_file.replace('\\', '/')}")
new_data <- read.csv("{data_path.replace('\\', '/')}")

# Convert character columns to factors
for (col in names(new_data)) {{
    if (is.character(new_data[[col]])) {{
        new_data[[col]] <- as.factor(new_data[[col]])
    }}
}}

preds <- predict(model, new_data, type = "{pred_type}")
write.csv(preds, "{out_pred_path.replace('\\', '/')}", row.names=FALSE)
'''
        with open(script_path, "w") as f:
            f.write(r_script)
            
        r_cmd = self._get_r_command()
        subprocess.run([r_cmd, script_path], check=True, capture_output=True)
        preds_df = pd.read_csv(out_pred_path)
        
        preds = preds_df.values.flatten()
        
        # For classification, convert probabilities to class labels
        if self.is_classifier:
            preds = (preds > 0.5).astype(int)
        
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability predictions for classification."""
        if not self.model_file or not self.is_classifier:
            raise ValueError("Model not trained or not a classifier.")
        
        run_id = str(uuid.uuid4())
        data_path = os.path.abspath(os.path.join(self.tmp_dir, f"proba_X_{run_id}.csv"))
        script_path = os.path.abspath(os.path.join(self.tmp_dir, f"proba_script_{run_id}.R"))
        out_pred_path = os.path.abspath(os.path.join(self.tmp_dir, f"proba_{run_id}.csv"))
        
        X.to_csv(data_path, index=False)
        
        r_script = f'''
library(mgcv)
model <- readRDS("{self.model_file.replace('\\', '/')}")
new_data <- read.csv("{data_path.replace('\\', '/')}")

for (col in names(new_data)) {{
    if (is.character(new_data[[col]])) {{
        new_data[[col]] <- as.factor(new_data[[col]])
    }}
}}

probs <- predict(model, new_data, type = "response")
write.csv(probs, "{out_pred_path.replace('\\', '/')}", row.names=FALSE)
'''
        with open(script_path, "w") as f:
            f.write(r_script)
        
        r_cmd = self._get_r_command()
        subprocess.run([r_cmd, script_path], check=True, capture_output=True)
        
        probs = pd.read_csv(out_pred_path).values.flatten()
        # Return as 2D array for compatibility
        return np.column_stack([1 - probs, probs])

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        preds = self.predict(X_test)
        
        if self.is_classifier:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
            
            # Get probabilities for AUC
            try:
                proba = self.predict_proba(X_test)[:, 1]
            except:
                proba = None
            
            metrics = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "f1": float(f1_score(y_test, preds, average='weighted')),
            }
            
            if proba is not None:
                try:
                    metrics["auc"] = float(roc_auc_score(y_test, proba))
                    metrics["log_loss"] = float(log_loss(y_test, proba))
                except:
                    pass
            
            return metrics
        else:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            return {
                "mse": float(mean_squared_error(y_test, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                "mae": float(mean_absolute_error(y_test, preds)),
                "r2": float(r2_score(y_test, preds))
            }

    def save(self, path: str):
        if not self.model_file:
            raise ValueError("No model.")
        import shutil
        shutil.copy(self.model_file, path)

    def load(self, path: str):
        self.model_file = path

    def get_explanations(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Generate GAM-specific explanations including partial dependence plots."""
        if not self.model_file:
            return {}
        
        run_id = str(uuid.uuid4())
        script_path = os.path.abspath(os.path.join(self.tmp_dir, f"explain_script_{run_id}.R"))
        results_path = os.path.abspath(os.path.join(self.tmp_dir, f"explain_{run_id}.json"))
        
        r_script = f'''
library(mgcv)
library(jsonlite)

model <- readRDS("{self.model_file.replace('\\', '/')}")
summ <- summary(model)

# Extract feature importance based on chi-square statistic
importance <- list()
if (!is.null(summ$s.table)) {{
    for (i in 1:nrow(summ$s.table)) {{
        term_name <- rownames(summ$s.table)[i]
        importance[[term_name]] <- list(
            edf = summ$s.table[i, "edf"],
            ref_df = summ$s.table[i, "Ref.df"],
            chi_sq = summ$s.table[i, "Chi.sq"],
            p_value = summ$s.table[i, "p-value"]
        )
    }}
}}

# Parametric terms
param_importance <- list()
if (!is.null(summ$p.table) && nrow(summ$p.table) > 0) {{
    for (i in 1:nrow(summ$p.table)) {{
        term_name <- rownames(summ$p.table)[i]
        param_importance[[term_name]] <- list(
            estimate = summ$p.table[i, "Estimate"],
            std_error = summ$p.table[i, "Std. Error"],
            t_value = summ$p.table[i, "t value"],
            p_value = summ$p.table[i, "Pr(>|t|)"]
        )
    }}
}}

results <- list(
    smooth_terms = importance,
    parametric_terms = param_importance,
    deviance_explained = summ$dev.expl,
    r_squared = summ$r.sq
)

write(toJSON(results, auto_unbox = TRUE, pretty = TRUE), "{results_path.replace('\\', '/')}")
'''
        
        with open(script_path, "w") as f:
            f.write(r_script)
        
        r_cmd = self._get_r_command()
        try:
            subprocess.run([r_cmd, script_path], check=True, capture_output=True)
            
            with open(results_path, "r") as f:
                explanations = json.load(f)
            
            # Format for frontend
            feature_importance = {}
            for term, info in explanations.get("smooth_terms", {}).items():
                # Extract feature name from term like "s(feature)"
                feature_name = term.replace("s(", "").replace(")", "").replace("`", "")
                feature_importance[feature_name] = {
                    "edf": info.get("edf", 0),
                    "chi_sq": info.get("chi_sq", 0),
                    "p_value": info.get("p_value", 1)
                }
            
            return {
                "type": "gam",
                "feature_importance": feature_importance,
                "smooth_terms": explanations.get("smooth_terms", {}),
                "parametric_terms": explanations.get("parametric_terms", {}),
                "deviance_explained": explanations.get("deviance_explained", 0),
                "r_squared": explanations.get("r_squared", 0)
            }
        except Exception as e:
            print(f"[GAM] Explanation extraction failed: {e}")
            return {"type": "gam", "feature_importance": {}}


# Register with factory
model_factory.register("mgcv", MGCVModel)
