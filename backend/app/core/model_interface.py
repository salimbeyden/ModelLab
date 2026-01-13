from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from app.core.schemas import PluginMeta, TaskType

class BaseModel(ABC):
    """
    Abstract Base Class for all model plugins.
    Ensures a consistent interface for the TrainingService.
    """
    
    @classmethod
    @abstractmethod
    def get_meta(cls) -> PluginMeta:
        """Return metadata for this model plugin."""
        pass

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        Returns a dictionary of training summaries/stats.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for the provided features.
        """
        pass

    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model and return a dictionary of metrics.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Serialize the model to disk.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load a serialized model from disk.
        """
        pass

    @abstractmethod
    def get_explanations(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Return model explanations (feature importance, local explanations, etc.)
        """
        pass
