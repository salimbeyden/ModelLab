from abc import ABC, abstractmethod
from typing import Any, Dict
from app.core.schemas import PluginMeta, RunConfig

class RunContext:
    """
    Context object passed to plugins during execution.
    Provides access to data, configuration, and artifact writing capabilities.
    """
    def __init__(self, run_id: str, config: RunConfig, dataset_path: str, output_dir: str):
        self.run_id = run_id
        self.config = config
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.X_train = None
        self.y_train = None
        self.model = None
        # Add other shared state as needed

    def log(self, message: str):
        # Implementation to write to logs.txt
        pass

class ModelPlugin(ABC):
    @property
    @abstractmethod
    def meta(self) -> PluginMeta:
        pass
    
    @abstractmethod
    def train(self, context: RunContext) -> Any:
        """
        Train the model using data in context. 
        Should save the model to artifact store and return it (or a reference).
        """
        pass
    
    @abstractmethod
    def explain(self, context: RunContext) -> Dict[str, Any]:
        """
        Generate global explanations.
        """
        pass
