from typing import Dict, Type, Any, List
from app.core.model_interface import BaseModel
from app.core.schemas import PluginMeta

class ModelFactory:
    """
    Registry and factory for model plugins.
    """
    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, model_type: str, model_class: Type[BaseModel]):
        cls._models[model_type] = model_class

    @classmethod
    def get_model(cls, model_type: str, **kwargs) -> BaseModel:
        if model_type not in cls._models:
            raise ValueError(f"Model type '{model_type}' is not registered.")
        return cls._models[model_type](model_id=model_type, **kwargs)

    @classmethod
    def list_available_models(cls) -> List[PluginMeta]:
        return [cl.get_meta() for cl in cls._models.values()]

model_factory = ModelFactory()
