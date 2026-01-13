from typing import Dict, List, Type
from app.plugins.base import ModelPlugin
from app.core.schemas import PluginMeta

class PluginRegistry:
    def __init__(self):
        self._models: Dict[str, ModelPlugin] = {}

    def register_model(self, plugin: ModelPlugin):
        # In a real app, might want to check for duplicates or validate
        self._models[plugin.meta.id] = plugin

    def get_model(self, model_id: str) -> ModelPlugin:
        return self._models.get(model_id)

    def list_models(self) -> List[PluginMeta]:
        return [p.meta for p in self._models.values()]

# Global registry instance
registry = PluginRegistry()
