import importlib
import yaml
from typing import Any, Callable, Dict

class ToolLoader:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.tools = {}
        self.load_tools()

    def load_tools(self):
        for tool in self.config.get('tools', []):
            func = self._import_func(tool['import_path'])
            self.tools[tool['name']] = {
                'func': func,
                'endpoint': tool['endpoint'],
                'input_schema': tool['input_schema']
            }

    def _import_func(self, import_path: str) -> Callable:
        module_path, func_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)

    def get_tool(self, name: str) -> Dict[str, Any]:
        return self.tools.get(name)
