import importlib
from collections.abc import Callable
from typing import Any, Dict

import yaml
<<<<<<< HEAD
from typing import Any  # keep this if Any is used
from collections.abc import Callable
=======

>>>>>>> 78c2a00bcbe09dbfd74780c53c87c391cdd9c4e4

class ToolLoader:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.tools = {}
        self.load_tools()

    def load_tools(self):
        for tool in self.config.get("tools", []):
            func = self._import_func(tool["import_path"])
            self.tools[tool["name"]] = {
                "func": func,
                "endpoint": tool["endpoint"],
                "input_schema": tool["input_schema"],
            }

    def _import_func(self, import_path: str) -> Callable:
        module_path, func_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)

    def get_tool(self, name: str) -> dict[str, Any]:
        return self.tools.get(name)
