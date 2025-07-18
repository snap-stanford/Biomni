import yaml
import importlib
from types import ModuleType
from collections.abc import Callable
from typing import Any  # If needed for type hints

class MCPAdapter:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.tools = {}
        self.load_tools()

    def load_tools(self):
        for tool in self.config.get('tools', []):
            if tool.get('remote_url'):
                # Register remote MCP tool
                self.tools[tool['name']] = self._make_remote_func(tool['remote_url'], tool['endpoint'], tool['input_schema'])
            else:
                # Register local tool
                func = self._import_func(tool['import_path'])
                self.tools[tool['name']] = func


    def _import_func(self, import_path: str) -> Callable:
        module_path, func_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)

    def _make_remote_func(self, base_url: str, endpoint: str, input_schema: dict) -> Callable:
        import requests
        def remote_func(**kwargs):
            payload = {k: kwargs[k] for k in input_schema.keys() if k in kwargs}
            url = base_url.rstrip('/') + endpoint
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            return resp.json().get('result', resp.json())
        return remote_func

    def __getattr__(self, name):
        if name in self.tools:
            return self.tools[name]
        raise AttributeError(f"No MCP tool named {name}")

# Usage:
# adapter = MCPAdapter("/Users/shinde/Desktop/Biomni/biomni_mcp_tools/genetics/tools_config.yaml")
# result = adapter.get_rna_seq_archs4(gene_name="TP53", K=10)
# For remote tools, add 'remote_url' in config:
#   remote_url: "http://localhost:8000"
