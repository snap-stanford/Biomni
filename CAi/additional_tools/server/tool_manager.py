"""tool_manager.py"""

import json
from pathlib import Path


class ToolManager:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.tools_dir = self.base_dir / "tools"
        self.tools = {}
        self.load_tools()

    def load_tools(self):
        for tool_dir in self.tools_dir.iterdir():
            if not tool_dir.is_dir():
                continue

            config_file = tool_dir / "config.json"
            if not config_file.exists():
                continue

            config = json.loads(config_file.read_text())
            tool_name = tool_dir.name
            actions = config.get("actions", {"default": "run.py"})
            scripts = {
                action_name: str((tool_dir / script_name).resolve()) for action_name, script_name in actions.items()
            }

            self.tools[tool_name] = {
                "name": tool_name,
                "env": config.get("conda_env"),
                "scripts": scripts,  # 保存所有 action 对应的脚本路径
                "config": config,
            }

    # 删掉了重复的 get_tool，保留一个就好
    def get(self, name):
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        return self.tools[name]


tool_manager = ToolManager()
