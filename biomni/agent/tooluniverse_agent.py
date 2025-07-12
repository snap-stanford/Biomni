import json
from biomni.agent.base_agent import base_agent
from biomni.tool.tooluniverse_registry import ToolUniverseRegistry

class ToolUniverseAgent(base_agent):
    """Agent that exposes all ToolUniverse tools as callable tools in the Biomni agent pipeline."""
    def __init__(self, llm="gpt-4", cheap_llm=None):
        super().__init__(llm, cheap_llm, tools=None)
        self.tu_registry = ToolUniverseRegistry()
        self.tu_tools = self.tu_registry.get_tool_schemas()
        self.log = []
        self.configure()

    def configure(self):
        """Configure the agent with ToolUniverse tools."""
        self.tool_schemas = {tool["name"]: tool for tool in self.tu_tools}

    def go(self, tool_name, **kwargs):
        """Call a ToolUniverse tool by name with arguments, log the call and return the result."""
        self.log.append(("user", f"Call ToolUniverse tool: {tool_name} with args: {kwargs}"))
        result = self.tu_registry.call_tool(tool_name, **kwargs)
        self.log.append(("assistant", json.dumps(result, indent=2)))
        return self.log, result
