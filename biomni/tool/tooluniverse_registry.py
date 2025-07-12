"""
ToolUniverseRegistry: Adapter for integrating ToolUniverse tools into Biomni's tool registry.
"""

from typing import Any

from tooluniverse import ToolUniverse


class ToolUniverseRegistry:
    """
    Adapter for integrating ToolUniverse tools into Biomni's tool registry.
    Loads available tools, exposes their schemas, and allows invocation by name.
    """

    def __init__(self) -> None:
        """Initialize and load all ToolUniverse tools."""
        self.tooluni = ToolUniverse()
        self.tooluni.load_tools()
        self.tool_name_list, self.tool_desc_list = self.tooluni.refresh_tool_name_desc()

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """
        Return all ToolUniverse tool schemas, mapped to Biomni's format.
        Returns:
            List of tool schema dicts with name, description, etc.
        """
        schemas = []
        for name, desc in zip(self.tool_name_list, self.tool_desc_list, strict=False):
            schema = {
                "name": name,
                "description": desc,
                "parameters": {},  # ToolUniverse does not expose input_schema directly
                "output_schema": {},  # ToolUniverse does not expose output_schema directly
                "module": "tooluniverse",
                "source": "ToolUniverse",
            }
            schemas.append(schema)
        return schemas

    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Call a ToolUniverse tool by name with given arguments.
        Args:
            tool_name: Name of the tool to call
            **kwargs: Arguments for the tool
        Returns:
            Result of the tool execution
        """
        query = {"name": tool_name, "arguments": kwargs}
        return self.tooluni.run(query)
