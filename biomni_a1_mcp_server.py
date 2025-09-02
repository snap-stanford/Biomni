#!/usr/bin/env python3
"""
MCP Server exposing Biomni A1 agent capabilities.
Run this from the Biomni directory where the package is installed.
"""
import sys
from biomni.agent.a1 import A1


def main():
    """Initialize and run the Biomni A1 MCP server."""
    try:
        # Initialize A1 agent
        agent = A1(
            path="./data",
            llm="claude-3-5-sonnet-20241022",
            use_tool_retriever=True,
            timeout_seconds=600
        )
        
        # Create MCP server exposing biomedical tools
        mcp_server = agent.create_mcp_server(tool_modules=[
            "biomni.tool.literature",
            "biomni.tool.database", 
            "biomni.tool.genomics",
            "biomni.tool.molecular_biology",
            "biomni.tool.cell_biology"
        ])
        
        print("Biomni A1 MCP server ready", file=sys.stderr)
        mcp_server.run(transport="stdio")
        
    except Exception as e:
        print(f"Failed to start Biomni A1 MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
