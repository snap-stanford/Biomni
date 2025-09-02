#!/usr/bin/env python3
"""
Standalone HTTP MCP Server exposing Biomni A1 agent.
Run this from the Biomni directory where the package is installed.
"""
import sys
import os
from biomni.agent.a1 import A1


def main():
    """Initialize and run the Biomni A1 HTTP MCP server."""
    try:
        print("Initializing Biomni A1 agent...", file=sys.stderr)
        
        # Initialize A1 agent with biomni_data in current directory
        agent = A1(
            path="./biomni_data",
            llm="claude-3-5-sonnet-20241022",
            use_tool_retriever=True,
            timeout_seconds=600,
        )
        
        # Create MCP server exposing key biomedical tool modules
        print("Creating MCP server...", file=sys.stderr)
        mcp_server = agent.create_mcp_server(
            tool_modules=[
                "biomni.tool.literature",
                "biomni.tool.database",
                "biomni.tool.genomics",
                "biomni.tool.molecular_biology",
                "biomni.tool.cell_biology",
                "biomni.tool.support_tools",
            ]
        )
        
        # Get port from environment or use default
        port = int(os.environ.get("BIOMNI_A1_PORT", "8080"))
        
        print(f"Starting Biomni A1 HTTP MCP server on port {port}...", file=sys.stderr)
        mcp_server.run(transport="sse", port=port)
        
    except Exception as e:
        print(f"Failed to start Biomni A1 MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
