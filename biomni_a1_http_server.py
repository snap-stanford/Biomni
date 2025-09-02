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
        
        # Get port from environment or use default
        port = int(os.environ.get("BIOMNI_A1_PORT", "3700"))
        host = os.environ.get("BIOMNI_A1_HOST", "127.0.0.1")
        
        # Create MCP server with proper HTTP transport configuration
        print("Creating MCP server...", file=sys.stderr)
        from mcp.server.fastmcp import FastMCP
        
        mcp_server = FastMCP(
            name="BiomniA1",
            host=host,
            port=port
        )
        
        # Add health check endpoint
        @mcp_server.tool()
        def health_check() -> dict:
            """Health check endpoint to verify server is running."""
            return {"status": "healthy", "server": "BiomniA1", "port": port}
        
        # Use the existing agent's MCP server creation method and extract tools
        print("Registering Biomni tools...", file=sys.stderr)
        try:
            # Create a temporary MCP server using agent's method to get the tools
            temp_mcp = agent.create_mcp_server(
                tool_modules=[
                    "biomni.tool.literature",
                    "biomni.tool.database", 
                    "biomni.tool.genomics",
                    "biomni.tool.molecular_biology",
                    "biomni.tool.cell_biology",
                    "biomni.tool.support_tools",
                ]
            )
            
            # Extract the registered tools from the temporary server
            # Note: This is a simplified approach - in production you might want 
            # to properly transfer all the tool configurations
            registered_tools = 0
            if hasattr(temp_mcp, '_tools'):
                for tool_name, tool_func in temp_mcp._tools.items():
                    try:
                        # Register the tool with our HTTP-configured server
                        mcp_server.tool()(tool_func)
                        registered_tools += 1
                    except Exception as e:
                        print(f"Warning: Failed to register tool '{tool_name}': {e}", file=sys.stderr)
                        
            print(f"Created MCP server with {registered_tools + 1} tools (including health check)", file=sys.stderr)
            
        except Exception as e:
            print(f"Warning: Could not register Biomni tools: {e}", file=sys.stderr)
            print("Server will run with health check only", file=sys.stderr)
        print(f"Starting Biomni A1 HTTP MCP server on {host}:{port}...", file=sys.stderr)
        
        # Run with SSE transport (HTTP-based) for MCP client compatibility
        mcp_server.run(transport="sse")
        
    except Exception as e:
        print(f"Failed to start Biomni A1 MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
