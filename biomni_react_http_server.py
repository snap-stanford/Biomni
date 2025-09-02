#!/usr/bin/env python3
"""
Standalone HTTP MCP Server exposing Biomni ReAct agent.
Run this from the Biomni directory where the package is installed.
"""
import sys
import os
from mcp.server.fastmcp import FastMCP
from biomni.agent.react import react


def main():
    """Initialize and run the Biomni ReAct HTTP MCP server."""
    try:
        print("Initializing Biomni ReAct agent...", file=sys.stderr)
        
        # Initialize ReAct agent
        agent = react(
            path="./biomni_data",
            llm="claude-3-5-sonnet-20241022",
            use_tool_retriever=True,
            timeout_seconds=300,
        )
        
        # Configure with controlled capabilities for eQMS compliance
        agent.configure(
            plan=True,  # Enable planning
            reflect=True,  # Enable reflection
            data_lake=True,  # Access to bio data lake
            library_access=True,  # Access to software libraries
        )
        
        # Get port from environment or use default
        port = int(os.environ.get("BIOMNI_REACT_PORT", "3800"))
        host = os.environ.get("BIOMNI_REACT_HOST", "127.0.0.1")
        
        # Create MCP server with proper HTTP transport configuration
        print("Creating MCP server...", file=sys.stderr)
        mcp_server = FastMCP(
            name="BiomniReAct",
            host=host,
            port=port
        )
        
        # Add health check endpoint
        @mcp_server.tool()
        def health_check() -> dict:
            """Health check endpoint to verify server is running."""
            return {"status": "healthy", "server": "BiomniReAct", "port": port}
        
        @mcp_server.tool()
        def analyze_biomedical_query(query: str, file_contexts: list[str] = None) -> dict:
            """
            Analyze biomedical queries using curated tools (no arbitrary code execution).
            
            Args:
                query: The biomedical question or analysis request
                file_contexts: Optional list of file contents to include in analysis
            """
            try:
                # Enhance query with file contexts if provided
                enhanced_query = query
                if file_contexts:
                    context_text = "\n".join(
                        [f"File {i + 1}:\n{content}\n" for i, content in enumerate(file_contexts)]
                    )
                    enhanced_query = f"{query}\n\nProvided file contexts:\n{context_text}"
                
                # Execute ReAct agent
                execution_log, final_response = agent.go(enhanced_query)
                
                return {
                    "success": True,
                    "response": final_response,
                    "execution_log": [str(step) for step in execution_log],
                    "agent_type": "react",
                }
                
            except Exception as e:
                return {"success": False, "error": str(e), "agent_type": "react"}
        
        print("Created MCP server with 2 tools (health_check, analyze_biomedical_query)", file=sys.stderr)
        print(f"Starting Biomni ReAct HTTP MCP server on {host}:{port}...", file=sys.stderr)
        
        # Run with SSE transport (HTTP-based) for MCP client compatibility
        mcp_server.run(transport="sse")
        
    except Exception as e:
        print(f"Failed to start Biomni ReAct MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
