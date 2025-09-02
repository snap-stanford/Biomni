#!/usr/bin/env python3
"""
MCP Server exposing Biomni ReAct agent capabilities.
Run this from the Biomni directory where the package is installed.
"""
import sys
from mcp.server.fastmcp import FastMCP
from biomni.agent.react import react


def main():
    """Initialize and run the Biomni ReAct MCP server."""
    try:
        # Initialize ReAct agent
        agent = react(
            path="./data",
            llm="claude-3-5-sonnet-20241022",
            use_tool_retriever=True,
            timeout_seconds=300
        )
        
        # Configure with controlled capabilities
        agent.configure(
            plan=True,
            reflect=True,
            data_lake=True,
            library_access=True
        )
        
        # Create MCP server
        mcp = FastMCP("BiomniReAct")
        
        @mcp.tool()
        def analyze_biomedical_query(query: str, file_contexts: list[str] = None) -> dict:
            """Analyze biomedical queries using curated tools (no arbitrary code)."""
            try:
                enhanced_query = query
                if file_contexts:
                    context_text = "\n".join([f"File {i+1}:\n{content}" for i, content in enumerate(file_contexts)])
                    enhanced_query = f"{query}\n\nFile contexts:\n{context_text}"
                
                execution_log, final_response = agent.go(enhanced_query)
                
                return {
                    "success": True,
                    "response": final_response,
                    "execution_log": [str(step) for step in execution_log]
                }
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        print("Biomni ReAct MCP server ready", file=sys.stderr)
        mcp.run(transport="stdio")
        
    except Exception as e:
        print(f"Failed to start Biomni ReAct MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
