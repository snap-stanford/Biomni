#!/usr/bin/env python3
"""
Unified HTTP MCP Server exposing both Biomni A1 and ReAct agents with intelligent routing.
Stateless server that receives conversation history from clients.
"""
import sys
import os
import re
from typing import Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from biomni.agent.a1 import A1
from biomni.agent.react import react



class QueryRouter:
    #TODO: Add a more sophisticated query router
    """Intelligent routing between A1 and ReAct agents."""
    
    @classmethod
    def classify_query(cls, query: str) -> str:
        """
        Classify query to determine which agent to use.
        
        Returns:
            "a1" for code execution needs, "react" for tool-based queries
        """
        return "a1"


def main():
    """Initialize and run the unified Biomni HTTP MCP server."""
    try:
        print("Initializing unified Biomni server...", file=sys.stderr)
        
        # Use absolute path to biomni_data to avoid path resolution issues
        biomni_data_path = os.path.abspath("./biomni_data")
        if not os.path.exists(biomni_data_path):
            print(f"Warning: biomni_data path not found at {biomni_data_path}", file=sys.stderr)
        
        # Initialize both agents
        print("Initializing A1 agent...", file=sys.stderr)
        a1_agent = A1(
            path=biomni_data_path,
            llm="claude-3-5-sonnet-20241022",
            use_tool_retriever=True,
            timeout_seconds=600,
        )
        
        print("Initializing ReAct agent...", file=sys.stderr)
        react_agent = react(
            path=biomni_data_path,
            llm="claude-3-5-sonnet-20241022", 
            use_tool_retriever=True,
            timeout_seconds=300,
        )
        
        # Configure ReAct agent
        react_agent.configure(
            plan=True,
            reflect=True,
            data_lake=True,
            library_access=True,
        )
        

        
        # Get server configuration
        port = int(os.environ.get("BIOMNI_PORT", "3900"))
        host = os.environ.get("BIOMNI_HOST", "127.0.0.1")
        
        # Create MCP server
        print("Creating unified MCP server...", file=sys.stderr)
        mcp_server = FastMCP(
            name="BiomniUnified",
            host=host,
            port=port
        )
        
        @mcp_server.tool()
        def health_check() -> dict:
            """Health check endpoint to verify server is running."""
            return {
                "status": "healthy", 
                "server": "BiomniUnified", 
                "port": port,
                "agents": ["a1", "react"],
                "data_path": biomni_data_path
            }
        
        @mcp_server.tool()
        def analyze_biomedical_query(
            query: str, 
            conversation_history: Optional[List[Dict[str, str]]] = None,
            file_contexts: Optional[List[str]] = None
        ) -> dict:
            """
            Unified biomedical analysis with intelligent agent routing.
            
            Args:
                query: The biomedical question or analysis request
                conversation_history: Optional list of previous conversation messages
                file_contexts: Optional list of file contents to include in analysis
            """
            try:
                # Build context with conversation history
                enhanced_query = query
                if conversation_history:
                    # Limit to recent conversation to avoid token overflow
                    recent_history = conversation_history[-6:]  # Last 6 messages (3 turns)
                    history_context = "\n".join([
                        f"{turn['role'].title()}: {turn['content']}" 
                        for turn in recent_history
                    ])
                    enhanced_query = f"Previous conversation:\n{history_context}\n\nCurrent query: {query}"
                
                # Add file contexts if provided
                if file_contexts:
                    context_text = "\n".join([
                        f"File {i + 1}:\n{content}\n" 
                        for i, content in enumerate(file_contexts)
                    ])
                    enhanced_query = f"{enhanced_query}\n\nProvided file contexts:\n{context_text}"
                
                # Classify query and route to appropriate agent
                agent_type = QueryRouter.classify_query(query)
                
                print(f"Routing query to {agent_type} agent", file=sys.stderr)
                
                if agent_type == "a1":
                    execution_log, final_response = a1_agent.go(enhanced_query)
                else:
                    execution_log, final_response = react_agent.go(enhanced_query)
                
                return {
                    "success": True,
                    "response": final_response,
                    "execution_log": [str(step) for step in execution_log],
                    "agent_type": agent_type,
                }
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error in analyze_biomedical_query: {error_msg}", file=sys.stderr)
                
                return {
                    "success": False, 
                    "error": error_msg,
                }
        

        
        print(f"Created unified MCP server with 2 tools", file=sys.stderr)
        print(f"- health_check: Server status and configuration", file=sys.stderr)
        print(f"- analyze_biomedical_query: Unified analysis with intelligent routing", file=sys.stderr)
        print(f"Starting unified Biomni HTTP MCP server on {host}:{port}...", file=sys.stderr)
        
        # Run with SSE transport for MCP client compatibility
        mcp_server.run(transport="sse")
        
    except Exception as e:
        print(f"Failed to start unified Biomni MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
