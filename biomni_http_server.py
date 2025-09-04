#!/usr/bin/env python3
"""
Unified HTTP MCP Server exposing both Biomni A1 and ReAct agents with intelligent routing.
Stateless server that receives conversation history from clients.

This is a significantly simplified and modularized version of the original server.
All complex logic has been extracted into logical modules in the collate package.

Environment Variables:
    BIOMNI_HOST: Server bind host (default: 0.0.0.0)
    BIOMNI_PORT: Server bind port (default: 3900)
    BIOMNI_DATA_PATH: Path to biomni data directory (default: ./biomni_data)
    BIOMNI_LLM_MODEL: LLM model to use (default: claude-3-5-sonnet-20241022)
    BIOMNI_A1_TIMEOUT: A1 agent timeout in seconds (default: 600)
    BIOMNI_REACT_TIMEOUT: ReAct agent timeout in seconds (default: 300)
    BIOMNI_DEFAULT_AGENT: Default agent for routing (a1|react, default: a1)
    BIOMNI_ENABLE_REACT: Enable ReAct agent (true|false, default: true)
"""
import sys
import os

# Configure matplotlib to use non-GUI backend before any imports that might use it
# This prevents "NSWindow should only be instantiated on the main thread!" errors on macOS
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

from mcp.server.fastmcp import FastMCP
from collate.biomni_server import (
    BiomniConfig,
    AgentManager,
    QueryRouter,
    StreamingManager,
    RequestHandlers,
)


def main():
    """Initialize and run the unified Biomni HTTP MCP server."""
    try:
        print("Initializing unified Biomni server...", file=sys.stderr)
        
        # Load and validate configuration
        config = BiomniConfig.from_environment()
        config.validate()
        
        # Initialize components
        agent_manager = AgentManager(config)
        agent_manager.initialize_agents()
        
        query_router = QueryRouter(config.default_agent, config.enable_react)
        streaming_manager = StreamingManager()
        
        request_handlers = RequestHandlers(
            config, agent_manager, query_router, streaming_manager
        )
        
        # Create MCP server
        print("Creating unified MCP server...", file=sys.stderr)
        mcp_server = FastMCP(
            name="BiomniUnified",
            host=config.host,
            port=config.port
        )
        
        # Register tools
        @mcp_server.tool()
        def health_check() -> dict:
            """Health check endpoint to verify server is running."""
            return request_handlers.health_check()
        
        @mcp_server.tool()
        def analyze_biomedical_query(
            query: str,
            conversation_history=None,
            file_contexts=None,
            files=None,
            stream: bool = False,
            stream_id=None,
        ) -> dict:
            """Unified biomedical analysis with intelligent agent routing."""
            return request_handlers.analyze_biomedical_query(
                query=query,
                conversation_history=conversation_history,
                file_contexts=file_contexts,
                files=files,
                stream=stream,
                stream_id=stream_id,
            )
        
        print(f"Created unified MCP server with 2 tools", file=sys.stderr)
        print(f"- health_check: Server status and configuration", file=sys.stderr)
        print(f"- analyze_biomedical_query: Unified analysis with optional streaming", file=sys.stderr)
        print(f"Starting unified Biomni HTTP MCP server on {config.host}:{config.port}...", file=sys.stderr)
        
        # Run with SSE transport for MCP client compatibility
        mcp_server.run(transport="sse")
        
    except Exception as e:
        print(f"Failed to start unified Biomni MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
