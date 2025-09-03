#!/usr/bin/env python3
"""
Unified HTTP MCP Server exposing both Biomni A1 and ReAct agents with intelligent routing.
Stateless server that receives conversation history from clients.
"""
import sys
import os
import re

# Configure matplotlib to use non-GUI backend before any imports that might use it
# This prevents "NSWindow should only be instantiated on the main thread!" errors on macOS
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

from typing import Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from biomni.agent.a1 import A1
from biomni.agent.react import react
from typing import Generator, Any
import uuid



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
            file_contexts: Optional[List[str]] = None,
            files: Optional[List[Dict[str, str]]] = None,
            stream_execution: bool = False
        ) -> dict:
            """
            Unified biomedical analysis with intelligent agent routing.
            
            Args:
                query: The biomedical question or analysis request
                conversation_history: Optional list of previous conversation messages
                file_contexts: Optional list of file contents to include in analysis
                stream_execution: If True, return execution steps as they happen (future enhancement)
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
                
                # Handle raw files (bytes_b64) by writing to temp files and passing paths
                temp_paths: list[str] = []
                if files:
                    import base64
                    import tempfile
                    for i, f in enumerate(files):
                        try:
                            name = f.get("name") or f"uploaded_file_{i}"
                            data_b64 = f.get("bytes_b64") or ""
                            raw = base64.b64decode(data_b64)
                            fd, tmp_path = tempfile.mkstemp(prefix="biomni_", suffix=f"_{name}")
                            with os.fdopen(fd, "wb") as fh:
                                fh.write(raw)
                            temp_paths.append(tmp_path)
                        except Exception:
                            continue

                    if temp_paths:
                        path_list = "\n".join(temp_paths)
                        enhanced_query = f"{enhanced_query}\n\nUploaded files (local paths):\n{path_list}"
                
                # Classify query and route to appropriate agent
                agent_type = QueryRouter.classify_query(query)
                
                print(f"Routing query to {agent_type} agent", file=sys.stderr)
                
                if agent_type == "a1":
                    execution_log, final_response = a1_agent.go(enhanced_query)
                else:
                    execution_log, final_response = react_agent.go(enhanced_query)
                
                try:
                    # Best-effort cleanup
                    for p in temp_paths:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                finally:
                    pass

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
        

        # In-memory stream registry for incremental step streaming
        streams: Dict[str, Dict[str, Any]] = {}

        @mcp_server.tool()
        def analyze_biomedical_query_streaming(
            query: str, 
            conversation_history: Optional[List[Dict[str, str]]] = None,
            file_contexts: Optional[List[str]] = None,
            files: Optional[List[Dict[str, str]]] = None,
        ) -> dict:
            """
            Streaming biomedical analysis that yields execution steps as they happen.
            
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
                
                # Handle raw files (bytes_b64) by writing to temp files and passing paths
                temp_paths: list[str] = []
                if files:
                    import base64
                    import tempfile
                    for i, f in enumerate(files):
                        try:
                            name = f.get("name") or f"uploaded_file_{i}"
                            data_b64 = f.get("bytes_b64") or ""
                            raw = base64.b64decode(data_b64)
                            fd, tmp_path = tempfile.mkstemp(prefix="biomni_", suffix=f"_{name}")
                            with os.fdopen(fd, "wb") as fh:
                                fh.write(raw)
                            temp_paths.append(tmp_path)
                        except Exception:
                            continue

                    if temp_paths:
                        path_list = "\n".join(temp_paths)
                        enhanced_query = f"{enhanced_query}\n\nUploaded files (local paths):\n{path_list}"
                
                # Classify query and route to appropriate agent
                agent_type = QueryRouter.classify_query(query)
                
                print(f"Routing streaming query to {agent_type} agent", file=sys.stderr)
                
                # Create a custom execution wrapper that can stream steps
                execution_steps = []
                
                if agent_type == "a1":
                    execution_log, final_response = a1_agent.go(enhanced_query)
                else:
                    execution_log, final_response = react_agent.go(enhanced_query)
                
                # For now, return the same format but mark it as streaming-ready
                try:
                    for p in temp_paths:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                finally:
                    pass

                return {
                    "success": True,
                    "response": final_response,
                    "execution_log": [str(step) for step in execution_log],
                    "agent_type": agent_type,
                    "streaming": True,
                }
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error in streaming analysis: {error_msg}", file=sys.stderr)
                
                return {
                    "success": False, 
                    "error": error_msg,
                    "streaming": True,
                }

        @mcp_server.tool()
        def start_biomni_stream(
            query: str,
            conversation_history: Optional[List[Dict[str, str]]] = None,
            file_contexts: Optional[List[str]] = None,
            files: Optional[List[Dict[str, str]]] = None,
        ) -> dict:
            """
            Start a streaming Biomni analysis session and return the first step.
            Returns a stream_id to be used with next_biomni_stream.
            """
            try:
                # Build enhanced query context
                enhanced_query = query
                if conversation_history:
                    recent_history = conversation_history[-6:]
                    history_context = "\n".join(
                        f"{turn['role'].title()}: {turn['content']}" for turn in recent_history
                    )
                    enhanced_query = f"Previous conversation:\n{history_context}\n\nCurrent query: {query}"

                temp_paths: list[str] = []
                if files:
                    import base64
                    import tempfile
                    for i, f in enumerate(files):
                        try:
                            name = f.get("name") or f"uploaded_file_{i}"
                            data_b64 = f.get("bytes_b64") or ""
                            raw = base64.b64decode(data_b64)
                            fd, tmp_path = tempfile.mkstemp(prefix="biomni_", suffix=f"_{name}")
                            with os.fdopen(fd, "wb") as fh:
                                fh.write(raw)
                            temp_paths.append(tmp_path)
                        except Exception:
                            continue

                    if temp_paths:
                        path_list = "\n".join(temp_paths)
                        enhanced_query = f"{enhanced_query}\n\nUploaded files (local paths):\n{path_list}"

                agent_type = QueryRouter.classify_query(query)
                print(f"Starting stream routed to {agent_type} agent", file=sys.stderr)

                # Build the generator for streaming steps
                if agent_type == "a1":
                    step_iter: Generator[dict, None, None] = a1_agent.go_stream(enhanced_query)
                else:
                    # Fallback: non-streaming agent -> wrap into a single-step generator
                    exec_log, final_response = react_agent.go(enhanced_query)
                    def singleton() -> Generator[dict, None, None]:
                        for entry in exec_log:
                            yield {"output": str(entry)}
                        return
                    step_iter = singleton()

                stream_id = str(uuid.uuid4())
                streams[stream_id] = {
                    "agent_type": agent_type,
                    "iterator": step_iter,
                    "final_response": None,
                    "temp_paths": temp_paths,
                }

                # Get first step if available
                try:
                    first = next(step_iter)
                    first_text = str(first.get("output", "")).strip()
                except StopIteration:
                    first_text = ""

                return {
                    "success": True,
                    "stream_id": stream_id,
                    "agent_type": agent_type,
                    "step": first_text,
                    "done": first_text == "",
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        @mcp_server.tool()
        def next_biomni_stream(stream_id: str) -> dict:
            """
            Advance a streaming Biomni session and return the next step.
            When done, includes a final_response if available and cleans up.
            """
            try:
                s = streams.get(stream_id)
                if not s:
                    return {"success": False, "error": "Invalid stream_id"}

                step_iter: Generator[dict, None, None] = s["iterator"]
                try:
                    nxt = next(step_iter)
                    return {
                        "success": True,
                        "done": False,
                        "step": str(nxt.get("output", "")).strip(),
                    }
                except StopIteration:
                    # Best-effort: obtain final output from accumulated log
                    final_text = ""
                    try:
                        if s["agent_type"] == "a1":
                            # A1 logs are accumulated inside the agent; reuse last known message
                            pass
                    except Exception:
                        final_text = ""
                    # Cleanup temp files created for this stream
                    try:
                        s_local = streams.get(stream_id) or {}
                        for p in s_local.get("temp_paths", []) or []:
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                    finally:
                        streams.pop(stream_id, None)
                    return {
                        "success": True,
                        "done": True,
                        "final_response": final_text,
                    }
            except Exception as e:
                return {"success": False, "error": str(e)}

        
        print(f"Created unified MCP server with 3 tools", file=sys.stderr)
        print(f"- health_check: Server status and configuration", file=sys.stderr)
        print(f"- analyze_biomedical_query: Unified analysis with intelligent routing", file=sys.stderr)
        print(f"- analyze_biomedical_query_streaming: Streaming analysis with real-time execution steps", file=sys.stderr)
        print(f"Starting unified Biomni HTTP MCP server on {host}:{port}...", file=sys.stderr)
        
        # Run with SSE transport for MCP client compatibility
        mcp_server.run(transport="sse")
        
    except Exception as e:
        print(f"Failed to start unified Biomni MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
