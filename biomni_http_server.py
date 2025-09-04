#!/usr/bin/env python3
"""
Unified HTTP MCP Server exposing both Biomni A1 and ReAct agents with intelligent routing.
Stateless server that receives conversation history from clients.
"""
import sys
import os
import re
import time

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
        
        # In-memory stream registry for incremental step streaming
        streams: Dict[str, Dict[str, Any]] = {}

        def _build_enhanced_query(query: str, conversation_history, files) -> tuple[str, list[str]]:
            # Build context with conversation history
            enhanced_query = query
            if conversation_history:
                # Limit to recent conversation to avoid token overflow
                recent_history = conversation_history[-6:]  # Last 6 messages (3 turns)
                history_context = "\n".join([f"{turn['role'].title()}: {turn['content']}" for turn in recent_history])
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

            return enhanced_query, temp_paths

        def _collect_generated_images(working_dir: str = "./") -> tuple[list[dict[str, str]], list[str]]:
            """Collect generated image files and return as base64 encoded data with file paths for cleanup."""
            import base64
            import glob
            import mimetypes
            
            image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.svg", "*.pdf", "*.html"]
            generated_images = []
            image_paths = []
            
            for ext in image_extensions:
                for img_path in glob.glob(os.path.join(working_dir, ext)):
                    try:
                        # Only collect files that were recently created (within last minute)
                        if os.path.getmtime(img_path) > (time.time() - 60):
                            with open(img_path, "rb") as f:
                                img_bytes = f.read()
                            
                            mime_type, _ = mimetypes.guess_type(img_path)
                            if not mime_type:
                                mime_type = "application/octet-stream"
                            
                            generated_images.append({
                                "name": os.path.basename(img_path),
                                "mime_type": mime_type,
                                "bytes_b64": base64.b64encode(img_bytes).decode("utf-8"),
                                "size": len(img_bytes)
                            })
                            image_paths.append(img_path)
                    except Exception:
                        continue
            
            return generated_images, image_paths

        @mcp_server.tool()
        def analyze_biomedical_query(
            query: str,
            conversation_history: Optional[List[Dict[str, str]]] = None,
            file_contexts: Optional[List[str]] = None,
            files: Optional[List[Dict[str, str]]] = None,
            stream: bool = False,
            stream_id: Optional[str] = None,
        ) -> dict:
            """
            Unified biomedical analysis with intelligent agent routing.

            Args:
                query: The biomedical question or analysis request
                conversation_history: Optional list of previous conversation messages
                file_contexts: Optional list of file contents to include in analysis (unused; reserved)
                files: Optional list of raw files as {name, bytes_b64}
                stream: If True, start/continue streaming steps
                stream_id: If provided with stream=True, advances an existing stream
            """
            try:
                if stream:
                    # Advance existing stream if stream_id provided
                    if stream_id:
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
                                # Collect any generated images before cleanup
                                generated_images, image_paths = _collect_generated_images()
                                
                                # Cleanup temp files and image files
                                try:
                                    s_local = streams.get(stream_id) or {}
                                    # Clean up temp input files
                                    for p in s_local.get("temp_paths", []) or []:
                                        try:
                                            os.remove(p)
                                        except Exception:
                                            pass
                                    # Clean up generated image files after sending
                                    for img_path in image_paths:
                                        try:
                                            os.remove(img_path)
                                        except Exception:
                                            pass
                                finally:
                                    streams.pop(stream_id, None)
                                return {
                                    "success": True,
                                    "done": True,
                                    "final_response": "",
                                    "generated_images": generated_images,
                                }
                        except Exception as e:
                            return {"success": False, "error": str(e)}

                    # Start a new streaming session
                    enhanced_query, temp_paths = _build_enhanced_query(query, conversation_history, files)
                    agent_type = QueryRouter.classify_query(query)
                    print(f"Starting unified stream routed to {agent_type} agent", file=sys.stderr)

                    # Build the generator for streaming steps
                    if agent_type == "a1":
                        step_iter: Generator[dict, None, None] = a1_agent.go_stream(enhanced_query)
                    else:
                        # Fallback: wrap non-streaming logs into a generator
                        exec_log, _ = react_agent.go(enhanced_query)
                        def singleton() -> Generator[dict, None, None]:
                            for entry in exec_log:
                                yield {"output": str(entry)}
                            return
                        step_iter = singleton()

                    sid = str(uuid.uuid4())
                    streams[sid] = {
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
                        "stream_id": sid,
                        "agent_type": agent_type,
                        "step": first_text,
                        "done": first_text == "",
                    }

                # Non-streaming path
                enhanced_query, temp_paths = _build_enhanced_query(query, conversation_history, files)
                agent_type = QueryRouter.classify_query(query)
                print(f"Routing unified query to {agent_type} agent", file=sys.stderr)

                if agent_type == "a1":
                    execution_log, final_response = a1_agent.go(enhanced_query)
                else:
                    execution_log, final_response = react_agent.go(enhanced_query)

                # Collect any generated images
                generated_images, image_paths = _collect_generated_images()

                try:
                    # Clean up temp input files
                    for p in temp_paths:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                    # Clean up generated image files after sending
                    for img_path in image_paths:
                        try:
                            os.remove(img_path)
                        except Exception:
                            pass
                finally:
                    pass

                return {
                    "success": True,
                    "response": final_response,
                    "execution_log": [str(step) for step in execution_log],
                    "agent_type": agent_type,
                    "generated_images": generated_images,
                }

            except Exception as e:
                error_msg = str(e)
                print(f"Error in analyze_biomedical_query: {error_msg}", file=sys.stderr)
                return {"success": False, "error": error_msg}

        
        print(f"Created unified MCP server with 2 tools", file=sys.stderr)
        print(f"- health_check: Server status and configuration", file=sys.stderr)
        print(f"- analyze_biomedical_query: Unified analysis with optional streaming", file=sys.stderr)
        print(f"Starting unified Biomni HTTP MCP server on {host}:{port}...", file=sys.stderr)
        
        # Run with SSE transport for MCP client compatibility
        mcp_server.run(transport="sse")
        
    except Exception as e:
        print(f"Failed to start unified Biomni MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
