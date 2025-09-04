"""HTTP request handlers for Biomni server."""

import sys
import tempfile
import shutil
from typing import Dict, List, Optional, Generator, Any
from ..config import BiomniConfig
from ..agents.manager import AgentManager
from ..agents.router import QueryRouter
from ..files.handler import FileHandler
from ..streaming.manager import StreamingManager


class RequestHandlers:
    """Handles HTTP requests for the Biomni server."""
    
    def __init__(
        self, 
        config: BiomniConfig,
        agent_manager: AgentManager,
        query_router: QueryRouter,
        streaming_manager: StreamingManager
    ):
        self.config = config
        self.agent_manager = agent_manager
        self.query_router = query_router
        self.streaming_manager = streaming_manager
    
    def health_check(self) -> dict:
        """Health check endpoint to verify server is running."""
        return {
            "status": "healthy",
            "server": "BiomniUnified",
            "port": self.config.port,
            "host": self.config.host,
            "agents": self.agent_manager.get_available_agents(),
            "default_agent": self.config.default_agent,
            "data_path": self.config.data_path,
            "llm_model": self.config.llm_model,
        }
    
    def analyze_biomedical_query(
        self,
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
                return self._handle_streaming_request(query, conversation_history, files, stream_id)
            else:
                return self._handle_non_streaming_request(query, conversation_history, files)
        except Exception as e:
            error_msg = str(e)
            print(f"Error in analyze_biomedical_query: {error_msg}", file=sys.stderr)
            return {"success": False, "error": error_msg}
    
    def _handle_streaming_request(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]],
        files: Optional[List[Dict[str, str]]],
        stream_id: Optional[str]
    ) -> dict:
        """Handle streaming requests."""
        # Advance existing stream if stream_id provided
        if stream_id:
            return self.streaming_manager.advance_stream(stream_id)
        
        # Start a new streaming session
        artifact_dir = tempfile.mkdtemp(prefix="biomni_artifacts_")
        enhanced_query, temp_paths = FileHandler.build_enhanced_query(
            query, conversation_history, files, artifact_dir
        )
        agent_type = self.query_router.classify_query(query)
        print(f"Starting unified stream routed to {agent_type} agent", file=sys.stderr)
        
        # Build the generator for streaming steps
        step_iter = self._create_step_iterator(agent_type, enhanced_query)
        
        # Create stream and get first event
        stream_id = self.streaming_manager.create_stream(
            step_iter, agent_type, temp_paths, artifact_dir
        )
        
        return self.streaming_manager.get_first_event(stream_id)
    
    def _handle_non_streaming_request(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]],
        files: Optional[List[Dict[str, str]]]
    ) -> dict:
        """Handle non-streaming requests."""
        artifact_dir = tempfile.mkdtemp(prefix="biomni_artifacts_")
        enhanced_query, temp_paths = FileHandler.build_enhanced_query(
            query, conversation_history, files, artifact_dir
        )
        agent_type = self.query_router.classify_query(query)
        print(f"Routing unified query to {agent_type} agent", file=sys.stderr)
        
        # Execute query with selected agent
        execution_log, final_response = self._execute_query(agent_type, enhanced_query)
        
        # Package generated files
        generated_files, artifact_paths = FileHandler.package_generated_files(
            artifact_dir, exclude_paths=temp_paths
        )
        
        # Cleanup
        try:
            shutil.rmtree(artifact_dir, ignore_errors=True)
        except Exception:
            pass
        
        return {
            "success": True,
            "response": final_response,
            "execution_log": [str(step) for step in execution_log],
            "agent_type": agent_type,
            "generated_files": generated_files,
            "generated_images": [
                f for f in generated_files 
                if str(f.get("mime_type", "")).startswith("image/")
            ],
        }
    
    def _create_step_iterator(self, agent_type: str, query: str) -> Generator[dict, None, None]:
        """Create appropriate step iterator for the given agent type."""
        if agent_type == "a1":
            a1_agent = self.agent_manager.get_agent_by_type("a1")
            return a1_agent.go_stream(query)
        else:
            # Use ReAct agent if available, otherwise fallback to A1
            react_agent = self.agent_manager.get_agent_by_type("react")
            if react_agent is not None:
                exec_log, _ = react_agent.go(query)
                def singleton() -> Generator[dict, None, None]:
                    for entry in exec_log:
                        yield {"output": str(entry)}
                    return
                return singleton()
            else:
                # Fallback to A1 if ReAct is disabled
                a1_agent = self.agent_manager.get_agent_by_type("a1")
                return a1_agent.go_stream(query)
    
    def _execute_query(self, agent_type: str, query: str) -> tuple:
        """Execute query with the specified agent type."""
        if agent_type == "a1":
            a1_agent = self.agent_manager.get_agent_by_type("a1")
            return a1_agent.go(query)
        else:
            # Use ReAct agent if available, otherwise fallback to A1
            react_agent = self.agent_manager.get_agent_by_type("react")
            if react_agent is not None:
                return react_agent.go(query)
            else:
                # Fallback to A1 if ReAct is disabled
                a1_agent = self.agent_manager.get_agent_by_type("a1")
                return a1_agent.go(query)
