"""Streaming state and event management."""

import uuid
import shutil
from typing import Dict, List, Any, Generator, Optional
from ..files.handler import FileHandler
from collate.message_parsing import parse_blocks


class StreamingManager:
    """Manages streaming state and block-based event processing."""
    
    def __init__(self):
        # In-memory stream registry for incremental step streaming
        self.streams: Dict[str, Dict[str, Any]] = {}
    
    def create_stream(
        self, 
        step_iterator: Generator[dict, None, None],
        agent_type: str,
        temp_paths: List[str],
        artifact_dir: str
    ) -> str:
        """
        Create a new streaming session.
        
        Args:
            step_iterator: Generator for streaming steps
            agent_type: Type of agent being used
            temp_paths: Temporary file paths for cleanup
            artifact_dir: Directory for artifacts
            
        Returns:
            Stream ID for the new session
        """
        stream_id = str(uuid.uuid4())
        self.streams[stream_id] = {
            "agent_type": agent_type,
            "iterator": step_iterator,
            "final_response": None,
            "temp_paths": temp_paths,
            "artifact_dir": artifact_dir,
            "events": [],
        }
        return stream_id
    
    def get_first_event(self, stream_id: str) -> Dict[str, Any]:
        """Get the first event from a stream."""
        stream_data = self.streams.get(stream_id)
        if not stream_data:
            return {"success": False, "error": "Invalid stream_id"}
        
        step_iter = stream_data["iterator"]
        
        try:
            first = next(step_iter)
            first_text = str(first.get("output", "")).strip()
            
            # Pre-populate events from the first output
            new_events = self._text_to_events(first_text)
            stream_data["events"] = new_events[1:] if len(new_events) > 1 else []
            first_event = new_events[0] if new_events else None
            
            return {
                "success": True,
                "stream_id": stream_id,
                "agent_type": stream_data["agent_type"],
                "done": first_text == "",
                "block": first_event,
                "step": first_event.get("text", "") if first_event and first_event.get("event") == "delta" else first_text,
            }
        except StopIteration:
            return {
                "success": True,
                "stream_id": stream_id,
                "agent_type": stream_data["agent_type"],
                "done": True,
                "step": "",
            }
    
    def advance_stream(self, stream_id: str) -> Dict[str, Any]:
        """Advance an existing stream to the next event."""
        stream_data = self.streams.get(stream_id)
        if not stream_data:
            return {"success": False, "error": "Invalid stream_id"}
        
        # If we have queued block events from prior outputs, serve those first
        events = stream_data.get("events") or []
        if events:
            event = events.pop(0)
            stream_data["events"] = events
            return {
                "success": True,
                "done": False,
                "block": event,
                "step": event.get("text", "") if event.get("event") == "delta" else "",
            }
        
        # Get next step from iterator
        step_iter = stream_data["iterator"]
        try:
            next_step = next(step_iter)
            output_text = str(next_step.get("output", "")).strip()
            
            # Convert output_text into block events
            new_events = self._text_to_events(output_text)
            stream_data["events"] = new_events[1:] if len(new_events) > 1 else []
            first_event = new_events[0] if new_events else {
                "event": "delta", 
                "block_id": str(uuid.uuid4()), 
                "block_type": "think", 
                "text": ""
            }
            
            return {
                "success": True,
                "done": False,
                "block": first_event,
                "step": first_event.get("text", "") if first_event.get("event") == "delta" else "",
            }
        except StopIteration:
            return self._finalize_stream(stream_id)
    
    def _finalize_stream(self, stream_id: str) -> Dict[str, Any]:
        """Finalize a stream and package generated files."""
        stream_data = self.streams.get(stream_id, {})
        artifact_dir = stream_data.get("artifact_dir", "./")
        temp_paths = stream_data.get("temp_paths", [])
        
        # Package any generated files
        generated_files, artifact_paths = FileHandler.package_generated_files(
            artifact_dir, 
            exclude_paths=temp_paths
        )
        
        # Cleanup artifact directory
        try:
            shutil.rmtree(artifact_dir, ignore_errors=True)
        except Exception:
            pass
        finally:
            self.streams.pop(stream_id, None)
        
        return {
            "success": True,
            "done": True,
            "final_response": "",
            "generated_files": generated_files,
            "generated_images": [
                f for f in generated_files 
                if str(f.get("mime_type", "")).startswith("image/")
            ],
        }
    
    def _text_to_events(self, text: str) -> List[Dict[str, Any]]:
        """Convert output text into block events."""
        if not text:
            return []
        
        events: List[Dict[str, Any]] = []
        blocks = parse_blocks(text) or []
        
        for block in blocks:
            block_id = str(uuid.uuid4())
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            block_text = getattr(block, "text", None) or (
                block.get("text") if isinstance(block, dict) else ""
            )
            
            if not block_type:
                continue
            
            events.append({
                "event": "block_start", 
                "block_id": block_id, 
                "block_type": block_type
            })
            if block_text:
                events.append({
                    "event": "delta", 
                    "block_id": block_id, 
                    "block_type": block_type, 
                    "text": block_text
                })
            events.append({
                "event": "block_end", 
                "block_id": block_id, 
                "block_type": block_type
            })
        
        # If no blocks were found, fallback to a single delta event
        if not events and text:
            block_id = str(uuid.uuid4())
            events = [
                {"event": "block_start", "block_id": block_id, "block_type": "think"},
                {"event": "delta", "block_id": block_id, "block_type": "think", "text": text},
                {"event": "block_end", "block_id": block_id, "block_type": "think"},
            ]
        
        return events
