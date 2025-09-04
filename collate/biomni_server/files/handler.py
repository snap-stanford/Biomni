"""File processing and artifact management."""

import os
import base64
import tempfile
import mimetypes
from typing import Optional, Dict, List, Tuple


class FileHandler:
    """Handles file processing, temporary files, and artifact management."""
    
    @staticmethod
    def build_enhanced_query(
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        files: Optional[List[Dict[str, str]]] = None,
        artifact_dir: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Build enhanced query with conversation history and file context.
        
        Args:
            query: The original query
            conversation_history: Optional conversation history
            files: Optional list of uploaded files
            artifact_dir: Optional directory for artifacts
            
        Returns:
            Tuple of (enhanced_query, temp_file_paths)
        """
        enhanced_query = query
        
        # Add conversation history context
        if conversation_history:
            # Limit to recent conversation to avoid token overflow
            recent_history = conversation_history[-6:]  # Last 6 messages (3 turns)
            history_context = "\n".join([
                f"{turn['role'].title()}: {turn['content']}" 
                for turn in recent_history
            ])
            enhanced_query = f"Previous conversation:\n{history_context}\n\nCurrent query: {query}"
        
        # Handle uploaded files
        temp_paths = FileHandler._process_uploaded_files(files, artifact_dir)
        if temp_paths:
            path_list = "\n".join(temp_paths)
            enhanced_query = f"{enhanced_query}\n\nUploaded files (local paths):\n{path_list}"
        
        # Add artifact directory info
        if artifact_dir:
            enhanced_query = f"{enhanced_query}\n\nArtifact output directory (write outputs here):\n{artifact_dir}"
        
        return enhanced_query, temp_paths
    
    @staticmethod
    def _process_uploaded_files(
        files: Optional[List[Dict[str, str]]], 
        artifact_dir: Optional[str]
    ) -> List[str]:
        """Process uploaded files and return list of temporary file paths."""
        temp_paths: List[str] = []
        
        if not files:
            return temp_paths
        
        for i, file_data in enumerate(files):
            try:
                # Sanitize filename
                name = file_data.get("name") or f"uploaded_file_{i}"
                name = os.path.basename(name)
                data_b64 = file_data.get("bytes_b64") or ""
                raw_data = base64.b64decode(data_b64)
                
                if artifact_dir:
                    # Write to artifact directory
                    try:
                        os.makedirs(artifact_dir, exist_ok=True)
                    except Exception:
                        pass
                    tmp_path = os.path.join(artifact_dir, name)
                    with open(tmp_path, "wb") as fh:
                        fh.write(raw_data)
                    temp_paths.append(tmp_path)
                else:
                    # Write to temporary file
                    fd, tmp_path = tempfile.mkstemp(prefix="biomni_", suffix=f"_{name}")
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(raw_data)
                    temp_paths.append(tmp_path)
            except Exception:
                # Skip problematic files
                continue
        
        return temp_paths
    
    @staticmethod
    def package_generated_files(
        artifact_dir: str, 
        exclude_paths: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """
        Package all files in artifact_dir (recursively), excluding uploaded inputs.
        
        Args:
            artifact_dir: Directory containing generated files
            exclude_paths: Paths to exclude from packaging
            
        Returns:
            Tuple of (generated_files_payload, file_paths_for_cleanup)
        """
        exclude_set = set(exclude_paths or [])
        generated_files: List[Dict[str, str]] = []
        file_paths: List[str] = []
        
        try:
            for root, _dirs, files in os.walk(artifact_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    if fpath in exclude_set:
                        continue
                    
                    try:
                        with open(fpath, "rb") as f:
                            file_bytes = f.read()
                        
                        mime_type, _ = mimetypes.guess_type(fpath)
                        if not mime_type:
                            mime_type = "application/octet-stream"
                        
                        generated_files.append({
                            "name": os.path.basename(fpath),
                            "mime_type": mime_type,
                            "bytes_b64": base64.b64encode(file_bytes).decode("utf-8"),
                            "size": len(file_bytes),
                        })
                        file_paths.append(fpath)
                    except Exception:
                        # Skip problematic files
                        continue
        except Exception:
            # Handle directory access issues gracefully
            pass
        
        return generated_files, file_paths
