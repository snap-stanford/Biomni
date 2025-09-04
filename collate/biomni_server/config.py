"""Configuration management for Biomni HTTP server."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BiomniConfig:
    """Configuration settings for the Biomni HTTP server."""
    
    # Server settings
    host: str
    port: int
    
    # Data and model settings
    data_path: str
    llm_model: str
    
    # Agent settings
    a1_timeout: int
    react_timeout: int
    default_agent: str
    enable_react: bool
    
    @classmethod
    def from_environment(cls) -> "BiomniConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.environ.get("BIOMNI_HOST", "0.0.0.0"),
            port=int(os.environ.get("BIOMNI_PORT", "3900")),
            data_path=os.path.abspath(os.environ.get("BIOMNI_DATA_PATH", "./biomni_data")),
            llm_model=os.environ.get("BIOMNI_LLM_MODEL", "claude-3-5-sonnet-20241022"),
            a1_timeout=int(os.environ.get("BIOMNI_A1_TIMEOUT", "600")),
            react_timeout=int(os.environ.get("BIOMNI_REACT_TIMEOUT", "300")),
            default_agent=os.environ.get("BIOMNI_DEFAULT_AGENT", "a1").lower(),
            enable_react=os.environ.get("BIOMNI_ENABLE_REACT", "true").lower() in ("true", "1", "yes"),
        )
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not os.path.exists(self.data_path):
            print(f"Warning: biomni_data path not found at {self.data_path}")
        
        if self.default_agent not in ("a1", "react"):
            raise ValueError(f"Invalid default_agent: {self.default_agent}. Must be 'a1' or 'react'")
        
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port}. Must be between 1 and 65535")
