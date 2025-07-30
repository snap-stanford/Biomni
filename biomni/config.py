"""
Simple configuration management for Biomni using Pydantic BaseSettings.
Reads environment variables with fallbacks to default values.
"""

from typing import Optional

from pydantic_settings import BaseSettings


class BiomniConfig(BaseSettings):
    """Simple Biomni configuration using Pydantic BaseSettings."""

    # Data and execution settings
    data_path: str = "./data"
    timeout_seconds: int = 600

    # LLM settings
    llm_model: str = "claude-sonnet-4-20250514"

    # Tool settings
    use_tool_retriever: bool = True

    # Custom model settings
    base_url: str | None = None
    api_key: str | None = None

    class Config:
        env_prefix = "BIOMNI_"
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global config instance
config = BiomniConfig()
