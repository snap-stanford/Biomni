"""
base_CAi Configuration Management

Simple configuration class for centralizing common settings.
Maintains full backward compatibility with existing code.
"""

import os
from dataclasses import dataclass


@dataclass
class Base_CAi_Config:
    """Central configuration for base_CAi agent.

    All settings are optional and have sensible defaults.
    API keys are still read from environment variables to maintain
    compatibility with existing .env file structure.

    Usage:
        # Create config with defaults
        config = base_CAiConfig()

        # Override specific settings
        config = base_CAiConfig(llm="gpt-4", timeout_seconds=1200)

        # Modify after creation
        config.path = "./custom_data"
    """

    # Data and execution settings
    path: str = "./data"
    timeout_seconds: int = 600

    # LLM settings (API keys still from environment)
    llm: str = "claude-sonnet-4-5"
    temperature: float = 0.7

    # Tool settings
    use_tool_retriever: bool = True

    # Data licensing settings
    commercial_mode: bool = False  # If True, excludes non-commercial datasets

    # Custom model settings (for custom LLM serving)
    base_url: str | None = None
    api_key: str | None = None  # Only for custom models, not provider API keys

    # LLM source (auto-detected if None)
    source: str | None = None

    # Third-party integrations
    protocols_io_access_token: str | None = None

    def __post_init__(self):
        """Load any environment variable overrides if they exist."""
        # Check for environment variable overrides (optional)
        # Support both old and new names for backwards compatibility
        if os.getenv("base_CAi_PATH") or os.getenv("base_CAi_DATA_PATH"):
            self.path = os.getenv("base_CAi_PATH") or os.getenv("base_CAi_DATA_PATH")
        if os.getenv("base_CAi_TIMEOUT_SECONDS"):
            self.timeout_seconds = int(os.getenv("base_CAi_TIMEOUT_SECONDS"))
        if os.getenv("base_CAi_LLM") or os.getenv("base_CAi_LLM_MODEL"):
            self.llm = os.getenv("base_CAi_LLM") or os.getenv("base_CAi_LLM_MODEL")
        if os.getenv("base_CAi_USE_TOOL_RETRIEVER"):
            self.use_tool_retriever = os.getenv("base_CAi_USE_TOOL_RETRIEVER").lower() == "true"
        if os.getenv("base_CAi_COMMERCIAL_MODE"):
            self.commercial_mode = os.getenv("base_CAi_COMMERCIAL_MODE").lower() == "true"
        if os.getenv("base_CAi_TEMPERATURE"):
            self.temperature = float(os.getenv("base_CAi_TEMPERATURE"))
        if os.getenv("base_CAi_CUSTOM_BASE_URL"):
            self.base_url = os.getenv("base_CAi_CUSTOM_BASE_URL")
        if os.getenv("base_CAi_CUSTOM_API_KEY"):
            self.api_key = os.getenv("base_CAi_CUSTOM_API_KEY")
        if os.getenv("base_CAi_SOURCE"):
            self.source = os.getenv("base_CAi_SOURCE")

        # Protocols.io access token (prefer specific env vars)
        env_token = os.getenv("PROTOCOLS_IO_ACCESS_TOKEN") or os.getenv("base_CAi_PROTOCOLS_IO_ACCESS_TOKEN")
        if env_token:
            self.protocols_io_access_token = env_token

    def to_dict(self) -> dict:
        """Convert config to dictionary for easy access."""
        return {
            "path": self.path,
            "timeout_seconds": self.timeout_seconds,
            "llm": self.llm,
            "temperature": self.temperature,
            "use_tool_retriever": self.use_tool_retriever,
            "commercial_mode": self.commercial_mode,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "source": self.source,
        }


# Global default config instance (optional, for convenience)
default_config = Base_CAi_Config()
