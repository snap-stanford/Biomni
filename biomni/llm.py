import os
from typing import Literal, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

SourceType = Literal["OpenAI", "AzureOpenAI", "Anthropic", "Ollama"]


def get_llm(
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.7,
    stop_sequences: list[str] | None = None,
    source: SourceType | None = None,
) -> BaseChatModel:
    """
    Get a language model instance based on the specified model name and source.
    This function supports models from OpenAI, Azure OpenAI, Anthropic, and Ollama.

    Args:
        model (str): The model name to use
        temperature (float): Temperature setting for generation
        stop_sequences (list): Sequences that will stop generation
        source (str): Source provider: "OpenAI", "AzureOpenAI", "Anthropic", or "Ollama"
                      If None, will attempt to auto-detect from model name
    """
    # Auto-detect source from model name if not specified
    if source is None:
        if model[:7] == "claude-":
            source = "Anthropic"
        elif model[:4] == "gpt-":
            source = "OpenAI"
        elif "/" in model or any(
            name in model.lower() for name in ["llama", "mistral", "qwen", "gemma", "phi", "dolphin", "orca", "vicuna"]
        ):
            source = "Ollama"
        else:
            raise ValueError("Unable to determine model source. Please specify 'source' parameter.")

    # Create appropriate model based on source
    if source == "OpenAI":
        return ChatOpenAI(model=model, temperature=temperature, stop_sequences=stop_sequences)
    elif source == "AzureOpenAI":
        API_VERSION = "2024-12-01-preview"
        return AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
            azure_deployment=model,
            openai_api_version=API_VERSION,
            temperature=temperature,
        )
    elif source == "Anthropic":
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
        )
    elif source == "Ollama":
        return ChatOllama(
            model=model,
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Invalid source: {source}. Valid options are 'OpenAI', 'AzureOpenAI', 'Anthropic', or 'Ollama'"
        )
