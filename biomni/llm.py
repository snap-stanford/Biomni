import os
from typing import Literal, Optional

import openai
from langchain_anthropic import ChatAnthropic

# from langchain_aws import ChatBedrock
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_aws import ChatBedrock
from langchain_upstage import ChatUpstage
from langchain_xai import ChatXAI

SourceType = Literal[
    "OpenAI",
    "AzureOpenAI",
    "Anthropic",
    "Ollama",
    "Gemini",
    "Bedrock",
    "Custom",
    "MISTRAL",
    "Upstage",
    "XAI",
]


def get_llm(
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.7,
    stop_sequences: list[str] | None = None,
    source: SourceType | None = None,
    base_url: str | None = None,
    api_key: str = "EMPTY",
) -> BaseChatModel:
    """
    Get a language model instance based on the specified model name and source.
    This function supports models from OpenAI, Azure OpenAI, Anthropic, Ollama, Gemini, Bedrock, and custom model serving.
    Args:
        model (str): The model name to use
        temperature (float): Temperature setting for generation
        stop_sequences (list): Sequences that will stop generation
        source (str): Source provider: "OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", or "Custom"
                      If None, will attempt to auto-detect from model name
        base_url (str): The base URL for custom model serving (e.g., "http://localhost:8000/v1"), default is None
        api_key (str): The API key for the custom llm
    """
    # Auto-detect source from model name if not specified
    if source is None:
        if model[:7] == "claude-":
            source = "Anthropic"
        elif model[:4] == "gpt-":
            source = "OpenAI"
        elif model[:7] == "gemini-":
            source = "Gemini"
        elif base_url is not None:
            source = "Custom"
        elif "/" in model or any(
            name in model.lower()
            for name in [
                "llama",
                "qwen",
                "gemma",
                "phi",
                "dolphin",
                "orca",
                "vicuna",
            ]
        ):
            source = "Ollama"
        elif model.startswith(
            (
                "us.anthropic.claude-",
                "amazon.titan-",
                "meta.llama-",
                "mistral.",
                "cohere.",
                "ai21.",
                "us.",
            )
        ):
            source = "Bedrock"
        elif model.startswith(("mistral-")):
            source = "MISTRAL"
        elif model.startswith(("solar-")):
            source = "Upstage"
        elif model.startswith(("grok-")):
            source = "XAI"
        else:
            raise ValueError(
                "Unable to determine model source. Please specify 'source' parameter."
            )
    # Create appropriate model based on source
    if source == "OpenAI":
        return ChatOpenAI(
            model=model, temperature=temperature, stop_sequences=stop_sequences
        )
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
    elif source == "Gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
        )
    elif source == "Ollama":
        return ChatOllama(
            model=model,
            temperature=temperature,
        )
    elif source == "MISTRAL":
        return ChatMistralAI(
            model=model,
            temperature=temperature,
        )
    elif source == "Bedrock":
        return ChatBedrock(
            model=model,
            temperature=temperature,
            model_kwargs={"stop_sequences": stop_sequences, "max_tokens": 8192 * 4},
        )
    elif source == "Upstage":
        return ChatUpstage(
            model=model,
            temperature=temperature,
        )
    elif source == "Custom":
        # Custom LLM serving such as SGLang. Must expose an openai compatible API.
        assert (
            base_url is not None
        ), "base_url must be provided for customly served LLMs"
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
            base_url=base_url,
            api_key=api_key,
        )
        return llm
    elif source == "XAI":
        print("XAI is selected")
        return ChatXAI(
            model=model,
            temperature=temperature,
            # stop_sequences=stop_sequences,
            # model_kwargs={"stop_sequences": stop_sequences},
        )
    else:
        raise ValueError(
            f"Invalid source: {source}. Valid options are 'OpenAI', 'AzureOpenAI', 'Anthropic', 'Gemini', 'Bedrock', or 'Ollama'"
        )
