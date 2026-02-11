#!/usr/bin/env python3
import os
import sys

# Fix Windows console encoding for emojis and ensure output is visible immediately (no buffering)
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
else:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

from dotenv import load_dotenv

load_dotenv()

# Amplicon tool loads CCLE.csv from the agent data path (./data by default) or from
# BIOMNI_PATH / BIOMNI_DATA_PATH. Place CCLE.csv in that directory or set the env var.


def _required_api_key_env_for_llm(model: str, source: str | None) -> str | None:
    """Return the env var name required for the given model/source, or None if no key needed."""
    if source is None:
        if model.startswith("claude-"):
            source = "Anthropic"
        elif model.startswith("azure-"):
            source = "AzureOpenAI"
        elif model.startswith("gpt-") and "oss" not in model:
            source = "OpenAI"
        elif model.startswith("gemini-"):
            source = "Gemini"
        elif "groq" in model.lower():
            source = "Groq"
        elif model.startswith(
            ("anthropic.claude-", "amazon.titan-", "meta.llama-", "mistral.", "cohere.", "ai21.", "us.")
        ):
            source = "Bedrock"
        elif os.getenv("BIOMNI_CUSTOM_BASE_URL") or os.getenv("BIOMNI_CUSTOM_API_KEY"):
            source = "Custom"
        else:
            source = "Ollama"  # local, no key
    key_map = {
        "OpenAI": "OPENAI_API_KEY",
        "AzureOpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Gemini": "GEMINI_API_KEY",
        "Groq": "GROQ_API_KEY",
        "Bedrock": None,  # AWS credentials from env/IAM
        "Custom": "BIOMNI_CUSTOM_API_KEY",
        "Ollama": None,
    }
    return key_map.get(source)


# Verify API key for the configured LLM provider
_llm = os.getenv("BIOMNI_LLM") or os.getenv("BIOMNI_LLM_MODEL") or "gpt-5"
_source = (
    os.getenv("LLM_SOURCE")
    if os.getenv("LLM_SOURCE")
    in ("OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom")
    else None
)
_key_var = _required_api_key_env_for_llm(_llm, _source)
if _key_var:
    _key = os.getenv(_key_var)
    if not _key or not str(_key).strip():
        print(f"ERROR: Set {_key_var} in .env for the configured LLM ({_llm}).")
        sys.exit(1)

print(f"Using LLM: {_llm}")

# Install Gradio if needed using uv
try:
    import gradio as gr

    if int(gr.__version__.split(".")[0]) >= 6:
        import subprocess

        subprocess.check_call(["uv", "pip", "install", "-q", "--python", sys.executable, "gradio>=5.0,<6.0"])
except ImportError:
    import subprocess

    subprocess.check_call(["uv", "pip", "install", "-q", "--python", sys.executable, "gradio>=5.0,<6.0"])

# Initialize agent
from biomni.agent import A1

agent = A1(path="./data", llm=_llm, expected_data_lake_files=[])

print("Starting web interface at http://localhost:7860")
print("Press Ctrl+C to stop")

# Launch web UI
agent.launch_gradio_demo(share=False, server_name="0.0.0.0")
