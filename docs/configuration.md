# Biomni Configuration Guide

## Quick Start

**Recommended approach**: Use environment variables or modify `default_config` for consistent behavior across your entire application.

```python
from biomni.config import default_config
from biomni.agent import A1

# Option 1: Modify global defaults (affects everything)
default_config.llm = "gpt-4"
default_config.timeout_seconds = 1200

# Option 2: Use environment variables (set in .env file)
# BIOMNI_LLM=gpt-4
# BIOMNI_TIMEOUT_SECONDS=1200

agent = A1()  # Uses your configuration
```

## Configuration Methods

### 1. Environment Variables (Recommended for Production)

Create a `.env` file in your project:

```bash
# Required API Keys (at least one)
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key

# Optional Settings
BIOMNI_LLM=claude-3-5-sonnet-20241022
BIOMNI_TIMEOUT_SECONDS=1200
BIOMNI_PATH=/path/to/data
```

### 2. Runtime Configuration (Recommended for Scripts)

```python
from biomni.config import default_config

# Changes apply to all agents and database queries
default_config.llm = "gpt-4"
default_config.timeout_seconds = 1200
```

### 3. Direct Parameters (Use with Caution)

```python
# ⚠️ Only affects this agent's reasoning, NOT database queries
agent = A1(llm="claude-3-5-sonnet-20241022")
```

## Common Examples

### Using Different Models

```python
# Use GPT-4 everywhere
default_config.llm = "gpt-4"
agent = A1()
```

### Cost Optimization (Different Models for Agent vs Database)

```python
# Cheaper model for database queries
default_config.llm = "claude-3-5-haiku-20241022"

# More powerful model for agent reasoning
agent = A1(llm="claude-3-5-sonnet-20241022")
```

### Custom/Local Models

```python
default_config.source = "Custom"
default_config.base_url = "http://localhost:8000/v1"
default_config.api_key = "local_key"
default_config.llm = "local-llama-70b"
```

## All Available Settings

### Environment Variables

```bash
# API Keys
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
AWS_BEARER_TOKEN_BEDROCK=your_key
AWS_REGION=us-east-1

# Azure OpenAI
OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Biomni Settings
BIOMNI_PATH=/path/to/data                   # Default: ./data
BIOMNI_TIMEOUT_SECONDS=1200                 # Default: 600
BIOMNI_LLM=model_name                        # Default: claude-sonnet-4-20250514
BIOMNI_TEMPERATURE=0.7                      # Default: 0.7
BIOMNI_USE_TOOL_RETRIEVER=true             # Default: true
BIOMNI_SOURCE=Anthropic                     # Auto-detected if not set
BIOMNI_CUSTOM_BASE_URL=http://localhost:8000/v1
BIOMNI_CUSTOM_API_KEY=custom_key
```

### Python Configuration

```python
from biomni.config import default_config

# All available settings
default_config.path = "./data"
default_config.timeout_seconds = 600
default_config.llm = "claude-sonnet-4-20250514"
default_config.temperature = 0.7
default_config.use_tool_retriever = True
default_config.source = None  # Auto-detected
default_config.base_url = None  # For custom models
default_config.api_key = None  # For custom models
```

## Important Notes

- **For pip-installed packages**: You can't edit the package files, but you can still use environment variables or modify `default_config` at runtime
- **Configuration consistency**: Database queries always use `default_config`, regardless of agent parameters
- **Priority order**: Direct params > Runtime config > Env vars > Defaults

## Troubleshooting

**API Key Not Found**:
- Check `.env` file exists in your working directory
- Verify with: `echo $ANTHROPIC_API_KEY`

**Configuration Not Applied**:
- Changes to `default_config` only affect agents created after the change
- Direct parameters only affect that specific agent, not database queries

**Model Not Found**:
- Check spelling of model name
- For Azure, prefix with "azure-" (e.g., "azure-gpt-4o")
- Ensure you have the right API key for that provider
