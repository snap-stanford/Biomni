FROM condaforge/mambaforge:latest AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ENV_FILE selects the conda environment:
#   biomni_env/fixed_env.yml  — reduced (~13GB, no R/CLI tools)
#   biomni_env/bio_env.yml    — full (~30GB, includes R)
ARG ENV_FILE=biomni_env/fixed_env.yml
COPY ${ENV_FILE} /app/environment.yml
RUN mamba env create -f /app/environment.yml -n biomni && \
    mamba clean -afy

# Activate the environment by default
ENV PATH=/opt/conda/envs/biomni/bin:$PATH
ENV CONDA_DEFAULT_ENV=biomni

# Install biomni package
COPY pyproject.toml README.md MANIFEST.in /app/
COPY biomni/ /app/biomni/
RUN pip install --no-cache-dir ".[gradio]"

# Copy entrypoint
COPY docker/entrypoint.py /app/entrypoint.py

# Gradio UI port
EXPOSE 7860
# MCP server port
EXPOSE 8000

ENV BIOMNI_DATA_PATH=/app/data
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "/app/entrypoint.py"]
