FROM condaforge/mambaforge:latest AS conda-base

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

# Split: extract conda deps (no pip section) and install them first
RUN sed '/^  - pip:/,$d' /app/environment.yml > /app/conda_only.yml && \
    mamba env create -f /app/conda_only.yml -n biomni && \
    mamba clean -afy && \
    rm /app/conda_only.yml

# Activate the environment
ENV PATH=/opt/conda/envs/biomni/bin:$PATH
ENV CONDA_DEFAULT_ENV=biomni

# Install pip deps as a separate layer
RUN sed -n '/^  - pip:/,$ { s/^      - //p }' /app/environment.yml > /app/pip_requirements.txt && \
    if [ -s /app/pip_requirements.txt ]; then \
        pip install --no-cache-dir -r /app/pip_requirements.txt; \
    fi && \
    rm /app/pip_requirements.txt

# Remove caches and unnecessary files to slim down
RUN find /opt/conda/envs/biomni -name '*.pyc' -delete && \
    find /opt/conda/envs/biomni -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; \
    find /opt/conda/envs/biomni -name '*.a' -delete 2>/dev/null; \
    rm -rf /opt/conda/envs/biomni/share/doc \
           /opt/conda/envs/biomni/share/man \
           /opt/conda/envs/biomni/share/info \
           /opt/conda/envs/biomni/lib/python3.11/test \
           /opt/conda/envs/biomni/lib/python3.11/unittest \
    ; true

# Install biomni package
COPY pyproject.toml README.md MANIFEST.in /app/
COPY biomni/ /app/biomni/
RUN pip install --no-cache-dir --no-deps ".[gradio,bedrock]"

# Install langchain-aws separately with --no-deps to avoid numpy<2 conflict
# (boto3 and langchain-core are already installed from the conda env)
RUN pip install --no-cache-dir --no-deps "langchain-aws>=0.2,<0.3"

# Copy entrypoint
COPY docker/entrypoint.py /app/entrypoint.py

# Gradio UI port
EXPOSE 7860
# MCP server port
EXPOSE 8000

ENV BIOMNI_DATA_PATH=/app/data
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "/app/entrypoint.py"]
