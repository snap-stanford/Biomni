FROM continuumio/miniconda3:latest

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV NON_INTERACTIVE=1
ENV BIOMNI_AUTO_INSTALL=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    git \
    curl \
    wget \
    unzip \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy environment files and setup scripts
COPY biomni_env/ ./biomni_env/
COPY biomni_http_server.py ./
COPY pyproject.toml ./
COPY README.md ./
COPY biomni/ ./biomni/
COPY collate_utils/ ./collate_utils/

# Initialize conda and set up channels
RUN conda config --add channels conda-forge && \
    conda config --add channels bioconda && \
    conda config --add channels defaults

# Create the base conda environment first (lightweight)
RUN cd biomni_env && \
    conda env create -n biomni_e1 -f environment.yml

# Activate environment and install the reduced package set for container deployment
# Using fixed_env.yml which is more comprehensive but manageable for containers
RUN cd biomni_env && \
    conda env update -n biomni_e1 -f fixed_env.yml

# Install biomni package in the conda environment
RUN /opt/conda/envs/biomni_e1/bin/pip install .

# Create biomni_data directory with proper permissions
RUN mkdir -p /app/biomni_data && chmod 755 /app/biomni_data

# Clean conda cache to reduce image size
RUN conda clean --all -f -y

# Set environment variables for the server
ENV BIOMNI_HOST=0.0.0.0
ENV BIOMNI_PORT=3900
ENV BIOMNI_DATA_PATH=/app/biomni_data
ENV BIOMNI_ENABLE_REACT=false
ENV BIOMNI_DEFAULT_AGENT=a1

# Expose port
EXPOSE 3900

# Set up conda activation in shell
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate biomni_e1" >> ~/.bashrc

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate biomni_e1\n\
exec python /app/biomni_http_server.py' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Use the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
