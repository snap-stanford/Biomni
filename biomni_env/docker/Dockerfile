# Biomni HITS Docker Image
# Base: Mambaforge for fast conda/mamba package management
#
# 빌드 방법: 프로젝트 루트에서 실행
#   docker build -t biomni-hits:latest -f biomni_env/docker/Dockerfile .
#
# 컨테이너 실행:
#   docker run -it --rm biomni-hits:latest
#   docker run -it --rm -v $(pwd):/workspace biomni-hits:latest

FROM docker.io/condaforge/mambaforge:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (needed for R and some bioinformatics tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    liblzma-dev \
    libbz2-dev \
    libpcre2-dev \
    libreadline-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy environment files first (for Docker layer caching)
COPY biomni_env/omics/environment.yml .
COPY biomni_env/omics/bio_env.yml .
COPY biomni_env/omics/r_packages.yml .
COPY biomni_env/omics/install_r_packages.R .

# Install packages from environment.yml directly to base environment
RUN mamba env update -n base -f environment.yml && \
    mamba clean -afy

# Install bio packages to base environment
RUN mamba env update -n base -f bio_env.yml && \
    mamba clean -afy

# Install R packages via conda
RUN mamba env update -n base -f r_packages.yml && \
    mamba clean -afy

# Install R packages via Rscript (this may take a while)
RUN Rscript install_r_packages.R

# Copy the entire project source code
COPY . /app/biomni_hits/

# Install the main package in editable mode
RUN pip install -e /app/biomni_hits/ && \
    pip cache purge

# Expose common ports (Jupyter, Chainlit, etc.)
EXPOSE 8888 8000 7860

# Default command
CMD ["bash"]

