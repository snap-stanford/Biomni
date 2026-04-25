# =============================================================================
# Tool Server Dockerfile
# Build context: CAi/ 目录（项目根）
#
# 构建命令（在 CAi/ 目录下执行）:
#   docker build -t tool-server -f additional_tools/server/Dockerfile .
#   docker compose -f additional_tools/server/docker-compose.yml up --build -d
# =============================================================================

# ─── 基础镜像 ────────────────────────────────────────────────────────────────
# CUDA 12.1 base（兼容环境中 CUDA 11.8 的 conda-bundled runtime）
# 若实验室有镜像代理，将 nvidia/ 前缀替换为代理地址
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ─── 1. 代理与环境变量 ────────────────────────────────────────────────────────
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY="localhost,127.0.0.1"

ENV HTTP_PROXY=$HTTP_PROXY \
    HTTPS_PROXY=$HTTPS_PROXY \
    http_proxy=$HTTP_PROXY \
    https_proxy=$HTTPS_PROXY \
    NO_PROXY=$NO_PROXY \
    no_proxy=$NO_PROXY \
    DEBIAN_FRONTEND=noninteractive

# ─── 2. 系统依赖 ─────────────────────────────────────────────────────────────
RUN if [ -n "$HTTP_PROXY" ]; then \
        echo "Acquire::http::Proxy \"$HTTP_PROXY\";"   > /etc/apt/apt.conf.d/95proxies && \
        echo "Acquire::https::Proxy \"$HTTPS_PROXY\";" >> /etc/apt/apt.conf.d/95proxies; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        wget git curl ca-certificates \
        libglib2.0-0 libxext6 libsm6 libxrender1 libgl1-mesa-glx \
        libboost-all-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# ─── 3. Miniconda ─────────────────────────────────────────────────────────────
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh

ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN if [ -n "$HTTP_PROXY" ]; then \
        conda config --set proxy_servers.http  "$HTTP_PROXY" && \
        conda config --set proxy_servers.https "$HTTPS_PROXY"; \
    fi && \
    conda install -n base -c conda-forge conda-libmamba-solver -y && \
    conda config --set solver libmamba && \
    conda clean -afy

# =============================================================================
# ─── 4. 拷贝服务源码与工具目录（install_all.sh 依赖相对路径 server/tools）──────
# =============================================================================
WORKDIR /app

COPY config.py                               /app/config.py
COPY additional_tools/server/app.py          /app/server/app.py
COPY additional_tools/server/job_manager.py  /app/server/job_manager.py
COPY additional_tools/server/tool_manager.py /app/server/tool_manager.py
COPY additional_tools/server/gpu_manager.py  /app/server/gpu_manager.py
COPY additional_tools/server/install_all.sh  /app/server/install_all.sh
COPY additional_tools/server/tools           /app/server/tools

# =============================================================================
# ─── 5. 使用 install_all.sh 一键安装工具 conda 环境 ───────────────────────────
# =============================================================================
RUN chmod +x /app/server/install_all.sh \
    && find /app/server/tools -name install.sh -type f -exec chmod +x {} \; \
    && bash /app/server/install_all.sh -y \
    && conda clean -afy

# =============================================================================
# ─── 6. 运行时配置 ────────────────────────────────────────────────────────────
# =============================================================================
ENV TOOL_SERVER_HOST=0.0.0.0 \
    TOOL_SERVER_PORT=8001 \
    PYTHONPATH=/app

EXPOSE 8001

# FastAPI 主进程运行在 scscore 环境中
CMD ["conda", "run", "--no-capture-output", "-n", "scscore", \
     "python", "/app/server/app.py"]
