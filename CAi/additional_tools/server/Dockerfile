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
# ─── 4. 分阶段 COPY：先只拷贝环境文件，充分利用 Docker 层缓存 ────────────────
#    只要 environment.yml 不变，下面所有 conda 安装层均命中缓存。
#    代码文件（run.py 等）在最后单独 COPY，改代码不会触发重新构建环境。
# =============================================================================
WORKDIR /app

# scscore（同时作为 FastAPI 主进程的运行环境）
COPY additional_tools/server/tools/scscore/environment.yml /tmp/envs/scscore.yml
# vina
COPY additional_tools/server/tools/vina/environment.yml /tmp/envs/vina.yml
# libinvent
COPY additional_tools/server/tools/libinvent/environment.yml /tmp/envs/libinvent.yml
# pmic（注意：yml 内部 name 为 chemprop，用 -n 覆盖为 config.json 中声明的 gflow312）
COPY additional_tools/server/tools/pmic/environment.yml /tmp/envs/pmic.yml
# scaffold
COPY additional_tools/server/tools/scaffold/environment.yml /tmp/envs/scaffold.yml
# toxicity
COPY additional_tools/server/tools/toxicity/environment.yml /tmp/envs/toxicity.yml
# rxnflow（yml 只含基础包，RxnFlow 源码需后续 pip install -e）
COPY additional_tools/server/tools/rxnflow/environment.yml /tmp/envs/rxnflow.yml

# =============================================================================
# ─── 5. 创建所有 conda 环境 ───────────────────────────────────────────────────
#    每个工具一个独立 RUN 块 = 独立缓存层，修改某工具不影响其他工具。
# =============================================================================

# ─── 环境 1: scscore（CPU，同时承载 FastAPI 主进程）─────────────────────────
RUN conda env create -n scscore -f /tmp/envs/scscore.yml \
    && conda clean -afy

# ─── 环境 2: vina（CPU，AutoDock Vina）─────────────────────────────────────
RUN conda env create -n vina -f /tmp/envs/vina.yml \
    && conda clean -afy

# ─── 环境 3: lib-invent（CPU，Lib-INVENT）───────────────────────────────────
RUN conda env create -n lib-invent -f /tmp/envs/libinvent.yml \
    && conda clean -afy

# ─── 环境 4: gflow312（GPU，PMIC/ChemProp；yml 内部 name=chemprop，此处覆盖）
RUN conda env create -n gflow312 -f /tmp/envs/pmic.yml \
    && conda clean -afy

# ─── 环境 5: chempro310（GPU，Scaffold 生成）────────────────────────────────
RUN conda env create -n chempro310 -f /tmp/envs/scaffold.yml \
    && conda clean -afy

# ─── 环境 6: toxicity（GPU，毒性预测）───────────────────────────────────────
RUN conda env create -n toxicity -f /tmp/envs/toxicity.yml \
    && conda clean -afy

# ─── 环境 7: rxnflow（GPU，RxnFlow；基础包，源码在下一步安装）──────────────
RUN conda env create -n rxnflow -f /tmp/envs/rxnflow.yml \
    && conda clean -afy

# ─── 环境 8: reinvent4（GPU，REINVENT4；无 environment.yml，PyPI 安装）──────
RUN conda create -n reinvent4 python=3.10 -y \
    && conda run -n reinvent4 pip install --no-cache-dir reinvent4 \
    && conda clean -afy

# ─── 环境 9: drugex（GPU，DrugEx；无 environment.yml，GitHub 安装）──────────
RUN conda create -n drugex python=3.10 -y \
    && conda run -n drugex pip install --no-cache-dir \
        "git+https://github.com/CDDLeiden/DrugEx.git@master" \
        "drugex[qsprpred] @ git+https://github.com/CDDLeiden/DrugEx.git@master" \
    && conda clean -afy

# =============================================================================
# ─── 6. 拷贝工具源码（频繁变更部分，放在环境安装之后避免缓存失效）────────────
# =============================================================================

# RxnFlow 源码需要 pip install -e（须在 COPY 源码后安装）
COPY additional_tools/server/tools/rxnflow/RxnFlow /app/server/tools/rxnflow/RxnFlow
RUN conda run -n rxnflow pip install --no-cache-dir \
        -e /app/server/tools/rxnflow/RxnFlow \
    && conda clean -afy

# 拷贝所有工具目录（包含 run.py、config.json、模型文件等）
COPY config.py                               /app/config.py
COPY additional_tools/server/app.py          /app/server/app.py
COPY additional_tools/server/job_manager.py  /app/server/job_manager.py
COPY additional_tools/server/tool_manager.py /app/server/tool_manager.py
COPY additional_tools/server/gpu_manager.py  /app/server/gpu_manager.py
COPY additional_tools/server/tools           /app/server/tools

# =============================================================================
# ─── 7. 运行时配置 ────────────────────────────────────────────────────────────
# =============================================================================
ENV TOOL_SERVER_HOST=0.0.0.0 \
    TOOL_SERVER_PORT=8001 \
    PYTHONPATH=/app

EXPOSE 8001

# FastAPI 主进程运行在 scscore 环境中
CMD ["conda", "run", "--no-capture-output", "-n", "scscore", \
     "python", "/app/server/app.py"]
