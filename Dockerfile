FROM python:3.11-slim

LABEL maintainer="Biomni Team"
LABEL description="Biomni - Biomedical AI Agent"

WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖相关文件，利用 Docker 缓存层
COPY pyproject.toml ./
COPY biomni/version.py ./biomni/version.py

# 安装 Python 依赖
RUN pip install --no-cache-dir .[gradio]

# 复制项目代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data /app/biomni_data

# 暴露 Gradio 默认端口
EXPOSE 7860

# 默认环境变量（运行时覆盖）
ENV BIOMNI_DATA_PATH=/app/data
ENV PYTHONUNBUFFERED=1

# 默认启动命令
CMD ["python", "-c", "from biomni.agent.a1 import A1; print('Biomni Agent loaded successfully.')"]
