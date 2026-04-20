#!/bin/bash
set -e

ENV_NAME="reinvent4"
PYTHON_VERSION="3.10"

echo "=== REINVENT4 环境安装 ==="

if conda env list | grep -q "^${ENV_NAME} \|^${ENV_NAME}$"; then
    echo "Conda 环境 '${ENV_NAME}' 已存在。"
    read -p "是否删除并重新安装? (y/N) " answer
    if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
        echo "已跳过，退出。"
        exit 0
    fi
    echo "删除旧环境..."
    conda env remove -n "${ENV_NAME}" -y
fi

echo "创建 conda 环境 '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

echo "安装 REINVENT4..."
conda run -n "${ENV_NAME}" pip install reinvent4

echo ""
echo "✅ REINVENT4 环境安装完成！"
echo "使用方法: conda activate ${ENV_NAME}"
