#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="lib-invent"
YML_FILE="$SCRIPT_DIR/environment.yml"

echo "=== Lib-INVENT 环境安装 ==="

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

echo "从 ${YML_FILE} 创建 conda 环境 '${ENV_NAME}'..."
conda env create -f "${YML_FILE}"

echo ""
echo "✅ Lib-INVENT 环境安装完成！"
echo "使用方法: conda activate ${ENV_NAME}"
