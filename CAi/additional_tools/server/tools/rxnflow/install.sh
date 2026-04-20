#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="rxnflow"
YML_FILE="$SCRIPT_DIR/environment.yml"
RXNFLOW_SRC="$SCRIPT_DIR/RxnFlow"

echo "=== RxnFlow 环境安装 ==="

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

# 安装 RxnFlow 本体及其依赖
if [ -d "${RXNFLOW_SRC}" ]; then
    echo "从本地源码安装 RxnFlow..."
    conda run -n "${ENV_NAME}" pip install -e "${RXNFLOW_SRC}"
else
    echo "警告: 未找到 ${RXNFLOW_SRC}，跳过 RxnFlow 源码安装。"
fi

echo ""
echo "✅ RxnFlow 环境安装完成！"
echo "使用方法: conda activate ${ENV_NAME}"
