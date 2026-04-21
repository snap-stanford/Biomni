#!/bin/bash
set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="drugex"
PYTHON_VERSION="3.10"

echo "======================================"
echo "      DrugEx 环境安装脚本"
echo "======================================"

# 1. 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} \|^${ENV_NAME}$"; then
    echo "Conda 环境 '${ENV_NAME}' 已存在。"
    read -p "是否删除并重新安装? (y/N) " answer
    if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
        echo "已跳过环境创建，尝试继续安装包..."
    else
        echo "正在删除旧环境 '${ENV_NAME}'..."
        conda env remove -n "${ENV_NAME}" -y
        echo "正在创建新环境 '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
        conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    fi
else
    echo "正在创建 Conda 环境 '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# 2. 安装 DrugEx 核心组件
echo ""
echo "🔄 正在安装 DrugEx 基础包 (来自 GitHub)..."
# 使用 conda run 确保在正确的环境下执行 pip
conda run -n "${ENV_NAME}" pip install git+https://github.com/CDDLeiden/DrugEx.git@master

# 3. 安装扩展组件
echo ""
echo "🔄 正在安装 DrugEx[qsprpred] 扩展组件..."
conda run -n "${ENV_NAME}" pip install "drugex[qsprpred] @ git+https://github.com/CDDLeiden/DrugEx.git@master"

echo ""
echo "✅ DrugEx 环境安装完成！"
echo "--------------------------------------"
echo "使用方法: conda activate ${ENV_NAME}"
echo "======================================"