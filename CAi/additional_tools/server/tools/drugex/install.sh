#!/bin/bash
set -e  # 出错即停

ENV_NAME="drugex"

# conda create --name $ENV_NAME python=3.10

echo "🔄 安装 DrugEx 基础包..."
conda run -n $ENV_NAME pip install git+https://github.com/CDDLeiden/DrugEx.git@master

echo "🔄 安装 DrugEx[qsprpred] 扩展..."
conda run -n $ENV_NAME pip install "drugex[qsprpred] @ git+https://github.com/CDDLeiden/DrugEx.git@master"

echo "✅ DrugEx 安装完成！"
echo "使用方法: conda activate $ENV_NAME"