#!/bin/bash
set -e

# 1. 动态获取当前脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="rxnflow"

# 2. 精确定位到包含 pyproject.toml 的 RxnFlow 目录
RXNFLOW_SRC="$SCRIPT_DIR/RxnFlow"

echo "=== RxnFlow 环境安装 (基于官方推荐路线) ==="
echo "工作目录: $SCRIPT_DIR"
echo "源码目录: $RXNFLOW_SRC"
echo "------------------------------------------"

# 3. 源码存在性前置检查 (如果没源码，直接提示并退出)
if [ ! -d "${RXNFLOW_SRC}" ] || [ ! -f "${RXNFLOW_SRC}/pyproject.toml" ]; then
    echo "❌ 致命错误: 未在当前目录下找到完整的 RxnFlow 源码包！"
    echo "寻找路径: ${RXNFLOW_SRC}"
    echo "------------------------------------------"
    echo "💡 解决方法:"
    echo "请先下载或克隆 RxnFlow 的源代码到当前目录中。"
    echo "如果你有该项目的 Git 权限，请在当前目录执行类似如下命令:"
    echo "  git clone <RxnFlow的Git仓库地址> RxnFlow"
    echo ""
    echo "请在确认 RxnFlow 文件夹和 pyproject.toml 文件就绪后，再次运行本脚本。"
    echo "------------------------------------------"
    exit 1
fi

# 4. 环境清理检查
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

# 5. 创建环境 (官方要求: python>=3.12,<3.13)
echo "创建 conda 环境 '${ENV_NAME}' (Python 3.12)..."
conda create -n "${ENV_NAME}" python=3.12 -y

# 6. 安装 UniDock (已添加 conda-forge 源解决找不到包的问题)
echo "安装 UniDock..."
conda install -n "${ENV_NAME}" unidock==1.1.2 -c conda-forge -y

# 7. 基于源码安装 RxnFlow 及其所有附加组件
echo "从本地源码安装 RxnFlow 及其所有依赖项 (基础+打分+口袋生成+开发工具)..."
conda run -n "${ENV_NAME}" pip install -e "${RXNFLOW_SRC}[unidock,pmnet,dev]" \
    --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html

echo "------------------------------------------"
echo "✅ RxnFlow 环境安装大功告成！"
echo "👉 请运行以下命令激活环境: conda activate ${ENV_NAME}"