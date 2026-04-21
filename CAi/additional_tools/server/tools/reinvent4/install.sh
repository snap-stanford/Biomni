#!/bin/bash
set -e

# 1. 动态获取当前脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="reinvent4"
REINVENT_SRC="$SCRIPT_DIR/REINVENT4"
REPO_URL="https://github.com/MolecularAI/REINVENT4.git"

echo "=== REINVENT4 环境安装 (基于官方脚本) ==="
echo "工作目录: $SCRIPT_DIR"
echo "------------------------------------------"

# 2. 源码自动获取与检查
if [ ! -d "${REINVENT_SRC}" ]; then
    echo "未检测到 REINVENT4 源码，正在从 GitHub 自动克隆..."
    git clone "${REPO_URL}" "${REINVENT_SRC}"
else
    echo "✅ REINVENT4 源码已存在于: ${REINVENT_SRC}"
fi

# 3. 环境清理检查
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

# 4. 创建纯净环境 (官方要求: python 3.10)
echo "创建 conda 环境 '${ENV_NAME}' (Python 3.10)..."
conda create -n "${ENV_NAME}" python=3.10 -y

# 5. 执行官方的一键安装脚本
echo "进入源码目录并执行官方安装命令 (安装 cu126 依赖)..."
cd "${REINVENT_SRC}"

# 运行前加一个保险检查，确保 install.py 确实存在
if [ ! -f "install.py" ]; then
    echo "❌ 致命错误: 在 ${REINVENT_SRC} 中未找到 install.py！请检查网络或克隆是否完整。"
    exit 1
fi

# 使用 conda run 来确保在目标环境内执行 python 命令
conda run -n "${ENV_NAME}" python install.py cu126

# 6. 安装完毕，清理源码目录 (仅保留环境用于推理)
echo "------------------------------------------"
echo "安装环境配置完成，正在清理 Git 源码文件夹以释放空间..."

# 注意：必须先 cd 切回上一级工作目录，否则在文件夹内部删除自己会报 "Device or resource busy"
cd "${SCRIPT_DIR}"
rm -rf "${REINVENT_SRC}"

echo "✅ 已成功删除源码文件夹: ${REINVENT_SRC}"
echo "------------------------------------------"
echo "✅ REINVENT4 (reinvent4) 推理环境部署大功告成！"
echo "👉 请运行以下命令激活环境: conda activate ${ENV_NAME}"