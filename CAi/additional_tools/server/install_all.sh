#!/bin/bash
# 一键安装所有工具的 conda 环境
# 每个工具都有独立的 install.sh，可单独运行

set -e

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tools"

# 所有工具列表（tool_name:install_script）
TOOLS=(
    "drugex"
    "libinvent"
    "pmic"
    "reinvent4"
    "rxnflow"
    "scaffold"
    "scscore"
    "toxicity"
    "vina"
)

# ============ 解析参数 ============
SELECTED=()
SKIP_CONFIRM=false

print_usage() {
    echo "用法: $0 [选项] [工具名...]"
    echo ""
    echo "选项:"
    echo "  -y, --yes       跳过确认提示，自动安装所有工具"
    echo "  -h, --help      显示帮助"
    echo ""
    echo "工具名（可指定多个，空格分隔）:"
    for t in "${TOOLS[@]}"; do
        echo "  $t"
    done
    echo ""
    echo "示例:"
    echo "  $0                   # 交互式选择安装哪些工具"
    echo "  $0 vina scscore      # 只安装 vina 和 scscore"
    echo "  $0 -y                # 全部安装，跳过确认"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes)
            SKIP_CONFIRM=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            SELECTED+=("$1")
            shift
            ;;
    esac
done

# 如果没有指定工具，默认安装全部
if [ ${#SELECTED[@]} -eq 0 ]; then
    SELECTED=("${TOOLS[@]}")
fi

# ============ 运行安装 ============
echo "======================================"
echo "  Biomni 工具环境一键安装脚本"
echo "======================================"
echo "将安装以下工具环境:"
for t in "${SELECTED[@]}"; do
    echo "  - $t"
done
echo ""

if [ "$SKIP_CONFIRM" = false ]; then
    read -p "确认开始安装? (y/N) " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "已取消。"
        exit 0
    fi
fi

FAILED=()
SUCCESS=()

for tool in "${SELECTED[@]}"; do
    INSTALL_SH="$TOOLS_DIR/$tool/install.sh"
    if [ ! -f "$INSTALL_SH" ]; then
        echo "⚠️  跳过 '$tool': 未找到 $INSTALL_SH"
        continue
    fi

    echo ""
    echo "======================================"
    echo "  安装: $tool"
    echo "======================================"
    chmod +x "$INSTALL_SH"

    # 在子 shell 中运行，避免 set -e 因单个失败中断整体
    if (export SKIP_REINSTALL_CONFIRM=y; bash "$INSTALL_SH"); then
        SUCCESS+=("$tool")
    else
        echo "❌ $tool 安装失败"
        FAILED+=("$tool")
    fi
done

# ============ 汇总结果 ============
echo ""
echo "======================================"
echo "  安装结果汇总"
echo "======================================"
if [ ${#SUCCESS[@]} -gt 0 ]; then
    echo "✅ 成功安装:"
    for t in "${SUCCESS[@]}"; do
        echo "   - $t"
    done
fi
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "❌ 安装失败:"
    for t in "${FAILED[@]}"; do
        echo "   - $t"
    done
    exit 1
fi
