#!/usr/bin/env python3
"""
工具后端脚本模板（单 action 版本）
=====================================
约定：
  - 脚本的 cwd 就是 Job 沙盒目录，直接用相对路径读写文件
  - 从 params.json 读取参数
  - 将结果写入 result.json
  - 调试信息用 print(..., file=sys.stderr)，不要用 print() 输出结果
"""

import json
import sys
from pathlib import Path


def run_my_tool(params: dict) -> dict:
    """
    在这里写你的核心计算逻辑。

    Args:
        params: 从 params.json 加载的参数字典

    Returns:
        符合标准格式的结果字典
    """
    # ---------- 解析参数 ----------
    smiles_list = params.get("smiles_list", [])
    # my_param   = params.get("my_param", "default_value")

    if not smiles_list:
        raise ValueError("参数 smiles_list 不能为空")

    # ---------- 核心计算 ----------
    # TODO: 替换为你的真实计算逻辑
    results = []
    for smi in smiles_list:
        results.append(
            {
                "smiles": smi,
                "score": 0.0,  # 替换为真实计算结果
            }
        )

    # ---------- 构造返回值 ----------
    return {
        "success": True,
        "summary": {
            "task": "MyTool Calculation",  # 任务描述
            "input_molecules": len(smiles_list),
            "processed_molecules": len(results),
        },
        "results": results,
        "errors": None,
    }


def main():
    cwd = Path.cwd()  # 沙盒目录
    result_payload = {"success": False, "error": None}

    try:
        # 1. 读取参数（固定写法）
        params_file = cwd / "params.json"
        if not params_file.exists():
            raise FileNotFoundError("当前沙盒目录下未找到 params.json")
        with open(params_file, encoding="utf-8") as f:
            params = json.load(f)

        print(f"[INFO] 收到参数: {list(params.keys())}", file=sys.stderr)

        # 2. 执行计算
        result_payload = run_my_tool(params)

    except Exception as e:
        result_payload = {"success": False, "error": str(e)}
        print(f"[ERROR] {e}", file=sys.stderr)

    # 3. 写入结果（固定写法）
    with open(cwd / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    if result_payload.get("success"):
        count = result_payload.get("summary", {}).get("processed_molecules", "?")
        print(f"[INFO] 完成！处理了 {count} 个分子。", file=sys.stderr)
    else:
        print(f"[ERROR] 失败: {result_payload.get('error')}", file=sys.stderr)


if __name__ == "__main__":
    main()
