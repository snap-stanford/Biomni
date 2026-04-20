#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REINVENT4 LibInvent 骨架装饰 (action: libinvent)
==================================================
在含 [*] 连接点的骨架上生成 R-基团变体。
不支持 @@ 手性标记。
"""

import json
import sys
from pathlib import Path

_TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOL_DIR))

from reinvent_generator import generate_molecules


def main():
    cwd = Path.cwd()
    result_payload = {"success": False, "error": None}

    try:
        # 1. 读取参数
        params_file = cwd / "params.json"
        if not params_file.exists():
            raise FileNotFoundError("当前沙盒目录下未找到 params.json")
        with open(params_file, "r", encoding="utf-8") as f:
            params = json.load(f)

        print(f"[INFO] libinvent action 收到参数: {list(params.keys())}", file=sys.stderr)

        # 2. 解析参数 — 支持 smiles_list（批量）或 smiles（单个）
        smiles_list = params.get("smiles_list", [])
        if not smiles_list and params.get("smiles"):
            smiles_list = [params["smiles"]]

        if not smiles_list:
            raise ValueError("参数 smiles_list 或 smiles 不能为空，需要含 [*] 的骨架 SMILES")

        num_variants = params.get("num_variants", params.get("num_decorations", 50))

        molecules = []
        for idx, smi in enumerate(smiles_list):
            molecules.append({
                "id": params.get("id", f"libinvent_{idx}"),
                "smiles": smi,
                "num_variants": num_variants,
            })

        config = {
            "mode": "libinvent",
            "output_dir": str(cwd / "output"),
            "device": params.get("device", "cuda:0"),
            "molecules": molecules
        }

        # 3. 调用核心生成逻辑
        report = generate_molecules(config)

        # 4. 转换为框架标准 result.json 格式
        all_smiles = []
        for r in report.get("results", []):
            if r.get("success") and r.get("output_file"):
                try:
                    import pandas as pd
                    df = pd.read_csv(r["output_file"])
                    smiles_col = "SMILES" if "SMILES" in df.columns else df.columns[0]
                    all_smiles.extend(df[smiles_col].dropna().tolist())
                except Exception:
                    pass

        result_payload = {
            "success": report.get("status") != "failed",
            "summary": {
                "task": "LibInvent Scaffold Decoration (REINVENT4)",
                "mode": "libinvent",
                "input_scaffolds": len(smiles_list),
                "requested_variants": num_variants,
                "generated_count": len(all_smiles),
                "successful_molecules": report.get("successful", 0),
                "failed_molecules": report.get("failed", 0),
            },
            "results": {
                "molecules": [{"smiles": s} for s in all_smiles],
                "input_scaffolds": smiles_list,
            },
            "errors": [
                {"molecule_id": r["molecule_id"], "error": r["error_message"]}
                for r in report.get("results", []) if not r.get("success")
            ] or None
        }

    except Exception as e:
        result_payload = {"success": False, "error": str(e)}
        print(f"[ERROR] {e}", file=sys.stderr)

    # 5. 写入 result.json
    with open(cwd / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    if result_payload.get("success"):
        count = result_payload.get("summary", {}).get("generated_count", "?")
        print(f"[INFO] libinvent 完成！生成了 {count} 个分子。", file=sys.stderr)
    else:
        print(f"[ERROR] libinvent 失败: {result_payload.get('error')}", file=sys.stderr)


if __name__ == "__main__":
    main()