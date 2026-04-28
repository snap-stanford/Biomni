#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REINVENT4 Mol2Mol 手性约束分子生成 (action: mol2mol)
=====================================================
基于带手性标记 (@@) 的参考分子生成结构变体。
不支持 [*] 连接点。
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

        print(f"[INFO] mol2mol action 收到参数: {list(params.keys())}", file=sys.stderr)

        # 2. 解析参数 — 支持 smiles_list（批量）或 smiles（单个）
        smiles_list = params.get("smiles_list", [])
        if not smiles_list and params.get("smiles"):
            smiles_list = [params["smiles"]]

        if not smiles_list:
            raise ValueError("参数 smiles_list 或 smiles 不能为空，需要完整的参考分子 SMILES")

        num_variants = params.get("num_variants", 50)
        strategy = params.get("strategy", "beamsearch")
        temperature = params.get("temperature", 1.0)

        molecules = []
        for idx, smi in enumerate(smiles_list):
            molecules.append({
                "id": params.get("id", f"mol2mol_{idx}"),
                "smiles": smi,
                "num_variants": num_variants,
                "strategy": strategy,
                "temperature": temperature,
            })

        config = {
            "mode": "mol2mol",
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
                "task": "Mol2Mol Chiral-aware Molecule Generation (REINVENT4)",
                "mode": "mol2mol",
                "input_molecules": len(smiles_list),
                "requested_variants": num_variants,
                "strategy": strategy,
                "temperature": temperature,
                "generated_count": len(all_smiles),
                "successful_molecules": report.get("successful", 0),
                "failed_molecules": report.get("failed", 0),
            },
            "results": {
                "molecules": [{"smiles": s} for s in all_smiles],
                "input_smiles": smiles_list,
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
        print(f"[INFO] mol2mol 完成！生成了 {count} 个分子。", file=sys.stderr)
    else:
        print(f"[ERROR] mol2mol 失败: {result_payload.get('error')}", file=sys.stderr)


if __name__ == "__main__":
    main()