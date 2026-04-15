#!/usr/bin/env python3

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

# 获取工具源码所在目录 (用于读取静态的 scoring.toml 模板)
BASE_DIR = Path(__file__).resolve().parent / "REINVENT4"


def extract_smiles_from_csv(csv_path):
    """从给定的 CSV 文件中提取 SMILES 列表"""
    smiles_list = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row.get("SMILES", "").strip()
            if smiles:
                smiles_list.append(smiles)
    return smiles_list


def main():
    result_payload = {"success": False, "summary": {}, "results": {}, "error": None}

    # 获取当前沙盒隔离目录
    cwd = Path.cwd()

    try:
        # 1. 加载参数
        params_file = cwd / "params.json"
        if not params_file.exists():
            raise FileNotFoundError("当前沙盒目录下未找到 params.json")

        with open(params_file, encoding="utf-8") as f:
            params = json.load(f)

        # 2. 灵活解析输入数据 (支持直接传 SMILES 列表，或者传外部 CSV 路径)
        smiles_list = params.get("smiles_list", [])
        source_csv_path = params.get("source_csv_path")

        if not smiles_list and source_csv_path:
            source_file = Path(source_csv_path)
            if not source_file.exists():
                raise FileNotFoundError(f"指定的源 CSV 文件不存在: {source_file}")
            smiles_list = extract_smiles_from_csv(source_file)

        if not smiles_list:
            raise ValueError("未提供有效的 SMILES 列表或 source_csv_path，无法执行打分")

        # 3. 将 SMILES 写入当前沙盒目录的 compounds.smi
        compounds_file = cwd / "compounds.smi"
        with open(compounds_file, "w", encoding="utf-8") as f:
            f.write("\n".join(smiles_list))

        # 4. 准备 REINVENT4 配置文件
        # ⚠️ 确保你的工具目录下有 REINVENT4/configs/scoring.toml 这个模板文件
        source_config = BASE_DIR / "REINVENT4" / "configs" / "scoring.toml"
        if not source_config.exists():
            raise FileNotFoundError(f"找不到打分配置文件模板: {source_config}")

        with open(source_config, encoding="utf-8") as f:
            config_content = f.read()

        # 将配置中的输入输出路径替换为当前沙盒目录的绝对路径
        scoring_csv_path = cwd / "scoring.csv"

        config_content = re.sub(
            r'smiles_file\s*=\s*"[^"]*"',
            f'smiles_file = "{compounds_file.as_posix()}"',  # 使用 as_posix 避免 Windows 路径转义问题
            config_content,
        )
        config_content = re.sub(
            r'output_csv\s*=\s*"[^"]*"', f'output_csv = "{scoring_csv_path.as_posix()}"', config_content
        )

        config_path = cwd / "scoring.toml"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)

        # 5. 执行 REINVENT4 打分 (在沙盒目录下执行)
        result = subprocess.run(
            ["reinvent", "-l", "scoring.log", "scoring.toml"], cwd=cwd, capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"REINVENT 执行失败:\n{result.stderr}")

        # 6. 读取打分结果 scoring.csv
        scores_data = []
        if scoring_csv_path.exists():
            with open(scoring_csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    score_entry = {
                        "smiles": row.get("SMILES", ""),
                        "score": float(row.get("Score", 0) or 0),
                        "qed": float(row.get("QED", 0) or 0),
                        "mw": float(row.get("MW", 0) or 0),
                        "tanimoto": float(row.get("Tanimoto similarity ECF6", 0) or 0),
                        "alerts": row.get("Alerts", ""),
                    }
                    scores_data.append(score_entry)

        # 7. 构造标准化结果
        result_payload["success"] = True
        result_payload["summary"] = {
            "task": "REINVENT4 Multi-parameter Scoring",
            "input_molecules": len(smiles_list),
            "scored_molecules": len(scores_data),
            "output_csv_path": str(scoring_csv_path),
        }
        result_payload["results"] = {"scored_data": scores_data}
        del result_payload["error"]

    except Exception as e:
        result_payload["success"] = False
        result_payload["error"] = str(e)

    # 8. 写入 result.json 到沙盒目录
    with open(cwd / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    if result_payload.get("success"):
        print(f"🎉 REINVENT4 打分完成！成功为 {result_payload['summary']['scored_molecules']} 个分子打分。")
    else:
        print(f"❌ 打分失败: {result_payload.get('error')}", file=sys.stderr)


if __name__ == "__main__":
    main()
