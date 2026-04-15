#!/usr/bin/env python3

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

# 获取工具源码所在目录
BASE_DIR = Path(__file__).resolve().parent / "REINVENT4"


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

        # 允许 Agent 控制生成数量，默认 50
        num_samples = params.get("num_samples", 50)

        # 2. 定位静态资源 (从 BASE_DIR 读取)
        source_config = BASE_DIR / "REINVENT4" / "configs" / "sampling.toml"
        model_file = BASE_DIR / "REINVENT4" / "priors" / "reinvent.prior"

        if not source_config.exists():
            raise FileNotFoundError(f"找不到配置文件模板: {source_config}")
        if not model_file.exists():
            raise FileNotFoundError(f"找不到模型文件: {model_file}")

        # 3. 读取并修改 sampling.toml
        with open(source_config, encoding="utf-8") as f:
            config_content = f.read()

        # 清除旧的 model_file 行
        lines = config_content.split("\n")
        new_lines = [line for line in lines if not line.strip().startswith("model_file")]
        config_content = "\n".join(new_lines)

        # 注入正确的绝对路径模型文件
        # 注意：使用 as_posix() 避免 Windows 路径反斜杠导致的转义问题
        config_content = re.sub(
            r"(\[parameters\]\s*)", r'\1\nmodel_file = "' + model_file.as_posix() + '"\n', config_content
        )

        # 动态修改生成数量 num_smiles
        config_content = re.sub(r"num_smiles\s*=\s*\d+", f"num_smiles = {num_samples}", config_content)

        # Redirect output CSV to the sandbox directory.
        # The template uses `output_file` (single quotes), so match both key name and quote style.
        sampling_csv_path = cwd / "sampling.csv"
        config_content = re.sub(
            r"output_file\s*=\s*['\"][^'\"]*['\"]", f'output_file = "{sampling_csv_path.as_posix()}"', config_content
        )

        # 将修改后的配置写入沙盒目录
        config_path = cwd / "sampling.toml"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)

        # 4. 执行分子生成 (在沙盒目录下执行)
        result = subprocess.run(
            ["reinvent", "-l", "sampling.log", "sampling.toml"], cwd=cwd, capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"REINVENT 执行失败:\n{result.stderr}")

        # 5. 解析生成的 sampling.csv
        molecules = []
        if sampling_csv_path.exists():
            with open(sampling_csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smiles = row.get("SMILES", "").strip()
                    if smiles:
                        molecules.append(
                            {
                                "smiles": smiles,
                                "state": int(row.get("SMILES_state", 0)),
                                "nll": float(row.get("NLL", 0.0)),  # 负对数似然
                            }
                        )

        if not molecules:
            raise ValueError("REINVENT 运行成功，但未生成任何有效的 SMILES")

        # 6. 构造标准结果
        result_payload["success"] = True
        result_payload["summary"] = {
            "task": "De novo Molecule Generation (REINVENT4)",
            "requested_samples": num_samples,
            "generated_count": len(molecules),
            "output_csv_path": str(sampling_csv_path),
        }
        result_payload["results"] = {"molecules": molecules}
        del result_payload["error"]

    except Exception as e:
        result_payload["success"] = False
        result_payload["error"] = str(e)

    # 7. 写入 result.json 到沙盒目录
    with open(cwd / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    if result_payload.get("success"):
        print(f"🎉 生成完毕！成功使用 REINVENT4 采样了 {result_payload['summary']['generated_count']} 个分子。")
    else:
        print(f"❌ 运行失败: {result_payload.get('error')}", file=sys.stderr)


if __name__ == "__main__":
    main()
