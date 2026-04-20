#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import subprocess
import os
import sys
import shutil
import csv
from pathlib import Path
import torch

# ==========================================
# 1. I/O 与参数解析模块
# ==========================================
def load_params(cwd: Path) -> dict:
    params_file = cwd / "params.json"
    if not params_file.exists():
        raise FileNotFoundError("当前沙盒目录下未找到 params.json")
    with open(params_file, "r", encoding="utf-8") as f:
        return json.load(f)

def write_result(cwd: Path, payload: dict):
    with open(cwd / "result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ==========================================
# 2. 环境准备与配置模块
# ==========================================
def prepare_drugex_env(cwd: Path, drugex_source: Path, base_dir: str, params: dict) -> tuple:
    generator_model = params.get("generator", "arl_graph_trans_FT")
    input_fragments = params.get("input_fragments", "arl_test_graph.txt")
    # 设置一个合理的默认生成数量，比如 50
    num_samples = params.get("num_samples", 50) 

    new_molecules_dir = drugex_source / base_dir / "new_molecules"
    if new_molecules_dir.exists():
        for item in new_molecules_dir.iterdir():
            if item.is_dir() and item.name.startswith("backup_"):
                shutil.rmtree(item, ignore_errors=True)
            elif item.is_file() and item.suffix == ".tsv":
                item.unlink(missing_ok=True) 

    generators_dir = drugex_source / base_dir / "generators"
    generators_dir.mkdir(parents=True, exist_ok=True)
    
    pkg_file = generators_dir / f"{generator_model}.pkg"
    if not pkg_file.exists():
        raise FileNotFoundError(f"严重错误: 找不到模型权重文件 {pkg_file}")

    has_gpu = torch.cuda.is_available()
    generator_config = _build_generator_config(base_dir, generator_model, num_samples, has_gpu)
    
    config_path = generators_dir / f"{generator_model}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(generator_config, f, ensure_ascii=False, indent=2)

    wrapper_script = cwd / "drugex_wrapper.py"
    # ⭐ 注意这里新增了 num_samples 传入
    _write_wrapper_script(wrapper_script, drugex_source, base_dir, input_fragments, generator_model, num_samples)

    expected_output_tsv = new_molecules_dir / f"{generator_model}.tsv"
    return wrapper_script, expected_output_tsv, generator_model, input_fragments

def _build_generator_config(base_dir: str, generator_model: str, num_samples: int, has_gpu: bool) -> dict:
    """构建 DrugEx 的核心模型配置字典（包含所有必备字段）"""
    return {
        "base_dir": base_dir,
        "input": "arl",
        "voc_files": ["arl"],
        "output": "arl",
        "agent_path": f"{base_dir}/models/pretrained/graph-trans/Papyrus05.5_graph_trans_PT/Papyrus05.5_graph_trans_PT.pkg",
        "prior_path": None,
        "training_mode": "FT" if "FT" in generator_model else "RL",
        "mol_type": "graph",
        "algorithm": "trans",
        "use_gru": False,
        "epochs": 2,
        "batch_size": 32,
        "use_gpus": [0] if has_gpu else [],
        "patience": 50,
        "n_samples": num_samples,
        "epsilon": 0.1,
        "beta": 0.0,
        "scheme": "PRCD",
        "predictor": [],
        "activity_threshold": 6.5,
        "active_targets": [],
        "inactive_targets": [],
        "window_targets": [],
        # 👇 下面这些就是我刚才手残删掉的“致命参数”，现在全回来了！
        "ligand_efficiency": False,
        "le_thresholds": [0.0, 0.5],
        "lipophilic_efficiency": False,
        "lipe_thresholds": [4.0, 6.0],
        "qed": False,
        "uniqueness": False,
        "sa_score": False,
        "ra_score": False,
        "molecular_weight": False,
        "mw_thresholds": [200, 600],
        "logP": False,
        "logP_thresholds": [-5, 5],
        "tpsa": False,
        "tpsa_thresholds": [0, 140],
        "similarity_mol": None,
        "similarity_type": "fraggle",
        "similarity_threshold": 0.5,
        "similarity_tversky_weights": [0.7, 0.3],
        "debug": False,
        "targets": [],
        "output_long": generator_model,
        "output_file_base": f"{base_dir}/generators/{generator_model}"
    }
def _write_wrapper_script(script_path: Path, drugex_source: Path, base_dir: str, input_fragments: str, generator_model: str, num_samples: int):
    code = f"""
import sys
import os
import torch
import runpy

if torch.cuda.is_available():
    _ = torch.zeros(1).cuda()
    gpu_arg = '0'
else:
    gpu_arg = ''

sys.path.insert(0, r"{drugex_source.as_posix()}")

original_load = torch.load
def safe_load(f, map_location=None, pickle_module=None, **pickle_load_args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return original_load(f, map_location=device, pickle_module=pickle_module, **pickle_load_args)
torch.load = safe_load
torch.cuda.set_device = lambda d: None

# ⚠️ 致命一击：强制覆盖命令行默认的 1,2,3,4 多卡和生成数量 1
sys.argv = [
    'drugex.generate', 
    '-b', r'{base_dir}', 
    '-i', r'{input_fragments}', 
    '-g', r'{generator_model}',
    '-gpu', gpu_arg,
    '-n', str({num_samples}),
    '-bs', '32',
    '--keep_undesired'
]

runpy.run_module('drugex.generate', run_name='__main__')
"""
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

# ==========================================
# 3. 执行引擎模块
# ==========================================
def execute_drugex(wrapper_script: Path, drugex_source: Path, base_dir: str):
    env = os.environ.copy()
    env["BASE_DIR"] = base_dir
    
    result = subprocess.run(
        ["python", str(wrapper_script)],
        cwd=drugex_source,
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"DrugEx 内部执行失败:\n{result.stderr}")
        
    return result.stdout, result.stderr

# ==========================================
# 4. 数据解析模块
# ==========================================
def parse_results(cwd: Path, expected_output_tsv: Path, generator_model: str, stdout: str, stderr: str) -> tuple:
    if not expected_output_tsv.exists():
        actual_files = []
        if expected_output_tsv.parent.exists():
            actual_files = [f.name for f in expected_output_tsv.parent.iterdir() if f.is_file()]
            
        error_msg = f"未找到输出文件: {expected_output_tsv.name}\n"
        error_msg += f"该模型可能未能根据提供的片段生成任何合法的分子。\n"
        error_msg += f"输出目录下实际存在的文件: {actual_files}\n"
        error_msg += f"--- DrugEx Stderr (后台错误) ---\n{stderr[-1500:]}"
        raise FileNotFoundError(error_msg)

    sandbox_tsv_path = cwd / f"{generator_model}_results.tsv"
    shutil.copy2(expected_output_tsv, sandbox_tsv_path)

    molecules = []
    with open(sandbox_tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            molecules.append({
                "smiles": row.get("SMILES", ""),
                "valid": float(row.get("Valid", 0) or 0),
                "accurate": float(row.get("Accurate", 0) or 0),
                "desired": float(row.get("Desired", 0) or 0),
                "qsarpred_a2ar": float(row.get("QSPRpred_A2AR_RandomForestClassifier", 0) or 0)
            })
            
    return sandbox_tsv_path, molecules

# ==========================================
# 5. 主流程编排
# ==========================================
def main():
    result_payload = {"success": False, "summary": {}, "results": {}, "error": None}
    cwd = Path.cwd()

    try:
        params = load_params(cwd)
        script_dir = Path(__file__).resolve().parent
        drugex_source = script_dir / "DrugEx"
        base_dir = "tutorial/CLI/examples"
        
        if not drugex_source.exists():
            raise FileNotFoundError(f"找不到 DrugEx 源码目录: {drugex_source}")

        wrapper_script, expected_tsv, model_name, input_frags = prepare_drugex_env(
            cwd, drugex_source, base_dir, params
        )

        stdout, stderr = execute_drugex(wrapper_script, drugex_source, base_dir)
        sandbox_tsv, molecules = parse_results(cwd, expected_tsv, model_name, stdout, stderr)

        result_payload["success"] = True
        result_payload["summary"] = {
            "task": "DrugEx Graph-based Molecule Generation",
            "model_used": model_name,
            "input_fragments": input_frags,
            "total_molecules_generated": len(molecules),
            "output_tsv_path": str(sandbox_tsv)
        }
        result_payload["results"] = {
            "molecules_preview": molecules[:10]
        }
        del result_payload["error"]

    except Exception as e:
        result_payload["success"] = False
        result_payload["error"] = str(e)

    write_result(cwd, result_payload)

    if result_payload.get("success"):
        print(f"🎉 DrugEx 生成完毕！成功生成 {result_payload['summary']['total_molecules_generated']} 个分子。")
    else:
        print(f"❌ 运行失败: {result_payload.get('error')}", file=sys.stderr)

if __name__ == "__main__":
    main()