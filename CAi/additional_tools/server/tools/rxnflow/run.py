#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import time
import warnings
from pathlib import Path
import torch
# 屏蔽底层警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 路径规范化设置
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
RXNFLOW_DIR = BASE_DIR / "RxnFlow"

# 将 RxnFlow 的 src 注入系统路径
if str(RXNFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(RXNFLOW_DIR))
    sys.path.insert(0, str(RXNFLOW_DIR))

def _resolve(rel_path):
    """将相对于 RxnFlow 目录的路径转为绝对路径 (用于加载模型权重和环境)"""
    return str(RXNFLOW_DIR / rel_path)

def parse_temperature(temperature_param):
    temperature_info = temperature_param.split("-")
    sample_dist = temperature_info[0]
    dist_params = list(map(float, temperature_info[1:]))
    assert sample_dist in ("constant", "uniform", "loguniform", "gamma", "beta")
    if sample_dist == "constant":
        assert len(dist_params) == 1
    else:
        assert len(dist_params) == 2
    return sample_dist, dist_params

ALLOWED_STRUCTURE_SUFFIXES = {".sdf", ".mol2", ".pdb"}


def resolve_input_path(file_path, cwd_path):
    """
    将输入路径统一解析为绝对 Path。
    支持相对路径和绝对路径。
    """
    path_obj = Path(file_path)
    if not path_obj.is_absolute():
        path_obj = cwd_path / path_obj
    return path_obj


def validate_structure_file(file_path, cwd_path, param_name):
    """
    校验结构文件是否存在，且后缀是否为支持格式。

    支持格式:
    - .sdf
    - .mol2
    - .pdb

    Args:
        file_path (str): 用户传入的文件路径
        cwd_path (Path): 当前沙盒目录
        param_name (str): 参数名，用于报错信息，例如 'protein_pdb_path' 或 'ref_ligand_path'

    Returns:
        Path: 解析后的绝对路径对象
    """
    if not file_path:
        raise ValueError(f"必须提供 '{param_name}' 参数。")

    if not isinstance(file_path, str):
        raise ValueError(f"'{param_name}' 必须是字符串路径。")

    path_obj = resolve_input_path(file_path, cwd_path)

    if not path_obj.exists():
        raise FileNotFoundError(f"{param_name} 指向的文件不存在: {path_obj}")

    if not path_obj.is_file():
        raise ValueError(f"{param_name} 不是一个有效文件: {path_obj}")

    suffix = path_obj.suffix.lower()
    if suffix not in ALLOWED_STRUCTURE_SUFFIXES:
        allowed_text = ", ".join(sorted(ALLOWED_STRUCTURE_SUFFIXES))
        raise ValueError(
            f"{param_name} 文件格式不受支持: {path_obj.name}。"
            f"仅支持以下格式: {allowed_text}。"
        )

    return path_obj

def normalize_center(center):
    """
    将不同格式的 center 统一解析为 [x, y, z] 的 float 列表。
    支持输入格式：
    - [x, y, z]
    - (x, y, z)
    - "x,y,z"
    - {"x": x, "y": y, "z": z}
    """
    if center is None:
        return None

    # 1) list / tuple
    if isinstance(center, (list, tuple)):
        if len(center) != 3:
            raise ValueError("'center' 必须包含 3 个坐标值，格式例如 [x, y, z]。")
        try:
            return [float(v) for v in center]
        except Exception:
            raise ValueError("'center' 中的 3 个坐标值必须都能转换为数值。")

    # 2) dict: {"x": ..., "y": ..., "z": ...}
    if isinstance(center, dict):
        required_keys = ["x", "y", "z"]
        if not all(k in center for k in required_keys):
            raise ValueError("'center' 字典格式必须包含 x、y、z 三个字段。")
        try:
            return [float(center["x"]), float(center["y"]), float(center["z"])]
        except Exception:
            raise ValueError("'center' 字典中的 x、y、z 必须都能转换为数值。")

    # 3) str: "x,y,z"
    if isinstance(center, str):
        parts = [p.strip() for p in center.split(",")]
        if len(parts) != 3:
            raise ValueError("'center' 字符串格式必须为 'x,y,z'。")
        try:
            return [float(v) for v in parts]
        except Exception:
            raise ValueError("'center' 字符串中的 3 个坐标值必须都能转换为数值。")

    raise ValueError(
        "'center' 只支持以下格式之一："
        "[x, y, z]、(x, y, z)、'x,y,z'、{'x': x, 'y': y, 'z': z}。"
    )

# ==========================================
# 2. 核心采样逻辑
# ==========================================
def rxnflow_zeroshot_sampling(
    protein_pdb_path,
    cwd_path, # 传入当前的沙盒目录
    center=None,
    ref_ligand_path=None,
    num_samples=100,
    env_dir=None,
    model_path=None,
    temperature="uniform-16-64",
    use_cuda=True,
    save_reward=True,
):
    try:
        from rxnflow.config import Config, init_empty
        from rxnflow.tasks.multi_pocket import ProxySampler
    except Exception as e:
        raise(e)
    # 1. 处理静态资源路径 (依赖于工具源码目录 RXNFLOW_DIR)
    if env_dir is None:
        env_dir = _resolve("data/envs/zincfrag")
    if model_path is None:
        model_path = _resolve("weights/qvina-unif-0-64_20250512.pt")

    # 2. 处理输入输出文件路径 (依赖于沙盒目录 cwd_path)
    protein_path_obj = Path(protein_pdb_path)
    if not protein_path_obj.is_absolute():
        protein_path_obj = cwd_path / protein_path_obj
    
    if ref_ligand_path:
        ref_path_obj = Path(ref_ligand_path)
        if not ref_path_obj.is_absolute():
            ref_path_obj = cwd_path / ref_path_obj
        ref_ligand_path = str(ref_path_obj)

    if not protein_path_obj.exists():
        raise FileNotFoundError(f"蛋白质 PDB 文件不存在: {protein_path_obj}")
    if center is None and ref_ligand_path is None:
        raise ValueError("center 和 ref_ligand_path 必须提供其一")
    if ref_ligand_path is not None and not os.path.exists(ref_ligand_path):
        raise FileNotFoundError(f"参考配体文件不存在: {ref_ligand_path}")

    # 3. 初始化 RxnFlow 采样器
    config = init_empty(Config())
    config.seed = 1
    config.env_dir = env_dir
    config.algo.num_from_policy = 100
    config.algo.action_subsampling.sampling_ratio = 0.1

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    sampler = ProxySampler(config, model_path, device)
    sample_dist, dist_params = parse_temperature(temperature)
    sampler.update_temperature(sample_dist, dist_params)
    sampler.set_pocket(str(protein_path_obj), center, ref_ligand_path)

    # 4. 执行采样
    tick = time.time()
    res = sampler.sample(num_samples, calc_reward=save_reward)
    elapsed = time.time() - tick

    # 5. 保存结果到沙盒目录
    preview_smiles = []
    if save_reward:
        output_path = cwd_path / "rxnflow_results.csv"
        with open(output_path, "w") as w:
            w.write(",SMILES,QED,Proxy\n")
            for idx, sample in enumerate(res):
                smiles = sample["smiles"]
                qed = sample["info"]["reward_qed"]
                proxy = sample["info"]["reward_vina"]
                if torch.is_tensor(qed): qed = qed.item()
                if torch.is_tensor(proxy): proxy = proxy.item()
                w.write(f"sample{idx},{smiles},{qed:.3f},{proxy:.3f}\n")
                if idx < 5:  # 截取前5个作为预览
                    preview_smiles.append({"smiles": smiles, "qed": round(float(qed), 3), "proxy_score": round(float(proxy), 3)})
    else:
        output_path = cwd_path / "rxnflow_results.smi"
        with open(output_path, "w") as w:
            for idx, sample in enumerate(res):
                smiles = sample["smiles"]
                w.write(f"{smiles}\tsample{idx}\n")
                if idx < 5:
                    preview_smiles.append({"smiles": smiles})

    return {
        "num_generated": len(res),
        "sampling_time_sec": round(elapsed, 3),
        "output_file": str(output_path),
        "preview": preview_smiles
    }


# ==========================================
# 3. 工具入口函数
# ==========================================
def main():
    result_payload = {"success": False, "summary": {}, "results": {}, "error": None}

    cwd = Path.cwd()

    try:
        # 1. 读取参数
        params_file = cwd / "params.json"
        if not params_file.exists():
            raise FileNotFoundError("当前沙盒目录下未找到 params.json")

        with open(params_file, "r", encoding="utf-8") as f:
            params = json.load(f)

        # 2. 必需参数校验：protein_pdb_path
        protein_pdb_path = params.get("protein_pdb_path")
        # if not protein_pdb_path:
        #     raise ValueError("必须提供 'protein_pdb_path' 参数。")
        # 校验蛋白质文件
        protein_path_obj = validate_structure_file(
            protein_pdb_path, cwd, "protein_pdb_path"
        )
        protein_pdb_path = str(protein_path_obj)
        # 3. 必需参数校验：center 或 ref_ligand_path 至少提供一个
        center = params.get("center")
        ref_ligand_path = params.get("ref_ligand_path")

        if center is None and not ref_ligand_path:
            raise ValueError(
                "必须提供 'center' 或 'ref_ligand_path' 其中之一。"
            )

        # 4. center 格式校验（如果提供）
        if center is not None:
            center = normalize_center(center)

        # 5. ref_ligand_path 基本校验（如果提供）
        if ref_ligand_path is not None and not isinstance(ref_ligand_path, str):
            raise ValueError("'ref_ligand_path' 必须是字符串路径。")
                # 如果给了参考配体，也校验格式
        if ref_ligand_path:
            ref_path_obj = validate_structure_file(
                ref_ligand_path, cwd, "ref_ligand_path"
            )
            ref_ligand_path = str(ref_path_obj)

        # 6. 其他参数保持默认或可选覆盖
        data = rxnflow_zeroshot_sampling(
            protein_pdb_path=protein_pdb_path,
            cwd_path=cwd,
            center=center,
            ref_ligand_path=ref_ligand_path,
            num_samples=params.get("num_samples", 10),
            env_dir=params.get("env_dir"),
            model_path=params.get("model_path"),
            temperature=params.get("temperature", "uniform-16-64"),
            use_cuda=params.get("use_cuda", True),
            save_reward=params.get("save_reward", True),
        )

        # 7. 构造标准结果
        result_payload["success"] = True
        result_payload["summary"] = {
            "task": "Target-aware Zero-shot Generation (RxnFlow)",
            "generated_count": data["num_generated"],
            "sampling_time_sec": data["sampling_time_sec"],
            "output_file": data["output_file"],
        }
        result_payload["results"] = {
            "generated_preview": data["preview"]
        }
        result_payload["error"] = None

    except Exception as e:
        result_payload["success"] = False
        result_payload["error"] = str(e)

    # 8. 写入结果
    with open(cwd / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    if result_payload.get("success"):
        print(
            f"🎉 靶向生成完成！用时 {result_payload['summary']['sampling_time_sec']} 秒，"
            f"生成了 {result_payload['summary']['generated_count']} 个分子。"
        )
    else:
        print(f"❌ 运行失败: {result_payload.get('error')}", file=sys.stderr)
        
if __name__ == "__main__":
    main()