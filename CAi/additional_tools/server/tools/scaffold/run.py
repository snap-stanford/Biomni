#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
from pathlib import Path
from rdkit import Chem
import warnings

# 忽略所有 UserWarning 和 DeprecationWarning
warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent / "scaffold_generate"

# 强行将工具源码目录加入 sys.path 的最前面。
# 这样即使在 workspace/jobs/<job_id> 下运行，也能正确 import 源码目录里的 py 文件
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ==========================================
# 1. 核心模块导入
# ==========================================
try:
    from scaffold_constrained_model import scaffold_constrained_RNN
    from data_structs import Vocabulary
    from utils import seq_to_smiles
except ImportError as e:
    print(f"导入失败！请确保本脚本与 scaffold_constrained_model.py 等文件在同一文件夹下。\n错误信息: {e}", file=sys.stderr)
    sys.exit(1)


def generate_analogs(scaffold_smiles, batch_size=100):
    """
    根据骨架生成类似物
    """
    if "*" not in scaffold_smiles:
        raise ValueError("输入骨架必须包含至少一个 '*' 作为生长位点。")

    # 2. 解析模型和词汇表路径 (使用 BASE_DIR 寻找静态权重)
    vocab_path = BASE_DIR / "data/DistributionLearningBenchmark/Voc"
    ckpt_path = BASE_DIR / "data/DistributionLearningBenchmark/Prior_ChEMBL_randomized.ckpt"

    if not vocab_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型权重或词汇表文件，请确认路径：\nVocab: {vocab_path}\nCkpt: {ckpt_path}")

    # 3. 初始化词汇表与模型
    voc = Vocabulary(init_from_file=str(vocab_path))
    agent = scaffold_constrained_RNN(voc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        agent.rnn.load_state_dict(torch.load(str(ckpt_path)))
    else:
        agent.rnn.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    
    agent.rnn.to(device)

    # 4. 执行采样生成
    seqs, agent_likelihood, entropy = agent.sample(pattern=scaffold_smiles, batch_size=batch_size)

    # 5. 解码并验证合法性与去重
    smiles_list = seq_to_smiles(seqs, voc)
    valid_unique_smiles = []

    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m:
            if s not in valid_unique_smiles:
                valid_unique_smiles.append(s)

    return valid_unique_smiles


def main():
    result = {"success": False, "summary": {}, "results": [], "error": None}
    
    try:
        # 1. 加载参数 (强制从当前沙盒目录 cwd 读取 params.json)
        params_file = Path("params.json")
        if not params_file.exists():
            raise FileNotFoundError("当前沙盒目录下未找到 params.json。")
            
        with open(params_file, "r", encoding="utf-8") as f:
            params = json.load(f)

        input_smiles = params.get("smiles")
        num_analogs = params.get("num_analogs", 10) 

        if not input_smiles:
            raise ValueError("必须提供 'smiles' 参数。")
        # scaffold-based generation 强制要求包含 *
        if "*" not in input_smiles:
            raise ValueError(
                "当前工具要求输入用于 scaffold-based generation 的 scaffold SMILES，"
                "并且必须包含至少一个 '*' 作为生长/连接位点。"
            )
        # if smiles with @@ 返回进行smiles调整重新输入
        # 如果 smiles 中包含 @@，直接报错并提示用户先去除后再输入
        if "@@" in input_smiles:
            raise ValueError(
                "当前工具暂不支持包含 '@@' 立体化学标记的 SMILES。"
                "请先去掉 '@@' 结构后再重新输入。"
            )
        if not isinstance(num_analogs, int):
            raise ValueError("'num_analogs' 参数必须是整数。")

        if num_analogs <= 0:
            raise ValueError("'num_analogs' 必须大于 0。")
        
        # 2. 执行核心生成逻辑
        generated_smiles = generate_analogs(input_smiles, batch_size=num_analogs)

        # 3. 构造输出结果
        result["success"] = True
        result["summary"] = {
            "input_scaffold": input_smiles,
            "requested_batch_size": num_analogs,
            "valid_unique_generated": len(generated_smiles)
        }
        # 将生成的 smiles 包装成字典列表形式，方便拓展
        result["results"] = [{"smiles": smi} for smi in generated_smiles]
        # 删除不需要序列化的 None 对象
        del result["error"] 

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)

    # 4. 写入 result.json (写到当前沙盒目录 cwd)
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 5. 打印状态供 stdout.log 捕获，方便排查
    if result.get("success"):
        print(f"🎉 骨架生成成功！生成了 {result['summary']['valid_unique_generated']} 个合法去重分子。结果已保存至 result.json。")
    else:
        print(f"❌ 工具运行失败: {result.get('error')}", file=sys.stderr)


if __name__ == "__main__":
    main()