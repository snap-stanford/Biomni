#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import torch
import warnings
from pathlib import Path

# 屏蔽底层依赖的警告，保持 Agent 读取的日志干净
warnings.filterwarnings("ignore")

# 1. 路径规范化设置
# BASE_DIR: 工具源码目录，用于加载本地的 .py 模块和静态模型权重
BASE_DIR = Path(__file__).resolve().parent / "chemprop_model"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from lightning import pytorch as pl
    from chemprop import data, featurizers, models
except ImportError as e:
    print(f"导入库失败，请确保安装了 chemprop 和 pytorch_lightning。\n错误信息: {e}", file=sys.stderr)
    sys.exit(1)


# ==========================================
# 2. 核心预测逻辑 (基于 Chemprop)
# ==========================================
def predict_single_pmic(model, smiles):
    """
    针对单个 SMILES 预测 pMIC 值并换算为 MIC (µM)
    """
    # 1. 必须使用与训练时完全相同的特征提取器
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    
    # 2. 构建 Datapoints
    datapoints = [data.MoleculeDatapoint.from_smi(smiles)]
    
    # 3. 构建 Dataset 和 DataLoader (num_workers=0 适合轻量级工具调用)
    dataset = data.MoleculeDataset(datapoints, featurizer)
    dataloader = data.build_dataloader(dataset, num_workers=0, shuffle=False)
    
    # 4. 配置轻量级的 Trainer 用于推理
    trainer = pl.Trainer(
        logger=False, 
        enable_progress_bar=False, # 关掉进度条保持终端输出干净
        accelerator="auto", 
        devices=1
    )
    
    # 5. 执行预测
    with torch.inference_mode():
        predictions = trainer.predict(model, dataloaders=dataloader)
    
    # 6. 处理预测结果
    preds_array = np.concatenate([batch.cpu().numpy() for batch in predictions], axis=0)
    pmic_val = float(preds_array.flatten()[0])
    
    # 7. 换算公式: MIC_uM = 10^(6 - pMIC)
    mic_uM = 10 ** (6.0 - pmic_val)
    
    return {
        "smiles": smiles,
        "pmic": round(pmic_val, 4),
        "mic_uM": round(mic_uM, 4)
    }

# ==========================================
# 3. toxicity prediction 函数入口
# ==========================================
def main():
    result = {"success": False, "summary": {}, "results": {}, "error": None}
    
    # 获取当前的沙盒隔离目录
    cwd = Path.cwd()

    try:
        # 1. 从沙盒目录读取参数
        params_path = cwd / "params.json"
        if not params_path.exists():
            raise FileNotFoundError("沙盒目录下未找到 params.json")
        
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        
        smiles = params.get("smiles")
        if not smiles:
            raise ValueError("必须提供 'smiles' 参数")

        # 2. 模型路径配置 (从 BASE_DIR 读取静态权重)
        CHECKPOINT_PATH = BASE_DIR / "checkpoints/best-pmic-epoch=13-val_loss=0.890.ckpt"
        
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"找不到模型权重文件: {CHECKPOINT_PATH}")

        # 3. 加载模型
        model = models.MPNN.load_from_checkpoint(str(CHECKPOINT_PATH))
        model.eval()

        # 4. 执行预测
        res = predict_single_pmic(model, smiles)

        # 5. 构造结果
        result["success"] = True
        result["summary"] = {
            "task": "Antibacterial pMIC Prediction (Chemprop)",
            "pMIC_value": res["pmic"],
            "estimated_MIC_uM": res["mic_uM"]
        }
        result["results"] = res
        del result["error"]

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)

    # 6. 写入结果到沙盒目录
    with open(cwd / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if result.get("success"):
        print(f"🎉 抗菌活性预测完成！pMIC: {result['summary']['pMIC_value']}, 预估 MIC: {result['summary']['estimated_MIC_uM']} µM")
    else:
        print(f"❌ 预测失败: {result.get('error')}", file=sys.stderr)

if __name__ == "__main__":
    main()