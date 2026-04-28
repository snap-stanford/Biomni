#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import warnings
import torch
import numpy as np
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Draw import rdMolDraw2D
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

# ==========================================
# 1. 路径规范化设置
# ==========================================
# BASE_DIR: 工具源码目录，用于加载本地的 .py 模块和静态模型权重
BASE_DIR = Path(__file__).resolve().parent / "toxicity_chemberta_tool"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from model_tox import ChemBERTaWithFingerprint
except ImportError as e:
    print(f"导入模型模块失败，请检查 model_tox.py 是否与本脚本在同一目录下。\n错误信息: {e}", file=sys.stderr)
    sys.exit(1)


# ==========================================
# 2. 化学结构拆分与 Masking 逻辑
# ==========================================
def split_smiles_into_substructures(smiles):
    """拆分SMILES为合理的化学子结构 (BRICS + 环 + 杂原子)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    atom_sets = []
    # 1. BRICS 断裂
    try:
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        bond_indices = [mol.GetBondBetweenAtoms(int(a1), int(a2)).GetIdx() 
                        for (a1, a2), _ in brics_bonds if mol.GetBondBetweenAtoms(int(a1), int(a2))]
        if bond_indices:
            frag_mol = Chem.FragmentOnBonds(mol, bond_indices, addDummies=False)
            atom_sets.extend(Chem.GetMolFrags(frag_mol, asMols=False))
    except Exception:
        pass

    # 2. 环系统保留
    for ring in Chem.GetSymmSSSR(mol):
        atom_sets.append(tuple(ring))

    # 3. 极性/杂原子兜底
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ["Cl", "Br", "S", "P"] or atom.GetFormalCharge() != 0:
            atom_sets.append((atom.GetIdx(),))

    # 去重
    unique_sets = []
    seen = set()
    for s in atom_sets:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            unique_sets.append(list(s))

    return unique_sets

def mask_fragment(smiles, atoms_to_remove):
    """从分子中移除指定原子，返回修改后的 SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return "CCO" # 兜底值
    
    emol = Chem.EditableMol(mol)
    for idx in sorted(atoms_to_remove, reverse=True):
        emol.RemoveAtom(idx)
        
    new_mol = emol.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol, canonical=True)
    except:
        return "CCO" # 如果破坏了化学结构无法解析，返回兜底


# ==========================================
# 3. 核心预测与解释引擎
# ==========================================
def predict_and_explain(smiles, model, tokenizer, device, fp_dim=1024):
    """计算整体毒性，并通过边际贡献计算各个片段的解释性 (SHAP)"""
    # ---- A. 计算整体分子预测值 ----
    inputs = tokenizer([smiles], truncation=True, padding=True, return_tensors="pt").to(device)
    dummy_fp = torch.zeros((1, fp_dim), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], fp=dummy_fp)
        base_prob = torch.softmax(logits, dim=1)[0, 1].item()
    
    # ---- B. 拆分片段并计算边际贡献 ----
    atom_groups = split_smiles_into_substructures(smiles)
    interpretation = []
    atom_contribs = []
    
    mol = Chem.MolFromSmiles(smiles)
    
    for group in atom_groups:
        masked_smiles = mask_fragment(smiles, group)
        
        # 预测移除该片段后的毒性
        m_inputs = tokenizer([masked_smiles], truncation=True, padding=True, return_tensors="pt").to(device)
        m_dummy_fp = torch.zeros((1, fp_dim), dtype=torch.float32).to(device)
        with torch.no_grad():
            m_logits = model(input_ids=m_inputs["input_ids"], attention_mask=m_inputs["attention_mask"], fp=m_dummy_fp)
            m_prob = torch.softmax(m_logits, dim=1)[0, 1].item()
            
        # 边际贡献 = 整体毒性 - 移除片段后的毒性 (正值代表该片段增加了整体毒性)
        contrib = base_prob - m_prob
        
        frag_smiles = "N/A"
        if mol:
            try:
                frag_smiles = Chem.MolFragmentToSmiles(mol, group)
            except:
                pass
                
        interpretation.append({
            "fragment": frag_smiles,
            "atom_indices": group,
            "contribution_score": round(contrib, 4),
            "effect": "Increases Toxicity" if contrib > 0 else "Decreases Toxicity"
        })
        atom_contribs.append((group, contrib))
        
    return base_prob, interpretation, atom_contribs


# ==========================================
# 4. 可视化生成
# ==========================================
def generate_highlight_image(smiles, atom_contribs, output_path: Path):
    """根据边际贡献值生成 RDKit 分子高亮图"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol or not atom_contribs:
        return False
        
    # 如果 atom_contribs 是空列表，规避 numpy 报错
    if len(atom_contribs) == 0:
        return False

    all_vals = np.array([v for _, v in atom_contribs])
    max_val = np.max(np.abs(all_vals))
    norm = max_val if max_val > 0 else 1.0
    
    atom_colors = {}
    for atoms, val in atom_contribs:
        intensity = min(abs(val) / norm, 1.0)
        if val >= 0:
            # 增加毒性: 浅红色到深红色高亮
            color = (1.0, 1.0 - intensity, 1.0 - intensity)
        else:
            # 降低毒性: 浅蓝色到深蓝色高亮
            color = (1.0 - intensity, 1.0 - intensity, 1.0)
            
        for aidx in atoms:
            atom_colors[aidx] = color
            
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    drawer.DrawMolecule(mol, highlightAtoms=list(atom_colors.keys()), highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    
    with open(output_path, "wb") as f:
        f.write(drawer.GetDrawingText())
    return True


# ==========================================
# 5. 工具入口函数
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
            
        # 2. 从源码目录 (BASE_DIR) 加载模型与配置
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL_NAME_OR_PATH = BASE_DIR / "models/chemberta_zinc_base_v1"
        WEIGHTS_PATH = BASE_DIR / "models/best_chemberta_ToxCast_APR_HepG2.pth"
        
        if not MODEL_NAME_OR_PATH.exists():
            raise FileNotFoundError(f"找不到模型目录: {MODEL_NAME_OR_PATH}")
            
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_NAME_OR_PATH))
        model = ChemBERTaWithFingerprint(chemberta_model_name=str(MODEL_NAME_OR_PATH), fp_dim=1024, num_labels=2)
        
        if WEIGHTS_PATH.exists():
            model.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location=DEVICE))
        else:
            print(f"⚠️ 未找到权重 {WEIGHTS_PATH}，将使用未微调的基座模型！", file=sys.stderr)
            
        model.to(DEVICE)
        model.eval()
        
        # 3. 执行解释性预测
        base_prob, interpretation, atom_contribs = predict_and_explain(
            smiles, model, tokenizer, DEVICE, fp_dim=1024
        )
        
        # 4. 生成可视化图片 (存放到沙盒目录 cwd)
        image_path = cwd / "toxicity_interpretation.png"
        has_image = generate_highlight_image(smiles, atom_contribs, image_path)
        import base64
        # ⭐ 新增：如果生成了图片，转成 Base64 字符串
        image_base64 = None
        if has_image:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')

        # 5. 构造结果
        result["success"] = True
        result["summary"] = {
            "task": "Toxicity Prediction with Marginal Contribution (SHAP)",
            "toxicity_probability": round(base_prob, 4),
            "is_toxic": base_prob > 0.5,
        }
        result["results"] = {
            "smiles": smiles,
            "interpretation": interpretation,
            "image_base64": image_base64  # 把图片数据塞给 Agent
        }
        del result["error"]

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)

    # 6. 写回结果到沙盒目录
    with open(cwd / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if result.get("success"):
        print(f"🎉 毒性预测完成！预测为 {'Toxic' if result['summary']['is_toxic'] else 'Non-Toxic'}。")
        if result["summary"].get("interpretation_image_saved"):
            print("📸 已生成高亮解析图。")
    else:
        print(f"❌ 预测失败: {result.get('error')}", file=sys.stderr)


if __name__ == "__main__":
    main()