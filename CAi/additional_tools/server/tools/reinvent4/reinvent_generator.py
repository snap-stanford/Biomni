#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REINVENT4 多模式分子生成统一接口 (Unified REINVENT4 Generator)

功能介绍:
    本脚本提供统一的 Python 接口来调用 REINVENT4 的三种分子生成模式:

    1. De Novo 生成 (reinvent.prior):
       从头生成全新分子，无需输入骨架。

    2. Mol2Mol 骨架约束生成 (mol2mol_scaffold.prior):
       基于带手性标记的参考分子生成结构变体 (支持 @@，不支持 [*])。

    3. LibInvent 骨架装饰 (libinvent.prior):
       在含 [*] 连接点的骨架上生成 R-基团变体 (不支持 @@)。
"""

import json
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# ========== 路径常量 ==========
# 模型文件相对于本脚本所在目录（即工具目录）
_TOOL_DIR = Path(__file__).resolve().parent
REINVENT_PRIOR_PATH  = str(_TOOL_DIR / "REINVENT4" / "priors" / "reinvent.prior")
MOL2MOL_PRIOR_PATH   = str(_TOOL_DIR / "REINVENT4" / "priors" / "mol2mol_scaffold.prior")
LIBINVENT_PRIOR_PATH = str(_TOOL_DIR / "REINVENT4" / "priors" / "libinvent.prior")

VALID_MODES = ["de_novo", "mol2mol", "libinvent"]


@dataclass
class GenerationResult:
    """生成结果数据结构"""
    success: bool
    molecule_id: str
    input_smiles: Optional[str]
    output_file: Optional[str]
    num_generated: int
    num_valid: int
    error_message: Optional[str] = None
    reinvent_error: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class REINVENTGenerator:
    """REINVENT4 多模式生成功能封装"""

    def __init__(self, output_dir: str, device: str = "cuda:0"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.device = device
        self.results = []

    def _create_toml_config(self, mode: str, molecules: List[Dict],
                            batch_idx: int = 0) -> Tuple[str, List[str]]:
        temp_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if mode == "de_novo":
            toml_content = f"""
# REINVENT4 De Novo Generation
run_type = "sampling"
device = "{self.device}"
json_out_config = "_de_novo_{timestamp}_{batch_idx}.json"

[parameters]
model_file = "{REINVENT_PRIOR_PATH}"
output_file = '{self.output_dir}/de_novo_variants_{batch_idx}.csv'
num_smiles = {molecules[0].get('num_variants', 100) if molecules else 100}
unique_molecules = true
randomize_smiles = true
"""
        else:
            smiles_file = self.log_dir / f"input_{mode}_{timestamp}_{batch_idx}.smi"
            temp_files.append(str(smiles_file))

            with open(smiles_file, 'w') as f:
                for mol in molecules:
                    f.write(f"{mol.get('smiles', '')}\n")

            if mode == "mol2mol":
                mol = molecules[0] if molecules else {}
                toml_content = f"""
# REINVENT4 Mol2Mol Scaffold Generation
run_type = "sampling"
device = "{self.device}"
json_out_config = "_mol2mol_{timestamp}_{batch_idx}.json"

[parameters]
model_file = "{MOL2MOL_PRIOR_PATH}"
smiles_file = "{smiles_file}"
output_file = '{self.output_dir}/mol2mol_variants_{batch_idx}.csv'
num_smiles = {mol.get('num_variants', 50)}
sample_strategy = "{mol.get('strategy', 'beamsearch')}"
temperature = {mol.get('temperature', 1.0)}
unique_molecules = true
randomize_smiles = false
"""
            elif mode == "libinvent":
                mol = molecules[0] if molecules else {}
                toml_content = f"""
# REINVENT4 LibInvent Scaffold Decoration
run_type = "sampling"
device = "{self.device}"
json_out_config = "_libinvent_{timestamp}_{batch_idx}.json"

[parameters]
model_file = "{LIBINVENT_PRIOR_PATH}"
smiles_file = "{smiles_file}"
output_file = '{self.output_dir}/libinvent_variants_{batch_idx}.csv'
num_smiles = {mol.get('num_variants', 50)}
unique_molecules = true
randomize_smiles = false
"""

        toml_file = self.log_dir / f"config_{mode}_{timestamp}_{batch_idx}.toml"
        with open(toml_file, 'w') as f:
            f.write(toml_content)

        return str(toml_file), temp_files

    def run_reinvent(self, toml_file: str, log_file: str) -> Tuple[bool, str, str]:
        cmd = ["reinvent", "-l", log_file, toml_file]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            stderr_content = result.stderr if result.stderr else ""
            if result.returncode != 0:
                error_lines = stderr_content.strip().split('\n')
                key_error = error_lines[-1] if error_lines else "未知错误"
                return False, key_error, stderr_content
            return True, "OK", stderr_content
        except subprocess.TimeoutExpired:
            return False, "REINVENT 执行超时（超过1小时）", ""
        except FileNotFoundError:
            return False, "未找到 reinvent 命令，请确保 REINVENT4 已安装并在 PATH 中", ""
        except Exception as e:
            return False, f"执行异常: {str(e)}", str(e)

    def parse_results(self, csv_file: str, mode: str, input_mol: Dict) -> GenerationResult:
        mol_id = input_mol.get('id', 'unknown')
        input_smiles = input_mol.get('smiles')

        if not os.path.exists(csv_file):
            return GenerationResult(
                success=False, molecule_id=mol_id, input_smiles=input_smiles,
                output_file=None, num_generated=0, num_valid=0,
                error_message="未找到输出文件"
            )
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            num_valid = len(df)
            if num_valid == 0 or df.empty:
                return GenerationResult(
                    success=False, molecule_id=mol_id, input_smiles=input_smiles,
                    output_file=csv_file, num_generated=0, num_valid=0,
                    error_message="生成的分子全部为无效 SMILES（CSV为空）"
                )
            return GenerationResult(
                success=True, molecule_id=mol_id, input_smiles=input_smiles,
                output_file=csv_file,
                num_generated=input_mol.get('num_variants', 50),
                num_valid=num_valid, warnings=[]
            )
        except Exception as e:
            return GenerationResult(
                success=False, molecule_id=mol_id, input_smiles=input_smiles,
                output_file=csv_file, num_generated=0, num_valid=0,
                error_message=f"解析结果文件失败: {str(e)}"
            )

    def generate(self, config: Dict) -> Dict:
        mode = config.get("mode", "unknown")

        if "mode" not in config:
            return {"status": "failed", "mode": "unknown",
                    "error": "缺少必填字段: mode", "results": [],
                    "timestamp": datetime.now().isoformat()}

        if mode not in VALID_MODES:
            return {"status": "failed", "mode": mode,
                    "error": f"无效模式: {mode}. 支持的模式: {VALID_MODES}",
                    "results": [], "timestamp": datetime.now().isoformat()}

        molecules = config.get("molecules", [])
        if mode == "de_novo" and not molecules:
            molecules = [{"id": "de_novo_run", "num_variants": 100}]

        results = []
        for idx, mol in enumerate(molecules):
            mol_id = mol.get('id', f'molecule_{idx}')
            input_smiles = mol.get('smiles', '')
            try:
                toml_file, temp_files = self._create_toml_config(mode, [mol], idx)
                log_file = self.log_dir / f"run_{mode}_{mol_id}.log"
                success, error_msg, raw_stderr = self.run_reinvent(toml_file, str(log_file))

                if not success:
                    results.append(GenerationResult(
                        success=False, molecule_id=mol_id, input_smiles=input_smiles,
                        output_file=None, num_generated=0, num_valid=0,
                        error_message=f"REINVENT 错误: {error_msg}",
                        reinvent_error=raw_stderr
                    ))
                else:
                    output_csv = self.output_dir / f"{mode}_variants_{idx}.csv"
                    result = self.parse_results(str(output_csv), mode, mol)
                    if result.success and os.path.exists(str(output_csv)):
                        final_name = self.output_dir / f"{mol_id}_variants.csv"
                        shutil.move(str(output_csv), str(final_name))
                        result.output_file = str(final_name)
                    results.append(result)

                for f in temp_files:
                    if os.path.exists(f):
                        os.remove(f)
            except Exception as e:
                results.append(GenerationResult(
                    success=False, molecule_id=mol_id, input_smiles=input_smiles,
                    output_file=None, num_generated=0, num_valid=0,
                    error_message=f"脚本异常: {str(e)}"
                ))

        report = {
            "status": "completed" if any(r.success for r in results) else "failed",
            "mode": mode,
            "total_molecules": len(molecules),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "output_directory": str(self.output_dir),
            "results": [asdict(r) for r in results],
            "timestamp": datetime.now().isoformat()
        }

        report_file = self.output_dir / f"{mode}_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report


def generate_molecules(config: Dict, output_dir: Optional[str] = None) -> Dict:
    """便捷的函数接口"""
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)
    if output_dir:
        config["output_dir"] = output_dir

    generator = REINVENTGenerator(
        output_dir=config.get("output_dir", "./output"),
        device=config.get("device", "cuda:0")
    )
    return generator.generate(config)