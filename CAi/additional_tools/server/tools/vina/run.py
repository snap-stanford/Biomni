#!/usr/bin/env python3

import json
import sys
import warnings
from pathlib import Path
import subprocess
import shutil

# 屏蔽不必要的警告
warnings.filterwarnings("ignore")

try:
    from vina import Vina
except ImportError as e:
    print(f"导入 Vina 失败，请确保安装了 vina 库 (pip install vina)。\n错误信息: {e}", file=sys.stderr)
    sys.exit(1)

def convert_structure_to_pdbqt(
    input_path: str,
    output_path: str,
    structure_role: str = "auto",
    overwrite: bool = True,
) -> str:
    """
    Convert a structure file to PDBQT format.

    Supported input formats:
        - .pdbqt  -> copied directly
        - .pdb    -> converted to .pdbqt
        - .sdf    -> converted to .pdbqt

    Args:
        input_path (str): Path to the input structure file.
        output_path (str): Path to the output PDBQT file.
        structure_role (str, optional): One of {"auto", "receptor", "ligand"}.
            - "receptor": prefer mk_prepare_receptor.py when possible
            - "ligand": prefer mk_prepare_ligand.py when possible
            - "auto": infer from filename and fall back as needed
        overwrite (bool, optional): Whether to overwrite output if it already exists.

    Returns:
        str: Absolute path to the generated PDBQT file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the input format is unsupported.
        RuntimeError: If conversion fails.
    """
    input_path_obj = Path(input_path).expanduser().resolve()
    output_path_obj = Path(output_path).expanduser().resolve()

    if not input_path_obj.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path_obj}")

    if not input_path_obj.is_file():
        raise ValueError(f"输入路径不是有效文件: {input_path_obj}")

    suffix = input_path_obj.suffix.lower()
    if suffix not in {".pdbqt", ".pdb", ".sdf"}:
        raise ValueError(
            f"不支持的输入格式: {suffix}。仅支持 .pdbqt、.pdb、.sdf"
        )

    if structure_role not in {"auto", "receptor", "ligand"}:
        raise ValueError(
            "structure_role 必须是 'auto'、'receptor' 或 'ligand'"
        )

    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if output_path_obj.exists():
        if overwrite:
            output_path_obj.unlink()
        else:
            raise FileExistsError(f"输出文件已存在: {output_path_obj}")

    # 1) 已经是 pdbqt，直接复制
    if suffix == ".pdbqt":
        shutil.copy2(input_path_obj, output_path_obj)
        return str(output_path_obj)

    # 2) 优先使用 Meeko
    # receptor: mk_prepare_receptor.py
    # ligand:   mk_prepare_ligand.py
    meeko_receptor = shutil.which("mk_prepare_receptor.py")
    meeko_ligand = shutil.which("mk_prepare_ligand.py")
    obabel = shutil.which("obabel")

    inferred_role = structure_role
    if inferred_role == "auto":
        name_lower = input_path_obj.name.lower()
        if "receptor" in name_lower or "protein" in name_lower:
            inferred_role = "receptor"
        elif "ligand" in name_lower:
            inferred_role = "ligand"
        else:
            # 保守默认：SDF 更像 ligand，PDB 更像 receptor
            inferred_role = "ligand" if suffix == ".sdf" else "receptor"

    # ---- receptor branch ----
    if inferred_role == "receptor" and meeko_receptor:
        # mk_prepare_receptor.py 通常用 basename 输出，-p 生成 pdbqt
        # 为了精确控制输出文件名，这里先输出到 basename，再重命名
        tmp_basename = output_path_obj.with_suffix("")
        cmd = [
            meeko_receptor,
            "-i", str(input_path_obj),
            "-o", str(tmp_basename),
            "-p",
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode == 0:
            # 常见输出：basename.pdbqt 或 basename_rigid.pdbqt
            candidate_files = [
                tmp_basename.with_suffix(".pdbqt"),
                Path(str(tmp_basename) + "_rigid.pdbqt"),
            ]
            for candidate in candidate_files:
                if candidate.exists():
                    shutil.move(str(candidate), str(output_path_obj))
                    return str(output_path_obj)

        # 如果 Meeko receptor 失败，再回退到 obabel
        # 不在这里立刻 raise，让 fallback 继续

    # ---- ligand branch ----
    if inferred_role == "ligand" and meeko_ligand and suffix == ".sdf":
        cmd = [
            meeko_ligand,
            "-i", str(input_path_obj),
            "-o", str(output_path_obj),
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode == 0 and output_path_obj.exists():
            return str(output_path_obj)
        # 同样失败则回退到 obabel

    # 3) Open Babel fallback
    if obabel:
        in_fmt = suffix.lstrip(".")
        cmd = [
            obabel,
            f"-i{in_fmt}", str(input_path_obj),
            "-opdbqt",
            "-O", str(output_path_obj),
        ]

        # 常见情况下保留氢更稳
        if suffix in {".pdb", ".sdf"}:
            cmd.append("-h")

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode == 0 and output_path_obj.exists() and output_path_obj.stat().st_size > 0:
            return str(output_path_obj)

        raise RuntimeError(
            "Open Babel 转换失败。\n"
            f"命令: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    # 4) 没有可用工具
    raise RuntimeError(
        "未找到可用的转换工具。请安装 Meeko 或 Open Babel。\n"
        "需要的命令之一：mk_prepare_receptor.py、mk_prepare_ligand.py、obabel"
    )

ALLOWED_INPUT_SUFFIXES = {".pdbqt", ".sdf", ".pdb"}


def resolve_input_path(file_path, cwd_path):
    path_obj = Path(file_path)
    if not path_obj.is_absolute():
        path_obj = cwd_path / path_obj
    return path_obj


def validate_existing_file(file_path, cwd_path, param_name):
    if not file_path:
        raise ValueError(f"必须提供 '{param_name}' 参数。")
    if not isinstance(file_path, str):
        raise ValueError(f"'{param_name}' 必须是字符串路径。")

    path_obj = resolve_input_path(file_path, cwd_path)

    if not path_obj.exists():
        raise FileNotFoundError(f"{param_name} 指向的文件不存在: {path_obj}")
    if not path_obj.is_file():
        raise ValueError(f"{param_name} 不是一个有效文件: {path_obj}")

    return path_obj


def prepare_pdbqt_file(input_file, cwd_path, param_name, output_filename):
    """
    将输入结构文件标准化为 pdbqt。
    支持输入格式：
    - .pdbqt：直接使用
    - .sdf：转换为 pdbqt
    - .pdb：转换为 pdbqt
    """
    input_path = validate_existing_file(input_file, cwd_path, param_name)
    suffix = input_path.suffix.lower()

    if suffix not in ALLOWED_INPUT_SUFFIXES:
        raise ValueError(
            f"{param_name} 文件格式不受支持: {input_path.name}。"
            "仅支持 .pdbqt、.sdf 或 .pdb 格式。"
        )

    if suffix == ".pdbqt":
        return input_path

    output_path = cwd_path / output_filename

    # 统一调用格式转换函数
    convert_structure_to_pdbqt(str(input_path), 
                               str(output_path),
                               structure_role="receptor" if param_name == "receptor_file" else "ligand")

    if not output_path.exists():
        raise RuntimeError(
            f"{param_name} 已尝试从 {suffix} 转换为 pdbqt，但未生成输出文件。"
        )

    return output_path

def vina_docking(
    cwd_path, receptor_pdbqt_file, ligand_pdbqt_file, center, box_size, exhaustiveness=32, n_poses=20, sf_name="vina"
):
    """
    使用 AutoDock Vina 进行分子对接。
    ...
    """
    # 1. 安全地处理文件路径 (相对于当前的沙盒目录 cwd)
    receptor_path = Path(receptor_pdbqt_file)
    if not receptor_path.is_absolute():
        receptor_path = cwd_path / receptor_path

    ligand_path = Path(ligand_pdbqt_file)
    if not ligand_path.is_absolute():
        ligand_path = cwd_path / ligand_path

    if not receptor_path.exists():
        raise FileNotFoundError(f"受体文件不存在: {receptor_path}")
    if not ligand_path.exists():
        raise FileNotFoundError(f"配体文件不存在: {ligand_path}")

    # 2. 定义输出文件路径 (保存在沙盒目录)
    output_path = cwd_path / "docked_poses.pdbqt"
    minimized_path = cwd_path / "minimized_pose.pdbqt"

    # 3. 初始化 Vina
    v = Vina(sf_name=sf_name)
    v.set_receptor(str(receptor_path))
    v.set_ligand_from_file(str(ligand_path))
    v.compute_vina_maps(center=center, box_size=box_size)

    # 4. 初始打分
    energy = v.score()
    score_before = float(energy[0])

    # 5. 局部最小化 (Minimization)
    energy_minimized = v.optimize()
    score_after = float(energy_minimized[0])
    v.write_pose(str(minimized_path), overwrite=True)

    # 6. 全局对接 (Docking)
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    v.write_poses(str(output_path), n_poses=5, overwrite=True)

    # 尝试获取对接后的最优构象打分
    try:
        energies = v.energies()
        best_docking_score = float(energies[0][0]) if len(energies) > 0 else None
    except Exception:
        best_docking_score = None

    return {
        "score_before_minimization": round(score_before, 3),
        "score_after_minimization": round(score_after, 3),
        "best_docking_score": round(best_docking_score, 3) if best_docking_score else None,
        "minimized_pose_file": str(minimized_path),
        "docked_poses_file": str(output_path),
    }


def main():
    result_payload = {"success": False, "summary": {}, "results": {}, "error": None}
    cwd = Path.cwd()

    try:
        # 1. 从沙盒目录读取 params.json
        params_file = cwd / "params.json"
        if not params_file.exists():
            raise FileNotFoundError("当前沙盒目录下未找到 params.json")

        with open(params_file, encoding="utf-8") as f:
            params = json.load(f)
            
        # 文件存在检查
        # 必要输入参数
        receptor_file = params.get("receptor_file")
        ligand_file = params.get("ligand_file")

        if not receptor_file:
            raise ValueError("必须提供 'receptor_file' 参数。")
        if not ligand_file:
            raise ValueError("必须提供 'ligand_file' 参数。")
        receptor_pdbqt_path = prepare_pdbqt_file(
            input_file=receptor_file,
            cwd_path=cwd,
            param_name="receptor_file",
            output_filename="prepared_receptor.pdbqt",
        )
        ligand_pdbqt_path = prepare_pdbqt_file(
            input_file=ligand_file,
            cwd_path=cwd,
            param_name="ligand_file",
            output_filename="prepared_ligand.pdbqt",
        )
        

        # 2. 执行对接计算
        data = vina_docking(
            cwd_path=cwd,
            receptor_pdbqt_file= str(receptor_pdbqt_path),
            ligand_pdbqt_file=str(ligand_pdbqt_path),
            center=params["center"],
            box_size=params["box_size"],
            exhaustiveness=params.get("exhaustiveness", 32),
            n_poses=params.get("n_poses", 20),
            sf_name=params.get("sf_name", "vina"),
        )

        # 3. 组装标准返回值
        result_payload["success"] = True
        result_payload["summary"] = {
            "task": "Molecular Docking (AutoDock Vina)",
            "best_docking_score": data["best_docking_score"],
            "score_after_minimization": data["score_after_minimization"],
        }
        result_payload["results"] = data
        del result_payload["error"]

    except Exception as e:
        result_payload["success"] = False
        result_payload["error"] = str(e)

    # 4. 将结果写回沙盒目录
    with open(cwd / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    if result_payload.get("success"):
        print(
            f"🎉 对接完成！最佳对接打分 (Best Docking Score): {result_payload['summary']['best_docking_score']} kcal/mol"
        )
    else:
        print(f"❌ 工具运行失败: {result_payload.get('error')}", file=sys.stderr)


if __name__ == "__main__":
    main()
