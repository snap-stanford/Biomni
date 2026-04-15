#!/usr/bin/env python3

import json
import sys
import warnings
from pathlib import Path

# 屏蔽不必要的警告
warnings.filterwarnings("ignore")

try:
    from vina import Vina
except ImportError as e:
    print(f"导入 Vina 失败，请确保安装了 vina 库 (pip install vina)。\n错误信息: {e}", file=sys.stderr)
    sys.exit(1)


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

        # 2. 执行对接计算
        data = vina_docking(
            cwd_path=cwd,
            receptor_pdbqt_file=params["receptor_pdbqt_file"],
            ligand_pdbqt_file=params["ligand_pdbqt_file"],
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
