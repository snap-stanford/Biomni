import json
import os
import time
from typing import Any

import requests


TOOL_SERVER_HOST = os.environ.get("TOOL_SERVER_HOST", "100.103.118.72")
TOOL_SERVER_PORT = os.environ.get("TOOL_SERVER_PORT", "8001")
BASE_URL = f"http://{TOOL_SERVER_HOST}:{TOOL_SERVER_PORT}"


def _call_worker_api(
    tool_name: str, payload: dict[str, Any], action: str = "default", timeout_mins: int = 5
) -> dict[str, Any]:
    """
    通用底层封装：向服务器提交任务并轮询等待结果。
    这个函数对 Agent 是隐藏的，只供上层工具函数调用。
    """
    run_url = f"{BASE_URL}/run/{tool_name}/{action}"
    job_url = f"{BASE_URL}/job"

    # 🌟 核心修复：显式声明不使用任何系统代理，防止内网 IP 被发往外网代理服务器导致 502
    bypass_proxies = {"http": None, "https": None}

    try:
        # 1. 提交任务
        r = requests.post(run_url, json=payload, timeout=10, proxies=bypass_proxies)
        r.raise_for_status()
        data = r.json()

        if "error" in data:
            return {"error": f"Task submission failed: {data['error']}"}

        job_id = data["job_id"]

        # 2. 轮询结果
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout_mins * 60:
                return {"error": f"Timeout: Task did not complete within {timeout_mins} minutes."}

            r = requests.get(f"{job_url}/{job_id}", timeout=10, proxies=bypass_proxies)
            status = r.json()
            state = status.get("status")

            if state == "running":
                time.sleep(3)
                continue
            elif state == "failed":
                return {"error": f"Server execution crashed: {status.get('data')}"}
            elif state == "finished":
                # 获取 result.json 中的数据
                result = status.get("data") or status.get("stdout")

                # 兼容旧接口的字符串形式输出
                if isinstance(result, str):
                    try:
                        result = json.loads(result.replace("'", '"'))
                    except json.JSONDecodeError:
                        return {"error": "Failed to parse string output into JSON.", "raw": result}

                if not result:
                    return {"error": "Task finished but returned no data."}

                if isinstance(result, dict) and "success" in result and not result["success"]:
                    return {"error": f"Tool execution failed: {result.get('error', 'Unknown error')}"}

                return result
            else:
                return {"error": f"Unknown state: {state}"}

    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP Error ({e.response.status_code}): {e.response.text}"}
    except Exception as e:
        return {"error": f"Network or API error: {str(e)}"}


# ==========================================
# 🛠️ Agent 工具 1：骨架衍生生成 -- update_v1
# ==========================================
def generate_scaffold_analogs(smiles: str, num_analogs: int = 10) -> str:
    """
    Tool Name:
        scaffold_based_analog_generation

    Description:
        Generate novel molecular analogs from a scaffold SMILES using a pre-trained RNN-based scaffold generation model.

    When to use:
        Use this tool when scaffold-based analog generation is needed and the user provides a scaffold SMILES with an explicit growth point.

    Do not use when:
        - The input is not a scaffold SMILES.
        - The SMILES does not contain '*'.
        - The SMILES contains '@@' stereochemical annotations.

    Inputs:
        smiles (str):
            Scaffold SMILES for analog generation.
            Must contain at least one '*' character indicating the attachment/growth point.
            Example: 'c1ccccc1*'
            SMILES containing '@@' are not supported.

        num_analogs (int, optional):
            Number of analogs to generate.
            Default: 10.
            Must be a positive integer.

    Output:
        JSON-formatted string.

        Success:
        {
            "status": "success",
            "generated_count": 42,
            "molecules": [
                "generated_smiles_1",
                "generated_smiles_2"
            ]
        }

        Failure:
        {
            "error": "Detailed error message"
        }
    Notes:
        - Returned molecules are valid and deduplicated analogs.
        - The actual number of generated molecules may be smaller than the requested number.
    """
    if "*" not in smiles:
        return json.dumps(
            {"error": "The input scaffold SMILES must contain at least one '*' character as the growth point."}
        )

    payload = {"smiles": smiles, "num_analogs": num_analogs}
    result = _call_worker_api("scaffold", payload)

    if "error" in result:
        return json.dumps(result)

    summary = result.get("summary", {})
    generated_smiles = [item["smiles"] for item in result.get("results", [])]

    agent_response = {
        "status": "success",
        "generated_count": summary.get("valid_unique_generated"),
        "molecules": generated_smiles,
    }

    return json.dumps(agent_response, ensure_ascii=False)


# ==========================================
# 🛠️ Agent 工具 2：毒性预测与 SHAP 解释 -- update_v1
# ==========================================
def predict_molecule_toxicity(smiles: str) -> str:
    """
    Use this tool to predict the toxicity based on the label of HepG2 from toxicast dataset of a complete molecule and provide a SHAP-style structural interpretation based on substructure-level contribution analysis.

    This tool performs whole-molecule toxicity prediction and then explains the prediction by estimating the contribution of molecular substructures/fragments. If available, it also returns a visualization image encoded as Base64.

    Args:
        smiles (str): The valid SMILES string of the complete molecule to be evaluated.

    Returns:
        Returns:
        str: A JSON-formatted string.

        Success output:
        {
            "verdict": "Toxic",  # or "Non-Toxic"
            "toxicity_probability": 0.8521,
            "structural_explanation": [
                {
                    "fragment": "c1ccccc1",
                    "contribution_score": 0.0521,
                    "effect": "Increases Toxicity"
                }
            ],
            "image_saved_at": "/absolute/path/to/latest_toxicity_explanation.png",
            "vision_prompt": "The structure interpretation image has been saved..."
        }

        Error output:
        {
            "error": "Detailed error message"
        }
    Notes:
        - Use this tool only for toxicity prediction of a complete molecule, not for scaffold-only input.
        - The returned toxicity_probability is the predicted probability that the molecule is toxic.
        - The field is_toxic is determined by whether toxicity_probability > 0.5.
        - The interpretation field contains fragment/substructure-level contribution results derived from SHAP-style marginal contribution analysis.
        - Positive contribution scores indicate fragments that increase predicted toxicity; negative contribution scores indicate fragments that decrease predicted toxicity.
        - If image_base64 is not null, it contains a Base64-encoded structural interpretation image that can be rendered or summarized by a vision-capable system.
    """
    payload = {"smiles": smiles}
    result = _call_worker_api("toxicity", payload)

    if "error" in result:
        return json.dumps({"error": result["error"]})

    summary = result.get("summary", {})
    results_data = result.get("results", {})

    agent_response = {
        "verdict": "Toxic" if summary.get("is_toxic") else "Non-Toxic",
        "toxicity_probability": summary.get("toxicity_probability"),
        "structural_explanation": results_data.get("interpretation", []),
    }

    # ⭐ 核心：处理服务器传回来的图片，适配沙盒环境
    image_base64 = results_data.get("image_base64")
    if image_base64:
        import base64
        import os

        # 方案 A：直接把图片保存在 Agent 当前的代码执行目录下
        # 很多高级 Agent 沙盒（如 OpenHands/E2B）会自动监听工作区的变化，将新图片传给视觉模型
        local_filename = "latest_toxicity_explanation.png"
        with open(local_filename, "wb") as f:
            f.write(base64.b64decode(image_base64))

        agent_response["image_saved_at"] = os.path.abspath(local_filename)
        agent_response["vision_prompt"] = (
            "The structure interpretation image has been saved locally. Please look at the image to analyze the toxic fragments."
        )

    return json.dumps(agent_response, ensure_ascii=False)


# ==========================================
# 🛠️ Agent 工具 3：SCScore 计算
# ==========================================
def calculate_scscore(
    smiles: str | None = None, smiles_list: list[str] | None = None, model_type: str = "1024bool"
) -> str:
    """
    Estimate the synthetic accessibility of molecules using the SCScore model.

    SCScore predicts how difficult a molecule is to synthesize from commercially
    available starting materials. The score ranges roughly from 1 (easy synthesis)
    to 5 (very difficult synthesis).

    Args:
        smiles (str, optional):
            A single SMILES string representing a molecule.
        smiles_list (list[str], optional):
            A list of SMILES strings for batch evaluation.
        model_type (str, optional):
            SCScore fingerprint model to use. Default is "1024bool".

    Returns:
        str: A JSON string strictly following this schema:

        SUCCESS:
        {
            "success": true,
            "summary": {
                "total": 2,
                "successful": 2,
                "failed": 0,
                "avg_scscore": 1.22,
                "min_scscore": 1.20,
                "max_scscore": 1.25
            },
            "results": [
                {
                    "input_smiles": "c1ccccc1",
                    "canonical_smiles": "c1ccccc1",
                    "scscore": 1.20,
                    "interpretation": "very easy synthesis"
                }
            ],
            "errors": null  # or a list of dictionaries if some molecules failed
        }

        ERROR (API level):
        {
            "success": false,
            "error": "Detailed error message"
        }
    """
    if smiles:
        smiles_list = [smiles]

    if not smiles_list:
        return json.dumps({"success": False, "error": "smiles or smiles_list must be provided"})

    payload = {
        "smiles_list": smiles_list,
        "model_type": model_type,
    }

    result = _call_worker_api("scscore", payload)

    # 统一将结果转为 JSON 字符串返回给 Agent
    return json.dumps(result, ensure_ascii=False)


# ==========================================
# 🛠️ Agent 工具 4：Lib-INVENT 骨架修饰 (Scaffold Decoration)
# ==========================================
def generate_libinvent_decorations(smiles: str, num_decorations: int = 3) -> str:
    """
    Use this tool to decorate a chemical scaffold using the Lib-INVENT reaction-based model.
    It generates decorated molecules by attaching substituents or side chains to the scaffold's
    attachment points.

    Args:
        smiles (str):
            The scaffold SMILES string to decorate.
            The scaffold MUST contain at least one valid attachment point, such as '[*]' or '[*:1]'.
            Example: '[*]c1ccccc1' or 'CC(=O)N[*]'.
            The input scaffold SMILES could NOT contain '@@'. 
        num_decorations (int, optional):
            The requested number of decorated molecules to generate. Defaults to 3.
            The actual number of successfully generated molecules may be smaller than this value,
            depending on scaffold validity and Lib-INVENT generation outcomes.

    Returns:
        str: A JSON-formatted string.

        On SUCCESS, the JSON follows this schema:
        {
            "status": "success",
            "input_scaffold": "O=C1N(C(=O)[*])CCSC(=O)C1(C)C",
            "requested_num_decorations": 10,
            "generated_count": 10,
            "csv_columns": ["SMILES", "status", "message"],
            "molecules_smiles": [...],
            "decorated_molecules_preview": [...]
        }

        On ERROR, the JSON follows this schema:
        {
            "error": "Detailed error message"
        }
    """
    if "*" not in smiles:
        return json.dumps(
            {"error": "The input scaffold SMILES must contain an attachment point (like '*' or '[*:1]')."}
        )
    if "@@" in smiles:
        return json.dumps(
            {"error": "The input scaffold SMILES could not contain '@@'. (like '@@' or '@@H')."}
        )
    payload = {"smiles": smiles, "number_of_decorations_per_scaffold": num_decorations}

    result = _call_worker_api("libinvent", payload)

    if "error" in result:
        return json.dumps({"error": result["error"]})

    summary = result.get("summary", {})
    results = result.get("results", [])

    columns = summary.get("columns", [])
    preview_rows = summary.get("preview", [])

    molecules_smiles = [row.get("SMILES") for row in results if row.get("SMILES")]

    input_scaffold = None
    if results:
        input_scaffold = results[0].get("input_scaffold")

    agent_response = {
        "status": "success",
        "input_scaffold": input_scaffold,
        "requested_num_decorations": num_decorations,
        "generated_count": summary.get("row_count"),
        "csv_columns": columns,
        "molecules_smiles": molecules_smiles,
        "decorated_molecules_preview": preview_rows,
    }

    return json.dumps(agent_response, ensure_ascii=False)


# ==========================================
# 🛠️ Agent 工具 5：抗菌活性 (pMIC) 预测 -- update_v1
# ==========================================
def predict_antibacterial_pmic(smiles: str) -> str:
    """
    Use this tool to predict the antibacterial activity of a complete molecule using a Chemprop-based MPNN model trained for pMIC prediction.

    This tool takes a valid molecular SMILES as input and predicts its antibacterial potency. It returns the predicted pMIC value and the corresponding estimated MIC value in µM. 
    Higher pMIC values and lower MIC_uM values indicate stronger predicted antibacterial activity.

    Args:
        smiles (str): The valid SMILES string of the complete molecule to be evaluated.

    Returns:
        str: A JSON-formatted string.

        Success output:
        {
            "status": "success",
            "pMIC_value": 6.42,
            "estimated_MIC_uM": 0.38,
            "interpretation": "Higher pMIC means stronger activity..."
        }

        Error output:
        {
            "error": "Detailed error message"
        }
    Notes:
        - Use this tool only for antibacterial activity prediction of a complete molecule.
        - Input must be a valid SMILES string of the full molecule.
        - The returned pMIC_value is the model-predicted antibacterial activity score.
        - The returned estimated_MIC_uM is the estimated minimum inhibitory concentration in µM.
        - Higher pMIC values generally indicate stronger predicted antibacterial activity.
        - Lower estimated_MIC_uM values generally indicate stronger predicted antibacterial activity.
    """
    payload = {"smiles": smiles}
    result = _call_worker_api("pmic", payload)

    if "error" in result:
        return json.dumps({"error": result["error"]})

    summary = result.get("summary", {})

    agent_response = {
        "status": "success",
        "pMIC_value": summary.get("pMIC_value"),
        "estimated_MIC_uM": summary.get("estimated_MIC_uM"),
        "interpretation": "Higher pMIC means stronger activity. A typical threshold for active compounds is often pMIC > 5.0 (MIC < 10 µM).",
    }

    return json.dumps(agent_response, ensure_ascii=False)


# ==========================================
# 🛠️ Agent 工具 6：RxnFlow 靶点口袋导向分子生成 --update_v1
# ==========================================
def generate_molecules_for_pocket(
    protein_pdb_path: str, center_xyz: list = None, ref_ligand_path: str = None, num_samples: int = 10
) -> str:
    """
    Use this tool to perform target-aware zero-shot molecular generation with RxnFlow.

    This tool generates candidate molecules for a protein target using either:
    1. a protein structure file plus a binding pocket center, or
    2. a protein structure file plus a reference ligand file.

    Required inputs:
        - protein_pdb_path
        - one of: center or ref_ligand_path

    Args:
        protein_pdb_path (str): Path to the target protein structure file.
            Supported formats: .sdf, .mol2, .pdb

        center (list | tuple | str | dict, optional): Binding pocket center coordinates.
            Accepted formats include:
            - [x, y, z]
            - (x, y, z)
            - "x,y,z"
            - {"x": x, "y": y, "z": z}
            This field is optional only if ref_ligand_path is provided.

        ref_ligand_path (str, optional): Path to the reference ligand structure file.
            Supported formats: .sdf, .mol2, .pdb
            This field is optional only if center is provided.

        num_samples (int, optional): Number of molecules to generate. Default: 100.
        env_dir (str, optional): Environment directory for RxnFlow. Uses the default internal setting if not provided.
        model_path (str, optional): Model checkpoint path. Uses the default internal setting if not provided.
        temperature (str, optional): Sampling temperature setting. Default: "uniform-16-64".
        use_cuda (bool, optional): Whether to use CUDA if available. Default: True.
        save_reward (bool, optional): Whether to calculate and save reward-related scores. Default: True.

    Returns:
        str: A JSON-formatted string.

        Success output:
        {
            "status": "success",
            "generated_count": 100,
            "sampling_time_sec": 12.345,
            "full_results_csv_path": "/sandbox/path/rxnflow_results.csv",
            "top_molecules_preview": [
                {
                    "smiles": "CCO...",
                    "qed": 0.812,
                    "proxy_score": -7.231
                }
            ]
        }

        Error output:
        {
            "error": "Detailed error message"
        }

    Notes:
        - protein_pdb_path is always required.
        - At least one of center or ref_ligand_path must be provided.
        - If both center and ref_ligand_path are missing, the tool will return an error.
        - protein_pdb_path and ref_ligand_path must be structure files in .sdf, .mol2, or .pdb format.
        - The actual output file may be a CSV file (if save_reward=True) or an SMI file (if save_reward=False).
        - generated_preview contains only a small preview of the generated molecules, not the full result set.
    """
    if not center_xyz and not ref_ligand_path:
        return json.dumps(
            {
                "error": 'You must provide either "center_xyz" coordinates OR a "ref_ligand_path" to define the binding pocket.'
            }
        )

    payload = {"protein_pdb_path": protein_pdb_path, "num_samples": num_samples, "save_reward": True}
    if center_xyz:
        payload["center"] = center_xyz
    if ref_ligand_path:
        payload["ref_ligand_path"] = ref_ligand_path

    # 这个工具跑得比较慢，允许 15 分钟超时
    result = _call_worker_api("rxnflow", payload, timeout_mins=15)

    summary = result.get("summary", {})
    results_data = result.get("results", {})

    agent_response = {
        "status": "success",
        "generated_count": summary.get("generated_count"),
        "sampling_time_sec": summary.get("sampling_time_sec"),
        "full_results_csv_path": summary.get("output_file"),
        "top_molecules_preview": results_data.get("generated_preview", []),
    }

    return json.dumps(agent_response, ensure_ascii=False)


# ==========================================
# 🛠️ Agent 工具 7：AutoDock Vina 分子对接 --update_v1
# ==========================================
def perform_molecular_docking_vina(
    receptor_pdbqt_path: str, ligand_pdbqt_path: str, center_xyz: list, box_size_xyz: list, exhaustiveness: int = 32
) -> str:
    """
    Use this tool to perform molecular docking of a small-molecule ligand into a protein receptor using AutoDock Vina.

    This tool accepts receptor and ligand structure files, automatically converts supported input formats to PDBQT when needed, and then runs docking with AutoDock Vina. It returns docking scores and the generated docking result files.

    Required inputs:
        - receptor_file
        - ligand_file
        - center
        - box_size

    Args:
        receptor_file (str): Path to the receptor structure file.
            Supported input formats: .pdbqt, .pdb, .sdf
            If the input is .pdb or .sdf, it will be converted to .pdbqt before docking.

        ligand_file (str): Path to the ligand structure file.
            Supported input formats: .pdbqt, .pdb, .sdf
            If the input is .pdb or .sdf, it will be converted to .pdbqt before docking.

        center (list | tuple | str | dict): The docking box center coordinates.

        box_size (list | tuple | str | dict): The docking box dimensions in Angstroms.

        exhaustiveness (int, optional): Exhaustiveness of the global search. Default: 32.
        n_poses (int, optional): Number of docking poses to generate. Default: 20.
        sf_name (str, optional): Scoring function name. Default: "vina".

    Returns:
        str: A JSON-formatted string.

        Success output:
        {
            "status": "success",
            "best_docking_score_kcal_mol": -8.4,
            "minimized_score_kcal_mol": -7.9,
            "docked_poses_file_path": "/path/to/docked_poses.pdbqt",
            "minimized_pose_file_path": "/path/to/minimized_pose.pdbqt",
            "interpretation": "More negative scores indicate stronger binding affinity."
        }

        Error output:
        {
            "error": "Detailed error message"
        }

    Notes:
        - receptor_file and ligand_file are both required.
        - Input files must exist before docking starts.
        - Supported input formats for both receptor_file and ligand_file are .pdbqt, .pdb, and .sdf.
        - If the input file is not already in .pdbqt format, the tool will automatically convert it to .pdbqt before docking.
        - More negative docking scores generally indicate stronger predicted binding affinity.
        - best_docking_score is the best score among sampled docking poses.
        - score_after_minimization is the score after local minimization.
    """
    payload = {
        "receptor_file": receptor_pdbqt_path,
        "ligand_file": ligand_pdbqt_path,
        "center": center_xyz,
        "box_size": box_size_xyz,
        "exhaustiveness": 32,
        "n_poses": 20,
    }

    # 分子对接可能非常耗时，特别是 exhaustiveness > 32 时，设置 20 分钟超时
    result = _call_worker_api("vina", payload, timeout_mins=20)

    if "error" in result:
        return json.dumps({"error": result["error"]})

    summary = result.get("summary", {})
    results_data = result.get("results", {})

    agent_response = {
        "status": "success",
        "best_docking_score_kcal_mol": summary.get("best_docking_score"),
        "minimized_score_kcal_mol": summary.get("score_after_minimization"),
        "docked_poses_file_path": results_data.get("docked_poses_file"),
        "minimized_pose_file_path": results_data.get("minimized_pose_file"), 
        "interpretation": "More negative scores indicate stronger binding affinity.",
    }

    return json.dumps(agent_response, ensure_ascii=False)


# ==========================================
# 🛠️ Agent 工具: REINVENT4 De Novo 从头分子生成
# ==========================================
def generate_molecules_reinvent4_denovo(num_variants: int = 100) -> str:
    """
    Use this tool to generate completely novel molecules from scratch using the REINVENT4 de novo prior model.
    No input scaffold or reference molecule is needed. Suitable for exploring vast chemical space.

    Args:
        num_variants (int, optional): The number of novel molecules to generate. Defaults to 100.

    Returns:
        str: A JSON string.
        SUCCESS:
        {
            "status": "success",
            "generated_count": 95,
            "molecules_smiles": ["CCO", "c1ccccc1", ...]
        }
        ERROR:
        {
            "error": "Detailed error message"
        }
    """
    payload = {"num_variants": num_variants}
    result = _call_worker_api("reinvent4", payload, action="de_novo", timeout_mins=10)

    if "error" in result:
        return json.dumps({"error": result["error"]})

    summary = result.get("summary", {})
    molecules_data = result.get("results", {}).get("molecules", [])
    smiles_list = [mol["smiles"] for mol in molecules_data if mol.get("smiles")]

    agent_response = {
        "status": "success",
        "generated_count": summary.get("generated_count", len(smiles_list)),
        "molecules_smiles": smiles_list
    }
    return json.dumps(agent_response, ensure_ascii=False)


# ==========================================
# 🛠️ Agent 工具: REINVENT4 LibInvent 骨架装饰
# ==========================================
def generate_molecules_reinvent4_libinvent(smiles: str, num_variants: int = 50) -> str:
    """
    Use this tool to decorate a chemical scaffold by generating R-group variants at [*] attachment points
    using the REINVENT4 LibInvent model.
    The input MUST be a scaffold SMILES containing at least one [*] wildcard (e.g., 'c1ccc([*])cc1').
    This mode does NOT support chiral annotations (@@). Use mol2mol mode for chiral molecules instead.

    Args:
        smiles (str): A scaffold SMILES string containing [*] attachment points.
        num_variants (int, optional): Number of decorated variants to generate. Defaults to 50.

    Returns:
        str: A JSON string.
        SUCCESS:
        {
            "status": "success",
            "input_scaffold": "c1ccc([*])cc1",
            "generated_count": 48,
            "molecules_smiles": ["c1ccc(N)cc1", "c1ccc(O)cc1", ...]
        }
        ERROR:
        {
            "error": "Detailed error message"
        }
    """
    if "[*]" not in smiles and "*" not in smiles:
        return json.dumps({"error": "The input scaffold SMILES must contain at least one [*] attachment point."})

    payload = {"smiles_list": [smiles], "num_variants": num_variants}
    result = _call_worker_api("reinvent4", payload, action="libinvent", timeout_mins=10)

    if "error" in result:
        return json.dumps({"error": result["error"]})

    summary = result.get("summary", {})
    molecules_data = result.get("results", {}).get("molecules", [])
    smiles_list = [mol["smiles"] for mol in molecules_data if mol.get("smiles")]

    agent_response = {
        "status": "success",
        "input_scaffold": smiles,
        "generated_count": summary.get("generated_count", len(smiles_list)),
        "molecules_smiles": smiles_list
    }
    return json.dumps(agent_response, ensure_ascii=False)


# ==========================================
# 🛠️ Agent 工具: REINVENT4 Mol2Mol 手性约束分子生成
# ==========================================
def generate_molecules_reinvent4_mol2mol(smiles: str, num_variants: int = 50, strategy: str = "beamsearch", temperature: float = 1.0) -> str:
    """
    Use this tool to generate structural analogs of a reference molecule while preserving its stereochemistry
    using the REINVENT4 Mol2Mol model.
    The input should be a complete SMILES string (supports @@ chiral annotations).
    This mode does NOT support [*] wildcards. Use libinvent mode for scaffold decoration instead.

    Args:
        smiles (str): A complete reference molecule SMILES (e.g., 'CC1(C)S[C@@H]2NC(=O)C(=O)N2[C@H]1C(=O)O').
        num_variants (int, optional): Number of analogs to generate. Defaults to 50.
        strategy (str, optional): Sampling strategy, 'beamsearch' or 'multinomial'. Defaults to 'beamsearch'.
        temperature (float, optional): Sampling temperature. Defaults to 1.0.

    Returns:
        str: A JSON string.
        SUCCESS:
        {
            "status": "success",
            "input_smiles": "CC1(C)S[C@@H]2NC(=O)C(=O)N2[C@H]1C(=O)O",
            "generated_count": 45,
            "molecules_smiles": ["CC1(C)SC2NC(=O)...", ...]
        }
        ERROR:
        {
            "error": "Detailed error message"
        }
    """
    if not smiles:
        return json.dumps({"error": "A reference molecule SMILES is required."})

    payload = {
        "smiles_list": [smiles],
        "num_variants": num_variants,
        "strategy": strategy,
        "temperature": temperature
    }
    result = _call_worker_api("reinvent4", payload, action="mol2mol", timeout_mins=10)

    if "error" in result:
        return json.dumps({"error": result["error"]})

    summary = result.get("summary", {})
    molecules_data = result.get("results", {}).get("molecules", [])
    smiles_list_out = [mol["smiles"] for mol in molecules_data if mol.get("smiles")]

    agent_response = {
        "status": "success",
        "input_smiles": smiles,
        "generated_count": summary.get("generated_count", len(smiles_list_out)),
        "molecules_smiles": smiles_list_out
    }
    return json.dumps(agent_response, ensure_ascii=False)


# ==========================================
# 🛠️ Agent 工具 10：DrugEx 图网络强化学习生成
# ==========================================
def generate_molecules_drugex(
    input_fragments: str = "arl_test_graph.txt", generator_model: str = "arl_graph_trans_RL", num_samples: int = 50
) -> str:
    """
    Use this tool to generate molecules using DrugEx (Graph-based DL + RL).
    It is specifically useful for multi-objective optimization based on input fragments.

    Args:
        input_fragments (str, optional): The name of the input fragment/graph file. Defaults to 'arl_test_graph.txt'.
        generator_model (str, optional): The pre-trained DrugEx model to use. Defaults to 'arl_graph_trans_RL'.
        num_samples (int, optional): The number of molecules to generate. Defaults to 50.

    Returns:
        str: A JSON-formatted string containing the following fields:
            - "status" (str): "success" or "error".
            - "model_used" (str): The name of the model actually used.
            - "requested_samples" (int): The number of samples requested.
            - "total_generated" (int): The actual number of valid molecules generated.
            - "top_molecules_preview" (list): A list of dictionaries representing the generated molecules.
              Each dictionary contains:
                - "smiles" (str): The SMILES string of the molecule.
                - "valid" (float): Validity score (usually 1.0 or 0.0).
                - "accurate" (float): Accuracy score.
                - "desired" (float): Desirability score based on the RL objectives.
                - "qsarpred_a2ar" (float): Predicted activity score for the A2AR target.
    """
    payload = {"input_fragments": input_fragments, "generator": generator_model, "num_samples": num_samples}
    result = _call_worker_api("drugex", payload, action="generate", timeout_mins=10)

    if "error" in result:
        return json.dumps({"error": result["error"]})

    summary = result.get("summary", {})
    agent_response = {
        "status": "success",
        "model_used": summary.get("model_used"),
        "requested_samples": num_samples,
        "total_generated": summary.get("total_molecules_generated"),
        "top_molecules_preview": result.get("results", {}).get("molecules_preview", []),
    }

    return json.dumps(agent_response, ensure_ascii=False)
