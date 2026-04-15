import os
import time
import json
import requests
from typing import Dict, Any, Optional, List

TOOL_SERVER_HOST = os.environ.get("TOOL_SERVER_HOST", "100.103.118.72")
TOOL_SERVER_PORT = os.environ.get("TOOL_SERVER_PORT", "8001")
BASE_URL = f"http://{TOOL_SERVER_HOST}:{TOOL_SERVER_PORT}"

def _call_worker_api(tool_name: str, payload: Dict[str, Any], action: str = "default", timeout_mins: int = 5) -> Dict[str, Any]:
    """
    通用底层封装：向服务器提交任务并轮询等待结果。
    这个函数对 Agent 是隐藏的，只供上层工具函数调用。
    """
    run_url = f"{BASE_URL}/run/{tool_name}/{action}"
    job_url = f"{BASE_URL}/job"

    # 🌟 核心修复：显式声明不使用任何系统代理，防止内网 IP 被发往外网代理服务器导致 502
    bypass_proxies = {
        "http": None, 
        "https": None
    }

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
# 🛠️ Agent 工具 1：骨架衍生生成
# ==========================================
def generate_scaffold_analogs(smiles: str, num_analogs: int = 10) -> str:
    """
    Use this tool to generate novel molecule analogs based on a specific chemical scaffold.
    
    Args:
        smiles (str): The SMILES string of the scaffold. MUST contain at least one '*' character to indicate the attachment/growth point (e.g., 'c1ccccc1*').
        num_analogs (int, optional): The number of analogs to generate. Defaults to 10. Max recommended is 100.
        
    Returns:
        str: A JSON-formatted string. 
        On SUCCESS, the JSON will strictly follow this schema:
        {
            "status": "success",
            "generated_count": 12,  # integer: the actual number of valid, unique molecules generated
            "molecules": [          # list of strings: the generated SMILES strings
                "c1cc(N)ccc1OC1CCN(C(C)C(NO)=O)CC1",
                "c1cc(N)ccc1S(=O)(=O)"
            ]
        }
        On ERROR, the JSON will follow this schema:
        {
            "error": "The error message detailing what went wrong."
        }
    """
    if "*" not in smiles:
        return json.dumps({"error": "The input scaffold SMILES must contain at least one '*' character as the growth point."})
        
    payload = {"smiles": smiles, "num_analogs": num_analogs}
    result = _call_worker_api("scaffold", payload)
    
    if "error" in result:
        return json.dumps(result)
        
    summary = result.get("summary", {})
    generated_smiles = [item["smiles"] for item in result.get("results", [])]
    
    agent_response = {
        "status": "success",
        "generated_count": summary.get("valid_unique_generated"),
        "molecules": generated_smiles
    }
    
    return json.dumps(agent_response, ensure_ascii=False)


# ==========================================
# 🛠️ Agent 工具 2：毒性预测与 SHAP 解释
# ==========================================
def predict_molecule_toxicity(smiles: str) -> str:
    """
    Use this tool to predict whether a given molecule is toxic and get a SHAP-based structural explanation.
    It returns the probability, the toxicity verdict, structural insights, and a file path to an interpretation image.
    
    Args:
        smiles (str): The valid SMILES string of the complete molecule to be evaluated.
        
    Returns:
        str: A JSON string strictly following this schema:
        
        SUCCESS:
        {
            "verdict": "Toxic" | "Non-Toxic",
            "toxicity_probability": 0.1234,
            "structural_explanation": [
                {"fragment": "c1ccccc1", "contribution_score": 0.05, "effect": "Increases Toxicity" | "Decreases Toxicity"}
            ],
            "image_saved_at": "/path/to/image.png",
            "vision_prompt": "..."
        }
        *CRITICAL*: If 'image_saved_at' is returned, you MUST use your Vision capability to read this image and summarize it for the user!
        
        ERROR:
        {
            "error": "Detailed error message"
        }
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
        "structural_explanation": results_data.get("interpretation", [])
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
        agent_response["vision_prompt"] = "The structure interpretation image has been saved locally. Please look at the image to analyze the toxic fragments."


    
    return json.dumps(agent_response, ensure_ascii=False)


# ==========================================
# 🛠️ Agent 工具 3：SCScore 计算
# ==========================================
def calculate_scscore(smiles: Optional[str] = None, smiles_list: Optional[List[str]] = None, model_type: str = "1024bool") -> str:
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
def generate_libinvent_decorations(smiles: str, num_decorations: int = 32) -> str:
    """
    Use this tool to decorate a specific chemical scaffold using the Lib-INVENT reaction-based model.
    It adds side-chains or functional groups to the specified attachment points.
    
    Args:
        smiles (str): The SMILES string of the scaffold. MUST contain at least one attachment point (e.g., '[*]c1ccccc1').
        num_decorations (int, optional): The number of decorated molecules to generate. Defaults to 32.
        
    Returns:
        str: A JSON-formatted string containing the successfully generated decorated molecules and their properties, or an error message.
    """
    if "*" not in smiles and "[" not in smiles:
        return json.dumps({"error": "The input scaffold SMILES must contain an attachment point (like '*' or '[*:1]')."})

    payload = {
        "smiles": smiles,
        "number_of_decorations_per_scaffold": num_decorations
    }

    result = _call_worker_api("libinvent", payload)

    if "error" in result:
        return json.dumps({"error": result["error"]})

    summary = result.get("summary", {})
    columns = summary.get("columns", [])
    preview_rows = summary.get("preview", [])

    # Extract complete molecule SMILES — Lib-INVENT CSV typically has a "SMILES" column
    # with the fully decorated molecule. Fall back to returning the raw rows with column info
    # so the agent knows exactly which key to read.
    smiles_key = next((c for c in columns if c.upper() == "SMILES"), None)
    if smiles_key:
        molecules_smiles = [row[smiles_key] for row in preview_rows if row.get(smiles_key)]
    else:
        molecules_smiles = []

    agent_response = {
        "status": "success",
        "generated_count": summary.get("row_count"),
        "csv_columns": columns,
        "molecules_smiles": molecules_smiles,
        "decorated_molecules_preview": preview_rows,
    }

    return json.dumps(agent_response, ensure_ascii=False)

# ==========================================
# 🛠️ Agent 工具 5：抗菌活性 (pMIC) 预测
# ==========================================
def predict_antibacterial_pmic(smiles: str) -> str:
    """
    Use this tool to predict the antibacterial activity of a molecule.
    It returns the predicted pMIC value and the estimated Minimum Inhibitory Concentration (MIC) in µM.
    Higher pMIC values (or lower MIC_uM values) indicate stronger antibacterial activity.
    
    Args:
        smiles (str): The valid SMILES string of the complete molecule to be evaluated.
        
    Returns:
        str: A JSON string containing the prediction results.
        The JSON structure is guaranteed to have the following keys:
        - "status" (str): "success" or "error".
        - "pMIC_value" (float): The predicted pMIC score.
        - "estimated_MIC_uM" (float): The estimated MIC value in µM.
        - "interpretation" (str): A brief explanation of the values.
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
        "interpretation": "Higher pMIC means stronger activity. A typical threshold for active compounds is often pMIC > 5.0 (MIC < 10 µM)."
    }
    
    
    return json.dumps(agent_response, ensure_ascii=False)

# ==========================================
# 🛠️ Agent 工具 6：RxnFlow 靶点口袋导向分子生成
# ==========================================
def generate_molecules_for_pocket(protein_pdb_path: str, center_xyz: list = None, ref_ligand_path: str = None, num_samples: int = 50) -> str:
    """
    Use this tool to perform structure-based zero-shot drug design (SBDD). 
    It generates novel molecules specifically tailored for a 3D protein pocket using RxnFlow.
    
    Args:
        protein_pdb_path (str): The absolute path to the protein target PDB file on the server.
        center_xyz (list of 3 floats, optional): The [x, y, z] coordinates of the binding pocket center. MUST provide either this or ref_ligand_path.
        ref_ligand_path (str, optional): The absolute path to a reference ligand PDB/SDF file to define the pocket. MUST provide either this or center_xyz.
        num_samples (int, optional): The number of molecules to generate. Defaults to 50.
        
    Returns:
        str: A JSON-formatted string containing the following fields:
            - "status" (str): "success" or "error".
            - "generated_count" (int): The actual number of molecules successfully generated.
            - "sampling_time_sec" (float): Total time taken for generation in seconds.
            - "full_results_csv_path" (str): Absolute path to the CSV file containing all generated SMILES and scores.
            - "top_molecules_preview" (list): A list of dictionaries for the top generated molecules.
              Each dictionary contains:
                - "smiles" (str): The SMILES string.
                - "qed" (float): QED score.
                - "proxy_score" (float): Vina proxy score (binding affinity estimation).
    """
    if not center_xyz and not ref_ligand_path:
        return json.dumps({"error": 'You must provide either "center_xyz" coordinates OR a "ref_ligand_path" to define the binding pocket.'})
        
    payload = {
        "protein_pdb_path": protein_pdb_path,
        "num_samples": num_samples,
        "save_reward": True
    }
    if center_xyz:
        payload["center"] = center_xyz
    if ref_ligand_path:
        payload["ref_ligand_path"] = ref_ligand_path
        
    # 这个工具跑得比较慢，允许 15 分钟超时
    result = _call_worker_api("rxnflow", payload, timeout_mins=15)

    if "error" in result:
        return json.dumps({"error": result["error"]})

    summary = result.get("summary", {})
    results_data = result.get("results", {})
    
    agent_response = {
        "status": "success",
        "generated_count": summary.get("generated_count"),
        "sampling_time_sec": summary.get("sampling_time_sec"),
        "full_results_csv_path": summary.get("output_file"),
        "top_molecules_preview": results_data.get("generated_preview", [])
    }
    
    
    return json.dumps(agent_response, ensure_ascii=False)

# ==========================================
# 🛠️ Agent 工具 7：AutoDock Vina 分子对接
# ==========================================
def perform_molecular_docking_vina(receptor_pdbqt_path: str, ligand_pdbqt_path: str, center_xyz: list, box_size_xyz: list, exhaustiveness: int = 32) -> str:
    """
    Use this tool to perform molecular docking of a small molecule ligand into a protein receptor using AutoDock Vina.
    It calculates the binding affinity (docking score) and generates the 3D docked poses.
    
    Args:
        receptor_pdbqt_path (str): The absolute path to the receptor PDBQT file on the server.
        ligand_pdbqt_path (str): The absolute path to the ligand PDBQT file on the server.
        center_xyz (list of 3 floats): The [x, y, z] coordinates of the binding box center.
        box_size_xyz (list of 3 floats): The dimensions [x, y, z] of the search box (in Angstroms).
        exhaustiveness (int, optional): The exhaustiveness of the global search. Default is 32. Higher values take longer but are more accurate.
        
    Returns:
        str: A JSON-formatted string containing the following fields:
            - "status" (str): "success" or "error".
            - "best_docking_score_kcal_mol" (float): The binding affinity of the best global docking pose (more negative is better).
            - "minimized_score_kcal_mol" (float): The score after local minimization of the input pose.
            - "docked_poses_file_path" (str): Absolute path to the generated PDBQT file containing the top docking poses.
            - "interpretation" (str): A brief guide on how to read the scores.
    """
    payload = {
        "receptor_pdbqt_file": receptor_pdbqt_path,
        "ligand_pdbqt_file": ligand_pdbqt_path,
        "center": center_xyz,
        "box_size": box_size_xyz,
        "exhaustiveness": exhaustiveness,
        "n_poses": 10
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
        "interpretation": "More negative scores indicate stronger binding affinity."
    }
    
    
    return json.dumps(agent_response, ensure_ascii=False)


# ==========================================
# 🛠️ Agent 工具 8：REINVENT4 分子多维综合打分
# ==========================================
def score_molecules_reinvent(smiles_list: list) -> str:
    """
    Use this tool to evaluate and score a list of molecules using REINVENT4.
    It calculates multiple physicochemical properties including QED (drug-likeness), MW (Molecular Weight), Tanimoto similarity, and provides a comprehensive composite Score.
    
    Args:
        smiles_list (list of str): A list of valid SMILES strings to be scored.
        
    Returns:
        str: A JSON-formatted string containing the following fields:
            - "status" (str): "success" or "error".
            - "scored_count" (int): The number of molecules successfully scored.
            - "top_molecules_sorted_by_score" (list): Top 20 molecules sorted by composite score (descending).
              Each dictionary contains:
                - "smiles" (str): The SMILES string.
                - "score" (float): The overall composite desirability score.
                - "qed" (float): Quantitative Estimate of Drug-likeness.
                - "mw" (float): Molecular Weight.
                - "tanimoto" (float): Tanimoto similarity to reference (if applicable).
                - "alerts" (str): Structural alerts or toxicity warnings (empty string if none).
    """
    if not smiles_list or not isinstance(smiles_list, list):
        return json.dumps({"error": "Please provide a valid list of SMILES strings."})

    payload = {
        "smiles_list": smiles_list
    }

    result = _call_worker_api("reinvent4", payload, action="score", timeout_mins=5)

    if "error" in result:
        return json.dumps({"error": result["error"]})
        
    summary = result.get("summary", {})
    scores_data = result.get("results", {}).get("scored_data", [])
    
    # 按照综合得分从高到低排序，帮助大模型优先关注好分子
    sorted_scores = sorted(scores_data, key=lambda x: x.get("score", 0), reverse=True)
    
    agent_response = {
        "status": "success",
        "scored_count": summary.get("scored_molecules"),
        "top_molecules_sorted_by_score": sorted_scores[:20] # 如果分子太多，只返回前 20 个避免超出上下文
    }
    
    
    return json.dumps(agent_response, ensure_ascii=False)

# ==========================================
# 🛠️ Agent 工具 9：REINVENT4 从头分子生成 (De novo Design)
# ==========================================
def generate_molecules_reinvent(num_samples: int = 50) -> str:
    """
    Use this tool to perform de novo molecule generation using the REINVENT4 prior model.
    It samples structurally novel and valid SMILES strings from scratch without requiring a protein pocket.
    
    Args:
        num_samples (int, optional): The number of novel molecules to generate. Defaults to 50.
        
    Returns:
        str: A JSON-formatted string containing the successfully generated SMILES strings and their Negative Log-Likelihood (NLL) scores.
    """
    payload = {
        "num_samples": num_samples
    }
        
    result = _call_worker_api("reinvent4", payload, action="sample", timeout_mins=10)

    if "error" in result:
        return json.dumps({"error": result["error"]})
        
    summary = result.get("summary", {})
    molecules_data = result.get("results", {}).get("molecules", [])
    
    # 提取纯 SMILES 列表方便大模型阅读
    smiles_list = [mol["smiles"] for mol in molecules_data]
    
    agent_response = {
        "status": "success",
        "generated_count": summary.get("generated_count"),
        "molecules_smiles": smiles_list
    }
    
    
    return json.dumps(agent_response, ensure_ascii=False)

# ==========================================
# 🛠️ Agent 工具 10：DrugEx 图网络强化学习生成
# ==========================================
def generate_molecules_drugex(input_fragments: str = "arl_test_graph.txt", generator_model: str = "arl_graph_trans_RL", num_samples: int = 50) -> str:
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
    result = _call_worker_api("drugex", payload, action = "generate",timeout_mins=10)
    
    if "error" in result:
        return json.dumps({"error": result["error"]})
    
    summary = result.get("summary", {})
    agent_response = {
        "status": "success",
        "model_used": summary.get("model_used"),
        "requested_samples": num_samples,
        "total_generated": summary.get("total_molecules_generated"),
        "top_molecules_preview": result.get("results", {}).get("molecules_preview", [])
    }
    
    return json.dumps(agent_response, ensure_ascii=False)