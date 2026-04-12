"""
化学工具集 - SCScore 合成复杂度评分（跨环境调用）

这些工具在独立的 scscore conda 环境中运行，使用 RDKit 和 SCScorer 模型
"""

from .cross_env_tools.base import CrossEnvTool

_scscore_tool = None


def _get_scscore_tool():
    """获取 SCScore 工具实例（延迟初始化）"""
    global _scscore_tool
    if _scscore_tool is None:
        _scscore_tool = CrossEnvTool(env_name="scscore", script_name="scscore_worker.py", timeout=60)
    return _scscore_tool


def calculate_scscore(smiles: str, model_type: str = "1024bool") -> dict:
    """
    计算分子的合成复杂度评分（SCScore）

    SCScore 是一个预测分子合成难度的机器学习模型，评分范围 1-5：
    - 1 分：非常容易合成（简单的起始材料）
    - 2 分：容易合成（需要少量合成步骤）
    - 3 分：中等难度（需要多步合成）
    - 4 分：较难合成（需要复杂的合成路线）
    - 5 分：非常难合成（需要高级合成技术）

    注意：此工具在独立的 scscore conda 环境中运行，使用 RDKit 和 SCScorer 模型

    Parameters:
        smiles: 分子的 SMILES 字符串，例如 "CCCOCCC" 或 "CC(=O)OC1=CC=CC=C1C(=O)O"
        model_type: 模型类型，可选 "1024bool"（默认）、"2048bool"、"1024uint8"

    Returns:
        包含以下字段的字典：
        - success: 是否成功（True/False）
        - input_smiles: 输入的 SMILES 字符串
        - canonical_smiles: 标准化的 SMILES 字符串
        - scscore: 合成复杂度评分（1-5）
        - model: 使用的模型类型
        - interpretation: 评分的文字解释
        - error: 如果失败，包含错误信息

    Examples:
        计算简单分子的 SCScore：
        result = calculate_scscore("CCCOCCC")
        # 返回: {'success': True, 'scscore': 1.43, 'interpretation': '非常容易合成'}

        计算阿司匹林的 SCScore：
        result = calculate_scscore("CC(=O)OC1=CC=CC=C1C(=O)O")
        # 返回: {'success': True, 'scscore': 1.59, 'interpretation': '容易合成'}
    """
    tool = _get_scscore_tool()
    return tool.call(function_name="calculate_scscore", smiles=smiles, model_type=model_type)


def batch_calculate_scscore(smiles_list: list, model_type: str = "1024bool") -> dict:
    """
    批量计算多个分子的合成复杂度评分（SCScore）

    批量计算比多次单独调用更高效，因为只需启动一次跨环境进程。
    适合需要评估多个候选分子的场景，如虚拟筛选、药物设计等。

    注意：此工具在独立的 scscore conda 环境中运行

    Parameters:
        smiles_list: SMILES 字符串列表，例如 ["CCCOCCC", "CCCNc1ccccc1", "CC(C)C"]
        model_type: 模型类型，可选 "1024bool"（默认）、"2048bool"、"1024uint8"

    Returns:
        包含以下字段的字典：
        - success: 是否成功（True/False）
        - summary: 统计信息
            - total: 总分子数
            - successful: 成功计算的数量
            - failed: 失败的数量
            - avg_scscore: 平均 SCScore
            - min_scscore: 最小 SCScore
            - max_scscore: 最大 SCScore
            - median_scscore: 中位数 SCScore
        - results: 每个分子的详细结果列表
        - errors: 失败的分子列表（如果有）

    Examples:
        批量计算多个分子：
        smiles_list = ["CCCOCCC", "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
        result = batch_calculate_scscore(smiles_list)
        # 返回: {'summary': {'total': 3, 'avg_scscore': 2.1}, 'results': [...]}
    """
    tool = _get_scscore_tool()
    return tool.batch_call(function_name="calculate_scscore", items=smiles_list, model_type=model_type)


def compare_synthesis_complexity(smiles_list: list, model_type: str = "1024bool") -> dict:
    """
    比较多个分子的合成复杂度并按难易程度排序

    此工具会计算所有分子的 SCScore，然后按照合成难度从易到难排序。
    适合用于选择最容易合成的候选分子，或者了解分子库的合成难度分布。

    注意：此工具在独立的 scscore conda 环境中运行

    Parameters:
        smiles_list: 要比较的 SMILES 字符串列表
        model_type: 模型类型，可选 "1024bool"（默认）、"2048bool"、"1024uint8"

    Returns:
        包含以下字段的字典：
        - total_molecules: 成功计算的分子总数
        - model: 使用的模型类型
        - ranked_molecules: 按 SCScore 从低到高排序的分子列表
        - easiest_to_synthesize: 最容易合成的分子（SCScore 最低）
        - hardest_to_synthesize: 最难合成的分子（SCScore 最高）
        - complexity_range: 复杂度范围信息
            - min: 最小 SCScore
            - max: 最大 SCScore
            - span: 范围跨度

    Examples:
        比较候选药物分子的合成难度：
        candidates = ["CC(C)C", "CCCOCCC", "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
        result = compare_synthesis_complexity(candidates)
        print(f"最容易合成: {result['easiest_to_synthesize']['canonical_smiles']}")
        print(f"最难合成: {result['hardest_to_synthesize']['canonical_smiles']}")
    """
    batch_result = batch_calculate_scscore(smiles_list, model_type)
    if "error" in batch_result:
        return batch_result
    results = batch_result.get("results", [])
    if not results:
        return {"error": "No successful calculations"}
    sorted_results = sorted(results, key=lambda x: x["scscore"])
    return {
        "total_molecules": len(results),
        "model": model_type,
        "ranked_molecules": sorted_results,
        "easiest_to_synthesize": sorted_results[0],
        "hardest_to_synthesize": sorted_results[-1],
        "complexity_range": {
            "min": sorted_results[0]["scscore"],
            "max": sorted_results[-1]["scscore"],
            "span": sorted_results[-1]["scscore"] - sorted_results[0]["scscore"],
        },
    }


def get_scscore_env_info() -> dict:
    """
    获取 SCScore 跨环境工具的配置信息

    此工具返回 SCScore 环境的配置详情，用于调试和验证环境设置。

    Returns:
        包含以下字段的字典：
        - env_name: 环境名称
        - conda_env: Conda 环境名称
        - python_path: Python 解释器路径
        - script_path: Worker 脚本路径
        - description: 环境描述
        - timeout: 超时设置（秒）
        - cache_enabled: 是否启用缓存

    Examples:
        检查 SCScore 环境配置：
        info = get_scscore_env_info()
        print(f"环境: {info['conda_env']}")
        print(f"Python: {info['python_path']}")
    """
    tool = _get_scscore_tool()
    return tool.get_env_info()
