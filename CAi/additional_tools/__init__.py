"""
Additional Tools for A1pro Agent

这个模块包含所有自定义工具，会被 A1pro 自动加载。

工具编写规范：
1. 每个工具都是一个独立的函数
2. 必须有完整的文档字符串（docstring）
3. 必须有类型注解
4. 复杂依赖在函数内部导入
5. 不要使用下划线开头的函数名（会被跳过）

示例：
    def my_tool(param1: str, param2: int) -> dict:
        '''
        工具描述

        Parameters:
            param1: 参数1描述
            param2: 参数2描述

        Returns:
            返回值描述
        '''
        # 在函数内部导入依赖
        import pandas as pd

        # 工具逻辑
        result = ...

        return result
"""

# 从子模块导入所有工具

from .template_tools import *

from .get_skills_content import get_skill_content, list_available_skills

# 如果你有其他工具模块，也在这里导入
# from .biology_tools import *
# from .data_tools import *

__all__ = [
    'get_skill_content',
    'list_available_skills',
    # 从 template_tools 导入的工具
    'calculate_scscore',
    'generate_scaffold_analogs',
    'predict_molecule_toxicity',
    'generate_molecules_for_pocket',
    'generate_libinvent_decorations',
    'predict_antibacterial_pmic',
    'perform_molecular_docking_vina',
    'generate_molecules_reinvent4_denovo',
    'generate_molecules_reinvent4_libinvent',
    'generate_molecules_reinvent4_mol2mol',
    'generate_molecules_drugex'
]
