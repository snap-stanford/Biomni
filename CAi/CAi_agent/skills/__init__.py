"""CAi Skills - 高级技能定义模块

Skills 以 Markdown 文件形式存储，展示如何组合使用工具来完成复杂任务。
它们会被延迟加载到系统提示词中，指导 Agent 如何处理特定类型的任务。
"""

from .loader import SkillLoader

__all__ = ["SkillLoader"]
