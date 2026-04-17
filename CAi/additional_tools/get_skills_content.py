
from functools import lru_cache
# ==========================================
# Skills 工具（供 Agent 和外部直接调用）
# ==========================================

@lru_cache(maxsize=1)
def _get_skill_loader():
    """Return a cached SkillLoader instance (lazy init)."""
    from CAi.CAi_agent.skills import SkillLoader
    return SkillLoader()


def get_skill_content(skill_id: str) -> str:
    """
    Get the complete workflow and guidance of a skill by its ID.

    Use this to get step-by-step instructions before executing a complex task.
    First call list_available_skills() to discover available skill IDs.

    Args:
        skill_id: The ID of the skill (e.g., 'molecule_analysis', 'virtual_screening')

    Returns:
        The complete skill content including workflow, examples, and best practices.
    """
    loader = _get_skill_loader()
    skill = loader.get_skill_by_id(skill_id)
    if skill:
        return skill.get("content_without_metadata", skill.get("content", ""))
    return f"Error: Skill '{skill_id}' not found. Call list_available_skills() to see available IDs."


def list_available_skills() -> str:
    """
    List all available skill IDs and their descriptions.

    Call this first to discover what skills exist, then use
    get_skill_content(skill_id) to get the detailed workflow.

    Returns:
        A formatted list of skill IDs and descriptions.
    """
    loader = _get_skill_loader()
    summaries = loader.get_skill_summaries()
    if not summaries:
        return "No skills available."
    lines = ["Available skills:"]
    for s in summaries:
        lines.append(f"  - {s['id']}: {s['description']}")
    return "\n".join(lines)