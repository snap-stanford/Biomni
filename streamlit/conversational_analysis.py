"""Conversational analysis runtime extracted from streamlit_app."""

from __future__ import annotations

import os
import re
from typing import Iterable

import streamlit as st

from omics_horizon_app import CURRENT_ABS_DIR
from omics_horizon_app.agent_runtime import (
    add_chat_message,
    build_agent_input_from_history,
    display_chat_files,
    format_agent_output_for_display,
    maybe_add_assistant_message,
)

CHAT_ATTACHMENT_PATTERNS: tuple[str, ...] = (
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.bmp",
)

_FINAL_ANSWER_SECTION_RE = re.compile(
    r"\n?\s*---\s*\n\s*🎯 \*\*최종 답변:\*\*\s*\n.*?(?:\n\s*---\s*)?",
    re.DOTALL,
)


def _collect_workspace_artifacts(patterns: Iterable[str]) -> set[str]:
    workspace = st.session_state.get("work_dir")
    if not workspace:
        return set()
    collected: set[str] = set()
    for pattern in patterns:
        for path in glob.glob(os.path.join(workspace, pattern)):
            collected.add(os.path.abspath(path))
    return collected


def _format_process_without_solution(raw_text: str) -> str:
    formatted = format_agent_output_for_display(raw_text)
    cleaned = _FINAL_ANSWER_SECTION_RE.sub("\n\n", formatted)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _sanitize_solution_text(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"<execute>.*?</execute>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<observation>.*?</observation>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"```[a-zA-Z0-9]*\n.*?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"Plan Update:.*?\n", "", cleaned)
    cleaned = re.sub(r"===.*?===", "", cleaned)
    cleaned = re.sub(r"🐍\s*\*\*코드 실행.*?\*\*", "", cleaned)
    cleaned = re.sub(r"📊\s*\*\*코드 실행.*?\*\*", "", cleaned)
    cleaned = re.sub(r"🔧\s*\*\*코드 실행.*?\*\*", "", cleaned)
    cleaned = re.sub(r"✅\s*\*\*실행 성공.*?\*\*", "", cleaned)
    cleaned = re.sub(r"❌\s*\*\*실행 오류.*?\*\*", "", cleaned)
    cleaned = re.sub(r"^---+$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*Here is my plan.*$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)
    cleaned = re.sub(r"^\s*I will now proceed.*$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)
    cleaned = re.sub(r"^\s*Perform bioinformatics analysis.*$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)
    cleaned = re.sub(r"^\s*#Analysis Instructions:.*$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)
    cleaned = re.sub(r"^\s*DATA FILES:.*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*DATA BRIEFING:.*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s*[⬜✅].*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*⬜.*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*✅ Step.*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_final_response(raw_text: str) -> str:
    if not raw_text:
        return ""

    solution_matches = re.findall(r"<solution>(.*?)</solution>", raw_text, re.DOTALL)
    for candidate in reversed(solution_matches):
        cleaned = _sanitize_solution_text(candidate)
        if cleaned:
            return cleaned

    observation_matches = re.findall(
        r"<observation>(.*?)</observation>", raw_text, re.DOTALL
    )
    for candidate in reversed(observation_matches):
        cleaned = _sanitize_solution_text(candidate)
        if cleaned:
            return cleaned

    return _sanitize_solution_text(raw_text)


# ... rest of the module remains unchanged
