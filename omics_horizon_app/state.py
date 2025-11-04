"\"\"\"Session state helper utilities for Omics Horizon.\"\"\""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Iterable

import streamlit as st

from .config import WORKSPACE_PATH

DEFAULT_STATE: Dict[str, Any] = {
    "language": "en",
    "data_files": [],
    "data_briefing": "",
    "paper_files": [],
    "analysis_method": "",
    "message_history": [],
    "qa_history": [],
    "chat_history": [],
    "is_streaming": False,
    "analysis_started": False,
    "should_run_agent": False,
}


def ensure_session_defaults(from_lims: bool = False, workspace_path: str | None = None) -> None:
    """Populate required Streamlit session-state keys with safe defaults."""
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value if not callable(value) else value()

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "work_dir" not in st.session_state:
        if from_lims and workspace_path:
            st.session_state.work_dir = workspace_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(WORKSPACE_PATH, f"session_{timestamp}")
            os.makedirs(session_dir, exist_ok=True)
            st.session_state.work_dir = session_dir


def reset_analysis_session(preserve_keys: Iterable[str] | None = None) -> None:
    """Clear Streamlit session state for a fresh analysis run."""
    preserved = set(preserve_keys or [])
    for key in list(st.session_state.keys()):
        if key not in preserved:
            del st.session_state[key]
