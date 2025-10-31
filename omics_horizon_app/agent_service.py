"\"\"\"Agent initialization helpers for Omics Horizon.\"\"\""

from __future__ import annotations

import streamlit as st

from biomni.agent import A1_HITS

from .config import create_agent_config


def get_or_create_agent(logger) -> A1_HITS:
    """Return session-scoped agent instance, creating it when needed."""
    agent = st.session_state.get("agent")
    if agent is not None:
        return agent

    session_config = create_agent_config()
    agent = A1_HITS(
        path=session_config.path,
        llm=session_config.llm,
        use_tool_retriever=session_config.use_tool_retriever,
        timeout_seconds=session_config.timeout_seconds,
        base_url=session_config.base_url,
        api_key=session_config.api_key,
        commercial_mode=session_config.commercial_mode,
    )
    st.session_state.agent = agent
    logger.info(
        "Agent initialized with model=%s, path=%s",
        session_config.llm,
        session_config.path,
    )
    return agent
