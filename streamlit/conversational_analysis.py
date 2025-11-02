"""Conversational analysis workflow extracted from the Streamlit app."""

from __future__ import annotations

import glob
import os
import re
from typing import Iterable

import streamlit as st
from langchain_core.messages import HumanMessage

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


def _collect_workspace_artifacts(patterns: Iterable[str]) -> set[str]:
    workspace = st.session_state.get("work_dir")
    if not workspace:
        return set()
    collected: set[str] = set()
    for pattern in patterns:
        for path in glob.glob(os.path.join(workspace, pattern)):
            collected.add(os.path.abspath(path))
    return collected


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


def render_analysis_conversation() -> None:
    """Display the conversational analysis interface."""
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("analysis_started", False)
    st.session_state.setdefault("should_run_agent", False)
    st.session_state.setdefault("is_streaming", False)

    st.markdown("### 💬 Analysis Conversation")

    if st.button(
        "Start Analysis",
        key="start_analysis_btn",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.analysis_started = True
        st.session_state.should_run_agent = True
        st.rerun()

    if not st.session_state.analysis_started:
        return

    if not st.session_state.chat_history:
        with st.chat_message(
            "assistant", avatar=f"{CURRENT_ABS_DIR}/logo/AI_assistant_logo.png"
        ):
            st.markdown("👋 **Hi! I am OmicsHorizon, your bioinformatics assistant.**")

    for message in st.session_state.chat_history:
        with st.chat_message(
            message["role"],
            avatar=(
                f"{CURRENT_ABS_DIR}/logo/AI_assistant_logo.png"
                if message["role"] == "assistant"
                else None
            ),
        ):
            if message["role"] == "assistant":
                st.markdown(format_agent_output_for_display(message["content"]))
            else:
                st.markdown(message["content"])
            if message.get("files"):
                display_chat_files(message["files"])

    user_input = st.chat_input("메시지를 입력하세요...", key="user_chat_input")
    if user_input:
        add_chat_message("user", user_input)
        st.session_state.should_run_agent = True
        st.rerun()

    if not st.session_state.should_run_agent:
        return

    original_dir = os.getcwd()
    os.chdir(st.session_state.work_dir)
    try:
        data_info = ", ".join([f"`{f}`" for f in st.session_state.data_files])
        prompt = f"""Perform bioinformatics analysis.
#Analysis Instructions:
{st.session_state.analysis_method}

DATA FILES: {data_info}

DATA BRIEFING:
{st.session_state.data_briefing if st.session_state.data_briefing else "Files are available in the working directory"}

"""

        has_assistant_history = any(
            msg.get("role") == "assistant" for msg in st.session_state.chat_history
        )
        agent_input = build_agent_input_from_history(
            initial_prompt=prompt, include_initial=not has_assistant_history
        )

        st.session_state.is_streaming = True
        attachments: list[str] = []
        result_text = ""
        last_node_text: dict[str, str] = {}

        def _compute_delta(node_id: str, new_text: str) -> str:
            previous = last_node_text.get(node_id, "")
            if previous and new_text.startswith(previous):
                delta = new_text[len(previous) :]
            else:
                delta = new_text
            last_node_text[node_id] = new_text
            return delta

        with st.chat_message(
            "assistant", avatar=f"{CURRENT_ABS_DIR}/logo/AI_assistant_logo.png"
        ):
            message_placeholder = st.empty()
            baseline_files = _collect_workspace_artifacts(CHAT_ATTACHMENT_PATTERNS)
            with st.spinner("AI is performing the analysis..."):
                try:
                    message_stream = st.session_state.agent.go_stream(agent_input)
                    for chunk in message_stream:
                        node = chunk[1][1]["langgraph_node"]
                        chunk_data = chunk[1][0]
                        if node not in {"generate", "execute"} or not hasattr(
                            chunk_data, "content"
                        ):
                            continue
                        content = chunk_data.content
                        if isinstance(content, list):
                            joined = "".join(
                                item for item in content if isinstance(item, str)
                            )
                        else:
                            joined = content if isinstance(content, str) else ""
                        delta = _compute_delta(node, joined)
                        if not delta:
                            continue
                        result_text += delta
                        message_placeholder.markdown(
                            format_agent_output_for_display(result_text)
                        )
                except Exception as exc:  # pragma: no cover - UI fallback
                    st.error(f"Agent execution failed: {exc}")
                finally:
                    st.session_state.is_streaming = False

            updated_files = _collect_workspace_artifacts(CHAT_ATTACHMENT_PATTERNS)
            new_files = sorted(
                updated_files - baseline_files,
                key=lambda path: os.path.getmtime(path),
            )
            attachments = new_files
            if new_files:
                display_chat_files(new_files)
    finally:
        os.chdir(original_dir)

    st.session_state.should_run_agent = False
    final_text = _extract_final_response(result_text)
    if attachments:
        figure_tokens = []
        for path in attachments:
            abs_path = os.path.abspath(path)
            if not os.path.isfile(abs_path):
                continue
            figure_tokens.append(f"[[FIGURE::{abs_path}]]")
        if figure_tokens and final_text:
            final_text = (final_text + "\n\n" + "\n\n".join(figure_tokens)).strip()
        elif figure_tokens:
            final_text = "\n\n".join(figure_tokens)
    if final_text:
        message_placeholder.markdown(
            format_agent_output_for_display(final_text)
        )
        maybe_add_assistant_message(final_text, files=attachments)
    st.rerun()


def get_analysis_context(max_messages: int = 5) -> str:
    """Build a lightweight context string for Q&A."""
    context_parts: list[str] = []

    if st.session_state.data_briefing:
        context_parts.append("=== DATA BRIEFING ===")
        context_parts.append(st.session_state.data_briefing[:1000])

    if st.session_state.analysis_method:
        context_parts.append("=== ANALYSIS WORKFLOW ===")
        context_parts.append(st.session_state.analysis_method[:1000])

    assistant_messages = [
        msg["content"]
        for msg in st.session_state.chat_history
        if msg.get("role") == "assistant" and msg.get("content")
    ]
    if assistant_messages:
        context_parts.append("=== RECENT ANALYSIS RESPONSES ===")
        for content in assistant_messages[-max_messages:]:
            context_parts.append(content.strip())

    return "\n\n".join(context_parts) if context_parts else "No analysis context available yet."


def answer_qa_question(question: str) -> str:
    """Answer a Q&A question based on current analysis context."""

    context = get_analysis_context()

    prompt = f"""You are a helpful bioinformatics analysis assistant. A user is asking a question about their ongoing analysis.

ANALYSIS CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Provide a clear, concise answer based on the analysis context
- If the information is not available in the context, say so politely
- Reference specific results when relevant
- Be technical but understandable
- If the user asks \"why\", provide reasoning based on the analysis

Answer the question:"""

    try:
        llm = st.session_state.agent.llm
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return f"Error generating answer: {exc}\n\nPlease try rephrasing your question."


__all__ = [
    "CHAT_ATTACHMENT_PATTERNS",
    "render_analysis_conversation",
    "get_analysis_context",
    "answer_qa_question",
]
