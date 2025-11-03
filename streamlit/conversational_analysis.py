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
    # Remove figure tokens from the middle of sections (they don't render well)
    cleaned = re.sub(r"\[\[FIGURE::.*?\]\]", "", cleaned, flags=re.DOTALL)
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


def _remove_prompt_artifacts(text: str, prompt: str = "") -> str:
    """Remove prompt artifacts from agent output.
    
    Args:
        text: The text to clean
        prompt: The actual prompt text to remove (if provided, removes it directly)
    """
    cleaned = text
    
    # First, remove the actual prompt if provided (most reliable method)
    if prompt:
        # Remove the prompt text directly
        cleaned = cleaned.replace(prompt, "")
        # Also try removing with leading/trailing whitespace variations
        cleaned = cleaned.replace(prompt.strip(), "")
        # Remove prompt if it appears at the start
        if cleaned.startswith(prompt):
            cleaned = cleaned[len(prompt):]
        if cleaned.startswith(prompt.strip()):
            cleaned = cleaned[len(prompt.strip()):]
    
    # Remove common boilerplate patterns that might appear
    cleaned = re.sub(r"^\s*Here is my plan.*?$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^\s*I will now proceed.*?$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
    
    # Clean up multiple consecutive newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    
    return cleaned.strip()


def _extract_final_response(raw_text: str) -> str:
    if not raw_text:
        return ""

    solution_matches = re.findall(r"<solution>(.*?)</solution>", raw_text, re.DOTALL)
    if solution_matches:
        # Get the last solution block
        candidate = solution_matches[-1]
        cleaned = _sanitize_solution_text(candidate)
        if cleaned:
            # # Also include any figure tokens that appear after the solution block
            # solution_end_pos = raw_text.rfind("</solution>")
            # if solution_end_pos > 0:
            #     text_after_solution = raw_text[solution_end_pos + len("</solution>"):].strip()
            #     # Check if there are figure tokens after the solution block
            #     if "[[FIGURE::" in text_after_solution:
            #         # Append figures after the cleaned solution content
            #         cleaned = cleaned + "\n\n" + text_after_solution
            return cleaned

    observation_matches = re.findall(
        r"<observation>(.*?)</observation>", raw_text, re.DOTALL
    )
    if observation_matches:
        # Get the last observation block
        candidate = observation_matches[-1]
        cleaned = _sanitize_solution_text(candidate)
        if cleaned:
            # Also include any figure tokens that appear after the observation block
            observation_end_pos = raw_text.rfind("</observation>")
            if observation_end_pos > 0:
                text_after_observation = raw_text[observation_end_pos + len("</observation>"):].strip()
                # Check if there are figure tokens after the observation block
                if "[[FIGURE::" in text_after_observation:
                    # Append figures after the cleaned observation content
                    cleaned = cleaned + "\n\n" + text_after_observation
            return cleaned

    return _sanitize_solution_text(raw_text)


def render_analysis_conversation2() -> None:
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
                st.markdown(
                    format_agent_output_for_display(message["content"]),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(message["content"])
            if message.get("files"):
                display_chat_files(message["files"])

    user_input = st.chat_input("Enter your message...", key="user_chat_input")
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
        # Track the last displayed formatted text to prevent duplicate display
        last_text = ""
        # Store the prompt to remove it from output dynamically
        prompt_to_remove = prompt.strip()
        # Store prompt in session state for use after streaming completes
        st.session_state.last_prompt = prompt_to_remove
        

        def _compute_delta(node_id: str, new_text: str) -> str:
            previous = last_node_text.get(node_id, "")
            
            if previous:
                if new_text.startswith(previous):
                    # Incremental: only return the new part
                    delta = new_text[len(previous) :]
                else:
                    delta = new_text[len(previous):] if previous in new_text else new_text

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
                            chunk_text = "".join(
                                item for item in content if isinstance(item, str)
                            )
                        else:
                            chunk_text = content if isinstance(content, str) else ""
                        
                        # Simple accumulation: just append chunks to result_text in order
                        # We'll only update display when a block completes (closing tag appears)
                        if chunk_text:
                            # Compute node-local delta to avoid repeated full re-sends
                            delta_text = _compute_delta(node, chunk_text)
                            if not delta_text:
                                continue
                                
                            if len(delta_text)>400:
                                if "<execute>" in result_text:
                                    result_text += '</execute>'
                                if "<observation>" in result_text:
                                    result_text += '</observation>'
                                if "<solution>" in result_text:
                                    result_text += '</solution>'
                                continue
                            
                            result_text += delta_text
                            
                            if result_text != last_text:

                                cleaned_result = _remove_prompt_artifacts(
                                    result_text, prompt_to_remove
                                )
                                
                                # Render the full formatted content into the placeholder (replace)
                                full_formatted_text = format_agent_output_for_display(
                                    cleaned_result
                                )

                                # Remove prompt artifacts from formatted text as well (belt-and-suspenders)
                                full_formatted_text = _remove_prompt_artifacts(
                                    full_formatted_text, prompt_to_remove
                                )
                            
                                message_placeholder.markdown(
                                    full_formatted_text, unsafe_allow_html=True
                                )
                                last_text = result_text
                            
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
    
    # Insert figure tokens into observation/solution blocks before extracting final response
    # This ensures figures are included in the extracted content
    if attachments:
        figure_tokens = []
        attachment_times = {}
        for path in attachments:
            abs_path = os.path.abspath(path)
            if not os.path.isfile(abs_path):
                continue
            figure_token = f"[[FIGURE::{abs_path}]]"
            figure_tokens.append(figure_token)
            # Store creation time for sorting
            attachment_times[figure_token] = os.path.getmtime(abs_path)
        
        # Check if figure tokens are already in result_text
        has_figure_tokens_in_text = any(
            f"[[FIGURE::{os.path.abspath(path)}]]" in result_text 
            for path in attachments 
            if os.path.isfile(os.path.abspath(path))
        )
        
        if not has_figure_tokens_in_text and figure_tokens:
            # Sort figures by creation time (oldest first)
            sorted_figures = sorted(figure_tokens, key=lambda x: attachment_times.get(x, 0))
            
            # Try to insert figures inside solution blocks (preferred) or after observation blocks
            # Note: Don't insert into observation blocks because they get converted to code blocks
            # which would prevent figure rendering
            solution_matches = list(re.finditer(r"<solution>(.*?)</solution>", result_text, re.DOTALL))
            if solution_matches:
                # Insert figures into the last solution block (best option)
                last_solution = solution_matches[-1]
                solution_content = last_solution.group(1)
                solution_start = last_solution.start()
                solution_end = last_solution.end()
                # Insert figures before closing tag
                new_solution_content = solution_content + "\n\n" + "\n\n".join(sorted_figures)
                result_text = result_text[:solution_start] + f"<solution>{new_solution_content}</solution>" + result_text[solution_end:]
            else:
                # If no solution blocks, insert after observation blocks (not inside!)
                observation_matches = list(re.finditer(r"<observation>(.*?)</observation>", result_text, re.DOTALL))
                if observation_matches:
                    # Insert figures AFTER the last observation block (not inside)
                    last_observation = observation_matches[-1]
                    obs_end = last_observation.end()
                    # Insert figures after the closing tag
                    result_text = result_text[:obs_end] + "\n\n" + "\n\n".join(sorted_figures) + result_text[obs_end:]
                else:
                    # No blocks found, append to result_text (will be included in final extraction)
                    result_text = result_text + "\n\n" + "\n\n".join(sorted_figures)
    
    final_text = _extract_final_response(result_text)

    if final_text:
        # Store the full result_text (including code execution, observations, etc.)
        # instead of just the final answer, so the analysis process is preserved
        cleaned_result = _remove_prompt_artifacts(result_text, prompt_to_remove)
        
        # Don't display again now (already streamed). Persist full content for chat history.
        maybe_add_assistant_message(cleaned_result, files=attachments)
    st.rerun()
    
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
                st.markdown(
                    format_agent_output_for_display(message["content"]),
                    unsafe_allow_html=True,
                )
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
        # Track the last displayed formatted text to prevent duplicate display
        last_displayed_text = ""
        # Track the number of completed blocks to detect when new blocks finish
        last_completed_block_count = 0
        # Store the prompt to remove it from output dynamically
        prompt_to_remove = prompt.strip()
        # Store prompt in session state for use after streaming completes
        st.session_state.last_prompt = prompt_to_remove

        def _compute_delta(node_id: str, new_text: str) -> str:
            previous = last_node_text.get(node_id, "")
            
            # Logging for debugging duplicate output
            prev_len = len(previous) if previous else 0
            new_len = len(new_text) if new_text else 0
            
            if previous:
                if new_text.startswith(previous):
                    # Incremental: only return the new part
                    delta = new_text[len(previous) :]
                    print(f"\n[DELTA] {node_id}: INCREMENTAL")
                    print(f"  prev_len: {prev_len}, new_len: {new_len}, delta_len: {len(delta)}")
                    print(f"  prev_tail: ...{previous[-100:] if prev_len > 100 else previous}")
                    print(f"  new_tail: ...{new_text[-100:] if new_len > 100 else new_text}")
                    print(f"  delta: {delta[:200] if len(delta) > 200 else delta}...")
                elif new_text == previous:
                    # Exact duplicate - skip
                    print(f"\n[DELTA] {node_id}: EXACT DUPLICATE - SKIPPED")
                    return ""
                else:
                    # Node completed and sent full text again
                    print(f"\n[DELTA] {node_id}: FULL RESEND")
                    print(f"  prev_len: {prev_len}, new_len: {new_len}")
                    print(f"  prev_tail: ...{previous[-100:] if prev_len > 100 else previous}")
                    print(f"  new_tail: ...{new_text[-100:] if new_len > 100 else new_text}")
                    
                    # Check if this is just a duplicate of what we already have
                    if len(new_text) <= len(previous) and new_text == previous[:len(new_text)]:
                        # Shorter version of previous - skip
                        print(f"  → SHORTER VERSION - SKIPPED")
                        return ""
                    
                    # Full replacement - only add if it's truly different
                    # Usually this means node restarted, so take it as new
                    delta = new_text[len(previous):] if previous in new_text else new_text
                    print(f"  delta_len: {len(delta)}")
                    print(f"  previous in new_text: {previous in new_text}")
                    if previous in new_text:
                        print(f"  delta: {delta[:200] if len(delta) > 200 else delta}...")
                    else:
                        print(f"  FULL REPLACEMENT (node restarted?): {new_text[:200] if new_len > 200 else new_text}...")
            else:
                delta = new_text
                print(f"\n[DELTA] {node_id}: FIRST CHUNK")
                print(f"  new_len: {new_len}")
                print(f"  new_text: {new_text[:200] if new_len > 200 else new_text}...")
            
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
                            chunk_text = "".join(
                                item for item in content if isinstance(item, str)
                            )
                        else:
                            chunk_text = content if isinstance(content, str) else ""
                        
                        # Simple accumulation: just append chunks to result_text in order
                        # We'll only update display when a block completes (closing tag appears)
                        if chunk_text:
                            # Compute node-local delta to avoid repeated full re-sends
                            delta_text = _compute_delta(node, chunk_text)
                            if not delta_text:
                                continue

                            # Cross-node de-duplication: trim any overlap where the end of the
                            # already-assembled text matches the beginning of this delta
                            if result_text and delta_text:
                                max_overlap_len = min(len(result_text), len(delta_text))
                                overlap_len = 0
                                for i in range(max_overlap_len, 0, -1):
                                    if result_text[-i:] == delta_text[:i]:
                                        overlap_len = i
                                        break
                                if overlap_len:
                                    delta_text = delta_text[overlap_len:]

                            if not delta_text:
                                continue

                            result_text += delta_text

                            # Check if this delta completed a block by looking for closing tags
                            block_completed = (
                                "</execute>" in delta_text
                                or "</observation>" in delta_text
                                or "</solution>" in delta_text
                            )

                            # Count completed blocks in result_text
                            completed_block_count = (
                                result_text.count("</execute>")
                                + result_text.count("</observation>")
                                + result_text.count("</solution>")
                            )

                            # Only update display if a block just completed and count increased
                            if block_completed and completed_block_count > last_completed_block_count:
                                # Remove prompt artifacts from result_text before formatting
                                cleaned_result = _remove_prompt_artifacts(
                                    result_text, prompt_to_remove
                                )

                                # Render the full formatted content into the placeholder (replace)
                                full_formatted_text = format_agent_output_for_display(
                                    cleaned_result
                                )

                                # Remove prompt artifacts from formatted text as well (belt-and-suspenders)
                                full_formatted_text = _remove_prompt_artifacts(
                                    full_formatted_text, prompt_to_remove
                                )

                                if full_formatted_text != last_displayed_text:
                                    message_placeholder.markdown(
                                        full_formatted_text, unsafe_allow_html=True
                                    )
                                    last_displayed_text = full_formatted_text
                                    last_completed_block_count = completed_block_count
                            
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
    
    # Insert figure tokens into observation/solution blocks before extracting final response
    # This ensures figures are included in the extracted content
    if attachments:
        figure_tokens = []
        attachment_times = {}
        for path in attachments:
            abs_path = os.path.abspath(path)
            if not os.path.isfile(abs_path):
                continue
            figure_token = f"[[FIGURE::{abs_path}]]"
            figure_tokens.append(figure_token)
            # Store creation time for sorting
            attachment_times[figure_token] = os.path.getmtime(abs_path)
        
        # Check if figure tokens are already in result_text
        has_figure_tokens_in_text = any(
            f"[[FIGURE::{os.path.abspath(path)}]]" in result_text 
            for path in attachments 
            if os.path.isfile(os.path.abspath(path))
        )
        
        if not has_figure_tokens_in_text and figure_tokens:
            # Sort figures by creation time (oldest first)
            sorted_figures = sorted(figure_tokens, key=lambda x: attachment_times.get(x, 0))
            
            # Try to insert figures inside solution blocks (preferred) or after observation blocks
            # Note: Don't insert into observation blocks because they get converted to code blocks
            # which would prevent figure rendering
            solution_matches = list(re.finditer(r"<solution>(.*?)</solution>", result_text, re.DOTALL))
            if solution_matches:
                # Insert figures into the last solution block (best option)
                last_solution = solution_matches[-1]
                solution_content = last_solution.group(1)
                solution_start = last_solution.start()
                solution_end = last_solution.end()
                # Insert figures before closing tag
                new_solution_content = solution_content + "\n\n" + "\n\n".join(sorted_figures)
                result_text = result_text[:solution_start] + f"<solution>{new_solution_content}</solution>" + result_text[solution_end:]
            else:
                # If no solution blocks, insert after observation blocks (not inside!)
                observation_matches = list(re.finditer(r"<observation>(.*?)</observation>", result_text, re.DOTALL))
                if observation_matches:
                    # Insert figures AFTER the last observation block (not inside)
                    last_observation = observation_matches[-1]
                    obs_end = last_observation.end()
                    # Insert figures after the closing tag
                    result_text = result_text[:obs_end] + "\n\n" + "\n\n".join(sorted_figures) + result_text[obs_end:]
                else:
                    # No blocks found, append to result_text (will be included in final extraction)
                    result_text = result_text + "\n\n" + "\n\n".join(sorted_figures)
    
    final_text = _extract_final_response(result_text)

    if final_text:
        # Store the full result_text (including code execution, observations, etc.)
        # instead of just the final answer, so the analysis process is preserved
        # Remove prompt artifacts but keep the full analysis process
        prompt_to_remove = getattr(st.session_state, 'last_prompt', "")
        cleaned_result = _remove_prompt_artifacts(result_text, prompt_to_remove)
        
        # Don't display again now (already streamed). Persist full content for chat history.
        maybe_add_assistant_message(cleaned_result, files=attachments)
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
