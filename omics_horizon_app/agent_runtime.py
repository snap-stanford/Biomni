"\"\"\"Agent interaction helpers for the Omics Horizon Streamlit app.\"\"\""

from __future__ import annotations

import hashlib
import os
import re
import time
from datetime import datetime
from typing import Iterable, List, Optional

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

MAX_OBSERVATION_DISPLAY_LENGTH = 2000

__all__ = [
    "MAX_OBSERVATION_DISPLAY_LENGTH",
    "format_agent_output_for_display",
    "parse_step_progress",
    "process_with_agent",
    "add_chat_message",
    "build_agent_input_from_history",
    "display_chat_files",
    "maybe_add_assistant_message",
]


def format_agent_output_for_display(
    raw_text: str, max_observation_length: int = MAX_OBSERVATION_DISPLAY_LENGTH
) -> str:
    """Format agent's raw output into clean, readable Markdown."""
    formatted = raw_text

    incomplete_execute = re.search(
        r"<execute>((?:(?!<execute>|</execute>).)*?)$", formatted, re.DOTALL
    )
    incomplete_code = None

    if incomplete_execute:
        incomplete_code = incomplete_execute.group(1)
        formatted = formatted[: incomplete_execute.start()]

    execution_count = [0]

    def replace_execute_block(match: re.Match) -> str:
        code = match.group(1).strip()
        execution_count[0] += 1

        if code.startswith("#!R"):
            language = "r"
            code = code[3:].strip()
            lang_emoji = "📊"
        elif code.startswith("#!BASH"):
            language = "bash"
            code = code[6:].strip()
            lang_emoji = "🔧"
        else:
            language = "python"
            lang_emoji = "🐍"

        return (
            f"\n\n---\n\n{lang_emoji} **코드 실행 #{execution_count[0]}:**\n"
            f"```{language}\n{code}\n```\n"
        )

    formatted = re.sub(
        r"<execute>\s*(.*?)\s*</execute>",
        replace_execute_block,
        formatted,
        flags=re.DOTALL,
    )

    def replace_observation_block(match: re.Match) -> str:
        result = match.group(1).strip()
        is_error = any(
            keyword in result
            for keyword in ["Error", "Exception", "Traceback", "Failed"]
        )

        if is_error:
            return f"\n\n❌ **실행 오류:**\n```\n{result}\n```\n"

        if len(result) > max_observation_length:
            result_preview = result[:max_observation_length]
            result = (
                result_preview
                + "\n\n... (출력이 길어 생략됨. 총 "
                + str(len(result))
                + "자)"
            )

        return f"\n\n✅ **실행 성공:**\n```\n{result}\n```\n"

    formatted = re.sub(
        r"<observation>\s*(.*?)</observation>",
        replace_observation_block,
        formatted,
        flags=re.DOTALL,
    )

    def replace_solution_block(match: re.Match) -> str:
        solution = match.group(1).strip()
        return f"\n\n---\n\n🎯 **최종 답변:**\n\n{solution}\n\n---\n"

    formatted = re.sub(
        r"<solution>\s*(.*?)</solution>",
        replace_solution_block,
        formatted,
        flags=re.DOTALL,
    )

    formatted = re.sub(
        r"^(\s*\d+\.\s*)\[✓\](.+?)(?:\(completed\))?$",
        r"\1✅ \2",
        formatted,
        flags=re.MULTILINE,
    )
    formatted = re.sub(
        r"^(\s*\d+\.\s*)\[✗\](.+?)$", r"\1❌ \2", formatted, flags=re.MULTILINE
    )
    formatted = re.sub(
        r"^(\s*\d+\.\s*)\[\s\](.+?)$", r"\1⬜ \2", formatted, flags=re.MULTILINE
    )
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)

    if incomplete_code:
        code_text = incomplete_code.strip()
        if code_text.startswith("#!R"):
            language = "r"
            code_text = code_text[3:].strip()
            lang_emoji = "📊"
        elif code_text.startswith("#!BASH"):
            language = "bash"
            code_text = code_text[6:].strip()
            lang_emoji = "🔧"
        else:
            language = "python"
            lang_emoji = "🐍"

        execution_count[0] += 1
        formatted += (
            f"\n\n---\n\n{lang_emoji} **코드 실행 #{execution_count[0]}:** ⏳ 실행 중...\n"
            f"```{language}\n{code_text}\n```\n"
        )

    incomplete_obs = re.search(
        r"<observation>((?:(?!<observation>|</observation>).)*?)$", formatted, re.DOTALL
    )
    if incomplete_obs:
        obs_content = incomplete_obs.group(1).strip()
        formatted = formatted[: incomplete_obs.start()]
        if obs_content:
            formatted += f"\n\n⏳ **실행 중...**\n```\n{obs_content}\n```\n"

    code_blocks: List[str] = []

    def save_code_block(match: re.Match) -> str:
        idx = len(code_blocks)
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{idx}__"

    formatted = re.sub(r"```[\s\S]*?```", save_code_block, formatted)

    lines = formatted.split("\n")
    protected_lines: List[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#") and not stripped.startswith("##"):
            if len(stripped) > 1 and not stripped[1:].strip().startswith("#"):
                indent = len(line) - len(stripped)
                protected_lines.append(" " * indent + "\\# " + stripped[1:])
            else:
                protected_lines.append(line)
        else:
            protected_lines.append(line)

    formatted = "\n".join(protected_lines)

    for idx, code_block in enumerate(code_blocks):
        formatted = formatted.replace(f"__CODE_BLOCK_{idx}__", code_block)

    return formatted


def parse_step_progress(accumulated_text: str) -> dict:
    """Parse current step progress from agent output."""
    all_checkboxes = re.findall(
        r"^\s*(\d+)\.\s*(?:\[([✓✗ ])\]|([✅❌⬜]))\s*(.+?)(?:\s*\(.*?\))?$",
        accumulated_text,
        re.MULTILINE,
    )

    checkbox_dict: dict[int, dict[str, str]] = {}

    for match in all_checkboxes:
        num_str = match[0]
        old_status = match[1]
        emoji_status = match[2]
        title = match[3]

        if old_status == "✓" or emoji_status == "✅":
            status = "completed"
        elif old_status == "✗" or emoji_status == "❌":
            status = "failed"
        else:
            status = "pending"

        step_num = int(num_str)
        checkbox_dict[step_num] = {"status": status, "title": title.strip()}

    parsed_checkboxes = [
        {"num": num, "status": data["status"], "title": data["title"]}
        for num, data in sorted(checkbox_dict.items())
    ]

    total_steps = len(parsed_checkboxes)
    completed_steps = sum(1 for cb in parsed_checkboxes if cb["status"] == "completed")

    current_marker = re.search(
        r"===\s*Step\s+(\d+)[:\s]+([^=]+?)===",
        accumulated_text,
        re.IGNORECASE,
    )

    current_step_num: Optional[int] = None
    current_step_title: Optional[str] = None

    if current_marker:
        current_step_num = int(current_marker.group(1))
        current_step_title = current_marker.group(2).strip()
    elif total_steps > 0 and parsed_checkboxes:
        last_completed_num = 0
        for cb in parsed_checkboxes:
            if cb["status"] == "completed":
                last_completed_num = max(last_completed_num, cb["num"])

        if last_completed_num > 0:
            next_steps = [
                cb for cb in parsed_checkboxes if cb["num"] > last_completed_num
            ]
            if next_steps:
                next_steps.sort(key=lambda x: x["num"])
                current_step_num = next_steps[0]["num"]
                current_step_title = next_steps[0]["title"]

        if current_step_num is None:
            for cb in parsed_checkboxes:
                if cb["status"] == "pending":
                    current_step_num = cb["num"]
                    current_step_title = cb["title"]
                    break

        if current_step_num is None:
            next_step_num = completed_steps + 1
            if next_step_num <= total_steps:
                for cb in parsed_checkboxes:
                    if cb["num"] == next_step_num:
                        current_step_num = next_step_num
                        current_step_title = cb["title"]
                        break

    is_executing = bool(
        re.search(r"<execute>(?!.*</execute>)", accumulated_text, re.DOTALL)
    )
    is_thinking = bool(
        re.search(r"(?:thinking|analyzing|processing)", accumulated_text[-500:], re.IGNORECASE)
    )

    return {
        "total_steps": total_steps,
        "completed_steps": completed_steps,
        "current_step_num": current_step_num,
        "current_step_title": current_step_title,
        "is_executing": is_executing,
        "is_thinking": is_thinking,
    }


def process_with_agent(prompt: str, show_process: bool = False, use_history: bool = False) -> str:
    """Execute the agent with progress feedback and optional streaming UI."""
    original_dir = os.getcwd()
    try:
        os.chdir(st.session_state.work_dir)

        if use_history:
            st.session_state.message_history.append({"role": "user", "content": prompt})
            agent_input: List[BaseMessage] = []
            for msg in st.session_state.message_history:
                if msg["role"] == "user":
                    agent_input.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    agent_input.append(AIMessage(content=msg["content"]))
        else:
            agent_input = [HumanMessage(content=prompt)]

        if show_process:
            progress_bar = st.progress(0)
            status_container = st.empty()
            step_info_container = st.empty()

            with st.expander("🔍 View Analysis Process", expanded=False):
                process_container = st.empty()
                result = ""
                message_stream = st.session_state.agent.go_stream(agent_input)

                for chunk in message_stream:
                    node = chunk[1][1]["langgraph_node"]
                    chunk_data = chunk[1][0]

                    if node in {"generate", "execute"} and hasattr(chunk_data, "content"):
                        result += chunk_data.content
                        formatted_result = format_agent_output_for_display(result)
                        process_container.markdown(formatted_result)

                        progress_info = parse_step_progress(result)
                        if progress_info["total_steps"] > 0:
                            progress = (
                                progress_info["completed_steps"]
                                / progress_info["total_steps"]
                            )
                            progress_bar.progress(min(progress, 0.99))
                            completed = progress_info["completed_steps"]
                            total = progress_info["total_steps"]

                            if (
                                progress_info["current_step_num"]
                                and progress_info["current_step_title"]
                            ):
                                status_emoji = (
                                    "⚙️" if progress_info["is_executing"] else "🧠"
                                )
                                step_info_container.markdown(
                                    f"{status_emoji} **In Progress: Step {progress_info['current_step_num']}/{total}** - "
                                    f"{progress_info['current_step_title']}"
                                )

                            status_container.info(
                                f"✅ Completed: {completed}/{total} steps | ⏳ In Progress: {total - completed} steps"
                            )
                        else:
                            status_container.info("🔍 Planning analysis steps...")

                progress_bar.progress(1.0)
                status_container.success("✅ Analysis complete!")
                step_info_container.empty()

                final_formatted = format_agent_output_for_display(result)
                st.session_state.analysis_process = final_formatted
        else:
            result = ""
            progress_bar = st.progress(0)
            status_container = st.empty()
            step_info_container = st.empty()

            message_stream = st.session_state.agent.go_stream(agent_input)

            for chunk in message_stream:
                node = chunk[1][1]["langgraph_node"]
                chunk_data = chunk[1][0]

                if node in {"generate", "execute"} and hasattr(chunk_data, "content"):
                    result += chunk_data.content

                progress_info = parse_step_progress(result)
                if progress_info["total_steps"] > 0:
                    progress = (
                        progress_info["completed_steps"] / progress_info["total_steps"]
                    )
                    progress_bar.progress(min(progress, 0.99))

                    completed = progress_info["completed_steps"]
                    total = progress_info["total_steps"]

                    if (
                        progress_info["current_step_num"]
                        and progress_info["current_step_title"]
                    ):
                        status_emoji = "⚙️" if progress_info["is_executing"] else "🧠"
                        step_info_container.markdown(
                            f"{status_emoji} **In Progress: Step {progress_info['current_step_num']}/{total}** - "
                            f"{progress_info['current_step_title']}"
                        )

                    status_container.info(
                        f"✅ Completed: {completed}/{total} steps | ⏳ In Progress: {total - completed} steps"
                    )
                else:
                    progress_bar.progress(0.05)
                    status_container.info("🔍 Planning analysis steps...")
                    step_info_container.empty()

            progress_bar.progress(1.0)
            status_container.success("✅ Analysis complete!")
            step_info_container.empty()
            time.sleep(0.5)
            progress_bar.empty()
            status_container.empty()

            if use_history:
                st.session_state.message_history.append(
                    {"role": "assistant", "content": result}
                )
            return result

        if use_history:
            st.session_state.message_history.append(
                {"role": "assistant", "content": result}
            )
        return result
    finally:
        os.chdir(original_dir)


def add_chat_message(
    role: str, content: str, files: Optional[Iterable[str]] = None, timestamp: str | None = None
) -> None:
    """Append a chat message to session history."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M:%S")

    message = {"role": role, "content": content, "timestamp": timestamp}
    if files:
        message["files"] = list(files)

    st.session_state.chat_history.append(message)
    if "message_history" not in st.session_state:
        st.session_state.message_history = []
    st.session_state.message_history.append({"role": role, "content": content})


def _normalize_file_list(files: Optional[Iterable[str]]) -> tuple[str, ...]:
    if not files:
        return ()
    normalized = {os.path.abspath(path) for path in files if path}
    return tuple(sorted(normalized))


def _compute_digest(text: str, files: Optional[Iterable[str]] = None) -> str:
    digest_payload = text or ""
    if files:
        digest_payload += "||" + "|".join(_normalize_file_list(files))
    return hashlib.md5(digest_payload.encode("utf-8")).hexdigest()


def maybe_add_assistant_message(content: str, files: Optional[Iterable[str]] = None) -> bool:
    """Add assistant message if it's not a duplicate of the last one."""
    if not content:
        return False
    file_list = _normalize_file_list(files)
    digest = _compute_digest(content, file_list)
    last_digest = st.session_state.get("last_assistant_digest")
    if last_digest == digest:
        return False
    add_chat_message("assistant", content, files=file_list)
    st.session_state.last_assistant_digest = digest
    return True


def build_agent_input_from_history(
    initial_prompt: Optional[str] = None, include_initial: bool = True
) -> List[BaseMessage]:
    """Build agent input from chat history."""
    agent_input: List[BaseMessage] = []
    if include_initial and initial_prompt:
        agent_input.append(HumanMessage(content=initial_prompt))

    for message in st.session_state.get("chat_history", []):
        if message["role"] == "user":
            agent_input.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            agent_input.append(AIMessage(content=message["content"]))

    return agent_input


def display_chat_files(files: Optional[Iterable[str]]) -> None:
    """Display files attached to a chat message."""
    if not files:
        return

    st.markdown("**📎 첨부 파일:**")
    for file_path in files:
        filename = os.path.basename(file_path)
        col1, col2 = st.columns([3, 1])
        with col1:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                st.image(file_path, use_container_width=True, caption=filename)
            else:
                st.markdown(f"📄 {filename}")
        with col2:
            try:
                with open(file_path, "rb") as f:
                    file_data = f.read()
                st.download_button(
                    label="⬇️",
                    data=file_data,
                    file_name=filename,
                    mime="application/octet-stream",
                    key=f"chat_download_{filename}_{len(st.session_state.chat_history)}",
                )
            except Exception:
                st.error(f"파일 로드 실패: {filename}")
