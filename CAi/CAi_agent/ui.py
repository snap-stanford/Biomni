import os
import re
import shutil
from datetime import datetime
from typing import Any

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage

from CAi.config import WORKSPACE_DIR
from CAi.logger import get_logger

logger = get_logger(__name__)


class AgentGradioUI:
    """Gradio UI wrapper for the agent with file management and chat interface."""

    SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
    SUPPORTED_TEXT_EXTENSIONS = (".txt", ".csv", ".py", ".json", ".md", ".sh", ".yaml", ".yml")
    MAX_TEXT_PREVIEW_CHARS = 10000

    def __init__(self, agent, thread_id: int = 42, require_verification: bool = False):
        self.agent = agent
        self.thread_id = thread_id
        self.require_verification = require_verification

        if not hasattr(self.agent, "main_history_copy"):
            self.agent.main_history_copy = []

        self.available_access_codes = ["CAi"]
        self.workspace_dir = str((WORKSPACE_DIR / "agent_workspace").resolve())
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.agent.workspace_dir = self.workspace_dir

    # ========== Access Control ==========

    def _verify_access_code(self, code: str) -> tuple[gr.update, gr.update, gr.update]:
        """Verify access code and toggle UI visibility."""
        if code in self.available_access_codes:
            return (
                gr.update(visible=False),  # Hide verification
                gr.update(visible=True),  # Show main interface
                gr.update(visible=False),  # Hide error message
            )
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="Incorrect access code.", visible=True),
        )

    # ========== File Management ==========

    def _get_current_files(self) -> list[list[str]]:
        """Get files in workspace as 2D array for Dataframe display."""
        if not os.path.exists(self.workspace_dir):
            return [["(暂无文件)"]]

        files = [f for f in os.listdir(self.workspace_dir) if os.path.isfile(os.path.join(self.workspace_dir, f))]
        return [[f] for f in files] if files else [["(暂无文件)"]]

    def _get_dropdown_choices(self) -> list[str]:
        """Get files in workspace as 1D list for dropdown."""
        if not os.path.exists(self.workspace_dir):
            return []

        return [f for f in os.listdir(self.workspace_dir) if os.path.isfile(os.path.join(self.workspace_dir, f))]

    def _clear_workspace(self) -> tuple[list[list[str]], gr.update]:
        """Clear workspace files only, preserve chat history."""
        try:
            if os.path.exists(self.workspace_dir):
                shutil.rmtree(self.workspace_dir)
            os.makedirs(self.workspace_dir, exist_ok=True)
            return self._get_current_files(), gr.update(choices=[])
        except Exception as e:
            logger.error(f"清空工作区失败: {e}")
            return self._get_current_files(), gr.update(choices=self._get_dropdown_choices())

    def _clear_all(self) -> tuple[list, list, list[list[str]], gr.update]:
        """Clear workspace files and chat history."""
        files, dropdown = self._clear_workspace()
        self.agent.main_history_copy = []
        if hasattr(self.agent, "_conversation_state"):
            self.agent._conversation_state = None
        if hasattr(self.agent, "log"):
            self.agent.log = []
        return [], [], files, dropdown

    def _save_conversation_history(self) -> tuple[list[list[str]], gr.update]:
        """Export conversation history to PDF via parent agent API."""
        try:
            if not hasattr(self.agent, "save_conversation_history"):
                gr.Warning("当前 Agent 不支持会话导出。")
                return self._get_current_files(), gr.update(choices=self._get_dropdown_choices())

            has_state = bool(getattr(self.agent, "_conversation_state", None))
            has_history = bool(getattr(self.agent, "main_history_copy", []))
            if not has_state and not has_history:
                gr.Warning("暂无可导出的会话，请先进行一次对话。")
                return self._get_current_files(), gr.update(choices=self._get_dropdown_choices())

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.workspace_dir, f"conversation_history_{timestamp}.pdf")
            self.agent.save_conversation_history(output_path, include_images=True, save_pdf=True)

            if os.path.exists(output_path):
                gr.Info(f"会话已导出: {os.path.basename(output_path)}")
            else:
                gr.Warning("导出执行完成，但未检测到 PDF 文件，请检查日志。")

            return self._get_current_files(), gr.update(choices=self._get_dropdown_choices())
        except Exception as e:
            logger.exception("导出会话失败")
            gr.Warning(f"导出失败: {e}")
            return self._get_current_files(), gr.update(choices=self._get_dropdown_choices())

    # ========== File Preview ==========

    def _preview_selected_file(self, evt: gr.SelectData) -> tuple[gr.update, gr.update, gr.update, gr.update]:
        """Handle file preview based on file type."""
        import base64

        filename = evt.value
        hide_all = (gr.update(visible=False),) * 4

        if filename == "(暂无文件)":
            return hide_all

        file_path = os.path.join(self.workspace_dir, filename)
        if not os.path.exists(file_path):
            return hide_all

        ext = os.path.splitext(filename)[1].lower()

        # Image preview
        if ext in self.SUPPORTED_IMAGE_EXTENSIONS:
            return (
                gr.update(value=file_path, visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        # Text/code preview
        if ext in self.SUPPORTED_TEXT_EXTENSIONS:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read(self.MAX_TEXT_PREVIEW_CHARS)
                    if len(content) == self.MAX_TEXT_PREVIEW_CHARS:
                        content += f"\n\n... (文件过长，仅展示前 {self.MAX_TEXT_PREVIEW_CHARS} 字符) ..."
                return (
                    gr.update(visible=False),
                    gr.update(value=content, visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
            except Exception:
                return hide_all

        # PDF preview with base64 embedding
        if ext == ".pdf":
            try:
                with open(file_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode("utf-8")
                pdf_html = (
                    f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                    f'width="100%" height="400px" style="border: none;"></iframe>'
                )
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value=pdf_html, visible=True),
                    gr.update(visible=False),
                )
            except Exception:
                return hide_all

        # Fallback for other file types
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=file_path, visible=True),
        )

    # ========== History Management ==========

    def _finalize_inner_history(self, history: list[dict]) -> list[dict]:
        """Mark all pending code blocks as done."""
        for item in history:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata")
            if isinstance(metadata, dict) and metadata.get("status") == "pending":
                metadata["status"] = "done"
        return history

    def _build_agent_messages(self, history: list[dict]) -> list:
        """Convert main_history_copy to LangChain message list."""
        messages = []
        for msg in history:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and "Executor is working" not in content:
                messages.append(AIMessage(content=content))
        return messages

    # ========== File Processing ==========

    def _process_observation_files(self, observation: str, temp_inner_history: list[dict]) -> None:
        """Extract and render files from observation text."""
        SUPPORTED_EXTENSIONS = self.SUPPORTED_IMAGE_EXTENSIONS + (".pdf", ".csv")

        if not any(ext in observation for ext in SUPPORTED_EXTENSIONS):
            return

        pattern = r"(\S+?(?:\.png|\.jpg|\.jpeg|\.gif|\.bmp|\.webp|\.pdf|\.csv))"
        matches = re.findall(pattern, observation)

        # Filter out invalid matches
        valid_matches = [
            m
            for m in matches
            if not (m.startswith("Warning:") or m.startswith("Error:") or m.startswith("'") or m.startswith("."))
        ]

        for file_path in valid_matches:
            file_path = file_path.strip("\"'").strip()

            # Try multiple candidate paths
            candidates = [
                file_path,
                os.path.join(self.workspace_dir, file_path),
                os.path.join(os.getcwd(), file_path),
            ]
            abs_path = next((p for p in candidates if os.path.exists(p)), None)

            if abs_path:
                # Copy to workspace if not already there
                if os.path.dirname(abs_path) != self.workspace_dir:
                    try:
                        shutil.copy2(abs_path, self.workspace_dir)
                    except Exception:
                        pass

                # Add to history based on file type
                if file_path.lower().endswith((".pdf", ".csv")):
                    temp_inner_history.append({"role": "assistant", "content": f"📄 生成了文件: {abs_path}"})
                else:
                    temp_inner_history.append(
                        {
                            "role": "assistant",
                            "content": (abs_path,),
                            "metadata": {"title": "🖼️ Image Output"},
                        }
                    )

    # ========== Response Generation ==========

    def _build_prompt_with_files(self, text_input: str, files: list[str], ref_files: list[str]) -> tuple[str, str]:
        """Build agent prompt and display text with file references."""
        agent_prompt = text_input
        display_text = text_input
        uploaded_filenames = []

        agent_prompt += f"\n\n[系统规则]: 你的指定工作目录是 '{self.workspace_dir}'。请将你在执行任务期间生成的所有文件（如图片、CSV、PDF等）直接保存到该绝对路径下。"
        # Process newly uploaded files
        for file_path in files:
            file_name = os.path.basename(file_path)
            target_path = os.path.join(self.workspace_dir, file_name)
            shutil.copy2(file_path, target_path)
            uploaded_filenames.append(file_name)
            agent_prompt += f"\n\n[系统消息]: 用户新上传了文件至: {target_path}。请按需读取分析。"

        # Process referenced files
        for file_name in ref_files:
            target_path = os.path.join(self.workspace_dir, file_name)
            uploaded_filenames.append(file_name)
            agent_prompt += f"\n\n[系统消息]: 用户指定引用了工作区已有文件: {target_path}。请按需读取分析。"

        # Generate display text
        if uploaded_filenames:
            file_display = ", ".join(uploaded_filenames)
            display_text = (
                f"{display_text}\n\n📎 [关联文件: {file_display}]" if display_text else f"📎 [关联文件: {file_display}]"
            )

        return agent_prompt, display_text

    def _stream_agent_execution(self, new_inner: list[dict], new_main: list[dict]):
        """Stream agent execution and process responses - yields updates."""
        agent_messages = self._build_agent_messages(self.agent.main_history_copy)
        inputs = {"messages": agent_messages, "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": self.thread_id}}
        original_msg_count = len(agent_messages)

        current_round_inner = []
        solution_content = None
        last_stream_state = None
        last_yield_count = 0
        processed_msg_count = 0  # Track how many messages we've processed

        for stream_state in self.agent.app.stream(inputs, stream_mode="values", config=config):
            last_stream_state = stream_state
            all_new_messages = stream_state["messages"][original_msg_count:]

            # Only process messages we haven't seen yet
            new_messages = all_new_messages[processed_msg_count:]
            processed_msg_count = len(all_new_messages)

            for msg in new_messages:
                if not isinstance(msg.content, str):
                    continue

                self._process_message_content(msg.content, current_round_inner)

                # Extract solution
                solution_match = re.search(r"<solution>(.*?)</solution>", msg.content, re.DOTALL)
                if solution_match:
                    solution_content = solution_match.group(1).strip()

            # Update main conversation
            updated_main = new_main[:-1] + [
                {
                    "role": "assistant",
                    "content": solution_content or "处理中...",
                    "metadata": {"title": "✅ Answer" if solution_content else "⏳ Processing"},
                }
            ]

            # Merge history and current round logs
            display_inner = new_inner + current_round_inner

            # Yield when there's new content
            if len(current_round_inner) > last_yield_count:
                last_yield_count = len(current_round_inner)
                logger.debug(
                    f"流式 yield: round={len(current_round_inner)}, "
                    f"total={len(display_inner)}, solution={bool(solution_content)}"
                )
                yield (display_inner, updated_main)

        # Finalize
        final_inner = new_inner + current_round_inner
        self._finalize_inner_history(final_inner)

        if last_stream_state is not None:
            self.agent._conversation_state = last_stream_state

        # Extract final solution if not found
        if not solution_content and last_stream_state and last_stream_state.get("messages"):
            solution_content = self._extract_final_solution(last_stream_state["messages"][-1].content)

        # Add completion marker
        final_inner = final_inner + [
            {"role": "assistant", "content": "👈 返回结果完成", "metadata": {"title": "🔄 Complete"}}
        ]

        # Ensure the final solution is always a string to avoid later type issues
        solution_content = solution_content or "任务执行完毕，请查看右侧执行日志。"

        # Final main conversation - collapsible answer
        final_main = new_main[:-1] + [
            {"role": "assistant", "content": solution_content, "metadata": {"title": "✅ Answer"}}
        ]

        # Save to history
        self.agent.main_history_copy.append({"role": "assistant", "content": solution_content})

        # Final yield
        logger.debug(f"最终 yield: inner={len(final_inner)}, main={len(final_main)}")
        yield (final_inner, final_main)

    def _process_message_content(self, content: str, current_round_inner: list[dict]) -> None:
        """Process message content and extract thinking, code, and observations."""
        # Extract thinking
        tag_positions = [content.find(tag) for tag in ["<execute>", "<solution>", "<observation>"] if tag in content]
        if tag_positions:
            thinking = content[: min(tag_positions)].strip()
            if thinking:
                current_round_inner.append(
                    {"role": "assistant", "content": thinking, "metadata": {"title": "🤔 Reasoning"}}
                )

        # Extract code
        execute_match = re.search(r"<execute>(.*?)</execute>", content, re.DOTALL)
        if execute_match:
            code = execute_match.group(1).strip()
            language = self._detect_code_language(code)
            code = self._clean_code_prefix(code, language)

            current_round_inner.append(
                {
                    "role": "assistant",
                    "content": f"##### Code: \n```{language}\n{code}\n```",
                    "metadata": {"title": f"🛠️ Executing {language}...", "status": "pending"},
                }
            )

        # Extract observation
        observation_match = re.search(r"<observation>(.*?)</observation>", content, re.DOTALL)
        if observation_match:
            observation = observation_match.group(1).strip()

            # Mark previous as done
            if current_round_inner:
                last_metadata = (
                    current_round_inner[-1].get("metadata") if isinstance(current_round_inner[-1], dict) else None
                )
                if isinstance(last_metadata, dict) and last_metadata.get("status") == "pending":
                    last_metadata["status"] = "done"

            current_round_inner.append(
                {
                    "role": "assistant",
                    "content": f"##### Observation: \n```\n{observation}\n```",
                    "metadata": {"title": "📊 Result", "collapsed": True},
                }
            )
            self._process_observation_files(observation, current_round_inner)

        # Extract solution
        solution_match = re.search(r"<solution>(.*?)</solution>", content, re.DOTALL)
        if solution_match:
            solution = solution_match.group(1).strip()
            current_round_inner.append(
                {
                    "role": "assistant",
                    "content": solution,
                    "metadata": {"title": "✅ Solution"},
                }
            )

    def _detect_code_language(self, code: str) -> str:
        """Detect programming language from code prefix."""
        if code.startswith("#!R"):
            return "r"
        elif code.startswith("#!BASH") or code.startswith("#!CLI"):
            return "bash"
        return "python"

    def _clean_code_prefix(self, code: str, language: str) -> str:
        """Remove language prefix from code."""
        if language == "r":
            return re.sub(r"^#!R", "", code, count=1).strip()
        elif language == "bash":
            return re.sub(r"^#!(?:BASH|CLI)", "", code, count=1).strip()
        return code

    def _extract_final_solution(self, final_message: str) -> str:
        """Extract final solution from message content."""
        cleaned = re.sub(r"<execute>.*?</execute>|<observation>.*?</observation>", "", final_message, flags=re.DOTALL)
        cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned).strip()
        return cleaned or "任务执行完毕，请查看右侧执行日志。"

    def _handle_error(
        self, error: Exception, inner_history: list[dict], main_history: list[dict]
    ) -> tuple[list[dict], list[dict], list[list[str]], gr.update]:
        """Handle execution errors and return error state."""
        error_msg = f"**系统执行错误**: `{str(error)}`\n请检查终端(Terminal)获取详细报错日志。"

        self._finalize_inner_history(inner_history)
        error_main = (
            main_history[:-1] + [{"role": "assistant", "content": error_msg}]
            if main_history and main_history[-1]["role"] == "assistant"
            else main_history + [{"role": "assistant", "content": error_msg}]
        )

        return (
            inner_history,
            error_main,
            self._get_current_files(),
            gr.update(choices=self._get_dropdown_choices()),
        )

    def generate_response(
        self,
        prompt_input: dict[str, Any] | None,
        ref_files: list[str] | None,
        inner_history: list[dict] | None = None,
        main_history: list[dict] | None = None,
    ):
        """Process user input and stream agent responses."""
        try:
            # Initialize
            main_history = main_history or []
            inner_history = inner_history or []
            prompt_input = prompt_input or {"text": "", "files": []}
            ref_files = ref_files or []

            text_input = prompt_input.get("text", "")
            files = prompt_input.get("files", [])

            # Return early if no input
            if not text_input and not files and not ref_files:
                yield (
                    inner_history,
                    main_history,
                    self._get_current_files(),
                    gr.update(choices=self._get_dropdown_choices()),
                )
                return

            # Build agent prompt and display text
            agent_prompt, display_text = self._build_prompt_with_files(text_input, files, ref_files)

            # Update histories (immutable)
            new_main = main_history + [
                {"role": "user", "content": display_text},
                {"role": "assistant", "content": "Executor is working on it 👉"},
            ]

            new_inner = inner_history + [
                {"role": "assistant", "content": "🚀 开始执行任务...", "metadata": {"title": "⏳ Starting"}}
            ]

            self.agent.main_history_copy.append({"role": "user", "content": agent_prompt})

            # Initial yield
            logger.debug(f"首次 yield: inner={len(new_inner)}, main={len(new_main)}")
            yield (new_inner, new_main, self._get_current_files(), gr.update(choices=self._get_dropdown_choices()))

            # Stream agent execution with yields
            for final_inner, final_main in self._stream_agent_execution(new_inner, new_main):
                yield (
                    final_inner,
                    final_main,
                    self._get_current_files(),
                    gr.update(choices=self._get_dropdown_choices()),
                )

        except Exception as e:
            logger.exception("Backend Execution Error")
            yield self._handle_error(e, inner_history, main_history)

    # ========== UI Builder ==========

    def build_ui(self):
        """Build and return the Gradio interface."""
        from .ui_layout import build_layout

        return build_layout(self)
