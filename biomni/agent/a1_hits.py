"""
Refactored A1_HITS Agent Module

This module provides a refactored version of the A1_HITS agent with improved:
- Code organization and readability
- Separation of concerns
- Reduced code duplication
- Better testability
"""

import glob
import re
import os
import time
import base64
from typing import Literal, TypedDict, List, Dict, Any, Set
from pathlib import Path
from pydantic import BaseModel, Field
from PIL import Image

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain_aws import BedrockEmbeddings
from biomni.env_desc import data_lake_dict, library_content_dict
from biomni.tool.support_tools import run_python_repl
from biomni.utils import (
    pretty_print,
    run_bash_script,
    run_r_code,
    run_with_timeout,
    textify_api_dict,
)
from biomni.agent.a1 import A1
from .a1 import AgentState
from biomni.llm import get_llm
from langchain_community.vectorstores import FAISS

try:
    from langchain.chains import ConversationalRetrievalChain
except:
    from langchain_classic.chains import ConversationalRetrievalChain

from biomni.model.retriever import ToolRetrieverByRAG
from biomni.utils.resource_filter import (
    apply_resource_filters,
    load_resource_filter_config,
    filter_data_lake_dict,
)
from biomni.config import default_config


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================


class Config:
    """Configuration constants for A1_HITS agent."""

    TOOL_LLM_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    ERROR_FIXING_LLM_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    # Timeout settings
    DEFAULT_TIMEOUT_SECONDS = 600  # 10 minutes

    # Message limits
    MAX_OUTPUT_LENGTH = 20000
    MAX_FILE_PREVIEW_LINES = 5
    MAX_CHARS_PER_LINE = 200

    # Workflow settings
    DEFAULT_RECURSION_LIMIT = 500
    DEFAULT_THREAD_ID = 42

    # Error handling
    MAX_PARSING_ERROR_COUNT = 2


class RAGConfig:
    """Configuration for RAG-based error fixing."""

    SEARCH_K = 10
    SCORE_THRESHOLD = 0.7
    AWS_REGION = "us-east-1"

    @staticmethod
    def get_rag_db_path():
        """Get the path to the FAISS index."""
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        return f"{current_file_dir}/../rag_db/faiss_index"


class FileConstants:
    """File-related constants."""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".pdf"}
    DATA_LAKE_SUBDIR = "data_lake"
    BIOMNI_DATA_SUBDIR = "biomni_data"


# ============================================================================
# UTILITY CLASSES
# ============================================================================


class PromptExtractor:
    """Utility class for extracting text from various prompt formats."""

    @staticmethod
    def extract_text(prompt) -> str:
        """
        Extract text content from various prompt formats.

        Args:
            prompt: Can be string, dict, list of messages, or message object

        Returns:
            Extracted text content
        """
        # Handle list of messages (e.g., [HumanMessage, AIMessage])
        if isinstance(prompt, list) and len(prompt) > 0:
            return PromptExtractor._extract_from_message_list(prompt)

        # Handle dict with 'content' key
        if isinstance(prompt, dict) and "content" in prompt:
            return PromptExtractor._extract_from_content(prompt["content"])

        # Handle message object with content attribute
        if hasattr(prompt, "content"):
            return PromptExtractor._extract_from_content(prompt.content)

        # Default: convert to string
        return str(prompt)

    @staticmethod
    def _extract_from_message_list(messages: list) -> str:
        """Extract text from a list of messages."""
        # Get the last message (user's message)
        last_message = messages[-1]

        if hasattr(last_message, "content"):
            content = last_message.content
        else:
            content = last_message

        return PromptExtractor._extract_from_content(content)

    @staticmethod
    def _extract_from_content(content) -> str:
        """Extract text from content that may contain images."""
        if isinstance(content, list):
            # Extract text parts only (filter out images)
            text_parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            return "\n".join(text_parts) if text_parts else ""

        if isinstance(content, str):
            return content

        return str(content)


class ResourceCollector:
    """Collects and organizes resources for agent execution."""

    def __init__(self, agent):
        """
        Initialize resource collector.

        Args:
            agent: The A1_HITS agent instance
        """
        self.agent = agent

    def gather_all_resources(self) -> Dict[str, List]:
        """
        Gather all available resources including tools, data lake, libraries, and know-how.

        Returns:
            Dictionary with 'tools', 'data_lake', 'libraries', and 'know_how' keys
        """
        return {
            "tools": self._get_tools(),
            "data_lake": self._get_data_lake_descriptions(),
            "libraries": self._get_library_descriptions(),
            "know_how": self._get_know_how_summaries(),
        }

    def _get_tools(self) -> List:
        """Get all available tools from the registry."""
        if hasattr(self.agent, "tool_registry"):
            return self.agent.tool_registry.tools
        return []

    def _get_data_lake_descriptions(self) -> List[Dict[str, str]]:
        """Get data lake items with their descriptions."""
        descriptions = []

        data_lake_path = self._get_data_lake_path()

        if os.path.exists(data_lake_path) and self.agent.data_lake_dict:
            data_lake_content = glob.glob(data_lake_path + "/*")

            # Filter to only include files in data_lake_dict (filtered by resource.yaml)
            data_lake_items = [
                os.path.basename(x)
                for x in data_lake_content
                if os.path.isfile(x)
                and os.path.basename(x) in self.agent.data_lake_dict
            ]

            # Create descriptions
            for item in data_lake_items:
                description = self.agent.data_lake_dict.get(
                    item, f"Data lake item: {item}"
                )
                descriptions.append({"name": item, "description": description})

        # Add custom data items
        if hasattr(self.agent, "_custom_data") and self.agent._custom_data:
            for name, info in self.agent._custom_data.items():
                descriptions.append({"name": name, "description": info["description"]})

        return descriptions

    def _get_data_lake_path(self) -> str:
        """Get the data lake directory path."""
        data_lake_path = os.path.join(self.agent.path, FileConstants.DATA_LAKE_SUBDIR)

        # Check if data_lake directory exists, if not try biomni_data/data_lake path
        if not os.path.exists(data_lake_path):
            data_lake_path = os.path.join(
                self.agent.path,
                FileConstants.BIOMNI_DATA_SUBDIR,
                FileConstants.DATA_LAKE_SUBDIR,
            )

        return data_lake_path

    def _get_library_descriptions(self) -> List[Dict[str, str]]:
        """Get library items with their descriptions."""
        descriptions = []

        # Add libraries from library_content_dict
        for lib_name, lib_desc in self.agent.library_content_dict.items():
            descriptions.append({"name": lib_name, "description": lib_desc})

        # Add custom software items
        if hasattr(self.agent, "_custom_software") and self.agent._custom_software:
            for name, info in self.agent._custom_software.items():
                # Avoid duplicates
                if not any(lib["name"] == name for lib in descriptions):
                    descriptions.append(
                        {"name": name, "description": info["description"]}
                    )

        return descriptions

    def _get_know_how_summaries(self) -> List[Dict[str, str]]:
        """Get know-how document summaries."""
        if hasattr(self.agent, "know_how_loader") and self.agent.know_how_loader:
            return self.agent.know_how_loader.get_document_summaries()
        return []

    @staticmethod
    def process_selected_resources(selected_resources: Dict) -> Dict[str, List]:
        """
        Process selected resources to extract just the names.
        Note: know-how documents are kept as full objects (not just names).

        Args:
            selected_resources: Dict with 'tools', 'data_lake', 'libraries', 'know_how'

        Returns:
            Dict with resource names (except know_how which keeps full objects)
        """
        result = {
            "tools": selected_resources.get("tools", []),
            "data_lake": [],
            "libraries": [],
            "know_how": [],  # Keep know-how documents as full objects
        }

        # Process libraries
        for lib in selected_resources.get("libraries", []):
            if isinstance(lib, dict):
                result["libraries"].append(lib["name"])
            else:
                result["libraries"].append(lib)

        # Process data lake items
        for item in selected_resources.get("data_lake", []):
            if isinstance(item, dict):
                result["data_lake"].append(item["name"])
            elif isinstance(item, str) and ": " in item:
                # Extract name from "name: description" format
                name = item.split(": ")[0]
                result["data_lake"].append(name)
            else:
                result["data_lake"].append(item)

        # Process know-how documents - keep Short description in the metadata section
        result["know_how"] = selected_resources.get("know_how", [])

        return result


class FileProcessor:
    """Handles file processing including new file detection and content extraction."""

    @staticmethod
    def process_new_files(
        files_before: Set[Path], files_after: Set[Path], observation: str
    ) -> List[Dict]:
        """
        Process newly created files and prepare message content.

        Args:
            files_before: Set of file paths before execution
            files_after: Set of file paths after execution
            observation: The execution observation text

        Returns:
            List of message content items (text + images)
        """
        new_files = files_after - files_before

        # Categorize new files
        new_images = []
        new_other_files = []

        for file_path in new_files:
            if file_path.is_file():
                if file_path.suffix.lower() in FileConstants.IMAGE_EXTENSIONS:
                    new_images.append(str(file_path))
                else:
                    new_other_files.append(str(file_path))

        # Prepare message content
        message_content = [{"type": "text", "text": observation}]

        # Add information about newly created files
        if new_images or new_other_files:
            files_info = "\n\n**Newly created files:**\n"

            # Process images
            if new_images:
                files_info += FileProcessor._process_images(new_images, message_content)

            # Process other files
            if new_other_files:
                files_info += FileProcessor._process_text_files(new_other_files)

            # Add files info to the first text content
            message_content[0]["text"] += files_info

        return message_content

    @staticmethod
    def _process_images(image_paths: List[str], message_content: List[Dict]) -> str:
        """Process image files and add them to message content."""
        files_info = "\n**Images:**\n"

        for img_path in image_paths:
            # Extract image resolution
            img_width, img_height = FileProcessor._get_image_dimensions(img_path)

            # Add image info
            if img_width and img_height:
                files_info += f"- {os.path.basename(img_path)}: {img_path} (resolution: {img_width}x{img_height} pixels)\n"
            else:
                files_info += f"- {os.path.basename(img_path)}: {img_path}\n"

            # Add image to message content
            FileProcessor._add_image_to_message(img_path, message_content)

        return files_info

    @staticmethod
    def _get_image_dimensions(img_path: str) -> tuple:
        """Get image dimensions (width, height)."""
        try:
            with Image.open(img_path) as img:
                return img.size
        except Exception:
            return None, None

    @staticmethod
    def _add_image_to_message(img_path: str, message_content: List[Dict]):
        """Encode image as base64 and add to message content."""
        try:
            with open(img_path, "rb") as img_file:
                img_data = img_file.read()
                base64_image = base64.b64encode(img_data).decode("utf-8")

                # Determine MIME type
                ext = Path(img_path).suffix.lower()
                mime_types = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".bmp": "image/bmp",
                    ".svg": "image/svg+xml",
                    ".pdf": "application/pdf",
                }
                mime_type = mime_types.get(ext, "image/png")

                # Add image to message content
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    }
                )
        except Exception as e:
            # Error is handled in the calling function
            pass

    @staticmethod
    def _process_text_files(file_paths: List[str]) -> str:
        """Process text files and read their contents."""
        files_info = "\n**Other files:**\n"

        for file_path in file_paths:
            files_info += f"- {file_path}\n"

            # Try to read file content
            try:
                content = FileProcessor._read_file_preview(file_path)
                if content:
                    files_info += f"```\n{content}```\n"
            except (UnicodeDecodeError, PermissionError):
                files_info += "  (Binary file or unable to read)\n"
            except Exception as e:
                files_info += f"  (Error reading file: {str(e)})\n"

        return files_info

    @staticmethod
    def _read_file_preview(file_path: str) -> str:
        """Read a preview of a text file (first few lines)."""
        with open(file_path, "r", encoding="utf-8") as f:
            lines = []
            line_count = 0

            for line in f:
                if line_count >= Config.MAX_FILE_PREVIEW_LINES:
                    lines.append("... (truncated: more lines)")
                    break

                # Truncate line if too long
                if len(line) > Config.MAX_CHARS_PER_LINE:
                    lines.append(line[: Config.MAX_CHARS_PER_LINE] + "...\n")
                else:
                    lines.append(line)

                line_count += 1

            return "".join(lines)


class MessageProcessor:
    """Handles message processing and transformations."""

    @staticmethod
    def remove_old_images_from_messages(messages: List) -> List:
        """
        Remove images from all HumanMessages except the most recent one.

        Args:
            messages: List of messages from the state

        Returns:
            List of processed messages with images removed from older HumanMessages
        """
        processed_messages = []

        # Find the index of the last HumanMessage
        last_human_msg_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_msg_idx = i
                break

        # Process each message
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage) and i != last_human_msg_idx:
                processed_messages.append(
                    MessageProcessor._remove_images_from_message(msg)
                )
            else:
                # For AIMessages and the most recent HumanMessage, keep as is
                processed_messages.append(msg)

        return processed_messages

    @staticmethod
    def _remove_images_from_message(msg: HumanMessage) -> HumanMessage:
        """Remove images from a single HumanMessage."""
        if isinstance(msg.content, list):
            # Filter out image_url content, keep only text
            text_only_content = [
                item for item in msg.content if item.get("type") != "image_url"
            ]

            # If only one text item remains, convert to simple string
            if (
                len(text_only_content) == 1
                and text_only_content[0].get("type") == "text"
            ):
                return HumanMessage(content=text_only_content[0]["text"])
            else:
                return HumanMessage(content=text_only_content)
        else:
            # Already simple text, keep as is
            return msg

    @staticmethod
    def create_observation_message(message_content: List[Dict]) -> HumanMessage:
        """
        Create an observation message from message content.

        Args:
            message_content: List of content items (text + images)

        Returns:
            HumanMessage with observation tags
        """
        if len(message_content) == 1:
            # Only text, use simple string content
            return HumanMessage(
                content=f"<observation>{message_content[0]['text']}</observation>"
            )
        else:
            # Text + images, use structured content
            message_content[0][
                "text"
            ] = f"<observation>{message_content[0]['text']}</observation>"
            return HumanMessage(content=message_content)


# ============================================================================
# WORKFLOW NODES
# ============================================================================


class WorkflowNodes:
    """Contains all workflow node functions for the agent."""

    def __init__(self, agent):
        """
        Initialize workflow nodes.

        Args:
            agent: The A1_HITS agent instance
        """
        self.agent = agent

    def generate(self, state: AgentState) -> AgentState:
        """
        Generate node: produces agent's reasoning and actions.

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        t1 = time.time()

        # Initialize chunk messages if not present
        if "chunk_messages" not in state:
            state["chunk_messages"] = []
        state["chunk_messages"].append("")  # dummy message

        # Process messages
        processed_messages = state["messages"]
        messages = [
            SystemMessage(content=self.agent.system_prompt)
        ] + processed_messages

        # Stream LLM response
        msg = ""
        for chunk in self.agent.llm.stream(messages):
            chunk_msg = chunk.content
            msg += chunk_msg
            state["chunk_messages"].append(chunk_msg)

        # Fix incomplete tags
        msg = self._fix_incomplete_tags(msg, state)

        # Parse response and determine next step
        think_match = re.search(r"<think>(.*?)</think>", msg, re.DOTALL)
        execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL)
        answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL)

        print(execute_match, answer_match, think_match)

        # Add message to state
        state["messages"].append(AIMessage(content=msg.strip()))

        # Determine next step
        if execute_match:
            state["next_step"] = "execute"
        elif answer_match:
            state["next_step"] = "end"
        elif think_match:
            state["next_step"] = "generate"
        else:
            state = self._handle_parsing_error(state)

        # Update timer
        t2 = time.time()
        self.agent.timer["generate"] += t2 - t1

        return state

    def _fix_incomplete_tags(self, msg: str, state: AgentState) -> str:
        """Fix incomplete XML tags in the message."""
        if "<execute>" in msg and "</execute>" not in msg:
            msg += "</execute>"
            state["messages"].append(AIMessageChunk(content="</execute>"))
        if "<solution>" in msg and "</solution>" not in msg:
            msg += "</solution>"
            state["messages"].append(AIMessageChunk(content="</solution>"))
        if "<think>" in msg and "</think>" not in msg:
            msg += "</think>"

        return msg

    def _handle_parsing_error(self, state: AgentState) -> AgentState:
        """Handle parsing errors when no valid tags are found."""
        print("parsing error...")

        # Check if we already added an error message to avoid infinite loops
        error_count = sum(
            1
            for m in state["messages"]
            if isinstance(m, AIMessage) and "there are no tags" in m.content.lower()
        )

        if error_count >= Config.MAX_PARSING_ERROR_COUNT:
            # End conversation after too many errors
            print("Detected repeated parsing errors, ending conversation")
            state["next_step"] = "end"
            state["messages"].append(
                AIMessage(
                    content="Execution terminated due to repeated parsing errors. "
                    "Please check your input and try again."
                )
            )
        else:
            # Try to correct it
            state["messages"].append(
                AIMessage(
                    content="Each response must include thinking process followed by "
                    "either <execute> or <solution> tag. But there are no tags "
                    "in the current response. Please follow the instruction, "
                    "fix and regenerate the response again."
                )
            )
            state["next_step"] = "generate"

        return state

    def execute(self, state: AgentState) -> AgentState:
        """
        Execute node: runs code and returns observations.

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        t1 = time.time()

        last_message = state["messages"][-1].content

        # Add closing tag if missing
        if "<execute>" in last_message and "</execute>" not in last_message:
            last_message += "</execute>"

        execute_match = re.search(r"<execute>(.*?)</execute>", last_message, re.DOTALL)

        if execute_match:
            print("START EXECUTING CODE!!!!!")
            code = execute_match.group(1)

            # Get list of files before execution
            current_dir = Path.cwd()
            files_before = set(current_dir.glob("*"))

            # Execute code
            result = self._execute_code(code)

            # Get list of files after execution
            files_after = set(current_dir.glob("*"))

            # Prepare observation
            observation = self._prepare_observation(result)

            # Process newly created files
            message_content = FileProcessor.process_new_files(
                files_before, files_after, observation
            )

            # Add error fixing guide if needed
            if self._should_add_error_fixing(observation, code):
                error_fixing_guide = self.error_fixing(code, result, state)
                if len(error_fixing_guide) > 0:
                    message_content[0]["text"] += (
                        f"\n\nPlease refer the following for fixing the error above:\n\n "
                        f"{error_fixing_guide}"
                    )

            # Add observation message
            state["messages"].append(
                MessageProcessor.create_observation_message(message_content)
            )

        # Update timer
        t2 = time.time()
        self.agent.timer["execute"] += t2 - t1

        return state

    def _execute_code(self, code: str) -> str:
        """Execute code based on its type (Python, R, or Bash)."""
        timeout = self.agent.timeout_seconds

        # Check if R code
        if self._is_r_code(code):
            r_code = re.sub(r"^#!R|^# R code|^# R script", "", code, 1).strip()
            return run_with_timeout(run_r_code, [r_code], timeout=timeout)

        # Check if Bash script or CLI command
        if self._is_bash_code(code):
            return self._execute_bash_code(code, timeout)

        # Default: Python code
        self.agent._inject_custom_functions_to_repl()
        return run_with_timeout(run_python_repl, [code], timeout=timeout)

    @staticmethod
    def _is_r_code(code: str) -> bool:
        """Check if code is R code."""
        return (
            code.strip().startswith("#!R")
            or code.strip().startswith("# R code")
            or code.strip().startswith("# R script")
        )

    @staticmethod
    def _is_bash_code(code: str) -> bool:
        """Check if code is Bash code."""
        return (
            code.strip().startswith("#!BASH")
            or code.strip().startswith("# Bash script")
            or code.strip().startswith("#!CLI")
        )

    @staticmethod
    def _execute_bash_code(code: str, timeout: int) -> str:
        """Execute Bash or CLI code."""
        if code.strip().startswith("#!CLI"):
            # For CLI commands, extract and run as single command
            cli_command = re.sub(r"^#!CLI", "", code, 1).strip()
            cli_command = cli_command.replace("\n", " ")
            return run_with_timeout(run_bash_script, [cli_command], timeout=timeout)
        else:
            # For Bash scripts
            bash_script = re.sub(r"^#!BASH|^# Bash script", "", code, 1).strip()
            return run_with_timeout(run_bash_script, [bash_script], timeout=timeout)

    def _prepare_observation(self, result: str) -> str:
        """Prepare observation from execution result."""
        if len(result) > Config.MAX_OUTPUT_LENGTH:
            result = (
                f"The output is too long to be added to context. "
                f"Here are the first {Config.MAX_OUTPUT_LENGTH} characters...\n"
                + result[: Config.MAX_OUTPUT_LENGTH]
            )
        return result

    @staticmethod
    def _should_add_error_fixing(observation: str, code: str) -> bool:
        """Check if error fixing guide should be added."""
        has_error_markers = (
            "Error Type" in observation and "Error Message" in observation
        )
        has_error_in_output = (
            "error" in observation.lower()
            and "try:" in code.lower()
            and "except" in code.lower()
        )
        return has_error_markers or has_error_in_output

    def error_fixing(self, code: str, output: str, state: AgentState) -> str:
        """
        Provides error fixing suggestions using RAG-based retrieval.

        Args:
            code: The code that caused the error
            output: The error output/message
            state: The current agent state

        Returns:
            Error fixing suggestion from the retrieval system
        """
        start_time = time.time()

        # Initialize error fixing history
        if "error_fixing_history" not in state:
            state["error_fixing_history"] = []

        # Set up retrieval components
        retriever = self._setup_error_fixing_retriever()
        llm = get_llm(model=Config.ERROR_FIXING_LLM_MODEL_ID)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever, response_if_no_docs_found=""
        )

        # Generate and execute query
        question = self._create_error_fixing_prompt(code, output)

        try:
            result = qa_chain.invoke(
                {
                    "question": question,
                    "chat_history": state["error_fixing_history"],
                }
            )
            answer = result["answer"]
        except Exception as e:
            answer = f"Error fixing retrieval failed: {str(e)}"

        # Update timing and history
        end_time = time.time()
        self.agent.timer["error_fixing"] += end_time - start_time
        state["error_fixing_history"].extend([question, answer])

        return answer

    @staticmethod
    def _setup_error_fixing_retriever():
        """Set up and return the FAISS retriever for error fixing."""
        embeddings = BedrockEmbeddings(
            normalize=True, region_name=os.getenv("AWS_REGION", RAGConfig.AWS_REGION)
        )

        db = FAISS.load_local(
            RAGConfig.get_rag_db_path(),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        return db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": RAGConfig.SEARCH_K,
                "score_threshold": RAGConfig.SCORE_THRESHOLD,
            },
        )

    @staticmethod
    def _create_error_fixing_prompt(code: str, output: str) -> str:
        """Create the error fixing prompt."""
        return f"""I encountered an error while executing the code.

1. Your purpose is to fix errors based on database queries only.
If you cannot obtain useful and relevant information from the database query, DO NOT provide any response.
Only answer using information retrieved from the database - DO NOT use your general knowledge.

2. In addition to the error line, check subsequent lines of code that may also contain potential errors.
Provide fix suggestions for these lines as well, but only based on database query results.

Provide a concise solution. Keep your answer short and focused.
If possible, don't modify the entire code, just provide the parts that need to be fixed and corresponding solution.

Code:
{code}
======================
Output:
{output}
"""

    def self_critic(self, state: AgentState, test_time_scale_round: int) -> AgentState:
        """
        Self-critic node: provides feedback on the solution.

        Args:
            state: Current agent state
            test_time_scale_round: Maximum number of critic rounds

        Returns:
            Updated agent state
        """
        if self.agent.critic_count < test_time_scale_round:
            # Generate feedback
            messages = state["messages"]
            feedback_prompt = f"""
            Here is a reminder of what is the user requested: {self.agent.user_task}
            Examine the previous executions, reaosning, and solutions.
            Critic harshly on what could be improved?
            Be specific and constructive.
            Think hard what are missing to solve the task.
            No question asked, just feedbacks.
            """
            feedback = self.agent.llm.invoke(
                messages + [HumanMessage(content=feedback_prompt)]
            )

            # Add feedback message
            state["messages"].append(
                AIMessage(
                    content=f"Wait... this is not enough to solve the task. "
                    f"Here are some feedbacks for improvement:\n{feedback.content}"
                )
            )
            self.agent.critic_count += 1
            state["next_step"] = "generate"
        else:
            state["next_step"] = "end"

        return state

    @staticmethod
    def routing_function(state: AgentState) -> Literal["execute", "generate", "end"]:
        """Route to the next node based on state."""
        next_step = state.get("next_step")

        if next_step in ["execute", "generate", "end"]:
            return next_step

        raise ValueError(f"Unexpected next_step: {next_step}")

    @staticmethod
    def routing_function_self_critic(state: AgentState) -> Literal["generate", "end"]:
        """Route to the next node for self-critic workflow."""
        next_step = state.get("next_step")

        if next_step in ["generate", "end"]:
            return next_step

        raise ValueError(f"Unexpected next_step: {next_step}")


# ============================================================================
# PROMPT GENERATION HELPER
# ============================================================================


class PromptGenerator:
    """Handles system prompt generation."""

    def __init__(self, agent):
        """
        Initialize prompt generator.

        Args:
            agent: The A1_HITS agent instance
        """
        self.agent = agent

    def format_item_with_description(self, name: str, description: str) -> str:
        """
        Format an item with its description in a readable way.

        Args:
            name: Item name
            description: Item description

        Returns:
            Formatted string
        """
        # Handle None or empty descriptions
        if not description:
            description = f"Data lake item: {name}"

        # Check if already formatted
        if isinstance(name, str) and ": " in name:
            return name

        # Wrap long descriptions
        max_line_length = 80
        if len(description) > max_line_length:
            wrapped_desc = self._wrap_description(description, max_line_length)
            formatted_desc = f"{name}:\n  " + "\n  ".join(wrapped_desc)
            return formatted_desc
        else:
            return f"{name}: {description}"

    @staticmethod
    def _wrap_description(description: str, max_line_length: int) -> List[str]:
        """Wrap long description into multiple lines."""
        wrapped_desc = []
        words = description.split()
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_line_length:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                wrapped_desc.append(current_line)
                current_line = word

        if current_line:
            wrapped_desc.append(current_line)

        return wrapped_desc

    def separate_custom_and_default_resources(
        self,
        data_lake_content: List,
        library_content_list: List,
        custom_data: List = None,
        custom_software: List = None,
    ) -> tuple:
        """
        Separate custom and default resources.

        Returns:
            Tuple of (default_data_lake, default_libraries, custom_data_names, custom_software_names)
        """
        custom_data_names = set()
        custom_software_names = set()

        if custom_data:
            custom_data_names = {
                item.get("name") if isinstance(item, dict) else item
                for item in custom_data
            }
        if custom_software:
            custom_software_names = {
                item.get("name") if isinstance(item, dict) else item
                for item in custom_software
            }

        # Separate default data lake items
        default_data_lake_content = [
            item
            for item in data_lake_content
            if self._get_item_name(item) not in custom_data_names
        ]

        # Separate default library items
        default_library_content_list = [
            lib
            for lib in library_content_list
            if self._get_item_name(lib) not in custom_software_names
        ]

        return (
            default_data_lake_content,
            default_library_content_list,
            custom_data_names,
            custom_software_names,
        )

    @staticmethod
    def _get_item_name(item) -> str:
        """Get item name from dict or string."""
        if isinstance(item, dict):
            return item.get("name", "")
        return item

    def format_data_lake_content(self, data_lake_content: List) -> List[str]:
        """Format data lake content with descriptions."""
        if isinstance(data_lake_content, list) and all(
            isinstance(item, str) for item in data_lake_content
        ):
            # Simple list of strings
            return self._format_string_list(
                data_lake_content, self.agent.data_lake_dict, "Data lake item"
            )
        else:
            # List with descriptions
            return self._format_dict_list(
                data_lake_content, self.agent.data_lake_dict, "Data lake item"
            )

    def format_library_content(self, library_content_list: List) -> List[str]:
        """Format library content with descriptions."""
        if isinstance(library_content_list, list) and all(
            isinstance(item, str) for item in library_content_list
        ):
            if (
                len(library_content_list) > 0
                and isinstance(library_content_list[0], str)
                and "," not in library_content_list[0]
            ):
                # Simple list of strings
                return self._format_string_list(
                    library_content_list,
                    self.agent.library_content_dict,
                    "Software library",
                )
            else:
                # Already formatted
                return library_content_list
        else:
            # List with descriptions
            return self._format_dict_list(
                library_content_list,
                self.agent.library_content_dict,
                "Software library",
            )

    def _format_string_list(
        self, items: List[str], description_dict: Dict, default_prefix: str
    ) -> List[str]:
        """Format a list of string items with descriptions."""
        formatted = []
        for item in items:
            if ": " in item:
                formatted.append(item)
            else:
                description = description_dict.get(item, f"{default_prefix}: {item}")
                formatted.append(self.format_item_with_description(item, description))
        return formatted

    def _format_dict_list(
        self, items: List, description_dict: Dict, default_prefix: str
    ) -> List[str]:
        """Format a list of dict items with descriptions."""
        formatted = []
        for item in items:
            if isinstance(item, dict):
                name = item.get("name", "")
                description = description_dict.get(name, f"{default_prefix}: {name}")
                formatted.append(self.format_item_with_description(name, description))
            elif isinstance(item, str) and ": " in item:
                formatted.append(item)
            else:
                description = description_dict.get(item, f"{default_prefix}: {item}")
                formatted.append(self.format_item_with_description(item, description))
        return formatted

    def format_custom_resources(
        self,
        custom_tools: List = None,
        custom_data: List = None,
        custom_software: List = None,
    ) -> Dict[str, List[str]]:
        """Format custom resources with highlighting."""
        result = {"tools": [], "data": [], "software": []}

        if custom_tools:
            for tool in custom_tools:
                if isinstance(tool, dict):
                    name = tool.get("name", "Unknown")
                    desc = tool.get("description", "")
                    module = tool.get("module", "custom_tools")
                    result["tools"].append(f"🔧 {name} (from {module}): {desc}")
                else:
                    result["tools"].append(f"🔧 {str(tool)}")

        if custom_data:
            for item in custom_data:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    result["data"].append(
                        f"📊 {self.format_item_with_description(name, desc)}"
                    )
                else:
                    desc = self.agent.data_lake_dict.get(item, f"Custom data: {item}")
                    result["data"].append(
                        f"📊 {self.format_item_with_description(item, desc)}"
                    )

        if custom_software:
            for item in custom_software:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    result["software"].append(
                        f"⚙️ {self.format_item_with_description(name, desc)}"
                    )
                else:
                    desc = self.agent.library_content_dict.get(
                        item, f"Custom software: {item}"
                    )
                    result["software"].append(
                        f"⚙️ {self.format_item_with_description(item, desc)}"
                    )

        return result


# ============================================================================
# MAIN A1_HITS CLASS
# ============================================================================


class A1_HITS(A1):
    """A1 HITS agent with improved code organization."""

    def __init__(self, *args, resource_filter_config_path=None, **kwargs):
        """
        Initialize A1_HITS agent with optional resource filtering.

        Args:
            *args: Arguments passed to parent A1 class
            resource_filter_config_path: Path to YAML file with resource filter configuration
            **kwargs: Keyword arguments passed to parent A1 class
        """
        # Load and apply resource filter config before calling super()
        self._apply_resource_filters_before_init(resource_filter_config_path, kwargs)

        # Initialize parent class
        super().__init__(*args, **kwargs)

        # Initialize LLM
        self.llm = get_llm(
            kwargs.get("llm", default_config.llm),
            source=kwargs.get("source", default_config.source),
            base_url=kwargs.get("base_url", default_config.base_url),
            api_key=kwargs.get(
                "api_key", default_config.api_key if default_config.api_key else "EMPTY"
            ),
            config=default_config,
        )

        # Apply resource filters after initialization
        self._apply_resource_filters_after_init(resource_filter_config_path)

        # Initialize timer
        self.timer = {"generate": 0.0, "execute": 0.0, "error_fixing": 0.0}

    def _apply_resource_filters_before_init(self, resource_filter_config_path, kwargs):
        """Apply resource filters before parent initialization."""
        resource_config = load_resource_filter_config(resource_filter_config_path)
        allowed_data_lake_items = resource_config.get("data_lake", None)

        # Only apply filtering if resource.yaml has data_lake items defined
        if (
            allowed_data_lake_items
            and len(allowed_data_lake_items) > 0
            and "expected_data_lake_files" not in kwargs
        ):
            # Determine commercial_mode
            commercial_mode = kwargs.get("commercial_mode", None)
            if commercial_mode is None:
                commercial_mode = default_config.commercial_mode

            # Load appropriate data_lake_dict
            if commercial_mode:
                from biomni.env_desc_cm import data_lake_dict as full_data_lake_dict
            else:
                from biomni.env_desc import data_lake_dict as full_data_lake_dict

            # Filter data_lake_dict based on resource.yaml
            filtered_data_lake_dict = filter_data_lake_dict(
                full_data_lake_dict, allowed_data_lake_items
            )

            # Pass filtered file list to parent
            filtered_files = list(filtered_data_lake_dict.keys())
            if filtered_files:
                kwargs["expected_data_lake_files"] = filtered_files
                print(
                    f"📥 Filtering data lake downloads: {len(filtered_files)} items "
                    f"allowed (from {len(full_data_lake_dict)} total)"
                )
            elif len(allowed_data_lake_items) > 0:
                kwargs["expected_data_lake_files"] = []
                print(
                    f"⚠️  Warning: Resource filter specified {len(allowed_data_lake_items)} "
                    f"data_lake items, but none matched available items. "
                    f"No data lake files will be downloaded."
                )

    def _apply_resource_filters_after_init(self, resource_filter_config_path):
        """Apply resource filters after parent initialization."""
        if (
            hasattr(self, "module2api")
            and hasattr(self, "data_lake_dict")
            and hasattr(self, "library_content_dict")
        ):
            (
                filtered_module2api,
                filtered_data_lake_dict,
                filtered_library_content_dict,
            ) = apply_resource_filters(
                self.module2api,
                self.data_lake_dict,
                self.library_content_dict,
                config_path=resource_filter_config_path,
            )

            # Update filtered resources
            self.module2api = filtered_module2api
            self.data_lake_dict = filtered_data_lake_dict
            self.library_content_dict = filtered_library_content_dict

            # Recreate tool registry if needed
            if hasattr(self, "tool_registry") and self.use_tool_retriever:
                from biomni.tool.tool_registry import ToolRegistry

                self.tool_registry = ToolRegistry(self.module2api)

    def go(self, prompt, additional_system_prompt=None):
        """
        Execute the agent with the given prompt (synchronous).

        Args:
            prompt: The user's query
            additional_system_prompt: Optional additional system prompt

        Yields:
            Messages from the agent execution
        """
        self.critic_count = 0
        self.user_task = prompt

        # Perform tool retrieval if enabled
        if self.use_tool_retriever:
            self._perform_tool_retrieval(prompt)

        # Add additional system prompt if provided
        if additional_system_prompt:
            self.system_prompt += "\n----\n" + additional_system_prompt

        # Prepare inputs
        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {
            "recursion_limit": Config.DEFAULT_RECURSION_LIMIT,
            "configurable": {"thread_id": Config.DEFAULT_THREAD_ID},
        }

        # Initialize log
        self.log = [self.system_prompt]
        yield self.system_prompt

        # Stream execution
        for s in self.app.stream(
            inputs, stream_mode="messages", config=config, subgraphs=True
        ):
            # Handle message chunks and complete messages
            if type(s[1][0]) == AIMessageChunk:
                print(s[1][0].content, end="")
            elif type(s[1][0]) in [AIMessage, HumanMessage]:
                message = s[1][0].content

                # Extract text from structured content
                if isinstance(message, list):
                    text_parts = [
                        item["text"] for item in message if item.get("type") == "text"
                    ]
                    text_message = "\n".join(text_parts) if text_parts else ""

                    if type(s[1][0]) == HumanMessage:
                        print(text_message)
                    self.log.append(message)
                    yield text_message
                else:
                    if type(s[1][0]) == HumanMessage:
                        print(message)
                    self.log.append(message)
                    yield message

        return self.log, message

    def go_stream(
        self, prompt, additional_system_prompt=None, skip_generate_system_prompt=False
    ):
        """
        Execute the agent with the given prompt (streaming).

        Args:
            prompt: The user's query (can be a list of messages)
            additional_system_prompt: Optional additional system prompt
            skip_generate_system_prompt: Skip system prompt generation

        Yields:
            Message chunks from the agent execution
        """
        self.critic_count = 0
        self.user_task = prompt

        # Perform tool retrieval if enabled
        if self.use_tool_retriever and not skip_generate_system_prompt:
            self._perform_tool_retrieval(prompt)

        # Add additional system prompt if provided
        if additional_system_prompt:
            self.system_prompt += "\n----\n" + additional_system_prompt

        # Prepare inputs
        inputs = {"messages": prompt, "next_step": None}
        config = {
            "recursion_limit": Config.DEFAULT_RECURSION_LIMIT,
            "configurable": {"thread_id": Config.DEFAULT_THREAD_ID},
        }

        # Initialize log
        self.log = [self.system_prompt]

        # Stream execution
        for s in self.app.stream(
            inputs, stream_mode="messages", config=config, subgraphs=True
        ):
            yield s

    def _perform_tool_retrieval(self, prompt):
        """Perform tool retrieval and update system prompt."""
        print("start tool retrieval")

        # Gather resources
        collector = ResourceCollector(self)
        resources = collector.gather_all_resources()

        # Extract text from prompt
        text_prompt = PromptExtractor.extract_text(prompt)

        # Perform retrieval
        tool_llm = get_llm(model=Config.TOOL_LLM_MODEL_ID)
        selected_resources = self.retriever.prompt_based_retrieval(
            text_prompt, resources, llm=tool_llm
        )

        print("end tool retrieval")
        print("Using prompt-based RAG retrieval with the agent's LLM")

        # Process selected resources
        selected_resources_names = ResourceCollector.process_selected_resources(
            selected_resources
        )

        # Update system prompt
        self.update_system_prompt_with_selected_resources(selected_resources_names)

    def configure(self, self_critic=False, test_time_scale_round=0):
        """
        Configure the agent workflow.

        Args:
            self_critic: Enable self-critic mode
            test_time_scale_round: Number of self-critic rounds
        """
        super().configure(
            self_critic=self_critic, test_time_scale_round=test_time_scale_round
        )

        # Create workflow nodes
        nodes = WorkflowNodes(self)

        # Create the workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate", nodes.generate)
        workflow.add_node("execute", nodes.execute)

        if self_critic:
            # Add self-critic node
            def execute_self_critic(state: AgentState) -> AgentState:
                return nodes.self_critic(state, test_time_scale_round)

            workflow.add_node("self_critic", execute_self_critic)

            # Add conditional edges for self-critic mode
            workflow.add_conditional_edges(
                "generate",
                nodes.routing_function,
                path_map={
                    "execute": "execute",
                    "generate": "generate",
                    "end": "self_critic",
                },
            )
            workflow.add_conditional_edges(
                "self_critic",
                nodes.routing_function_self_critic,
                path_map={"generate": "generate", "end": END},
            )
        else:
            # Add conditional edges for normal mode
            workflow.add_conditional_edges(
                "generate",
                nodes.routing_function,
                path_map={"execute": "execute", "generate": "generate", "end": END},
            )

        workflow.add_edge("execute", "generate")
        workflow.add_edge(START, "generate")

        # Compile the workflow
        self.app = workflow.compile()
        
        # Set up persistent memory if enabled (controlled by default_config)
        if default_config.use_persistent_memory:
            self.checkpointer = MemorySaver()
            self.app.checkpointer = self.checkpointer
        else:
            self.checkpointer = None

    def _load_prompt_template(self, template_name: str) -> str:
        """
        Load a prompt template file from the prompts directory.

        Args:
            template_name: Name of the template file

        Returns:
            Content of the template file
        """
        template_path = os.path.join(
            os.path.dirname(__file__), "prompts", template_name
        )

        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Prompt template '{template_name}' not found at {template_path}"
            )

    def _generate_system_prompt(
        self,
        tool_desc,
        data_lake_content,
        library_content_list,
        self_critic=False,
        is_retrieval=False,
        custom_tools=None,
        custom_data=None,
        custom_software=None,
        know_how_docs=None,
    ):
        """
        Generate the system prompt based on the provided resources.

        Args:
            tool_desc: Dictionary of tool descriptions
            data_lake_content: List of data lake items
            library_content_list: List of libraries
            self_critic: Not used (kept for backward compatibility)
            is_retrieval: Whether this is for retrieval (True) or initial configuration (False)
            custom_tools: List of custom tools to highlight
            custom_data: List of custom data items to highlight
            custom_software: List of custom software items to highlight
            know_how_docs: List of know-how documents with best practices and protocols

        Returns:
            The generated system prompt
        """
        # Create prompt generator
        generator = PromptGenerator(self)

        # Separate custom and default resources
        (
            default_data_lake_content,
            default_library_content_list,
            custom_data_names,
            custom_software_names,
        ) = generator.separate_custom_and_default_resources(
            data_lake_content, library_content_list, custom_data, custom_software
        )

        # Format default resources
        data_lake_formatted = generator.format_data_lake_content(
            default_data_lake_content
        )
        libraries_formatted = generator.format_library_content(
            default_library_content_list
        )

        # Format custom resources
        custom_resources = generator.format_custom_resources(
            custom_tools, custom_data, custom_software
        )

        # Format know-how documents - include FULL content (metadata already stripped)
        know_how_formatted = []
        if know_how_docs:
            for doc in know_how_docs:
                if isinstance(doc, dict):
                    name = doc.get("name", "Unknown")
                    content = doc.get("content", "")
                    # Include full content in system prompt (metadata already removed)
                    know_how_formatted.append(f"📚 {name}:\n{content}")

        # Load base prompt template
        prompt_modifier = self._load_prompt_template("base_system_prompt.md")

        # Build custom resources section
        has_custom_resources = any(custom_resources.values()) or know_how_formatted
        if has_custom_resources:
            custom_sections = []

            # Add know-how section first (highest priority)
            if know_how_formatted:
                custom_sections.append(
                    "📚 KNOW-HOW DOCUMENTS (BEST PRACTICES & PROTOCOLS - ALREADY LOADED):\n"
                    "{know_how_docs}\n\n"
                    "IMPORTANT: These documents are ALREADY AVAILABLE in your context. You do NOT need to\n"
                    "retrieve them or \"review\" them as a separate step. You can DIRECTLY reference and use\n"
                    "the information from these documents to answer questions, provide protocols, suggest\n"
                    "parameters, and offer troubleshooting guidance.\n\n"
                    "These documents contain expert knowledge, protocols, and troubleshooting guidance.\n"
                    "Reference them directly for experimental design, methodology, and problem-solving.\n"
                )

            if custom_resources["tools"]:
                custom_sections.append(
                    "🔧 CUSTOM TOOLS (USE THESE FIRST):\n{custom_tools}\n"
                )

            if custom_resources["data"]:
                custom_sections.append(
                    "📊 CUSTOM DATA (PRIORITIZE THESE DATASETS):\n{custom_data}\n"
                )

            if custom_resources["software"]:
                custom_sections.append(
                    "⚙️ CUSTOM SOFTWARE (USE THESE LIBRARIES):\n{custom_software}\n"
                )

            # Load custom resources template
            custom_resources_template = self._load_prompt_template(
                "custom_resources_section.md"
            )
            prompt_modifier += custom_resources_template.format(
                custom_sections="\n".join(custom_sections)
            )

        # Add environment resources section
        prompt_modifier += self._load_prompt_template(
            "environment_resources_section.md"
        )

        # Set appropriate text based on context
        if is_retrieval:
            function_intro = "Based on your query, I've identified the following most relevant functions that you can use in your code:"
            data_lake_intro = "Based on your query, I've identified the following most relevant datasets:"
            library_intro = "Based on your query, I've identified the following most relevant libraries that you can use:"
            import_instruction = "IMPORTANT: When using any function, you MUST first import it from its module. For example:\nfrom [module_name] import [function_name]"
        else:
            function_intro = "In your code, you will need to import the function location using the following dictionary of functions:"
            data_lake_intro = "You can write code to understand the data, process and utilize it for the task. Here is the list of datasets. I recommend you to find the schema of the dataset first before using it:"
            library_intro = "The environment supports a list of libraries that can be directly used. Do not forget the import statement:"
            import_instruction = ""

        # Format content
        library_content_formatted = (
            "\n".join(libraries_formatted)
            if libraries_formatted
            else "No specific libraries have been pre-identified for this task.\nUse standard Python libraries (pandas, numpy, scipy, matplotlib, etc.) as needed."
        )

        data_lake_content_formatted = (
            "\n".join(data_lake_formatted)
            if data_lake_formatted
            else f"No specific datasets have been pre-identified for this task.\nExplore the data lake directory if you need biological data: {self.path}/data_lake"
        )

        # Handle tool_desc
        if isinstance(tool_desc, dict):
            tool_desc_formatted = (
                textify_api_dict(tool_desc)
                if tool_desc
                else "No specific functions have been pre-identified for this task.\nUse standard Python functions and methods as needed."
            )
        else:
            tool_desc_formatted = (
                tool_desc
                if tool_desc
                else "No specific functions have been pre-identified for this task.\nUse standard Python functions and methods as needed."
            )

        # Build format dictionary
        format_dict = {
            "function_intro": function_intro,
            "tool_desc": tool_desc_formatted,
            "import_instruction": import_instruction,
            "data_lake_path": self.path + "/data_lake",
            "data_lake_intro": data_lake_intro,
            "data_lake_content": data_lake_content_formatted,
            "library_intro": library_intro,
            "library_content_formatted": library_content_formatted,
        }

        # Add custom resources to format dict
        if know_how_formatted:
            format_dict["know_how_docs"] = "\n\n".join(know_how_formatted)
        if custom_resources["tools"]:
            format_dict["custom_tools"] = "\n".join(custom_resources["tools"])
        if custom_resources["data"]:
            format_dict["custom_data"] = "\n".join(custom_resources["data"])
        if custom_resources["software"]:
            format_dict["custom_software"] = "\n".join(custom_resources["software"])

        # Format and return prompt
        formatted_prompt = prompt_modifier.format(**format_dict)
        return formatted_prompt