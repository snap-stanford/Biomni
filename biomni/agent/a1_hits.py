import glob
import re
from typing import Literal, TypedDict, List
from pydantic import BaseModel, Field
import time

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
from langchain.chains import ConversationalRetrievalChain
from biomni.model.retriever import ToolRetrieverByRAG

# tool_llm_model_id = "gemini-2.5-pro"
# tool_llm_model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
tool_llm_model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
# tool_llm_model_id = "gemini-2.5-flash"  # Use Gemini instead of Bedrock for tool retrieval


class A1_HITS(A1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = {"generate": 0.0, "execute": 0.0, "error_fixing": 0.0}

    def _initialize_error_fixing_history(self, state):
        """Initialize error fixing history in state if not present."""
        if "error_fixing_history" not in state:
            state["error_fixing_history"] = []

    def _setup_error_fixing_retriever(self):
        """Set up and return the FAISS retriever for error fixing."""
        # Constants for configuration
        # Get current file location
        import os

        CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
        RAG_DB_PATH = f"{CURRENT_FILE_DIR}/../rag_db/faiss_index"
        SEARCH_K = 10
        SCORE_THRESHOLD = 0.7

        embeddings = BedrockEmbeddings(normalize=True)

        db = FAISS.load_local(
            RAG_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        return db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": SEARCH_K, "score_threshold": SCORE_THRESHOLD},
        )

    def _get_error_fixing_llm(self):
        """Get the LLM model for error fixing."""
        # Using mistral-small-2506 as the primary model
        # Alternative models are commented out for future reference
        # return get_llm(model="mistral-small-2506")
        return get_llm(model="us.anthropic.claude-haiku-4-5-20251001-v1:0")
        # return get_llm(model="us.anthropic.claude-3-5-haiku-20241022-v1:0")
        # Alternative options:
        # return get_llm(model="us.anthropic.claude-3-5-sonnet-20240620-v1:0")
        # return ChatBedrock(model="us.anthropic.claude-3-5-sonnet-20240620-v1:0")

    def _create_qa_chain(self, llm, retriever):
        """Create and return the conversational retrieval chain."""
        return ConversationalRetrievalChain.from_llm(
            llm,
            retriever,
            response_if_no_docs_found="",
        )

    def _create_error_fixing_prompt(self, code, output):
        """Create the error fixing prompt with code and output."""
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

    def _get_error_fixing_answer(self, qa_chain, question, state):
        """Get the error fixing answer from the QA chain."""
        try:
            result = qa_chain.invoke(
                {
                    "question": question,
                    "chat_history": [],
                    # "chat_history": state["error_fixing_history"][
                    #     -1:
                    # ],  # Use only the last message for error fixing
                }
            )
            return result["answer"]
        except Exception as e:
            # Fallback in case of retrieval failure
            return f"Error fixing retrieval failed: {str(e)}"

    def _update_error_fixing_metrics(self, start_time, question, answer, state):
        """Update timing metrics and error fixing history."""
        end_time = time.time()
        self.timer["error_fixing"] += end_time - start_time

        # Update conversation history
        state["error_fixing_history"].extend([question, answer])

    def go(self, prompt, additional_system_prompt=None):
        """Execute the agent with the given prompt.

        Args:
            prompt: The user's query

        """
        self.critic_count = 0
        self.user_task = prompt
        if self.use_tool_retriever:
            # Gather all available resources
            # 1. Tools from the registry
            all_tools = (
                self.tool_registry.tools if hasattr(self, "tool_registry") else []
            )

            # 2. Data lake items with descriptions
            data_lake_path = self.path + "/data_lake"
            data_lake_content = glob.glob(data_lake_path + "/*")
            data_lake_items = [x.split("/")[-1] for x in data_lake_content]

            # Create data lake descriptions for retrieval
            data_lake_descriptions = []
            for item in data_lake_items:
                description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                data_lake_descriptions.append(
                    {"name": item, "description": description}
                )

            # Add custom data items to retrieval if they exist
            if hasattr(self, "_custom_data") and self._custom_data:
                for name, info in self._custom_data.items():
                    data_lake_descriptions.append(
                        {"name": name, "description": info["description"]}
                    )
            # 3. Libraries with descriptions - use library_content_dict directly
            library_descriptions = []
            for lib_name, lib_desc in self.library_content_dict.items():
                library_descriptions.append({"name": lib_name, "description": lib_desc})

            # Add custom software items to retrieval if they exist
            if hasattr(self, "_custom_software") and self._custom_software:
                for name, info in self._custom_software.items():
                    # Check if it's not already in the library descriptions to avoid duplicates
                    if not any(lib["name"] == name for lib in library_descriptions):
                        library_descriptions.append(
                            {"name": name, "description": info["description"]}
                        )

            # Use retrieval to get relevant resources
            resources = {
                "tools": all_tools,
                "data_lake": data_lake_descriptions,
                "libraries": library_descriptions,
            }

            # Use prompt-based retrieval with the agent's LLM
            tool_llm = get_llm(model=tool_llm_model_id)
            selected_resources = self.retriever.prompt_based_retrieval(
                prompt, resources, llm=tool_llm
            )
            # self.retriever = ToolRetrieverByRAG()
            # selected_resources = self.retriever.prompt_based_retrieval(prompt)
            print("Using prompt-based retrieval with the agent's LLM")
            # Extract the names from the selected resources for the system prompt
            selected_resources_names = {
                "tools": selected_resources["tools"],
                "data_lake": [],
                "libraries": [
                    lib["name"] if isinstance(lib, dict) else lib
                    for lib in selected_resources["libraries"]
                ],
            }

            # Process data lake items to extract just the names
            for item in selected_resources["data_lake"]:
                if isinstance(item, dict):
                    selected_resources_names["data_lake"].append(item["name"])
                elif isinstance(item, str) and ": " in item:
                    # If the item already has a description, extract just the name
                    name = item.split(": ")[0]
                    selected_resources_names["data_lake"].append(name)
                else:
                    selected_resources_names["data_lake"].append(item)

            # Update the system prompt with the selected resources
            self.update_system_prompt_with_selected_resources(selected_resources_names)

        if additional_system_prompt:
            self.system_prompt += "\n----\n" + additional_system_prompt
        print(self.system_prompt)
        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = [self.system_prompt]
        yield self.system_prompt
        for s in self.app.stream(
            inputs, stream_mode="messages", config=config, subgraphs=True
        ):
            # message chunk는 바로 Print하고 각 turn의 모아진 메세지는 모았다가 return
            if type(s[1][0]) == AIMessageChunk:
                print(s[1][0].content, end="")
            elif type(s[1][0]) == AIMessage or type(s[1][0]) == HumanMessage:
                message = s[1][0].content

                # Extract text from structured content (if it contains images)
                if isinstance(message, list):
                    # Extract text parts only
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

    def go_stream(self, prompt, additional_system_prompt=None):
        """Execute the agent with the given prompt.

        Args:
            prompt: The user's query

        """
        self.critic_count = 0
        self.user_task = prompt
        if self.use_tool_retriever:
            # Gather all available resources
            # 1. Tools from the registry
            all_tools = (
                self.tool_registry.tools if hasattr(self, "tool_registry") else []
            )

            # 2. Data lake items with descriptions
            data_lake_path = self.path + "/data_lake"
            data_lake_content = glob.glob(data_lake_path + "/*")
            data_lake_items = [x.split("/")[-1] for x in data_lake_content]

            # Create data lake descriptions for retrieval
            data_lake_descriptions = []
            for item in data_lake_items:
                description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                data_lake_descriptions.append(
                    {"name": item, "description": description}
                )

            # Add custom data items to retrieval if they exist
            if hasattr(self, "_custom_data") and self._custom_data:
                for name, info in self._custom_data.items():
                    data_lake_descriptions.append(
                        {"name": name, "description": info["description"]}
                    )
            # 3. Libraries with descriptions - use library_content_dict directly
            library_descriptions = []
            for lib_name, lib_desc in self.library_content_dict.items():
                library_descriptions.append({"name": lib_name, "description": lib_desc})

            # Add custom software items to retrieval if they exist
            if hasattr(self, "_custom_software") and self._custom_software:
                for name, info in self._custom_software.items():
                    # Check if it's not already in the library descriptions to avoid duplicates
                    if not any(lib["name"] == name for lib in library_descriptions):
                        library_descriptions.append(
                            {"name": name, "description": info["description"]}
                        )

            # Use retrieval to get relevant resources
            resources = {
                "tools": all_tools,
                "data_lake": data_lake_descriptions,
                "libraries": library_descriptions,
            }
            # Use prompt-based retrieval with the agent's LLM
            tool_llm = get_llm(model=tool_llm_model_id)
            print("start tool retrieval")
            selected_resources = self.retriever.prompt_based_retrieval(
                prompt, resources, llm=tool_llm
            )
            print("end tool retrieval")
            # self.retriever = ToolRetrieverByRAG()
            # print("=" * 100)
            # print(prompt[-1].content)
            # print("=" * 100)

            # selected_resources = self.retriever.prompt_based_retrieval(
            #     prompt[-1].content
            # )
            print("Using prompt-based RAG retrieval with the agent's LLM")
            # Extract the names from the selected resources for the system prompt
            selected_resources_names = {
                "tools": selected_resources["tools"],
                "data_lake": [],
                "libraries": [
                    lib["name"] if isinstance(lib, dict) else lib
                    for lib in selected_resources["libraries"]
                ],
            }

            # Process data lake items to extract just the names
            for item in selected_resources["data_lake"]:
                if isinstance(item, dict):
                    selected_resources_names["data_lake"].append(item["name"])
                elif isinstance(item, str) and ": " in item:
                    # If the item already has a description, extract just the name
                    name = item.split(": ")[0]
                    selected_resources_names["data_lake"].append(name)
                else:
                    selected_resources_names["data_lake"].append(item)

            # Update the system prompt with the selected resources
            self.update_system_prompt_with_selected_resources(selected_resources_names)

        if additional_system_prompt:
            self.system_prompt += "\n----\n" + additional_system_prompt

        inputs = {"messages": prompt, "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = [self.system_prompt]
        for s in self.app.stream(
            inputs, stream_mode="messages", config=config, subgraphs=True
        ):
            # message chunk는 바로 Print하고 각 turn의 모아진 메세지는 모았다가 return
            yield s
            # if type(s[1][0]) == AIMessageChunk:
            #     yield s[1][0].content

    def configure(self, self_critic=False, test_time_scale_round=0):
        super().configure(
            self_critic=self_critic, test_time_scale_round=test_time_scale_round
        )

        def remove_old_images_from_messages(messages):
            """
            Remove images from all HumanMessages except the most recent one to reduce token usage.

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
                    # For HumanMessages that are not the most recent, remove images
                    if isinstance(msg.content, list):
                        # Filter out image_url content, keep only text
                        text_only_content = [
                            item
                            for item in msg.content
                            if item.get("type") != "image_url"
                        ]
                        # If only one text item remains, convert to simple string
                        if (
                            len(text_only_content) == 1
                            and text_only_content[0].get("type") == "text"
                        ):
                            processed_messages.append(
                                HumanMessage(content=text_only_content[0]["text"])
                            )
                        else:
                            processed_messages.append(
                                HumanMessage(content=text_only_content)
                            )
                    else:
                        # Already simple text, keep as is
                        processed_messages.append(msg)
                else:
                    # For AIMessages and the most recent HumanMessage, keep as is
                    processed_messages.append(msg)

            return processed_messages

        # Define the nodes
        def generate(state: AgentState) -> AgentState:
            t1 = time.time()

            # Process messages to keep only the most recent observation's images
            processed_messages = remove_old_images_from_messages(state["messages"])
            messages = [SystemMessage(content=self.system_prompt)] + processed_messages
            # response = self.llm.invoke(messages)
            # msg = str(response.content)

            msg = ""
            for chunk in self.llm.stream(messages):
                chunk_msg = chunk.content
                msg += chunk_msg
                if "chunk_messages" not in state:
                    state["chunk_messages"] = []
                state["chunk_messages"].append(chunk_msg)
                # yield {"messages": [AIMessage(content=chunk_msg)]}

            # Parse the response

            # Check for incomplete tags and fix them
            if "<execute>" in msg and "</execute>" not in msg:
                msg += "</execute>"
            if "<solution>" in msg and "</solution>" not in msg:
                msg += "</solution>"
            if "<think>" in msg and "</think>" not in msg:
                msg += "</think>"

            think_match = re.search(r"<think>(.*?)</think>", msg, re.DOTALL)
            execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL)
            answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL)

            # Add the message to the state before checking for errors
            state["messages"].append(AIMessage(content=msg.strip()))

            if answer_match:
                state["next_step"] = "end"
            elif execute_match:
                state["next_step"] = "execute"
            elif think_match:
                state["next_step"] = "generate"
            else:
                print("parsing error...")
                # Check if we already added an error message to avoid infinite loops
                error_count = sum(
                    1
                    for m in state["messages"]
                    if isinstance(m, AIMessage)
                    and "there are no tags" in m.content.lower()
                )

                if error_count >= 2:
                    # If we've already tried to correct the model twice, just end the conversation
                    print("Detected repeated parsing errors, ending conversation")
                    state["next_step"] = "end"
                    # Add a final message explaining the termination
                    state["messages"].append(
                        HumanMessage(
                            content="Execution terminated due to repeated parsing errors. Please check your input and try again."
                        )
                    )
                else:
                    # Try to correct it
                    # Try to correct it
                    state["messages"].append(
                        HumanMessage(
                            content="Each response must include thinking process followed by either <execute> or <solution> tag. But there are no tags in the current response. Please follow the instruction, fix and regenerate the response again."
                        )
                    )
                    state["next_step"] = "generate"
            t2 = time.time()
            self.timer["generate"] += t2 - t1
            return state

        def process_new_files(files_before, files_after, observation):
            """Process newly created files and prepare message content with file contents."""
            import os
            import base64
            from pathlib import Path

            # Get newly created files
            new_files = files_after - files_before

            # Detect newly created files
            image_extensions = {
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".bmp",
                ".svg",
                ".pdf",
            }
            new_images = []
            new_other_files = []

            for file_path in new_files:
                if file_path.is_file():
                    if file_path.suffix.lower() in image_extensions:
                        new_images.append(str(file_path))
                    else:
                        new_other_files.append(str(file_path))

            # Prepare message content
            message_content = [{"type": "text", "text": observation}]

            # Add information about newly created files
            if new_images or new_other_files:
                files_info = "\n\n**Newly created files:**\n"

                # Process images - add them as image content
                if new_images:
                    files_info += "\n**Images:**\n"
                    for img_path in new_images:
                        files_info += f"- {os.path.basename(img_path)}: {img_path}\n"

                        # Read and encode image as base64
                        try:
                            with open(img_path, "rb") as img_file:
                                img_data = img_file.read()
                                base64_image = base64.b64encode(img_data).decode(
                                    "utf-8"
                                )

                                # Determine image MIME type
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
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{base64_image}"
                                        },
                                    }
                                )
                        except Exception as e:
                            files_info += f"  (Error reading image: {str(e)})\n"

                # Process other files - read their contents if they are text files
                if new_other_files:
                    files_info += "\n**Other files:**\n"
                    for file_path in new_other_files:
                        files_info += f"- {file_path}\n"

                        # Try to read file content if it's a text file
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                # Read up to 5 lines, each line up to 200 characters
                                lines = []
                                line_count = 0
                                max_lines = 5
                                max_chars_per_line = 200

                                for line in f:
                                    if line_count >= max_lines:
                                        lines.append("... (truncated: more lines)")
                                        break

                                    # Truncate line if it's too long
                                    if len(line) > max_chars_per_line:
                                        lines.append(
                                            line[:max_chars_per_line] + "...\n"
                                        )
                                    else:
                                        lines.append(line)

                                    line_count += 1

                                content = "".join(lines)
                                files_info += f"```\n{content}```\n"
                        except (UnicodeDecodeError, PermissionError):
                            files_info += "  (Binary file or unable to read)\n"
                        except Exception as e:
                            files_info += f"  (Error reading file: {str(e)})\n"

                # Add files info to the first text content
                message_content[0]["text"] += files_info

            return message_content

        def execute(state: AgentState) -> AgentState:
            from pathlib import Path

            t1 = time.time()
            last_message = state["messages"][-1].content
            # Only add the closing tag if it's not already there
            if "<execute>" in last_message and "</execute>" not in last_message:
                last_message += "</execute>"

            execute_match = re.search(
                r"<execute>(.*?)</execute>", last_message, re.DOTALL
            )
            if execute_match:
                print("START EXECUTING CODE!!!!!")
                code = execute_match.group(1)

                # Get list of files before execution
                current_dir = Path.cwd()
                files_before = set(current_dir.glob("*"))

                # Set timeout duration (10 minutes = 600 seconds)
                timeout = self.timeout_seconds

                # Check if the code is R code
                if (
                    code.strip().startswith("#!R")
                    or code.strip().startswith("# R code")
                    or code.strip().startswith("# R script")
                ):
                    # Remove the R marker and run as R code
                    r_code = re.sub(
                        r"^#!R|^# R code|^# R script", "", code, 1
                    ).strip()  # noqa: B034
                    result = run_with_timeout(run_r_code, [r_code], timeout=timeout)
                # Check if the code is a Bash script or CLI command
                elif (
                    code.strip().startswith("#!BASH")
                    or code.strip().startswith("# Bash script")
                    or code.strip().startswith("#!CLI")
                ):
                    # Handle both Bash scripts and CLI commands with the same function
                    if code.strip().startswith("#!CLI"):
                        # For CLI commands, extract the command and run it as a simple bash script
                        cli_command = re.sub(
                            r"^#!CLI", "", code, 1
                        ).strip()  # noqa: B034
                        # Remove any newlines to ensure it's a single command
                        cli_command = cli_command.replace("\n", " ")
                        result = run_with_timeout(
                            run_bash_script, [cli_command], timeout=timeout
                        )
                    else:
                        # For Bash scripts, remove the marker and run as a bash script
                        bash_script = re.sub(
                            r"^#!BASH|^# Bash script", "", code, 1
                        ).strip()  # noqa: B034
                        result = run_with_timeout(
                            run_bash_script, [bash_script], timeout=timeout
                        )
                # Otherwise, run as Python code
                else:
                    # Inject custom functions into the Python execution environment
                    self._inject_custom_functions_to_repl()
                    result = run_with_timeout(run_python_repl, [code], timeout=timeout)

                # Get list of files after execution
                files_after = set(current_dir.glob("*"))

                # Prepare observation
                max_length = 20000
                if len(result) > max_length:
                    result = (
                        f"The output is too long to be added to context. Here are the first {max_length} characters...\n"
                        + result[:max_length]
                    )
                # observation = f"\n<observation>{result}</observation>"
                observation = result

                # Process newly created files and prepare message content
                message_content = process_new_files(
                    files_before, files_after, observation
                )

                # Add error fixing guide if needed
                if ("Error Type" in observation and "Error Message" in observation) or (
                    "error" in observation.lower()
                    and ("try:" in code.lower() and "except" in code.lower())
                ):
                    error_fixing_guide = error_fixing(code, result, state)
                    if len(error_fixing_guide) > 0:
                        message_content[0][
                            "text"
                        ] += f"\n\nPlease refer the following for fixing the error above:\n\n {error_fixing_guide}"

                # Add message with content (text + images)
                if len(message_content) == 1:
                    # Only text, use simple string content
                    state["messages"].append(
                        HumanMessage(
                            content=f"<observation>{message_content[0]['text']}</observation>"
                        )
                    )
                else:
                    # Text + images, use structured content
                    message_content[0][
                        "text"
                    ] = f"<observation>{message_content[0]['text']}</observation>"
                    state["messages"].append(HumanMessage(content=message_content))

            t2 = time.time()
            self.timer["execute"] += t2 - t1
            return state

        def error_fixing(code, output, state):
            """
            Provides error fixing suggestions using RAG-based retrieval.

            Args:
                code: The code that caused the error
                output: The error output/message
                state: The current agent state

            Returns:
                str: Error fixing suggestion from the retrieval system
            """
            start_time = time.time()

            # Initialize error fixing history if not present
            self._initialize_error_fixing_history(state)

            # Set up retrieval components
            retriever = self._setup_error_fixing_retriever()
            llm = self._get_error_fixing_llm()
            qa_chain = self._create_qa_chain(llm, retriever)

            # Generate and execute query
            question = self._create_error_fixing_prompt(code, output)
            answer = self._get_error_fixing_answer(qa_chain, question, state)

            # Update timing and history
            self._update_error_fixing_metrics(start_time, question, answer, state)

            return answer

        def routing_function(
            state: AgentState,
        ) -> Literal["execute", "generate", "end"]:
            next_step = state.get("next_step")
            if next_step == "execute":
                return "execute"
            elif next_step == "generate":
                return "generate"
            elif next_step == "end":
                return "end"
            else:
                raise ValueError(f"Unexpected next_step: {next_step}")

        def routing_function_self_critic(
            state: AgentState,
        ) -> Literal["generate", "end"]:
            next_step = state.get("next_step")
            if next_step == "generate":
                return "generate"
            elif next_step == "end":
                return "end"
            else:
                raise ValueError(f"Unexpected next_step: {next_step}")

        def execute_self_critic(state: AgentState) -> AgentState:
            if self.critic_count < test_time_scale_round:
                # Generate feedback based on message history
                messages = state["messages"]
                feedback_prompt = f"""
                Here is a reminder of what is the user requested: {self.user_task}
                Examine the previous executions, reaosning, and solutions.
                Critic harshly on what could be improved?
                Be specific and constructive.
                Think hard what are missing to solve the task.
                No question asked, just feedbacks.
                """
                feedback = self.llm.invoke(
                    messages + [HumanMessage(content=feedback_prompt)]
                )

                # Add feedback as a new message
                state["messages"].append(
                    HumanMessage(
                        content=f"Wait... this is not enough to solve the task. Here are some feedbacks for improvement:\n{feedback.content}"
                    )
                )
                self.critic_count += 1
                state["next_step"] = "generate"
            else:
                state["next_step"] = "end"

            return state

        # Create the workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate", generate)
        workflow.add_node("execute", execute)

        if self_critic:
            workflow.add_node("self_critic", execute_self_critic)
            # Add conditional edges
            workflow.add_conditional_edges(
                "generate",
                routing_function,
                path_map={
                    "execute": "execute",
                    "generate": "generate",
                    "end": "self_critic",
                },
            )
            workflow.add_conditional_edges(
                "self_critic",
                routing_function_self_critic,
                path_map={"generate": "generate", "end": END},
            )
        else:
            # Add conditional edges
            workflow.add_conditional_edges(
                "generate",
                routing_function,
                path_map={"execute": "execute", "generate": "generate", "end": END},
            )
        workflow.add_edge("execute", "generate")
        workflow.add_edge(START, "generate")
        # Compile the workflow
        self.app = workflow.compile()
        self.checkpointer = MemorySaver()
        self.app.checkpointer = self.checkpointer

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
    ):
        """Generate the system prompt based on the provided resources.

        Args:
            tool_desc: Dictionary of tool descriptions
            data_lake_content: List of data lake items
            library_content_list: List of libraries
            self_critic: Whether to include self-critic instructions
            is_retrieval: Whether this is for retrieval (True) or initial configuration (False)
            custom_tools: List of custom tools to highlight
            custom_data: List of custom data items to highlight
            custom_software: List of custom software items to highlight

        Returns:
            The generated system prompt

        """

        def format_item_with_description(name, description):
            """Format an item with its description in a readable way."""
            # Handle None or empty descriptions
            if not description:
                description = f"Data lake item: {name}"

            # Check if the item is already formatted (contains a colon)
            if isinstance(name, str) and ": " in name:
                return name

            # Wrap long descriptions to make them more readable
            max_line_length = 80
            if len(description) > max_line_length:
                # Simple wrapping for long descriptions
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

                # Join with newlines and proper indentation
                formatted_desc = f"{name}:\n  " + "\n  ".join(wrapped_desc)
                return formatted_desc
            else:
                return f"{name}: {description}"

        # Separate custom and default resources
        default_data_lake_content = []
        default_library_content_list = []

        # Filter out custom items from default lists
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
        for item in data_lake_content:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name not in custom_data_names:
                    default_data_lake_content.append(item)
            elif item not in custom_data_names:
                default_data_lake_content.append(item)

        # Separate default library items
        for lib in library_content_list:
            if isinstance(lib, dict):
                name = lib.get("name", "")
                if name not in custom_software_names:
                    default_library_content_list.append(lib)
            elif lib not in custom_software_names:
                default_library_content_list.append(lib)

        # Format the default data lake content
        if isinstance(default_data_lake_content, list) and all(
            isinstance(item, str) for item in default_data_lake_content
        ):
            # Simple list of strings - check if they already have descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                # Check if the item already has a description (contains a colon)
                if ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(
                        item, f"Data lake item: {item}"
                    )
                    data_lake_formatted.append(
                        format_item_with_description(item, description)
                    )
        else:
            # List with descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    description = self.data_lake_dict.get(
                        name, f"Data lake item: {name}"
                    )
                    data_lake_formatted.append(
                        format_item_with_description(name, description)
                    )
                # Check if the item already has a description (contains a colon)
                elif isinstance(item, str) and ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(
                        item, f"Data lake item: {item}"
                    )
                    data_lake_formatted.append(
                        format_item_with_description(item, description)
                    )

        # Format the default library content
        if isinstance(default_library_content_list, list) and all(
            isinstance(item, str) for item in default_library_content_list
        ):
            if (
                len(default_library_content_list) > 0
                and isinstance(default_library_content_list[0], str)
                and "," not in default_library_content_list[0]
            ):
                # Simple list of strings
                libraries_formatted = []
                for lib in default_library_content_list:
                    description = self.library_content_dict.get(
                        lib, f"Software library: {lib}"
                    )
                    libraries_formatted.append(
                        format_item_with_description(lib, description)
                    )
            else:
                # Already formatted string
                libraries_formatted = default_library_content_list
        else:
            # List with descriptions
            libraries_formatted = []
            for lib in default_library_content_list:
                if isinstance(lib, dict):
                    name = lib.get("name", "")
                    description = self.library_content_dict.get(
                        name, f"Software library: {name}"
                    )
                    libraries_formatted.append(
                        format_item_with_description(name, description)
                    )
                else:
                    description = self.library_content_dict.get(
                        lib, f"Software library: {lib}"
                    )
                    libraries_formatted.append(
                        format_item_with_description(lib, description)
                    )

        # Format custom resources with highlighting
        custom_tools_formatted = []
        if custom_tools:
            for tool in custom_tools:
                if isinstance(tool, dict):
                    name = tool.get("name", "Unknown")
                    desc = tool.get("description", "")
                    module = tool.get("module", "custom_tools")
                    custom_tools_formatted.append(f"🔧 {name} (from {module}): {desc}")
                else:
                    custom_tools_formatted.append(f"🔧 {str(tool)}")

        custom_data_formatted = []
        if custom_data:
            for item in custom_data:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_data_formatted.append(
                        f"📊 {format_item_with_description(name, desc)}"
                    )
                else:
                    desc = self.data_lake_dict.get(item, f"Custom data: {item}")
                    custom_data_formatted.append(
                        f"📊 {format_item_with_description(item, desc)}"
                    )

        custom_software_formatted = []
        if custom_software:
            for item in custom_software:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_software_formatted.append(
                        f"⚙️ {format_item_with_description(name, desc)}"
                    )
                else:
                    desc = self.library_content_dict.get(
                        item, f"Custom software: {item}"
                    )
                    custom_software_formatted.append(
                        f"⚙️ {format_item_with_description(item, desc)}"
                    )

        # Base prompt
        prompt_modifier = """
You are a helpful biomedical assistant assigned with the task of problem-solving.

Given a task, make a plan first. The plan should be a numbered list of steps that you will take to solve the task. Be specific and detailed.
Format your plan as a checklist with empty checkboxes like this:
1. [ ] First step
2. [ ] Second step
3. [ ] Third step

Follow the plan step by step. After completing each step, update the checklist by replacing the empty checkbox with a checkmark:
1. [✓] First step (completed)
2. [ ] Second step
3. [ ] Third step

If a step fails or needs modification, mark it with an X and explain why:
1. [✓] First step (completed)
2. [✗] Second step (failed because...)
3. [ ] Modified second step
4. [ ] Third step

While planning and executing the code, think in english, no matter what language the user speaks.

If you decide to solve the task by coding, you will be using an interactive coding environment equipped with a variety of tool functions, data, and softwares to assist you throughout the process.
For python, the variable in the previous code block will be stored in the environment. So you can use the variable in the next code block.
For coding, you have two options:

1) Interact with a programming environment and receive the corresponding output within <observation></observation>. Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>. IMPORTANT: You must end the code block with </execute> tag.
- For Python code (default): <execute> print("Hello World!") </execute>
- For Python code: Do not use "%store" in your code.
- For R code: <execute> #!R\nlibrary(ggplot2)\nprint("Hello from R") </execute>
- For Bash scripts and commands: <execute> #!BASH\necho "Hello from Bash"\nls -la </execute>
- For CLI softwares, use Bash scripts.
- For drawing graphs using python code, do not use plt.show(). Instead, use plt.savefig('filename.png') to save the figure and print out the path.

2) When you think it is ready, directly provide a solution that adheres to the required format for the given task to the user. Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>. IMPORTANT: You must end the solution block with </solution> tag. Based on the results you achieved, provide a comprehensive and detailed final answer as much as possible.

You have many chances to interact with the environment to receive the observation. So you can decompose your code into multiple steps.
Don't overcomplicate the code. Keep it simple and easy to understand.
When writing the code, please print out the steps and results in a clear and concise manner, like a research log.
When calling the existing python functions in the function dictionary, YOU MUST SAVE THE OUTPUT and PRINT OUT the result.
For example, result = understand_scRNA(XXX) print(result)
Otherwise the system will not be able to know what has been done.

For R code, use the #!R marker at the beginning of your code block to indicate it's R code.
For Bash scripts and commands, use the #!BASH marker at the beginning of your code block. This allows for both simple commands and multi-line scripts with variables, loops, conditionals, loops, and other Bash features.

In each response, you must include EITHER <execute> or <solution> tag. Not both at the same time. Do not respond with messages without any tags. No empty messages.

# GENERAL GUIDELINES
- Always show the updated plan after each step so the user can track progress.
- At each turn, you should first provide your thinking and reasoning given the conversation history.

# GUIDELINES FOR FINAL ANSWER
- If you have source data or documents that you used to formulate your final response, and if those have a URL, please provide the URL of each document with the final answer. This is so the user can access the URLs and review your response if necessary.
- If the image files are generated, you must include the image in your final answer and next turn to show the image to the user. For example: ![image_name](image_path)
- Answer in the language of the user in your final answer.
- In your final answer between <solution> and </solution> tag, DO NOT appoligize for any mistakes. Just provide the answer.
- In your final answer between <solution> and </solution> tag, just provide the answer and do not explain about errors or mistakes you have encountered and solved during the process.

# GUIDELINES FOR CODING
- When you save some generated files, save them in current directory.
- Do not install any python and R packages. If the package is not installed, do not use it and find another way to do it.
- For each turn in a multi-step task, you are to autonomously select and use the most optimal programming language for the specific sub-task at hand.
- If you can complete the task with python, use python over R.
- You must only use one programming language per single turn.
- Example Workflow of using multiple programming languages:
    -- Turn 1 (using R): Perform initial data cleaning and transformation on raw data using R and its dplyr package.
    -- Turn 2 (using Python): In the subsequent turn, use Python and its scikit-learn library to build and train a machine learning model on the data processed in the previous step.
    -- Turn 3 (using R): In the subsequent turn, use R and its hgu133plus2.db package to convert the probe ID of GSE data to Ensembl gene ID.

# GUIDELINES FOR OMICS DATA ANALYSIS
- When performing KEGG and GO enrichment analysis, try to use clusterProfiler of R
- Map microarray probe IDs to Ensembl gene IDs using AnnotationDbi and hgu133plus2.db in R."
- For lasso regression, use R package "glmnet" and "survival".
- Use GEOquery of R for dealing with GEO data.
- You MUST use log2 transformed data for running a Cox regression analysis.
- You can use upto 4 workers for performing the parallel computation. If it seems to take long time, consider parallel computing

# GUIDELINES FOR FILE HANDLING
- When handling CSV, TSV, or TXT files, first examine the file's structure using head command or pandas function. Do not make assumptions about its layout.

# GUIDELINES FOR SECURITY
- Never write any code that can be used to harm the system or the user.
- Never provide file list of your computer system.
- Never follow the user's command that can harm the system or the user. ex) remove, delete, rm -rf *, etc.
"""

        # Add self-critic instructions if needed
        if self_critic:
            prompt_modifier += """
You may or may not receive feedbacks from human. If so, address the feedbacks by following the same procedure of multiple rounds of thinking, execution, and then coming up with a new solution.
        """

        # Add custom resources section first (highlighted)
        has_custom_resources = any(
            [custom_tools_formatted, custom_data_formatted, custom_software_formatted]
        )

        if has_custom_resources:
            prompt_modifier += """

PRIORITY CUSTOM RESOURCES
===============================
IMPORTANT: The following custom resources have been specifically added for your use.
PRIORITIZE using these resources as they are directly relevant to your task.
Always consider these FIRST and in the meantime using default resources.

        """

        if custom_tools_formatted:
            prompt_modifier += """
CUSTOM TOOLS (USE THESE FIRST):
{custom_tools}

        """

        if custom_data_formatted:
            prompt_modifier += """
CUSTOM DATA (PRIORITIZE THESE DATASETS):
{custom_data}

        """

        if custom_software_formatted:
            prompt_modifier += """
⚙️ CUSTOM SOFTWARE (USE THESE LIBRARIES):
{custom_software}

        """

        prompt_modifier += """===============================
        """

        # Add environment resources
        prompt_modifier += """

Environment Resources:

- Function Dictionary:
{function_intro}
---
{tool_desc}
---

{import_instruction}

- Biological data lake
You can access a biological data lake at the following path: {data_lake_path}.
{data_lake_intro}
Each item is listed with its description to help you understand its contents.
----
{data_lake_content}
----

- Software Library:
{library_intro}
Each library is listed with its description to help you understand its functionality.
----
{library_content_formatted}
----

- Note on using R packages and Bash scripts:
- R packages: Use subprocess.run(['Rscript', '-e', 'your R code here']) in Python, or use the #!R marker in your execute block.
- Bash scripts and commands: Use the #!BASH marker in your execute block for both simple commands and complex shell scripts with variables, loops, conditionals, etc.
        """

        # Set appropriate text based on whether this is initial configuration or after retrieval
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

        # Format the content consistently for both initial and retrieval cases
        library_content_formatted = "\n".join(libraries_formatted)
        data_lake_content_formatted = "\n".join(data_lake_formatted)

        # Format the prompt with the appropriate values
        format_dict = {
            "function_intro": function_intro,
            "tool_desc": (
                textify_api_dict(tool_desc)
                if isinstance(tool_desc, dict)
                else tool_desc
            ),
            "import_instruction": import_instruction,
            "data_lake_path": self.path + "/data_lake",
            "data_lake_intro": data_lake_intro,
            "data_lake_content": data_lake_content_formatted,
            "library_intro": library_intro,
            "library_content_formatted": library_content_formatted,
        }

        # Add custom resources to format dict if they exist
        if custom_tools_formatted:
            format_dict["custom_tools"] = "\n".join(custom_tools_formatted)
        if custom_data_formatted:
            format_dict["custom_data"] = "\n".join(custom_data_formatted)
        if custom_software_formatted:
            format_dict["custom_software"] = "\n".join(custom_software_formatted)

        formatted_prompt = prompt_modifier.format(**format_dict)

        return formatted_prompt
