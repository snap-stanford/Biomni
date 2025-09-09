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
)
from biomni.agent.a1 import A1
from .a1 import AgentState
from biomni.llm import get_llm
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


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

        CURRENT_FILE_PATH = os.path.abspath(__file__)
        RAG_DB_PATH = "{CURRENT_FILE_PATH}/../rag_db/faiss_index"
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
        return get_llm(model="mistral-small-2506")
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
                {"question": question, "chat_history": state["error_fixing_history"]}
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
            tool_llm = get_llm(model="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
            selected_resources = self.retriever.prompt_based_retrieval(
                prompt, resources, llm=tool_llm
            )
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

        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = [self.system_prompt]
        yield self.system_prompt
        print(self.system_prompt)
        for s in self.app.stream(
            inputs, stream_mode="messages", config=config, subgraphs=True
        ):
            # message chunk는 바로 Print하고 각 turn의 모아진 메세지는 모았다가 return
            if type(s[1][0]) == AIMessageChunk:
                print(s[1][0].content, end="")
            elif type(s[1][0]) == AIMessage or type(s[1][0]) == HumanMessage:
                message = s[1][0].content
                if type(s[1][0]) == HumanMessage:
                    print(message)
                # message = s["messages"][-1]
                # out = pretty_print(message, printout=False)
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
            tool_llm = get_llm(model="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
            selected_resources = self.retriever.prompt_based_retrieval(
                prompt, resources, llm=tool_llm
            )
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

        # Define the nodes
        def generate(state: AgentState) -> AgentState:
            t1 = time.time()
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
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
                    if isinstance(m, AIMessage) and "There are no tags" in m.content
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
                    state["messages"].append(
                        HumanMessage(
                            content="Each response must include thinking process followed by either <execute> or <solution> tag. But there are no tags in the current response. Please follow the instruction, fix and regenerate the response again."
                        )
                    )
                    state["next_step"] = "generate"
            t2 = time.time()
            self.timer["generate"] += t2 - t1
            return state

        def execute(state: AgentState) -> AgentState:
            t1 = time.time()
            last_message = state["messages"][-1].content
            # Only add the closing tag if it's not already there
            if "<execute>" in last_message and "</execute>" not in last_message:
                last_message += "</execute>"

            execute_match = re.search(
                r"<execute>(.*?)</execute>", last_message, re.DOTALL
            )
            if execute_match:
                code = execute_match.group(1)

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

                max_length = 20000
                if len(result) > max_length:
                    result = (
                        f"The output is too long to be added to context. Here are the first {max_length} characters...\n"
                        + result[:max_length]
                    )
                observation = f"\n<observation>{result}</observation>"

                if ("Error Type" in observation and "Error Message" in observation) or (
                    "error" in observation.lower()
                    and ("try:" in code.lower() and "except" in code.lower())
                ):
                    error_fixing_guide = error_fixing(code, result, state)
                    state["messages"].append(
                        HumanMessage(
                            content=f"{observation}\n\nPlease refer the following for fixing the error above:\n\n {error_fixing_guide}"
                        )
                    )
                else:
                    state["messages"].append(HumanMessage(content=f"{observation}"))
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
