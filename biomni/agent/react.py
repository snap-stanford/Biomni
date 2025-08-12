import glob
import inspect
import json
import os
import signal
from collections.abc import Sequence
from functools import wraps
from multiprocessing import Process, Queue
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from biomni.env_desc import data_lake_dict, library_content_dict
from biomni.llm import get_llm
from biomni.model.retriever import ToolRetriever
from biomni.tool.tool_registry import ToolRegistry
from biomni.utils import (
    api_schema_to_langchain_tool,
    function_to_api_schema,
    pretty_print,
    read_module2api,
)


# Define the AgentState TypedDict for our custom implementation
class AgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer that combines message sequences
    messages: Annotated[Sequence[BaseMessage], add_messages]


class react:
    def __init__(
        self,
        path="./data",
        llm="claude-3-7-sonnet-latest",
        use_tool_retriever=False,
        timeout_seconds=600,
    ):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
            ### TODO: Download the data
        else:
            print(f"Data directory already exists: {path}, loading...")

        module2api = read_module2api()

        self.llm = get_llm(llm)
        tools = []
        for module, api_list in module2api.items():
            print("Registering tools from module:", module)
            tools += [api_schema_to_langchain_tool(api, mode="custom_tool", module_name=module) for api in api_list]
        self.tools = tools
        self.module2api = module2api
        self.use_tool_retriever = use_tool_retriever

        # Store dictionaries for data lake and library content
        self.data_lake_dict = data_lake_dict
        self.library_content_dict = library_content_dict

        if self.use_tool_retriever:
            self.tool_registry = ToolRegistry(module2api)
            self.retriever = ToolRetriever()

        self.timeout_seconds = timeout_seconds  # 10 minutes default timeout

        # When wrapping tools with timeout
        self.tools = self._add_timeout_to_tools(self.tools)
        self.prompt = ""
        self.system_prompt = ""

    def _add_timeout_to_tools(self, tools):
        """Apply timeout wrapper to all tool functions using multiprocessing."""

        def create_timed_func(original_func, timeout):
            """Factory function that creates a unique timed function for each tool."""
            tool_name = getattr(original_func, "__name__", "unknown")
            # print(f"Applying timeout wrapper to tool: {tool_name}")

            def process_func(func, args, kwargs, result_queue):
                """Function to run in a separate process."""
                try:
                    result = func(*args, **kwargs)
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", str(e)))

            @wraps(original_func)
            def timed_func(*args, **kwargs):
                # print(f"Executing tool with timeout: {tool_name}")
                result_queue = Queue()

                # Start a separate process
                proc = Process(
                    target=process_func,
                    args=(original_func, args, kwargs, result_queue),
                )
                proc.start()

                # Wait for the specified timeout
                proc.join(timeout)

                # Check if the process is still running after timeout
                if proc.is_alive():
                    print(f"TIMEOUT: Tool {tool_name} execution timed out after {timeout} seconds")
                    # Force terminate the process
                    proc.terminate()
                    proc.join(1)  # Give it a second to terminate

                    # If it's still not dead, kill it with more force
                    if proc.is_alive():
                        os.kill(proc.pid, signal.SIGKILL)

                    return f"ERROR: Tool execution timed out after {timeout} seconds. Please try with simpler inputs or break your task into smaller steps."

                # Get the result from the queue
                if not result_queue.empty():
                    status, result = result_queue.get()
                    if status == "success":
                        return result
                    else:
                        return f"Error in tool execution: {result}"

                return "Error: Tool execution completed but no result was returned"

            return timed_func

        wrapped_tools = []
        for tool in tools:
            wrapped_tool = tool
            wrapped_tool.func = create_timed_func(tool.func, self.timeout_seconds)
            wrapped_tools.append(wrapped_tool)

        return wrapped_tools

    def add_tool(self, api):
        function_code = inspect.getsource(api)
        schema = function_to_api_schema(function_code, self.llm)
        new_tool = api_schema_to_langchain_tool(schema, mode="custom_tool", module_name=api.__module__)

        # Create a single wrapped tool using the existing _add_timeout_to_tools method
        wrapped_tools = self._add_timeout_to_tools([new_tool])

        # Get the wrapped tool and add it to our tools list
        if wrapped_tools:
            self.tools.append(wrapped_tools[0])

    def configure(
        self,
        plan=False,
        reflect=False,
        data_lake=False,
        react_code_search=False,
        library_access=False,
    ):
        data_lake_path = self.path + "/data_lake"
        data_lake_content = glob.glob(data_lake_path + "/*")
        data_lake_items = [x.split("/")[-1] for x in data_lake_content]

        if react_code_search:
            tools = [i for i in self.tools if i.name in ["run_python_repl", "search_google"]]

            prompt_modifier = """
You are a helpful biomedical assistant assigned with the task of problem-solving.

You have access to two tools:
1) run_python_repl: to write and run python code
2) search_google: to search google for information

You can use them to solve the problem.
            """
        else:
            tools = self.tools
            if (not plan) and (not reflect):
                prompt_modifier = """You are a helpful biologist and expert geneticist.
                """
            elif plan and (not reflect):
                prompt_modifier = """You are a helpful biologist and expert geneticist.
                Given the question from the user,
                - First, come up with a high level plan based on your understanding of the problem and available tools and record it in the Research Plan and Status. You can revise the plan later.
                - Research Plan and Status should well organized and succinctly keep track of 1) high level plan (can be revised), 2) what steps have been done and what steps are in progress, 3) short results and conclusions of each step after it has been performed. Do not perform action in research plan.
                - Research Plan and Status must only include progress that has been made by previous steps. It should not include results not directly confirmed by the previous observation.
                - Follow the plan and try to achieve the goal as straightforwardly as possible. Use tools as necessary.
                """
            elif (not plan) and reflect:
                prompt_modifier = """You are a helpful biologist and expert geneticist.
                In each round after the tool is used, conduct "reflection" step: reflect on the current state of the problem and the results of the last round. What does the observation mean? If there is an error, what caused the error and how to debug?
                """
            else:
                prompt_modifier = """You are a helpful biologist and expert geneticist.
                Given the question from the user,
                - First, come up with a high level plan based on your understanding of the problem and available tools and record it in the Research Plan and Status. You can revise the plan later.
                - Research Plan and Status should well organized and succinctly keep track of 1) high level plan (can be revised), 2) what steps have been done and what steps are in progress, 3) short results and conclusions of each step after it has been performed. Do not perform action in research plan.
                - Research Plan and Status must only include progress that has been made by previous steps. It should not include results not directly confirmed by the previous observation.
                - Follow the plan and try to achieve the goal as straightforwardly as possible. Use tools as necessary.
                In each round after the tool is used, conduct "reflection" step: reflect on the current state of the problem and the results of the last round. What does the observation mean? If there is an error, what caused the error and how to debug?
                You have access to write_python_code and run_python_repl tool to write and run your own code if tools fail, or if the given tools are not enough. Please always make sure to write code when dealing with substantial data, including finding the length of long sequences or elements at different positions.
                """

        if data_lake:
            # Format data lake items with descriptions
            data_lake_formatted = []
            for item in data_lake_items:
                description = data_lake_dict.get(item, f"Data lake item: {item}")
                data_lake_formatted.append(f"{item}: {description}")

            prompt_modifier += """
You can also access a biological data lake at the following path: {data_lake_path}. You can use the run_python_repl tool to write code to understand the data, process and utilize it for the task.
Here is the list of datasets with their descriptions:
----
{data_lake_formatted}
----
            """.format(
                data_lake_path=data_lake_path,
                data_lake_formatted="\n".join(data_lake_formatted),
            )

        if library_access:
            # Format library content with descriptions
            library_formatted = []
            for lib_name, lib_desc in library_content_dict.items():
                library_formatted.append(f"{lib_name}: {lib_desc}")

            prompt_modifier += """
You also have access to a list of software packages that can be used to perform various tasks.
You can use the run_python_repl tool to write code to access and utilize the library for the task.
Don't forget the import statement.
Here is the list of available libraries with their descriptions:
----
{library_formatted}
----
            """.format(library_formatted="\n".join(library_formatted))

        print("=" * 25 + "System Prompt" + "=" * 25)
        print(prompt_modifier)
        self.system_prompt = prompt_modifier
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_modifier),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # Store the tools for later use
        self.active_tools = tools

        # Create a custom implementation of the ReAct agent using LangGraph
        self.app = self._create_custom_react_agent(self.llm, tools, self.prompt)

    def _create_custom_react_agent(self, llm, tools, prompt):
        """Create a custom ReAct agent using LangGraph."""
        # Create a dictionary mapping tool names to tool objects for easy lookup
        tools_by_name = {tool.name: tool for tool in tools}

        # Bind the tools to the language model
        llm_with_tools = llm.bind_tools(tools)

        # Define the node that calls the model
        def call_model(state: AgentState, config: RunnableConfig = None):
            """Node that calls the language model to get the next action."""
            system_message = SystemMessage(content=self.system_prompt)
            messages = [system_message] + state["messages"]
            response = llm_with_tools.invoke(messages, config=config)
            return {"messages": [response]}

        # Define the node that executes tools
        def tool_node(state: AgentState):
            """Node that executes tools based on the LLM's decisions."""
            outputs = []
            for tool_call in state["messages"][-1].tool_calls:
                try:
                    tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                    outputs.append(
                        ToolMessage(
                            content=json.dumps(tool_result),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )
                except Exception as e:
                    # Handle any errors that occur during tool execution
                    outputs.append(
                        ToolMessage(
                            content=json.dumps({"error": str(e)}),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )
            return {"messages": outputs}

        # Define the conditional edge that determines whether to continue or not
        def should_continue(state: AgentState):
            """Determine if we should continue running the graph or finish."""
            messages = state["messages"]
            last_message = messages[-1]
            # If there is no tool call, then we finish
            if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                return "end"
            # Otherwise if there is, we continue
            else:
                return "continue"

        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)

        # Set the entrypoint as `agent`
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )

        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")

        # Compile the graph
        return workflow.compile()

    def go(self, prompt):
        """Execute the agent with the given prompt.

        Args:
            prompt: The user's query

        """
        if self.use_tool_retriever:
            # Gather all available tools from the registry
            all_tools = self.tool_registry.tools if hasattr(self, "tool_registry") else []

            # Get data lake items with descriptions
            data_lake_path = self.path + "/data_lake"
            data_lake_content = glob.glob(data_lake_path + "/*")
            data_lake_items = [x.split("/")[-1] for x in data_lake_content]

            # Create data lake descriptions for retrieval
            data_lake_descriptions = []
            for item in data_lake_items:
                description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                data_lake_descriptions.append({"name": item, "description": description})

            # Libraries with descriptions
            library_descriptions = []
            for lib_name, lib_desc in self.library_content_dict.items():
                library_descriptions.append({"name": lib_name, "description": lib_desc})

            # Prepare resources for retrieval
            resources = {
                "tools": all_tools,
                "data_lake": data_lake_descriptions,
                "libraries": library_descriptions,
            }

            # Use prompt-based retrieval with the agent's LLM
            selected_resources = self.retriever.prompt_based_retrieval(prompt, resources, llm=self.llm)
            print("Using prompt-based retrieval with the agent's LLM")

            # If we're using prompt or embedding based retrieval, print the selected resources
            print("\nSelected tools:")
            for tool in selected_resources["tools"]:
                if isinstance(tool, dict):
                    print(f"- {tool.get('name', 'Unknown')}: {tool.get('description', '')}")
                else:
                    print(f"- {getattr(tool, 'name', str(tool))}: {getattr(tool, 'description', '')}")

            print("\nSelected data lake items:")
            for item in selected_resources["data_lake"]:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    description = self.data_lake_dict.get(name, f"Data lake item: {name}")
                    print(f"- {name}: {description}")
                elif isinstance(item, str) and ": " in item:
                    # If the item already has a description, print it as is
                    print(f"- {item}")
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    print(f"- {item}: {description}")

            print("\nSelected libraries:")
            for lib in selected_resources["libraries"]:
                if isinstance(lib, dict):
                    print(f"- {lib.get('name', 'Unknown')}: {lib.get('description', '')}")
                else:
                    print(f"- {lib}")

            # Convert selected tools to langchain tool objects
            tool_names = [
                tool["name"] if isinstance(tool, dict) else getattr(tool, "name", str(tool))
                for tool in selected_resources["tools"]
            ]
            retrieved_list_of_tools = []

            # Get the tool objects by name
            for tool_name in tool_names:
                # Find the tool in the original tools list
                matching_tools = [t for t in self.tools if getattr(t, "name", None) == tool_name]
                if matching_tools:
                    retrieved_list_of_tools.append(matching_tools[0])

            # Add back coding tool if not already included
            if len([i for i in retrieved_list_of_tools if i.name == "run_python_repl"]) == 0:
                retrieved_list_of_tools = retrieved_list_of_tools + [
                    i for i in self.tools if i.name == "run_python_repl"
                ]

            print("Retrieved tools: \n" + "\n".join([l.name + ": " + l.description for l in retrieved_list_of_tools]))
            # Recreate the custom agent with the retrieved tools
            self.app = self._create_custom_react_agent(self.llm, retrieved_list_of_tools, self.prompt)

        # Default behavior (no tool retriever or retrieval_method is 'none')
        config = {"recursion_limit": 50}
        inputs = {"messages": [("user", prompt)]}
        self.log = []
        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            self.log.append(out)
        return self.log, s["messages"][-1].content

    def result_formatting(self, output_class, task_intention):
        self.format_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are evaluateGPT, tasked with extract and parse the task output based on the history of an agent. "
                        "Review the entire history of messages provided. "
                        "Here is the task output requirement: \n"
                        f"'{task_intention.replace('{', '{{').replace('}', '}}')}'.\n"
                    ),
                ),
                ("placeholder", "{messages}"),
            ]
        )

        checker_llm = self.format_check_prompt | self.llm.with_structured_output(output_class)
        result = checker_llm.invoke({"messages": [("user", str(self.log))]}).dict()
        return result
