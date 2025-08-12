"""
POPPERAgent: Enhanced POPPER agent with biomni capabilities for automatic hypothesis validation.

This agent extends POPPER's sequential falsification testing framework with biomni's
rich biological tools, databases, and execution environment.
"""

# Standard Library Imports
import contextlib
import io
import json
import logging
import multiprocessing
import os
import re
import sys
import traceback
from pathlib import Path

# Typing and Pydantic
from typing import Annotated, Any, Literal, Optional, TypedDict, Union

# Third-Party Imports
import numpy as np
import pandas as pd
import scipy.stats as stats
from langchain_core.messages.base import get_msg_title_repr

# LangChain and LangGraph Imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.interactive_env import is_interactive_env
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from scipy.stats import chi2

from biomni.env_desc import data_lake_dict, library_content_dict

# Biomni Imports
from biomni.llm import get_llm
from biomni.model.retriever import ToolRetriever
from biomni.tool.support_tools import run_python_repl
from biomni.tool.tool_registry import ToolRegistry
from biomni.utils import api_schema_to_langchain_tool, pretty_print, read_module2api, run_with_timeout

# POPPER Imports (we'll need to handle these carefully)
# Since POPPER modules might not be in the path, we'll implement the key functions here
# or import them conditionally

# Logging Configuration
logging.getLogger("httpx").setLevel(logging.WARNING)


class TimeoutException(Exception):
    """Exception raised when execution surpasses time limit."""

    pass


def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutException("Execution surpassed time limit.")


def parse_output(solution):
    """Parse structured output from LLM.

    When we add 'include_raw=True' to structured output,
    it will return a dict w 'raw', 'parsed', 'parsing_error'.
    """
    if not solution["parsed"]:
        print("code solution fail to produce")
        print(solution)
    return solution["parsed"]


# ==================== Data Structures ====================


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
        status : Current status of the execution
        captured_output : Output from code execution
        p_val : P-value from statistical test
        tools : Available tools for the current context
    """

    error: str
    messages: list
    generation: str
    iterations: int
    status: str
    captured_output: str
    p_val: float
    tools: list | None


class code(BaseModel):
    """Code output structure."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


class test_specification(BaseModel):
    """Test specification for falsification tests."""

    test_name: str | None = Field(description="name of the test")
    test_description: str | None = Field(description="test description")
    null_hypothesis: str | None = Field(description="null hypothesis")
    alternate_hypothesis: str | None = Field(description="alternate hypothesis")


class LogLikelihoodRatioInput(BaseModel):
    """Input for likelihood ratio calculation."""

    likelihood_h1: float = Field(description="probability of data given hypothesis is alternative, P(data|h1)")
    likelihood_h0: float = Field(description="probability of data given hypothesis is null, P(data|h0)")


class parser_yes_no(BaseModel):
    """Parser for checking if output contains p-value."""

    check_output_error: str | None = Field(
        description="Does the given text contains a p-value? Yes if it has; No if does not."
    )
    p_val: str | None = Field(
        description="The p-value extracted from the text (e.g., '0.05', '3.485e-01', '1.234e-05')"
    )


class data_input_check_result(BaseModel):
    """Check if code makes up fake data entries."""

    fake_data_entries: str = Field(
        description="Does the code make up fake data entries? Yes if it does; No if does not."
    )


class relevance_subhypothesis(BaseModel):
    """Check relevance of subhypothesis to main hypothesis."""

    relevance_reasoning: str | None = Field(description="What is the reason behind this relevance score?")
    relevance_score: str | None = Field(description="relevance score")


class OutputSpecification(BaseModel):
    """Output specification for the hypothesis testing."""

    main_hypothesis: str | None = Field(description="The main hypothesis under study")
    falsification_test_result: str | None = Field(description="The result of the sequential falsification test")
    reasoning: str | None = Field(description="Reasoning, summarizing, and analyzing these results")
    conclusion: bool | None = Field(description="Conclusion on whether the hypothesis is true or false (True/False)")
    rationale: str | None = Field(description="Rationale behind the conclusion")


# ==================== Statistical Methods ====================


def likelihood_ratio_e_value(likelihood_ratio, alpha=0.1):
    """Calculate e-value from likelihood ratios."""
    likelihood_ratio = np.array(likelihood_ratio)
    cum_e = 1 / np.prod(likelihood_ratio)
    if cum_e < alpha:
        return True, cum_e
    else:
        return False, cum_e


def e_value_kappa_calibrator(p_values, alpha=0.1, kappa=0.5):
    """Calculate e-value using kappa calibrator."""
    p_values = np.array(p_values)
    e_values = kappa * p_values ** (kappa - 1)
    cum_e = np.prod(e_values)

    if cum_e > 1 / alpha:
        return True, cum_e
    else:
        return False, cum_e


def e_value_integral_calibrator(p_values, alpha=0.1):
    """Calculate e-value using integral calibrator."""
    p_values = np.array(p_values)
    e_values = (1 - p_values + p_values * np.log(p_values)) / (p_values * (-np.log(p_values)) ** 2)
    cum_e = np.prod(e_values)

    if cum_e > 1 / alpha:
        return True, cum_e
    else:
        return False, cum_e


def fishers_method(p_values, alpha=0.1):
    """Apply Fisher's method for combining p-values."""
    p_values = np.array(p_values)
    chi_square_stat = -2 * np.sum(np.log(p_values))
    degrees_of_freedom = 2 * len(p_values)
    combined_p_value = 1 - chi2.cdf(chi_square_stat, degrees_of_freedom)

    if combined_p_value < alpha:
        return True, combined_p_value
    else:
        return False, combined_p_value


# ==================== Data Loader ====================


class UnifiedDataLoader:
    """Unified data loader that combines POPPER and biomni data access."""

    def __init__(self, data_path: str | None = None, use_biomni_lake: bool = True):
        """
        Initialize the unified data loader.

        Args:
            data_path: Path to local data directory (for POPPER compatibility)
            use_biomni_lake: Whether to use biomni's data lake
        """
        self.data_path = data_path
        self.use_biomni_lake = use_biomni_lake
        self.table_dict = {}
        self.data_desc = ""

        if use_biomni_lake:
            self._init_biomni_data()

        if data_path:
            self._init_local_data()

    def _init_biomni_data(self):
        """Initialize biomni data lake access."""
        # Create description of available data
        desc_parts = ["Available datasets from biomni data lake:"]
        for name, description in data_lake_dict.items():
            desc_parts.append(f"- {name}: {description}")

        self.data_desc += "\n".join(desc_parts)

        # Note: Actual data loading would happen on-demand through tools
        self.biomni_datasets = data_lake_dict
        self.data_lake_path = "/dfs/project/bioagentos/biomni_data_test/biomni_data/data_lake"

    def _init_local_data(self):
        """Initialize local data (POPPER style)."""
        if not os.path.exists(self.data_path):
            print(f"Warning: Data path {self.data_path} does not exist")
            return

        # Load any CSV/parquet files in the data path
        for file in Path(self.data_path).glob("*.csv"):
            name = file.stem
            self.table_dict[name] = pd.read_csv(file)
            self.data_desc += f"\nLocal dataset: {name} (shape: {self.table_dict[name].shape})"

        for file in Path(self.data_path).glob("*.parquet"):
            name = file.stem
            self.table_dict[name] = pd.read_parquet(file)
            self.data_desc += f"\nLocal dataset: {name} (shape: {self.table_dict[name].shape})"

    def get_data_description(self) -> str:
        """Get description of all available data."""
        return self.data_desc

    def load_dataset(self, name: str) -> pd.DataFrame:
        """Load a specific dataset by name."""
        if name in self.table_dict:
            return self.table_dict[name]
        elif self.use_biomni_lake and name in self.biomni_datasets:
            # This would typically load from biomni's data infrastructure
            # For now, we'll return a placeholder
            print(f"Loading {name} from biomni data lake...")
            # Actual implementation would use biomni's data loading tools
            return pd.DataFrame()
        else:
            raise ValueError(f"Dataset {name} not found")


# ==================== Enhanced Test Agents ====================


class EnhancedTestProposalAgent:
    """Enhanced test proposal agent with biomni capabilities."""

    def __init__(
        self,
        data_loader: UnifiedDataLoader,
        llm,
        domain="biology",
        tool_retriever: ToolRetriever | None = None,
        tools: list | None = None,
    ):
        """
        Initialize the enhanced test proposal agent.

        Args:
            data_loader: Unified data loader
            llm: Language model
            domain: Scientific domain
            tool_retriever: Tool retriever for selecting relevant tools
            tools: List of available tools
        """
        self.data_loader = data_loader
        self.data = data_loader.get_data_description()
        self.llm = llm
        self.domain = domain
        self.tool_retriever = tool_retriever
        self.tools = tools or []
        self.existing_tests = []
        self.failed_tests = []

        # Set up prompt template
        system_prompt = self._get_system_prompt()
        self.system_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        self.chain = self.system_prompt | self.llm.with_structured_output(test_specification)
        self.output_parser = self.llm.with_structured_output(test_specification)

    def _get_system_prompt(self):
        """Get system prompt for test proposal."""
        return f"""You are a scientific hypothesis testing expert specializing in {self.domain}.
        Your task is to propose falsification tests for hypotheses.

        You have access to the following data and tools:
        - Biomni data lake with biological datasets
        - Literature search capabilities
        - Database query tools (UniProt, NCBI, etc.)
        - Statistical analysis tools
        - Molecular biology tools

        When proposing tests:
        1. Consider what data is available
        2. Design tests that can definitively falsify the hypothesis
        3. Use appropriate statistical methods
        4. Leverage available tools and databases
        """

    def go(self, main_hypothesis: str, test_results: str | None = None, log: dict | None = None) -> str:
        """Generate a test proposal."""
        if not test_results:
            test_results = "No tests conducted yet"

        # Get relevant tools if retriever is available
        relevant_tools = []
        if self.tool_retriever:
            # Use prompt-based retrieval to get relevant tools
            try:
                # Create resources dict for retrieval
                tools_list = [
                    {"name": tool.name, "description": tool.description} for tool in self.tools[:50]
                ]  # Limit for context
                resources = {"tools": tools_list, "data_lake": [], "libraries": []}

                retrieved = self.tool_retriever.prompt_based_retrieval(main_hypothesis, resources, llm=self.llm)
                tool_indices = retrieved.get("tools", [])
                relevant_tools = []
                for idx in tool_indices:
                    if isinstance(idx, int) and 0 <= idx < len(tools_list):
                        relevant_tools.append(tools_list[idx]["name"])
            except Exception as e:
                print(f"Tool retrieval failed: {e}")
                # Fallback to relevant tools based on hypothesis keywords
                main_hypothesis.lower()
                relevant_tools = []
                for tool in self.tools[:20]:  # Check first 20 tools
                    tool_name = tool.name.lower()
                    if any(keyword in tool_name for keyword in ["gene", "expression", "gtex", "query", "analyze"]):
                        relevant_tools.append(tool.name)
                    if len(relevant_tools) >= 5:
                        break
                if not relevant_tools:
                    relevant_tools = [tool.name for tool in self.tools[:5]]

        prompt = f"""
        Main hypothesis: {main_hypothesis}

        Previous test results:
        {test_results}

        Available data:
        {self.data}

        Relevant tools: {", ".join(relevant_tools) if relevant_tools else "All biomni tools available"}

        Failed tests to avoid: {self.failed_tests}

        Please propose a new falsification test that:
        1. Tests a specific aspect of the hypothesis
        2. Can be implemented with available data and tools
        3. Provides clear statistical output (p-value or likelihood ratio)
        """

        # Use React agent for complex reasoning
        app = create_react_agent(self.llm, [])
        config = {"recursion_limit": 500}
        inputs = {"messages": [("user", prompt)]}

        if log:
            log["designer"].append(f"Proposing test for: {main_hypothesis}")

        for s in app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            if log:
                log["designer"].append(message.content)

        # Parse output
        for _ in range(10):
            res = self.output_parser.invoke(s["messages"][-1].content)
            if res:
                break

        # Format as question
        question = f"""Main hypothesis: {main_hypothesis}
Falsification Test name: {res.test_name}
Falsification Test description: {res.test_description}
Falsification Test Null sub-hypothesis: {res.null_hypothesis}
Falsification Test Alternate sub-hypothesis: {res.alternate_hypothesis}"""

        return question

    def add_to_existing_tests(self, test: str):
        """Add test to existing tests list."""
        self.existing_tests.append(test)

    def add_to_failed_tests(self, test: str):
        """Add test to failed tests list."""
        self.failed_tests.append(test)


class EnhancedTestCodingAgent:
    """Enhanced test coding agent with biomni capabilities."""

    def __init__(self, data_loader: UnifiedDataLoader, llm, max_retry=10, time_limit=10, tools=None, verbose=True):
        """
        Initialize the enhanced test coding agent.

        Args:
            data_loader: Unified data loader
            llm: Language model
            max_retry: Maximum number of retries
            time_limit: Time limit in minutes
            tools: Available tools
            verbose: Whether to print verbose output
        """
        self.data_loader = data_loader
        self.data = data_loader.get_data_description()
        self.llm = llm
        self.max_retry = max_retry
        self.time_limit = time_limit
        self.tools = tools or []
        self.verbose = verbose

        # Set up prompts
        self._setup_prompts()

        # Set up workflow
        self._setup_workflow()

    def _setup_prompts(self):
        """Set up prompt templates."""
        system_prompt = f"""You are a scientific programmer specializing in hypothesis testing.
        Generate Python code to test the given hypothesis using available data and tools.

        Available data:
        {self.data}

        Available libraries:
        - All standard Python scientific libraries (numpy, scipy, pandas, etc.)
        - Bioinformatics libraries (biopython, scanpy, etc.)
        - Statistical testing libraries

        Requirements:
        1. Your code must produce a p-value for the statistical test
        2. Use actual data, do not make up fake data
        3. Import all necessary libraries
        4. Print the p-value clearly at the end
        5. Handle errors gracefully
        """

        self.code_gen_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("placeholder", "{messages}")]
        )

        # Format checker
        self.format_check_prompt = ChatPromptTemplate.from_messages(
            [("system", "Check if the output contains a p-value."), ("placeholder", "{messages}")]
        )

        self.tool_404_parser_llm = self.format_check_prompt | self.llm.with_structured_output(parser_yes_no)

        # Data checker
        self.data_check_prompt = ChatPromptTemplate.from_messages(
            [("system", "Check if code makes up fake data entries."), ("placeholder", "{messages}")]
        )

        self.data_checker = self.data_check_prompt | self.llm.with_structured_output(data_input_check_result)

    def _setup_workflow(self):
        """Set up the LangGraph workflow."""
        structured_llm = self.llm.with_structured_output(code, include_raw=True)
        code_gen_chain = self.code_gen_prompt | structured_llm | parse_output

        def generate(state: GraphState):
            """Generate code solution."""
            if self.verbose:
                print("---GENERATING CODE SOLUTION---")

            messages = state["messages"]
            iterations = state["iterations"]
            error = state.get("error")

            if error == "yes":
                messages += [("user", "Try again. Fix the error and ensure you output a p-value.")]

            # Add tool context if available - append to user message instead of system
            if self.tools and state.get("tools"):
                tool_context = f"\nAvailable tools: {', '.join([t.name for t in state['tools']])}"
                if messages:
                    # Append to the last user message
                    last_msg = messages[-1]
                    if last_msg[0] == "user":
                        messages[-1] = (last_msg[0], last_msg[1] + tool_context)
                    else:
                        messages.append(("user", tool_context))

            # Generate code
            for _ in range(20):
                code_solution = code_gen_chain.invoke({"messages": messages})
                if code_solution:
                    break

            messages += [
                ("assistant", f"{code_solution.prefix}\nImports: {code_solution.imports}\nCode: {code_solution.code}")
            ]

            iterations += 1
            return {"generation": code_solution, "messages": messages, "iterations": iterations}

        def code_check(state: GraphState):
            """Check code execution and output."""
            if self.verbose:
                print("---CHECKING CODE---")

            messages = state["messages"]
            code_solution = state["generation"]
            iterations = state["iterations"]

            imports = code_solution.imports
            code_block = code_solution.code

            # Check for fake data
            from langchain_core.messages import HumanMessage

            data_check = self.data_checker.invoke(
                {"messages": [HumanMessage(content=imports + "\n\n" + code_block)]}
            ).model_dump()
            if data_check["fake_data_entries"].lower() == "yes":
                print("Data input check failed")
                messages += [("user", "Do NOT make up fake data. Use actual data from the available datasets.")]
                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "yes",
                    "status": "Failed test",
                }

            # Execute code with biomni's execution environment
            full_code = imports + "\n\n" + code_block

            # Make data available in execution context
            exec_globals = globals().copy()
            exec_globals.update(__builtins__)

            # Add data tables to context
            for name, df in self.data_loader.table_dict.items():
                exec_globals[name] = df

            # Execute with timeout
            try:
                result = run_with_timeout(
                    lambda: self._execute_code(full_code, exec_globals), timeout=self.time_limit * 60
                )
                captured_output = result
            except TimeoutError:
                print("---CODE BLOCK CHECK: TIMEOUT---")
                messages += [("user", "Execution timeout. Please optimize your code.")]
                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "yes",
                    "status": "Failed test",
                }
            except Exception as e:
                print(f"---CODE BLOCK CHECK: FAILED---\n{e}")
                messages += [("user", f"Code execution failed: {e}")]
                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "yes",
                    "status": "Failed test",
                }

            print(f"Captured output: {captured_output}")

            # Check for p-value
            if not captured_output:
                messages += [("user", "No output captured. Ensure your code prints the p-value.")]
                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "yes",
                    "status": "Failed test",
                }

            from langchain_core.messages import HumanMessage

            checker = self.tool_404_parser_llm.invoke(
                {"messages": [HumanMessage(content=captured_output)]}
            ).model_dump()

            if checker["check_output_error"].lower() == "no":
                messages += [("user", "No p-value found in output. Ensure you calculate and print a p-value.")]
                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "yes",
                    "status": "Failed test",
                }

            try:
                p_val = float(checker["p_val"])
                if np.isnan(p_val) or p_val == 0:
                    raise ValueError(f"Invalid p-value: {p_val}")
            except Exception as e:
                messages += [("user", f"Invalid p-value: {e}")]
                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "yes",
                    "status": "Failed test",
                }

            # Success
            if self.verbose:
                print("---NO CODE TEST FAILURES---")

            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "no",
                "status": "success",
                "captured_output": captured_output,
                "p_val": p_val,
            }

        def reflect(state: GraphState):
            """Reflect on errors."""
            if self.verbose:
                print("---REFLECTING ON ERRORS---")

            messages = state["messages"]
            iterations = state["iterations"]

            reflection_prompt = "Analyze the error and suggest how to fix it."
            messages += [("user", reflection_prompt)]

            return {"messages": messages, "iterations": iterations}

        def decide_to_finish(state: GraphState):
            """Decide whether to finish or retry."""
            error = state["error"]
            iterations = state["iterations"]

            if error == "no" or iterations >= self.max_retry:
                if self.verbose:
                    print("---DECISION: FINISH---")
                return "end"
            else:
                if self.verbose:
                    print("---DECISION: RE-TRY SOLUTION---")
                return "reflect"

        # Build workflow
        workflow = StateGraph(GraphState)
        workflow.add_node("generate", generate)
        workflow.add_node("check_code", code_check)
        workflow.add_node("reflect", reflect)

        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "check_code")
        workflow.add_conditional_edges("check_code", decide_to_finish, {"end": END, "reflect": "reflect"})
        workflow.add_edge("reflect", "generate")

        self.app = workflow.compile()

    def _execute_code(self, code: str, exec_globals: dict) -> str:
        """Execute code and capture output."""
        output_capture = io.StringIO()

        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
            exec(code, exec_globals)

        return output_capture.getvalue()

    def go(self, question: str, log: dict | None = None) -> dict:
        """Execute the test coding agent."""
        print(f"Testing: {question}")

        # Get relevant tools if available
        tools = []
        if self.tools:
            # Could use tool retriever here to select relevant tools
            tools = self.tools[:10]  # Limit tools for context

        config = {"recursion_limit": 500}
        initial_state = {"messages": [("user", question)], "iterations": 0, "tools": tools}

        graph = self.app.invoke(initial_state, config=config)

        if log:
            log["executor"].append(f"Test result: {graph.get('status')}")
            if graph.get("p_val"):
                log["executor"].append(f"P-value: {graph.get('p_val')}")

        return graph


# ==================== Main POPPERAgent Class ====================


class BasePOPPERAgent:
    """
    Enhanced POPPER agent with biomni capabilities for automatic hypothesis validation.

    This agent combines POPPER's sequential falsification testing with biomni's
    rich biological tools and data resources.
    """

    def __init__(
        self,
        llm: str = "claude-sonnet-4-20250514",
        data_path: str | None = None,
        use_biomni_tools: bool = True,
        use_biomni_data: bool = True,
        is_local: bool = False,
        base_url: str | None = None,
        api_key: str = "EMPTY",
    ):
        """
        Initialize the POPPERAgent.

        Args:
            llm: Language model to use
            data_path: Path to local data (for POPPER compatibility)
            use_biomni_tools: Whether to use biomni's tool ecosystem
            use_biomni_data: Whether to use biomni's data lake
            is_local: Whether using a locally served model
            base_url: Base URL for local model (e.g., "http://localhost:8000/v1")
            api_key: API key for model
        """
        if is_local:
            assert base_url is not None, "Base URL required for local model"

        self.llm_name = llm
        self.base_url = base_url
        self.api_key = api_key
        self.llm = get_llm(llm, temperature=0.0, base_url=base_url, api_key=api_key)

        # Initialize data loader
        self.data_loader = UnifiedDataLoader(data_path=data_path, use_biomni_lake=use_biomni_data)

        # Initialize tool system
        self.tools = []
        self.tool_retriever = None
        if use_biomni_tools:
            self._init_biomni_tools()

        # Initialize output parser
        self.output_parser = self.llm.with_structured_output(OutputSpecification)

        # Initialize tracking
        self.tracked_tests = []
        self.tracked_stat = []
        self.num_of_tests = 0
        self.res = False
        self.res_stat = None

        # Initialize log
        self.log = {"designer": [], "executor": [], "relevance_checker": [], "summarizer": [], "sequential_testing": []}

        # Set up relevance checker
        self._setup_relevance_checker()

    def _init_biomni_tools(self):
        """Initialize biomni tool system."""
        print("Initializing biomni tool system...")

        # Load tool descriptions
        tools_dict = read_module2api()

        # Initialize tool registry with tools
        self.tool_registry = ToolRegistry(tools_dict)

        # Convert to LangChain tools
        self.tools = []
        for module_name, tool_list in tools_dict.items():
            for desc in tool_list:
                try:
                    tool = api_schema_to_langchain_tool(desc, mode="custom_tool", module_name=module_name)
                    self.tools.append(tool)
                except Exception as e:
                    print(f"Failed to load tool {desc.get('name', 'unknown')}: {e}")

        print(f"Loaded {len(self.tools)} biomni tools")

        # Initialize tool retriever
        if len(self.tools) > 0:
            # Create retriever for dynamic tool selection
            self.tool_retriever = ToolRetriever()
        else:
            self.tool_retriever = None

    def _setup_relevance_checker(self):
        """Set up relevance checker for hypotheses."""
        relevance_prompt = """You are evaluating whether a proposed falsification test
        is relevant to the main hypothesis.

        Score from 0 to 1:
        - 1.0: Directly tests the hypothesis
        - 0.8-0.9: Tests important aspects
        - 0.5-0.7: Somewhat relevant
        - Below 0.5: Not relevant

        Provide reasoning for your score."""

        self.relevance_checker_prompt = ChatPromptTemplate.from_messages(
            [("system", relevance_prompt), ("placeholder", "{messages}")]
        )

        self.relevance_checker = self.relevance_checker_prompt | self.llm.with_structured_output(
            relevance_subhypothesis
        )

    def configure(
        self,
        alpha: float = 0.1,
        beta: float = 0.1,
        aggregate_test: str = "E-value",
        llm_approx: bool = False,
        max_num_of_tests: int = 10,
        time_limit: int = 10,
        max_retry: int = 10,
        domain: str = "biology",
        max_failed_tests: int = 10,
        relevance_checker: bool = True,
        use_react_agent: bool = False,
    ):
        """
        Configure the POPPERAgent.

        Args:
            alpha: Significance level for hypothesis testing
            beta: Type II error rate
            aggregate_test: Method for aggregating p-values ('Fisher', 'E-value', 'E-value_integral')
            llm_approx: Whether to use LLM approximation for likelihood ratios
            max_num_of_tests: Maximum number of falsification tests
            time_limit: Time limit for each test in minutes
            max_retry: Maximum retries for each test
            domain: Scientific domain
            max_failed_tests: Maximum failed tests before stopping
            relevance_checker: Whether to check test relevance
            use_react_agent: Whether to use React-style agent
        """
        self.alpha = alpha
        self.beta = beta
        self.aggregate_test = aggregate_test
        self.llm_approx = llm_approx
        self.max_num_of_tests = max_num_of_tests
        self.domain = domain
        self.max_failed_tests = max_failed_tests
        self.use_relevance_checker = relevance_checker

        # Make data available globally (for POPPER compatibility)
        for name, df in self.data_loader.table_dict.items():
            globals()[name] = df

        # Initialize agents
        self.test_proposal_agent = EnhancedTestProposalAgent(
            data_loader=self.data_loader,
            llm=self.llm,
            domain=self.domain,
            tool_retriever=self.tool_retriever,
            tools=self.tools,
        )

        self.test_coding_agent = EnhancedTestCodingAgent(
            data_loader=self.data_loader,
            llm=self.llm,
            max_retry=max_retry,
            time_limit=time_limit,
            tools=self.tools[:20] if self.tools else None,  # Limit tools for context
            verbose=True,
        )

        # Set up workflow
        self._setup_workflow()

    def _setup_workflow(self):
        """Set up the main workflow using LangGraph."""

        class State(TypedDict):
            messages: Annotated[list, add_messages]
            cur_test_proposal: str

        def design_falsification_test(state: State):
            """Design a falsification test."""
            test_results = self._get_test_results_summary()

            if self.use_relevance_checker:
                for _i in range(self.max_failed_tests):
                    proposal = self.test_proposal_agent.go(self.main_hypothesis, test_results, self.log)

                    # Check relevance
                    check_msg = f"Subhypothesis: {proposal}; Main hypothesis: {self.main_hypothesis}"
                    from langchain_core.messages import HumanMessage

                    proposal_check = self.relevance_checker.invoke(
                        {"messages": [HumanMessage(content=check_msg)]}
                    ).model_dump()

                    if float(proposal_check["relevance_score"]) < 0.8:
                        self.test_proposal_agent.add_to_failed_tests(proposal)
                        print(f"Test not relevant enough (score: {proposal_check['relevance_score']})")
                        self.log["relevance_checker"].append(
                            f"Rejected: {proposal} (score: {proposal_check['relevance_score']})"
                        )
                    else:
                        print(f"Test approved (relevance: {proposal_check['relevance_score']})")
                        self.log["relevance_checker"].append(
                            f"Approved: {proposal} (score: {proposal_check['relevance_score']})"
                        )
                        return {
                            "cur_test_proposal": proposal,
                            "messages": [("assistant", f"Proposed test: {proposal}")],
                        }
            else:
                proposal = self.test_proposal_agent.go(self.main_hypothesis, test_results, self.log)
                return {"cur_test_proposal": proposal, "messages": [("assistant", f"Proposed test: {proposal}")]}

        def implement_falsification_test(state: State):
            """Implement the falsification test."""
            out = self.test_coding_agent.go(state["cur_test_proposal"], self.log)

            if out["status"] == "Failed test":
                self.implementation_success_status = False
                self.test_proposal_agent.add_to_failed_tests(state["cur_test_proposal"])
                return {"messages": [("assistant", "Failed to implement test")]}
            else:
                self.implementation_success_status = True
                self.test_proposal_agent.add_to_existing_tests(state["cur_test_proposal"])
                self.tracked_tests.append(state["cur_test_proposal"])

                if self.llm_approx:
                    # Would implement likelihood estimation here
                    # For now, use p-value
                    self.tracked_stat.append(float(out["p_val"]))
                else:
                    self.tracked_stat.append(float(out["p_val"]))

                return {"messages": [("assistant", f"Test complete. P-value: {out['p_val']}")]}

        def sequential_testing(state: State):
            """Perform sequential testing."""
            print("---SEQUENTIAL TESTING---")

            if self.aggregate_test == "Fisher":
                self.res, self.res_stat = fishers_method(self.tracked_stat, alpha=self.alpha)
            elif self.aggregate_test == "E-value":
                self.res, self.res_stat = e_value_kappa_calibrator(self.tracked_stat, alpha=self.alpha)
            elif self.aggregate_test == "E-value_integral":
                self.res, self.res_stat = e_value_integral_calibrator(self.tracked_stat, alpha=self.alpha)

            self.num_of_tests += 1
            res_log = "sufficient evidence - PASS" if self.res else "insufficient evidence - CONTINUE"

            output = f"P-values: {self.tracked_stat}\nCombined statistic: {self.res_stat}\nResult: {res_log}"
            print(output)
            self.log["sequential_testing"].append(output)

            return {"messages": [("assistant", output)]}

        def implementation_status(state: State) -> Literal["sequential_testing", "design_falsification_test"]:
            """Check implementation status."""
            if self.implementation_success_status:
                return "sequential_testing"
            else:
                return "design_falsification_test"

        def test_decision(state: State) -> Literal["design_falsification_test", "summarizer"]:
            """Decide next step based on test results."""
            if self.res:
                return "summarizer"
            else:
                return "design_falsification_test"

        def summarizer(state: State):
            """Summarize the results."""
            return self.summarize()

        # Build graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("design_falsification_test", design_falsification_test)
        graph_builder.add_node("implement_falsification_test", implement_falsification_test)
        graph_builder.add_node("sequential_testing", sequential_testing)
        graph_builder.add_node("summarizer", summarizer)

        graph_builder.add_edge(START, "design_falsification_test")
        graph_builder.add_edge("design_falsification_test", "implement_falsification_test")
        graph_builder.add_conditional_edges("implement_falsification_test", implementation_status)
        graph_builder.add_conditional_edges("sequential_testing", test_decision)
        graph_builder.add_edge("summarizer", END)

        self.graph = graph_builder.compile()

    def _get_test_results_summary(self) -> str:
        """Get summary of test results."""
        if not self.tracked_tests:
            return "No tests conducted yet"

        results = []
        for i, (test, stat) in enumerate(zip(self.tracked_tests, self.tracked_stat, strict=False)):
            results.append(f"Test {i + 1}: {test}\nP-value: {stat}")

        return "\n\n".join(results)

    def summarize(self) -> dict:
        """Summarize the hypothesis testing results."""
        print("---SUMMARIZING RESULTS---")

        test_summary = self._get_test_results_summary()

        if self.aggregate_test == "Fisher":
            method = f"Fisher's method (combined p-value: {self.res_stat})"
        elif self.aggregate_test == "E-value":
            method = f"E-value kappa calibrator (combined e-value: {self.res_stat})"
        elif self.aggregate_test == "E-value_integral":
            method = f"E-value integral calibrator (combined e-value: {self.res_stat})"
        else:
            method = f"{self.aggregate_test} (statistic: {self.res_stat})"

        summary = f"""
        Hypothesis Testing Summary
        ==========================

        Main Hypothesis: {self.main_hypothesis}

        Tests Conducted: {self.num_of_tests}

        Test Results:
        {test_summary}

        Aggregation Method: {method}

        Conclusion: {"Hypothesis REJECTED" if self.res else "Insufficient evidence to reject hypothesis"}

        The sequential falsification testing {"found sufficient evidence to reject" if self.res else "did not find sufficient evidence to reject"} the hypothesis.
        """

        self.log["summarizer"].append(summary)

        # Use LLM to generate structured output
        self.output_parser.invoke(summary)

        return {"messages": [("assistant", summary)]}

    def go(self, hypothesis: str) -> tuple[dict, str, dict]:
        """
        Run the hypothesis testing pipeline.

        Args:
            hypothesis: The hypothesis to test

        Returns:
            Tuple of (log, summary, result_dict)
        """
        # Reset state
        self.log = {"designer": [], "executor": [], "relevance_checker": [], "summarizer": [], "sequential_testing": []}
        self.tracked_tests = []
        self.tracked_stat = []
        self.num_of_tests = 0
        self.res = False
        self.res_stat = None
        self.main_hypothesis = hypothesis

        print(f"Testing hypothesis: {hypothesis}")

        # Run workflow
        config = {"recursion_limit": 500}

        for s in self.graph.stream({"messages": [("user", hypothesis)]}, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = message.content

            # Check stopping conditions
            if (
                self.num_of_tests >= self.max_num_of_tests
                or len(self.test_proposal_agent.failed_tests) >= self.max_failed_tests
            ):
                print("Maximum tests reached. Summarizing...")
                self.log["summarizer"].append("Maximum tests reached")
                out = self.summarize()["messages"][0][1]
                break

        # Parse final result
        try:
            result = self.output_parser.invoke(out)
            result_dict = result.model_dump()
        except Exception:
            result_dict = {
                "main_hypothesis": hypothesis,
                "falsification_test_result": f"Tests: {self.num_of_tests}, Result: {self.res}",
                "reasoning": out,
                "conclusion": self.res,
                "rationale": "Based on sequential falsification testing",
            }

        return self.log, out, result_dict


# ==================== Convenience Functions ====================


def create_base_popper_agent(
    llm: str = "claude-sonnet-4-20250514", data_path: str | None = None, use_biomni: bool = True, **kwargs
) -> BasePOPPERAgent:
    """
    Create a configured BasePOPPERAgent instance.

    Args:
        llm: Language model to use
        data_path: Path to local data
        use_biomni: Whether to use biomni capabilities
        **kwargs: Additional configuration parameters

    Returns:
        Configured BasePOPPERAgent instance
    """
    # Extract biomni-specific parameters
    base_url = kwargs.pop("base_url", None)
    api_key = kwargs.pop("api_key", "EMPTY")
    is_local = kwargs.pop("is_local", False)

    agent = BasePOPPERAgent(
        llm=llm,
        data_path=data_path,
        use_biomni_tools=use_biomni,
        use_biomni_data=use_biomni,
        base_url=base_url,
        api_key=api_key,
        is_local=is_local,
    )

    # Configure with defaults
    agent.configure(
        alpha=kwargs.get("alpha", 0.1),
        beta=kwargs.get("beta", 0.1),
        aggregate_test=kwargs.get("aggregate_test", "E-value"),
        max_num_of_tests=kwargs.get("max_num_of_tests", 10),
        domain=kwargs.get("domain", "biology"),
    )

    return agent


# ==================== Simplified POPPER Agent ====================


class A1TestCodingAgent:
    """Test coding agent that uses biomni A1 agent for code generation."""

    def __init__(self, data_loader: UnifiedDataLoader, llm, a1_agent, max_retry=10, time_limit=10, verbose=True):
        """
        Initialize the A1 test coding agent.

        Args:
            data_loader: Unified data loader
            llm: Language model (for parsing outputs)
            a1_agent: Instance of biomni A1 agent
            max_retry: Maximum number of retries
            time_limit: Time limit in minutes
            verbose: Whether to print verbose output
        """
        self.data_loader = data_loader
        self.data = data_loader.get_data_description()
        self.llm = llm
        self.a1_agent = a1_agent
        self.max_retry = max_retry
        self.time_limit = time_limit
        self.verbose = verbose

        # Set up output parser
        self.output_parser = self.llm.with_structured_output(parser_yes_no)

    def _create_hypothesis_testing_prompt(self, question: str) -> str:
        """Create a prompt for A1 agent focused on hypothesis testing."""
        # Get list of available query functions from A1 agent
        available_tools = []
        if hasattr(self, "a1_agent") and hasattr(self.a1_agent, "module2api"):
            for module, tools in self.a1_agent.module2api.items():
                if module == "biomni.tool.database":
                    for tool in tools:
                        available_tools.append(
                            f"  - {tool.get('name', 'unknown')}: {tool.get('description', 'No description')[:80]}..."
                        )

        tools_section = ""
        if available_tools:
            tools_section = f"""
Available database query functions:
{chr(10).join(available_tools)}
"""

        return f"""
{question}

IMPORTANT INSTRUCTIONS:
1. You MUST perform a statistical hypothesis test based on the given hypotheses
2. Use the available data from the data lake and database query functions
3. Calculate a p-value for the test
4. When you have completed your analysis and are ready to report the final p-value, use the following format:

<solution>
p-value: X.XXXe-XX
</solution>

(Replace X.XXXe-XX with the actual p-value, e.g., 3.485e-01 or 0.05)

5. Do NOT make up fake data - use only real data from available sources
6. The <solution> tag should only appear once at the very end with the final p-value

Available data:
{self.data}
{tools_section}
Remember: The goal is to test the hypothesis statistically and report a p-value in the <solution> tag.
"""

    def go(self, question: str, log: dict | None = None) -> dict:
        """Execute the test using A1 agent."""
        if self.verbose:
            print(f"Testing: {question}")

        # Create hypothesis testing prompt
        prompt = self._create_hypothesis_testing_prompt(question)

        # Run A1 agent
        for attempt in range(self.max_retry):
            try:
                # Execute A1 agent
                a1_log, a1_output = self.a1_agent.go(prompt)

                # Extract the output
                if isinstance(a1_output, str):
                    captured_output = a1_output
                else:
                    # Try to find observation blocks in the log
                    captured_output = ""
                    import re

                    for entry in a1_log:
                        if isinstance(entry, str) and "<observation>" in entry:
                            obs_match = re.search(r"<observation>(.*?)</observation>", entry, re.DOTALL)
                            if obs_match:
                                captured_output += obs_match.group(1) + "\n"

                    # If no observations, use the full output
                    if not captured_output:
                        captured_output = str(a1_output)

                if self.verbose:
                    print(f"A1 output: {captured_output[:500]}...")

                # Check for p-value
                from langchain_core.messages import HumanMessage

                try:
                    checker = self.output_parser.invoke([HumanMessage(content=captured_output)]).model_dump()
                except Exception as e:
                    if self.verbose:
                        print("parsing error...")
                        print(f"Error invoking output parser: {e}")
                        print(f"Output content (first 500 chars): {captured_output[:500]}")

                    # Try to extract p-value directly with regex as fallback
                    import re

                    # First try to find p-value in solution tag
                    solution_match = re.search(
                        r"<solution>.*?p-value:\s*([0-9.]+e?[-+]?\d*).*?</solution>",
                        captured_output,
                        re.IGNORECASE | re.DOTALL,
                    )
                    if solution_match:
                        p_val_match = solution_match
                    else:
                        # Fallback to general p-value search
                        p_val_match = re.search(r"p-value:\s*([0-9.]+e?[-+]?\d*)", captured_output, re.IGNORECASE)

                    if p_val_match:
                        try:
                            p_val = float(p_val_match.group(1))
                            if not np.isnan(p_val) and p_val != 0:
                                if self.verbose:
                                    print(f"Successfully extracted p-value via regex: {p_val}")

                                if log:
                                    log["executor"].append("Test completed successfully")
                                    log["executor"].append(f"P-value: {p_val}")

                                return {
                                    "status": "success",
                                    "p_val": p_val,
                                    "captured_output": captured_output,
                                    "generation": None,
                                    "messages": [],
                                    "iterations": attempt + 1,
                                    "error": "no",
                                }
                        except ValueError:
                            pass
                    continue

                if checker["check_output_error"].lower() == "yes":
                    # Extract p-value
                    try:
                        p_val = float(checker["p_val"])
                        if np.isnan(p_val) or p_val == 0:
                            raise ValueError(f"Invalid p-value: {p_val}")

                        # Success
                        if self.verbose:
                            print(f"Successfully extracted p-value: {p_val}")

                        if log:
                            log["executor"].append("Test completed successfully")
                            log["executor"].append(f"P-value: {p_val}")

                        return {
                            "status": "success",
                            "p_val": p_val,
                            "captured_output": captured_output,
                            "generation": None,
                            "messages": [],
                            "iterations": attempt + 1,
                            "error": "no",
                        }
                    except Exception as e:
                        if self.verbose:
                            print(f"Error parsing p-value: {e}")
                        continue
                else:
                    if self.verbose:
                        print("No p-value found in output, retrying...")

            except Exception as e:
                if self.verbose:
                    print(f"Attempt {attempt + 1} failed: {e}")
                if log:
                    log["executor"].append(f"Attempt {attempt + 1} failed: {e}")

        # All attempts failed
        if log:
            log["executor"].append("Failed to generate valid test after all attempts")

        return {
            "status": "Failed test",
            "p_val": None,
            "captured_output": "",
            "generation": None,
            "messages": [],
            "iterations": self.max_retry,
            "error": "yes",
        }


class POPPERAgent(BasePOPPERAgent):
    """
    POPPER agent that uses only biomni data lake and database query functions.
    Uses biomni A1 agent for test implementation instead of custom code generation.
    """

    def __init__(
        self,
        llm: str = "claude-sonnet-4-20250514",
        data_path: str | None = None,
        use_biomni_data: bool = True,
        is_local: bool = False,
        base_url: str | None = None,
        api_key: str = "EMPTY",
        use_only_database_tools: bool = True,
        a1_llm: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize the SimplifiedPOPPERAgent.

        Args:
            llm: Language model to use for POPPER logic
            data_path: Path to local data (for POPPER compatibility)
            use_biomni_data: Whether to use biomni's data lake
            is_local: Whether using a locally served model
            base_url: Base URL for local model
            api_key: API key for model
            use_only_database_tools: Whether to filter tools to only database queries
            a1_llm: Language model to use for A1 agent
        """
        self.use_only_database_tools = use_only_database_tools
        self.a1_llm = a1_llm

        # Initialize parent class
        super().__init__(
            llm=llm,
            data_path=data_path,
            use_biomni_tools=True,  # We'll filter them ourselves
            use_biomni_data=use_biomni_data,
            is_local=is_local,
            base_url=base_url,
            api_key=api_key,
        )

    def _init_biomni_tools(self):
        """Initialize biomni tool system with filtering for database tools only."""
        print("Initializing simplified biomni tool system...")

        # Load tool descriptions
        tools_dict = read_module2api()

        if self.use_only_database_tools:
            # Filter to keep only database module
            filtered_tools_dict = {}
            if "biomni.tool.database" in tools_dict:
                # Further filter to keep only query_* functions
                database_tools = []
                for tool in tools_dict["biomni.tool.database"]:
                    if tool.get("name", "").startswith("query_"):
                        database_tools.append(tool)

                if database_tools:
                    filtered_tools_dict["biomni.tool.database"] = database_tools

                print(f"Filtered to {len(database_tools)} database query tools")

            tools_dict = filtered_tools_dict

        # Initialize tool registry with filtered tools
        self.tool_registry = ToolRegistry(tools_dict)

        # Convert to LangChain tools
        self.tools = []
        for module_name, tool_list in tools_dict.items():
            for desc in tool_list:
                try:
                    tool = api_schema_to_langchain_tool(desc, mode="custom_tool", module_name=module_name)
                    self.tools.append(tool)
                except Exception as e:
                    print(f"Failed to load tool {desc.get('name', 'unknown')}: {e}")

        print(f"Loaded {len(self.tools)} tools for simplified POPPER agent")

        # Initialize tool retriever
        if len(self.tools) > 0:
            self.tool_retriever = ToolRetriever()
        else:
            self.tool_retriever = None

    def configure(
        self,
        alpha: float = 0.1,
        beta: float = 0.1,
        aggregate_test: str = "E-value",
        llm_approx: bool = False,
        max_num_of_tests: int = 10,
        time_limit: int = 10,
        max_retry: int = 10,
        domain: str = "biology",
        max_failed_tests: int = 10,
        relevance_checker: bool = True,
    ):
        """Configure the SimplifiedPOPPERAgent with A1 integration."""

        # Store configuration
        self.alpha = alpha
        self.beta = beta
        self.aggregate_test = aggregate_test
        self.llm_approx = llm_approx
        self.max_num_of_tests = max_num_of_tests
        self.domain = domain
        self.max_failed_tests = max_failed_tests
        self.use_relevance_checker = relevance_checker

        # Make data available globally (for POPPER compatibility)
        for name, df in self.data_loader.table_dict.items():
            globals()[name] = df

        # Initialize simplified test proposal agent
        self._setup_test_proposal_agent()

        # Initialize A1 agent for test coding
        print("Initializing A1 agent for test implementation...")
        from biomni.agent.a1 import A1

        # Create A1 agent with limited tools
        # Use the biomni data path
        a1_data_path = "/dfs/project/bioagentos/biomni_data_test"
        self.a1_agent = A1(
            path=a1_data_path,
            llm=self.a1_llm,
            use_tool_retriever=False,  # We'll provide specific tools
            timeout_seconds=time_limit * 60,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        # Add only database tools to A1 agent
        if self.use_only_database_tools and self.tools:
            # Clear default tools first
            self.a1_agent.module2api = {"biomni.tool.database": []}

            # Add our filtered database tools
            for tool in self.tools:
                if hasattr(tool, "name") and tool.name.startswith("query_"):
                    # Extract tool info and add to A1
                    tool_info = {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": getattr(tool, "args_schema", {}).schema() if hasattr(tool, "args_schema") else {},
                        "required_parameters": [],
                        "module": "biomni.tool.database",
                    }
                    self.a1_agent.module2api["biomni.tool.database"].append(tool_info)

        # Reconfigure A1 agent with updated tools
        self.a1_agent.configure()

        # Initialize A1 test coding agent
        self.test_coding_agent = A1TestCodingAgent(
            data_loader=self.data_loader,
            llm=self.llm,
            a1_agent=self.a1_agent,
            max_retry=max_retry,
            time_limit=time_limit,
            verbose=True,
        )

        # Set up workflow (same as parent)
        self._setup_workflow()

    def _setup_test_proposal_agent(self):
        """Set up a simplified test proposal agent."""

        # Create a simplified version with updated system prompt
        class SimplifiedTestProposalAgent(EnhancedTestProposalAgent):
            def _get_system_prompt(self):
                """Get simplified system prompt for test proposal."""
                return f"""You are a scientific hypothesis testing expert specializing in {self.domain}.
Your task is to propose falsification tests for hypotheses.

You have access to:
- Biomni data lake with biological datasets
- Database query tools (UniProt, NCBI, KEGG, STRING, etc.)

When proposing tests:
1. Focus on using available data from the data lake
2. Leverage database queries to gather additional information
3. Design tests that can definitively falsify the hypothesis
4. Ensure tests can produce clear statistical outputs (p-values)
5. Keep tests simple and focused on data analysis

Available data:
{self.data}

Note: The implementation will use biomni's A1 agent to execute the tests, so design tests that can be implemented through data analysis and database queries."""

        self.test_proposal_agent = SimplifiedTestProposalAgent(
            data_loader=self.data_loader,
            llm=self.llm,
            domain=self.domain,
            tool_retriever=self.tool_retriever,
            tools=self.tools,
        )


def create_popper_agent(llm: str = "claude-sonnet-4-20250514", data_path: str | None = None, **kwargs) -> POPPERAgent:
    """
    Create a configured POPPERAgent instance.

    Args:
        llm: Language model to use
        data_path: Path to local data
        **kwargs: Additional configuration parameters

    Returns:
        Configured POPPERAgent instance
    """
    # Extract parameters
    base_url = kwargs.pop("base_url", None)
    api_key = kwargs.pop("api_key", "EMPTY")
    is_local = kwargs.pop("is_local", False)
    a1_llm = kwargs.pop("a1_llm", llm)  # Use same LLM by default

    agent = POPPERAgent(
        llm=llm,
        data_path=data_path,
        base_url=base_url,
        api_key=api_key,
        is_local=is_local,
        use_only_database_tools=kwargs.pop("use_only_database_tools", True),
        a1_llm=a1_llm,
    )

    # Configure with defaults
    agent.configure(
        alpha=kwargs.get("alpha", 0.1),
        beta=kwargs.get("beta", 0.1),
        aggregate_test=kwargs.get("aggregate_test", "E-value"),
        max_num_of_tests=kwargs.get("max_num_of_tests", 10),
        domain=kwargs.get("domain", "biology"),
        max_retry=kwargs.get("max_retry", 10),
        time_limit=kwargs.get("time_limit", 10),
    )

    return agent


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Example usage
    agent = create_popper_agent(llm="claude-sonnet-4-20250514", use_biomni=True)

    # Test a hypothesis
    hypothesis = "Gene X is essential for cell division in cancer cells"
    log, summary, result = agent.go(hypothesis)

    print("\n=== FINAL RESULT ===")
    print(summary)
    print("\nConclusion:", result["conclusion"])
