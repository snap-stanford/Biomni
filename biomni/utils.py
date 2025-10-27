import ast
import enum
import importlib
import json
import os
import pickle
import subprocess
import tempfile
import traceback
import zipfile
from typing import Any, ClassVar
from urllib.parse import urljoin

import pandas as pd
import requests
import tqdm  # Add tqdm for progress bar
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.tools import StructuredTool
from langchain_core.utils.interactive_env import is_interactive_env
from pydantic import BaseModel, Field, ValidationError


# Add these new functions for running R code and CLI commands
def run_r_code(code: str) -> str:
    """Run R code using subprocess.

    Args:
        code: R code to run

    Returns:
        Output of the R code

    """
    try:
        # Create a temporary file to store the R code
        with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
            f.write(code)
            temp_file = f.name

        # Run the R code using Rscript
        result = subprocess.run(["Rscript", temp_file], capture_output=True, text=True, check=False)

        # Clean up the temporary file
        os.unlink(temp_file)

        # Return the output
        if result.returncode != 0:
            return f"Error running R code:\n{result.stderr}"
        else:
            return result.stdout
    except Exception as e:
        return f"Error running R code: {str(e)}"


def run_bash_script(script: str) -> str:
    """Run a Bash script using subprocess.

    Args:
        script: Bash script to run

    Returns:
        Output of the Bash script

    Example:
        This is how to use the function

        .. code-block:: python

            # Example of a complex Bash script
            script = '''
            #!/bin/bash

            # Define variables
            DATA_DIR="/path/to/data"
            OUTPUT_FILE="results.txt"

            # Create output directory if it doesn't exist
            mkdir -p $(dirname $OUTPUT_FILE)

            # Loop through files
            for file in $DATA_DIR/*.txt; do
                echo "Processing $file..."
                # Count lines in each file
                line_count=$(wc -l < $file)
                echo "$file: $line_count lines" >> $OUTPUT_FILE
            done

            echo "Processing complete. Results saved to $OUTPUT_FILE"
            '''
            result = run_bash_script(script)
            print(result)

    """
    try:
        # Trim any leading/trailing whitespace
        script = script.strip()

        # If the script is empty, return an error
        if not script:
            return "Error: Empty script"

        # Create a temporary file to store the Bash script
        with tempfile.NamedTemporaryFile(suffix=".sh", mode="w", delete=False) as f:
            # Add shebang if not present
            if not script.startswith("#!/"):
                f.write("#!/bin/bash\n")
            # Add set -e to exit on error
            if "set -e" not in script:
                f.write("set -e\n")
            f.write(script)
            temp_file = f.name

        # Make the script executable
        os.chmod(temp_file, 0o755)

        # Get current environment variables and working directory
        env = os.environ.copy()
        cwd = os.getcwd()

        # Run the Bash script with the current environment and working directory
        result = subprocess.run(
            [temp_file],
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=cwd,
        )

        # Clean up the temporary file
        os.unlink(temp_file)

        # Return the output
        if result.returncode != 0:
            traceback.print_stack()
            print(result)
            return f"Error running Bash script (exit code {result.returncode}):\n{result.stderr}"
        else:
            return result.stdout
    except Exception as e:
        traceback.print_exc()
        return f"Error running Bash script: {str(e)}"


# Keep the run_cli_command for backward compatibility
def run_cli_command(command: str) -> str:
    """Run a CLI command using subprocess.

    Args:
        command: CLI command to run

    Returns:
        Output of the CLI command

    """
    try:
        # Trim any leading/trailing whitespace
        command = command.strip()

        # If the command is empty, return an error
        if not command:
            return "Error: Empty command"

        # Split the command into a list of arguments, handling quoted arguments correctly
        import shlex

        args = shlex.split(command)

        # Run the command
        result = subprocess.run(args, capture_output=True, text=True, check=False)

        # Return the output
        if result.returncode != 0:
            return f"Error running command '{command}':\n{result.stderr}"
        else:
            return result.stdout
    except Exception as e:
        return f"Error running command '{command}': {str(e)}"


def run_with_timeout(func, args=None, kwargs=None, timeout=600):
    """Run a function with a timeout using threading instead of multiprocessing.
    This allows variables to persist in the global namespace between function calls.
    Returns the function result or a timeout error message.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    import ctypes
    import queue
    import threading

    result_queue = queue.Queue()

    def thread_func(func, args, kwargs, result_queue):
        """Function to run in a separate thread."""
        try:
            result = func(*args, **kwargs)
            result_queue.put(("success", result))
        except Exception as e:
            result_queue.put(("error", str(e)))

    # Start a separate thread
    thread = threading.Thread(target=thread_func, args=(func, args, kwargs, result_queue))
    thread.daemon = True  # Set as daemon so it will be killed when main thread exits
    thread.start()

    # Wait for the specified timeout
    thread.join(timeout)

    # Check if the thread is still running after timeout
    if thread.is_alive():
        print(f"TIMEOUT: Code execution timed out after {timeout} seconds")

        # Unfortunately, there's no clean way to force terminate a thread in Python
        # The recommended approach is to use daemon threads and let them be killed when main thread exits
        # Here, we'll try to raise an exception in the thread to make it stop
        try:
            # Get thread ID and try to terminate it
            thread_id = thread.ident
            if thread_id:
                # This is a bit dangerous and not 100% reliable
                # It attempts to raise a SystemExit exception in the thread
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
                if res > 1:
                    # Oops, we raised too many exceptions
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
        except Exception as e:
            print(f"Error trying to terminate thread: {e}")

        return f"ERROR: Code execution timed out after {timeout} seconds. Please try with simpler inputs or break your task into smaller steps."

    # Get the result from the queue if available
    try:
        status, result = result_queue.get(block=False)
        return result if status == "success" else f"Error in execution: {result}"
    except queue.Empty:
        return "Error: Execution completed but no result was returned"


class api_schema(BaseModel):
    """api schema specification."""

    api_schema: str | None = Field(description="The api schema as a dictionary")


def function_to_api_schema(function_string, llm):
    prompt = """
    Based on a code snippet and help me write an API docstring in the format like this:

    {{'name': 'get_gene_set_enrichment',
    'description': 'Given a list of genes, identify a pathway that is enriched for this gene set. Return a list of pathway name, p-value, z-scores.',
    'required_parameters': [{{'name': 'genes',
    'type': 'List[str]',
    'description': 'List of g`ene symbols to analyze',
    'default': None}}],
    'optional_parameters': [{{'name': 'top_k',
    'type': 'int',
    'description': 'Top K pathways to return',
    'default': 10}},  {{'name': 'database',
    'type': 'str',
    'description': 'Name of the database to use for enrichment analysis',
    'default': "gene_ontology"}}]}}

    Strictly follow the input from the function - don't create fake optional parameters.
    For variable without default values, set them as None, not null.
    For variable with boolean values, use capitalized True or False, not true or false.
    Do not add any return type in the docstring.
    Be as clear and succint as possible for the descriptions. Please do not make it overly verbose.
    Here is the code snippet:
    {code}
    """
    llm = llm.with_structured_output(api_schema)

    for _ in range(7):
        try:
            api = llm.invoke(prompt.format(code=function_string)).dict()["api_schema"]
            return ast.literal_eval(api)  # -> prefer "default": None
            # return json.loads(api) # -> prefer "default": null
        except Exception as e:
            print("API string:", api)
            print("Error parsing the API string:", e)
            continue

    return "Error: Could not parse the API schema"
    # return


def get_all_functions_from_file(file_path):
    with open(file_path) as file:
        file_content = file.read()

    # Parse the file content into an AST (Abstract Syntax Tree)
    tree = ast.parse(file_content)

    # List to hold the top-level functions as strings
    functions = []

    # Walk through the AST nodes
    for node in tree.body:  # Only consider top-level nodes in the body
        if isinstance(node, ast.FunctionDef):  # Check if the node is a function definition
            # Skip if function name starts with underscore
            if node.name.startswith("_"):
                continue

            start_line = node.lineno - 1  # Get the starting line of the function
            end_line = node.end_lineno  # Get the ending line of the function (only available in Python 3.8+)
            func_code = file_content.splitlines()[start_line:end_line]
            functions.append("\n".join(func_code))  # Join lines of the function and add to the list

    return functions


def write_python_code(request: str):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    template = """Write some python code to solve the user's problem.

    Return only python code in Markdown format, e.g.:

    ```python
    ....
    ```"""
    prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])

    def _sanitize_output(text: str):
        _, after = text.split("```python")
        return after.split("```")[0]

    chain = prompt | model | StrOutputParser() | _sanitize_output
    return chain.invoke({"input": "write a code that " + request})


def execute_graphql_query(
    query: str,
    variables: dict,
    api_address: str = "https://api.genetics.opentargets.org/graphql",
) -> dict:
    """Executes a GraphQL query with variables and returns the data as a dictionary."""
    headers = {"Content-Type": "application/json"}
    response = requests.post(api_address, json={"query": query, "variables": variables}, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(response.text)
        response.raise_for_status()


def get_tool_decorated_functions(relative_path):
    import ast
    import importlib.util
    import os

    # Get the directory of the current file (__init__.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path from the relative path
    file_path = os.path.join(current_dir, relative_path)

    with open(file_path) as file:
        tree = ast.parse(file.read(), filename=file_path)

    tool_function_names = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Name)
                    and decorator.id == "tool"
                    or (
                        isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "tool"
                    )
                ):
                    tool_function_names.append(node.name)

    # Calculate the module name from the relative path
    package_path = os.path.relpath(file_path, start=current_dir)
    module_name = package_path.replace(os.path.sep, ".").rsplit(".", 1)[0]

    # Import the module and get the function objects
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tool_functions = [getattr(module, name) for name in tool_function_names]

    return tool_functions


def process_bio_retrieval_ducoment(documents_df):
    ir_corpus = {}
    corpus2tool = {}
    for row in documents_df.itertuples():
        doc = row.document_content
        ir_corpus[row.docid] = (
            (doc.get("name", "") or "")
            + ", "
            + (doc.get("description", "") or "")
            + ", "
            + (doc.get("url", "") or "")
            + ", "
            + ", required_params: "
            + json.dumps(doc.get("required_parameters", ""))
            + ", optional_params: "
            + json.dumps(doc.get("optional_parameters", ""))
        )

        corpus2tool[
            (doc.get("name", "") or "")
            + ", "
            + (doc.get("description", "") or "")
            + ", "
            + (doc.get("url", "") or "")
            + ", "
            + ", required_params: "
            + json.dumps(doc.get("required_parameters", ""))
            + ", optional_params: "
            + json.dumps(doc.get("optional_parameters", ""))
        ] = doc["name"]
    return ir_corpus, corpus2tool


def load_pickle(file):
    import pickle

    with open(file, "rb") as f:
        return pickle.load(f)


def pretty_print(message, printout=True):
    if isinstance(message, tuple):
        title = message
    elif isinstance(message.content, list):
        title = get_msg_title_repr(message.type.title().upper() + " Message", bold=is_interactive_env())
        if message.name is not None:
            title += f"\nName: {message.name}"

        for i in message.content:
            if i["type"] == "text":
                title += f"\n{i['text']}\n"
            elif i["type"] == "tool_use":
                title += f"\nTool: {i['name']}"
                title += f"\nInput: {i['input']}"
        if printout:
            print(f"{title}")
    else:
        title = get_msg_title_repr(message.type.title() + " Message", bold=is_interactive_env())
        if message.name is not None:
            title += f"\nName: {message.name}"
        title += f"\n\n{message.content}"
        if printout:
            print(f"{title}")
    return title


class CustomBaseModel(BaseModel):
    api_schema: ClassVar[dict] = None  # Class variable to store api_schema

    # Add model_config with arbitrary_types_allowed=True
    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def set_api_schema(cls, schema: dict):
        cls.api_schema = schema

    @classmethod
    def model_validate(cls, obj):
        try:
            return super().model_validate(obj)
        except (ValidationError, AttributeError) as e:
            if not cls.api_schema:
                raise e  # If no api_schema is set, raise original error

            error_msg = "Required Parameters:\n"
            for param in cls.api_schema["required_parameters"]:
                error_msg += f"- {param['name']} ({param['type']}): {param['description']}\n"

            error_msg += "\nErrors:\n"
            for err in e.errors():
                field = err["loc"][0] if err["loc"] else "input"
                error_msg += f"- {field}: {err['msg']}\n"

            if not obj:
                error_msg += "\nNo input provided"
            else:
                error_msg += "\nProvided Input:\n"
                for key, value in obj.items():
                    error_msg += f"- {key}: {value}\n"

                missing_params = {param["name"] for param in cls.api_schema["required_parameters"]} - set(obj.keys())
                if missing_params:
                    error_msg += "\nMissing Parameters:\n"
                    for param in missing_params:
                        error_msg += f"- {param}\n"

            # # Create proper validation error structure
            raise ValidationError.from_exception_data(
                title="Validation Error",
                line_errors=[
                    {
                        "type": "value_error",
                        "loc": ("input",),
                        "input": obj,
                        "ctx": {
                            "error": error_msg,
                        },
                    }
                ],
            ) from None


def safe_execute_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return str(e)

    return wrapper


def api_schema_to_langchain_tool(api_schema, mode="generated_tool", module_name=None):
    if mode == "generated_tool":
        module = importlib.import_module("biomni.tool.generated_tool." + api_schema["tool_name"] + ".api")
    elif mode == "custom_tool":
        module = importlib.import_module(module_name)

    api_function = getattr(module, api_schema["name"])
    api_function = safe_execute_decorator(api_function)

    # Define a mapping from string type names to actual Python type objects
    type_mapping = {
        "string": str,
        "integer": int,
        "boolean": bool,
        "pandas": pd.DataFrame,  # Use the imported pandas.DataFrame directly
        "str": str,
        "int": int,
        "bool": bool,
        "List[str]": list[str],
        "List[int]": list[int],
        "Dict": dict,
        "Any": Any,
    }

    # Create the fields and annotations
    annotations = {}
    for param in api_schema["required_parameters"]:
        param_type = param["type"]
        if param_type in type_mapping:
            annotations[param["name"]] = type_mapping[param_type]
        else:
            # For types not in the mapping, try a safer approach than direct eval
            try:
                annotations[param["name"]] = eval(param_type)
            except (NameError, SyntaxError):
                # Default to Any for unknown types
                annotations[param["name"]] = Any

    fields = {param["name"]: Field(description=param["description"]) for param in api_schema["required_parameters"]}

    # Create the ApiInput class dynamically
    ApiInput = type("Input", (CustomBaseModel,), {"__annotations__": annotations, **fields})
    # Set the api_schema
    ApiInput.set_api_schema(api_schema)

    # Create the StructuredTool
    api_tool = StructuredTool.from_function(
        func=api_function,
        name=api_schema["name"],
        description=api_schema["description"],
        args_schema=ApiInput,
        return_direct=True,
    )

    return api_tool


class ID(enum.Enum):
    ENTREZ = "Entrez"
    ENSEMBL = "Ensembl without version"  # e.g. ENSG00000123374
    ENSEMBL_W_VERSION = "Ensembl with version"  # e.g. ENSG00000123374.10 (needed for GTEx)


def get_gene_id(gene_symbol: str, id_type: ID):
    """Get the ID for a gene symbol. If no match found, returns None."""
    if id_type == ID.ENTREZ:
        return _get_gene_id_entrez(gene_symbol)
    elif id_type == ID.ENSEMBL:
        return _get_gene_id_ensembl(gene_symbol)
    elif id_type == ID.ENSEMBL_W_VERSION:
        return _get_gene_id_ensembl_with_version(gene_symbol)
    else:
        raise ValueError(f"ID type {id_type} not supported")


def _get_gene_id_entrez(gene_symbol: str):
    """Get the Entrez ID for a gene symbol. If no match found, returns None
    e.g. 1017 (CDK2).
    """
    api_call = f"https://mygene.info/v3/query?species=human&q=symbol:{gene_symbol}"
    response = requests.get(api_call)
    response_json = response.json()

    if len(response_json["hits"]) == 0:
        return None
    else:
        return response_json["hits"][0]["entrezgene"]


def _get_gene_id_ensembl(gene_symbol):
    """Get the Ensembl ID for a gene symbol. If no match found, returns None
    e.g. ENSG00000123374.
    """
    api_call = f"https://mygene.info/v3/query?species=human&fields=ensembl&q=symbol:{gene_symbol}"
    response = requests.get(api_call)
    response_json = response.json()

    if len(response_json["hits"]) == 0:
        return None
    else:
        ensembl = response_json["hits"][0]["ensembl"]
        if isinstance(ensembl, list):
            return ensembl[0][
                "gene"
            ]  # Sometimes returns a list, for example RNH1 (first elem is on chr11, second is on scaffold_hschr11)
        else:
            return ensembl["gene"]


def _get_gene_id_ensembl_with_version(gene_symbol):
    """Get the Ensembl ID for a gene symbol. If no match found, returns None
    e.g. ENSG00000123374.10.
    """
    api_base = "https://gtexportal.org/api/v2/reference/gene"
    params = {"geneId": gene_symbol}
    response_json = requests.get(api_base, params=params).json()

    if len(response_json["data"]) == 0:
        return None
    else:
        return response_json["data"][0]["gencodeId"]


def save_pkl(f, filename):
    with open(filename, "wb") as file:
        pickle.dump(f, file)


def load_pkl(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


_TEXT_COLOR_MAPPING = {
    "blue": "36;1",
    "yellow": "33;1",
    "pink": "38;5;200",
    "green": "32;1",
    "red": "31;1",
}


def color_print(text, color="blue"):
    color_str = _TEXT_COLOR_MAPPING[color]
    print(f"\u001b[{color_str}m\033[1;3m{text}\u001b[0m")


class PromptLogger(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        for message in messages[0]:
            color_print(message.pretty_repr(), color="green")


class NodeLogger(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):  # response of type LLMResult
        for generations in response.generations:  # response.generations of type List[List[Generations]] becuase "each input could have multiple candidate generations"
            for generation in generations:
                generated_text = generation.message.content
                # token_usage = generation.message.response_metadata["token_usage"]
                color_print(generated_text, color="yellow")

    def on_agent_action(self, action, **kwargs):
        color_print(action.log, color="pink")

    def on_agent_finish(self, finish, **kwargs):
        color_print(finish, color="red")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name")
        color_print(f"Calling {tool_name} with inputs: {input_str}", color="pink")

    def on_tool_end(self, output, **kwargs):
        output = str(output)
        color_print(output, color="blue")


def check_or_create_path(path=None):
    # Set a default path if none is provided
    if path is None:
        path = os.path.join(os.getcwd(), "tmp_directory")

    # Check if the path exists
    if not os.path.exists(path):
        # If it doesn't exist, create the directory
        os.makedirs(path)
        print(f"Directory created at: {path}")
    else:
        print(f"Directory already exists at: {path}")

    return path


def langchain_to_gradio_message(message):
    # Build the title and content based on the message type
    if isinstance(message.content, list):
        # For a message with multiple content items (like text and tool use)
        gradio_messages = []
        for item in message.content:
            gradio_message = {
                "role": "user" if message.type == "human" else "assistant",
                "content": "",
                "metadata": {},
            }

            if item["type"] == "text":
                item["text"] = item["text"].replace("<think>", "\n")
                item["text"] = item["text"].replace("</think>", "\n")
                gradio_message["content"] += f"{item['text']}\n"
                gradio_messages.append(gradio_message)
            elif item["type"] == "tool_use":
                if item["name"] == "run_python_repl":
                    gradio_message["metadata"]["title"] = "🛠️ Writing code..."
                    # input = "```python {code_block}```\n".format(code_block=item['input']["command"])
                    gradio_message["metadata"]["log"] = "Executing Code block..."
                    gradio_message["content"] = f"##### Code: \n ```python \n {item['input']['command']} \n``` \n"
                else:
                    gradio_message["metadata"]["title"] = f"🛠️ Used tool ```{item['name']}```"
                    to_print = ";".join([i + ": " + str(j) for i, j in item["input"].items()])
                    gradio_message["metadata"]["log"] = f"🔍 Input -- {to_print}\n"
                gradio_message["metadata"]["status"] = "pending"
                gradio_messages.append(gradio_message)

    else:
        gradio_message = {
            "role": "user" if message.type == "human" else "assistant",
            "content": "",
            "metadata": {},
        }
        print(message)
        content = message.content
        content = content.replace("<think>", "\n")
        content = content.replace("</think>", "\n")
        content = content.replace("<solution>", "\n")
        content = content.replace("</solution>", "\n")

        gradio_message["content"] = content
        gradio_messages = [gradio_message]
    return gradio_messages


def parse_hpo_obo(file_path):
    """Parse the HPO OBO file and create a dictionary mapping HP IDs to phenotype descriptions.

    Args:
        file_path (str): Path to the HPO OBO file.

    Returns:
        dict: A dictionary where keys are HP IDs and values are phenotype descriptions.

    """
    hp_dict = {}
    current_id = None
    current_name = None

    with open(file_path) as file:
        for line in file:
            line = line.strip()
            if line.startswith("[Term]"):
                # If a new term block starts, save the previous term
                if current_id and current_name:
                    hp_dict[current_id] = current_name
                current_id = None
                current_name = None
            elif line.startswith("id: HP:"):
                current_id = line.split(": ")[1]
            elif line.startswith("name:"):
                current_name = line.split(": ", 1)[1]

        # Add the last term to the dictionary
        if current_id and current_name:
            hp_dict[current_id] = current_name

    return hp_dict


def textify_api_dict(api_dict):
    """Convert a nested API dictionary to a nicely formatted string."""
    lines = []
    for category, methods in api_dict.items():
        lines.append(f"Import file: {category}")
        lines.append("=" * (len("Import file: ") + len(category)))
        for method in methods:
            lines.append(f"Method: {method.get('name', 'N/A')}")
            lines.append(f"  Description: {method.get('description', 'No description provided.')}")

            # Process required parameters
            req_params = method.get("required_parameters", [])
            if req_params:
                lines.append("  Required Parameters:")
                for param in req_params:
                    param_name = param.get("name", "N/A")
                    param_type = param.get("type", "N/A")
                    param_desc = param.get("description", "No description")
                    param_default = param.get("default", "None")
                    lines.append(f"    - {param_name} ({param_type}): {param_desc} [Default: {param_default}]")

            # Process optional parameters
            opt_params = method.get("optional_parameters", [])
            if opt_params:
                lines.append("  Optional Parameters:")
                for param in opt_params:
                    param_name = param.get("name", "N/A")
                    param_type = param.get("type", "N/A")
                    param_desc = param.get("description", "No description")
                    param_default = param.get("default", "None")
                    lines.append(f"    - {param_name} ({param_type}): {param_desc} [Default: {param_default}]")

            lines.append("")  # Empty line between methods
        lines.append("")  # Extra empty line after each category

    return "\n".join(lines)


def read_module2api():
    fields = [
        "literature",
        "biochemistry",
        "bioimaging",
        "bioengineering",
        "biophysics",
        "glycoengineering",
        "cancer_biology",
        "cell_biology",
        "molecular_biology",
        "genetics",
        "genomics",
        "immunology",
        "microbiology",
        "pathology",
        "pharmacology",
        "physiology",
        "synthetic_biology",
        "systems_biology",
        "support_tools",
        "database",
        "lab_automation",
    ]

    module2api = {}
    for field in fields:
        module_name = f"biomni.tool.tool_description.{field}"
        module = importlib.import_module(module_name)
        module2api[f"biomni.tool.{field}"] = module.description
    return module2api


def download_and_unzip(url: str, dest_dir: str) -> str:
    """Download a zip file from a URL and extract it to the destination directory.

    Args:
        url: The URL to download the zip file from.
        dest_dir: The directory to extract the contents to.

    Returns:
        The path to the extracted directory, or an error message.

    """
    try:
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Downloading from {url} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            chunk_size = 8192
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                with tqdm.tqdm(
                    total=total_size / (1024**3),
                    unit="GB",
                    unit_scale=True,
                    desc="Downloading",
                    ncols=80,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            tmp_file.write(chunk)
                            pbar.update(len(chunk) / (1024**3))
                tmp_zip_path = tmp_file.name
        print(f"Downloaded to {tmp_zip_path}. Extracting...")
        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
        os.unlink(tmp_zip_path)
        print(f"Extraction complete to {dest_dir}")
        return dest_dir
    except Exception as e:
        print(f"Error downloading or extracting zip: {e}")
        return f"Error: {e}"


def check_and_download_s3_files(
    s3_bucket_url: str, local_data_lake_path: str, expected_files: list[str], folder: str = "data_lake"
) -> dict[str, bool]:
    """Check for missing files in the local data lake and download them from S3 bucket.

    Args:
        s3_bucket_url: Base URL of the S3 bucket (e.g., "https://biomni-release.s3.amazonaws.com")
        local_data_lake_path: Local path to the data lake directory
        expected_files: List of expected file names in the data lake
        folder: S3 folder name ("data_lake" or "benchmark")

    Returns:
        Dictionary mapping file names to download success status
    """

    os.makedirs(local_data_lake_path, exist_ok=True)
    download_results = {}

    def download_with_progress(url: str, file_path: str, desc: str) -> bool:
        """Download file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(file_path, "wb") as f:
                if total_size > 0:
                    with tqdm.tqdm(total=total_size, unit="B", unit_scale=True, desc=desc, ncols=80) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            print(f"✗ Failed to download {desc}: {e}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            return False

    def cleanup_file(file_path: str):
        """Clean up file if it exists."""
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

    # Handle benchmark folder (download as zip)
    if folder == "benchmark":
        print(f"Downloading entire {folder} folder structure...")
        s3_zip_url = urljoin(s3_bucket_url + "/", folder + ".zip")

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
            tmp_zip_path = tmp_zip.name

            if download_with_progress(s3_zip_url, tmp_zip_path, f"{folder}.zip"):
                print(f"Extracting {folder}.zip...")
                try:
                    with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
                        zip_ref.extractall(local_data_lake_path)
                    print(f"✓ Successfully downloaded and extracted {folder} folder")
                    download_results = dict.fromkeys(expected_files, True)
                except Exception as e:
                    print(f"✗ Error extracting {folder}.zip: {e}")
                    download_results = dict.fromkeys(expected_files, False)
                finally:
                    cleanup_file(tmp_zip_path)
            else:
                download_results = dict.fromkeys(expected_files, False)

        return download_results

    # Handle data_lake folder (download individual files)
    for filename in expected_files:
        local_file_path = os.path.join(local_data_lake_path, filename)

        if os.path.exists(local_file_path):
            download_results[filename] = True
            continue

        s3_file_url = urljoin(s3_bucket_url + "/" + folder + "/", filename)
        print(f"Downloading {filename} from {folder}...")

        if download_with_progress(s3_file_url, local_file_path, filename):
            print(f"✓ Successfully downloaded: {filename}")
            download_results[filename] = True
        else:
            download_results[filename] = False

    return download_results


def clean_message_content(content: str) -> str:
    """Clean message content by removing ANSI escape codes.

    This function removes ANSI escape sequences (like color codes) from text content
    that might be present in terminal output or console messages. This ensures clean
    text for markdown generation and PDF conversion.

    Args:
        content: The raw message content that may contain ANSI escape codes

    Returns:
        Cleaned content with ANSI escape codes removed

    Example:
        >>> clean_message_content("Hello \x1b[31mworld\x1b[0m!")
        "Hello world!"
    """
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", content)


def should_skip_message(clean_output: str) -> bool:
    """Check if message should be skipped during markdown generation.

    This function determines whether a message should be excluded from the final
    markdown output. It skips empty or meaningless messages but preserves important
    error messages that should be displayed to users.

    Args:
        clean_output: The cleaned message content to evaluate

    Returns:
        True if the message should be skipped, False otherwise

    Note:
        Parsing error messages are intentionally not skipped as they provide
        important feedback to users about conversation flow issues.
    """
    return (
        clean_output.strip() in ["", "None", "null", "undefined"]
        # Don't skip parsing error messages - they should be displayed and increment step counter
        # or "There are no tags" in clean_output
        # or "Execution terminated due to repeated parsing errors" in clean_output
    )


def has_execution_results(clean_output: str, execution_results) -> bool:
    """Check if message contains code execution and has associated results.

    This function determines whether a message contains executable code and has
    corresponding execution results available for display in the markdown output.

    Args:
        clean_output: The cleaned message content to check for execute tags
        execution_results: List of execution results from the agent's execution history

    Returns:
        True if the message contains <execute> tags and has execution results available
    """
    return "<execute>" in clean_output and execution_results is not None and execution_results


def find_matching_execution(clean_output: str, execution_results) -> dict | None:
    """Find the execution result that matches the given message content.

    This function searches through the execution results to find the one that
    corresponds to the current message. It matches based on the triggering message
    content to associate execution results with their originating AI messages.

    Args:
        clean_output: The cleaned message content to match against
        execution_results: List of execution result dictionaries containing
                         triggering messages and execution data

    Returns:
        The matching execution result dictionary if found, None otherwise

    Note:
        The matching is bidirectional - it checks if either the triggering message
        is contained in the current output or vice versa to handle partial matches.
    """
    for exec_result in execution_results:
        if exec_result["triggering_message"] in clean_output or clean_output in exec_result["triggering_message"]:
            return exec_result
    return None


def create_parsing_error_html() -> str:
    """Create HTML markup for displaying parsing errors in markdown output.

    This function generates a styled HTML block that displays parsing errors
    when the agent's response doesn't contain the required tags. The HTML
    uses CSS classes for consistent styling in the final PDF output.

    Returns:
        HTML string containing a styled parsing error message box

    Note:
        The returned HTML uses CSS classes defined in get_pdf_css_content()
        for consistent styling across the document.
    """
    return """
<div class="parsing-error-box">
    <div class="parsing-error-header">Parsing Error</div>
    <div class="parsing-error-content">Each response must include thinking process followed by either execute or solution tag. But there are no tags in the current response.</div>
</div>
"""


def parse_tool_calls_from_code(code: str, module2api: dict, custom_functions: dict = None) -> list[str]:
    """Parse code to detect imported tools by analyzing import statements.

    This function analyzes Python code to identify which tools/functions are being
    imported and used. It extracts tool names from import statements and function
    calls, then returns a deduplicated list of detected tool names.

    Args:
        code: The Python code string to analyze for tool imports
        module2api: Dictionary mapping module names to their available API tools
        custom_functions: Optional dictionary of custom functions that have been
                         added to the agent

    Returns:
        Sorted list of unique tool names detected in the code

    Example:
        >>> code = "from biomni.tool import analyze_data\nimport pandas as pd"
        >>> parse_tool_calls_from_code(code, module2api)
        ['analyze_data', 'pandas']
    """
    tool_module_pairs = parse_tool_calls_with_modules(code, module2api, custom_functions)
    return sorted({pair[0] for pair in tool_module_pairs})


def parse_tool_calls_with_modules(code: str, module2api: dict, custom_functions: dict = None) -> list[tuple[str, str]]:
    """Parse code to detect imported tools and their associated modules.

    This function performs detailed analysis of Python code to identify which
    tools/functions are being imported and which modules they belong to. It
    handles various import patterns including direct imports, from-imports,
    and module.function patterns.

    Args:
        code: The Python code string to analyze for tool imports
        module2api: Dictionary mapping module names to their available API tools
        custom_functions: Optional dictionary of custom functions that have been
                         added to the agent

    Returns:
        List of tuples containing (tool_name, module_name) pairs for each
        detected tool and its associated module

    Note:
        The function uses regex patterns to match various import statement
        formats and also detects direct function calls without explicit imports.
    """
    import re

    detected_tools = set()

    # Get all available tools from module2api
    all_tools = {}
    for module_name, module_tools in module2api.items():
        for tool in module_tools:
            if isinstance(tool, dict) and "name" in tool:
                tool_name = tool["name"]
                if tool_name not in all_tools:
                    all_tools[tool_name] = []
                all_tools[tool_name].append(module_name)

    # Add custom tools
    if custom_functions:
        for tool_name in custom_functions.keys():
            if tool_name not in all_tools:
                all_tools[tool_name] = []
            all_tools[tool_name].append("custom_tools")

    # Look for import statements in the code
    import_patterns = [
        r"from\s+([\w.]+)\s+import\s+([\w,\s]+)",  # from module import tool1, tool2
        r"import\s+([\w.]+)",  # import module
    ]

    for pattern in import_patterns:
        matches = re.findall(pattern, code)
        for match in matches:
            if len(match) == 2:  # from module import tools
                module_name, tools_str = match
                # Split tools by comma and clean up
                tools = [tool.strip() for tool in tools_str.split(",")]

                for tool in tools:
                    # Check if this tool exists in any module
                    if tool in all_tools:
                        # Find the best matching module
                        best_module = find_best_module_match(module_name, all_tools[tool])
                        detected_tools.add((tool, best_module))
                    # Also check if it's a module.function pattern
                    elif "." in tool:
                        parts = tool.split(".")
                        if len(parts) == 2:
                            module_part, func_part = parts
                            if func_part in all_tools:
                                best_module = find_best_module_match(module_part, all_tools[func_part])
                                detected_tools.add((func_part, best_module))

            elif len(match) == 1:  # import module
                module_name = match[0]
                # Check if any tools from this module are used
                for tool_name, modules in all_tools.items():
                    if any(module_name in mod for mod in modules):
                        # Look for usage of this tool in the code
                        if re.search(rf"\b{tool_name}\s*\(", code):
                            best_module = find_best_module_match(module_name, modules)
                            detected_tools.add((tool_name, best_module))

    # Also look for direct function calls without imports
    function_call_pattern = r"(\w+)\s*\("
    function_calls = re.findall(function_call_pattern, code)

    for func_call in function_calls:
        if func_call in all_tools:
            # For direct calls, use the first available module
            best_module = all_tools[func_call][0]
            detected_tools.add((func_call, best_module))

    return sorted(detected_tools)


def find_best_module_match(target_module: str, available_modules: list[str]) -> str:
    """Find the best matching module from a list of available modules.

    This function attempts to match a target module name against a list of
    available modules using various matching strategies: exact match, partial
    substring matches, and fallback to the first available module.

    Args:
        target_module: The module name we're trying to match
        available_modules: List of available module names to search through

    Returns:
        The best matching module name from the available modules list.
        Returns "unknown" if no modules are available.

    Note:
        The matching strategy prioritizes exact matches, then partial matches
        (where either the target is contained in the module name or vice versa),
        and finally falls back to the first available module.
    """
    # First try exact match
    if target_module in available_modules:
        return target_module

    # Try partial matches
    for module in available_modules:
        if target_module in module or module in target_module:
            return module

    # Return the first available module as fallback
    return available_modules[0] if available_modules else "unknown"


def inject_custom_functions_to_repl(custom_functions: dict):
    """Inject custom functions into the Python REPL execution environment.

    This function makes custom tools available during code execution by injecting
    them into both the persistent execution namespace and the builtins module.
    This allows the agent to call custom functions that users have added via
    agent.add_tool() when executing Python code in <execute> blocks.

    Args:
        custom_functions: Dictionary mapping function names to their callable objects

    Note:
        The function modifies both the persistent namespace used by run_python_repl
        and the builtins module to ensure maximum compatibility and accessibility
        of custom functions during code execution.
    """
    if custom_functions:
        # Access the persistent namespace used by run_python_repl
        from biomni.tool.support_tools import _persistent_namespace

        # Inject all custom functions into the execution namespace
        for name, func in custom_functions.items():
            _persistent_namespace[name] = func

        # Also make them available in builtins for broader access
        import builtins

        if not hasattr(builtins, "_biomni_custom_functions"):
            builtins._biomni_custom_functions = {}
        builtins._biomni_custom_functions.update(custom_functions)


def format_execute_tags_in_content(content: str, parse_tool_calls_with_modules_func) -> str:
    """Format execute tags in content by extracting code and creating highlighted tool call blocks.

    This function processes content that contains <execute>...</execute> tags and
    converts them into styled HTML blocks that display the code with syntax highlighting
    and information about which tools are being used.

    Args:
        content: The content string that may contain <execute> tags
        parse_tool_calls_with_modules_func: Function to parse tool calls with modules
                                          (typically parse_tool_calls_with_modules)

    Returns:
        Formatted content with execute tags converted to highlighted tool call blocks.
        Also processes <solution> tags in the same pass.

    Note:
        The function also calls format_solution_tags_in_content() to handle
        solution tags in the same processing pass.
    """
    import re

    # Pattern to match <execute>...</execute> blocks
    execute_pattern = r"<execute>(.*?)</execute>"

    def replace_execute_tag(match):
        code_content = match.group(1).strip()
        language, tool_name = detect_code_language_and_tool(code_content)
        code_content = clean_code_content(code_content, language)

        # Parse tools from the code content with module information
        detected_tool_modules = parse_tool_calls_with_modules_func(code_content)

        # Create the formatted block
        formatted_block = create_tool_call_block(code_content, language, tool_name, detected_tool_modules)
        return formatted_block

    # Replace all execute tags with formatted tool call blocks
    formatted_content = re.sub(execute_pattern, replace_execute_tag, content, flags=re.DOTALL)

    # Also format solution tags
    formatted_content = format_solution_tags_in_content(formatted_content)

    return formatted_content


def detect_code_language_and_tool(code_content: str) -> tuple[str, str]:
    """Detect the programming language and tool name from code content.

    This function analyzes code content to determine the programming language
    and appropriate tool name based on language markers at the beginning of
    the code block.

    Args:
        code_content: The code content to analyze for language markers

    Returns:
        Tuple containing (language, tool_name) where:
        - language: The detected programming language ("python", "r", "bash")
        - tool_name: The human-readable tool name for display

    Example:
        >>> detect_code_language_and_tool("#!R\nlibrary(ggplot2)")
        ("r", "R REPL")
        >>> detect_code_language_and_tool("#!BASH\necho 'hello'")
        ("bash", "Bash Script")
    """
    if code_content.startswith("#!R") or code_content.startswith("# R code") or code_content.startswith("# R script"):
        return "r", "R REPL"
    elif code_content.startswith("#!BASH") or code_content.startswith("# Bash script"):
        return "bash", "Bash Script"
    elif code_content.startswith("#!CLI"):
        return "bash", "CLI Command"
    else:
        return "python", "Python REPL"


def clean_code_content(code_content: str, language: str) -> str:
    """Clean code content by removing language markers.

    This function removes language-specific markers from the beginning of code
    content to prepare it for display in code blocks. The markers are used
    internally for language detection but should not appear in the final output.

    Args:
        code_content: The raw code content that may contain language markers
        language: The detected programming language ("python", "r", "bash")

    Returns:
        Cleaned code content with language markers removed

    Example:
        >>> clean_code_content("#!R\nlibrary(ggplot2)", "r")
        "library(ggplot2)"
        >>> clean_code_content("#!BASH\necho 'hello'", "bash")
        "echo 'hello'"
    """
    import re

    if language == "r":
        return re.sub(r"^#!R|^# R code|^# R script", "", code_content, count=1).strip()
    elif language == "bash":
        if code_content.startswith("#!BASH") or code_content.startswith("# Bash script"):
            return re.sub(r"^#!BASH|^# Bash script", "", code_content, count=1).strip()
        elif code_content.startswith("#!CLI"):
            return re.sub(r"^#!CLI", "", code_content, count=1).strip()
    return code_content


def create_tool_call_block(code_content: str, language: str, tool_name: str, detected_tool_modules: list) -> str:
    """Create the HTML block for tool call highlighting.

    This function generates a styled HTML block that displays code execution
    information including the code itself, syntax highlighting, and a list of
    tools that were used during execution.

    Args:
        code_content: The cleaned code content to display
        language: The programming language for syntax highlighting
        tool_name: The default tool name to display if no specific tools detected
        detected_tool_modules: List of (tool_name, module_name) tuples for tools used

    Returns:
        HTML string containing a styled tool call block with code and tool information

    Note:
        The HTML uses CSS classes defined in get_pdf_css_content() for styling.
        If no specific tools are detected, it falls back to a default tool name.
    """
    # Create the formatted block with code and tools used
    formatted_block = f"""<div class="tool-call-highlight">
<div class="tool-call-header">
<strong>Code Execution</strong>
</div>
<div class="tool-call-input">
```{language}
{code_content}
```
</div>"""

    # Add tools used section
    if detected_tool_modules:
        tools_list = format_detected_tools(detected_tool_modules)
        formatted_block += f"""
<div class="tools-used">
<strong>Tools Used:</strong> {tools_list}
</div>"""
    else:
        formatted_block += format_default_tool_name(language, tool_name)

    formatted_block += "</div>"
    return formatted_block


def format_detected_tools(detected_tool_modules: list) -> str:
    """Format detected tools with their modules for display.

    This function takes a list of (tool_name, module_name) tuples and formats
    them into a human-readable string for display in the tool call blocks.
    It handles special cases for common tools and formats module names appropriately.

    Args:
        detected_tool_modules: List of (tool_name, module_name) tuples

    Returns:
        Comma-separated string of formatted tool descriptions

    Example:
        >>> format_detected_tools([("analyze_data", "biomni.tool"), ("pandas", "pandas")])
        "biomni → analyze_data, pandas → pandas"
    """
    tool_descriptions = []
    for tool_name, module_name in detected_tool_modules:
        if tool_name == "python_repl":
            tool_descriptions.append("Python REPL")
        elif tool_name == "r_repl":
            tool_descriptions.append("R REPL")
        elif "bash" in tool_name.lower():
            tool_descriptions.append("Bash Script")
        else:
            # Extract the last part of the module name for display
            display_module = module_name.split(".")[-1] if "." in module_name else module_name
            tool_descriptions.append(f"{display_module} → {tool_name}")

    return ", ".join(sorted(tool_descriptions))


def format_default_tool_name(language: str, tool_name: str) -> str:
    """Format default tool name based on programming language.

    This function generates HTML for displaying the default tool name when
    no specific tools are detected in the code. It maps programming languages
    to their appropriate default tool names.

    Args:
        language: The programming language ("python", "r", "bash")
        tool_name: The detected tool name (used for bash CLI vs script distinction)

    Returns:
        HTML string containing a styled tools-used section

    Note:
        For bash, it distinguishes between CLI commands and bash scripts
        based on the tool_name parameter.
    """
    if language == "r":
        return """
<div class="tools-used">
<strong>Tools Used:</strong> R REPL
</div>"""
    elif language == "bash":
        if tool_name == "CLI Command":
            return """
<div class="tools-used">
<strong>Tools Used:</strong> CLI Command
</div>"""
        else:
            return """
<div class="tools-used">
<strong>Tools Used:</strong> Bash Script
</div>"""
    else:
        return """
<div class="tools-used">
<strong>Tools Used:</strong> Python REPL
</div>"""


def format_solution_tags_in_content(content: str) -> str:
    """Format solution tags in content by extracting text and formatting as solution blocks.

    This function processes content that contains <solution>...</solution> tags and
    converts them into styled HTML blocks that display solution content with appropriate
    formatting and CSS classes.

    Args:
        content: The content string that may contain <solution> tags

    Returns:
        Formatted content with solution tags converted to styled solution blocks

    Note:
        The solution blocks use the "title-text summary" CSS class for consistent
        styling with other content blocks in the markdown output.
    """
    import re

    # Pattern to match <solution>...</solution> blocks
    solution_pattern = r"<solution>(.*?)</solution>"

    def replace_solution_tag(match):
        solution_content = match.group(1).strip()
        # Format as regular text, not terminal
        return f"""<div class="title-text summary">
<div class="title-text-header">
<strong>Summary and Solution</strong>
</div>
<div class="title-text-content">
{solution_content}
</div>
</div>"""

    # Replace all solution tags with formatted solution blocks
    formatted_content = re.sub(solution_pattern, replace_solution_tag, content, flags=re.DOTALL)

    return formatted_content


def format_observation_as_terminal(content: str) -> str | None:
    """Format observation content with terminal-like styling.

    This function processes observation content from the agent's execution results
    and formats it as a styled terminal block. It handles both text and image content,
    with length limits to ensure the output fits within PDF page constraints.

    Args:
        content: The observation content string, potentially containing <observation> tags

    Returns:
        Formatted HTML content with terminal styling, or None if observation is
        empty, invalid, or contains only meaningless content

    Note:
        - Content is limited to 10,000 characters to fit within 2 A4 pages
        - Handles both text and base64-encoded images
        - Uses CSS classes for consistent styling with other content blocks
    """
    import re

    # Character limit for 2 A4 pages (approximately 10,000 characters)
    MAX_OBSERVATION_LENGTH = 10000

    # Remove the <observation> tags and extract the content
    observation_pattern = r"<observation>(.*?)</observation>"
    observation_match = re.search(observation_pattern, content, re.DOTALL)

    if observation_match:
        observation_content = observation_match.group(1).strip()
    else:
        # Fallback if no observation tags found - check if content is meaningful
        if not (content.strip() and content.strip() not in ["", "None", "null", "undefined"]):
            return None
        observation_content = content.strip()

    # Skip empty observations
    if not observation_content or observation_content in ["", "None", "null", "undefined"]:
        return None

    # Check if observation is too long for 2 pages
    if len(observation_content) > MAX_OBSERVATION_LENGTH:
        cropped_content = observation_content[:MAX_OBSERVATION_LENGTH]
        truncation_notice = f"\n\n[Output truncated - content was too long to display here ({len(observation_content)} characters total)]"
        observation_content = cropped_content + truncation_notice

    # Check if it contains plot data (base64 images)
    if "data:image/" in observation_content:
        content_html = process_observation_with_images(observation_content)
    else:
        # Regular text output - format as terminal output
        content_html = f"```terminal\n{observation_content}\n```"

    return f"""<div class="title-text observation">
<div class="title-text-header">
<strong>Observation</strong>
</div>
<div class="title-text-content">
{content_html}
</div>
</div>"""


def process_observation_with_images(observation_content: str) -> str:
    """Process observation content that contains both text and base64-encoded images.

    This function handles observation content that includes both text output and
    base64-encoded images (typically plots from data analysis). It separates the
    text and image content and formats them appropriately for markdown display.

    Args:
        observation_content: The observation content containing both text and images

    Returns:
        HTML string containing formatted text (as terminal blocks) and images
        (as markdown image tags)

    Note:
        The function uses "data:image/" as a delimiter to split content into
        text and image parts, then processes each part separately.
    """
    # Split content into text and image parts
    parts = observation_content.split("data:image/")
    text_parts = []
    image_parts = []

    for i, part in enumerate(parts):
        if i == 0:
            # First part is text only
            if part.strip():
                text_parts.append(part.strip())
        else:
            # Find the end of the base64 data
            end_markers = ["\n", "\r", " ", "\t", ">", "<", "]", ")", "}"]
            image_end = len(part)
            for marker in end_markers:
                marker_pos = part.find(marker)
                if marker_pos != -1 and marker_pos < image_end:
                    image_end = marker_pos

            # Extract image data
            image_data = "data:image/" + part[:image_end]
            image_parts.append(image_data)

            # Extract remaining text
            remaining_text = part[image_end:].strip()
            if remaining_text:
                text_parts.append(remaining_text)

    # Build the content
    content_html = ""
    if text_parts:
        # Add text content as terminal output
        text_content = "\n".join(text_parts)
        content_html += f"```terminal\n{text_content}\n```\n\n"

    if image_parts:
        # Add image content
        for image_data in image_parts:
            content_html += f"![Plot]({image_data})\n\n"

    return content_html


def remove_emojis_from_text(text: str) -> str:
    """Remove emojis from text for markdown/PDF output.

    This function removes common emojis used in the system prompt and configuration
    display from text content before it's converted to markdown or PDF. This ensures
    clean, professional output while preserving emojis in the console display.

    Args:
        text: The text content that may contain emojis

    Returns:
        Text content with emojis removed

    Note:
        The function targets specific emojis used in the Biomni system:
        - 🔧 for tools
        - 📊 for data
        - ⚙️ for software
        - 📋 for configuration
        - 🤖 for agent
    """
    import re

    # Remove common emojis used in the system prompt, this makes conversion simpler
    emoji_patterns = [
        r"🔧\s*",  # Tool emoji
        r"📊\s*",  # Data emoji
        r"⚙️\s*",  # Software emoji
        r"📋\s*",  # Config emoji
        r"🤖\s*",  # Agent emoji
    ]

    for pattern in emoji_patterns:
        text = re.sub(pattern, "", text)

    return text


def format_lists_in_text(text: str) -> str:
    """Format numbered lists and bullet points in text to proper markdown format.

    This function processes text content to identify and format various types of lists,
    including numbered lists with checkboxes, regular lists, and plan structures.
    It also handles preprocessing tasks like removing bold formatting from plan titles
    and removing emojis for clean PDF output.

    Args:
        text: The text content to process for list formatting

    Returns:
        Formatted text with properly structured lists and cleaned formatting

    Note:
        The function performs several preprocessing steps:
        - Removes bold formatting from plan titles
        - Removes emojis for PDF output
        - Identifies and formats checkbox lists
        - Processes regular text blocks
    """
    import re

    # Preprocess to remove bold formatting from plan titles
    # Remove **Plan:**, **Updated Plan:**, **Completed Plan:**, etc.
    text = re.sub(r"\*\*([Pp]lan|Updated [Pp]lan|Completed [Pp]lan|Final [Pp]lan):\*\*", r"\1:", text)
    # Also handle cases without colons
    text = re.sub(r"\*\*([Pp]lan|Updated [Pp]lan|Completed [Pp]lan|Final [Pp]lan)\*\*", r"\1", text)
    # Handle any other bold formatting patterns for plan titles
    text = re.sub(r"<strong>([Pp]lan|Updated [Pp]lan|Completed [Pp]lan|Final [Pp]lan):</strong>", r"\1:", text)
    text = re.sub(r"<strong>([Pp]lan|Updated [Pp]lan|Completed [Pp]lan|Final [Pp]lan)</strong>", r"\1", text)

    # Remove emojis from the text for markdown/PDF output
    text = remove_emojis_from_text(text)

    lines = text.split("\n")
    list_blocks = identify_list_blocks(lines)

    # Process each block
    result_blocks = []
    for block_text, is_checkbox_list in list_blocks:
        if is_checkbox_list:
            result_blocks.append(format_single_list(block_text))
        else:
            result_blocks.append(block_text)

    return "\n".join(result_blocks)


def identify_list_blocks(lines: list) -> list[tuple[str, bool]]:
    """Identify blocks of text that contain lists.

    This function analyzes a list of text lines to identify contiguous blocks
    that contain numbered lists with checkboxes. It groups lines into blocks
    and marks whether each block contains a checkbox list or regular text.

    Args:
        lines: List of text lines to analyze

    Returns:
        List of tuples containing (block_text, is_checkbox_list) where:
        - block_text: The text content of the block
        - is_checkbox_list: True if the block contains numbered items with checkboxes

    Note:
        The function looks for patterns like "1. [ ]", "2. [✓]", "3. [✗]" to
        identify checkbox sequences and groups them into separate blocks.
    """
    import re

    list_blocks = []
    current_block = []
    in_checkbox_sequence = False

    for line in lines:
        line_stripped = line.strip()

        # Check if this line starts a numbered item with checkbox
        if re.match(r"^\d+\.\s*\[[ ✓✗]\]", line_stripped):
            if not in_checkbox_sequence:
                # Start of a new checkbox sequence
                if current_block:
                    list_blocks.append(("\n".join(current_block), False))
                current_block = [line]
                in_checkbox_sequence = True
            else:
                # Continue the sequence
                current_block.append(line)
        else:
            if in_checkbox_sequence:
                # End of checkbox sequence
                if current_block:
                    list_blocks.append(("\n".join(current_block), True))
                current_block = []
                in_checkbox_sequence = False
            current_block.append(line)

    # Handle the last block
    if current_block:
        if in_checkbox_sequence:
            list_blocks.append(("\n".join(current_block), True))
        else:
            list_blocks.append(("\n".join(current_block), False))

    return list_blocks


def format_single_list(text: str) -> str:
    """Format a single list block with checkboxes and plan titles.

    This function processes a text block that may contain numbered lists with
    checkboxes and plan titles. It converts checkbox symbols to HTML list items
    and wraps the content in a styled container with appropriate CSS classes.

    Args:
        text: The text block to format, potentially containing numbered lists

    Returns:
        HTML string containing either a formatted list with plan title or
        regular text if no list items are found

    Note:
        The function recognizes plan titles like "Plan", "Updated Plan", "Completed Plan"
        and converts checkbox symbols (✓, ✗) to HTML format ([x], [ ]).
    """
    import re

    lines = text.split("\n")
    list_items = []
    has_list_items = False
    plan_title = "Plan"  # Default title

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for plan title patterns
        if re.match(r"^(Plan|Updated Plan|Completed Plan)$", line, re.IGNORECASE):
            plan_title = line
            continue

        # Check for numbered lists with checkboxes (1. [ ] or 1. [✓] or 1. [✗])
        if re.match(r"^\d+\.\s*\[[ ✓✗]\]", line):
            has_list_items = True
            # Extract the content after the checkbox
            content = re.sub(r"^\d+\.\s*\[[ ✓✗]\]\s*", "", line)

            # Replace checkbox symbols with text format
            if "[✓]" in line:
                list_items.append(f"<li><strong>[x]</strong> {content}</li>")
            elif "[✗]" in line:
                list_items.append(f"<li><strong>[ ]</strong> {content}</li>")
            else:
                list_items.append(f"<li><strong>[ ]</strong> {content}</li>")
        else:
            # Regular text - add as is (don't convert to list items)
            list_items.append(line)

    if has_list_items and list_items:
        # This is a list - return with container div and styled title
        return f"""<div class="title-text plan">
<div class="title-text-header">
<span class="plan-title">{plan_title}</span>
</div>
<div class="title-text-content">
<ul>
{chr(10).join(list_items)}
</ul>
</div>
</div>"""
    else:
        # Regular text
        return "\n".join(list_items)


def convert_markdown_to_pdf(markdown_path: str, pdf_path: str) -> None:
    """Convert markdown file to PDF using weasyprint or fallback libraries.

    This function converts a markdown file to PDF format using multiple fallback
    strategies. It prioritizes weasyprint for better layout control, then falls back
    to markdown2pdf and finally pandoc if the preferred libraries are not available.

    Args:
        markdown_path: Path to the input markdown file
        pdf_path: Path where the output PDF file should be saved

    Raises:
        ImportError: If no PDF conversion library is available
        Exception: If PDF conversion fails for any other reason

    Note:
        The function uses minimal markdown extensions for better performance
        and applies custom CSS styling for consistent formatting.
    """
    try:
        # Try weasyprint first (better for complex layouts)
        from weasyprint import HTML
        from weasyprint.text.fonts import FontConfiguration

        # Read markdown content
        with open(markdown_path, encoding="utf-8") as f:
            markdown_content = f.read()

        # Convert markdown to HTML with minimal extensions for better performance
        import markdown

        # Use minimal extensions to improve performance
        html_content = markdown.markdown(
            markdown_content,
            extensions=["fenced_code"],  # Removed codehilite for better performance
        )

        # Add CSS styling
        css_content = get_pdf_css_content()

        # Create HTML document
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Biomni Conversation History</title>
            <style>{css_content}</style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Convert to PDF with performance optimizations
        font_config = FontConfiguration()
        html_obj = HTML(string=html_doc)
        html_obj.write_pdf(pdf_path, font_config=font_config, optimize_images=True)

    except ImportError:
        # Fallback to markdown2pdf if weasyprint is not available
        try:
            from markdown2pdf import markdown2pdf

            markdown2pdf(markdown_path, pdf_path)
        except ImportError:
            # Final fallback - try using pandoc if available
            import subprocess

            try:
                subprocess.run(["pandoc", markdown_path, "-o", pdf_path], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise ImportError(
                    "No PDF conversion library available. Please install weasyprint, markdown2pdf, or pandoc."
                ) from e
    except Exception as e:
        raise Exception(f"PDF conversion failed: {e}") from e


def get_pdf_css_content() -> str:
    """Get the CSS content for PDF generation.

    This function returns a comprehensive CSS stylesheet designed specifically
    for PDF generation from markdown content. It includes styling for all
    HTML elements that may appear in the converted markdown, with optimized
    typography, spacing, and layout for print media.

    Returns:
        CSS string containing all styles needed for PDF generation

    Note:
        The CSS includes styles for:
        - Typography and font families
        - Headings and text formatting
        - Code blocks and syntax highlighting
        - Tables and lists
        - Custom classes for tool calls, observations, and plans
        - Print-optimized spacing and layout
    """
    return """
    body {
        /* Previously: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif, 'Noto Color Emoji', 'Apple Color Emoji', 'Segoe UI Emoji', 'Twemoji', 'EmojiOne Color' */
        font-family: sans-serif;
        font-size: 9pt;
        line-height: 1.4;
        max-width: 800px;
        margin: 0 auto;
        padding: 15px;
        color: #333;
    }
    h1, h2, h3, h4, h5, h6 {
        /* Previously: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif, 'Noto Color Emoji', 'Apple Color Emoji', 'Segoe UI Emoji', 'Twemoji', 'EmojiOne Color' */
        font-family: sans-serif;
        color: #2c3e50;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    h1 {
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
        font-size: 16pt;
    }
    h2 {
        border-bottom: 1px solid #bdc3c7;
        padding-bottom: 3px;
        font-size: 14pt;
    }
    h3 {
        font-size: 12pt;
    }
    h4 {
        font-size: 10pt;
        margin-top: 0.8em;
        margin-bottom: 0.3em;
    }
    h5, h6 {
        font-size: 9pt;
        margin-top: 0.6em;
        margin-bottom: 0.2em;
    }
    code {
        background-color: #f8f9fa;
        padding: 1px 3px;
        border-radius: 2px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 8pt;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    pre {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 3px;
        overflow-x: auto;
        border-left: 3px solid #3498db;
        white-space: pre-wrap;
        word-wrap: break-word;
        font-size: 8pt;
        margin: 0.5em 0;
    }
    pre code {
        background-color: transparent;
        padding: 0;
        border-radius: 0;
        font-size: 8pt;
    }
    /* Code header styling */
    strong {
        font-size: 9pt;
        font-weight: normal;
        color: #6c757d;
        font-style: italic;
    }
    blockquote {
        border-left: 3px solid #bdc3c7;
        margin: 0.5em 0;
        padding-left: 15px;
        color: #7f8c8d;
        font-size: 8pt;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 0.5em 0;
        font-size: 8pt;
    }
    th, td {
        border: 1px solid #bdc3c7;
        padding: 4px 8px;
        text-align: left;
    }
    th {
        background-color: #ecf0f1;
        font-weight: bold;
    }
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 10px auto;
        border: 1px solid #ddd;
        border-radius: 3px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    p {
        /* Previously: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif, 'Noto Color Emoji', 'Apple Color Emoji', 'Segoe UI Emoji', 'Twemoji', 'EmojiOne Color' */
        font-family: sans-serif;
        margin: 0.3em 0;
    }
    /* Tool call highlighting - matching observation and code formatting */
    .tool-call-highlight {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 3px;
        padding: 0;
        margin: 10px 0;
        overflow: hidden;
    }
    .tool-call-header {
        background-color: #e9ecef;
        color: #495057;
        padding: 8px 12px;
        margin: 0;
        font-weight: normal;
        font-size: 9pt;
        font-style: italic;
        border-bottom: 1px solid #dee2e6;
    }
    .tool-call-input {
        background-color: #f8f9fa;
        border: none;
        border-radius: 0;
        padding: 10px 12px;
        margin: 0;
        color: #333;
        font-size: 8pt;
        line-height: 1.4;
    }
    .tool-call-input strong {
        color: #495057;
        font-weight: normal;
        font-size: 8pt;
        font-style: italic;
    }
    .tool-call-input pre {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 3px;
        padding: 10px;
        margin: 0;
        font-size: 8pt;
        line-height: 1.4;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .tool-call-input code {
        background-color: transparent;
        padding: 0;
        border-radius: 0;
        font-size: 8pt;
        color: #2c3e50;
    }
    .tools-used {
        background-color: #f8f9fa;
        border-top: 1px solid #dee2e6;
        padding: 8px 12px;
        margin: 0;
        font-size: 8pt;
        color: #6c757d;
    }
    .tools-used strong {
        color: #6c757d;
        font-weight: normal;
        font-size: 8pt;
        font-style: italic;
    }
    /* Title-text styling - unified for observations, plans, and solutions */
    .title-text {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 3px;
        padding: 0;
        margin: 10px 0;
        overflow: hidden;
    }
    .title-text-header {
        background-color: #e9ecef;
        color: #495057;
        padding: 8px 12px;
        margin: 0;
        font-weight: normal;
        font-size: 9pt;
        font-style: italic;
        border-bottom: 1px solid #dee2e6;
    }
    .title-text-header strong {
        color: #495057;
        font-weight: normal;
        font-size: 9pt;
        font-style: italic;
    }
    .title-text-content {
        background-color: #f8f9fa;
        border: none;
        border-radius: 0;
        padding: 10px 12px;
        margin: 0;
        color: #333;
        font-size: 8pt;
        line-height: 1.4;
    }
    /* Plan-specific styling - soft blue pastel */
    .title-text.plan {
        background-color: #e3f2fd;
        border-color: #bbdefb;
    }
    .title-text.plan .title-text-header {
        background-color: #bbdefb;
        color: #1976d2;
    }
    .title-text.plan .title-text-content {
        background-color: #e3f2fd;
    }
    .plan-title {
        font-style: italic;
        font-weight: normal;
        color: #1565c0;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .plan-title strong {
        font-weight: normal;
    }
    /* Code execution-specific styling - matching title-text styling */
    .tool-call-highlight {
        background-color: #f8f9fa;
        border-color: #e9ecef;
    }
    .tool-call-header {
        background-color: #e9ecef;
        color: #495057;
    }
    .tool-call-input {
        background-color: #f8f9fa;
        color: #333;
    }
    /* Observation-specific styling - soft purple pastel */
    .title-text.observation {
        background-color: #f3e5f5;
        border-color: #e1bee7;
    }
    .title-text.observation .title-text-header {
        background-color: #e1bee7;
        color: #7b1fa2;
    }
    .title-text.observation .title-text-content {
        background-color: #f3e5f5;
    }
    /* Summary and solution-specific styling - soft orange pastel, no overlay */
    .title-text.summary {
        background-color: #fff3e0;
        border-color: #ffcc02;
    }
    .title-text.summary .title-text-header {
        background-color: #ffcc02;
        color: #f57c00;
    }
    .title-text.summary .title-text-content {
        background-color: #fff3e0;
    }
    .title-text-content ul {
        background-color: transparent;
        border: none;
        border-radius: 0;
        padding: 0;
        margin: 0;
        color: #333;
        font-size: 8pt;
        line-height: 1.4;
    }
    .title-text-content li {
        margin: 3px 0;
        color: #333;
    }
    .title-text-content li strong {
        color: #495057;
        font-weight: normal;
        font-size: 8pt;
        font-style: italic;
    }
    .title-text-content li code {
        background-color: #e9ecef;
        color: #333;
        padding: 1px 3px;
        border-radius: 2px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 7pt;
    }
    .title-text-content pre {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 3px;
        padding: 10px;
        margin: 0;
        font-size: 8pt;
        line-height: 1.4;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .title-text-content code {
        background-color: transparent;
        padding: 0;
        border-radius: 0;
        font-size: 8pt;
        color: #2c3e50;
    }
    /* Parsing error display styling */
    .parsing-error-box {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 8px 0;
        font-size: 9pt;
        color: #c62828;
        box-shadow: 0 2px 4px rgba(244, 67, 54, 0.1);
    }
    .parsing-error-header {
        font-weight: bold;
        margin-bottom: 4px;
        color: #d32f2f;
    }
    .parsing-error-content {
        font-family: 'Courier New', monospace;
        background-color: #ffcdd2;
        padding: 4px 6px;
        border-radius: 2px;
        margin-top: 4px;
        font-size: 8pt;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    """
