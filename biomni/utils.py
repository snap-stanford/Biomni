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
        "bioengineering",
        "biophysics",
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
