import os
import ast
from docstring_parser import parse

def get_description(filepath):
    """
    Extracts all functions and their docstrings from the given file path.

    Args:
        filepath (str): Path to the Python file.

    Returns:
        dict: A dictionary with function names as keys and their docstrings as values.
              If a function has no docstring, the value is None.
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return {}

    with open(filepath, "r", encoding="utf-8") as f:
        source_code = f.read()

    tree = ast.parse(source_code)
    description = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            doc = parse(docstring)
            function_info = {
                "long_description": doc.long_description,
                "description": doc.short_description,
                "return": doc.returns.description if doc.returns else None,
                "name": node.name,
                "required_parameters": [],
                "optional_parameters": [],
            }
            for param in doc.params:
                value = {
                    "default": param.default,
                    "description": param.description,
                    "name": param.arg_name,
                    "type": param.type_name,
                }
                if param.default is None or param.default == "":
                    function_info["required_parameters"].append(value)
                else:
                    function_info["optional_parameters"].append(value)
            description.append(function_info)

    return description


description = get_description(
    os.path.dirname(os.path.abspath(__file__)) + "/../statistics.py"
)