import inspect
import ast
import os
from docstring_parser import parse


def get_description(filepath):
    """
    주어진 파일 경로에서 모든 함수와 해당 docstring을 추출합니다.

    Args:
        filepath (str): 파이썬 파일의 경로.

    Returns:
        dict: 함수 이름을 키로 하고 docstring을 값으로 하는 딕셔너리.
              docstring이 없는 경우 값은 None입니다.
    """
    if not os.path.exists(filepath):
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다.")
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
    os.path.dirname(os.path.abspath(__file__)) + "/../proteomics.py"
)
