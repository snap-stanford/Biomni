from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from docstring_parser import DocstringParam, parse


MANUAL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "find_roi_from_image": {
        "description": "Detect regions of interest (ROI) in gel or blot images.",
        "long_description": (
            "Automatically detects blot bands or colonies by combining blob detection, "
            "adaptive thresholding, and contour analysis. Annotated ROIs and their indices "
            "are saved as an image that mirrors the input path."
        ),
        "return": "Absolute path to the annotated image highlighting detected ROIs.",
        "parameters": {
            "image_path": {
                "description": "Path to the grayscale image to analyse for ROI detection.",
                "type": "str",
            }
        },
    }
}


def _safe_literal_eval(node: ast.expr | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node) if hasattr(ast, "unparse") else None


def _annotation_to_str(annotation: ast.expr | None) -> str | None:
    if annotation is None:
        return None
    return ast.unparse(annotation) if hasattr(ast, "unparse") else None


def _iter_positional_args(
    node: ast.FunctionDef,
) -> Iterable[Tuple[ast.arg, ast.expr | None]]:
    pos_args: List[ast.arg] = []
    pos_args.extend(getattr(node.args, "posonlyargs", []))
    pos_args.extend(node.args.args)

    defaults: List[ast.expr | None] = [None] * (
        len(pos_args) - len(node.args.defaults)
    ) + list(node.args.defaults)
    return zip(pos_args, defaults)


def _iter_kwonly_args(
    node: ast.FunctionDef,
) -> Iterable[Tuple[ast.arg, ast.expr | None]]:
    return zip(node.args.kwonlyargs, node.args.kw_defaults)


def _merge_manual_parameter_override(
    parameter: Dict[str, Any], override: Dict[str, Dict[str, Any]]
) -> None:
    manual = override.get(parameter["name"])
    if not manual:
        return
    for key, value in manual.items():
        if value is not None:
            parameter[key] = value


def _build_parameter_entry(
    arg: ast.arg,
    default_node: ast.expr | None,
    doc_param: DocstringParam | None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "name": arg.arg,
        "type": None,
        "description": None,
        "default": None,
    }

    annotation = _annotation_to_str(arg.annotation)
    default_value = _safe_literal_eval(default_node)

    if doc_param:
        entry["description"] = doc_param.description or None
        entry["type"] = doc_param.type_name or annotation
        if doc_param.default not in ("", None):
            entry["default"] = doc_param.default
    if entry["type"] is None:
        entry["type"] = annotation
    if entry["default"] is None:
        entry["default"] = default_value

    return entry


def _classify_parameter(entry: Dict[str, Any]) -> str:
    default = entry.get("default")
    if default in (None, "None", ""):
        return "required_parameters"
    return "optional_parameters"


def get_description(filepath: str) -> List[Dict[str, Any]]:
    """
    Extract structured tool metadata from public functions in the target file.

    Args:
        filepath (str): Absolute path to the module defining tool functions.

    Returns:
        list[dict]: Each dictionary contains ``name``, ``description``,
            ``long_description``, ``return``, ``required_parameters``, and
            ``optional_parameters`` fields derived from function docstrings.
    """
    module_path = Path(filepath)
    if not module_path.exists():
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다.")
        return []

    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    description: List[Dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue

        docstring = ast.get_docstring(node)
        doc = parse(docstring) if docstring else None
        doc_params = {param.arg_name: param for param in doc.params} if doc else {}

        function_info: Dict[str, Any] = {
            "name": node.name,
            "description": doc.short_description if doc else None,
            "long_description": doc.long_description if doc else None,
            "return": doc.returns.description if doc and doc.returns else None,
            "required_parameters": [],
            "optional_parameters": [],
        }

        manual_override = MANUAL_OVERRIDES.get(node.name, {})
        for key in ("description", "long_description", "return"):
            if not function_info[key]:
                function_info[key] = manual_override.get(key, function_info[key])

        parameter_entries: List[Dict[str, Any]] = []

        for arg, default in _iter_positional_args(node):
            if arg.arg == "self":
                continue
            doc_param = doc_params.pop(arg.arg, None)
            parameter_entries.append(_build_parameter_entry(arg, default, doc_param))

        for kw_arg, default in _iter_kwonly_args(node):
            doc_param = doc_params.pop(kw_arg.arg, None)
            parameter_entries.append(
                _build_parameter_entry(kw_arg, default, doc_param)
            )

        if manual_override:
            param_overrides = manual_override.get("parameters", {})
            for entry in parameter_entries:
                _merge_manual_parameter_override(entry, param_overrides)

        for entry in parameter_entries:
            bucket = _classify_parameter(entry)
            function_info[bucket].append(entry)

        description.append(function_info)

    return description


MODULE_PATH = (Path(__file__).resolve().parent.parent / "bio.py").resolve()
description = get_description(str(MODULE_PATH))

