from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from docstring_parser import DocstringParam, parse


MANUAL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "analyze_pixel_distribution": {
        "description": "Summarise pixel intensity characteristics for gel or blot images.",
        "long_description": (
            "Loads a grayscale image, computes percentiles, histogram-derived brightness "
            "bucket counts, and aggregates statistics such as min/max, mean, and standard "
            "deviation of pixel intensities. Useful for quickly assessing exposure levels "
            "or spotting saturation across the image."
        ),
        "return": (
            "Dictionary containing: 'shape' (image dimensions), 'min_intensity', 'max_intensity', "
            "'mean_intensity', 'std_intensity' (individual stats), 'intensity_stats' (dict with "
            "keys 'min', 'max', 'mean', 'std'), 'percentiles_label', 'percentiles_values', and "
            "'pixel_brightness_distribution' (list of formatted brightness range strings)."
        ),
        "parameters": {
            "image_path": {
                "description": (
                    "Absolute or relative path to the grayscale image. Automatically "
                    "appends `.png` when no extension is provided."
                ),
                "type": "str",
            }
        },
    },
    "find_roi_from_image": {
        "description": "Detect regions of interest (ROI) in gel or blot images.",
        "long_description": (
            "Automatically detects blot bands or colonies by combining blob detection, "
            "adaptive thresholding, and contour analysis. Annotated ROIs and their indices "
            "are saved as an image that mirrors the input path. A binary mask used for "
            "detection is generated with the provided intensity thresholds and saved next to "
            "the source image."
        ),
        "return": (
            "Tuple containing the absolute path to the annotated image and a list "
            "of ROI coordinates in `(x, y, width, height)` format."
        ),
        "parameters": {
            "image_path": {
                "description": "Path to the grayscale image to analyze for ROI detection.",
                "type": "str",
            },
            "lower_threshold": {
                "description": "Lower bound of the intensity window used to build the binary mask.",
                "type": "int",
            },
            "upper_threshold": {
                "description": "Upper bound of the intensity window used to build the binary mask.",
                "type": "int",
            },
        },
    },
    "quantify_bands": {
        "description": "Compute band intensities for provided western blot or DNA electrophoresis image and detected ROIs.",
        "long_description": (
            "Loads the target grayscale image and, for each ROI, "
            "measures the average pixel intensity, samples the surrounding background "
            "according to the chosen strategy, and reports the positive signal after "
            "background subtraction."
        ),
        "return": (
            "List of floating-point intensities aligned with the input ROIs. Invalid ROIs "
            "yield NaN and empty/background-free ROIs resolve to zero."
        ),
        "parameters": {
            "image_path": {
                "description": "Absolute or relative path to the grayscale image file.",
                "type": "str",
            },
            "rois": {
                "description": (
                    "Sequence of ROI tuples in `(x, y, width, height)` format defining band locations."
                ),
                "type": "Sequence[Sequence[float]]",
            },
            "background_width": {
                "description": (
                    "Number of pixels to expand around each ROI when gathering background samples."
                ),
                "type": "int",
                "default": 3,
            },
            "back_pos": {
                "description": (
                    "Background sampling strategy. Choose among `all`, `top/bottom`, or `sides`."
                ),
                "type": "str",
                "default": "all",
            },
            "back_type": {
                "description": (
                    "Aggregation method for background pixels. Either `mean` or `median`."
                ),
                "type": "str",
                "default": "median",
            },
        },
    },
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


MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "bio_image_processing.py"
).resolve()
description = get_description(str(MODULE_PATH))

