"""
Resource filtering utilities for restricting tools, data lake, and libraries
based on YAML configuration files.
"""

import os
import yaml
import importlib
import inspect
from typing import Dict, List, Any, Optional, Set
from pathlib import Path


def load_resource_filter_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load resource filter configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file. If None, looks for
                     'resource_filter.yaml' in the current directory and parent directories,
                     including Biomni_HITS directory.

    Returns:
        Dictionary with 'tools', 'data_lake', and 'libraries' keys.
        Returns empty dict if file not found or invalid.
    """
    if config_path is None:
        # Try to find resource_filter.yaml or resource.yaml in current directory and parent directories
        current_dir = Path.cwd()
        search_paths = [
            current_dir / "resource_filter.yaml",
            current_dir / "resource.yaml",
            current_dir / "Biomni_HITS" / "resource_filter.yaml",
            current_dir / "Biomni_HITS" / "resource.yaml",
            current_dir / "chainlit" / "resource.yaml",
        ]

        # Add parent directories
        for parent in list(current_dir.parents)[:5]:  # Check up to 5 levels up
            search_paths.append(parent / "resource_filter.yaml")
            search_paths.append(parent / "resource.yaml")
            search_paths.append(parent / "Biomni_HITS" / "resource_filter.yaml")
            search_paths.append(parent / "Biomni_HITS" / "resource.yaml")
            search_paths.append(parent / "chainlit" / "resource.yaml")

        # Also try relative to this file's location
        try:
            import biomni

            biomni_path = Path(biomni.__file__).parent.parent
            search_paths.append(biomni_path / "resource_filter.yaml")
            search_paths.append(biomni_path / "resource.yaml")
            search_paths.append(biomni_path / "chainlit" / "resource.yaml")
        except:
            pass

        for potential_path in search_paths:
            if potential_path.exists():
                config_path = str(potential_path)
                break

    if config_path is None or not os.path.exists(config_path):
        print(f"No resource filter configuration found. Using all resources.")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        # Extract allowed_resources section
        allowed_resources = config.get("allowed_resources", {})

        print(f"Loaded resource filter from: {config_path}")
        return allowed_resources
    except Exception as e:
        print(f"Error loading resource filter config: {e}")
        return {}


def _parse_tool_spec(spec: Any) -> Dict[str, Any]:
    """
    Parse a tool specification from YAML.

    Supports:
    - String: tool name (e.g., "query_uniprot")
    - Dict with "file" key: Python file path (e.g., {"file": "biomni/tool/genetics.py"})
    - Dict with "module" key: module name (e.g., {"module": "biomni.tool.genetics"})

    Returns:
        Dict with 'type' ('name', 'file', 'module') and 'value' keys.
    """
    if isinstance(spec, str):
        return {"type": "name", "value": spec}
    elif isinstance(spec, dict):
        if "file" in spec:
            return {"type": "file", "value": spec["file"]}
        elif "module" in spec:
            return {"type": "module", "value": spec["module"]}
        else:
            # If it's a dict but no recognized key, treat as tool name
            # This shouldn't happen in normal usage, but handle gracefully
            return {"type": "name", "value": str(spec)}
    else:
        # Fallback: convert to string
        return {"type": "name", "value": str(spec)}


def _get_tools_from_module(module_name: str) -> Set[str]:
    """
    Get all tool names from a module by inspecting its description attribute.

    Args:
        module_name: Full module name (e.g., "biomni.tool.tool_description.genetics")

    Returns:
        Set of tool names from that module.
    """
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "description") and isinstance(module.description, list):
            return {
                tool.get("name")
                for tool in module.description
                if isinstance(tool, dict) and "name" in tool
            }
        return set()
    except Exception as e:
        print(f"Warning: Could not import module {module_name}: {e}")
        return set()


def _get_tools_from_file(file_path: str, base_path: Optional[str] = None) -> Set[str]:
    """
    Get all function names from a Python file.

    Args:
        file_path: Path to Python file (can be relative or absolute)
        base_path: Base path for resolving relative paths

    Returns:
        Set of function names from that file.
    """
    try:
        # Resolve path
        if not os.path.isabs(file_path):
            if base_path:
                resolved_path = os.path.join(base_path, file_path)
            else:
                # Try to resolve relative to current working directory
                resolved_path = os.path.abspath(file_path)
        else:
            resolved_path = file_path

        if not os.path.exists(resolved_path):
            print(f"Warning: File not found: {resolved_path}")
            return set()

        # Read and parse the file
        with open(resolved_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Parse AST to find function definitions
        import ast

        tree = ast.parse(source)
        functions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.add(node.name)

        return functions
    except Exception as e:
        print(f"Warning: Could not parse file {file_path}: {e}")
        return set()


def filter_module2api(
    module2api: Dict[str, List[Dict]], allowed_tools: Optional[List[Any]] = None
) -> Dict[str, List[Dict]]:
    """
    Filter module2api dictionary based on allowed tools specification.

    Args:
        module2api: Dictionary mapping module names to lists of tool descriptions.
        allowed_tools: List of tool specifications. Each can be:
            - String tool name
            - Dict with "file" key pointing to Python file
            - Dict with "module" key pointing to module name
            If None or empty, returns original module2api (no filtering).

    Returns:
        Filtered module2api dictionary.
    """
    if not allowed_tools:
        return module2api

    # Build set of allowed tool names
    allowed_tool_names: Set[str] = set()

    # Also track which modules/files are fully allowed
    allowed_modules: Set[str] = set()

    for spec in allowed_tools:
        parsed = _parse_tool_spec(spec)

        if parsed["type"] == "name":
            allowed_tool_names.add(parsed["value"])
        elif parsed["type"] == "module":
            module_name = parsed["value"]
            # Check if it's a tool_description module
            if not module_name.startswith("biomni.tool.tool_description."):
                # Try to convert module name to tool_description format
                if module_name.startswith("biomni.tool."):
                    # Convert "biomni.tool.genetics" to "biomni.tool.tool_description.genetics"
                    tool_name = module_name.replace("biomni.tool.", "")
                    tool_desc_module = f"biomni.tool.tool_description.{tool_name}"
                    module_name = tool_desc_module

            # Get all tools from this module
            tools = _get_tools_from_module(module_name)
            allowed_tool_names.update(tools)
            # Also allow matching module prefixes in module2api
            # module2api uses keys like "biomni.tool.genetics" (without tool_description)
            if module_name.startswith("biomni.tool.tool_description."):
                # Extract the base module name (e.g., "genetics")
                base_module = module_name.replace("biomni.tool.tool_description.", "")
                # Add both tool_description and regular module formats
                allowed_modules.add(f"biomni.tool.tool_description.{base_module}")
                allowed_modules.add(f"biomni.tool.{base_module}")
            else:
                allowed_modules.add(module_name)
        elif parsed["type"] == "file":
            file_path = parsed["value"]
            # Get all functions from the file
            functions = _get_tools_from_file(file_path)
            allowed_tool_names.update(functions)

    # Filter module2api
    filtered_module2api = {}
    for module_name, api_list in module2api.items():
        filtered_apis = []

        # Check if entire module is allowed
        module_allowed = False
        for allowed_mod in allowed_modules:
            # Check exact match or if module_name matches allowed module pattern
            if module_name == allowed_mod:
                module_allowed = True
                break
            # Check if module_name is "biomni.tool.X" and allowed_mod is "biomni.tool.tool_description.X"
            if allowed_mod.startswith("biomni.tool.tool_description."):
                base_name = allowed_mod.replace("biomni.tool.tool_description.", "")
                if module_name == f"biomni.tool.{base_name}":
                    module_allowed = True
                    break
            # Check reverse: if allowed_mod is "biomni.tool.X" and module_name is "biomni.tool.tool_description.X"
            if module_name.startswith("biomni.tool.tool_description."):
                base_name = module_name.replace("biomni.tool.tool_description.", "")
                if allowed_mod == f"biomni.tool.{base_name}":
                    module_allowed = True
                    break

        for api in api_list:
            tool_name = api.get("name", "")
            # Check if tool is allowed by name or if entire module is allowed
            if tool_name in allowed_tool_names or module_allowed:
                filtered_apis.append(api)

        if filtered_apis:
            filtered_module2api[module_name] = filtered_apis

    print(
        f"Filtered tools: {sum(len(apis) for apis in filtered_module2api.values())} tools from {len(filtered_module2api)} modules"
    )
    return filtered_module2api


def filter_data_lake_dict(
    data_lake_dict: Dict[str, str], allowed_items: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Filter data_lake_dict based on allowed items.

    Args:
        data_lake_dict: Dictionary mapping data lake item names to descriptions.
        allowed_items: List of allowed data lake item names. If None or empty, returns original dict.

    Returns:
        Filtered data_lake_dict.
    """
    if not allowed_items:
        return data_lake_dict

    allowed_set = set(allowed_items)
    filtered = {k: v for k, v in data_lake_dict.items() if k in allowed_set}

    print(
        f"Filtered data lake: {len(filtered)} items (from {len(data_lake_dict)} total)"
    )
    return filtered


def filter_library_content_dict(
    library_content_dict: Dict[str, str], allowed_libraries: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Filter library_content_dict based on allowed libraries.

    Args:
        library_content_dict: Dictionary mapping library names to descriptions.
        allowed_libraries: List of allowed library names. If None or empty, returns original dict.

    Returns:
        Filtered library_content_dict.
    """
    if not allowed_libraries:
        return library_content_dict

    allowed_set = set(allowed_libraries)
    filtered = {k: v for k, v in library_content_dict.items() if k in allowed_set}

    print(
        f"Filtered libraries: {len(filtered)} libraries (from {len(library_content_dict)} total)"
    )
    return filtered


def apply_resource_filters(
    module2api: Dict[str, List[Dict]],
    data_lake_dict: Dict[str, str],
    library_content_dict: Dict[str, str],
    config_path: Optional[str] = None,
) -> tuple[Dict[str, List[Dict]], Dict[str, str], Dict[str, str]]:
    """
    Apply all resource filters based on YAML configuration.

    Args:
        module2api: Dictionary mapping module names to lists of tool descriptions.
        data_lake_dict: Dictionary mapping data lake item names to descriptions.
        library_content_dict: Dictionary mapping library names to descriptions.
        config_path: Path to YAML configuration file. If None, auto-detects.

    Returns:
        Tuple of (filtered_module2api, filtered_data_lake_dict, filtered_library_content_dict).
    """
    config = load_resource_filter_config(config_path)

    # Filter tools
    allowed_tools = config.get("tools", None)
    filtered_module2api = filter_module2api(module2api, allowed_tools)

    # Filter data lake
    allowed_data_lake = config.get("data_lake", None)
    filtered_data_lake_dict = filter_data_lake_dict(data_lake_dict, allowed_data_lake)

    # Filter libraries
    allowed_libraries = config.get("libraries", None)
    filtered_library_content_dict = filter_library_content_dict(
        library_content_dict, allowed_libraries
    )

    return filtered_module2api, filtered_data_lake_dict, filtered_library_content_dict
