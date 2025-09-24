import ast
import asyncio
import importlib
import io
import json
import os
import re
import tempfile
import time
import traceback
import urllib.request
import zipfile
from typing import Any

from biomni.llm import get_llm

# ------------------------------------------------------------
# Dynamic PyLabRobot documentation/content loader
# ------------------------------------------------------------

_MAX_DOC_CHARS = int("50000")


def _collect_from_installed_package(section: str) -> list[str]:
    """Collect short, up-to-date metadata from the installed pylabrobot.

    This does not rely on docs being packaged; it introspects modules and includes
    small docstrings and lists of available resources.
    """
    docs: list[str] = []
    try:
        pr = importlib.import_module("pylabrobot.resources")
        names = [n for n in dir(pr) if not n.startswith("_")]
        important = [
            n for n in names if any(k in n.lower() for k in ["tip", "rack", "carrier", "plate", "hamilton", "iswap"])
        ]
        lines = [
            "Installed PyLabRobot resources detected:",
            ", ".join(sorted(important)[:400]),
        ]
        # Removed explicit resource name checks per request

        docs.append("\n".join(lines))
    except Exception:
        pass
    return docs


def _load_pylabrobot_tutorial_content(section: str) -> str:
    """Load PLR tutorial/docs text from multiple sources with graceful fallback.

    Precedence:
      1) Docs from installed pylabrobot package (docs/user_guide/...)
      2) Introspect installed package (pylabrobot)
    """
    docs: list[str] = []

    # 1) Fetch from GitHub repo zip (pinned commit by default)
    repo = "PyLabRobot/pylabrobot"
    ref = "106aef9c8699ceb826d8c9c894eba304a082f24d"

    gh_docs = _collect_docs_from_github_zip(repo=repo, ref=ref, section=section)

    if gh_docs and isinstance(gh_docs[0], tuple):
        if section == "liquid":
            formatted = _format_liquid_user_guide(gh_docs)  # returns single string
            if formatted:
                docs.append(formatted)
        else:
            # Concatenate texts in stable order
            docs.append("\n\n".join(text for _, text in gh_docs if text))
    else:
        docs.extend(gh_docs)

    # 2) Installed package introspection
    docs.extend(_collect_from_installed_package(section))

    if not docs:
        return ""

    text = "\n\n".join([d for d in docs if d])
    if len(text) > _MAX_DOC_CHARS:
        text = text[:_MAX_DOC_CHARS]
    return text


def _collect_docs_from_github_zip(repo: str, ref: str, section: str) -> list[tuple[str, str]] | list[str]:
    """Download GitHub repo zip and extract user_guide docs for the section.

    Uses:
      - docs/user_guide/00_liquid-handling for section == "liquid"
      - docs/user_guide/01_material-handling for section == "material"
    """
    if not repo:
        return []

    # Try commit zip first if ref looks like a commit SHA, then branch, then tag
    url = f"https://github.com/{repo}/archive/{ref}.zip"

    data = None
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = resp.read()
    except Exception:
        return []

    # Restrict to specific subfolders by section
    if section == "liquid":
        # Only the Hamilton STAR(let) folder for now
        target_subdir = "docs/user_guide/00_liquid-handling/hamilton-star"
    else:
        target_subdir = "docs/user_guide/01_material-handling"
    target_subdir = target_subdir.lower()

    collected_named: list[tuple[str, str]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # Select relevant names within target_subdir
            candidate_names = []
            for name in zf.namelist():
                lower = name.lower()
                if target_subdir not in lower:
                    continue
                if lower.endswith("/"):
                    continue
                if not (
                    lower.endswith(".md")
                    or lower.endswith(".rst")
                    or lower.endswith(".txt")
                    or lower.endswith(".ipynb")
                ):
                    continue
                candidate_names.append(name)

            # Deterministic order; for liquid, ensure basic.ipynb first
            if section == "liquid":
                candidate_names.sort(key=lambda n: (0 if n.lower().endswith("basic.ipynb") else 1, n.lower()))
            else:
                candidate_names.sort(key=lambda n: n.lower())

            for name in candidate_names:
                lower = name.lower()
                try:
                    with zf.open(name) as f:
                        content_bytes = f.read()
                        if lower.endswith(".ipynb"):
                            try:
                                nb = json.loads(content_bytes.decode("utf-8"))
                            except Exception:
                                continue
                            cells = nb.get("cells") or nb.get("worksheets", [{}])[0].get("cells", [])
                            parts = []
                            for c in cells:
                                ctype = c.get("cell_type") or c.get("type")
                                src = c.get("source") or c.get("input") or []
                                if isinstance(src, list):
                                    src = "".join(src)
                                if not isinstance(src, str):
                                    continue
                                if ctype == "markdown":
                                    parts.append(src)
                                elif ctype == "code":
                                    keep_lines = []
                                    for line in src.splitlines():
                                        l = line.strip()
                                        if l.startswith("#"):
                                            continue
                                        if "pylabrobot" in l and ("import " in l or "from " in l):
                                            keep_lines.append(l)
                                    if keep_lines:
                                        parts.append("Code refs:\n" + "\n".join(keep_lines[:20]))
                                if sum(len(p) for p in parts) > 5000:
                                    break
                            text = "\n\n".join(parts)
                        else:
                            try:
                                text = content_bytes.decode("utf-8")
                            except Exception:
                                text = str(content_bytes)
                        if text:
                            collected_named.append((lower, text[:5000]))
                except Exception:
                    continue
    except Exception:
        return []

    return collected_named


def _format_liquid_user_guide(named_docs: list[tuple[str, str]]) -> str:
    """Assemble liquid-handling docs into a curated order with headings.

    named_docs: list of (filename_lower, text) from GitHub.
    Returns a single formatted string.
    """
    sections = [
        (
            "Getting started with liquid handling on a Hamilton STAR(let)",
            ["/hamilton-star/", "basic.ipynb", "basic", "getting-started"],
        ),
        ("iSWAP Module", ["iswap"]),
        ("Liquid level detection on Hamilton STAR(let)", ["liquid-level", "lld", "level_detection", "level-detection"]),
        ("Z-probing", ["z-probing", "z_probing", "z-probing", "zprobing", "z-prob"]),
        ("Foil", ["foil"]),
        ("Using the 96 head", ["96", "head", "mca", "96-head", "head-96"]),
        ("Debugging STAR issues", ["debug", "troubleshoot"]),
        ("Hamilton STAR Hardware Guide", ["hardware", "star-hardware"]),
        (
            "Using “Hamilton Liquid Classes” with Pylabrobot",
            ["liquid-classes", "liquid_classes", "hamilton-liquid-classes"],
        ),
    ]

    used: set[int] = set()
    out_parts: list[str] = []

    def pick_first(keywords: list[str]) -> str:
        for idx, (fname, text) in enumerate(named_docs):
            if idx in used:
                continue
            if any(k in fname for k in keywords):
                used.add(idx)
                return text
        return ""

    for heading, keywords in sections:
        text = pick_first(keywords)
        if text:
            out_parts.append(f"## {heading}\n\n{text}")

    # Append any remaining docs not matched, to avoid losing content
    for idx, (fname, text) in enumerate(named_docs):
        if idx not in used and text:
            # Derive a nice title from filename
            leaf = fname.rsplit("/", 1)[-1]
            title = leaf.replace("_", " ").replace("-", " ")
            title = title.rsplit(".", 1)[0].strip().title()
            out_parts.append(f"## {title}\n\n{text}")

    return "\n\n".join(out_parts)


def pylabrobot_liquid_handling_script(
    prompt: str = None,
    task_description: str = None,
    verbose: bool = True,
    save_to_dir: str = None,
) -> dict[str, Any]:
    """Generate a PyLabRobot liquid handling script based on natural language description.

    Uses the comprehensive PyLabRobot tutorial database and Claude to generate
    Python scripts for liquid handling tasks using the Hamilton STARBackend.
    Focuses on pipetting, aspirating, dispensing, and liquid transfer operations.

    Args:
        prompt (str, optional): Natural language description of the liquid handling task
        task_description (str, optional): Alternative parameter name for task description
        verbose (bool): If True, return full response; if False, return formatted results
        save_to_dir (str, optional): Directory to save the generated script. If provided,
            the script will be saved as a .py file in this directory.

    Returns:
        dict: Dictionary containing:
            - success (bool): Whether script generation was successful
            - script (str): Generated Python script (if successful)
            - description (str): Description of what the script does
            - error (str): Error message (if failed)
            - raw_response (str): Raw LLM response (if verbose)
            - saved_path (str): Path to saved file (if save_to_dir provided and successful)

    Example:
        >>> result = generate_liquid_handling_script("Move 100μL of water from wells A1:A3 to B1:B3")
        >>> print(result["script"])

        >>> # Save script to file
        >>> result = generate_liquid_handling_script(
        ...     "Move 100μL of water from wells A1:A3 to B1:B3", save_to_dir="/path/to/scripts"
        ... )
        >>> print(f"Script saved to: {result['saved_path']}")
    """
    # Use either prompt or task_description
    task_prompt = prompt or task_description

    if not task_prompt:
        return {"success": False, "error": "Either 'prompt' or 'task_description' parameter is required"}

    try:
        # Load PyLabRobot tutorial content (dynamic docs)
        tutorial_content = _load_pylabrobot_tutorial_content("liquid")
        if not tutorial_content:
            return {"success": False, "error": "Failed to load PyLabRobot docs/tutorial content"}
    except Exception as e:
        return {"success": False, "error": f"Failed to load tutorial content: {str(e)}"}

    # Create system prompt template
    system_template = """You are an expert laboratory automation programmer specialized in PyLabRobot with Hamilton STARBackend liquid handling systems.

Your task is to generate a complete, working Python script based on the user's natural language description of a liquid handling task.

PYLABROBOT TUTORIAL CONTENT:
{tutorial_content}

Based on the tutorial content above and the user's request, generate a complete Python script that focuses on LIQUID HANDLING operations such as:

1. Uses proper PyLabRobot imports and setup
2. Configures the Hamilton STARBackend appropriately
3. Sets up the deck layout with appropriate carriers, tip racks, and plates.
4. Implements the requested liquid handling operations (aspirate/dispense/transfer)
5. Includes proper tip handling and liquid level detection
6. Includes proper error handling and resource cleanup
7. Follows the tutorial content's syntax and example code as closely as possible.
8. Use only available resources from the tutorial content.

Extra Notes:
- Use hamilton_96_tiprack_1000uL_filter instead of HTF (deprecated). Note the capital L in uL.
- Use Cor_96_wellplate_360ul_Fb instead of Corning_96_wellplate_360ul_Fb.
- You must name all your plates, tip racks, and carriers.
- Assign labware into carriers via slot assignment (tip_car[0] = tiprack). Assign plates to rails using lh.deck.assign_child_resource(plate_car, rails=14).
- Rails must be between -4 and 32.
- There are some methods that are not async, including lh.summary(). Do not use await for these methods.


Focus specifically on:
- Pipetting operations
- Liquid transfers between wells/plates
- Serial dilutions
- Multi-channel operations
- Volume handling and accuracy

Your response should be a JSON object with the following fields:
1. "script": The complete Python script as a string
2. "description": A brief description of what the script accomplishes
3. "requirements": List of any special requirements or setup needed
4. "safety_notes": Any important safety considerations

The script should be ready to run (assuming proper hardware setup) and should include:
- Proper async/await usage
- Resource setup and cleanup
- Error handling
- Comments explaining key steps

Return ONLY the JSON object with no additional text."""

    try:
        # Get LLM instance
        llm = get_llm()

        # Format the system prompt with tutorial content
        system_prompt = system_template.format(tutorial_content=tutorial_content)

        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a PyLabRobot liquid handling script for: {task_prompt}"},
        ]

        # Query the LLM
        response = llm.invoke(messages)
        llm_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # Try to parse JSON response using robust method from database.py
        try:
            # Find JSON boundaries (in case LLM adds explanations)
            json_start = llm_text.find("{")
            json_end = llm_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_text = llm_text[json_start:json_end]
            else:
                json_text = llm_text

            # Handle triple quotes by converting them to proper JSON strings
            # Replace """ with proper JSON string escaping
            json_text = re.sub(
                r':\s*"""([^"]*(?:"(?!""")[^"]*)*)"""',
                lambda m: ': "'
                + m.group(1).replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
                + '"',
                json_text,
                flags=re.DOTALL,
            )

            parsed_response = json.loads(json_text)

            result = {
                "success": True,
                "script": parsed_response.get("script", ""),
                "description": parsed_response.get("description", ""),
                "requirements": parsed_response.get("requirements", []),
                "safety_notes": parsed_response.get("safety_notes", ""),
            }

            # Save script to file if directory provided
            if save_to_dir and result["script"]:
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(save_to_dir, exist_ok=True)

                    # Generate filename with timestamp for uniqueness
                    import datetime

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pylabrobot_liquid_handling_{timestamp}.py"

                    # Ensure unique filename
                    filepath = os.path.join(save_to_dir, filename)
                    counter = 1
                    while os.path.exists(filepath):
                        name_parts = filename.rsplit(".", 1)
                        filepath = os.path.join(save_to_dir, f"{name_parts[0]}_{counter}.{name_parts[1]}")
                        counter += 1

                    # Write the script to file
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(result["script"])

                    result["saved_path"] = filepath

                except Exception as save_error:
                    result["save_error"] = f"Failed to save script: {str(save_error)}"

            if verbose:
                result["raw_response"] = llm_text

            return result

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            return {
                "success": False,
                "error": f"Failed to parse LLM response: {str(e)}",
                "raw_response": llm_text if verbose else None,
            }

    except Exception as e:
        return {"success": False, "error": f"Failed to generate script: {str(e)}"}


def pylabrobot_material_handling_script(
    prompt: str = None,
    task_description: str = None,
    verbose: bool = True,
    save_to_dir: str = None,
) -> dict[str, Any]:
    """Generate a PyLabRobot material handling script based on natural language description.

    Uses the comprehensive PyLabRobot tutorial database and Claude to generate
    Python scripts for material handling tasks using the Hamilton STARBackend iSWAP module.
    Focuses on plate movement, gripper operations, and robotic manipulation.

    Args:
        prompt (str, optional): Natural language description of the material handling task
        task_description (str, optional): Alternative parameter name for task description
        verbose (bool): If True, return full response; if False, return formatted results
        save_to_dir (str, optional): Directory to save the generated script. If provided,
            the script will be saved as a .py file in this directory.

    Returns:
        dict: Dictionary containing:
            - success (bool): Whether script generation was successful
            - script (str): Generated Python script (if successful)
            - description (str): Description of what the script does
            - error (str): Error message (if failed)
            - raw_response (str): Raw LLM response (if verbose)
            - saved_path (str): Path to saved file (if save_to_dir provided and successful)

    Example:
        >>> result = generate_material_handling_script("Move plate from position A to position B")
        >>> print(result["script"])

        >>> # Save script to file
        >>> result = generate_material_handling_script(
        ...     "Move plate from position A to position B", save_to_dir="/path/to/scripts"
        ... )
        >>> print(f"Script saved to: {result['saved_path']}")
    """
    # Use either prompt or task_description
    task_prompt = prompt or task_description

    if not task_prompt:
        return {"success": False, "error": "Either 'prompt' or 'task_description' parameter is required"}

    try:
        # Load PyLabRobot tutorial content (dynamic docs)
        tutorial_content = _load_pylabrobot_tutorial_content("material")
        if not tutorial_content:
            return {"success": False, "error": "Failed to load PyLabRobot docs/tutorial content"}
    except Exception as e:
        return {"success": False, "error": f"Failed to load tutorial content: {str(e)}"}

    # Create system prompt template focused on material handling
    system_template = """You are an expert laboratory automation programmer specialized in PyLabRobot with Hamilton STARBackend material handling systems, particularly the iSWAP robotic gripper module.

Your task is to generate a complete, working Python script based on the user's natural language description of a material handling task.

PYLABROBOT TUTORIAL CONTENT:
{tutorial_content}

Based on the tutorial content above and the user's request, generate a complete Python script that focuses on MATERIAL HANDLING operations such as:

1. Uses proper PyLabRobot imports and setup
2. Configures the Hamilton STARBackend appropriately
3. Sets up the deck layout with appropriate carriers, tip racks, and plates.
4. Implements the requested liquid handling operations (aspirate/dispense/transfer)
5. Includes proper tip handling and liquid level detection
6. Includes proper error handling and resource cleanup
7. Follows the tutorial content's syntax and example code as closely as possible.
8. Use only available resources from the tutorial content.


Focus specifically on:
- Plate transfers and positioning
- iSWAP gripper operations (open/close/rotate)
- Safe movement protocols
- Position calibration
- Manual positioning when needed

Your response should be a JSON object with the following fields:
1. "script": The complete Python script as a string
2. "description": A brief description of what the script accomplishes
3. "requirements": List of any special requirements or setup needed
4. "safety_notes": Any important safety considerations

The script should be ready to run (assuming proper hardware setup) and should include:
- Proper async/await usage
- Resource setup and cleanup
- Error handling
- Comments explaining key steps
- iSWAP-specific safety protocols

Return ONLY the JSON object with no additional text."""

    try:
        # Get LLM instance
        llm = get_llm()

        # Format the system prompt with tutorial content
        system_prompt = system_template.format(tutorial_content=tutorial_content)

        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a PyLabRobot material handling script for: {task_prompt}"},
        ]

        # Query the LLM
        response = llm.invoke(messages)
        llm_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # Try to parse JSON response using robust method from database.py
        try:
            # Find JSON boundaries (in case LLM adds explanations)
            json_start = llm_text.find("{")
            json_end = llm_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_text = llm_text[json_start:json_end]
            else:
                json_text = llm_text

            # Handle triple quotes by converting them to proper JSON strings
            # Replace """ with proper JSON string escaping
            json_text = re.sub(
                r':\s*"""([^"]*(?:"(?!""")[^"]*)*)"""',
                lambda m: ': "'
                + m.group(1).replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
                + '"',
                json_text,
                flags=re.DOTALL,
            )

            parsed_response = json.loads(json_text)

            result = {
                "success": True,
                "script": parsed_response.get("script", ""),
                "description": parsed_response.get("description", ""),
                "requirements": parsed_response.get("requirements", []),
                "safety_notes": parsed_response.get("safety_notes", ""),
            }

            # Save script to file if directory provided
            if save_to_dir and result["script"]:
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(save_to_dir, exist_ok=True)

                    # Generate filename with timestamp for uniqueness
                    import datetime

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pylabrobot_material_handling_{timestamp}.py"

                    # Ensure unique filename
                    filepath = os.path.join(save_to_dir, filename)
                    counter = 1
                    while os.path.exists(filepath):
                        name_parts = filename.rsplit(".", 1)
                        filepath = os.path.join(save_to_dir, f"{name_parts[0]}_{counter}.{name_parts[1]}")
                        counter += 1

                    # Write the script to file
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(result["script"])

                    result["saved_path"] = filepath

                except Exception as save_error:
                    result["save_error"] = f"Failed to save script: {str(save_error)}"

            if verbose:
                result["raw_response"] = llm_text

            return result

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            return {
                "success": False,
                "error": f"Failed to parse LLM response: {str(e)}",
                "raw_response": llm_text if verbose else None,
            }

    except Exception as e:
        return {"success": False, "error": f"Failed to generate script: {str(e)}"}


def test_pylabrobot_script(
    script_input: str,
    enable_tracking: bool = False,
    timeout_seconds: int = 60,
    save_test_report: bool = False,
    test_report_dir: str = None,
) -> dict[str, Any]:
    """Test a PyLabRobot script using simulation and validation.

    Uses PyLabRobot's ChatterboxBackend and tracking systems to
    validate generated scripts without requiring physical hardware.

    Args:
        script_input (str): Either the PyLabRobot script code as a string, or a file path to a .py file
        enable_tracking (bool): Enable tip and volume tracking for error detection
        timeout_seconds (int): Maximum execution time before timeout
        save_test_report (bool): Whether to save detailed test results to file
        test_report_dir (str, optional): Directory to save test reports

    Returns:
        dict: Dictionary containing:
            - success (bool): Whether the script passed all tests
            - test_results (dict): Detailed test results for each validation step
            - execution_summary (dict): Summary of operations performed
            - errors (list): List of errors encountered
            - warnings (list): List of warnings
            - test_report_path (str): Path to saved report if requested

    Example:
        >>> # Test with script content string
        >>> script = "async def main(): ..."
        >>> result = test_pylabrobot_script(script)

        >>> # Test with file path
        >>> result = test_pylabrobot_script("/path/to/script.py")

        >>> if result["success"]:
        ...     print("Test passed!")
        ... else:
        ...     print(f"Test failed: {result['errors']}")
    """
    start_time = time.time()
    test_results = {
        "syntax_valid": False,
        "imports_valid": False,
        "simulation_successful": False,
        "tracking_enabled": enable_tracking,
    }
    execution_summary = {"operations_performed": 0, "tips_used": 0, "liquid_transferred": 0.0, "execution_time": 0.0}
    errors = []
    warnings = []

    # Determine if input is a file path or script content
    script_content = ""
    try:
        # Check if input looks like a file path and exists
        if (
            script_input.endswith(".py") and os.path.isfile(script_input) and "\n" not in script_input[:100]
        ):  # Basic heuristic: file paths shouldn't have newlines
            try:
                with open(script_input, encoding="utf-8") as f:
                    script_content = f.read()
                test_results["input_type"] = "file"
                test_results["file_path"] = script_input
            except Exception as e:
                errors.append(f"Failed to read script file '{script_input}': {str(e)}")
                return _create_test_result(
                    False,
                    test_results,
                    execution_summary,
                    errors,
                    warnings,
                    start_time,
                    save_test_report,
                    test_report_dir,
                )
        else:
            # Treat as script content string
            script_content = script_input
            test_results["input_type"] = "string"

    except Exception as e:
        errors.append(f"Failed to process script input: {str(e)}")
        return _create_test_result(
            False, test_results, execution_summary, errors, warnings, start_time, save_test_report, test_report_dir
        )

    if not script_content.strip():
        errors.append("Script content is empty")
        return _create_test_result(
            False, test_results, execution_summary, errors, warnings, start_time, save_test_report, test_report_dir
        )

    try:
        # Step 1: Syntax Validation
        try:
            ast.parse(script_content)
            test_results["syntax_valid"] = True
        except SyntaxError as e:
            errors.append(f"Syntax Error: {str(e)} at line {e.lineno}")
            return _create_test_result(
                False, test_results, execution_summary, errors, warnings, start_time, save_test_report, test_report_dir
            )

        # Step 2: Import Validation
        import_results = _validate_pylabrobot_imports(script_content)
        test_results["imports_valid"] = import_results["success"]
        if not import_results["success"]:
            errors.extend(import_results["errors"])
            warnings.extend(import_results["warnings"])

        # Step 3: Replace backends with ChatterboxBackend for simulation
        modified_script = _modify_script_for_testing(script_content, enable_tracking)

        # Step 4: Execute script in controlled environment
        execution_result = _execute_script_safely(modified_script, timeout_seconds)

        test_results["simulation_successful"] = execution_result["success"]
        execution_summary.update(execution_result["summary"])

        if not execution_result["success"]:
            errors.extend(execution_result["errors"])

        warnings.extend(execution_result.get("warnings", []))

    except Exception as e:
        errors.append(f"Unexpected error during testing: {str(e)}")
        traceback.print_exc()

    overall_success = (
        test_results["syntax_valid"] and test_results["imports_valid"] and test_results["simulation_successful"]
    )

    return _create_test_result(
        overall_success,
        test_results,
        execution_summary,
        errors,
        warnings,
        start_time,
        save_test_report,
        test_report_dir,
    )


def _validate_pylabrobot_imports(script_content: str) -> dict[str, Any]:
    """Validate that all PyLabRobot imports in the script are available."""
    import_errors = []
    import_warnings = []

    try:
        # Parse the script to find import statements
        tree = ast.parse(script_content)
        pylabrobot_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "pylabrobot" in alias.name:
                        pylabrobot_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and "pylabrobot" in node.module:
                    for alias in node.names:
                        full_import = f"{node.module}.{alias.name}"
                        pylabrobot_imports.append(full_import)

        # Try to import each PyLabRobot module/class
        for import_name in pylabrobot_imports:
            try:
                # Handle different import patterns
                if "." in import_name:
                    parts = import_name.split(".")
                    module_parts = parts[:-1]
                    class_name = parts[-1]

                    # Import the module
                    module_name = ".".join(module_parts)
                    module = __import__(module_name, fromlist=[class_name])

                    # Check if the class/function exists
                    if not hasattr(module, class_name):
                        import_errors.append(f"Cannot find '{class_name}' in module '{module_name}'")
                else:
                    # Direct module import
                    __import__(import_name)

            except ImportError as e:
                import_errors.append(f"Failed to import '{import_name}': {str(e)}")
            except Exception as e:
                import_warnings.append(f"Warning validating import '{import_name}': {str(e)}")

    except Exception as e:
        import_errors.append(f"Error parsing imports: {str(e)}")

    return {"success": len(import_errors) == 0, "errors": import_errors, "warnings": import_warnings}


def _modify_script_for_testing(script_content: str, enable_tracking: bool) -> str:
    """Modify script to use simulation backends and enable tracking."""
    modified_script = script_content

    # Replace STARBackend with LiquidHandlerChatterboxBackend for simulation
    replacements = [("STARBackend()", "LiquidHandlerChatterboxBackend()")]

    for old, new in replacements:
        modified_script = modified_script.replace(old, new)

    lines = modified_script.split("\n")
    insert_at = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(("from ", "import ", "#")):
            insert_at = i + 1
            continue
        if line.strip().startswith(("async def", "def", "class", "if __name__")):
            break
        insert_at = i + 1
    lines.insert(insert_at, "from pylabrobot.liquid_handling.backends import LiquidHandlerChatterboxBackend")
    modified_script = "\n".join(lines)

    # Add tracking imports and setup at the beginning
    if enable_tracking:
        tracking_setup = """
# Enable PyLabRobot tracking for validation
try:
    from pylabrobot.resources import set_tip_tracking, set_volume_tracking
    set_tip_tracking(True)
    set_volume_tracking(True)
except ImportError:
    pass  # Tracking not available in this PyLabRobot version

"""
        # Insert after imports but before main function
        lines = modified_script.split("\n")
        insert_index = 0
        for i, line in enumerate(lines):
            if (
                line.strip().startswith("async def")
                or line.strip().startswith("def")
                or line.strip().startswith("if __name__")
            ):
                insert_index = i
                break

        lines.insert(insert_index, tracking_setup)
        modified_script = "\n".join(lines)

    return modified_script


def _execute_script_safely(script_content: str, timeout_seconds: int) -> dict[str, Any]:
    """Execute the modified script in a safe environment."""
    errors = []
    warnings = []
    summary = {"operations_performed": 0, "tips_used": 0, "liquid_transferred": 0.0}

    try:
        # Create a temporary file for the script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script_content)
            temp_script_path = f.name

        # Execute the script with timeout
        try:
            # Use threading for timeout control
            import threading

            result = None
            exception = None

            def target():
                nonlocal result, exception
                try:
                    result = _run_script_with_monitoring(temp_script_path)
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                errors.append(f"Script execution timed out after {timeout_seconds} seconds")
            elif exception:
                raise exception
            elif result:
                summary.update(result.get("summary", {}))
                warnings.extend(result.get("warnings", []))

                return {"success": True, "summary": summary, "errors": errors, "warnings": warnings}
            else:
                errors.append("Script execution completed but returned no result")
        except Exception as e:
            errors.append(f"Script execution failed: {str(e)}")

    except Exception as e:
        errors.append(f"Failed to prepare script execution: {str(e)}")
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_script_path)
        except OSError:
            pass

    return {"success": False, "summary": summary, "errors": errors, "warnings": warnings}


def _run_script_with_monitoring(script_path: str) -> dict[str, Any]:
    """Run the script and monitor its execution."""
    # Note: This is a simplified version. In practice, you might want to
    # use subprocess or other isolation methods for safety

    warnings = []
    summary = {"operations_performed": 0, "tips_used": 0, "liquid_transferred": 0.0}

    try:
        # Read and execute the script
        with open(script_path) as f:
            script_content = f.read()

        # Create a namespace for execution
        namespace = {
            "__name__": "__main__",
            "__file__": script_path,
        }

        # Execute the script
        exec(script_content, namespace)

        # If the script has a main function, run it
        if "main" in namespace and callable(namespace["main"]):
            if asyncio.iscoroutinefunction(namespace["main"]):
                # Run async main function using asyncio.run if not in event loop
                try:
                    asyncio.get_running_loop()
                    # We're in an event loop, create a new thread to run asyncio.run
                    import threading

                    result = None
                    exception = None

                    def run_async():
                        nonlocal result, exception
                        try:
                            asyncio.run(namespace["main"]())
                        except Exception as e:
                            exception = e

                    thread = threading.Thread(target=run_async)
                    thread.start()
                    thread.join()

                    if exception:
                        raise exception
                except RuntimeError:
                    # No event loop running, safe to use asyncio.run
                    asyncio.run(namespace["main"]())
            else:
                namespace["main"]()

        # Try to extract execution summary (this would need to be enhanced
        # based on actual PyLabRobot API for getting execution statistics)
        summary["operations_performed"] = 1  # Placeholder

    except Exception as e:
        raise Exception(f"Script execution error: {str(e)}") from e

    return {"summary": summary, "warnings": warnings}


def _create_test_result(
    success: bool,
    test_results: dict,
    execution_summary: dict,
    errors: list,
    warnings: list,
    start_time: float,
    save_test_report: bool,
    test_report_dir: str,
) -> dict[str, Any]:
    """Create the final test result dictionary."""
    # Calculate total execution time
    total_execution_time = time.time() - start_time
    execution_summary["total_execution_time"] = total_execution_time

    result = {
        "success": success,
        "test_results": test_results,
        "execution_summary": execution_summary,
        "errors": errors,
        "warnings": warnings,
    }

    # Save test report if requested
    if save_test_report:
        try:
            if test_report_dir:
                os.makedirs(test_report_dir, exist_ok=True)
            else:
                test_report_dir = tempfile.gettempdir()

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_filename = f"pylabrobot_test_report_{timestamp}.json"
            report_path = os.path.join(test_report_dir, report_filename)

            with open(report_path, "w") as f:
                json.dump(result, f, indent=2)

            result["test_report_path"] = report_path

        except Exception as e:
            warnings.append(f"Failed to save test report: {str(e)}")

    return result
