import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biomni.config import BiomniConfig
    from biomni.llm import SourceType


# Safe allowlist: import name -> pip package name
AUTO_INSTALL_IMPORT_TO_PIP = {
    "PyPDF2": "PyPDF2",
    "arxiv": "arxiv",
    "googlesearch": "googlesearch-python",
    "pymed": "pymed",
    "scholarly": "scholarly",
}


def _detect_python_environment() -> dict[str, str | bool]:
    """Detect current Python runtime environment (conda/venv/system)."""
    conda_prefix = os.getenv("CONDA_PREFIX")
    virtual_env = os.getenv("VIRTUAL_ENV")
    in_venv = bool(getattr(sys, "base_prefix", sys.prefix) != sys.prefix or virtual_env)
    in_conda = bool(conda_prefix)

    if in_conda:
        env_type = "conda"
        env_path = conda_prefix or ""
    elif in_venv:
        env_type = "venv"
        env_path = virtual_env or sys.prefix
    else:
        env_type = "system"
        env_path = sys.prefix

    return {
        "env_type": env_type,
        "env_path": env_path,
        "python_executable": sys.executable,
        "is_virtual_like": bool(in_conda or in_venv),
    }


def _is_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _detect_missing_tool_dependencies(target_modules: set[str] | None = None) -> tuple[list[str], dict[str, list[str]]]:
    """Scan biomni/tool/*.py imports and report missing third-party modules."""
    tool_dir = Path(__file__).resolve().parent / "tool"
    stdlib_modules = set(getattr(sys, "stdlib_module_names", set()))
    ignored = {"biomni"}
    extra_ignored = {
        x.strip()
        for x in os.getenv("BIOMNI_TOOL_DEP_IGNORE", "").split(",")
        if x.strip()
    }
    ignored |= extra_ignored

    package_to_files: dict[str, set[str]] = {}
    for py_file in tool_dir.glob("*.py"):
        if py_file.name.startswith("_") or py_file.name == "__init__.py":
            continue
        module_stem = py_file.stem
        if target_modules and module_stem not in target_modules and py_file.name not in target_modules:
            continue

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except Exception:
            # Ignore unparsable files in startup checks.
            continue

        imports_in_file: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports_in_file.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    continue
                if node.module:
                    imports_in_file.add(node.module.split(".")[0])

        for module_name in imports_in_file:
            if not module_name or module_name in ignored or module_name in stdlib_modules:
                continue
            package_to_files.setdefault(module_name, set()).add(py_file.name)

    missing = sorted([pkg for pkg in package_to_files if not _is_module_available(pkg)])
    missing_usage = {pkg: sorted(package_to_files[pkg]) for pkg in missing}
    return missing, missing_usage


def _normalize_tool_dep_mode(mode: str) -> tuple[str, str | None]:
    normalized = mode.strip().lower()
    if normalized in {"off", "warn", "strict"}:
        return normalized, None
    return "warn", f"Invalid BIOMNI_TOOL_DEP_CHECK_MODE={mode!r}, fallback to 'warn'. Expected off|warn|strict."


def _normalize_check_timing(value: str) -> tuple[str, str | None]:
    normalized = value.strip().lower()
    if normalized in {"startup", "post_retrieval", "both"}:
        return normalized, None
    return (
        "startup",
        f"Invalid BIOMNI_TOOL_DEP_CHECK_TIMING={value!r}, fallback to 'startup'. Expected startup|post_retrieval|both.",
    )


def _tool_dependency_messages(
    *,
    target_modules: set[str] | None,
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    missing_pkgs, usage = _detect_missing_tool_dependencies(target_modules=target_modules)
    messages: list[str] = []
    if missing_pkgs:
        preview = ", ".join(missing_pkgs[:12])
        more = "" if len(missing_pkgs) <= 12 else f" ... (+{len(missing_pkgs) - 12} more)"
        messages.append(
            f"Missing tool dependencies ({len(missing_pkgs)}): {preview}{more}. "
            "Set BIOMNI_TOOL_DEP_IGNORE to skip specific imports."
        )
        first_pkg = missing_pkgs[0]
        messages.append(f"Example missing package usage: `{first_pkg}` imported by {', '.join(usage.get(first_pkg, []))}.")
    return messages, usage, missing_pkgs


def _is_truthy_env(var_name: str, default: str = "false") -> bool:
    return os.getenv(var_name, default).strip().lower() in {"1", "true", "yes", "on"}


def _attempt_auto_install_missing_imports(missing_imports: list[str]) -> None:
    if not missing_imports:
        return

    pip_packages: list[str] = []
    unmapped: list[str] = []
    for import_name in missing_imports:
        pip_name = AUTO_INSTALL_IMPORT_TO_PIP.get(import_name)
        if pip_name:
            pip_packages.append(pip_name)
        else:
            unmapped.append(import_name)

    pip_packages = sorted(set(pip_packages))

    env_info = _detect_python_environment()
    allow_global_install = _is_truthy_env("BIOMNI_AUTO_INSTALL_ALLOW_GLOBAL", "false")
    if not env_info["is_virtual_like"] and not allow_global_install:
        print("\n" + "=" * 50)
        print("📦 AUTO-INSTALL MISSING DEPS")
        print("=" * 50)
        print(f"  Python env: {env_info['env_type']} ({env_info['env_path']})")
        print("  Status: SKIP (non-virtual environment and BIOMNI_AUTO_INSTALL_ALLOW_GLOBAL=false)")
        print("  Hint: activate venv/conda, or set BIOMNI_AUTO_INSTALL_ALLOW_GLOBAL=true")
        print("=" * 50 + "\n")
        return

    print("\n" + "=" * 50)
    print("📦 AUTO-INSTALL MISSING DEPS")
    print("=" * 50)
    print(f"  Python env: {env_info['env_type']} ({env_info['env_path']})")
    print(f"  Python executable: {env_info['python_executable']}")
    if pip_packages:
        print(f"  Install targets: {', '.join(pip_packages)}")
    if unmapped:
        print(f"  Unmapped imports (not auto-installed): {', '.join(unmapped)}")

    if not pip_packages:
        print("  Status: SKIP (no allowlisted packages to install)")
        print("=" * 50 + "\n")
        return

    if _is_truthy_env("BIOMNI_AUTO_INSTALL_DRY_RUN", "false"):
        print("  Status: DRY RUN (set BIOMNI_AUTO_INSTALL_DRY_RUN=false to execute)")
        print("=" * 50 + "\n")
        return

    # Use current interpreter so installs land in the active environment.
    cmd = [sys.executable, "-m", "pip", "install", *pip_packages]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        print("  Status: INSTALL OK")
    else:
        print("  Status: INSTALL FAILED")
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        for line in tail:
            print(f"    {line}")
    print("=" * 50 + "\n")


def run_startup_checks(
    *,
    path: str,
    llm: str | None,
    retrieval_llm: str | None,
    source: "SourceType | None",
    base_url: str | None,
    api_key: str | None,
    config: "BiomniConfig",
) -> None:
    """
    Fail-fast checks for environment/config/tool dependencies.

    Env toggles:
      - BIOMNI_STARTUP_CHECKS=0/false/off  -> disable all checks
      - BIOMNI_TOOL_DEP_CHECK_MODE=off|warn|strict (default: warn)
      - BIOMNI_TOOL_DEP_CHECK_TIMING=startup|post_retrieval|both (default: startup)
      - BIOMNI_AUTO_INSTALL_MISSING_DEPS=true|false (default: false)
      - BIOMNI_AUTO_INSTALL_DRY_RUN=true|false (default: false)
      - BIOMNI_AUTO_INSTALL_ALLOW_GLOBAL=true|false (default: false)
    """
    checks_enabled = os.getenv("BIOMNI_STARTUP_CHECKS", "true").strip().lower() not in {"0", "false", "off"}
    if not checks_enabled:
        print("⚪ Startup checks disabled by BIOMNI_STARTUP_CHECKS")
        return

    errors: list[str] = []
    warnings: list[str] = []

    effective_source = source if source is not None else config.source
    effective_llm = llm if llm is not None else config.llm
    effective_retrieval_llm = retrieval_llm if retrieval_llm is not None else getattr(config, "retrieval_llm", None)
    effective_base_url = base_url if base_url is not None else config.base_url
    effective_api_key = api_key if api_key is not None else config.api_key

    # Provider-level package checks
    if effective_source == "Anthropic":
        if not _is_module_available("langchain_anthropic"):
            errors.append("Missing package `langchain_anthropic` for source=Anthropic.")
        if not os.getenv("ANTHROPIC_API_KEY"):
            warnings.append("`ANTHROPIC_API_KEY` is not set in environment.")
        if effective_base_url:
            warnings.append(
                "`source=Anthropic` with `base_url` detected. In current get_llm(), Anthropic path ignores base_url; "
                "use `source=Custom` if you need relay/proxy base_url."
            )
    elif effective_source in {"OpenAI", "AzureOpenAI", "Gemini", "Groq", "Custom"}:
        if not _is_module_available("langchain_openai"):
            errors.append(f"Missing package `langchain_openai` for source={effective_source}.")

    if effective_source == "Custom":
        if not effective_base_url:
            errors.append("source=Custom requires `base_url`.")
        if not effective_api_key or effective_api_key == "EMPTY":
            warnings.append("source=Custom is using empty api_key.")

    # Basic path check
    if not path:
        errors.append("`path` is empty.")

    # Tool dependency checks
    tool_dep_mode, mode_warn = _normalize_tool_dep_mode(os.getenv("BIOMNI_TOOL_DEP_CHECK_MODE", "warn"))
    if mode_warn:
        warnings.append(mode_warn)
    check_timing, timing_warn = _normalize_check_timing(os.getenv("BIOMNI_TOOL_DEP_CHECK_TIMING", "startup"))
    if timing_warn:
        warnings.append(timing_warn)
    target_modules_env = os.getenv("BIOMNI_TOOL_DEP_TARGET_MODULES", "")
    target_modules = {x.strip() for x in target_modules_env.split(",") if x.strip()} or None

    if tool_dep_mode != "off" and check_timing in {"startup", "both"}:
        dep_messages, _usage, missing_pkgs = _tool_dependency_messages(target_modules=target_modules)
        if missing_pkgs and _is_truthy_env("BIOMNI_AUTO_INSTALL_MISSING_DEPS", "false"):
            _attempt_auto_install_missing_imports(missing_pkgs)
            dep_messages, _usage, missing_pkgs = _tool_dependency_messages(target_modules=target_modules)
        if dep_messages:
            if tool_dep_mode == "strict":
                errors.append(dep_messages[0])
                warnings.extend(dep_messages[1:])
            else:
                warnings.extend(dep_messages)

    print("\n" + "=" * 50)
    print("🚦 STARTUP CHECKS")
    print("=" * 50)
    print(f"  LLM: {effective_llm}")
    print(f"  Retrieval LLM: {effective_retrieval_llm}")
    print(f"  Source: {effective_source}")
    print(f"  Tool dep mode: {tool_dep_mode}")
    print(f"  Tool dep timing: {check_timing}")
    print(f"  Tool dep targets: {', '.join(sorted(target_modules)) if target_modules else 'ALL'}")

    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    - {w}")

    if errors:
        print("  Errors:")
        for e in errors:
            print(f"    - {e}")
        print("=" * 50 + "\n")
        raise RuntimeError(
            "Startup checks failed. Fix the errors above or set BIOMNI_STARTUP_CHECKS=false to bypass temporarily."
        )

    print("  Status: PASS")
    print("=" * 50 + "\n")


def run_post_retrieval_tool_checks(selected_tools: list[dict | str] | None) -> None:
    """Check dependencies for selected tool modules after tool retrieval."""
    tool_dep_mode, _mode_warn = _normalize_tool_dep_mode(os.getenv("BIOMNI_TOOL_DEP_CHECK_MODE", "warn"))
    check_timing, _timing_warn = _normalize_check_timing(os.getenv("BIOMNI_TOOL_DEP_CHECK_TIMING", "startup"))
    if tool_dep_mode == "off":
        print("⏭️ POST-RETRIEVAL TOOL DEP CHECK skipped: BIOMNI_TOOL_DEP_CHECK_MODE=off")
        return
    if check_timing not in {"post_retrieval", "both"}:
        print(
            "⏭️ POST-RETRIEVAL TOOL DEP CHECK skipped: "
            f"BIOMNI_TOOL_DEP_CHECK_TIMING={check_timing!r} (need 'post_retrieval' or 'both')"
        )
        return

    selected_modules: set[str] = set()
    selected_tool_names: list[str] = []
    tools_without_module: list[str] = []
    for tool in selected_tools or []:
        if isinstance(tool, dict):
            tool_name = str(tool.get("name") or "<unnamed_tool>")
            selected_tool_names.append(tool_name)
            module_name = tool.get("module")
            if isinstance(module_name, str) and module_name:
                selected_modules.add(module_name)
            else:
                tools_without_module.append(tool_name)
        else:
            selected_tool_names.append(str(tool))
            tools_without_module.append(str(tool))

    print("\n" + "=" * 50)
    print("🧩 POST-RETRIEVAL TOOL DEP CHECK")
    print("=" * 50)
    print(f"  Selected tools: {', '.join(selected_tool_names) if selected_tool_names else '(none)'}")
    print(f"  Derived modules: {', '.join(sorted(selected_modules)) if selected_modules else '(none)'}")
    print(f"  Mode: {tool_dep_mode}")
    print(f"  Auto install: {_is_truthy_env('BIOMNI_AUTO_INSTALL_MISSING_DEPS', 'false')}")
    if tools_without_module:
        print(f"  Tools missing module metadata: {', '.join(tools_without_module)}")

    if not selected_modules:
        if tool_dep_mode == "strict":
            print("  Errors:")
            print("    - No module metadata found on selected tools; cannot perform dependency checks reliably.")
            print("=" * 50 + "\n")
            raise RuntimeError("Post-retrieval dependency check failed: missing module metadata on selected tools.")
        print("  Status: WARN (skipped due to missing module metadata)")
        print("=" * 50 + "\n")
        return

    messages, _usage, missing_pkgs = _tool_dependency_messages(target_modules=selected_modules)
    if missing_pkgs:
        print("  Missing deps before install:")
        for msg in messages:
            print(f"    - {msg}")

    if missing_pkgs and _is_truthy_env("BIOMNI_AUTO_INSTALL_MISSING_DEPS", "false"):
        _attempt_auto_install_missing_imports(missing_pkgs)
        messages, _usage, missing_pkgs = _tool_dependency_messages(target_modules=selected_modules)
        if missing_pkgs:
            print("  Missing deps after install:")
            for msg in messages:
                print(f"    - {msg}")
        else:
            print("  Missing deps after install: none")

    if messages:
        if tool_dep_mode == "strict":
            print("  Errors:")
            for msg in messages:
                print(f"    - {msg}")
            print("=" * 50 + "\n")
            raise RuntimeError(
                "Post-retrieval dependency check failed for selected tools. Install missing packages or set "
                "BIOMNI_TOOL_DEP_CHECK_MODE=warn."
            )
        else:
            print("  Warnings:")
            for msg in messages:
                print(f"    - {msg}")
            print("  Status: WARN")
            print("=" * 50 + "\n")
            return

    print("  Status: PASS")
    print("=" * 50 + "\n")
