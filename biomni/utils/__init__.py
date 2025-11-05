"""
Biomni utilities package.

This package re-exports all functions from biomni.utils.py to maintain
backward compatibility when biomni.utils is imported as a package.
"""

import os
import importlib.util

# Get the path to the parent utils.py file
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_utils_py_path = os.path.join(_parent_dir, "utils.py")

# Load utils.py as a module and re-export everything
spec = importlib.util.spec_from_file_location("biomni._utils_module", _utils_py_path)
if spec and spec.loader:
    _utils_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_utils_module)

    # Re-export everything from utils.py (except private attributes)
    for name in dir(_utils_module):
        if not name.startswith("_"):
            globals()[name] = getattr(_utils_module, name)

# Import resource_filter module
from . import resource_filter  # noqa: F401
