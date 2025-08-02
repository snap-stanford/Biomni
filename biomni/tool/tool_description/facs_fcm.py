import inspect
import ast
import os
from docstring_parser import parse


from .proteomics import get_description


description = get_description(
    os.path.dirname(os.path.abspath(__file__)) + "/../facs_fcm.py"
)
