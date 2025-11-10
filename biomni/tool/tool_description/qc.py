import os
from .statistics import get_description


description = get_description(
    os.path.dirname(os.path.abspath(__file__)) + "/../qc.py"
)