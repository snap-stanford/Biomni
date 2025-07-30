import os

from biomni.tool.tool_description.proteomics import get_description

description = get_description(
    os.path.dirname(os.path.abspath(__file__)) + "/../genomics.py"
)
