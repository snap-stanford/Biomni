# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "biomni"
copyright = "2025, Biomni"
author = "Biomni"
release = "0.0.6"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []
napoleon_use_param = True

autodoc_mock_imports = [
    "biomni.agent.base_agent",
    "biomni.",
    "PyPDF2",
    "bioagentos",
    "Bio",
    "gseapy",
    "googlesearch",
    "scanpy",
    "torch",
    "gget",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
