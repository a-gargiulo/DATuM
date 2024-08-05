# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
import os

sys.path.insert(0, os.path.abspath("../../code"))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DATuM"
copyright = "2023, Aldo Gargiulo"
author = "Aldo Gargiulo"
release = "1.0"
version = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = ["sphinx.ext.autodoc", "sphinx_autodoc_typehints", "sphinx_new_tab_link"]
# autodoc_default_flags = ['members', 'undoc-members']
autodoc_member_order = "bysource"
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme = 'editorial_sphinx_theme'
html_static_path = ["_static"]
# html_static_path = [os.path.join('source', '_static')]
