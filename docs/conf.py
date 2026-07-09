# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../"))
os.environ["OMP_DISPLAY_ENV"] = "FALSE"
os.environ["OMP_DISPLAY_AFFINITY"] = "FALSE"

project = "ROMS-Tools"
copyright = "2024, ROMS-Tools developers"
author = "ROMS-Tools developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinxcontrib.bibtex",
    "sphinx_design",
]
myst_enable_extensions = ["dollarmath", "amsmath"]

numpydoc_show_class_members = True
napolean_google_docstring = False
napolean_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_custom_sections = [
    ("Returns", "params_style"),
    ("Sets Attributes", "params_style"),
    ("Required Parameter Sections", "params_style"),
    ("Assumptions", "notes_style"),
    ("Example Config YAML File", "example"),
]

# autodoc_default_options = {
#     "inherited-members": "BaseModel, pydantic.BaseModel, pydantic.main.BaseModel"
# }

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
# html_theme = 'alabaster'
html_static_path = ["_static"]

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

html_theme_options = {
    "repository_url": "https://github.com/CWorthy-ocean/roms-tools",
    "use_repository_button": True,
}


def export_notebooks_as_scripts(app):
    """Export select tutorial notebooks as plain .py scripts for download.

    Not all tutorials benefit from a script version, so this is opt-in via
    the `SCRIPT_EXPORT_NOTEBOOKS` list rather than applied to every notebook.
    """
    from nbconvert import PythonExporter

    docs_dir = os.path.abspath(os.path.dirname(__file__))
    static_dir = os.path.join(docs_dir, "_static")
    os.makedirs(static_dir, exist_ok=True)

    exporter = PythonExporter()
    for name in SCRIPT_EXPORT_NOTEBOOKS:
        notebook_path = os.path.join(docs_dir, f"{name}.ipynb")
        script, _ = exporter.from_filename(notebook_path)
        with open(os.path.join(static_dir, f"{name}.py"), "w") as f:
            f.write(script)


SCRIPT_EXPORT_NOTEBOOKS = ["end_to_end"]


def setup(app):
    app.connect("builder-inited", export_notebooks_as_scripts)
