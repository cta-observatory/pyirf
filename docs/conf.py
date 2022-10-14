# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from pyirf import __version__

# -- Project information -----------------------------------------------------

project = "pyirf"
copyright = "2020, Maximilian Nöthe, Michele Peresano, Thomas Vuillaume"
author = "Maximilian Nöthe, Michele Peresano, Thomas Vuillaume"

# The full version, including alpha/beta/rc tags
version = __version__

# -- General configuration ---------------------------------------------------

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]
# source_suffix = '.rst'

# The master toctree document.
master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "numpydoc",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
]

# nbsphinx
# nbsphinx_execute = "never"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png', }",
    "--InlineBackend.rc={'figure.dpi': 300}",
]

numpydoc_show_class_members = False
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    'canonical_url': 'https://cta-observatory.github.io/pyirf',
    'display_version': True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.7", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/latest/", None),
    "gammapy": ("https://docs.gammapy.org/0.18/", None),
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
