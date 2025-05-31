# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# Point to the project root to find the genostockpy package
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GenoStockPy'
copyright = '2024, GenoStockPy Contributors' # Replace with actual year/authors
author = 'GenoStockPy Contributors'
release = '0.1.0' # Initial development version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Include documentation from docstrings
    'sphinx.ext.napoleon',     # Support for Google and NumPy style docstrings
    'sphinx.ext.intersphinx',  # Link to other projects' documentation (Python, NumPy, Pandas)
    'sphinx.ext.viewcode',     # Add links to source code
    'sphinx.ext.githubpages',  # Helps with GitHub Pages deployment
    'myst_parser',             # For Markdown support
    'sphinx.ext.autosummary',  # Create summary tables
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Source suffix for markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown', # if using .txt for markdown
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index' # In Sphinx 3.x+ this is 'root_doc', but 'master_doc' is widely used.
                     # For Sphinx 5.x+, explicitly use root_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme' # ReadTheDocs theme is popular and clean
# html_theme = 'alabaster' # Default Sphinx theme
html_static_path = ['_static']

# -- Options for autodoc -----------------------------------------------------
autodoc_member_order = 'bysource' # Order members by source order
autodoc_default_options = {
    'members': True,
    'undoc-members': True, # Show members with no docstrings (useful during development)
    'private-members': False,
    'special-members': '__init__', # Show __init__ methods
    'show-inheritance': True,
}
autosummary_generate = True # Turn on autosummary extension

# -- Options for Napoleon (Google/NumPy docstrings) --------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True # Set to False if primarily using Google style
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# Ensure that the setup function is present for autodoc if needed
# def setup(app):
#     app.add_css_file('custom.css') # Example for custom CSS

# For Sphinx 5.x+
root_doc = 'index'
highlight_language = 'python3'
pygments_style = 'sphinx'
todo_include_todos = False
html_show_sourcelink = True # If source code is public
htmlhelp_basename = 'GenoStockPydoc'
latex_elements = {} # For PDF output via LaTeX
latex_documents = [
  (root_doc, 'GenoStockPy.tex', 'GenoStockPy Documentation',
   author, 'manual'),
]
man_pages = [
    (root_doc, 'genostockpy', 'GenoStockPy Documentation',
     [author], 1)
]
texinfo_documents = [
  (root_doc, 'GenoStockPy', 'GenoStockPy Documentation',
   author, 'GenoStockPy', 'Quantitative Genetics and Genomics in Python.',
   'Miscellaneous'),
]
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']
```

Now, I'll create the main `index.rst`.
