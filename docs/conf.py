"""Sphinx configuration for pyBehaviorLab documentation.

Build locally with::

    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html

On Read the Docs, the ``.readthedocs.yaml`` at the repo root drives
the build.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# Put the project root on sys.path so autodoc can import modules.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# -------------------------------------------------------------------- #
# Project metadata
# -------------------------------------------------------------------- #

project = "pyBehaviorLab"
author = "pyBehaviorLab contributors"
copyright = f"{datetime.now().year}, {author}"
release = os.environ.get("PYBL_VERSION", "dev")
version = release

# -------------------------------------------------------------------- #
# Extensions
# -------------------------------------------------------------------- #

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "myst_parser",
]

# -------------------------------------------------------------------- #
# Source files
# -------------------------------------------------------------------- #

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = [
    "_build",
    "_archive",
    "_trash",
    "Thumbs.db",
    ".DS_Store",
]
# Every file under docs/ is a published page. Dated plans, audits and session
# notes are kept outside the repository entirely, so nothing here has to be
# excluded by name, the old per-file exclude list went stale each time a
# report was written and left orphan warnings behind it.

autosectionlabel_prefix_document = True
nitpicky = False

# -------------------------------------------------------------------- #
# MyST
# -------------------------------------------------------------------- #

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "linkify",
    "substitution",
    "tasklist",
    "attrs_inline",
]
myst_heading_anchors = 3

# -------------------------------------------------------------------- #
# Autodoc
# -------------------------------------------------------------------- #

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__weakref__,__dict__,__module__",
}
autodoc_typehints = "signature"
autodoc_class_signature = "mixed"
autosummary_generate = True
autosummary_imported_members = False

# These modules can't be imported at doc-build time (Qt requires a
# display, hardware-specific deps may not be installed on RTD). Mock
# them so autodoc can still introspect signatures and docstrings.
autodoc_mock_imports = [
    "PySide6",
    "cv2",
    "numpy",
    "serial",
    "deeplabcut",
    "sleap",
    "esptool",
    "matplotlib",
    "scipy",
    "pandas",
    "sklearn",
    "tensorflow",
    "torch",
    "PySpin",
    "ximea",
]

# Napoleon: support both Google and NumPy docstring styles.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -------------------------------------------------------------------- #
# Cross-project references
# -------------------------------------------------------------------- #

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -------------------------------------------------------------------- #
# HTML output
# -------------------------------------------------------------------- #

html_theme = "pybehaviorlab"
html_theme_path = ["_theme"]
html_theme_options = {}
# Ensure _static exists (holds the logo/favicon + intro media; the bespoke
# theme in _theme/ supplies the layout, CSS and JS).
_STATIC = Path(__file__).parent / "_static"
_STATIC.mkdir(exist_ok=True)
html_static_path = ["_static"]
html_css_files = []
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_title = f"{project} {release}"
html_short_title = project
html_show_sourcelink = False

# -------------------------------------------------------------------- #
# Code styling
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
# Mermaid
# -------------------------------------------------------------------- #
# sphinxcontrib-mermaid 2.x picks a theme per colour scheme and, left alone,
# uses its built-in "dark" theme. That renders any node without a classDef as a
# black box, on a page that is otherwise light. Pinning BOTH themes to "base"
# with explicit variables makes every node legible whichever branch it takes.
#
# Note the option names: this version reads ``mermaid_init_config`` (a dict),
# not ``mermaid_init_js``. Setting the latter has no effect at all.
# Served from _static, not from a CDN. A rig PC is often off the network, and
# on one the CDN import simply fails: every diagram degrades to its raw
# "flowchart LR A-->B" source, which reads as a broken page rather than as a
# missing library. The bundle is code split, so _static carries the entry file
# and its 73 chunks (2.6 MB); tools/vendor_mermaid.py refreshes them.
mermaid_use_local = "mermaid.esm.min.mjs"
mermaid_version = "11.12.1"

mermaid_light_theme = "base"
mermaid_dark_theme = "base"
mermaid_init_config = {
    "startOnLoad": False,
    "securityLevel": "loose",
    "theme": "base",
    "fontFamily": 'Inter, "Segoe UI", system-ui, sans-serif',
    "themeVariables": {
        "fontSize": "16px",
        "background": "#ffffff",
        "mainBkg": "#eef4fb",
        "primaryColor": "#eef4fb",
        "primaryTextColor": "#12283d",
        "primaryBorderColor": "#3b6ea5",
        "secondaryColor": "#f1effc",
        "secondaryTextColor": "#231d47",
        "secondaryBorderColor": "#5a4ae0",
        "tertiaryColor": "#fdf0e3",
        "tertiaryTextColor": "#3a2a15",
        "tertiaryBorderColor": "#c07a2c",
        "lineColor": "#5f5c73",
        "textColor": "#1c1b24",
        "nodeTextColor": "#12283d",
        "clusterBkg": "#faf9fd",
        "clusterBorder": "#cbc6dc",
        "edgeLabelBackground": "#ffffff",
        "titleColor": "#1c1b24",
    },
    "flowchart": {
        "useMaxWidth": True,
        "htmlLabels": True,
        "curve": "basis",
        "nodeSpacing": 45,
        "rankSpacing": 50,
        "padding": 12,
    },
}

pygments_style = "friendly"
# ``none`` so Pygments doesn't reject ASCII-art / placeholder blocks.
# Individual fenced blocks opt in with ``` python / ``` json / etc.
highlight_language = "none"

# -------------------------------------------------------------------- #
# Copybutton
# -------------------------------------------------------------------- #

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
