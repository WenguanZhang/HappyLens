"""Sphinx configuration for the HappyLens documentation."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "HappyLens"
author = "Wenguan Zhang and HappyLens contributors"
copyright = "2023-2026, Wenguan Zhang"
version = "1.0"
release = "1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosectionlabel_prefix_document = True
myst_heading_anchors = 4
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]

html_theme = "sphinx_rtd_theme"
html_title = "HappyLens Documentation"
html_logo = str(ROOT / "website" / "assets" / "happylens-logo.svg")
html_favicon = str(ROOT / "website" / "assets" / "happylens-icon.svg")
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = True

html_theme_options = {
    "logo_only": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
