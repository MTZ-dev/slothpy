# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "SlothPy"
copyright = "2023, Mikołaj Tadeusz Żychowicz"
author = "Mikołaj Tadeusz Żychowicz"
release = "0.1.14"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "numpydoc",
    "sphinx_togglebutton",
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.autosectionlabel",
]

templates_path = ["_templates"]
exclude_patterns = []
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
toc_object_entries_show_parents = "hide"
html_theme_options = {
    "show_nav_level": 8,
    "navigation_depth": 8,
    "logo": {
        "text": "SlothPy",
        "image_dark": "_static/slothpy_3.png",
        "alt_text": "SlothPy",
    },
    "show_toc_level": 1,
    "secondary_sidebar_items": [
        "edit-this-page",
        "sourcelink",
    ],  # Here add "page-toc" for table of contents on the right
    "navbar_align": "content",
    "navbar_center": ["navbar-nav"],
    "navbar_end": [
        "custom_version.html",
        "theme-switcher",
        "navbar-icon-links",
    ],
    "header_links_before_dropdown": 10,
    "icon_links": [
        {
            "name": "Twitter",
            "url": "https://twitter.com/multilumimater",
            "icon": "fa-brands fa-twitter",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/MTZ-dev/slothpy",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/slothpy/",
            "icon": "fa-custom fa-pypi",
        },
        {
            "name": "MultiLumiMater",
            "url": "https://multilumimater.pl/",
            "icon": "_static/mlmg.png",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
    ],
    "use_edit_page_button": True,
    "show_version_warning_banner": True,
}
html_context = {
    "github_user": "MTZ-dev",
    "github_repo": "Sloth",
    "github_version": "dev-doc",
    "doc_path": "doc",
}
html_static_path = ["_static"]
html_additional_pages = {
    "custom_version.html": "_templates/custom_version.html"
}
html_css_files = ["css/custom.css"]
html_js_files = ["custom-icon.js", "collapsible_toc.js"]
html_sidebars = {
    "**": [
        "search-field.html",
        "sidebar-nav-bs.html",
        "globaltoc.html",
    ],
}

html_logo = "_static/slothpy_3.png"
html_favicon = "_static/slothpy_3.png"

nbsphinx_execute = "never"
