# -*- coding: utf-8 -*-

import sys
import os
import re
import sphinx_rtd_theme

# Prefer to use the version of the theme in this repo
# and not the installed version of the theme.
sys.path.insert(0, os.path.abspath('../..'))  # Source code dir relative to this file
# sys.path.append(os.path.abspath('./demo/'))

from sphinx_rtd_theme import __version__ as theme_version
from sphinx_rtd_theme import __version_full__ as theme_version_full
from sphinx.locale import _


# -- Project information -----------------------------------------------------
project = u'Sparse Deep GP' #u'Read the Docs Sphinx Theme'
slug = re.sub(r'\W+', '-', project.lower())
version = theme_version
release = theme_version_full
author = 'Wenyuan Zhao, Haoyuan Chen' #u'Dave Snider, Read the Docs, Inc. & contributors'
copyright = '2024, Wenyuan Zhao, Haoyuan Chen' #author
language = 'en'


# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    "sphinx.ext.coverage",
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

autosummary_generate = True  # Turn on sphinx.ext.autosummary

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md'] #'.rst'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", '**.ipynb_checkpoints']

locale_dirs = ['locale/']
gettext_compact = False

# The master toctree document
master_doc = 'index'
suppress_warnings = ['image.nonlocal_uri']
pygments_style = 'default'

# if sys.version_info < (3, 0):
#     tags.add("python2")
# else:
#     tags.add("python3")


# Configuration for intersphinx: refer to the Python standard library
intersphinx_mapping = {
    'rtd': ('https://docs.readthedocs.io/en/stable/', None),
    'rtd-dev': ('https://dev.readthedocs.io/en/stable/', None),
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': True,
    'navigation_depth': 5,
}
html_context = {}

if not 'READTHEDOCS' in os.environ:
    html_static_path = ['_static/']
    html_js_files = ['debug.js']

    # Add fake versions for local QA of the menu
    html_context['test_versions'] = list(map(
        lambda x: str(x / 10),
        range(1, 100)
    ))

html_logo = "demo/static/logo-wordmark-light.svg"
html_show_sourcelink = True
html_favicon = "demo/static/favicon.ico"

htmlhelp_basename = slug


latex_documents = [
  ('index', '{0}.tex'.format(slug), project, author, 'manual'),
]

man_pages = [
    ('index', slug, project, [author], 1)
]

texinfo_documents = [
  ('index', slug, project, author, slug, project, 'Miscellaneous'),
]


# Extensions to theme docs
def setup(app):
    from sphinx.domains.python import PyField
    from sphinx.util.docfields import Field

    app.add_object_type(
        'confval',
        'confval',
        objname='configuration value',
        indextemplate='pair: %s; configuration value',
        doc_field_types=[
            PyField(
                'type',
                label=_('Type'),
                has_arg=False,
                names=('type',),
                bodyrolename='class'
            ),
            Field(
                'default',
                label=_('Default'),
                has_arg=False,
                names=('default',),
            ),
        ]
    )