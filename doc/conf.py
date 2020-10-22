import os
import sys
import tempfile
import subprocess

sys.path.insert(0, os.path.abspath('..'))

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    # set ORT_RELEASE to nonsense
    os.environ['ORT_RELEASE'] = "FAIL"

# -- Project information -----------------------------------------------------

project = 'DaceML'
copyright = '2020, Scalable Parallel Computing Laboratory, ETH Zurich'
author = 'Scalable Parallel Computing Laboratory, ETH Zurich, and the DaceML authors'


# -- Configuration -----------------------------------------------------------

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx_autodoc_typehints']

autodoc_typehints = 'description'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

doctest_global_setup = '''
import torch
import torch.nn as nn
import os
'''

html_sidebars = {
        '**': [
            'localtoc.html',
            'relations.html',
            'sourcelink.html',
            'searchbox.html']}

typehints_fully_qualified = False
add_module_names = False
autoclass_content = 'both'
