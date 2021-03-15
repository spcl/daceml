import os
import sys
import tempfile
import subprocess
import importlib
import inspect

sys.path.insert(0, os.path.abspath('..'))

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    # set ORT_RELEASE to nonsense
    os.environ['ORT_RELEASE'] = "FAIL"

# -- Project information -----------------------------------------------------

project = 'DaCeML'
copyright = '2020, Scalable Parallel Computing Laboratory, ETH Zurich'
author = 'Scalable Parallel Computing Laboratory, ETH Zurich, and the DaCeML authors'

# -- Configuration -----------------------------------------------------------

extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx_autodoc_typehints',
    'sphinx.ext.linkcode'
]

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
import dace
'''

html_sidebars = {
    '**':
    ['localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']
}

typehints_fully_qualified = False
add_module_names = False
autoclass_content = 'both'


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None

    # main idea taken from https://gitlab.merchise.org/merchise/odoo
    module = info['module']
    fullname = info['fullname']

    if not module:
        return None

    obj = importlib.import_module(module)
    for item in fullname.split('.'):
        obj = getattr(obj, item, None)

    if obj is None:
        return None

    # get original from decorated methods
    try:
        objc = getattr(obj, '_orig')
    except AttributeError:
        pass

    try:
        obj_source_path = inspect.getsourcefile(obj)
        _, line = inspect.getsourcelines(obj)
    except (TypeError, IOError):
        # obj doesn't have a module, or something
        return None

    project_root = "../"
    return "https://github.com/spcl/daceml/blob/master/{}{}".format(
        os.path.relpath(obj_source_path, project_root),
        "#L{}".format(line) if line is not None else "")
