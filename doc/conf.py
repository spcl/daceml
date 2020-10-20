import os
import sys
import tempfile
import subprocess

sys.path.insert(0, os.path.abspath('..'))

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    # download and install ORT_RELEASE
    dir_name = tempfile.mkdtemp()
    subprocess.check_call(
        ["wget", "https://github.com/orausch/onnxruntime/releases/download/build1/onnxruntime_dist_cpu.tar.gz"],
        cwd=dir_name
    )
    subprocess.check_call(
        ["tar", "-xzf", "onnxruntime_dist_cpu.tar.gz"],
        cwd=dir_name
    )
    os.environ['ORT_RELEASE'] = os.path.join(dir_name, "onnxruntime_dist_cpu")

# -- Project information -----------------------------------------------------

project = 'DaceML'
copyright = '2020, Scalable Parallel Computing Laboratory, ETH Zurich'
author = 'Scalable Parallel Computing Laboratory, ETH Zurich, and the DaceML authors'


# -- General configuration ---------------------------------------------------

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx_autodoc_typehints']

autodoc_typehints = 'description'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

doctest_global_setup = '''
import torch
import torch.nn as nn

# download efficientnet
import os
import subprocess
model_path = os.path.join("..", "tests", "onnx_files", "efficientnet.onnx")
# Download model
if not os.path.exists(model_path):
    subprocess.check_call([
        "wget",
        "https://github.com/onnx/models/raw/master/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        "--output-document={}".format(model_path)
    ])
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
