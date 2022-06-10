import os
import sys
import tempfile
import subprocess
import importlib
import inspect
import time
from io import BytesIO
from zipfile import ZipFile

import requests

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'DaCeML'
copyright = '2020, Scalable Parallel Computing Laboratory, ETH Zurich'
author = 'Scalable Parallel Computing Laboratory, ETH Zurich, and the DaCeML authors'

# -- Configuration -----------------------------------------------------------

extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx_autodoc_typehints',
    'sphinx.ext.linkcode', 'sphinx_gallery.gen_gallery'
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
dace.Config.set('debugprint', value=False)
'''

html_sidebars = {
    '**':
    ['localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']
}

typehints_fully_qualified = False
add_module_names = False
autoclass_content = 'both'

build_fpga_docs = "DACEML_DOC_BUILD_FPGA" in os.environ and os.environ[
    "DACEML_DOC_BUILD_FPGA"] == 'True'
build_cuda_docs = "DACEML_DOC_BUILD_CUDA" in os.environ and os.environ[
    "DACEML_DOC_BUILD_CUDA"] == 'True'
if build_cuda_docs and build_fpga_docs:
    pattern = "/plot_"
elif not build_cuda_docs and build_fpga_docs:
    pattern = "/plot(?!_cuda)_"
elif build_cuda_docs and not build_fpga_docs:
    pattern = "/plot(?!_fpga)_"
else:
    pattern = "/plot(?!(_fpga)|(_cuda))_"

print()
print(pattern)

sphinx_gallery_conf = {
    'default_thumb_file': 'dace.png',
    'filename_pattern': pattern
}


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


# the following is inspired by https://github.com/dfm/rtds-action/blob/main/src/rtds_action/rtds_action.py
def download_auto_examples_artifact(gh_token):
    print("Downloading auto_examples")
    try:
        git_hash = (subprocess.check_output(["git", "rev-parse",
                                             "HEAD"]).strip().decode("ascii"))
    except subprocess.CalledProcessError:
        raise RuntimeError("can't get git hash")

    expected_name = f"auto_examples_{git_hash}"
    out_dir = os.path.join(os.path.dirname(__file__), "auto_examples")

    # 3 retries
    for i in range(3):
        # read the artifacts
        r = requests.get(
            f"https://api.github.com/repos/spcl/daceml/actions/artifacts",
            params=dict(per_page=100),
        )
        if r.status_code != 200:
            raise RuntimeError(f"Can't list files ({r.status_code})")

        result = r.json()
        for artifact in result.get("artifacts", []):
            if artifact["name"] == expected_name:
                print(f"Found artifact {artifact}")
                r = requests.get(
                    artifact["archive_download_url"],
                    headers={"Authorization": f"token {gh_token}"},
                )

                if r.status_code != 200:
                    raise ValueError(
                        f"Can't download artifact ({r.status_code})")

                with ZipFile(BytesIO(r.content)) as f:
                    f.extractall(path=out_dir)

                return

        print(f"Couldn't find expected artifact '{expected_name}' "
              f"at https://api.github.com/repos/spcl/daceml/actions/artifacts")
        time.sleep(30)
        print("Retrying...")

    raise ValueError(
        f"Couldn't find expected artifact '{expected_name}' "
        f"at https://api.github.com/repos/spcl/daceml/actions/artifacts")


# RTD-specific steps
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    # set ORT_RELEASE to nonsense
    os.environ['ORT_RELEASE'] = "FAIL"

    # the github access token can only be accessed from master builds
    token = os.environ["GH_TOKEN"]
    download_auto_examples_artifact(token)
