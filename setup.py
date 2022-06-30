import os
from setuptools import setup, find_packages
import itertools
import glob
import os

# Find runtime and external library files by obtaining the module path and
# trimming the absolute path of the resulting files.
daceml_path = os.path.dirname(os.path.abspath(__file__)) + '/daceml/'
runtime_files = [
    f[len(daceml_path):] for f in itertools.chain(
        glob.glob(daceml_path + '**/*.h', recursive=True),
        glob.glob(daceml_path + '**/*.cuh', recursive=True),
        glob.glob(daceml_path + '**/*.cu', recursive=True))
]

with open("README.md", "r") as fp:
    long_description = fp.read()

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
setup(
    name='daceml',
    version='0.2.0a',
    url='https://github.com/spcl/daceml',
    author='SPCL @ ETH Zurich',
    author_email='rauscho@ethz.ch',
    description='DaCe frontend for machine learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6, <3.10',
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_data={'': (['*.cpp'] + runtime_files)},
    install_requires=[
        'dace@git+https://github.com/spcl/dace.git@40d735',
        'onnx == 1.8.0',  # we support opset v12
        'torch',
        'protobuf == 3.19',
        'dataclasses; python_version < "3.7"',
        'onnx-simplifier == 0.3.10'
    ],
    # install with pip and --find-links (see Makefile)
    # See https://github.com/pypa/pip/issues/5898
    extras_require={
        'testing': [
            'coverage', 'pytest', 'yapf==0.31', 'pytest-cov', 'transformers',
            'pytest-xdist', 'torchvision', 'tabulate', 'efficientnet_pytorch',
            'pytest-timeout'
        ],
        'docs': [
            'sphinx', 'sphinx_rtd_theme', 'sphinx-autodoc-typehints',
            'sphinx-gallery', 'matplotlib'
        ],
    })
