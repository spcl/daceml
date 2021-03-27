import os
from setuptools import setup

with open("README.md", "r") as fp:
    long_description = fp.read()

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
setup(
    name='daceml',
    version='0.1.0a',
    url='https://github.com/spcl/dace-onnx',
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
    python_requires='>=3.6',
    packages=['daceml'],
    package_data={'': ['*.cpp']},
    install_requires=[
        'dace@git+https://github.com/orausch/dace.git@dml_param_to_trans',
        'onnx == 1.7.0', 'torch', 'dataclasses; python_version < "3.7"'
    ],
    # install with pip and --find-links (see Makefile)
    # See https://github.com/pypa/pip/issues/5898
    extras_require={
        'testing': [
            'coverage', 'pytest', 'yapf', 'pytest-cov', 'transformers',
            'pytest-xdist', 'torchvision'
        ],
        'docs': [
            'sphinx==3.2.1', 'sphinx_rtd_theme==0.5.0',
            'sphinx-autodoc-typehints==1.11.1'
        ],
        'debug': ['onnxruntime']
    })
