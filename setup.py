import os
from setuptools import setup, find_packages

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
        # from https://github.com/pypa/pip/issues/6658#issuecomment-506841157
        'dace@file://localhost' + PKG_DIR + "/dace",
        'onnx == 1.7.0',
        'numba'
    ],
    extras_require={'testing': ['coverage', 'pytest', 'yapf']})
