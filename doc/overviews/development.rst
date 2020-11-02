.. _dev:

Development
===========
The ``Makefile`` contains a few commands for development tasks such as running tests, checking formatting or installing the package.

For example, the following command would install the package and run tests::

        VENV_PATH='' make install test

If you would like to create a virtual environment and install to it, remove `VENV_PATH=''` from the above command.

Makefile Targets
----------------
The CI runs several tests using the ``Makefile``:

``make test``, ``make test-parallel`` & ``make test-gpu``
    Run pytest on the ``tests/`` directory. The pytest runner takes a custom argument ``--gpu`` to run GPU tests.
    CPU tests can be run in parallel using the ``test-parallel`` target.

``make doctest``
    Run doctests; this executes the code samples in the documentation and docstrings.

``make doc``
    Build the documentation.

``make check-formatting``
    This runs the formatting checks. The DaceML codebase is formatted using ``yapf``.
