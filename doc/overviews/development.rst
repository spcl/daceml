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
    Run pytest on the ``tests/`` directory. CPU tests can be run in parallel using the ``test-parallel`` target.

``make doctest``
    Run doctests; this executes the code samples in the documentation and docstrings.

``make doc``
    Build the documentation.

``make check-formatting``
    This runs the formatting checks. The DaceML codebase is formatted using ``yapf``. Use ``check-formatting-names`` to
    only print the names of the misformatted files.

Testing
-------
DaceML uses ``pytest`` to run tests. The pytest runner takes a custom argument ``--gpu`` to run GPU tests.
Tests can be parallelized using ``xdist`` by passing the arguments ``-n auto --dist loadfile``.

Some tests are marked with ``pytest.mark.slow``. Slow tests are skipped on Travis CI (see ``.travis.yml``).

If you provide the fixture (i.e. an argument to the test) with name ``gpu``, then the test will be parameterized to pass
both ``True`` and ``False`` to that argument.

Setting the default implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Nodes can be expanded to different implementations. To control the default implementation that is used, tests can be
decorated with the following two markers.

``pytest.mark.ort``
    Use the ONNXRuntime expansion as default

``pytest.mark.pure``
    Use the pure expansion as default when possible (fallback is always ORT)

If you provide the fixture (i.e. an argument to the test) with name ``default_implementation``, then the test will be
parameterized to test both implementations.
