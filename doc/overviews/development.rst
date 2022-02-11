.. _dev:

Development
===========
The ``Makefile`` contains a few commands for development tasks such as running tests, checking formatting or installing the package.

For example, the following command would install the package and run tests::

        VENV_PATH='' make install test

If you would like to create a virtual environment and install to it, remove `VENV_PATH=''` from the above command.

Specific Package Versions
-------------------------
The `DACE_VERSION` and `TORCH_VERSION` variables can be used to install specific versions of those packages over the
recommended ones. For example, you can use a local dace repository using::

        DACE_VERSION='-e /path/to/dace/' make clean install

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
    This runs the formatting checks. The DaCeML codebase is formatted using ``yapf``. Use ``check-formatting-names`` to
    only print the names of the misformatted files.

Testing
-------
DaCeML uses ``pytest`` to run tests.
Tests can be parallelized using ``xdist`` by passing the arguments ``-n auto --dist loadfile``.

Fixtures and Arguments
~~~~~~~~~~~~~~~~~~~~~~
The pytest runner takes several custom arguments: ``--gpu``, ``--gpu-only``,
``--skip-cpu-blas``.

There are also a few useful fixtures in use. For example the ``gpu`` fixture
parameterizes the test to run twice, once with ``True`` and once with
``False``. The fixture is written to interact correctly with the pytest
arguments like ``--gpu``.

There are some subtle, but hopefully intutitive interactions between the
fixtures and the arguments.
See `test_fixtures.py
<https://github.com/spcl/daceml/blob/master/tests/test_fixtures.py>`_ to
understand the expected behavior, and
`conftest.py <https://github.com/spcl/daceml/blob/master/tests/conftest.py>`_ for their implementation.

Setting the default implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Nodes can be expanded to different implementations (See :ref:`node_implementations`). To control the default
implementation that is used, tests can be decorated with the following two markers.

``pytest.mark.ort``
    Use the ONNXRuntime expansion as default

``pytest.mark.pure``
    Use the pure expansion as default when possible (falling back to ONNXRuntime)

If you provide the fixture (i.e., an argument to the test) with name ``default_implementation``, then the test will be
parameterized to test both implementations.

Useful Snippets
---------------

GPU Leak checker
~~~~~~~~~~~~~~~~
Put this code in ``tests/conf.py``::

    import xml.etree.ElementTree as ET
    import subprocess
    import shlex

    import torch
    # initialize torch cuda context
    a = torch.ones(1, 1).cuda()


    def _get_gpu_mem_usage():
        result = subprocess.check_output(shlex.split("nvidia-smi -x -q"))
        usage_str = ET.fromstring(result).find("gpu").find("fb_memory_usage").find("used").text
        if not usage_str.endswith(" MiB"):
            raise RuntimeError("Couldn't parse nvidia-smi output")

        return int(usage_str[:-4])

    @pytest.fixture(autouse=True)
    def memory_printer():
        before = _get_gpu_mem_usage()
        log.debug(f"Usage before: {before}")
        yield
        after = _get_gpu_mem_usage()
        log.debug(f"Usage after: {after}, delta: {after - before}")
        assert after - before < 200
