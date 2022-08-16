Installation
============

DaCeML can be installed by using ``pip install git+https://github.com/spcl/daceml``. It is recommended to install the desired version of PyTorch first.

Alternatively, clone the repository and install using::

    VENV_PATH='' make install

See :ref:`dev` for more details on the ``Makefile``.

.. _ort-installation:

Installing ONNXRuntime
----------------------
Many ONNX operators include data-centric implementations that DaCeML uses for
lowering and analysis (:ref:`Pure Implementations`).
When such an implementation doesn't exist,
DaCeML can execute ONNX operators using `ONNXRuntime <https://github.com/microsoft/onnxruntime>`_.
To enable this, a
patched version [#f1]_ of ONNXRuntime needs to be installed and setup.

ONNXRuntime can be installed from source or from a prebuilt release.

Prebuilt Release (CPU only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The prebuilt release only supports executing ONNX operators with ONNXRuntime on the CPU execution provider.

.. note::
    The CPU-only restriction is not relevant for nodes which have pure SDFG implementations. Nodes with pure implementations can be executed on CPUs, GPUs and FPGAs.

To install, download the prebuilt release and extract it somewhere.

.. code-block:: bash

    wget https://github.com/orausch/onnxruntime/releases/download/v2/onnxruntime-daceml-patched.tgz
    tar -xzf onnxruntime-daceml-patched.tgz

Afterwards, ensure that the environment variable ``ORT_RELEASE`` points to the location of the extracted folder.

Build from Source
~~~~~~~~~~~~~~~~~
Clone the `patched onnxruntime <https://github.com/orausch/onnxruntime>`_ repository somewhere and build it using the following commands.

.. code-block:: bash

    git checkout master
    ./build.sh --build_shared_lib --parallel --config Release

To enable CUDA, add the relevant arguments. For instance::

     ./build.sh --use_cuda --cuda_version=10.1 --cuda_home=/usr/local/cuda --cudnn_home=/usr/local/cuda --build_shared_lib --parallel --config Release

See ``onnxruntime/BUILD.md`` for more details on building ONNXRuntime.

Afterwards, ensure that the environment variable ``ORT_ROOT`` points to the root of cloned repository.

.. [#f1] The patched version will be required until `#4453 <https://github.com/microsoft/onnxruntime/pull/4453>`_ has been merged and released.
