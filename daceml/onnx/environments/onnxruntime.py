import os
import logging

import dace.library
from dace.config import Config

log = logging.getLogger(__name__)

if 'ORT_ROOT' not in os.environ and 'ORT_RELEASE' not in os.environ:
    raise ValueError("This environment expects the environment variable "
                     "ORT_ROOT or ORT_RELEASE to be set (see README.md)")

if Config.get("compiler", "cuda", "max_concurrent_streams") != -1:
    log.info("Setting compiler.cuda.max_concurrent_streams to -1")
    Config.set("compiler", "cuda", "max_concurrent_streams", value=-1)


def _get_src_includes():
    """
    Get the includes and dll path when ORT is built from source
    """
    ort_path = os.path.abspath(os.environ['ORT_ROOT'])
    cand_path = os.path.join(ort_path, "build", "Linux",
                             dace.Config.get("compiler", "build_type"))

    if os.path.isdir(cand_path):
        ort_build_path = cand_path
    else:
        ort_build_path = os.path.join(ort_path, "build", "Linux", "Release")

    ort_dll_path = os.path.join(ort_build_path, "libonnxruntime.so")
    includes = [
        os.path.join(ort_path, "include", "onnxruntime", "core", "session"),
        os.path.join(ort_path, "include", "onnxruntime", "core", "providers",
                     "cpu"),
        os.path.join(ort_path, "include", "onnxruntime", "core", "providers",
                     "cuda")
    ]
    return includes, ort_dll_path


def _get_dist_includes():
    """
    Get the includes and dll path when ORT is used from the distribution package
    """
    ort_path = os.path.abspath(os.environ['ORT_RELEASE'])
    includes = [os.path.join(ort_path, 'include')]
    ort_dll_path = os.path.join(ort_path, 'lib', 'libonnxruntime.so')
    return includes, ort_dll_path


if 'ORT_RELEASE' in os.environ:
    log.debug("Using ORT_RELEASE")
    INCLUDES, ORT_DLL_PATH = _get_dist_includes()
else:
    log.debug("Using ORT_ROOT")
    INCLUDES, ORT_DLL_PATH = _get_src_includes()


@dace.library.environment
class ONNXRuntime:
    """ Environment used to run ONNX operator nodes using ONNX Runtime. This environment expects the environment variable
        ``ORT_ROOT`` to be set to the root of the patched onnxruntime repository (https://github.com/orausch/onnxruntime)

        Furthermore, both the runtime and the protobuf shared libs should be built:

        ``./build.sh --build_shared_lib --parallel --config Release``
        ``mkdir build-protobuf && cd build-protobuf && cmake ../cmake/external/protobuf/cmake -Dprotobuf_BUILD_SHARED_LIBS=ON && make``

        (add ``-jN`` to the make command for parallel builds)
        See ``onnxruntime/BUILD.md`` for more details.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = INCLUDES
    cmake_libraries = [ORT_DLL_PATH]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = [
        "../include/dace_onnx.h",
        "onnxruntime_c_api.h",
        "cpu_provider_factory.h",
        "cuda_provider_factory.h",
    ]
    init_code = ""
    finalize_code = ""
