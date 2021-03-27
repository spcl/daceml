import pytest
import daceml.onnx as donnx
import sys

# the bert encoder is very nested, and exceeds the recursion limit while serializing
sys.setrecursionlimit(2000)


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="Run tests using gpu")
    parser.addoption("--gpu-only",
                     action="store_true",
                     help="Run tests using gpu, and skip CPU tests")


@pytest.fixture
def skip_non_gpu_test():
    pytest.skip('Skipping test since --gpu-only was passed')


@pytest.fixture
def skip_gpu_test_on_cpu():
    pytest.skip('Skipping test since --gpu or --gpu-only were not passed')


def pytest_generate_tests(metafunc):
    if "gpu" in (m.name for m in metafunc.definition.iter_markers()):
        if not (metafunc.config.getoption("--gpu")
                or metafunc.config.getoption("--gpu-only")):
            metafunc.fixturenames.insert(0, "skip_gpu_test_on_cpu")

    if "gpu" in metafunc.fixturenames:
        if metafunc.config.getoption("--gpu"):
            metafunc.parametrize("gpu", [True, False])
        elif metafunc.config.getoption("--gpu-only"):
            metafunc.parametrize("gpu", [True])
        else:
            metafunc.parametrize("gpu", [False])
    elif metafunc.config.getoption("--gpu-only"):
        metafunc.fixturenames.insert(0, "skip_non_gpu_test")

    if "default_implementation" in metafunc.fixturenames:
        metafunc.parametrize("default_implementation", [
            pytest.param("pure", marks=pytest.mark.pure),
            pytest.param("onnxruntime", marks=pytest.mark.ort)
        ])


@pytest.fixture
def sdfg_name(request):
    return request.node.name.replace("[", "-").replace("]",
                                                       "").replace("-", "_")


@pytest.fixture(autouse=True)
def setup_default_implementation(request):
    # this fixture is used for all tests (autouse)

    old_default = donnx.default_implementation

    pure_marker = request.node.get_closest_marker("pure")
    ort_marker = request.node.get_closest_marker("ort")

    if pure_marker is not None:
        donnx.default_implementation = "pure"
        yield

    if ort_marker is not None:
        donnx.default_implementation = "onnxruntime"
        yield

    if ort_marker is None and pure_marker is None:
        yield

    donnx.default_implementation = old_default
