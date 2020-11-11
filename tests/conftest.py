import pytest
import daceml.onnx as donnx


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="Run tests using gpu")
    parser.addoption("--gpu-only",
                     action="store_true",
                     help="Run tests using gpu, and skip CPU tests")


def pytest_generate_tests(metafunc):
    if "gpu" in metafunc.fixturenames:
        if metafunc.config.getoption("--gpu"):
            metafunc.parametrize("gpu", [True, False])
        elif metafunc.config.getoption("--gpu-only"):
            metafunc.parametrize("gpu", [True])
        else:
            metafunc.parametrize("gpu", [False])
    if "default_implementation" in metafunc.fixturenames:
        metafunc.parametrize("default_implementation", [
            pytest.param("pure", marks=pytest.mark.pure),
            pytest.param("onnxruntime", marks=pytest.mark.pure)
        ])


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
