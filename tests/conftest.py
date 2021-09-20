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


def pytest_runtest_setup(item):
    # if @pytest.mark.gpu is applied skip the test on CPU
    if "gpu" in (m.name for m in item.iter_markers()):
        if not (item.config.getoption("--gpu")
                or item.config.getoption("--gpu-only")):
            pytest.skip(
                'Skipping test since --gpu or --gpu-only were not passed')
    # else: if the gpu fixture is used, the test is parameterized: don't skip on CPU
    elif "gpu" in item.fixturenames:
        pass
    # otherwise: this test is not marked with @pytest.mark.gpu, and doesn't have the gpu fixture:
    # skip it if --gpu-only is passed
    elif item.config.getoption("--gpu-only"):
        pytest.skip('Skipping test since --gpu-only was passed')


def pytest_generate_tests(metafunc):
    if "gpu" in metafunc.fixturenames:
        if metafunc.config.getoption("--gpu"):
            metafunc.parametrize("gpu", [
                pytest.param(True, id="use_gpu"),
                pytest.param(False, id="use_cpu")
            ])
        elif metafunc.config.getoption("--gpu-only"):
            metafunc.parametrize("gpu", [pytest.param(True, id="use_gpu")])
        else:
            metafunc.parametrize("gpu", [pytest.param(False, id="use_cpu")])

    if "default_implementation" in metafunc.fixturenames:
        metafunc.parametrize("default_implementation", [
            pytest.param("pure", marks=pytest.mark.pure),
            pytest.param("onnxruntime", marks=pytest.mark.ort)
        ])

    if "use_cpp_dispatcher" in metafunc.fixturenames:
        metafunc.parametrize("use_cpp_dispatcher", [
            pytest.param(True, id="use_cpp_dispatcher"),
            pytest.param(False, id="no_use_cpp_dispatcher"),
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
