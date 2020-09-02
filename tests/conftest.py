def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="Run tests using gpu")


def pytest_generate_tests(metafunc):
    if "gpu" in metafunc.fixturenames:
        if metafunc.config.getoption("--gpu"):
            metafunc.parametrize("gpu", [True, False])
        else:
            metafunc.parametrize("gpu", [False])
