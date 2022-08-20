"""
Pytest plugin to mute all ranks that are not 0

Requires mpi4py
"""

import pytest

from mpi4py import MPI


def pytest_addoption(parser):
    parser.addoption("--unmute-all-ranks",
                     action="store_true",
                     help="Unmute all MPI ranks")


@pytest.mark.trylast
def pytest_configure(config):
    unmute = config.getoption("--unmute-all-ranks")
    if MPI.COMM_WORLD.Get_rank() != 0 and not unmute:
        # unregister the standard reporter
        standard_reporter = config.pluginmanager.getplugin('terminalreporter')
        config.pluginmanager.unregister(standard_reporter)
