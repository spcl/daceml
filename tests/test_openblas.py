"""
Test that openblas is found and automatically selected on a matrix multiplication

If this fails, CPU CI will be very slow.
"""

import pytest
import dace
import dace.libraries.blas as blas


@pytest.mark.cpublas
def test_openblas_is_installed():
    assert blas.environments.OpenBLAS.is_installed()
