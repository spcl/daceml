"""
Test that openblas is found and automatically selected on a matrix multiplication

If this fails, CPU CI will be very slow.
"""

import dace
import dace.libraries.blas as blas


def test_openblas_is_installed():
    assert blas.environments.OpenBLAS.is_installed()
