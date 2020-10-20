class AutoDiffException(Exception):
    """Base class for all exceptions related to automatic differentiation"""
    pass

from .autodiff import add_backward_pass



