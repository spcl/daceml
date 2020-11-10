class AutoDiffException(Exception):
    """ Base class for all exceptions related to automatic differentiation failures. """
    pass


from .autodiff import add_backward_pass
from .pytorch import make_backward_function
from .backward_implementation_abc import BackwardImplementation, BackwardContext, BackwardResult
from .backward_pass_generator import BackwardPassGenerator
