import dace.library

from . import library
from . import python_frontend
from . import torch_integration

dace.library.register_library(__name__, "autodiff")
