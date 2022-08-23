import dace.library

from .node import DistributedMemlet, FULLY_REPLICATED_RANK
from . import grid_mapped_array

dace.library.register_library(__name__, "communication")
