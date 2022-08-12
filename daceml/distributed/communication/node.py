
import dace
import dace.library
from dace import nodes, properties, symbolic, SDFG, SDFGState, subsets

from . import sendrecv
from . import subarrays
from . import solver

from daceml.util import utils


#: a placeholder variable that is used for all fully replicated ranks
FULLY_REPLICATED_RANK ='FULLY_REPLICATED_RANK'

@dace.library.node
class DistributedMemlet(nodes.LibraryNode):
    """
    Communication node that distributes input to output based on symbolic
    expressions for the rank-local subsets.

    These expressions (``src_subset`` and ``dst_subset``) should include
    symbolic variables given in ``rank_variables``.
    """

    # Global properties
    implementations = {
        "subarrays": subarrays.CommunicateSubArrays,
    }
    default_implementation = "subarrays"

    src_rank_variables = properties.ListProperty(
        element_type=str,
        desc="List of variables used in the indexing expressions that represent rank identifiers in the source expression"
    )
    src_pgrid = properties.Property(dtype=str, allow_none=True)
    src_subset = properties.RangeProperty(
        default=subsets.Range([]),
        desc=
        "Subset of the input array that is held on each rank"
    )
    src_global_array = properties.DataProperty()

    dst_rank_variables = properties.ListProperty(
        element_type=str,
        desc="List of variables used in the indexing expressions that represent rank identifiers in the destination expression"
    )
    dst_pgrid = properties.Property(dtype=str, allow_none=True)
    dst_subset = properties.RangeProperty(
        default=subsets.Range([]),
        desc=
        "Subset of the output array that is held on each rank"
    )
    dst_global_array = properties.DataProperty()

    def __init__(self, name, src_rank_variables, src_pgrid, src_subset, src_global_array, dst_rank_variables, dst_pgrid, dst_subset, dst_global_array):
        super().__init__(name)
        self.src_rank_variables = src_rank_variables
        self.src_pgrid = src_pgrid
        self.src_subset = src_subset
        self.src_global_array = src_global_array
        self.dst_rank_variables = dst_rank_variables
        self.dst_pgrid = dst_pgrid
        self.dst_subset = dst_subset
        self.dst_global_array = dst_global_array


    def validate(self, sdfg: SDFG, state: SDFGState):

        if self.src_global_array not in sdfg.arrays:
            raise ValueError(f"{self.src_global_array} is not an array in the SDFG")
        if self.dst_global_array not in sdfg.arrays:
            raise ValueError(f"{self.dst_global_array} is not an array in the SDFG")

        src_free_vars = self.src_subset.free_symbols
        dst_free_vars = self.dst_subset.free_symbols

        if self.src_pgrid is None and self.dst_pgrid is None:
            raise ValueError("At least one process grid must be specified")

        if src_free_vars.difference(self.src_rank_variables):
            raise ValueError("Source subset has free variables that are not rank variables")
        if dst_free_vars.difference(self.dst_rank_variables):
            raise ValueError("Destination subset has free variables that are not rank variables")

        if FULLY_REPLICATED_RANK in src_free_vars or FULLY_REPLICATED_RANK in dst_free_vars:
            raise RuntimeError("Fully replicated rank appeared in free variables, this should not happen")


        inp_buffer, out_buffer = None, None

        if state.out_degree(self) != 1:
            raise ValueError("SymbolicCommunication node must have exactly one output edge")
        if state.in_degree(self) != 1:
            raise ValueError("SymbolicCommunication node must have exactly one input edge")
        out_buffer = sdfg.arrays[state.out_edges(self)[0].data.data]
        inp_buffer = sdfg.arrays[state.in_edges(self)[0].data.data]
        if inp_buffer.dtype != out_buffer.dtype:
            raise ValueError("Input and output buffers must have the same data type")


        # Check that subset sizes are correct
        if not utils.all_equal(self.src_subset.size_exact(), inp_buffer.shape):
            raise ValueError(f"Source subset size {self.src_subset.size_exact()} does not match input buffer size {inp_buffer.shape}")
        if not utils.all_equal(self.dst_subset.size_exact(), out_buffer.shape):
            raise ValueError(f"Destination subset size {self.dst_subset.size_exact()} does not match output buffer size {out_buffer.shape}")


        # Check process grids
        if self.src_pgrid and self.src_pgrid not in sdfg.process_grids:
            raise ValueError("Source process grid not found")
        if self.dst_pgrid and self.dst_pgrid not in sdfg.process_grids:
            raise ValueError("Destination process grid not found")


        return inp_buffer, out_buffer
    
    def __str__(self):
        return f"{self.src_global_array}[{self.src_subset}] -> {self.dst_global_array}[{self.dst_subset}]"

