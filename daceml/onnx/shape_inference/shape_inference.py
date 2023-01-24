import functools

from onnx import helper

from daceml.onnx.shape_inference.symbolic_shape_infer import SymbolicShapeInference


def _compute_matmul_shape(self: SymbolicShapeInference, node, output_dtype=None, rhs_shape=None):
    """Slightly modified function from SymbolicShapeInference that
    allows supplying rhs_shape (instead of getting it from a node)."""
    lhs_shape = self._get_shape(node, 0)
    if rhs_shape is None:
        rhs_shape = self._get_shape(node, 1)
    lhs_rank = len(lhs_shape)
    rhs_rank = len(rhs_shape)
    lhs_reduce_dim = 0
    rhs_reduce_dim = 0
    assert lhs_rank > 0 and rhs_rank > 0
    if lhs_rank == 1 and rhs_rank == 1:
        new_shape = []
    elif lhs_rank == 1:
        rhs_reduce_dim = -2
        new_shape = rhs_shape[:rhs_reduce_dim] + [rhs_shape[-1]]
    elif rhs_rank == 1:
        lhs_reduce_dim = -1
        new_shape = lhs_shape[:lhs_reduce_dim]
    else:
        lhs_reduce_dim = -1
        rhs_reduce_dim = -2
        new_shape = self._broadcast_shapes(lhs_shape[:-2], rhs_shape[:-2]) + [lhs_shape[-2]] + [rhs_shape[-1]]
    # merge reduce dim
    self._check_merged_dims([lhs_shape[lhs_reduce_dim], rhs_shape[rhs_reduce_dim]], allow_broadcast=False)
    if output_dtype is None:
        # infer output_dtype from input type when not specified
        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
    vi = self.known_vi_[node.output[0]]
    vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_shape))

# Overwrite _compute_matul_shape with the modified version.
SymbolicShapeInference._compute_matmul_shape = _compute_matmul_shape

def make_init_fn(original_init, placeholder_id_to_module):
    """Initializes the SSI object and adds the inference functions for the replaced modules."""
    def new_init(self, int_max, auto_merge, guess_output_rank, verbose):
        original_init(self, int_max, auto_merge, guess_output_rank, verbose)
        from daceml.onnx import MODULES_TO_REPLACE
        for module_name, replacement_info in MODULES_TO_REPLACE.items():
            infer_fn = replacement_info.infer_shape
            self.dispatcher_[module_name] = functools.partial(infer_fn, placeholder_id_to_module, self)

    return new_init


def infer_shapes(onnx_model, placeholder_id_to_module, auto_merge=False):
    # Override _compute_matmul_shape and the init functions with modified versions that handle module replacements.
    original_init_fn = SymbolicShapeInference.__init__
    SymbolicShapeInference.__init__ = make_init_fn(original_init_fn, placeholder_id_to_module)

    result = SymbolicShapeInference.infer_shapes(onnx_model, auto_merge=auto_merge)

    # Need to revert to be able to run twice.
    SymbolicShapeInference.__init__ = original_init_fn

    if result is None:
        raise ValueError("Symbolic shape inference failed")
    return result
