"""
Runs the onnx backend tests for every operator that has a pure implementation

Resources:
https://github.com/onnx/onnx/blob/main/docs/ImplementingAnOnnxBackend.md
https://github.com/onnx/onnx-coreml/blob/master/tests/onnx_backend_node_test.py
"""

import inspect
import pytest

import onnx.backend.test

from daceml.onnx import DaCeMLBackend
from daceml.onnx.forward_implementation_abc import ONNXForward

ALL_PURE_OPS = set()
for impl, args in ONNXForward.extensions().items():
    if "op" in args:
        ALL_PURE_OPS.add(args["op"])


class DaCeMLPureBackend(DaCeMLBackend):
    @classmethod
    def is_compatible(cls, model, device='CPU', **kwargs):
        ops = {n.op_type for n in model.graph.node}
        # empty difference means all ops are compatible
        return not ops.difference(ALL_PURE_OPS)


# pytest magic to print report
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.runner.Runner(DaCeMLPureBackend, __name__)
backend_test.enable_report()

EXCLUDED = [
    'test_basic_conv_with_padding_cpu',
    'test_batchnorm_epsilon_cpu',
    'test_batchnorm_example_cpu',
    'test_cast_DOUBLE_to_FLOAT16_cpu',
    'test_cast_FLOAT16_to_DOUBLE_cpu',
    'test_cast_FLOAT16_to_FLOAT_cpu',
    'test_cast_FLOAT_to_FLOAT16_cpu',
    'test_cast_FLOAT_to_STRING_cpu',
    'test_cast_STRING_to_FLOAT_cpu',
    'test_clip_cpu',
    'test_clip_default_inbounds_cpu',
    'test_clip_default_int8_inbounds_cpu',
    'test_clip_default_int8_max_cpu',
    'test_clip_default_int8_min_cpu',
    'test_clip_default_max_cpu',
    'test_clip_default_min_cpu',
    'test_clip_example_cpu',
    'test_clip_inbounds_cpu',
    'test_clip_outbounds_cpu',
    'test_clip_splitbounds_cpu',
    'test_conv_with_strides_and_asymmetric_padding_cpu',
    'test_conv_with_strides_no_padding_cpu',
    'test_conv_with_strides_padding_cpu',
    'test_einsum_batch_diagonal_cpu',
    'test_einsum_inner_prod_cpu',
    'test_expand_dim_changed_cpu',
    'test_expand_dim_unchanged_cpu',
    'test_gather_negative_indices_cpu',
    'test_gemm_all_attributes_cpu',
    'test_gemm_alpha_cpu',
    'test_gemm_beta_cpu',
    'test_gemm_default_no_bias_cpu',
    'test_gemm_default_scalar_bias_cpu',
    'test_gemm_default_single_elem_vector_bias_cpu',
    'test_gemm_default_vector_bias_cpu',
    'test_gemm_default_zero_bias_cpu',
    'test_gemm_transposeA_cpu',
    'test_gemm_transposeB_cpu',
    'test_maxpool_1d_default_cpu',
    'test_maxpool_2d_ceil_cpu',
    'test_maxpool_2d_dilations_cpu',
    'test_maxpool_2d_pads_cpu',
    'test_maxpool_2d_precomputed_pads_cpu',
    'test_maxpool_2d_precomputed_same_upper_cpu',
    'test_maxpool_2d_same_lower_cpu',
    'test_maxpool_2d_same_upper_cpu',
    'test_maxpool_2d_uint8_cpu',
    'test_maxpool_3d_default_cpu',
    'test_maxpool_with_argmax_2d_precomputed_pads_cpu',
    'test_maxpool_with_argmax_2d_precomputed_strides_cpu',
    'test_slice_cpu',
    'test_slice_default_axes_cpu',
    'test_slice_default_steps_cpu',
    'test_slice_end_out_of_bounds_cpu',
    'test_slice_neg_cpu',
    'test_slice_neg_steps_cpu',
    'test_slice_negative_axes_cpu',
    'test_slice_start_out_of_bounds_cpu',
    'test_split_equal_parts_1d_cpu',
    'test_split_equal_parts_2d_cpu',
    'test_split_equal_parts_default_axis_cpu',
]
for test in EXCLUDED:
    backend_test.exclude(test)

cases = backend_test.test_cases['OnnxBackendNodeModelTest']
for name, func in inspect.getmembers(cases):
    if name.startswith("test"):
        setattr(cases, name, pytest.mark.onnx(func))
    if name.endswith("cuda"):
        setattr(cases, name, pytest.mark.gpu(func))
