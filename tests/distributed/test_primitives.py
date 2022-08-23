from typing import List, Sequence
import pytest
import numpy as np
import itertools

import dace
from daceml.distributed.communication.grid_mapped_array import ScatterOntoGrid, GatherFromGrid, AxisScheme, AxisType
from daceml.distributed import utils as distr_utils
from daceml.util import utils

fst = lambda x: x[0]
snd = lambda x: x[1]


def get_block(X: np.ndarray, coords: Sequence[int], grid_shape: Sequence[int],
              mapping: List[AxisScheme]) -> np.ndarray:
    """
    Get the block of X that would be associated with the given process grid
    coords if mapped to the grid with grid_shape and mapping.
    """

    # build up the subset of X
    subset = []
    for i, (grid_size, scheme) in enumerate(zip(grid_shape, mapping)):
        if scheme.scheme == AxisType.PARTITION:
            bs = X.shape[scheme.axis] // grid_size
            subset.append((scheme.axis, f"i{i}*{bs}:(i{i}+1)*{bs}"))
        elif scheme.scheme == AxisType.REPLICATE:
            subset.append((scheme.axis, f":"))
        elif scheme.scheme == AxisType.BROADCAST:
            # don't add anything, this is not an axis that exists in X
            pass

    subset_str = ",".join(map(snd, sorted(subset, key=fst)))

    index_subset_str = f"lambda arr, {', '.join('i' + str(i) for i in range(len(grid_shape)))}: arr[{subset_str}]"
    index_subset = eval(index_subset_str)

    return index_subset(X, *coords).copy()


def str_to_mapping(mapping: List[str]) -> List[AxisScheme]:
    def axis_type_from_string(t: str) -> AxisScheme:
        scheme, axis = t.split(":")
        axis = int(axis) if axis else None
        scheme = getattr(AxisType, scheme)
        return AxisScheme(scheme=scheme, axis=axis)

    return list(map(axis_type_from_string, mapping))


def compute_subarr_shape(global_shape: List[int], mapping: List[AxisScheme],
                         grid_shape: List[int]) -> List[int]:

    subshape = []

    for scheme, grid_size in zip(mapping, grid_shape):
        if scheme.scheme == AxisType.PARTITION:
            subshape.append(
                (scheme.axis, global_shape[scheme.axis] // grid_size))
        elif scheme.scheme == AxisType.REPLICATE:
            subshape.append((scheme.axis, global_shape[scheme.axis]))
        elif scheme.scheme == AxisType.BROADCAST:
            pass
        else:
            raise ValueError(f"Unknown scheme {scheme}")
    return list(map(snd, sorted(subshape, key=fst)))


@pytest.mark.parametrize(
    "grid_shape, array_shape, mapping",
    [
        # no broadcasting
        ([2, 1, 1], [4, 4, 4], ['PARTITION:0', 'PARTITION:1', 'PARTITION:2']),
        ([2, 2, 1], [4, 4, 4], ['PARTITION:1', 'PARTITION:2', 'PARTITION:0']),
        ([2, 1, 1], [4, 4, 4], ['PARTITION:0', 'PARTITION:2', 'PARTITION:1']),
        # replicating
        ([2, 1], [4, 4], ['PARTITION:1', 'REPLICATE:0']),
        ([2, 2], [4, 4], ['PARTITION:1', 'REPLICATE:0']),
        ([2, 2, 2], [4, 4], ['PARTITION:0', 'PARTITION:1', 'BROADCAST:']),
        ([2, 2, 2], [4, 4], ['REPLICATE:0', 'BROADCAST:', 'PARTITION:1']),
        ([2, 2], [4], ['REPLICATE:0', 'BROADCAST:']),
        ([2, 2], [4], ['BROADCAST:', 'PARTITION:0']),
    ])
def test_scatter(grid_shape, array_shape, mapping, sdfg_name):
    mapping = str_to_mapping(mapping)
    sdfg = dace.SDFG(sdfg_name)

    state = sdfg.add_state()

    subarr_shape = compute_subarr_shape(array_shape, mapping, grid_shape)
    sdfg.add_array("X", shape=array_shape, dtype=dace.int64)
    sdfg.add_array("__return", shape=subarr_shape, dtype=dace.int64)
    process_grid_name = sdfg.add_pgrid(grid_shape)
    distr_utils.initialize_fields(state, [
        f'MPI_Comm {process_grid_name}_comm;',
        f'MPI_Group {process_grid_name}_group;',
        f'int {process_grid_name}_coords[{len(grid_shape)}];',
        f'int {process_grid_name}_dims[{len(grid_shape)}];',
        f'int {process_grid_name}_rank;',
        f'int {process_grid_name}_size;',
        f'bool {process_grid_name}_valid;',
    ])

    node = ScatterOntoGrid(
        "scatter",
        grid_name=process_grid_name,
        axis_mapping=mapping,
    )
    node.add_in_connector("_inp_buffer")
    node.add_out_connector("_out_buffer")

    state.add_node(node)
    state.add_edge(state.add_read("X"), None, node, "_inp_buffer",
                   sdfg.make_array_memlet("X"))
    state.add_edge(node, "_out_buffer", state.add_write("__return"), None,
                   sdfg.make_array_memlet("__return"))

    # setup inputs
    X = distr_utils.arange_with_size(array_shape)

    rank_to_coords = itertools.product(*(range(i) for i in grid_shape))
    expected_outputs = [
        get_block(X, coords, grid_shape, mapping) for coords in rank_to_coords
    ]
    num_ranks = utils.prod(grid_shape)
    inputs = [
        dict(X=X) if i == 0 else dict(X=np.zeros_like(X, shape=(1, )))
        for i in range(num_ranks)
    ]

    distr_utils.compile_and_call(sdfg, inputs, expected_outputs, num_ranks)


@pytest.mark.parametrize(
    "grid_shape, global_array_shape, mapping",
    [
        # no reduction, just gather
        ([2, 1, 1], [4, 4, 4], ['PARTITION:0', 'PARTITION:1', 'PARTITION:2']),
        ([2, 2, 1], [4, 4, 4], ['PARTITION:1', 'PARTITION:2', 'PARTITION:0']),
        ([2, 1, 1], [4, 4, 4], ['PARTITION:0', 'PARTITION:2', 'PARTITION:1']),
        # reductions
        ([2, 1, 1], [4, 4], ['PARTITION:0', 'PARTITION:1', 'BROADCAST:']),
        ([2, 1, 1], [4, 4], ['PARTITION:1', 'PARTITION:0', 'BROADCAST:']),
        ([2, 2, 1], [4, 4], ['PARTITION:1', 'PARTITION:0', 'BROADCAST:']),
        ([2, 2, 1], [4, 4], ['BROADCAST:', 'PARTITION:1', 'PARTITION:0']),
        ([2, 2, 2], [4, 4], ['PARTITION:1', 'BROADCAST:', 'REPLICATE:0']),
    ])
def test_gather(grid_shape, global_array_shape, mapping, sdfg_name):
    sdfg = dace.SDFG(sdfg_name)
    mapping = str_to_mapping(mapping)

    state = sdfg.add_state()

    subarr_shape = compute_subarr_shape(global_array_shape, mapping,
                                        grid_shape)

    sdfg.add_array("X", shape=subarr_shape, dtype=dace.int64)
    sdfg.add_array("__return", shape=global_array_shape, dtype=dace.int64)

    process_grid_name = sdfg.add_pgrid(shape=grid_shape)
    distr_utils.initialize_fields(state, [
        f'MPI_Comm {process_grid_name}_comm;',
        f'MPI_Group {process_grid_name}_group;',
        f'int {process_grid_name}_coords[{len(grid_shape)}];',
        f'int {process_grid_name}_dims[{len(grid_shape)}];',
        f'int {process_grid_name}_rank;',
        f'int {process_grid_name}_size;',
        f'bool {process_grid_name}_valid;',
    ])

    node = GatherFromGrid(
        "gather",
        grid_name=process_grid_name,
        axis_mapping=mapping,
    )
    node.add_in_connector("_inp_buffer")
    node.add_out_connector("_out_buffer")

    state.add_node(node)
    state.add_edge(state.add_read("X"), None, node, "_inp_buffer",
                   sdfg.make_array_memlet("X"))
    state.add_edge(node, "_out_buffer", state.add_write("__return"), None,
                   sdfg.make_array_memlet("__return"))

    # setup inputs
    X = distr_utils.arange_with_size(global_array_shape)
    rank_to_coords = list(itertools.product(*(range(i) for i in grid_shape)))
    inputs = [{
        "X": get_block(X, coords, grid_shape, mapping)
    } for coords in rank_to_coords]
    for v in inputs:
        assert utils.all_equal(v["X"].shape, subarr_shape)

    # setup output
    order_by_first = lambda x: tuple(map(snd, sorted(x, key=fst)))

    # number of blocks, in the shape of the array
    grid_shape_reduced = tuple(
        (scheme.axis, dim_size if scheme.scheme == AxisType.PARTITION else 1)
        for scheme, dim_size in zip(mapping, grid_shape)
        if scheme.scheme != AxisType.BROADCAST)
    array_num_blocks = order_by_first(grid_shape_reduced)

    reduced_array = np.empty(array_num_blocks, dtype=np.ndarray)
    for coords in rank_to_coords:
        # get the block residing at these process grid coords
        block = get_block(X, coords, grid_shape, mapping)
        coords_to_keep = tuple(
            (s.axis, c if s.scheme == AxisType.PARTITION else 0)
            for c, s in zip(coords, mapping) if s.scheme != AxisType.BROADCAST)
        coords_to_keep = order_by_first(coords_to_keep)
        if reduced_array[coords_to_keep] is None:
            reduced_array[coords_to_keep] = block
        else:
            reduced_array[coords_to_keep] += block
    expected = np.block(reduced_array.tolist())
    assert utils.all_equal(expected.shape, global_array_shape)

    distr_utils.compile_and_call(sdfg, inputs, expected,
                                 utils.prod(grid_shape))
