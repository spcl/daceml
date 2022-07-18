import dace
import torch
from dace.library import change_default
from dace.transformation.auto.auto_optimize import auto_optimize as dace_auto_optimize
from dace.transformation.dataflow import TrivialMapRangeElimination, MapFusion

from daceml import onnx as donnx
from daceml.transformation import TaskletFusion
from daceml.util import utils


def _specialize_memory(sdfg):
    from dace.sdfg.scope import is_devicelevel_gpu

    # Make memory persistent
    for state in sdfg.nodes():
        for dnode in state.data_nodes():
            if is_devicelevel_gpu(sdfg, state, dnode):
                continue
            if 'reserved_' in dnode.data:
                continue
            arr = sdfg.arrays[dnode.data]
            if (arr.transient and not isinstance(arr, dace.data.View)
                    and arr.storage != dace.StorageType.Register
                    and dnode.data != 'features'):
                print(dnode.data)
                arr.lifetime = dace.AllocationLifetime.Persistent

    # Disable OpenMP sections
    sdfg.openmp_sections = False


def specialize_mem_onnx(mod):
    def spec(module):
        for sd in module.sdfg.all_sdfgs_recursive():
            _specialize_memory(sd)

    mod.append_post_onnx_hook("specializemem", spec)


def apply_dace_auto_optimize(module):
    sdfg = module.sdfg
    dace_auto_optimize(sdfg,
                       device=dace.dtypes.DeviceType.GPU if torch.cuda.is_available() else dace.dtypes.DeviceType.CPU)


def opitmize(model):
    # reset the compiled sdfg
    model.reset_sdfg()

    # expand the onnx nodes, and apply automatic transformations like inlining
    def expand_and_strict_transforms(module):
        # use the pure expansions of operators
        with change_default(donnx, "pure"):
            utils.auto_optimize(module.sdfg, cuda=True)

    model.append_post_onnx_hook("auto_optimize", expand_and_strict_transforms)

    # # apply subgraph fusion
    def fuse_sg(module):
        sdfg = module.sdfg
        sdfg.apply_transformations_repeated(TrivialMapRangeElimination)
        sdfg.apply_transformations_repeated(MapFusion)
        # SubgraphFusion.apply_to(sdfg, *sdfg.node(0).nodes())

    model.append_post_onnx_hook("subgraph_fusion", fuse_sg)
    #
    # apply tasklet fusion
    model.append_post_onnx_hook("fuse_tasklets", lambda x:
    x.dace_model.sdfg.apply_transformations_repeated(TaskletFusion))
    #
    # # apply vectorization    # def set_transients_to_persistent(module):
    #     sdfg = module.sdfg
    #     for _, name, desc in sdfg.arrays_recursive():
    #         if isinstance(desc, dace.data.Array) and desc.transient:
    #             print(name, desc.shape, desc.lifetime)
    #             desc.lifetime = dace.AllocationLifetime.Persistent
    #
    # dace_model.append_post_onnx_hook("set_transients_to_persistent", set_transients_to_persistent)
    # def vectorize(module):
    #     module.sdfg.apply_transformations(Vectorization)
    #
    # model.append_post_onnx_hook("vectorize", vectorize)
