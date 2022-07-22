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
                    and arr.storage != dace.StorageType.Register):
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


def make_maps_dynamic(module):
    sdfg = module.sdfg
    for node in sdfg.all_nodes_recursive():
        if isinstance(node[0], dace.sdfg.nodes.MapEntry) \
                and node[0].schedule == dace.dtypes.ScheduleType.Sequential\
                and len(node[0].map.params) == 1\
                and node[0].label in ['assign_167_16_map', 'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_161_4_162']:
            print("Changing schdeule to TB dynamic: ", node[0].map)
            node[0].schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock_Dynamic
