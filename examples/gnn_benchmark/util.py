import dace
import torch
from dace.transformation.auto.auto_optimize import auto_optimize as dace_auto_optimize


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
                print('Setting lifetime to persistent: ', dnode.data)
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
                and node[0].schedule == dace.dtypes.ScheduleType.Sequential \
                and len(node[0].map.params) == 1 \
                and node[0].label in [
            # 'assign_167_16_map',
            'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_155_4_156',
            'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_154_4_155',
            'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_157_4_158',
            'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_153_4_154',
            'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_69_4_70',
            'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_69_4_70',
            # '_Mult__map',
            # 'assign_160_16_map',
        ]:
            print("Changing schdeule to TB dynamic: ", node[0].map)
            node[0].schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock_Dynamic
