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
                print(f'Setting lifetime from {arr.lifetime} to persistent: {dnode.data}')
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
                and len(node[0].map.params):
            if node[0].label in [
                # The entries that are commented out cause compile errors.
                # GCN
                'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_85_4_86',
                # '_Mult__map',

                # GAT
                # 'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_160_4_161',
                # 'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_207_4_208',
                'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_208_4_209',
                'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_209_4_210',
                'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_161_4_162',
                # '_Div__map',
                # 'assign_218_12_map',
                # '_Mult__map', # Is a 2-d map.
            ]:
                print("Changing schedule to TB dynamic: ", node[0].map)
                node[0].schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock_Dynamic
