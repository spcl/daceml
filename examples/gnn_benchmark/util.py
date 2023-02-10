import logging

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


def make_maps_dynamic(module, exclude_loops=None):
    sdfg = module.sdfg
    # Count which loops were excluded to be able to produce a warning
    # in case a loop was not found in the graph.
    exclude_loops = {name: 0 for name in exclude_loops} or {}
    for node in sdfg.all_nodes_recursive():
        if isinstance(node[0], dace.sdfg.nodes.MapEntry) \
                and node[0].schedule == dace.dtypes.ScheduleType.Sequential \
                and len(node[0].map.params):
            if node[0].label not in exclude_loops:
                print("Changing schedule to TB dynamic: ", node[0].map)
                node[0].schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock_Dynamic
            else:
                exclude_loops[node[0].label] += 1
                print("Keeping schedule sequential for ", node[0].map)

    not_excluded = [name for name, count in exclude_loops.items() if count == 0]
    if not_excluded:
        logging.warning("Following loops were marked as excluded from thread-block dynamic scheduling "
                        "but were not found in the SDFG: %s", not_excluded)
