"""
Profile Bandwidth across storage types
"""
import copy
import argparse 
from typing import List, Tuple
import statistics

import numpy as np

import dace
from dace.transformation import interstate

# these will be measured using a copy
PAIRS_COPY = [
    # GPU_Global
    (dace.StorageType.GPU_Global, dace.StorageType.GPU_Global),
    (dace.StorageType.Default, dace.StorageType.GPU_Global),
    (dace.StorageType.GPU_Global, dace.StorageType.Default),
    (dace.StorageType.GPU_Global, dace.StorageType.GPU_Shared),
    (dace.StorageType.GPU_Shared, dace.StorageType.GPU_Global),
]

# these will be measured using a kernel
PAIRS_REGISTER = [
    (dace.StorageType.GPU_Global, dace.StorageType.Register),
    (dace.StorageType.Register, dace.StorageType.GPU_Global),
    # GPU_Shared
    (dace.StorageType.GPU_Shared, dace.StorageType.Register),
    (dace.StorageType.Register, dace.StorageType.GPU_Shared),
]


def sizeof_fmt(num, suffix="B"):
    """
    From https://stackoverflow.com/a/1094933
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

RESULT_T = List[Tuple[int, float]]


def run_for_sizes(sdfg: dace.SDFG) -> RESULT_T:
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.NestedSDFG):
            n.sdfg.node(0).instrument = dace.InstrumentationType.GPU_Events

    num_elements = [2**i for i in range(16, 25, 4)]

    results = []

    for n in num_elements:
        specialized_sdfg = copy.deepcopy(sdfg)
        specialized_sdfg.name = f"{specialized_sdfg.name}_{n}"
        specialized_sdfg.replace('N', n)
        compiled_sdfg = specialized_sdfg.compile()

        A = np.random.rand(n).astype(np.float32)
        B = np.random.rand(n).astype(np.float32)
        compiled_sdfg(A=A, B=B,N=n, T=1000)
        def get_first(d: dict):
            assert len(d.keys()) == 1
            key = next(iter(d.keys()))
            return d[key]
        report = compiled_sdfg.sdfg.get_latest_report()
        times = get_first(get_first(report.durations))
        median = statistics.median(times)
        results.append((n, median))
        bandwidth = (n * 4) / median
        print(f"{n} -> \t{sizeof_fmt(bandwidth)}/s")
    return results

def profile_register(other: dace.StorageType) -> Tuple[RESULT_T, RESULT_T]:
    """
    Profile data movement bandwidth to and from a register.
    :param other: the source or destination storage type
    :returns: results for read and write as a tuple
    """
    N = dace.symbol("N") 
    T = dace.symbol("T") 

    @dace.program
    def write(B: dace.float32[N]):
        B[:] = 0

    write_sdfg = write.to_sdfg()
    write_sdfg.arrays["B"].storage = other
    write_sdfg.apply_gpu_transformations()

    @dace.program
    def profile_write():
        B_storage = dace.define_local([N], dtype=dace.float32)
        for i in range(T):
            write_sdfg(B=B_storage)

    sdfg = profile_write.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(interstate.StateFusion)
    sdfg.arrays["B_storage"].storage = other

    print("WRITE")
    result_write = run_for_sizes(sdfg)
    return (result_write, result_write)




def profile_copy(src: dace.StorageType, dst: dace.StorageType) -> RESULT_T:
    """
    Profile the bandwidth between two storage types using a copy edge
    :returns: bandwith in bytes/s, for different num_elements
    """

    N = dace.symbol("N") 
    T = dace.symbol("T") 

    @dace.program
    def copy(A: dace.float32[N], B: dace.float32[N]):
        B[:] = A
    
    copy_sdfg = copy.to_sdfg()
    copy_sdfg.arrays["A"].storage = src
    copy_sdfg.arrays["B"].storage = dst


    @dace.program
    def profile_copy(A: dace.float32[N], B: dace.float32[N]):
        A_storage = dace.define_local([N], dtype=dace.float32)
        B_storage = dace.define_local([N], dtype=dace.float32)
        A_storage[:] = A
        for i in range(T):
            copy_sdfg(A_storage, B_storage)
        B[:] = B_storage

    sdfg = profile_copy.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(interstate.StateFusion)
    sdfg.arrays["A_storage"].storage = src
    sdfg.arrays["B_storage"].storage = src

    return run_for_sizes(sdfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str)
    parser.add_argument("dst", type=str)
    args = parser.parse_args()

    src = dace.StorageType[args.src]
    dst = dace.StorageType[args.dst]
    if (src, dst) in PAIRS_COPY:
        print("Profiling using copy")
        profile_copy(src, dst)         
    elif (src, dst) in PAIRS_REGISTER:
        print("Profiling using register")
        if src is dace.StorageType.Register:
            profile_register(dst)
        elif dst is dace.StorageType.Register:
            profile_register(src)
        else:
            raise ValueError("Can only profile register to register")
    else:
        raise ValueError("Unsupported pair")
