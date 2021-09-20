import numpy as np
import torch
import typing


def torch_tensors_close(name,
                        torch_v,
                        dace_v,
                        rtol=1e-5,
                        atol=1e-4,
                        summary=False,
                        raise_on_fail=True):
    """ Assert that the two torch tensors are close. Prints a nice error string if not.
    """
    if not torch.allclose(
            torch_v, dace_v, rtol=rtol, atol=atol, equal_nan=True):
        print()
        print(name + " was not close")
        if summary:
            torch_v = torch_v.detach().cpu().numpy()
            dace_v = dace_v.detach().cpu().numpy()
            max_error_idx = np.argmax(torch_v - dace_v)
            max_error = np.max(torch_v - dace_v)
            failed_mask = np.abs(torch_v -
                                 dace_v) > atol + rtol * np.abs(dace_v)
            print(f"Num wrong: {failed_mask.sum()}")
            print(
                f"Maximum error: {max_error}, Torch value: {torch_v.flatten()[max_error_idx]},"
                f" Dace value: {dace_v.flatten()[max_error_idx]}")
        else:
            print("torch value: ", torch_v)
            print("dace value: ", dace_v)
            print("diff: ", torch.abs(dace_v - torch_v))

            torch_v = torch_v.detach().cpu().numpy()
            dace_v = dace_v.detach().cpu().numpy()
            failed_mask = np.abs(torch_v -
                                 dace_v) > atol + rtol * np.abs(dace_v)
            print(f"wrong elements torch: {torch_v[failed_mask]}")
            print(f"wrong elements dace: {dace_v[failed_mask]}")

            for x, y, _ in zip(torch_v[failed_mask], dace_v[failed_mask],
                               range(100)):
                print(f"lhs_failed: {abs(x - y)}")
                print(
                    f"rhs_failed: {atol} + {rtol * abs(y)} = {atol + rtol * abs(y)}"
                )

        if raise_on_fail:
            assert False, f"{name} was not close)"


T = typing.TypeVar("T")


def copy_to_gpu(gpu: bool, tensor: T) -> T:
    if gpu:
        return tensor.cuda()
    else:
        return tensor
