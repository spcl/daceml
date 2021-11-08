import os
import typing
import urllib.request, urllib.parse
import pathlib

import daceml
import numpy as np
import torch


def get_data_file(url, directory_name=None) -> str:
    """ Get a data file from ``url``, cache it locally and return the local file path to it.

        :param url: the url to download from.
        :param directory_name: an optional relative directory path where the file will be downloaded to.
        :returns: the path of the downloaded file.
    """

    data_directory = (pathlib.Path(daceml.__file__).parent.parent / 'tests' /
                      'data')

    if directory_name is not None:
        data_directory /= directory_name

    data_directory.mkdir(exist_ok=True, parents=True)

    file_name = os.path.basename(urllib.parse.urlparse(url).path)
    file_path = str(data_directory / file_name)

    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
    return file_path


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
