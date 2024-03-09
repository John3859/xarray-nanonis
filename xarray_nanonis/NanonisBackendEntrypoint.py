"""
Adding Nanonis backend for reading support to Xarray
"""

__all__ = ["NanonisBackendEntrypoint"]

from xarray.backends import BackendEntrypoint
from xarray_nanonis.nanonis import Read_NanonisFile
import os


class NanonisBackendEntrypoint(BackendEntrypoint):
    def open_dataset(self, filename_or_obj, *, drop_variables=None, **kwargs):
        divider = kwargs.pop("divider", 1)
        if kwargs:
            raise KeyError(
                "{} keyword arguments are not supported".format(kwargs.keys())
            )
        ds = Read_NanonisFile(filename_or_obj).dataset
        if divider != 1:
            try:
                ds["bias"] = ds["bias"] / divider
            except KeyError:
                pass
        return ds

    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".sxm", ".3ds", ".dat"}

    description = "Use Nanonis files (.sxm, .dat, .3ds) in Xarray"
