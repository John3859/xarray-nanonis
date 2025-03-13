"""
Adding Nanonis backend for reading support to Xarray
"""

__all__ = ["NanonisBackendEntrypoint"]

import os

from xarray.backends import BackendEntrypoint

from xarray_nanonis.nanonis import Read_NanonisFile


class NanonisBackendEntrypoint(BackendEntrypoint):
    def open_dataset(self, filename_or_obj, *, drop_variables=None, **kwargs):
        divider = kwargs.pop("divider", 1)
        if kwargs:
            raise KeyError("{} keyword arguments are not supported".format(kwargs.keys()))
        ds = Read_NanonisFile(filename_or_obj, divider=divider).dataset
        return ds

    open_dataset_parameters = ("filename_or_obj", "drop_variables")

    def guess_can_open(self, filename_or_obj):
        if isinstance(filename_or_obj, str) or isinstance(filename_or_obj, os.PathLike):
            filename_or_obj = str(filename_or_obj)
        else:
            return False
        _, ext = os.path.splitext(filename_or_obj)
        return ext in {".sxm", ".3ds", ".dat"}

    description = "Use Nanonis files (.sxm, .dat, .3ds) in Xarray"
