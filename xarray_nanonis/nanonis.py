"""
Read Nanonis files to xarray.Dataset.

There are three kinds of Nanonis data file:
- Nanonis scan file format (.sxm, V2): Containing topography.
- Nanonis ASCII data file format (.dat): Containing point spectroscopy.
- Nanonis Binary file format (.3ds, V1): Containing linecut and mapping.

These file formats consist of ASCII header which is followed by experiment data.
"""

__all__: list[str] = [
    "Read_NanonisFile",
    "Read_NanonisScanFile",
    "Read_NanonisBinaryFile",
    "Read_NanonisASCIIFile",
]

import os
from pathlib import Path
from typing import Any, BinaryIO, TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from xarray_nanonis.utils import (
    _construct_var,
    _get_std_name,
    _parse_header_table,
    _separate_name_unit,
    _handle_multilevel_header,
)

import numpy.typing as npt

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from xarray import Dataset

# End tag of header for different Nanonis file formats.
nanonis_end_tags = {".3ds": ":HEADER_END:", ".sxm": "SCANIT_END", ".dat": "[DATA]"}


class HeaderNotFoundError(Exception):
    # To be raised when header is not found in file
    pass


class Read_NanonisFile:
    """
    Read Nanonis file.

    Base class for all Nanonis file readers.
    This class contains following information and methods shared across all formats:
        - name and path of the file.
        - the position of end tag, by which the file is divided into header and data.
        - read and encode header.

    Header parsing and data loading are accomplished in subclasses:
        - Read_NanonisScanFile, for reading .sxm file;
        - Read_NanonisBinaryFile, for reading .3ds file;
        - Read_NanonisASCIIFile, for reading .dat file.
    """

    def __new__(cls, file_path: str, divider: int = 1):
        """
        Use corresponding subclass according to the file extension.

        Parameters
        ----------
        file_path: str
            Path of the Nanonis file.
        """
        if cls is Read_NanonisFile:
            _, file_ext = os.path.splitext(file_path)
            if file_ext == ".sxm":
                cls = Read_NanonisScanFile
            elif file_ext == ".3ds":
                cls = Read_NanonisBinaryFile
            elif file_ext == ".dat":
                cls = Read_NanonisASCIIFile
        return object.__new__(cls)

    def __init__(self, file_path: str, divider: int = 1) -> None:
        """
        Read Nanonis file.

        The initialization has the following steps:
            - Handle the file path and file name;
            - Locate the position of end tag;
            - Read header into dictionary;;
            - Read data into xarray.Dataset.

        Parameters
        ----------
        file_path: str
            Path of the Nanonis file.
        """

        # Path and suffix of the file.
        self.path: Path = Path(file_path)
        self.suffix: str = self.path.suffix
        # divider
        self.divider: int = divider
        # check is the file exits.
        if not self.path.exists():
            raise FileNotFoundError("Nanonis file {} does not exist".format(self.path))
        self.header: dict[str, Any] | None = None
        self.dataset: xr.Dataset | None = None
        # Read the file and save the information into self.header and self.dataset.
        self._read_file()

    def _read_header(self, file: BinaryIO) -> list[str]:
        """
        Read header from file.

        This function reads each line of the file until encountering the end tag.
        Every line is decoded to string and saved into list.

        Returns
        -------
        list[str]
            List containing each line of header
        """
        # The end tag according to file extension.
        tag: bytes = nanonis_end_tags[self.suffix].encode()
        header: list[str] = []

        # read header from the start of the file
        file.seek(0)
        # Read every line and stop after the end tag.
        for line in file:
            if tag not in line:
                # Decode line to text and save it into list.
                header.append(line.decode(encoding="utf-8", errors="replace"))
            else:
                # If tag is detected, stop iterating.
                break

        return header

    def _parse_header(self, header_raw: list[str]) -> dict[str, Any]:
        """
        Translate the infomation in the header into dictionary.

        Parameters
        ----------
        header_raw : list[str]
            List containing every line of the header.

        Returns
        -------
        dict[str, Any]
        """
        raise NotImplementedError

    def _read_data(self, file: BinaryIO) -> npt.NDArray[np.double] | pd.DataFrame:
        """
        Read data from file.

        Parameters
        ----------
        file : BinaryIO
            The file object returned by open().

        Returns
        -------
        numpy.ndarray[Any, dtype[numpy.double]] | pandas.DataFrame
            The data from the file.
        """
        raise NotImplementedError

    def _organize_data(
        self, data_raw: npt.NDArray[np.double] | pd.DataFrame
    ) -> npt.NDArray[np.double] | pd.DataFrame:
        """
        Organize the data to particular structure by either reshaping or sorting.

        Parameters
        ----------
        data_raw : numpy.ndarray[Any, dtype[numpy.double]] | pandas.DataFrame
            The data returned by the _read_data() method.

        Returns
        -------
        numpy.ndarray[Any, dtype[numpy.double]] | pandas.DataFrame
        """
        raise NotImplementedError

    def _load_dataset(self, data: npt.NDArray[np.double] | pd.DataFrame) -> xr.Dataset:
        """
        Save both data and header into xarray.Dataset.

        Parameters
        ----------
        data: numpy.ndarray[Any, dtype[numpy.double]] | pandas.DataFrame
            The data from _organize_data() method.

        Returns
        -------
        xarray.Dataset
        """
        raise NotImplementedError

    def _read_file(self) -> xr.Dataset:
        """
        Read nanonis file and store the content into xarray.Dataset.

        Nanonis file is read and separated into header and data firstly.
        The header is parsed and stored in a dictionary.
        The data is organized and saved into Dataset, along with the header.

        Returns
        -------
        xarray.Dataset
        """
        # Open nanonis file.
        with open(self.path, "rb") as file:
            # Read header from file.
            header_raw = self._read_header(file)
            # Read data from file.
            data_raw = self._read_data(file)
        # The information in the header is parsed and saved in self.header.
        self.header = self._parse_header(header_raw)
        # Organize the data with information from header.
        data = self._organize_data(data_raw)
        # Store both data and header into Dataset.
        self.dataset = self._load_dataset(data)
        return self.dataset


class Read_NanonisScanFile(Read_NanonisFile):
    """
    Read Nanonis scan file (.sxm).
    """

    def __init__(self, file_path: str, divider: int = 1) -> None:
        self.data_format: str = ">f4"
        super().__init__(file_path, divider=divider)

    @property
    def pixels(self) -> npt.NDArray[np.short]:
        """
        Pixels along x-axis and y-axis.

        Returns
        -------
        numpy.ndarray[Any, dtype[np.short]]
        """
        return np.asarray(self.header["SCAN_PIXELS"].split(), dtype=np.short)

    @property
    def ranges(self) -> npt.NDArray[np.double]:
        """
        Scan range along x-axis and y-axis.
        Returns
        -------
        numpy.ndarray[Any, dtype[np.double]]
        """
        return np.array(self.header["SCAN_RANGE"].split(), dtype=np.double)

    @property
    def range(self) -> npt.NDArray[np.double]:
        """
        Same as ranges.
        Returns
        -------
        numpy.ndarray[Any, dtype[np.double]]
        """
        return self.ranges

    @property
    def center(self):
        """
        Absolulte location of the scan field.
        Returns
        -------
        numpy.ndarray[Any, dtype[np.double]]
        """
        return np.array(self.header["SCAN_OFFSET"].split(), dtype=np.double)

    @property
    def angle(self):
        """
        Rotation angle of the image.
        Returns
        -------
        float
        """
        return float(self.header["SCAN_ANGLE"])

    @property
    def corner(self):
        """
        Left-bottom of the scan field.
        Returns
        -------
        numpy.ndarray[Any, dtype[np.double]]
        """
        return self.center - self.ranges / 2

    @property
    def channels(self) -> list[str]:
        """
        Names of channels recorded while scanning.
        Returns
        -------
        list[str]
        """
        return self.header["DATA_INFO"]["Name"]

    @property
    def units(self) -> dict[str, str]:
        """
        Name and unit pairs of each channel.
        Returns
        -------
        dict[str, str]
        """
        # Units of these channels
        _units = self.header["DATA_INFO"]["Unit"]
        units_dict = dict(zip(self.channels, _units))
        units_dict["x"] = "m"
        units_dict["y"] = "m"
        return units_dict

    @property
    def bias(self) -> float:
        """
        Bias voltage during the scanning.

        This is the bias voltage recorded by nanonis.
        Not always the true sample bias.

        Returns
        -------
        float
        """
        return float(self.header["BIAS"]) / self.divider

    def _parse_header(self, header_raw: list[str]) -> dict[str, str | dict[str, str]]:
        """
        Parse the header of scan file.

        Parameters
        ----------
        header_raw: list[str]
            Every line of header returned by _read_header() method.

        Returns
        -------
        dict[str, Any]
        """
        # Remove the empty line at the end.
        header_raw: list[str] = header_raw[:-1]
        header_dict: dict[str, str | list[str] | dict[str, str]] = dict()

        # Convert header to dictionary.
        for i, line in enumerate(header_raw):
            line = line.rstrip("\n")
            # The header consists of tags surrounded by colons (':')
            # followed by one or more lines of values.
            if line.startswith(":"):
                line: str = line.strip(":")
                count = 1

                # Count how many lines does each value have.
                while not header_raw[i + count].startswith(":"):
                    count += 1
                    if i + count == len(header_raw):
                        break

                if count == 2:
                    # When count is 2, there is only a single line of value.
                    value = header_raw[i + 1].strip()
                    _handle_multilevel_header(header_dict, line, value)
                else:
                    # When There are multiple lines of values,
                    # the values are parsed as table.
                    header_dict[line] = _parse_header_table(
                        header_raw[i + 1 : i + count]
                    )
        return header_dict

    def _read_data(self, file: BinaryIO) -> npt.NDArray[np.double]:
        """
        Read data from file.

        Parameters
        ----------
        file : BinaryIO
            File object returned by open().

        Returns
        -------
        numpy.ndarray[Any, dtype[numpy.double]]
        """
        # the binary data begins with /1A/04
        file.seek(4, os.SEEK_CUR)
        data: npt.NDArray[np.double] = np.fromfile(file, dtype=self.data_format)
        return data

    def _organize_data(self, data_raw: npt.NDArray[np.double]):
        """
        Reshape the data to multidimensional array.

        In the data, the channels are stored one after the other,
        forward scan followed by backward scan. The data is stored
        chronologically as it is recorded. On an up-scan, the first
        point corresponds to the lower left corner of the scanfield (forward scan).
        On a down-scan, it is the upper left corner of the scanfield.
        Hence, backward scan data start on the right side of the scanfield.

        Parameters
        ----------
        data_raw : numpy.ndarray[Any, dtype[numpy.double]]
            The array returned by _read_data() method.

        Returns
        -------
        numpy.ndarray[Any, dtype[numpy.double]]
        """
        # Number of channels
        n_channel = len(self.channels)
        # Number of scan directions
        n_direction = 2
        # Number of scan pixels
        n_x, n_y = self.pixels
        # Reshape the array according to the order the data is stored.
        try:
            data = data_raw.reshape(n_channel, n_direction, n_y, n_x)
        except ValueError:
            data = np.empty((n_channel, n_direction, n_y, n_x), dtype=np.double)
            data[:, 0, :, :] = data_raw.reshape(n_channel, 1, n_y, n_x)
            data[:, 1, :, :] = data[:, 0, :, :]
        # Reorder the array so that up-scan, down-scan, forward-scan and backward-scan
        # all have the lower left corner of the scanfield
        if self.header["SCAN_DIR"] == "down":
            # When the scan direction is down and forward,
            # the first pixel is the upper left corner.
            # The data is mirrored horizontally,
            # so that the first pixel is the lower left corner.
            data[:, 0, :, :] = np.flip(data[:, 0, :, :], axis=-2)
            # When the scan direction is down and backward,
            # the first pixel is the upper right corner.
            # The data is mirrored both horizontally and vertically.
            data[:, 1, :, :] = np.flip(data[:, 1, :, :], (-1, -2))
        else:
            # When the scan direction is up and forward,
            # the first pixel is already the lower left corner.
            # Hence, no mirror is needed.
            # When the scan direction is up and backward,
            # the first pixel is the lower right corner.
            # The data is mirrored vertically.
            data[:, 1, :, :] = np.flip(data[:, 1, :, :], axis=-1)
        return data

    def _load_dataset(self, data: npt.NDArray[np.double]) -> xr.Dataset:
        """
        Save header and data into Dataset.

        Parameters
        ----------
        data : numpy.ndarray[Any, dtype[numpy.double]]
            Array returned by _organize_data() method, have particular structure.

        Returns
        -------
        xarray.Dataset
        """
        # Number of channels
        chan_name: list[str] = self.channels
        # the number of pixels
        n_x, n_y = self.pixels
        # The range in x-axis and y-axis
        range_x, range_y = self.range

        # Coordinates of the Dataset
        coords_var = dict()
        # x-axis
        coords_var["x"] = _construct_var(
            "x", np.linspace(0, range_x, n_x), _get_std_name("x"), "m"
        )
        # y-axis
        coords_var["y"] = _construct_var(
            "y", np.linspace(0, range_y, n_y), _get_std_name("y"), "m"
        )
        # Scan directions
        coords_var["dir"] = _construct_var(
            "dir", ["forward", "backward"], _get_std_name("dir"), None
        )

        # Split scan data from different channels,
        # and convert them to xarray.Variable
        # All the variables are saved into a dictionary.
        shared_attrs = {
            "pixels": (n_x, n_y),
            "ranges": (range_x, range_y),
            "steps": (range_x / (n_x - 1), range_y / (n_y - 1)),
            "bias": self.bias,
            "center": self.center,
            "corner": self.corner,
            "angle": self.angle,
        }
        data_var: dict[str, xr.Variable] = dict()
        for i, chan in enumerate(chan_name):
            data_var[chan] = _construct_var(
                ["dir", "y", "x"],
                data[i],
                _get_std_name(chan),
                self.units[chan],
                attrs=shared_attrs,
            )

        # Store the data into xarray.DataSet
        data_ds = xr.Dataset(
            data_var,
            coords=coords_var,
            attrs=self.header,
        )
        return data_ds


class Read_NanonisBinaryFile(Read_NanonisFile):
    """
    Read Nanonis binary file (.3ds).
    """

    def __init__(self, file_path: str, divider: int = 1):
        """
        Read Nanonis grid file.
        Parameters
        ----------
        file_path : str
            File path of the file.
        """
        self._data_format: str = ">f4"
        super().__init__(file_path, divider=divider)

    @property
    def points(self) -> int:
        """
        Number of Points of the bias spectroscopy.
        Returns
        -------
        int
        """
        n_point = int(self.header["Points"])
        return n_point

    @property
    def pixels(self) -> tuple[int, int]:
        """
        Number of points along x and y direction
        Returns
        -------
        tuple[int, int]
        """
        n_x, n_y = np.asarray(self.header["Grid dim"].split("x"), dtype=np.short)
        return n_x, n_y

    @property
    def grid_setting(self) -> tuple[float, ...]:
        """
        Grid Setting.
        Returns
        -------
        numpy.ndarray[Any, dtype[numpy.double]]
        """
        return tuple(np.array(self.header["Grid settings"], dtype=np.double))

    @property
    def wh(self) -> tuple[float, float]:
        """
        Width and height of the grid.
        Returns
        -------
        tuple[float, float]
        """
        w, h = self.grid_setting[2:4]
        return w, h

    @property
    def center(self) -> tuple[float, float]:
        """
        Center of the grid.
        Returns
        -------
        tuple[float, float]
        """
        cx, cy = self.grid_setting[:2]
        return cx, cy

    @property
    def corner(self) -> tuple[float, float]:
        """
        Left-bottom corner of the grid.
        Returns
        -------
        tuple[float, float]
        """
        cx, cy = self.center
        w, h = self.wh
        return cx - w / 2, cy - h / 2

    @property
    def angle(self) -> float:
        """
        Rotation angle of the grid.
        Returns
        -------
        float
        """
        return self.grid_setting[-1]

    @property
    def steps(self) -> tuple[float, float]:
        """
        Distance between each pixel of the grid.
        Returns
        -------
        tuple[float, float]
        """
        w, h = self.wh
        n_x, n_y = self.pixels
        step_x = w / (n_x - 1)
        if n_y != 1:
            step_y = h / (n_y - 1)
        else:
            step_y = 0
        return step_x, step_y

    @property
    def bias_range(self) -> tuple[float, float]:
        """
        Start and end value of the bias sweep.
        Returns
        -------
        tuple[float, float]
        """
        bias_start = (
            float(self.header["Bias Spectroscopy"]["Sweep Start (V)"]) / self.divider
        )
        bias_end = (
            float(self.header["Bias Spectroscopy"]["Sweep End (V)"]) / self.divider
        )
        return bias_start, bias_end

    def _parse_header(self, header_raw) -> dict[str, str | list[str] | dict[str, str]]:
        """
        Parse key-value pairs in grid file header.

        Key-value pairs from header are read and split, and then stored into a dictionary.
        Some values from header are series of values seperated by ";". They are split,
        and stored into a tuple.

        All the information from header are stored into dictionary using string or list of
        strings, without any type conversion. When using data stored in header, convert its
        type manually.

        Parameters
        ----------
        header_raw : list[str]
            List of each line of the header. It is returned by _read_header() method.

        Returns
        -------
        dict[str, str | list[str]]
            Key-value pairs from header.
        """
        # The dictionary to store key-value pairs from header.
        header_dic: dict[str, str | list[str] | dict[str, str]] = dict()

        for line in header_raw:
            # Each line in header is ended with "\r\n".
            # Key-value pairs are seperated with '='.
            key, val = line.rstrip("\r\n").split("=", 1)
            # Remove the '"' surrounding the values.
            # A series of values are seperated with ';'. Split them into tuple.
            if ";" in line:
                val: list[str] | str = val.strip('"').split(";")
            else:
                val = val.strip('"')
            _handle_multilevel_header(header_dic, key, val)
        return header_dic

    def _read_data(self, file: BinaryIO) -> npt.NDArray[np.double]:
        """
        Read data from file.
        Parameters
        ----------
        file: BinaryIO
            File object returned by open().

        Returns
        -------
        numpy.ndarray[Any, dtype[numpy.double]]
        """
        data: npt.NDArray[np.double] = np.fromfile(file, dtype=self._data_format)
        return data

    def _organize_data(self, data: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        """
        Organize data into multidimensional array with particular structure.

        The experiments aren't separated, all data is written into
        the file continuously. Each experiment starts with the fixed
        parameters, followed by the experiment parameters and the
        experiment data (Channels one after the other). The size of
        the experiment data is defined in the header, so it's easy to
        read a specific experiment. From the start of the binary data an
        experiment including the fixed and experiment parameters always
        takes (# Parameters) * 4 + (Experiment size) bytes.

        Parameters
        ----------
        data: numpy.ndarray[Any, dtype[numpy.double]]
            The array returned by _read_data() method.

        Returns
        -------
        numpy.ndarray[Any, dtype[numpy.double]]
        """
        # Read size of grid data from header.
        # Number of parameters.
        n_para: int = int(self.header["# Parameters (4 byte)"])
        # Number of channels.
        n_chan: int = len(self.header["Channels"])
        # Number of points.
        n_point: int = self.points
        # Pixels of the grid.
        n_x, n_y = self.pixels

        # The size depends on whether backward sweep is on or not.
        back_status: bool = self._check_back_status()

        # If back sweep is on, both forward and backward data are stored
        if back_status:
            exp_size: int = n_para + 2 * n_chan * n_point
        # Otherwise, only forward data are stored.
        else:
            exp_size = n_para + n_chan * n_point

        # Reshape raw data into 3D array.
        # axis in order ['y', 'x', 'parameter + channel'].
        data = data.reshape(n_y, n_x, exp_size)
        # Move axis so that the order of the axis is ['param + channel', 'y', 'x']
        data = np.moveaxis(data, 2, 0)
        return data

    def _load_dataset(self, data: npt.NDArray[np.double]) -> xr.Dataset:
        """
        Load parameters and signals from raw data file.

        The data in grid file contains two parts, parameters and signals.
        Parameters and signals are stored in a single Dataset.

        Signals are a series of 3-dimensional array and kept in xarray
        DataSet. Each channel can be obtained by using its name('name'
        for forward sweep and 'name_bwd' for backward sweep).

        Parameters are stored into the same Dataset of signals as coordinates.

        Parameters
        ----------
        data : numpy.ndarray[Any, dtype[numpy.double]]
            The array returned by _organize_data() method.

        Returns
        -------
        xarray.Dataset
        """

        # Read size of grid data from header.
        # Number of parameters
        n_para: int = int(self.header["# Parameters (4 byte)"])
        # Number of points
        n_point: int = self.points
        # Pixels of the grid
        n_x, n_y = self.pixels
        # Width and height of the grid.
        width, height = self.wh
        # Distance between grid pixels.
        x_step, y_step = self.steps
        # Whether back sweep is on or off.
        back_status: bool = self._check_back_status()

        # Raw data contains both parameters and signals from different channels.
        # Separate these apart and store parameters into Dataset's coordinates.
        exp_param_name, exp_param_units = _separate_name_unit(
            self.header["Experiment parameters"]
        )
        fixed_param_name, fixed_param_units = _separate_name_unit(
            self.header["Fixed parameters"]
        )
        para_name: list[str] = fixed_param_name + exp_param_name
        param_units: dict[str, str] = fixed_param_units | exp_param_units
        para_array: npt.NDArray[np.double] = data[0:n_para, :, :]

        # Store parameters into a dictionary of Variable.
        para_var: dict[str, xr.Variable] = dict()
        for i, param in enumerate(para_name):
            para_var[param] = _construct_var(
                ["y", "x"], para_array[i, :, :], param, param_units[param]
            )

        # Generate axis 'x', 'y' and 'Bias'.
        para_var["x"] = _construct_var(
            "x", np.linspace(0, width, n_x), _get_std_name("x"), "m"
        )
        para_var["y"] = _construct_var(
            "y", np.linspace(0, height, n_y), _get_std_name("y"), "m"
        )
        # Get the start and end value of bias sweep.
        try:
            bias_start, bias_end = self.bias_range
        # If This information is not stored in header, get them from parameters.
        except KeyError:
            bias_start = para_var["Sweep_Start"][0, 0] / self.divider
            bias_end = para_var["Sweep_End"][0, 0] / self.divider
        # Reverse the bias dimension if initial bias voltage is larger.
        bias_reverse = False
        if bias_start > bias_end:
            bias_reverse = True
            bias_start, bias_end = bias_end, bias_start
        para_var["bias"] = _construct_var(
            "bias",
            np.linspace(bias_start, bias_end, n_point),
            _get_std_name("bias"),
            "V",
        )

        # Separate signals from parameters in raw data.
        data_array: npt.NDArray[np.double] = data[n_para:, :, :]
        # Split signals from different channels apart,
        # and store them into a dictionary.
        chan_name, chan_units = _separate_name_unit(self.header["Channels"])
        data_var: dict[str, xr.Variable] = dict()
        shared_attrs = {
            "pixels": (n_x, n_y),
            "points": n_point,
            "ranges": (width, height),
            "steps": (x_step, y_step),
            "bias_range": (bias_start, bias_end),
            "center": self.center,
            "angle": self.angle,
            "corner": self.corner,
        }
        for i, chan in enumerate(chan_name):
            std_name = _get_std_name(chan)
            if back_status:
                # If backward sweep is on, backward signals are also stored.
                i *= 2
                if bias_reverse:
                    if i == 0:
                        fwd_slice = slice((i + 1) * n_point - 1, None, -1)
                    else:
                        fwd_slice = slice((i + 1) * n_point - 1, i * n_point - 1, -1)
                    bwd_slice = slice((i + 2) * n_point - 1, (i + 1) * n_point - 1, -1)
                else:
                    fwd_slice = slice(i * n_point, (i + 1) * n_point)
                    bwd_slice = slice((i + 1) * n_point, (i + 2) * n_point)
                data_var[chan] = _construct_var(
                    ["bias", "y", "x"],
                    data_array[fwd_slice, :, :],
                    std_name,
                    chan_units[chan],
                    attrs=shared_attrs,
                )
                data_var[chan + "_bwd"] = _construct_var(
                    ["bias", "y", "x"],
                    data_array[bwd_slice, :, :],
                    std_name + "_bwd",
                    chan_units[chan + "_bwd"],
                    attrs=shared_attrs,
                )
            else:
                # If backward sweep is off, only forward signals are stored
                if bias_reverse:
                    if i == 0:
                        fwd_slice = slice((i + 1) * n_point - 1, None, -1)
                    else:
                        fwd_slice = slice((i + 1) * n_point - 1, i * n_point - 1, -1)
                else:
                    fwd_slice = slice(i * n_point, (i + 1) * n_point)
                data_var[chan] = _construct_var(
                    ["bias", "y", "x"],
                    data_array[fwd_slice, :, :],
                    std_name,
                    chan_units[chan],
                    attrs=shared_attrs,
                )

        # Store into Dataset
        data_ds: xr.Dataset = xr.Dataset(
            data_var,
            coords=para_var,
            attrs=self.header,
        )

        # Line-cut only have 1 pixel along y direction,
        # remove this dimension from the Dataset.
        if n_y == 1:
            data_ds = data_ds.squeeze("y")
            data_ds["x"].attrs["standard_name"] = "Distance"

        # sort the bias dimension
        data_ds = data_ds.sortby("bias")
        return data_ds

    def _check_back_status(self) -> bool:
        """
        Check whether backward sweep is enabled.
        Returns
        -------
        bool
        """
        # Convert back sweep status into bool.
        try:
            back_status: str = self.header["Bias Spectroscopy"]["backward sweep"]
        except KeyError:
            # If backward sweep status is not saved in file,
            # decide the backward status using experiment size.
            exp_size = int(self.header["Experiment size (bytes)"])
            n_point: int = self.points
            # The experimental size when backward sweep is enabled.
            # Each floating point number uses 4 bytes.
            # When acquiring 1 channel forward and backward, 256 points,
            # this will be 2 x 256 x 4 bytes = 2048 bytes.
            exp_size_bwd: int = 8 * n_point
            if exp_size == exp_size_bwd:
                return True
            else:
                return False

        if back_status == "FALSE":
            return False
        else:
            return True


class Read_NanonisASCIIFile(Read_NanonisFile):
    """
    Read Nanonis ASCII file (.dat).
    """

    def __init__(self, file_path: str | Path, divider: int = 1) -> None:
        super().__init__(file_path, divider=divider)

    @property
    def bias_range(self) -> tuple[float, float]:
        """
        Start and end value of the bias sweep.
        Returns
        -------
        tuple[float, float]
        """
        bias_start = (
            float(self.header["Bias Spectroscopy"]["Sweep Start (V)"]) / self.divider
        )
        bias_end = (
            float(self.header["Bias Spectroscopy"]["Sweep End (V)"]) / self.divider
        )
        return bias_start, bias_end

    @property
    def center(self) -> tuple[float, float]:
        """
        The absolute location of the bias sweep.
        Returns
        -------
        tuple[float, float]
        """
        loc_x = float(self.header["X (m)"])
        loc_y = float(self.header["Y (m)"])
        return loc_x, loc_y

    def _parse_header(self, header: list[str]) -> dict[str, str | dict[str, str]]:
        """
        Parse the raw header of spectroscopy.

        Every line of the header contains a key-value pair. Each value and key is followed by '\t'.
        This function uses '\t' as delimiter to separate key and value and converts them into a
        dictionary.

        Since no type conversion applied, both keys and values are *string*.

        Parameters
        ----------
        header : list[str]
            Each line of the header, returned by _read_header() method.

        Returns
        -------
        dict[str, str]
        """

        # Load the raw header from file and split it into lists of lines.
        header_raw: list[str] = header[:-1]
        header_dict: dict[str, str | dict[str, str]] = dict()

        # pretreatment of header entries.
        for line in header_raw:
            line = line.rstrip("\r\n")
            if line[-1] == "\t":
                # Delete \t at the end of values
                line: str = line.rstrip("\t")
            if "\t" not in line:
                # Make sure every key is followed by \t
                line += "\t"

            key, val = line.split("\t")
            _handle_multilevel_header(header_dict, key, val)
        self.header = header_dict
        return header_dict

    def _read_data(self, file: BinaryIO) -> pd.DataFrame:
        """
        Read data from file to Dataframe.

        Parameters
        ----------
        file: BinaryIO
            File object returned by open().

        Returns
        -------
        pandas.DataFrame
        """
        data: pd.DataFrame = pd.read_csv(file, sep="\t", index_col=0)
        return data

    def _organize_data(self, data_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Sort data according to the bias voltage.

        Parameters
        ----------
        data_raw: pandas.DataFrame
            The raw data read from file.

        Returns
        -------
        pandas.Dataframe
            Data sorted according to the bias voltage.
        """
        data = data_raw.sort_index()
        return data

    def _load_dataset(self, data: pd.DataFrame) -> xr.Dataset:
        """
        This function loads signal data from .dat file.

        Parameters
        ----------
        data : pandas.DataFrame
            Data returned by _organize_data() method.

        Returns
        -------
        xarray.Dataset
        """
        # Read channels' names from Nanonis file.
        # The names from the file contain both name and unit of each channel.
        # names and units of all the channels
        _channels = data.columns.values
        names, units = _separate_name_unit(_channels)

        # Sample bias index.
        bias = (
            _construct_var(
                "bias",
                data.index,
                standard_name=_get_std_name("bias"),
                units="V",
            )
            / self.divider
        )
        # bias range
        try:
            bias_start, bias_end = self.bias_range
            if bias_start > bias_end:
                bias_start, bias_end = bias_end, bias_start
        except KeyError:
            bias_start = bias[0]
            bias_end = bias[-1]

        # Dictionary that maps channel names to experimental data.
        shared_attrs = {"bias_range": (bias_start, bias_end), "center": self.center}
        split_signal = dict()
        for chan, name in zip(_channels, names):
            split_signal[name] = _construct_var(
                "bias",
                data[chan],
                standard_name=_get_std_name(name),
                units=units[name],
                attrs=shared_attrs,
            )

        # Save spectroscopic data into Xarray.Dataset.
        data_ds = xr.Dataset(split_signal, coords={"bias": bias}, attrs=self.header)
        self.dataset = data_ds
        return data_ds
