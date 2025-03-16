"""
Testing for reading NANONIS .3ds files

This module provides comprehensive test cases for the xarray_nanonis package's
ability to read and parse NANONIS .3ds files, which are binary files containing
scanning tunneling spectroscopy (STS) data. Tests cover:

1. Standard 3ds files with 2D grids of spectroscopy points
2. Line spectroscopy files (1D cuts where one grid dimension is 1)
3. Multi-Linear Segments (MLS) files with varying bias voltage ranges

Each test class creates synthetic file data that mimics the structure of real
NANONIS files, including both the header information and binary data portions.
The tests validate that xarray_nanonis correctly:
- Parses header information into dataset attributes
- Extracts fixed and experimental parameters
- Loads channel data with proper dimensions
- Handles special cases like MLS bias coordinates
"""

import numpy as np
import pytest
import xarray as xr

import xarray_nanonis  # noqa: F401


class TestStd3dsFile:
    """
    Test suite for validating standard NANONIS 3ds file reading functionality.

    This class tests the xarray_nanonis package's ability to correctly read and
    parse standard NANONIS .3ds files, which contain grid spectroscopy data.
    The tests verify:

    1. Header information is correctly parsed into dataset attributes
    2. Fixed parameters from the file are properly extracted
    3. Experiment parameters are correctly imported
    4. Channel data is correctly loaded into the dataset with appropriate dimensions

    The class uses fixtures to generate synthetic 3ds file data that mimics
    the structure of real NANONIS files, including headers and binary data
    representing spectroscopy measurements across a 2D grid.
    """

    @pytest.fixture(scope="class")
    def grid_dim(self):
        """
        Fixture to create a grid dimension
        """
        return 256, 256

    @pytest.fixture(scope="class")
    def grid_points(self):
        """
        Fixture to create a grid points
        """
        return 101

    @pytest.fixture(scope="class")
    def exp_channels(self):
        """
        Fixture to create channels
        """
        return ("Current (A)", "LI_Demod_1_X (A)", "LI_Demod_1_Y (A)", "LI_Demod_2_X (A)", "LI_Demod_2_Y (A)")

    @pytest.fixture(scope="class")
    def grid_data(self, grid_dim, grid_points, exp_channels):
        """
        Fixture to create a grid data
        """

        n_channels = len(exp_channels)
        # Calculate total data points: (grid_points * n_channels + 8) * grid_x * grid_y
        n_data = (grid_points * n_channels + 8) * grid_dim[0] * grid_dim[1]

        # Create random data
        rng = np.random.default_rng()
        data = rng.random(n_data).reshape(grid_dim[1], grid_dim[0], -1)

        # Set fixed parameters
        data[:, :, 0] = -1  # Sweep Start
        data[:, :, 1] = 1  # Sweep End
        return data.flatten()

    @pytest.fixture(scope="class")
    def grid_header(self, grid_dim, grid_points, exp_channels):
        """
        Fixture to create a grid header
        """
        return "\r\n".join(
            [
                'Grid dim="{} x {}"'.format(grid_dim[0], grid_dim[1]),
                "Grid settings=0.000000E+0;0.000000E+0;6.880776E-9;6.880776E-9;2.244028E+1",
                'Sweep Signal="Bias (V)"',
                'Fixed parameters="Sweep Start;Sweep End"',
                'Experiment parameters="X (m);Y (m);Z (m);Z Offset (m);Settling time (s);Integration time (s)"',
                "# Parameters (4 byte)=8",
                f"Experiment size (bytes)={grid_points * len(exp_channels) * 4}",
                'Points="{}"'.format(grid_points),
                'Channels="{}"'.format(";".join(exp_channels)),
                'Experiment="Grid Spectroscopy"',
                "Lock-in>Lock-in status=ON",
                'Lock-in>Modulated signal="Bias (V)"',
                "Lock-in>Amplitude=20E-3",
                'Lock-in>Demodulated signal="Current (A)"',
                ":HEADER_END:\r\n",
            ]
        )

    @pytest.fixture(scope="class")
    def header_dict(self, grid_dim, grid_points, exp_channels):
        """
        Fixture to create a header dictionary
        """
        return {
            "Grid dim": f"{grid_dim[0]} x {grid_dim[1]}",
            "Grid settings": ("0.000000E+0", "0.000000E+0", "6.880776E-9", "6.880776E-9", "2.244028E+1"),
            "Sweep Signal": "Bias (V)",
            "Fixed parameters": ("Sweep Start", "Sweep End"),
            "Experiment parameters": (
                "X (m)",
                "Y (m)",
                "Z (m)",
                "Z Offset (m)",
                "Settling time (s)",
                "Integration time (s)",
            ),
            "# Parameters (4 byte)": "8",
            "Experiment size (bytes)": f"{grid_points * len(exp_channels) * 4}",
            "Points": f"{grid_points}",
            "Channels": exp_channels,
            "Experiment": "Grid Spectroscopy",
            "Lock-in": {
                "Lock-in status": "ON",
                "Modulated signal": "Bias (V)",
                "Amplitude": "20E-3",
                "Demodulated signal": "Current (A)",
            },
        }

    @pytest.fixture(scope="class", autouse=True)
    def grid_data_3ds(self, tmp_path_factory, grid_data, grid_header):
        """
        Fixture to create a grid data 3ds
        """
        file_path = tmp_path_factory.mktemp("3ds_file") / "grid.3ds"
        with open(file_path, "wb") as f:
            f.write(grid_header.encode("utf-8"))
            f.write(grid_data.astype(">f4").tobytes())
        return file_path

    @pytest.fixture(scope="class")
    def ds(self, grid_data_3ds):
        """
        Fixture to create a dataset
        """
        return xr.load_dataset(grid_data_3ds)

    def test_header(self, ds, header_dict):
        """
        Test to read the header
        """
        assert ds.attrs == header_dict

    @pytest.fixture(scope="class")
    def data_parts(self, ds, grid_data, grid_dim, header_dict, grid_points, exp_channels):
        """
        Test of reading data values
        """
        # Reshape the data and handle incomplete data by filling with NaN
        expected_shape = (grid_dim[1], grid_dim[0], grid_points * len(exp_channels) + 8)
        total_elements = np.prod(expected_shape)

        # If the data is incomplete, pad with NaN values
        if len(grid_data) < total_elements:
            padded_data = np.full(total_elements, np.nan, dtype=float)
            padded_data[: len(grid_data)] = grid_data
            data = padded_data.reshape(expected_shape)
        else:
            data = grid_data.reshape(expected_shape)

        data_fixed = data[:, :, : len(header_dict["Fixed parameters"])]
        data_param = data[:, :, len(header_dict["Fixed parameters"]) : int(header_dict["# Parameters (4 byte)"])]

        data_channels = data[:, :, int(header_dict["# Parameters (4 byte)"]) :].reshape(
            grid_dim[1], grid_dim[0], len(exp_channels), grid_points
        )

        return data_fixed, data_param, data_channels

    def test_fixed_parameter_values(self, ds, data_parts, header_dict):
        """
        Test to read fixed parameters
        """
        data_fixed, _, _ = data_parts
        for i, name in enumerate(header_dict["Fixed parameters"]):
            name = name.replace(" ", "_")
            desired_value = np.squeeze(data_fixed[:, :, i])
            np.testing.assert_equal(ds[name].values, desired_value)

    def test_experiment_parameter_values(self, ds, data_parts, header_dict):
        """
        Test to read experiment parameters
        """
        _, data_param, _ = data_parts
        for i, name in enumerate(header_dict["Experiment parameters"]):
            name = name.rsplit(" (", 1)[0].strip(" ")
            name = name.replace(" ", "_")
            desired_value = np.squeeze(data_param[:, :, i])
            np.testing.assert_almost_equal(ds[name].values, desired_value)

    def test_channel_values(self, ds, data_parts, header_dict):
        """
        Test to read channels
        """
        _, _, data_channels = data_parts
        for i, name in enumerate(header_dict["Channels"]):
            name = name.rsplit(" (", 1)[0].strip(" ")
            name = name.replace(" ", "_")
            desired_value = np.squeeze(np.moveaxis(data_channels[:, :, i], -1, 0))
            np.testing.assert_almost_equal(ds[name].values, desired_value)


class TestLine3dsFile(TestStd3dsFile):
    """
    Class to test 3ds file for line cuts (grid dimension of y is 1)
    """

    @pytest.fixture(scope="class")
    def grid_dim(self):
        """
        Fixture to create a grid dimension
        """
        return 256, 1


class TestMLS3dsFile(TestStd3dsFile):
    """
    Test class for MLS (Multi-Linear Segments) 3DS files.

    This class extends TestStd3dsFile to test the specific functionality
    of Multi-Linear Segments data files, which contain spectroscopy data
    collected over multiple bias voltage segments. The class provides fixtures
    for generating the appropriate grid headers and header dictionaries for
    MLS files, and tests the proper construction of bias coordinates based on
    the segment information.

    MLS files differ from standard 3DS files in that they contain spectroscopy
    data collected across multiple bias segments with different voltage ranges
    and step counts. This requires special handling for constructing the bias
    coordinate array from the concatenation of these segments.
    """

    @pytest.fixture(scope="class")
    def grid_points(self):
        return 18

    @pytest.fixture(scope="class")
    def grid_header(self, grid_dim, grid_points, exp_channels):
        """
        Fixture to create a grid header
        """
        segments_info = [
            "-100E-3,-20E-3,15E-3,360E-3,9,0E+0,0E+0",
            "-20E-3,20E-3,15E-3,360E-3,2,0E+0,0E+0",
            "20E-3,100E-3,15E-3,360E-3,9,0E+0,0E+0",
        ]

        return "\r\n".join(
            [
                'Grid dim="{} x {}"'.format(grid_dim[0], grid_dim[1]),
                "Grid settings=0.000000E+0;0.000000E+0;6.880776E-9;6.880776E-9;2.244028E+1",
                "Filetype=MLS",
                'Sweep Signal="Bias (V)"',
                'Fixed parameters="Sweep Start;Sweep End"',
                'Experiment parameters="X (m);Y (m);Z (m);Z Offset (m);Settling time (s);Integration time (s)"',
                "# Parameters (4 byte)=8",
                "Segment Start (V), Segment End (V), Settling (s), Integration (s), Steps (xn), Lockin, Init. Settling (s)={}".format(
                    ";".join(segments_info)
                ),
                f"Experiment size (bytes)={grid_points * len(exp_channels) * 4}",
                'Points="{}"'.format(grid_points),
                'Channels="{}"'.format(";".join(exp_channels)),
                'Experiment="Grid Spectroscopy"',
                "Lock-in>Lock-in status=ON",
                'Lock-in>Modulated signal="Bias (V)"',
                "Lock-in>Amplitude=20E-3",
                'Lock-in>Demodulated signal="Current (A)"',
                ":HEADER_END:\r\n",
            ]
        )

    @pytest.fixture(scope="class")
    def header_dict(self, grid_dim, grid_points, exp_channels):
        """
        Fixture to create a header dictionary
        """
        return {
            "Grid dim": f"{grid_dim[0]} x {grid_dim[1]}",
            "Filetype": "MLS",
            "Grid settings": ("0.000000E+0", "0.000000E+0", "6.880776E-9", "6.880776E-9", "2.244028E+1"),
            "Sweep Signal": "Bias (V)",
            "Fixed parameters": ("Sweep Start", "Sweep End"),
            "Experiment parameters": (
                "X (m)",
                "Y (m)",
                "Z (m)",
                "Z Offset (m)",
                "Settling time (s)",
                "Integration time (s)",
            ),
            "Segment Start (V), Segment End (V), Settling (s), Integration (s), Steps (xn), Lockin, Init. Settling (s)": (
                "-100E-3,-20E-3,15E-3,360E-3,9,0E+0,0E+0",
                "-20E-3,20E-3,15E-3,360E-3,2,0E+0,0E+0",
                "20E-3,100E-3,15E-3,360E-3,9,0E+0,0E+0",
            ),
            "# Parameters (4 byte)": "8",
            "Experiment size (bytes)": f"{grid_points * len(exp_channels) * 4}",
            "Points": f"{grid_points}",
            "Channels": exp_channels,
            "Experiment": "Grid Spectroscopy",
            "Lock-in": {
                "Lock-in status": "ON",
                "Modulated signal": "Bias (V)",
                "Amplitude": "20E-3",
                "Demodulated signal": "Current (A)",
            },
        }

    def test_MLS_coord(self, ds, header_dict):
        """
        Test the MLS (Multi Line Spectroscopy) bias coordinate creation.

        This test validates that the bias coordinate in the dataset is correctly
        constructed from the segment information in the header dictionary.

        For each segment, it extracts the start voltage, end voltage, and number of points,
        then builds a bias coordinate list by concatenating linspace arrays between each
        start and end point. For segments after the first one, it skips the first point
        to avoid duplicates at segment boundaries.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset containing the bias coordinate
        header_dict : dict
            Dictionary containing header information with segment data
        """
        segments = header_dict[
            "Segment Start (V), Segment End (V), Settling (s), Integration (s), Steps (xn), Lockin, Init. Settling (s)"
        ]
        bias_coord = []
        for i, segment in enumerate(segments):
            start, end = segment.split(",")[0:2]
            start = float(start)
            end = float(end)
            points = int(segment.split(",")[4])
            if i == 0:
                bias_coord += list(np.linspace(start, end, points))
            else:
                bias_coord += list(np.linspace(start, end, points)[1:])
        np.testing.assert_almost_equal(ds["bias"].values, bias_coord)


class TestIncomplete3dsFile(TestStd3dsFile):
    """
    Test class for incomplete 3DS files.
    """

    @pytest.fixture(scope="class")
    def grid_data(self, grid_dim, grid_points, exp_channels):
        """
        Fixture to create a grid data
        """
        n_channels = len(exp_channels)
        # Calculate total data points: (grid_points * n_channels + 8) * grid_x * grid_y
        n_data = (grid_points * n_channels + 8) * grid_dim[0] * grid_dim[1]

        # Create random data
        rng = np.random.default_rng()
        data = rng.random(n_data).reshape(grid_dim[1], grid_dim[0], -1)

        # Set fixed parameters
        data[:, :, 0] = -1  # Sweep Start
        data[:, :, 1] = 1  # Sweep End

        # Simulate incomplete acquisition by truncating data
        cutoff_line = grid_dim[0] // 2  # Cut off halfway through the grid
        # Reshape to keep only data up to the cutoff line
        truncated_data = data[:cutoff_line, :, :]
        # Reshape back to original format but with fewer lines
        data = truncated_data
        return data.flatten()
