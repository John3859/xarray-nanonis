"""
Testing for reading of Nanonis .dat files
"""

import numpy as np
import pytest
import xarray as xr

import xarray_nanonis  # noqa: F401


class TestStdDatfile:
    """
    Test the reading of standard .dat files
    """

    @pytest.fixture(scope="class")
    def dat_header_dict(self):
        """
        Provides a dictionary of header key-value pairs from a typical Nanonis .dat file.

        This fixture contains only the simple header entries (no multilevel entries with '>' in the key).
        The dictionary represents metadata values typically found in Nanonis bias spectroscopy measurements.

        Returns:
            dict: Dictionary with header keys and their corresponding string values.
        """
        return {
            "Experiment": "bias spectroscopy",
            "X (m)": "-4.53963E-9",
            "Y (m)": "-7.82028E-9",
            "Z (m)": "-18.4467E-9",
            "Z offset (m)": "0E+0",
            "Settling time (s)": "10E-3",
            "Integration time (s)": "20E-3",
            "Z-Ctrl hold": "TRUE",
            "Final Z (m)": "N/A",
        }

    @pytest.fixture(scope="class")
    def dat_header(self, dat_header_dict):
        """
        Generates a string representation of a Nanonis .dat file header.

        This fixture uses the dat_header_dict fixture to create a properly formatted header string
        with tab-separated key-value pairs as found in Nanonis .dat files. Each line ends with
        a tab character followed by a newline, matching the format of real data files.

        Args:
            dat_header_dict (dict): Dictionary containing header key-value pairs.

        Returns:
            str: A formatted header string as it would appear in a Nanonis .dat file.
        """
        header_str = ""
        for key, value in dat_header_dict.items():
            header_str += f"{key}\t{value}\t\n"

        return header_str

    @pytest.fixture(scope="class")
    def dat_data(self):
        """
        Generates random data mimicking the structure of a Nanonis .dat file.

        Returns:
            numpy.ndarray: Random data array with 5 columns and 201 rows with dtype float32.
        """
        # Create a random number generator
        rng = np.random.default_rng()

        # Fixed number of data points (201)
        n_points = 201

        # Generate bias voltage values (column 1 and 5)
        bias_values = np.linspace(-1, 1, n_points, dtype=np.float32)

        # Generate random current values (column 2)
        # Use normally distributed values with mean and standard deviation similar to example data
        current_values = rng.normal(-500e-12, 500e-12, n_points).astype(np.float32)

        # Generate random lock-in values (columns 3 and 4)
        li_demod_1 = rng.normal(5e-12, 1e-12, n_points).astype(np.float32)
        li_demod_2 = rng.normal(5e-12, 1e-12, n_points).astype(np.float32)

        # Stack the columns to create the data array
        data = np.column_stack((bias_values, current_values, li_demod_1, li_demod_2, bias_values))

        return data

    @pytest.fixture(scope="class")
    def dat_std_file(self, tmp_path_factory, dat_header, dat_data):
        """
        Creates a temporary .dat file for testing with random data.

        Args:
            tmp_path_factory: pytest fixture that provides a factory for temporary directories.
            dat_header (str): The header string for the .dat file.
            dat_random_data (numpy.ndarray): The data to write to the .dat file.

        Returns:
            Path: Path to the created temporary test file.
        """
        # Create a temporary directory
        tmp_path = tmp_path_factory.mktemp("test_data")

        # Create column names
        columns = ["Bias calc (V)", "Current (A)", "LI Demod 1 X (A)", "LI Demod 2 X (A)", "Bias (V)"]

        # Create the temporary file path
        dat_file = tmp_path / "test_bias_spectroscopy.dat"

        with open(dat_file, "w") as f:
            # Write the header
            f.write(dat_header)
            f.write("\n")

            # Add the data section marker
            f.write("[DATA]\n")

            # Write column names
            f.write("\t".join(columns) + "\n")

            # Write the data rows
            for row in dat_data:
                row_str = "\t".join([f"{val:E}" for val in row])
                f.write(row_str + "\n")

        return dat_file

    @pytest.fixture(scope="class", autouse=True)
    def ds(self, dat_std_file):
        """
        Reads the .dat file into an xarray dataset.

        Args:
            dat_std_file (Path): Path to the temporary .dat file created for testing.

        Returns:
            xarray.Dataset: The dataset created from the .dat file.
        """
        return xr.load_dataset(dat_std_file)

    def test_header(self, ds, dat_header_dict):
        """
        Tests if the header information is correctly read into the dataset.

        Args:
            ds (xarray.Dataset): The dataset created from the .dat file.
            dat_header_dict (dict): Dictionary containing expected header key-value pairs.
        """
        # Check if the header information is correctly read
        assert ds.attrs == dat_header_dict

    def test_data_values(self, ds, dat_data):
        """
        Tests if the data values in the dataset match the expected random data.

        Args:
            ds (xarray.Dataset): The dataset created from the .dat file.
            dat_data (numpy.ndarray): The random data used to create the .dat file.
        """
        # Check if the data values are correctly read
        np.testing.assert_almost_equal(ds["bias"].values, dat_data[:, 0])
        np.testing.assert_almost_equal(ds["Current"].values, dat_data[:, 1])
        np.testing.assert_almost_equal(ds["LI_Demod_1_X"].values, dat_data[:, 2])
        np.testing.assert_almost_equal(ds["LI_Demod_2_X"].values, dat_data[:, 3])
        np.testing.assert_almost_equal(ds["Bias"].values, dat_data[:, 4])

    def test_data_attrs(self, ds):
        """
        Tests if the data variables in the dataset have the correct attributes.

        Args:
            ds (xarray.Dataset): The dataset created from the .dat file.
        """
        # Check if the data variables have the correct attributes
        assert ds["bias"].attrs == {"standard_name": "Sample bias", "units": "V"}
        assert ds["Current"].attrs == {"standard_name": "Current", "units": "A"}
        assert ds["LI_Demod_1_X"].attrs == {"standard_name": "dI/dV", "units": "A"}
        assert ds["LI_Demod_2_X"].attrs == {"standard_name": "dI/dV", "units": "A"}
        assert ds["Bias"].attrs == {"standard_name": "Sample bias", "units": "V"}
