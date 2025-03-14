"""
Test reading of sxm files
"""

import numpy as np
import pytest
import xarray as xr

import xarray_nanonis


@pytest.fixture(scope="class")
def sxm_data(sxm_pixels):
    rng = np.random.default_rng()
    return rng.random(3 * 2 * sxm_pixels[0] * sxm_pixels[1], dtype=np.float32)


@pytest.fixture(scope="class")
def sxm_header_dict(sxm_pixels, sxm_scan_direction, sxm_scan_range):
    return {
        "NANONIS_VERSION": "2",
        "SCANIT_TYPE": "FLOAT MSBFIRST",
        "SCAN_PIXELS": f"{sxm_pixels[0]} {sxm_pixels[1]}",
        "SCAN_RANGE": sxm_scan_range,
        "SCAN_OFFSET": "1E-7 -1E-7",
        "SCAN_ANGLE": "0E+0",
        "SCAN_DIR": sxm_scan_direction,
        "BIAS": "-1E+0",
        "COMMENT": "",
        "Bias": {
            "Bias (V)": "-1E+0",
            "Calibration (V/V)": "-1E+0",
            "Offset (V)": "-1E-3",
        },
        "DATA_INFO": {
            "Channel": ("14", "12", "13"),
            "Name": ("Z", "X", "Y"),
            "Unit": ("m", "m", "m"),
            "Direction": ("both", "both", "both"),
            "Calibration": ("1.000E-8", "7.000E-8", "7.000E-8"),
            "Offset": ("0.000E+0", "0.000E+0", "0.000E+0"),
        },
    }


@pytest.fixture(scope="class")
def sxm_header(sxm_pixels, sxm_scan_direction, sxm_scan_range) -> str:
    header = "\n".join(
        [
            ":NANONIS_VERSION:",
            "2",
            ":SCANIT_TYPE:",
            "FLOAT MSBFIRST",
            ":SCAN_PIXELS:",
            f"{sxm_pixels[0]} {sxm_pixels[1]}",
            ":SCAN_RANGE:",
            sxm_scan_range,
            ":SCAN_OFFSET:",
            "1E-7 -1E-7",
            ":SCAN_ANGLE:",
            "0E+0",
            ":SCAN_DIR:",
            sxm_scan_direction,
            ":BIAS:",
            "-1E+0",
            ":COMMENT:",
            "",
            ":Bias>Bias (V):",
            "-1E+0",
            ":Bias>Calibration (V/V):",
            "-1E+0",
            ":Bias>Offset (V):",
            "-1E-3",
            ":DATA_INFO:",
            "\t" + "\t".join(["Channel", "Name", "Unit", "Direction", "Calibration", "Offset"]),
            "\t" + "\t".join(["14", "Z", "m", "both", "1.000E-8", "0.000E+0"]),
            "\t" + "\t".join(["12", "X", "m", "both", "7.000E-8", "0.000E+0"]),
            "\t" + "\t".join(["13", "Y", "m", "both", "7.000E-8", "0.000E+0"]),
            "",
            ":SCANIT_END:\n\n\n",
        ]
    )
    return header


@pytest.fixture(scope="class")
def sxm_std_file(tmp_path_factory, sxm_data, sxm_header):
    file_path = tmp_path_factory.mktemp("sxm_file") / "sxm_std.sxm"
    with open(file_path, mode="wb") as f:
        # Write the header in UTF-8 encoding
        f.write(sxm_header.encode("utf-8"))
        # Write the start of the data section
        f.write(b"\x1a\x04")
        # Write binary data - float32 in MSB (big-endian) order
        f.write(sxm_data.astype(">f4").tobytes())
    return file_path


class TestSxmFile:
    @pytest.fixture(scope="class")
    def sxm_pixels(self):
        """
        Fixture to provide the number of pixels in the sxm file
        """
        return 512, 512

    @pytest.fixture(scope="class")
    def sxm_scan_range(self):
        return "1E-7 1E-7"

    @pytest.fixture(scope="class")
    def sxm_scan_direction(self):
        return "down"

    @pytest.fixture(scope="class", autouse=True)
    def ds(self, sxm_std_file):
        """
        Fixture to provide the sxm dataset
        """
        ds = xr.load_dataset(sxm_std_file)
        return ds

    def test_data_values(self, ds, sxm_data, sxm_pixels):
        """
        Test the data values of the dataset.
        """
        data = sxm_data.reshape((3, 2, sxm_pixels[1], sxm_pixels[0]))
        for i in range(3):
            data[i, 0] = data[i, 0][::-1, :]
            data[i, 1] = data[i, 1][::-1, ::-1]
        for i, var in enumerate(ds.keys()):
            np.testing.assert_equal(ds[var].values, data[i])

    def test_dim(self, ds, sxm_pixels, sxm_scan_range):
        """
        Test the dimensions of the dataset.
        """
        scan_range = [float(x) for x in sxm_scan_range.split()]
        np.testing.assert_equal(ds["x"].values, np.linspace(0, scan_range[0], sxm_pixels[0]))
        np.testing.assert_equal(ds["y"].values, np.linspace(0, scan_range[1], sxm_pixels[1]))
        assert ds["x"].attrs["units"] == "m"
        assert ds["y"].attrs["units"] == "m"

    def test_header_attrs(self, ds, sxm_header_dict):
        """
        Test the header attributes of the dataset.
        """
        assert ds.attrs == sxm_header_dict
        # for key in sxm_header_dict.keys():
        #     if isinstance(sxm_header_dict[key], dict):
        #         for subkey in sxm_header_dict[key].keys():
        #             assert ds.attrs[key][subkey] == sxm_header_dict[key][subkey]
        #     else:
        #         assert ds.attrs[key] == sxm_header_dict[key]


class TestUpSxmFile(TestSxmFile):
    @pytest.fixture(scope="class")
    def sxm_scan_direction(self):
        return "up"

    def test_data_values(self, ds, sxm_data, sxm_pixels):
        data = sxm_data.reshape((3, 2, sxm_pixels[1], sxm_pixels[0]))
        for i in range(3):
            data[i, 0] = data[i, 0][:, :]
            data[i, 1] = data[i, 1][:, ::-1]
        np.testing.assert_equal(ds["Z"].values, data[0])
        np.testing.assert_equal(ds["X"].values, data[1])
        np.testing.assert_equal(ds["Y"].values, data[2])


class TestNonSquareSxmFile(TestSxmFile):
    @pytest.fixture(scope="class")
    def sxm_pixels(self):
        """
        Fixture to provide the number of pixels in the sxm file
        """
        return 512, 1024

    @pytest.fixture(scope="class")
    def sxm_scan_range(self):
        return "1E-7 2E-7"


class TestOneDIRSxmFile(TestSxmFile):
    @pytest.fixture(scope="class")
    def sxm_data(self, sxm_pixels):
        rng = np.random.default_rng()
        return rng.random(sxm_pixels[0] * sxm_pixels[1], dtype=np.float32)

    @pytest.fixture(scope="class")
    def sxm_header_dict(self, sxm_pixels, sxm_scan_direction, sxm_scan_range):
        return {
            "NANONIS_VERSION": "2",
            "SCANIT_TYPE": "FLOAT MSBFIRST",
            "SCAN_PIXELS": f"{sxm_pixels[0]} {sxm_pixels[1]}",
            "SCAN_RANGE": sxm_scan_range,
            "SCAN_OFFSET": "1E-7 -1E-7",
            "SCAN_ANGLE": "0E+0",
            "SCAN_DIR": sxm_scan_direction,
            "BIAS": "-1E+0",
            "COMMENT": "",
            "Bias": {
                "Bias (V)": "-1E+0",
                "Calibration (V/V)": "-1E+0",
                "Offset (V)": "-1E-3",
            },
            "DATA_INFO": {
                "Channel": ("14",),
                "Name": ("Z",),
                "Unit": ("m",),
                "Direction": ("both",),
                "Calibration": ("1.000E-8",),
                "Offset": ("0.000E+0",),
            },
        }

    @pytest.fixture(scope="class")
    def sxm_header(self, sxm_pixels, sxm_scan_range, sxm_scan_direction):
        header = "\n".join(
            [
                ":NANONIS_VERSION:",
                "2",
                ":SCANIT_TYPE:",
                "FLOAT MSBFIRST",
                ":SCAN_PIXELS:",
                f"{sxm_pixels[0]} {sxm_pixels[1]}",
                ":SCAN_RANGE:",
                sxm_scan_range,
                ":SCAN_OFFSET:",
                "1E-7 -1E-7",
                ":SCAN_ANGLE:",
                "0E+0",
                ":SCAN_DIR:",
                sxm_scan_direction,
                ":BIAS:",
                "-1E+0",
                ":COMMENT:",
                "",
                ":Bias>Bias (V):",
                "-1E+0",
                ":Bias>Calibration (V/V):",
                "-1E+0",
                ":Bias>Offset (V):",
                "-1E-3",
                ":DATA_INFO:",
                "\t" + "\t".join(["Channel", "Name", "Unit", "Direction", "Calibration", "Offset"]),
                "\t" + "\t".join(["14", "Z", "m", "both", "1.000E-8", "0.000E+0"]),
                "",
                ":SCANIT_END:\n\n\n",
            ]
        )
        return header

    def test_data_values(self, ds, sxm_data, sxm_pixels):
        data = sxm_data.reshape(sxm_pixels[0], sxm_pixels[1])
        data = data[::-1, :]
        np.testing.assert_array_equal(ds["Z"].values[0], data)
        np.testing.assert_array_equal(ds["Z"].values[1], np.ones_like(data))


class TestIncompleteSxmFile(TestSxmFile):
    @pytest.fixture(scope="class")
    def sxm_data(self, sxm_pixels):
        rng = np.random.default_rng()
        data = rng.random(3 * 2 * sxm_pixels[0] * sxm_pixels[1], dtype=np.float32)
        data = data.reshape(3, 2, sxm_pixels[1], sxm_pixels[0])
        data[:, :, 300:, :] = np.nan
        return data.flatten()


class TestMultipassSxmFile(TestSxmFile):
    @pytest.fixture(scope="class")
    def sxm_header(self, sxm_pixels, sxm_scan_direction, sxm_scan_range):
        header = "\n".join(
            [
                ":NANONIS_VERSION:",
                "2",
                ":SCANIT_TYPE:",
                "FLOAT MSBFIRST",
                ":SCAN_PIXELS:",
                f"{sxm_pixels[0]} {sxm_pixels[1]}",
                ":SCAN_RANGE:",
                sxm_scan_range,
                ":SCAN_OFFSET:",
                "1E-7 -1E-7",
                ":SCAN_ANGLE:",
                "0E+0",
                ":SCAN_DIR:",
                sxm_scan_direction,
                ":BIAS:",
                "-1E+0",
                ":COMMENT:",
                "",
                ":Bias>Bias (V):",
                "-1E+0",
                ":Bias>Calibration (V/V):",
                "-1E-3",
                ":Bias>Offset (V):",
                "-1E-3",
                ":Multipass-Config:",
                "\t"
                + "\t".join(
                    [
                        "Record-Ch",
                        "Playback",
                        "Playback-Offset",
                        "BOL-delay_[cycles]",
                        "Bias_override",
                        "Bias_override_value",
                        "Z_Setp_override",
                        "Z_Setp_override_value",
                        "Speed_factor",
                    ]
                ),
                "\t"
                + "\t".join(["-1", "FALSE", "0.000E+0", "40000", "TRUE", "-2.000E-2", "TRUE", "3.000E-10", "1.000"]),
                "\t"
                + "\t".join(["-1", "FALSE", "0.000E+0", "40000", "TRUE", "-4.000E-2", "TRUE", "3.000E-10", "1.000"]),
                "\t"
                + "\t".join(["-1", "FALSE", "0.000E+0", "40000", "TRUE", "-6.000E-2", "TRUE", "4.000E-10", "1.000"]),
                "\t"
                + "\t".join(["-1", "FALSE", "0.000E+0", "40000", "TRUE", "-8.000E-2", "TRUE", "4.000E-10", "1.000"]),
                "\t"
                + "\t".join(["-1", "FALSE", "0.000E+0", "40000", "TRUE", "-7.000E-2", "TRUE", "4.000E-10", "1.000"]),
                "\t"
                + "\t".join(["-1", "FALSE", "0.000E+0", "40000", "TRUE", "-5.000E-2", "TRUE", "4.000E-10", "1.000"]),
                ":DATA_INFO:",
                "\t" + "\t".join(["Channel", "Name", "Unit", "Direction", "Calibration", "Offset"]),
                "\t" + "\t".join(["14", "[P1]_Z", "m", "both", "1.000E-8", "0.000E+0"]),
                "\t" + "\t".join(["14", "[P2]_Z", "m", "both", "1.000E-8", "0.000E+0"]),
                "\t" + "\t".join(["14", "[P3]_Z", "m", "both", "1.000E-8", "0.000E+0"]),
                "",
                ":SCANIT_END:\n\n\n",
            ]
        )
        return header

    @pytest.fixture(scope="class")
    def sxm_header_dict(self, sxm_pixels, sxm_scan_direction, sxm_scan_range):
        return {
            "NANONIS_VERSION": "2",
            "SCANIT_TYPE": "FLOAT MSBFIRST",
            "SCAN_PIXELS": f"{sxm_pixels[0]} {sxm_pixels[1]}",
            "SCAN_RANGE": sxm_scan_range,
            "SCAN_OFFSET": "1E-7 -1E-7",
            "SCAN_ANGLE": "0E+0",
            "SCAN_DIR": sxm_scan_direction,
            "BIAS": "-1E+0",
            "COMMENT": "",
            "Bias": {
                "Bias (V)": "-1E+0",
                "Calibration (V/V)": "-1E-3",
                "Offset (V)": "-1E-3",
            },
            "Multipass-Config": {
                "fwd_0": {
                    "Record-Ch": "-1",
                    "Playback": "FALSE",
                    "Playback-Offset": "0.000E+0",
                    "BOL-delay_[cycles]": "40000",
                    "Bias_override": "TRUE",
                    "Bias_override_value": "-2.000E-2",
                    "Z_Setp_override": "TRUE",
                    "Z_Setp_override_value": "3.000E-10",
                    "Speed_factor": "1.000",
                },
                "bwd_0": {
                    "Record-Ch": "-1",
                    "Playback": "FALSE",
                    "Playback-Offset": "0.000E+0",
                    "BOL-delay_[cycles]": "40000",
                    "Bias_override": "TRUE",
                    "Bias_override_value": "-4.000E-2",
                    "Z_Setp_override": "TRUE",
                    "Z_Setp_override_value": "3.000E-10",
                    "Speed_factor": "1.000",
                },
                "fwd_1": {
                    "Record-Ch": "-1",
                    "Playback": "FALSE",
                    "Playback-Offset": "0.000E+0",
                    "BOL-delay_[cycles]": "40000",
                    "Bias_override": "TRUE",
                    "Bias_override_value": "-6.000E-2",
                    "Z_Setp_override": "TRUE",
                    "Z_Setp_override_value": "4.000E-10",
                    "Speed_factor": "1.000",
                },
                "bwd_1": {
                    "Record-Ch": "-1",
                    "Playback": "FALSE",
                    "Playback-Offset": "0.000E+0",
                    "BOL-delay_[cycles]": "40000",
                    "Bias_override": "TRUE",
                    "Bias_override_value": "-8.000E-2",
                    "Z_Setp_override": "TRUE",
                    "Z_Setp_override_value": "4.000E-10",
                    "Speed_factor": "1.000",
                },
                "fwd_2": {
                    "Record-Ch": "-1",
                    "Playback": "FALSE",
                    "Playback-Offset": "0.000E+0",
                    "BOL-delay_[cycles]": "40000",
                    "Bias_override": "TRUE",
                    "Bias_override_value": "-7.000E-2",
                    "Z_Setp_override": "TRUE",
                    "Z_Setp_override_value": "4.000E-10",
                    "Speed_factor": "1.000",
                },
                "bwd_2": {
                    "Record-Ch": "-1",
                    "Playback": "FALSE",
                    "Playback-Offset": "0.000E+0",
                    "BOL-delay_[cycles]": "40000",
                    "Bias_override": "TRUE",
                    "Bias_override_value": "-5.000E-2",
                    "Z_Setp_override": "TRUE",
                    "Z_Setp_override_value": "4.000E-10",
                    "Speed_factor": "1.000",
                },
            },
            "DATA_INFO": {
                "Channel": ("14", "14", "14"),
                "Name": ("[P1]_Z", "[P2]_Z", "[P3]_Z"),
                "Unit": ("m", "m", "m"),
                "Direction": ("both", "both", "both"),
                "Calibration": ("1.000E-8", "1.000E-8", "1.000E-8"),
                "Offset": ("0.000E+0", "0.000E+0", "0.000E+0"),
            },
        }
