"""
utils
"""

import re
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

# Regular expressions for channel names
lockin_re = re.compile(r"LI[_ ]Demod[_ ][1, 2][_ ][X, Y].*")


def _construct_var(dims, data, standard_name, units, attrs=None):
    if attrs is None:
        attrs = {"standard_name": standard_name, "units": units}
    else:
        attrs["standard_name"] = standard_name
        attrs["units"] = units
    coord = xr.Variable(
        dims,
        data,
        attrs=attrs,
    )
    return coord


def _parse_header_table(table_list):
    """
    Parse table in header into dictionary.

    :param table_list: list of every line of a table.
    :type table_list: list of str
    :return: dictionary of every column of the table.
    :rtype: dict
    """
    # Use table_array to store the processed array
    table_array = []
    # split each line in table_list to arraies.
    for entry in table_list:
        entry = entry.strip("\t").split("\t")
        table_array.append(entry)

    table_header = table_array[0]
    table_values = zip(*table_array[1:])

    table_dict = dict(zip(table_header, table_values))

    return table_dict


def _get_std_name(name: str) -> str:
    """
    Get standard name of channel

    Parameters
    ----------
    name : str
        Name of the channel (from the file).

    Returns
    -------
    str
        Standard name of the channel.
    """
    if name == "bias":
        std_name = "Sample bias"
    elif lockin_re.match(name):
        std_name = "dI/dV"
    elif name == "Z":
        std_name = "Height"
    elif name == "dir":
        std_name = "Scan direction"
    else:
        std_name = name
    return std_name


def _name(name: str | list[str]) -> str:
    """
    Process the name to have consistent format

    Replace all spaces within the name with "_".

    Parameters
    ----------
    name : str | list[str]
        Name of the channel.

    Returns
    -------
    str | list[str]
        Name with consistent format.
    """
    if isinstance(name, str):
        name = name.replace(" ", "_")
    else:
        name = [s.replace(" ", "_") for s in name]
    return name


def _separate_name_unit(channels: list[str]) -> tuple[list[str], dict[str, str]]:
    """
    Separate the names and units of every channel.

    Parameters
    ----------
    channels : list[str]
        Information of channels read from file.

    Returns
    -------
    channel_names : list[str]
        List of names of all the channels.
    channel_units : dict[str, str]
        Dictionary containing name and unit pairs of each channel.
    """
    names: list[str] = []
    units: dict[str, str] = {}
    # regular expression matches unit.
    re_pattern = re.compile(r" \(.\)")
    for channel in channels:
        if re_pattern.search(channel):
            # split the channel names by the match of unit.
            name_split = re_pattern.split(channel)
            # combine all the split strings to form the name of channel.
            channel_name = ""
            for i in name_split:
                channel_name += i
            channel_name = _name(channel_name)
            # search the unit of the channel
            channel_unit = re_pattern.search(channel).group().strip("() ")
        else:
            # If the channel does not contain unit, the unit is set to None.
            channel_name = _name(channel)
            channel_unit = None
        names.append(channel_name)
        units[channel_name] = channel_unit
    return names, units


def _handle_multilevel_header(header_dict, key, value):
    if ">" in key:
        level_0, level_1 = key.split(">", 1)
        sub_dict = header_dict.setdefault(level_0, dict())
        sub_dict[level_1] = value
    else:
        header_dict[key] = value
    return header_dict
