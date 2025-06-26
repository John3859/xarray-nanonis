# xarray-nanonis

[![PyPI version](https://badge.fury.io/py/xarray-nanonis.svg)](https://badge.fury.io/py/xarray-nanonis)
[![Python](https://img.shields.io/pypi/pyversions/xarray-nanonis.svg)](https://pypi.org/project/xarray-nanonis/)
[![License](https://img.shields.io/github/license/John3859/xarray-nanonis.svg)](LICENSE)

**Extending xarray to read Nanonis files with coordinates and metadata.**

`xarray-nanonis` is a Python package that integrates Nanonis data files into the xarray ecosystem. It provides support for reading `.sxm`, `.dat`, and `.3ds` files directly into xarray Datasets, complete with proper dimensions, units, coordinate information, and metadata preservation.

## Features

- **Xarray integration**: Load Nanonis files directly as xarray Datasets using `open_dataset()`;
- **Full metadata preservation**: Header information is preserved as Dataset attributes;
- **Proper dimensions and coordinates**: Spatial (x, y) and energy (bias) coordinates with units;
- **Multi-channel support**: Multiple measurement channels can be retrieved from datasets easily; 
- **Unit handling**: Automatic parsing and assignment of physical units;
- **Scan direction support**: Forward/backward scan data for `.sxm` files;
- **Advanced spectroscopy features**: Support for Multi-Linear Segments (MLS) bias sweeps;

## Installation

Install from PyPI:

```bash
pip install xarray-nanonis
```

## Quick Start

### Loading Nanonis files with xarray

Once installed, you can load Nanonis files directly using xarray's `open_dataset()`:

```python
import xarray as xr

# Load a topography scan
ds_topo = xr.open_dataset('topography.sxm')
# or
ds_topo = xr.open_dataset('topography.sxm', engine='nanonis')

# Load point spectroscopy data  
ds_spec = xr.open_dataset('spectrum.dat')
# or
ds_spec = xr.open_dataset('spectrum.dat', engine='nanonis')

# Load grid spectroscopy data
ds_grid = xr.open_dataset('grid_spectroscopy.3ds')
# or 
ds_grid = xr.open_dataset('grid_spectroscopy.3ds', engine='nanonis')
```

## Usage Examples

### Topography Data (.sxm files)

```python
import xarray as xr
import matplotlib.pyplot as plt

# Load topography data
ds = xr.open_dataset('topography.sxm', engine='nanonis')

# The dataset contains multi-channel data
print(ds.data_vars)  # Shows available channels (e.g., 'Z', 'Current', etc.)

# Metadata is stored in attributes
print(ds.attrs)  # Contains scan parameters, bias, etc.

# Plot topography
ds['Z'].sel(dir='forward').plot.pcolormesh() # Forward scan topography
plt.title(f"Topography - Bias: {ds.attrs['BIAS']} V")
plt.show()

# Access scan parameters
print(f"Scan size: {ds.attrs['SCAN_RANGE']} m")
print(f"Pixels: {ds.attrs['SCAN_PIXELS']}")
```

### Point Spectroscopy (.dat files)

```python
# Load I-V spectroscopy
ds = xr.open_dataset('dIdV_spectrum.dat', engine='nanonis')

# Plot dI/dV vs bias
ds['LI_Demod_1_X'].plot(x='bias')
plt.show()
```

### Grid Spectroscopy (.3ds files)

```python
# Load grid spectroscopy data
ds = xr.open_dataset('grid_spectroscopy.3ds', engine='nanonis')

# Access spectroscopy grid
dIdV_map = ds['LI_Demod_1_X']  # Shape: (bias, y, x)

# Plot dI/dV map at specific bias
dIdV_map.sel(bias=0.1, method='nearest').plot.pcolormesh()
plt.show()
```

### Advanced: Custom voltage divider

For systems with voltage dividers, you can specify the division factor:

```python
# For a system with 10:1 voltage divider
ds = xr.open_dataset('data.sxm', engine='nanonis', divider=10)
```

However, this only works for coordinates, not metadata.

## Data Structure

All Nanonis files are loaded as xarray Datasets with:

- **Coordinates**: Spatial (x, y) and energy (bias) dimensions with units
- **Data variables**: Measurement channels (Current, Topography, Lock-in signals, etc.)
- **Attributes**: Complete header information
- **Units**: Proper unit handling for all variables and coordinates

Example dataset structure:
```
<xarray.Dataset>
Dimensions:  (x: 256, y: 256, dir: 2)
Coordinates:
  * x        (x) float64 0.0 1.95e-10 3.91e-10 ... 4.98e-08 5.0e-08
  * y        (y) float64 0.0 1.95e-10 3.91e-10 ... 4.98e-08 5.0e-08  
  * dir      (dir) <U8 'forward' 'backward'
Data variables:
    Z        (dir, y, x) float32 ...
    Current  (dir, y, x) float32 ...
Attributes:
    SCAN_PIXELS:  512 512
    SCAN_RANGE:   5.0e-08 5.0e-08
    BIAS:         0.1
    ...
```

## Requirements

- Python ≥ 3.9
- xarray
- numpy  
- pandas

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This package extends the functionality of [xarray](https://xarray.pydata.org/) for the scanning probe microscopy community. Special thanks to the [xarray](https://xarray.pydata.org/) development team for creating such a powerful and flexible data analysis toolkit, and to the [nanonispy](https://github.com/underchemist/nanonispy) project for providing the foundation for parsing Nanonis file formats.
