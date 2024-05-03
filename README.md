# ROMS-tools

## Overview

A suite of python tools for setting up a [ROMS](https://github.com/CESR-lab/ucla-roms) simulation.

_Note these tools are for the [Center for Earth Systems Research Group](http://research.atmos.ucla.edu/cesr/) at UCLA's version of ROMS._

## Installation instructions

### Install via pip

```bash
pip install roms-tools
```

### Installation from GitHub

ROMS-tools is under active development. To obtain the latest development version, you may clone the source repository and install it:
```bash
git clone https://github.com/CWorthy-ocean/roms-tools.git
cd roms-tools
pip install -e . --no-deps 
```

### Running the tests

Check the installation has worked by running the tests (you will need to also install pytest to run these)
```bash
pytest
```

Dependencies required are xarray, scipy, netcdf4, and pooch, plus matplotlib and cartopy for visualising grids.

ROMS-tools should run on any platform that can install the above dependencies.

You can set up a conda environment with all required as follows:
```bash
cd roms-tools
conda env create -f ci/environment.yml
conda activate romstools-test
```

## Usage instructions

To set up all the input files for a new ROMS simulation from scratch, follow these steps in order.

### Step 1: Make Grid

The first step is choosing the domain size, location, and resolution. Do this by creating an instance of the `Grid` class:

```python
from roms_tools import Grid

grid = Grid(
    nx=100,          # number of points in the x-direction (not including 2 boundary cells on either end)
    ny=100,          # number of points in the y-direction (not including 2 boundary cells on either end)
    size_x=1800,     # size of the domain in the x-direction (in km)
    size_y=2400,     # size of the domain in the y-direction (in km)
    center_lon=-21,  # longitude of the center of the domain
    center_lat=61,   # latitude of the center of the domain
    rot=20,          # rotation of the grid's x-direction from lines of constant longitude, with positive values being a counter-clockwise rotation
)
```

To visualize the grid we have just created, use the `.plot` method:

```python
grid.plot()
```

![iceland_grid](https://github.com/CWorthy-ocean/roms-tools/assets/35968931/de8c03ab-3c61-4ba5-a9b7-65592fd9280f)

To see the values of the grid variables you can examine the `xarray.Dataset` object returned by the `.ds` property

```python
grid.ds
```
```
<xarray.Dataset>
Dimensions:    (eta_rho: 3, xi_rho: 3, one: 1)
Dimensions without coordinates: eta_rho, xi_rho, one
Data variables:
    angle      (eta_rho, xi_rho) float64 0.0 0.0 0.0 -1.46e-16 ... 0.0 0.0 0.0
    f0         (eta_rho, xi_rho) float64 4.565e-06 4.565e-06 ... -4.565e-06
    pn         (eta_rho, xi_rho) float64 5e-06 5e-06 5e-06 ... 5e-06 5e-06 5e-06
    lon_rho    (eta_rho, xi_rho) float64 339.1 340.0 340.9 ... 339.1 340.0 340.9
    lat_rho    (eta_rho, xi_rho) float64 1.799 1.799 1.799 ... -1.799 -1.799
    spherical  (one) <U1 'T'
    tra_lon    (one) int64 -20
    tra_lat    (one) int64 0
    rotate     (one) int64 0
Attributes:
    Title:    ROMS grid. Settings: nx: 1 ny: 1  xsize: 0.1 ysize: 0.1 rotate:...
    Date:     2023-11-20
    Type:     ROMS grid produced by roms-tools
```

Once we are happy with our grid, we can save it as a netCDF file via the `.save` method:

```python
grid.save('grids/my_new_roms_grid.nc')
```

The basic grid domain is now ready for use by ROMS.


### Steps 2-7:

Coming soon!


## Feedback and contributions

**If you find a bug, have a feature suggestion, or any other kind of feedback, please start a Discussion.**

We also accept contributions in the form of Pull Requests.


## See also

- [ROMS source code](https://github.com/CESR-lab/ucla-roms)
- [C-Star](https://github.com/CWorthy-ocean/C-Star)
