# ROMS-tools

## Overview

A suite of python tools for setting up a [ROMS](https://github.com/CESR-lab/ucla-roms) simulation.

_Note these tools are for the [Center for Earth Systems Research Group](http://research.atmos.ucla.edu/cesr/) at UCLA's version of ROMS._

## Installation instructions

### Install via pip

```bash
pip install roms-tools
```

### Install from GitHub

ROMS-tools is under active development. To obtain the latest development version, you may clone the source repository and install it:
```bash
git clone https://github.com/CWorthy-ocean/roms-tools.git
cd roms-tools
pip install -e . --no-deps 
```

### Run the tests

Check the installation has worked by running the tests (you will need to also install pytest to run these)
```bash
pytest
```

Dependencies required are xarray, scipy, netcdf4, and pooch, plus matplotlib and cartopy for visualising grids.

ROMS-tools should run on any platform that can install the above dependencies.

You can set up a conda environment with all required dependencies as follows:
```bash
cd roms-tools
conda env create -f ci/environment.yml
conda activate romstools-test
```

## Usage instructions

To set up all the input files for a new ROMS simulation from scratch, follow these steps in order.

### Step 1: Make Grid and Topography

The first step is choosing the domain size, location, resolution, and topography options. Do this by creating an instance of the `Grid` class:

```python
from roms_tools import Grid

grid = Grid(
    nx=100,                     # number of points in the x-direction (not including 2 boundary cells on either end)
    ny=100,                     # number of points in the y-direction (not including 2 boundary cells on either end)
    size_x=1800,                # size of the domain in the x-direction (in km)
    size_y=2400,                # size of the domain in the y-direction (in km)
    center_lon=-21,             # longitude of the center of the domain
    center_lat=61,              # latitude of the center of the domain
    rot=20,                     # rotation of the grid's x-direction from lines of constant longitude, with positive values being a counter-clockwise rotation
    topography_source='etopo5', # data source to use for the topography
    smooth_factor=2,            # smoothing factor used in the global Gaussian smoothing of the topography, default: 2
    hmin=5,                     # minimum ocean depth (in m), default: 5
    rmax=0.2,                   # maximum slope parameter (in m), default: 0.2
)
```

To visualize the grid we have just created, use the `.plot` method:

```python
grid.plot(bathymetry=True)
```

<img width="786" alt="Screenshot 2024-05-14 at 2 29 03 PM" src="https://github.com/NoraLoose/roms-tools/assets/23617395/6e364b50-d367-49e0-b1b5-f7e1662b1338">

To see the values of the grid variables you can examine the `xarray.Dataset` object returned by the `.ds` property

```python
grid.ds
```
```
<xarray.Dataset>
Dimensions:   (eta_rho: 102, xi_rho: 102)
Coordinates:
    lat_rho   (eta_rho, xi_rho) float64 47.84 47.91 47.97 ... 73.51 73.53
    lon_rho   (eta_rho, xi_rho) float64 333.0 333.3 333.5 ... 352.6 353.2
Dimensions without coordinates: eta_rho, xi_rho
Data variables:
    angle     (eta_rho, xi_rho) float64 0.4177 0.4177 ... 0.1146 0.1146
    f0        (eta_rho, xi_rho) float64 0.0001078 0.0001079 ... 0.0001395
    pm        (eta_rho, xi_rho) float64 4.209e-05 4.208e-05 ... 4.209e-05
    pn        (eta_rho, xi_rho) float64 5.592e-05 5.592e-05 ... 5.592e-05
    tra_lon   int64 -21
    tra_lat   int64 61
    rotate    int64 20
    hraw      (eta_rho, xi_rho) float64 2.662e+03 2.837e+03 ... 3.19e+03
    mask_rho  (eta_rho, xi_rho) int64 1 1 1 1 1 1 1 1 1 ... 1 1 1 1 1 1 1 1
    h         (eta_rho, xi_rho) float64 2.875e+03 2.875e+03 ... 2.972e+03
Attributes:
    Title:                     ROMS grid. Settings: nx: 100 ny: 100 xsize: 1....
    Date:                      2024-05-14
    Type:                      ROMS grid produced by roms-tools
    Topography source:         etopo5
    Topography modifications:  Global smoothing with factor 2; Minimal depth:...
```

Once we are happy with our grid, we can save it as a netCDF file via the `.save` method:

```python
grid.save('grids/my_new_roms_grid.nc')
```

The basic grid domain is now ready for use by ROMS.


### More steps:

Coming soon!


## Feedback and contributions

**If you find a bug, have a feature suggestion, or any other kind of feedback, please start a Discussion.**

We also accept contributions in the form of Pull Requests.


## See also

- [ROMS source code](https://github.com/CESR-lab/ucla-roms)
- [C-Star](https://github.com/CWorthy-ocean/C-Star)
