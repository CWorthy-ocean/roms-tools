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

![Screenshot 2024-05-20 at 10 34 05 AM](https://github.com/NoraLoose/roms-tools/assets/23617395/4c0411fd-8195-4fcf-9837-9ec32f7ff23b)

To see the values of the grid variables you can examine the `xarray.Dataset` object returned by the `.ds` property

```python
grid.ds
```
```
<xarray.Dataset> Size: 749kB
Dimensions:   (eta_rho: 102, xi_rho: 102)
Coordinates:
    lat_rho   (eta_rho, xi_rho) float64 83kB 47.84 47.91 47.97 ... 73.51 73.53
    lon_rho   (eta_rho, xi_rho) float64 83kB 333.0 333.3 333.5 ... 352.6 353.2
Dimensions without coordinates: eta_rho, xi_rho
Data variables:
    angle     (eta_rho, xi_rho) float64 83kB 0.4177 0.4177 ... 0.1146 0.1146
    f         (eta_rho, xi_rho) float64 83kB 0.0001078 0.0001079 ... 0.0001395
    pm        (eta_rho, xi_rho) float64 83kB 4.209e-05 4.208e-05 ... 4.209e-05
    pn        (eta_rho, xi_rho) float64 83kB 5.592e-05 5.592e-05 ... 5.592e-05
    tra_lon   int64 8B -21
    tra_lat   int64 8B 61
    rotate    int64 8B 20
    hraw      (eta_rho, xi_rho) float64 83kB 2.662e+03 2.837e+03 ... 3.19e+03
    mask_rho  (eta_rho, xi_rho) int64 83kB 1 1 1 1 1 1 1 1 1 ... 1 1 1 1 1 1 1 1
    h         (eta_rho, xi_rho) float64 83kB 2.875e+03 2.875e+03 ... 2.972e+03
Attributes:
    Type:               ROMS grid produced by roms-tools
    size_x:             1800
    size_y:             2400
    topography_source:  etopo5
    smooth_factor:      2
    hmin:               5
    rmax:               0.2

```

Once we are happy with our grid, we can save it as a netCDF file via the `.save` method:

```python
grid.save('grids/my_new_roms_grid.nc')
```

The basic grid domain is now ready for use by ROMS.

We can also create a grid from an existing file:

```python
the_same_grid = Grid.from_file('grids/my_new_roms_grid.nc')
```

`grid` and `the_same_grid` are identical!

### Step 2: Make Tidal Forcing

Once we have created a grid in Step 1, we can make the tidal forcing for this grid. 

```python
from roms_tools import TidalForcing
from datetime import datetime

tidal_forcing = TidalForcing(
    grid=grid,                                 # The grid object representing the ROMS grid associated with the tidal forcing data
    filename='tidal_data.nc',                  # The path to the tidal dataset file.
    nc=10,                                     # Number of constituents to consider. Maximum number is 14. Default is 10.
    model_reference_date=datetime(2000, 1, 1)  # The reference date for the ROMS simulation. Default is datetime(2000, 1, 1).
    source="tpxo",                             # The source of the tidal data. Default is "tpxo".
    allan_factor=2.0                           # The Allan factor used in tidal model computation. Default is 2.0.
)
```
The tidal forcing is held by the `xarray.Dataset` object that is returned by the `.ds` property

```python
tidal_forcing.ds
```

```
<xarray.Dataset> Size: 7MB
Dimensions:  (ntides: 10, eta_rho: 102, xi_rho: 102, xi_u: 101, eta_v: 101)
Coordinates:
    lat_rho  (eta_rho, xi_rho) float64 83kB 47.84 47.91 47.97 ... 73.51 73.53
    lon_rho  (eta_rho, xi_rho) float64 83kB 354.0 354.3 354.5 ... 13.64 14.21
Dimensions without coordinates: ntides, eta_rho, xi_rho, xi_u, eta_v
Data variables:
    omega    (ntides) float64 80B 0.0001405 0.0001454 ... 2.639e-06 5.323e-06
    ssh_Re   (ntides, eta_rho, xi_rho) float64 832kB 1.329 1.394 ... -0.01962
    ssh_Im   (ntides, eta_rho, xi_rho) float64 832kB 0.7651 0.7742 ... -0.01428
    pot_Re   (ntides, eta_rho, xi_rho) float64 832kB 0.05112 ... -0.0005051
    pot_Im   (ntides, eta_rho, xi_rho) float64 832kB 0.1091 0.1075 ... 0.003186
    u_Re     (ntides, eta_rho, xi_u) float64 824kB 0.2155 0.1961 ... 0.0003719
    u_Im     (ntides, eta_rho, xi_u) float64 824kB 0.716 0.6522 ... 0.0001938
    v_Re     (ntides, eta_v, xi_rho) float64 824kB -0.5712 ... -0.0003684
    v_Im     (ntides, eta_v, xi_rho) float64 824kB 0.828 0.7922 ... 0.0001878
```

To visualize any of the tidal forcing components, use the `.plot` method:

```python
tidal_forcing.plot("ssh_Re", nc=0)
```
![Screenshot 2024-05-20 at 5 02 25 PM](https://github.com/NoraLoose/roms-tools/assets/23617395/f0e35759-c6a1-4c19-a683-cb7e272aa910)


### More steps:

Coming soon!


## Feedback and contributions

**If you find a bug, have a feature suggestion, or any other kind of feedback, please start a Discussion.**

We also accept contributions in the form of Pull Requests.


## See also

- [ROMS source code](https://github.com/CESR-lab/ucla-roms)
- [C-Star](https://github.com/CWorthy-ocean/C-Star)
