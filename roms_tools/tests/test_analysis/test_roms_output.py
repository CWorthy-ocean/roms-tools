import pytest
from pathlib import Path
import xarray as xr
from roms_tools import Grid, ROMSOutput
from roms_tools.download import download_test_data
import os

def test_successful_initialization_with_restart_files(use_dask):

    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)
    
    # single file
    fname_restart1 = Path(download_test_data("eastpac25km_rst.19980106000000.nc"))

    single_restart = ROMSOutput(
        grid=grid, path=fname_restart1, type="restart", use_dask=use_dask
    )
    assert isinstance(single_restart.ds, xr.Dataset)

    # list of files
    fname_restart2 = Path(download_test_data("eastpac25km_rst.19980126000000.nc"))
    two_restarts = ROMSOutput(
        grid=grid, path=[fname_restart1, fname_restart2], type="restart", use_dask=use_dask
    )
    assert isinstance(two_restarts.ds, xr.Dataset)

    # directory
    directory = os.path.dirname(fname_restart1)
    restarts = ROMSOutput(
        grid=grid, path=directory, type="restart", use_dask=use_dask
    )
    assert isinstance(restarts.ds, xr.Dataset)



