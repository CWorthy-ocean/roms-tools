import pytest
from datetime import datetime
import numpy as np
import xarray as xr
from roms_tools import Grid
from roms_tools.setup.datasets import GLORYSDataset
from roms_tools.setup.download import download_test_data
from roms_tools.setup.utils import extrapolate_deepest_to_bottom, get_target_coords
from roms_tools.setup.regrid import LateralRegrid, VerticalRegrid


@pytest.fixture()
def midocean_glorys_data(request, use_dask):
    fname = download_test_data("GLORYS_NA_2012.nc")

    data = GLORYSDataset(
        filename=fname,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2013, 1, 1),
        use_dask=use_dask,
    )
    data.post_process()

    # transform longitude to [-180, 180 range]; this is usually handled by data.choose_subdomain()
    ds = data.ds
    lon = ds["longitude"]
    ds[data.dim_names["longitude"]] = xr.where(lon > 180, lon - 360, lon)
    object.__setattr__(data, "ds", ds)

    # extrapolate deepest value to bottom so all levels can use the same surface mask
    for var in data.var_names:
        if var != "zeta":
            data.ds[data.var_names[var]] = extrapolate_deepest_to_bottom(
                data.ds[data.var_names[var]], data.dim_names["depth"]
            )

    # remove upper and deep layers to be able to check vertical extrapolation
    ds = data.ds.isel(depth=slice(1, 3))
    object.__setattr__(data, "ds", ds)

    return data


def test_vertical_regrid_no_nans(midocean_glorys_data):

    grid = Grid(
        nx=4,
        ny=4,
        size_x=300,
        size_y=300,
        center_lon=-8,
        center_lat=60,
        rot=0,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    target_coords = get_target_coords(grid)

    lateral_regrid = LateralRegrid(
        midocean_glorys_data, target_coords["lon"], target_coords["lat"]
    )
    vertical_regrid = VerticalRegrid(midocean_glorys_data, grid)

    varnames = ["temp", "salt", "u", "v"]
    for var in varnames:
        regridded = lateral_regrid.apply(
            midocean_glorys_data.ds[midocean_glorys_data.var_names[var]]
        )
        regridded = vertical_regrid.apply(regridded, fill_nans=True)

        assert not regridded.isnull().any()


def vertical_regridder(depth_values, layer_depth_rho_values):
    class DataContainer:
        """Mock class for holding data and dimension names."""

        def __init__(self, ds):
            self.ds = ds
            self.dim_names = {"depth": "depth"}

    class Grid:
        """Mock class representing the grid object with layer depth information."""

        def __init__(self, ds):
            self.ds = ds

    # Creating minimal mock data for testing
    # Depth levels in meters

    # Create mock datasets for DataContainer and Grid
    data_ds = xr.Dataset({"depth": (["depth"], depth_values)})
    grid_ds = xr.Dataset({"layer_depth_rho": (["s_rho"], layer_depth_rho_values)})

    # Instantiate DataContainer and Grid objects with mock datasets
    mock_data = DataContainer(data_ds)
    mock_grid = Grid(grid_ds)

    return VerticalRegrid(mock_data, mock_grid)


@pytest.mark.parametrize(
    "depth_values, layer_depth_rho_values, temp_data",
    [
        ([5, 50, 100, 150], [130, 100, 70, 30, 10], [30, 25, 10, 2]),
        ([5, 50, 100, 150], [130, 100, 70, 30, 2], [30, 25, 10, 2]),
        ([5, 50, 100, 150], [200, 100, 70, 30, 10], [30, 25, 10, 2]),
        ([5, 50, 100, 150], [200, 100, 70, 30, 1], [30, 25, 10, 2]),
    ],
)
def test_vertical_regrid(request, depth_values, layer_depth_rho_values, temp_data):

    vertical_regrid = vertical_regridder(
        depth_values=depth_values, layer_depth_rho_values=layer_depth_rho_values
    )
    data = xr.Dataset({"temp_data": (["depth"], temp_data)})

    # without filling in NaNs
    regridded = vertical_regrid.apply(data.temp_data, fill_nans=False)
    expected = np.interp(
        layer_depth_rho_values, depth_values, temp_data, left=np.nan, right=np.nan
    )
    assert np.allclose(expected, regridded.data, equal_nan=True)

    # with filling in NaNs
    regridded = vertical_regrid.apply(data.temp_data, fill_nans=True)
    expected = np.interp(layer_depth_rho_values, depth_values, temp_data)
    assert np.allclose(expected, regridded.data, equal_nan=True)
