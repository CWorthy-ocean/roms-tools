import pytest
from datetime import datetime
import numpy as np
import xarray as xr
from roms_tools import Grid
from roms_tools.setup.datasets import GLORYSDataset
from roms_tools.setup.download import download_test_data
from roms_tools.setup.utils import extrapolate_deepest_to_bottom, get_target_coords
from roms_tools.setup.fill import LateralFill
from roms_tools.setup.regrid import LateralRegrid, VerticalRegrid

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
