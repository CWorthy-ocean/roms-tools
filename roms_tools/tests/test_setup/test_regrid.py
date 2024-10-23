import pytest
import numpy as np
import xarray as xr
from roms_tools.setup.regrid import VerticalRegrid


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
    target_depth = xr.DataArray(data=layer_depth_rho_values, dims=["s_rho"])
    # Instantiate DataContainer and Grid objects with mock datasets
    mock_data = DataContainer(data_ds)

    return VerticalRegrid(mock_data, target_depth)


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
