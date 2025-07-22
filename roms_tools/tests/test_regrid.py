import numpy as np
import pytest
import xarray as xr

from roms_tools.regrid import VerticalRegridToROMS

try:
    import xesmf  # type: ignore
except ImportError:
    xesmf = None

from roms_tools.regrid import LateralRegridFromROMS


# Lateral regridding
@pytest.mark.skipif(xesmf is None, reason="xesmf required")
def test_lateral_regrid_with_curvilinear_grid():
    """Test that LateralRegridFromROMS regrids data correctly from a curvilinear ROMS
    grid.
    """
    # Define ROMS curvilinear grid dimensions
    eta_rho, xi_rho = 10, 20

    # Create a mock ROMS grid with curvilinear coordinates
    lat_rho = np.linspace(-10, 10, eta_rho).reshape(-1, 1) * np.ones((1, xi_rho))
    lon_rho = np.linspace(120, 140, xi_rho).reshape(1, -1) * np.ones((eta_rho, 1))

    ds_in = xr.Dataset(
        {
            "temp": (("eta_rho", "xi_rho"), np.random.rand(eta_rho, xi_rho)),
        },
        coords={
            "lat": (("eta_rho", "xi_rho"), lat_rho),
            "lon": (("eta_rho", "xi_rho"), lon_rho),
        },
    )

    # Define target latitude and longitude coordinates
    target_coords = {
        "lat": np.linspace(-5, 5, 5),
        "lon": np.linspace(125, 135, 10),
    }

    # Instantiate the regridder
    regridder = LateralRegridFromROMS(ds_in, target_coords, method="bilinear")

    # Apply the regridding to the input data
    regridded_da = regridder.apply(ds_in["temp"])

    # Assertions to verify that the output is as expected
    assert isinstance(regridded_da, xr.DataArray)
    assert regridded_da.shape == (5, 10)
    assert np.allclose(regridded_da.coords["lat"], target_coords["lat"])
    assert np.allclose(regridded_da.coords["lon"], target_coords["lon"])


@pytest.mark.skipif(xesmf is not None, reason="xesmf has to be missing")
def test_lateral_regrid_import_error():
    """Test that LateralRegridFromROMS raises ImportError when xesmf is missing."""
    # Define mock ROMS curvilinear grid dimensions
    eta_rho, xi_rho = 10, 20

    # Create a mock ROMS grid with curvilinear coordinates
    lat_rho = np.linspace(-10, 10, eta_rho).reshape(-1, 1) * np.ones((1, xi_rho))
    lon_rho = np.linspace(120, 140, xi_rho).reshape(1, -1) * np.ones((eta_rho, 1))

    ds_in = xr.Dataset(
        {
            "temp": (("eta_rho", "xi_rho"), np.random.rand(eta_rho, xi_rho)),
        },
        coords={
            "lat": (("eta_rho", "xi_rho"), lat_rho),
            "lon": (("eta_rho", "xi_rho"), lon_rho),
        },
    )

    # Define target latitude and longitude coordinates
    target_coords = {
        "lat": np.linspace(-5, 5, 5),
        "lon": np.linspace(125, 135, 10),
    }

    # Check that ImportError is raised when xesmf is missing
    with pytest.raises(
        ImportError, match="xesmf is required for this regridding task.*"
    ):
        LateralRegridFromROMS(ds_in, target_coords, method="bilinear")


# Vertical regridding
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

    target_depth = xr.DataArray(data=layer_depth_rho_values, dims=["s_rho"])
    source_depth = xr.DataArray(data=depth_values, dims=["depth"])

    return VerticalRegridToROMS(target_depth, source_depth)


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
