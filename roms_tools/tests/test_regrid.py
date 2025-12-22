import numpy as np
import pytest
import xarray as xr

from roms_tools.regrid import VerticalRegrid, VerticalRegridToROMS

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


# Test VerticalRegrid
def test_vertical_regrid_2d_depths_different_vertical_levels():
    """
    Vertical regridding with 2D (eta, xi) depth coordinates where
    source and target have different numbers of vertical levels.
    """
    # --- dimensions ---
    time = xr.DataArray(
        np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]"),
        dims="time",
    )

    s_rho_src = xr.DataArray(np.arange(6), dims="s_rho")  # source: 6 levels
    s_rho_tgt = xr.DataArray(np.arange(10), dims="s_rho")  # target: 10 levels

    eta = xr.DataArray(np.arange(3), dims="eta")
    xi = xr.DataArray(np.arange(4), dims="xi")

    # --- source depth coords: 2D + s_rho ---
    base_depth = -(100 + 10 * eta.values[:, None] + 5 * xi.values[None, :])

    source_depth = xr.DataArray(
        base_depth,
        dims=("eta", "xi"),
        coords={"eta": eta, "xi": xi},
    ).expand_dims(time=time, s_rho=s_rho_src)

    source_depth = source_depth + 20 * s_rho_src

    # --- target depth coords: 2D + different s_rho ---
    target_depth = xr.DataArray(
        base_depth,
        dims=("eta", "xi"),
        coords={"eta": eta, "xi": xi},
    ).expand_dims(s_rho=s_rho_tgt)

    target_depth = target_depth + 20 * s_rho_tgt

    # --- synthetic temperature field: varies linearly with depth ---
    temp_data = xr.DataArray(
        np.broadcast_to(
            s_rho_src.values[None, :, None, None] * 2.0,
            (len(time), len(s_rho_src), len(eta), len(xi)),
        ),
        dims=("time", "s_rho", "eta", "xi"),
        coords={
            "time": time,
            "s_rho": s_rho_src,
            "eta": eta,
            "xi": xi,
        },
        name="temp",
    )

    ds = xr.Dataset(
        {
            "temp": temp_data,
        },
        coords={
            "time": time,
            "s_rho": s_rho_src,
            "eta": eta,
            "xi": xi,
        },
    )

    # --- regrid ---
    regridder = VerticalRegrid(ds)
    out = regridder.apply(
        temp_data,
        source_depth_coords=source_depth,
        target_depth_coords=target_depth,
    )

    # --- assertions ---
    assert isinstance(out, xr.DataArray)

    # vertical dimension changed to target resolution
    assert out.sizes["s_rho"] == s_rho_tgt.size

    # horizontal + time preserved
    assert out.sizes["time"] == time.size
    assert out.sizes["eta"] == eta.size
    assert out.sizes["xi"] == xi.size

    # output contains finite values
    assert np.isfinite(out).any()

    # interpolation stays within source bounds
    assert out.min() >= temp_data.min()
    assert out.max() <= temp_data.max()


def test_vertical_regrid_mask_edges():
    """Values outside source depth range should be masked when mask_edges=True."""
    s_rho = xr.DataArray(np.linspace(-1, 0, 5), dims="s_rho")
    source_depth = xr.DataArray([-100, -75, -50, -25, 0], dims="s_rho")

    target_depth = xr.DataArray([-200, -50, 10], dims="s_rho")

    data = xr.DataArray(
        source_depth.values,
        dims="s_rho",
        coords={"s_rho": s_rho},
    )

    ds = xr.Dataset(
        {"data": data, "depth": source_depth},
        coords={"s_rho": s_rho},
    )

    regridder = VerticalRegrid(ds)
    out = regridder.apply(data, source_depth, target_depth, mask_edges=True)

    assert np.isnan(out[0])
    assert np.isnan(out[-1])
    assert not np.isnan(out[1])
