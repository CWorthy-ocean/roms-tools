import numpy as np
import pytest
import xarray as xr

from roms_tools.vertical_coordinate import (
    compute_cs,
    compute_depth,
    compute_depth_coordinates,
    sigma_stretch,
)


def test_compute_cs():
    sigma = np.linspace(-1, 0, 10)
    theta_s, theta_b = 5, 2
    cs = compute_cs(sigma, theta_s, theta_b)
    assert cs.shape == sigma.shape
    assert np.all(cs <= 0) and np.all(cs >= -1)

    with pytest.raises(ValueError, match="theta_s must be between 0 and 10"):
        compute_cs(sigma, 15, 2)

    with pytest.raises(ValueError, match="theta_b must be between 0 and 4"):
        compute_cs(sigma, 5, 5)


def test_sigma_stretch():
    theta_s, theta_b, N = 5, 2, 10
    cs, sigma = sigma_stretch(theta_s, theta_b, N, "r")
    assert cs.shape == sigma.shape
    assert isinstance(cs, xr.DataArray)
    assert isinstance(sigma, xr.DataArray)

    with pytest.raises(
        ValueError,
        match="Type must be either 'w' for vertical velocity points or 'r' for rho-points.",
    ):
        sigma_stretch(theta_s, theta_b, N, "invalid")


def test_compute_depth():
    zeta = xr.DataArray(0.5)
    h = xr.DataArray(10.0)
    hc = 5.0
    cs = xr.DataArray(np.linspace(-1, 0, 10), dims="s_rho")
    sigma = xr.DataArray(np.linspace(-1, 0, 10), dims="s_rho")

    depth = compute_depth(zeta, h, hc, cs, sigma)
    assert depth.shape == sigma.shape
    assert isinstance(depth, xr.DataArray)


def test_compute_depth_coordinates():
    grid_ds = xr.Dataset(
        {
            "h": xr.DataArray([[10, 20], [30, 40]], dims=("eta_rho", "xi_rho")),
            "Cs_r": xr.DataArray(np.linspace(-1, 0, 10), dims="s_rho"),
            "sigma_r": xr.DataArray(np.linspace(-1, 0, 10), dims="s_rho"),
            "Cs_w": xr.DataArray(np.linspace(-1, 0, 11), dims="s_w"),
            "sigma_w": xr.DataArray(np.linspace(-1, 0, 11), dims="s_w"),
        },
        attrs={"hc": 5.0},
    )

    depth = compute_depth_coordinates(grid_ds, depth_type="layer", location="rho")
    assert isinstance(depth, xr.DataArray)
    assert "eta_rho" in depth.dims and "xi_rho" in depth.dims

    with pytest.raises(ValueError, match="Invalid depth_type"):
        compute_depth_coordinates(grid_ds, depth_type="invalid")

    with pytest.raises(ValueError, match="Invalid location"):
        compute_depth_coordinates(grid_ds, location="invalid")
