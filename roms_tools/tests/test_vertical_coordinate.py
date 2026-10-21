import numpy as np
import pytest
import xarray as xr

from roms_tools.constants import UPPER_BOUND_THETA_B, UPPER_BOUND_THETA_S
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

    with pytest.raises(
        ValueError, match=f"theta_s must be between 0 and {UPPER_BOUND_THETA_S}"
    ):
        compute_cs(sigma, UPPER_BOUND_THETA_S + 1, theta_b)

    with pytest.raises(
        ValueError, match=f"theta_b must be between 0 and {UPPER_BOUND_THETA_B}"
    ):
        compute_cs(sigma, theta_s, UPPER_BOUND_THETA_B + 1)

    with pytest.raises(ValueError, match="vstretching must be 1 to 5"):
        compute_cs(sigma, theta_s, theta_b, vstretching=6)


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


@pytest.mark.parametrize(
    "vstretching, theta_s, theta_b, expected",
    [
        # Cs_r for N = 5 from Rutgers ROMS set_scoord.F
        (1, 4.5, 0.95, [-0.966713, -0.835826, -0.480212, -0.129113, -0.015680]),
        (2, 5.0, 0.4, [-0.845328, -0.454989, -0.180037, -0.044655, -0.002771]),
        (3, 0.65, 0.58, [-0.880325, -0.656024, -0.480223, -0.302246, -0.089888]),
        (4, 5.0, 2.0, [-0.809019, -0.400746, -0.151298, -0.041949, -0.004025]),
        (5, 5.0, 2.0, [-0.578471, -0.114358, -0.017609, -0.001017, -0.000019]),
    ],
)
def test_sigma_stretch_matches_rutgers_roms(vstretching, theta_s, theta_b, expected):
    cs, _ = sigma_stretch(theta_s, theta_b, 5, "r", vstretching)
    np.testing.assert_allclose(cs, expected, atol=1e-6)


def test_compute_depth():
    zeta = xr.DataArray(0.5)
    h = xr.DataArray(10.0)
    hc = 5.0
    cs = xr.DataArray(np.linspace(-1, 0, 10), dims="s_rho")
    sigma = xr.DataArray(np.linspace(-1, 0, 10), dims="s_rho")

    depth = compute_depth(zeta, h, hc, cs, sigma)
    assert depth.shape == sigma.shape
    assert isinstance(depth, xr.DataArray)

    # Vtransform 1 reaches from the bottom to the surface, with z = hc * sigma if h = hc
    depth = compute_depth(zeta, h, hc, cs, sigma, vtransform=1)
    np.testing.assert_allclose(depth[[0, -1]], [10.0, -0.5])
    np.testing.assert_allclose(
        compute_depth(0, h, 10.0, cs, sigma, vtransform=1), -10.0 * sigma
    )


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
