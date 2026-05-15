import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from roms_tools import BoundaryForcing, Grid
from roms_tools.datasets.download import download_test_data
from roms_tools.setup.utils import (
    _compute_bgc_source_density,
    _infer_valid_boundaries_from_mask,
    check_and_set_boundaries,
    compute_potential_density,
    get_target_coords,
    validate_names,
)


class DummyGrid:
    """Lightweight grid wrapper mimicking the real Grid API for testing."""

    def __init__(self, ds: xr.Dataset, straddle: bool):
        """Initialize grid wrapper."""
        self.ds = ds
        self.straddle = straddle


class TestGetTargetCoords:
    def make_rho_grid(self, lons, lats, with_mask=False):
        """Helper to create a minimal rho grid dataset."""
        eta, xi = len(lats), len(lons)
        lon_rho, lat_rho = np.meshgrid(lons, lats)
        ds = xr.Dataset(
            {
                "lon_rho": (("eta_rho", "xi_rho"), lon_rho),
                "lat_rho": (("eta_rho", "xi_rho"), lat_rho),
                "angle": (("eta_rho", "xi_rho"), np.zeros_like(lon_rho)),
            },
            coords={"eta_rho": np.arange(eta), "xi_rho": np.arange(xi)},
        )
        if with_mask:
            ds["mask_rho"] = (("eta_rho", "xi_rho"), np.ones_like(lon_rho))
        return ds

    def test_basic_rho_grid(self):
        ds = self.make_rho_grid(lons=[-10, -5, 0, 5, 10], lats=[50, 55])
        grid = DummyGrid(ds, straddle=True)
        result = get_target_coords(grid)
        assert "lat" in result and "lon" in result
        assert np.allclose(result["lon"], ds.lon_rho)

    def test_wrap_longitudes_to_minus180_180(self):
        ds = self.make_rho_grid(lons=[190, 200], lats=[0, 1])
        grid = DummyGrid(ds, straddle=True)
        result = get_target_coords(grid)
        # longitudes >180 should wrap to -170, -160
        expected = np.array([[-170, -160], [-170, -160]])
        assert np.allclose(result["lon"].values, expected)

    def test_convert_to_0_360_if_far_from_greenwich(self):
        ds = self.make_rho_grid(lons=[-170, -160], lats=[0, 1])
        grid = DummyGrid(ds, straddle=False)
        result = get_target_coords(grid)
        # Should convert to 190, 200 since domain is far from Greenwich
        expected = np.array([[190, 200], [190, 200]])
        assert np.allclose(result["lon"].values, expected)
        assert result["straddle"] is False

    def test_close_to_greenwich_stays_minus180_180(self):
        ds = self.make_rho_grid(lons=[-2, -1], lats=[0, 1])
        grid = DummyGrid(ds, straddle=False)
        result = get_target_coords(grid)
        # Should remain unchanged (-2, -1), not converted to 358, 359
        expected = np.array([[-2, -1], [-2, -1]])
        assert np.allclose(result["lon"].values, expected)
        assert result["straddle"] is True

    def test_includes_optional_fields(self):
        ds = self.make_rho_grid(lons=[-10, -5], lats=[0, 1], with_mask=True)
        grid = DummyGrid(ds, straddle=True)
        result = get_target_coords(grid)
        assert result["mask"] is not None

    def test_coarse_grid_selection(self):
        lon = np.array([[190, 200]])
        lat = np.array([[10, 10]])
        ds = xr.Dataset(
            {
                "lon_coarse": (("eta_coarse", "xi_coarse"), lon),
                "lat_coarse": (("eta_coarse", "xi_coarse"), lat),
                "angle_coarse": (("eta_coarse", "xi_coarse"), np.zeros_like(lon)),
                "mask_coarse": (("eta_coarse", "xi_coarse"), np.ones_like(lon)),
            },
            coords={"eta_coarse": [0], "xi_coarse": [0, 1]},
        )
        grid = DummyGrid(ds, straddle=True)
        result = get_target_coords(grid, use_coarse_grid=True)
        # Should wrap longitudes to -170, -160
        expected = np.array([[-170, -160]])
        assert np.allclose(result["lon"].values, expected)
        assert "mask" in result


# Test yaml roundtrip with multiple source files
@pytest.fixture()
def boundary_forcing_from_multiple_source_files(request, use_dask):
    """Fixture for creating a BoundaryForcing object."""
    grid = Grid(
        nx=5,
        ny=5,
        size_x=100,
        size_y=100,
        center_lon=-8,
        center_lat=60,
        rot=10,
        N=3,  # number of vertical levels
    )

    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))

    return BoundaryForcing(
        grid=grid,
        start_time=datetime(2011, 1, 1),
        end_time=datetime(2013, 1, 1),
        source={"name": "GLORYS", "path": [fname1, fname2]},
        use_dask=use_dask,
    )


def test_roundtrip_yaml(
    boundary_forcing_from_multiple_source_files, request, tmp_path, use_dask
):
    """Test that creating a BoundaryForcing object, saving its parameters to yaml file,
    and re-opening yaml file creates the same object.
    """
    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str
        boundary_forcing_from_multiple_source_files.to_yaml(filepath)

        bdry_forcing_from_file = BoundaryForcing.from_yaml(filepath, use_dask=use_dask)

        assert boundary_forcing_from_multiple_source_files == bdry_forcing_from_file

        filepath = Path(filepath)
        filepath.unlink()


# test validate_names function

VALID_NAMES = ["a", "b", "c", "d"]
SENTINEL = "ALL"
MAX_TO_PLOT = 3


def test_valid_names_no_truncation():
    names = ["a", "b"]
    result = validate_names(names, VALID_NAMES, SENTINEL, MAX_TO_PLOT, label="test")
    assert result == names


def test_valid_names_with_truncation(caplog):
    names = ["a", "b", "c", "d"]
    with caplog.at_level(logging.WARNING):
        result = validate_names(
            names, VALID_NAMES, SENTINEL, max_to_plot=2, label="test"
        )
        assert result == ["a", "b"]
        assert "Only the first 2 tests will be plotted" in caplog.text


def test_include_all_sentinel():
    result = validate_names(SENTINEL, VALID_NAMES, SENTINEL, MAX_TO_PLOT, label="test")
    assert result == VALID_NAMES[:MAX_TO_PLOT]


def test_invalid_name_raises():
    with pytest.raises(ValueError, match="Invalid tests: z"):
        validate_names(["a", "z"], VALID_NAMES, SENTINEL, MAX_TO_PLOT, label="test")


def test_non_list_input_raises():
    with pytest.raises(ValueError, match="`test_names` should be a list of strings."):
        validate_names("a", VALID_NAMES, SENTINEL, MAX_TO_PLOT, label="test")


def test_non_string_elements_in_list_raises():
    with pytest.raises(
        ValueError, match="All elements in `test_names` must be strings."
    ):
        validate_names(["a", 2], VALID_NAMES, SENTINEL, MAX_TO_PLOT, label="test")


def test_custom_label_in_errors():
    with pytest.raises(ValueError, match="Invalid foozs: z"):
        validate_names(["z"], VALID_NAMES, SENTINEL, MAX_TO_PLOT, label="fooz")


# test _infer_valid_boundaries_from_mask
@pytest.fixture
def simple_mask():
    data = np.array(
        [
            [0, 0, 0, 0],  # south
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],  # north
        ]
    )
    return xr.DataArray(data, dims=("eta_rho", "xi_rho"))


def test_infer_valid_boundaries_partial(simple_mask):
    out = _infer_valid_boundaries_from_mask(simple_mask)

    assert out == {
        "south": False,
        "north": True,
        "west": True,
        "east": True,
    }


def test_infer_valid_boundaries_all_ocean():
    mask = xr.DataArray(np.ones((3, 3)), dims=("eta_rho", "xi_rho"))

    out = _infer_valid_boundaries_from_mask(mask)
    assert all(out.values())


# test check_and_set_boundaries


def test_check_and_set_default_boundaries(simple_mask, monkeypatch):
    monkeypatch.setattr(
        "roms_tools.setup.utils._infer_valid_boundaries_from_mask",
        lambda mask: {
            "south": True,
            "north": False,
            "west": True,
            "east": False,
        },
    )

    result = check_and_set_boundaries(None, simple_mask)

    assert result == {
        "south": True,
        "north": False,
        "west": True,
        "east": False,
    }


def test_check_and_set_partial_boundaries(simple_mask, monkeypatch):
    monkeypatch.setattr(
        "roms_tools.setup.utils._infer_valid_boundaries_from_mask",
        lambda mask: {
            "south": True,
            "north": False,
            "west": True,
            "east": False,
        },
    )

    user = {"south": False}  # user overrides south → False

    result = check_and_set_boundaries(user, simple_mask)

    assert result == {
        "south": False,  # user-preserved
        "north": False,  # inferred
        "west": True,  # inferred
        "east": False,  # inferred
    }


def test_check_and_set_type_error(simple_mask):
    with pytest.raises(TypeError):
        check_and_set_boundaries({"south": "yes"}, simple_mask)


def test_check_and_set_invalid_key(simple_mask):
    with pytest.raises(ValueError):
        check_and_set_boundaries({"northeast": True}, simple_mask)


def test_check_and_set_full_user_boundaries(simple_mask):
    boundaries = {"south": False, "north": True, "east": False, "west": True}

    result = check_and_set_boundaries(boundaries, simple_mask)

    assert result == boundaries  # unchanged


# test compute_potential_density


def test_compute_potential_density_known_value():
    """Verify sigma-0 against a known seawater value (T=20°C, S=35 PSU → ~24.64 kg/m³)."""
    temp = xr.DataArray([[20.0]])
    salt = xr.DataArray([[35.0]])
    result = compute_potential_density(temp, salt)
    assert float(result.values.flat[0]) == pytest.approx(24.64, abs=0.1)


def test_compute_potential_density_dask():
    """Verify compute_potential_density returns a lazy dask-backed array."""
    import dask.array as da

    temp = xr.DataArray(da.from_array([[20.0]], chunks=(1, 1)))
    salt = xr.DataArray(da.from_array([[35.0]], chunks=(1, 1)))
    result = compute_potential_density(temp, salt)
    assert result.chunks is not None


def test_compute_bgc_source_density_constant_TS():
    """For constant T/S, source density at each BGC depth equals sigma0(S,T) + monotonicity perturbation."""
    import gsw

    n_phys = 4
    n_bgc = 3
    T_const, S_const = 10.0, 35.0
    phys_temp = xr.DataArray(
        np.full((1, 1, n_phys), T_const), dims=["xi", "eta", "depth"]
    )
    phys_salt = xr.DataArray(
        np.full((1, 1, n_phys), S_const), dims=["xi", "eta", "depth"]
    )
    phys_depth = xr.DataArray(
        np.array([10.0, 100.0, 500.0, 2000.0]),
        dims=["depth"],
        coords={"depth": [10.0, 100.0, 500.0, 2000.0]},
    )
    bgc_depth = xr.DataArray(
        np.array([20.0, 300.0, 1500.0]),
        dims=["depth"],
        coords={"depth": [20.0, 300.0, 1500.0]},
    )

    result = _compute_bgc_source_density(
        phys_temp,
        phys_salt,
        "depth",
        phys_depth,
        bgc_depth,
        "depth",
    )

    # gsw.sigma0 is invariant in space for constant T/S; only the monotonicity
    # perturbation along the BGC depth axis (added in _compute_bgc_source_density)
    # should change the value across depth.
    expected_base = gsw.sigma0(S_const, T_const)
    expected = expected_base + np.arange(n_bgc) * 1e-7
    np.testing.assert_allclose(
        result.values[0, 0],
        expected,
        rtol=0.0,
        atol=1e-6,
    )


def test_compute_bgc_source_density_linear_TS_profile():
    """Density at BGC depths is the interpolation of physics densities to BGC depths."""
    import gsw

    # Linear T/S profiles. Physics depths bracket the BGC depths so result is
    # purely the linear interpolation in depth space (no extrapolation).
    phys_depths = np.array([0.0, 100.0, 500.0, 2000.0])
    phys_T = np.array([20.0, 15.0, 8.0, 2.0])
    phys_S = np.array([34.5, 34.8, 34.9, 35.0])
    bgc_depths = np.array([50.0, 250.0, 1200.0])

    phys_temp = xr.DataArray(phys_T.reshape(1, 1, -1), dims=["xi", "eta", "depth"])
    phys_salt = xr.DataArray(phys_S.reshape(1, 1, -1), dims=["xi", "eta", "depth"])
    phys_depth = xr.DataArray(
        phys_depths, dims=["depth"], coords={"depth": phys_depths}
    )
    bgc_depth = xr.DataArray(bgc_depths, dims=["depth"], coords={"depth": bgc_depths})

    result = _compute_bgc_source_density(
        phys_temp,
        phys_salt,
        "depth",
        phys_depth,
        bgc_depth,
        "depth",
    )

    # Manual reference: compute density at phys depths with the same perturbation
    # the function adds, then linearly interpolate to BGC depths, then add the
    # BGC-depth perturbation.
    phys_density_perturbed = (
        gsw.sigma0(phys_S, phys_T) + np.arange(len(phys_depths)) * 1e-7
    )
    expected = np.interp(bgc_depths, phys_depths, phys_density_perturbed)
    expected = expected + np.arange(len(bgc_depths)) * 1e-7

    np.testing.assert_allclose(
        result.values[0, 0],
        expected,
        rtol=1e-10,
        atol=1e-10,
    )
