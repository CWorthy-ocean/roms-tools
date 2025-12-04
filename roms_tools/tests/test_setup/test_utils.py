import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from roms_tools import BoundaryForcing, Grid
from roms_tools.download import download_test_data
from roms_tools.setup.lat_lon_datasets import ERA5Correction
from roms_tools.setup.utils import (
    get_target_coords,
    interpolate_from_climatology,
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


def test_interpolate_from_climatology(use_dask):
    fname = download_test_data("ERA5_regional_test_data.nc")
    era5_times = xr.open_dataset(fname).time

    climatology = ERA5Correction(use_dask=use_dask)
    field = climatology.ds["ssr_corr"]
    field["time"] = field["time"].dt.days

    interpolated_field = interpolate_from_climatology(field, "time", era5_times)
    assert len(interpolated_field.time) == len(era5_times)


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
