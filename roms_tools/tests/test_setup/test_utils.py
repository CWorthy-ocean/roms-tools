import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from roms_tools import BoundaryForcing, Grid
from roms_tools.datasets.download import download_test_data
from roms_tools.setup.utils import (
    _compute_density_coord,
    _infer_valid_boundaries_from_mask,
    calendar_midmonth_dates,
    check_and_set_boundaries,
    compute_potential_density,
    expand_monthly_climatology_time_axis,
    get_target_coords,
    interpolate_dynamic_bgc_by_calendar_year,
    month_to_time_index,
    tile_monthly_climatology_on_calendar,
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


def test_compute_density_coord_constant_TS():
    """For constant T/S, the density coordinate equals sigma0(S,T) plus the
    monotonicity perturbation along the depth dimension.
    """
    import gsw

    n_depth = 4
    T_const, S_const = 10.0, 35.0
    temp = xr.DataArray(np.full((n_depth, 1, 1), T_const), dims=["depth", "eta", "xi"])
    salt = xr.DataArray(np.full((n_depth, 1, 1), S_const), dims=["depth", "eta", "xi"])

    result = _compute_density_coord(temp, salt, "depth")

    # gsw.sigma0 is invariant for constant T/S; only the per-index perturbation
    # added along the depth axis changes the value across depth.
    expected = gsw.sigma0(S_const, T_const) + np.arange(n_depth) * 1e-7
    np.testing.assert_allclose(result.values[:, 0, 0], expected, rtol=0.0, atol=1e-6)


def test_compute_density_coord_monotonic_and_chunked():
    """The density coordinate is strictly increasing along the depth dim and is
    single-chunked there (required by xgcm.transform).
    """
    # A stably stratified column: density should already increase with depth, and
    # the perturbation guarantees strict monotonicity even for ties.
    temp = xr.DataArray(
        np.array([20.0, 20.0, 12.0, 4.0]).reshape(-1, 1, 1),
        dims=["s_rho", "eta", "xi"],
    ).chunk({"s_rho": 2})
    salt = xr.DataArray(
        np.array([34.0, 34.0, 34.8, 35.0]).reshape(-1, 1, 1),
        dims=["s_rho", "eta", "xi"],
    ).chunk({"s_rho": 2})

    result = _compute_density_coord(temp, salt, "s_rho")

    profile = result.values[:, 0, 0]
    assert np.all(np.diff(profile) > 0), "density coordinate must be strictly monotonic"
    # single chunk along the transformed dim
    assert result.chunks is not None
    s_axis = result.dims.index("s_rho")
    assert len(result.chunks[s_axis]) == 1


def test_density_space_interpolation_returns_correct_values():
    """End-to-end correctness of density-space interpolation on a synthetic column.

    A tracer placed on a source density coordinate (built from source T/S) and
    interpolated onto a target density coordinate (built from target T/S) must equal
    a direct 1-D linear interpolation of the tracer in density space. This pins the
    full density-interpolation composition (``_compute_density_coord`` feeding
    ``VerticalRegrid.apply``) used by InitialConditions and BoundaryForcing.
    """
    from roms_tools.regrid import VerticalRegrid

    # Source column: stably stratified (colder/denser with depth) so the density
    # coordinate is monotonic; a tracer that increases with depth.
    src_temp = xr.DataArray(
        np.array([25.0, 20.0, 15.0, 10.0]).reshape(-1, 1, 1),
        dims=["depth", "eta", "xi"],
    )
    src_salt = xr.DataArray(np.full((4, 1, 1), 35.0), dims=["depth", "eta", "xi"])
    tracer = xr.DataArray(
        np.array([2000.0, 2100.0, 2200.0, 2300.0]).reshape(-1, 1, 1),
        dims=["depth", "eta", "xi"],
        name="ALK",
    )

    # Target ROMS sigma levels, with T/S chosen so the target densities fall strictly
    # inside the source range (genuine interpolation, not edge clamping).
    tgt_temp = xr.DataArray(
        np.array([22.0, 17.0, 12.0]).reshape(-1, 1, 1),
        dims=["s_rho", "eta", "xi"],
    )
    tgt_salt = xr.DataArray(np.full((3, 1, 1), 35.0), dims=["s_rho", "eta", "xi"])

    source_density = _compute_density_coord(src_temp, src_salt, "depth")
    target_density = _compute_density_coord(tgt_temp, tgt_salt, "s_rho")

    src_rho = source_density.values[:, 0, 0]
    tgt_rho = target_density.values[:, 0, 0]
    # sanity: target densities are interior to the source range
    assert tgt_rho.min() > src_rho.min()
    assert tgt_rho.max() < src_rho.max()

    ds = xr.Dataset({"ALK": tracer})
    vertical_regrid = VerticalRegrid(ds, source_dim="depth")
    regridded = vertical_regrid.apply(
        ds["ALK"],
        source_depth_coords=source_density,
        target_depth_coords=target_density,
    ).transpose("s_rho", "eta", "xi")

    expected = np.interp(tgt_rho, src_rho, tracer.values[:, 0, 0])
    np.testing.assert_allclose(regridded.values[:, 0, 0], expected, rtol=0.0, atol=1e-6)


class TestMonthlyClimatologyExpansion:
    @staticmethod
    def _monthly_climatology_ds() -> xr.Dataset:
        months = np.arange(1, 13)
        return xr.Dataset(
            {
                "river_volume": (
                    ("river_time", "nriver"),
                    np.arange(12 * 2, dtype=float).reshape(12, 2),
                )
            },
            coords={
                "river_time": months,
                "month": ("river_time", months),
                "nriver": [0, 1],
            },
            attrs={"climatology": "True"},
        )

    def test_month_to_time_index(self):
        assert month_to_time_index(np.array([1, 2, 3])) == {1: 0, 2: 1, 3: 2}

    def test_tile_monthly_climatology_on_calendar(self):
        ds = self._monthly_climatology_ds()
        dates = [
            datetime(2020, 1, 15),
            datetime(2020, 2, 15),
            datetime(2021, 1, 15),
        ]
        tiled = tile_monthly_climatology_on_calendar(ds, dates)
        assert tiled.sizes["river_time"] == 3
        np.testing.assert_allclose(
            tiled["river_volume"].isel(river_time=0).values,
            ds["river_volume"].isel(river_time=0).values,
        )
        np.testing.assert_allclose(
            tiled["river_volume"].isel(river_time=2).values,
            ds["river_volume"].isel(river_time=0).values,
        )
        assert "month" not in tiled

    def test_expand_monthly_climatology_time_axis(self):
        ds = self._monthly_climatology_ds()
        expanded = expand_monthly_climatology_time_axis(
            ds,
            datetime(2020, 1, 1),
            datetime(2020, 3, 15),
            datetime(2000, 1, 1),
            discharge_climatology_attr="discharge_climatology",
        )
        assert expanded.sizes["river_time"] == 3
        assert expanded.attrs["discharge_climatology"] == "True"
        assert "climatology" not in expanded.attrs
        assert "abs_time" in expanded
        assert calendar_midmonth_dates(datetime(2020, 1, 1), datetime(2020, 3, 15)) == [
            datetime(2020, m, 15) for m in (1, 2, 3)
        ]

    def test_calendar_midmonth_dates_short_window_without_15th(self):
        # A sub-month window that skips the 15th must still yield one in-window
        # date tagged to the correct month, not raise.
        dates = calendar_midmonth_dates(datetime(2020, 1, 20), datetime(2020, 1, 28))
        assert dates == [datetime(2020, 1, 20)]
        assert dates[0].month == 1


class TestInterpolateDynamicBGCByCalendarYear:
    @staticmethod
    def _monthly_abs_time(years: tuple[int, ...]) -> xr.DataArray:
        dates = [datetime(year, month, 15) for year in years for month in range(1, 13)]
        return xr.DataArray(
            np.array(dates, dtype="datetime64[ns]"),
            dims=["river_time"],
        )

    def test_interior_gap_year_is_linearly_interpolated(self):
        abs_time = self._monthly_abs_time((2000, 2001, 2002))
        years = abs_time.dt.year.values
        values = np.empty((len(years), 1), dtype=np.float32)
        for i, year in enumerate(years):
            if year == 2001:
                values[i, 0] = np.nan
            elif year == 2000:
                values[i, 0] = 100.0
            else:
                values[i, 0] = 200.0
        dic = xr.DataArray(
            values,
            dims=["river_time", "nriver"],
            coords={"river_time": abs_time, "nriver": [0]},
        )
        result = interpolate_dynamic_bgc_by_calendar_year({"DIC": dic}, abs_time)
        gap = result["DIC"].sel(river_time=abs_time.dt.year == 2001)
        np.testing.assert_allclose(gap.values, 150.0, rtol=1e-6)

    def test_leading_and_trailing_nan_years_remain_nan(self):
        abs_time = self._monthly_abs_time((2000, 2001, 2002))
        years = abs_time.dt.year.values
        values = np.where(years == 2001, 100.0, np.nan).astype(np.float32)[
            :, np.newaxis
        ]
        dic = xr.DataArray(
            values,
            dims=["river_time", "nriver"],
            coords={"river_time": abs_time, "nriver": [0]},
        )
        result = interpolate_dynamic_bgc_by_calendar_year({"DIC": dic}, abs_time)
        assert np.all(np.isnan(result["DIC"].sel(river_time=abs_time.dt.year == 2000)))
        assert np.all(np.isnan(result["DIC"].sel(river_time=abs_time.dt.year == 2002)))

    def test_zero_export_year_is_not_interpolated(self):
        abs_time = self._monthly_abs_time((2000, 2001, 2002))
        years = abs_time.dt.year.values
        values = np.empty((len(years), 1), dtype=np.float32)
        for i, year in enumerate(years):
            if year == 2001:
                values[i, 0] = 0.0
            elif year == 2000:
                values[i, 0] = 100.0
            else:
                values[i, 0] = 200.0
        dic = xr.DataArray(
            values,
            dims=["river_time", "nriver"],
            coords={"river_time": abs_time, "nriver": [0]},
        )
        result = interpolate_dynamic_bgc_by_calendar_year({"DIC": dic}, abs_time)
        gap = result["DIC"].sel(river_time=abs_time.dt.year == 2001)
        np.testing.assert_allclose(gap.values, 0.0)
