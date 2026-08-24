"""Tests for the curvilinear source datasets."""

from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from roms_tools.datasets.curvilinear_datasets import (
    CONUS404_RADIATION_NOISE_FLOOR_W_M2,
    CONUS404Dataset,
)
from roms_tools.datasets.utils import specific_humidity_from_dewpoint
from roms_tools.tests.conus404_test_utils import (
    GRID_SPACING_M,
    synthetic_conus404,
    write_conus404_store,
)

# Differencing two float32 accumulations can lose up to one ULP from each
# endpoint, so allow twice the single-step noise floor.
RADIATION_TOL = 2 * CONUS404_RADIATION_NOISE_FLOOR_W_M2

START = datetime(2020, 6, 15, 3)
END = datetime(2020, 6, 15, 12)


@pytest.fixture(scope="module")
def synthetic_store(tmp_path_factory):
    """A synthetic CONUS404 zarr store plus its exact expected outputs."""
    ds, expected = synthetic_conus404()
    path = write_conus404_store(
        tmp_path_factory.mktemp("conus404") / "synthetic", ds, fmt="zarr"
    )
    return path, expected


@pytest.fixture(scope="module")
def dataset(synthetic_store):
    path, expected = synthetic_store
    data = CONUS404Dataset(
        filename=str(path), start_time=START, end_time=END, use_dask=True
    )
    return data, expected


def _aligned(data, expected, key):
    """Return (actual, expected) for a variable, aligned on the selected times."""
    actual = data.ds[data.var_names[key]].compute()
    # `expected` covers records 1: of the full synthetic series; select the ones
    # the dataset actually kept.
    idx = np.searchsorted(expected["times"][1:], actual.time.values)
    return actual.values, expected[key][idx]


# --- radiation -------------------------------------------------------------


def test_radiation_recovers_flux_within_noise_floor(dataset):
    """Differenced accumulations recover the true flux to the quantization floor."""
    data, expected = dataset
    for key in ("swrad", "lwrad"):
        actual, exp = _aligned(data, expected, key)
        err = np.abs(actual - exp)
        assert err.max() <= RADIATION_TOL, (
            f"{key} max error {err.max():.3f} W/m2 exceeds {RADIATION_TOL:.3f}"
        )


def test_radiation_is_quantized_not_exact(dataset):
    """The float32 accumulation really does cost precision.

    Guards the fixture itself: if this ever passes exactly, the synthetic data has
    stopped reproducing the store's quantization and the tolerance above would be
    testing nothing.
    """
    data, expected = dataset
    actual, exp = _aligned(data, expected, "swrad")
    daytime = exp > 1.0
    assert not np.allclose(actual[daytime], exp[daytime], atol=1e-6)


def test_night_radiation_is_exactly_zero(dataset):
    """Night-time shortwave comes back as exactly 0.0, as on the real store."""
    data, expected = dataset
    actual, exp = _aligned(data, expected, "swrad")
    night = exp == 0.0
    assert night.any(), "fixture produced no night-time records"
    assert (actual[night] == 0.0).all()


def test_radiation_is_never_negative(dataset):
    """The ULP artifacts that make raw differences slightly negative are clipped."""
    data, _ = dataset
    for key in ("swrad", "lwrad"):
        assert (data.ds[data.var_names[key]].compute().values >= 0.0).all()


def test_swrad_is_net_not_downward(dataset):
    """swrad must be net shortwave (down minus up), matching ERA5's `ssr`."""
    data, expected = dataset
    from roms_tools.tests.conus404_test_utils import ALBEDO

    actual, exp_net = _aligned(data, expected, "swrad")
    exp_down = exp_net / (1.0 - ALBEDO)
    daytime = exp_net > 50.0
    assert np.abs(actual - exp_net)[daytime].max() <= RADIATION_TOL
    # And clearly distinguishable from the downward-only alternative.
    assert np.abs(actual - exp_down)[daytime].max() > 10 * RADIATION_TOL


# --- winds -----------------------------------------------------------------


def test_wind_rotated_to_earth_relative(dataset):
    """U10/V10 are rotated from the model grid to east/north."""
    data, expected = dataset
    for key in ("uwnd", "vwnd"):
        actual, exp = _aligned(data, expected, key)
        np.testing.assert_allclose(actual, exp, rtol=1e-6, atol=1e-5)


def test_wind_rotation_actually_changes_the_values(dataset):
    """Guard against a no-op rotation silently passing the test above."""
    data, expected = dataset
    u_actual, _ = _aligned(data, expected, "uwnd")
    # The synthetic grid-relative wind is a constant (5, 2) m/s.
    assert np.abs(u_actual - 5.0).min() > 0.1


def test_wind_rotation_preserves_speed(dataset):
    """A rotation cannot change wind magnitude."""
    data, expected = dataset
    u, _ = _aligned(data, expected, "uwnd")
    v, _ = _aligned(data, expected, "vwnd")
    np.testing.assert_allclose(
        np.hypot(u, v), np.hypot(5.0, 2.0), rtol=1e-6, atol=1e-5
    )


# --- unit conversions ------------------------------------------------------


def test_temperature_in_celsius(dataset):
    data, expected = dataset
    actual, exp = _aligned(data, expected, "Tair")
    np.testing.assert_allclose(actual, exp, rtol=1e-6, atol=1e-4)


def test_rain_in_cm_per_day(dataset):
    data, expected = dataset
    actual, exp = _aligned(data, expected, "rain")
    np.testing.assert_allclose(actual, exp, rtol=1e-6, atol=1e-6)


def test_units_attrs_match_era5_conventions(dataset):
    data, _ = dataset
    vn = data.var_names
    assert data.ds[vn["swrad"]].attrs["units"] == "W/m^2"
    assert data.ds[vn["lwrad"]].attrs["units"] == "W/m^2"
    assert data.ds[vn["Tair"]].attrs["units"] == "degrees C"
    assert data.ds[vn["rain"]].attrs["units"] == "cm/day"
    assert data.ds["qair"].attrs["units"] == "kg/kg"


# --- humidity --------------------------------------------------------------


def test_qair_uses_psfc_by_default(dataset):
    data, _ = dataset
    tair = data.ds[data.var_names["Tair"]].compute()
    # The synthetic PSFC is a constant 101300 Pa.
    exp = specific_humidity_from_dewpoint(tair, tair - 5.0, patm=1013.0)
    np.testing.assert_allclose(
        data.ds["qair"].compute().values, exp.values, rtol=1e-6
    )


def test_qair_era5_magnus_matches_era5_bit_for_bit(synthetic_store):
    """`qair_method="era5_magnus"` reproduces ERA5's arithmetic exactly.

    This is the regression test protecting a blend seam: if someone changes one
    source's humidity formula without the other, this fails.
    """
    path, _ = synthetic_store
    data = CONUS404Dataset(
        filename=str(path),
        start_time=START,
        end_time=END,
        use_dask=True,
        qair_method="era5_magnus",
    )
    tair = data.ds[data.var_names["Tair"]].compute()
    exp = specific_humidity_from_dewpoint(tair, tair - 5.0, patm=1010.0)
    np.testing.assert_array_equal(data.ds["qair"].compute().values, exp.values)


def test_era5_magnus_does_not_read_psfc(synthetic_store):
    path, _ = synthetic_store
    data = CONUS404Dataset(
        filename=str(path),
        start_time=START,
        end_time=END,
        use_dask=True,
        qair_method="era5_magnus",
    )
    assert "PSFC" not in data.ds.data_vars


# --- variable bookkeeping --------------------------------------------------


def test_var_names_after_post_process(dataset):
    """Only the seven ROMS surface-forcing fields survive."""
    data, _ = dataset
    assert set(data.var_names) == {
        "uwnd",
        "vwnd",
        "swrad",
        "lwrad",
        "Tair",
        "rain",
        "qair",
    }
    assert data.opt_var_names == {}


def test_auxiliary_variables_are_dropped(dataset):
    """Nothing SurfaceForcing would try to regrid and fail on is left behind."""
    data, _ = dataset
    for name in ("ACSWUPB", "TD2", "PSFC", "Q2", "COSALPHA", "SINALPHA", "LANDMASK"):
        assert name not in data.ds.data_vars, f"{name} survived post_process"


def test_source_mask_is_all_ones(dataset):
    """An atmospheric source is valid everywhere inside its footprint.

    A land/sea mask here would become xESMF's `mask_in`, renormalizing the
    bilinear weights over water cells only and punching NaN holes through coastal
    ROMS points.
    """
    data, _ = dataset
    assert (data.ds["mask"].values == 1).all()


def test_land_mask_kept_separately(dataset):
    data, _ = dataset
    assert "mask_land" in data.ds
    assert set(np.unique(data.ds["mask_land"].values)) <= {0, 1}


def test_staggered_coords_dropped(dataset):
    data, _ = dataset
    assert "x_stag" not in data.ds.dims
    assert "y_stag" not in data.ds.dims


# --- time selection --------------------------------------------------------


@pytest.mark.parametrize(
    ("pad", "expected_first"),
    [(True, START - timedelta(hours=1)), (False, START)],
)
def test_extra_leading_record_is_consumed_by_the_diff(
    synthetic_store, pad, expected_first
):
    """The 1 h widening leaves exactly the series an undifferenced source would."""
    path, _ = synthetic_store
    data = CONUS404Dataset(
        filename=str(path),
        start_time=START,
        end_time=END,
        use_dask=True,
        start_time_pad=pad,
    )
    assert data.ds.time.values[0] == np.datetime64(expected_first)


def test_start_before_store_beginning_raises(tmp_path):
    """A missing lead record is an error, not a silently shortened series."""
    ds, _ = synthetic_conus404(start=datetime(2020, 6, 15, 0), ntime=12)
    path = write_conus404_store(tmp_path / "late_start", ds, fmt="zarr")
    with pytest.raises(ValueError, match="accumulated since the model start"):
        CONUS404Dataset(
            filename=str(path),
            start_time=datetime(2020, 6, 15, 0),
            end_time=datetime(2020, 6, 15, 6),
            use_dask=True,
        )


# --- geometry --------------------------------------------------------------


def test_minimal_grid_spacing_reads_projected_axes(dataset):
    """The projected x/y axes give the spacing exactly, not approximately."""
    data, _ = dataset
    assert data.compute_minimal_grid_spacing(data.ds) == pytest.approx(GRID_SPACING_M)


def test_resolution_is_about_four_km_in_degrees(dataset):
    data, _ = dataset
    assert data.resolution == pytest.approx(4000.0 / 111_320.0, rel=0.05)


def test_is_global_is_false(dataset):
    data, _ = dataset
    assert data.is_global is False


def test_choose_subdomain_selects_a_subset(dataset):
    data, expected = dataset
    lat, lon = expected["lat"], expected["lon"]
    target = {
        "lat": xr.DataArray(lat[15:25, 20:30], dims=("eta_rho", "xi_rho")),
        "lon": xr.DataArray(lon[15:25, 20:30], dims=("eta_rho", "xi_rho")),
        "straddle": True,
    }
    sub = data.choose_subdomain(target, buffer_points=2, return_copy=True)
    assert sub.ds.sizes["y"] < data.ds.sizes["y"]
    assert sub.ds.sizes["x"] < data.ds.sizes["x"]
    # The requested window must still be fully inside the selection.
    assert float(sub.ds.lat.min()) <= lat[15:25, 20:30].min()
    assert float(sub.ds.lat.max()) >= lat[15:25, 20:30].max()


def test_choose_subdomain_outside_footprint_raises(dataset):
    data, _ = dataset
    target = {
        "lat": xr.DataArray(np.array([[-40.0, -39.0]]), dims=("eta_rho", "xi_rho")),
        "lon": xr.DataArray(np.array([[20.0, 21.0]]), dims=("eta_rho", "xi_rho")),
        "straddle": True,
    }
    with pytest.raises(ValueError, match="does not intersect the CONUS404Dataset"):
        data.choose_subdomain(target, buffer_points=1, return_copy=True)


def test_missing_2d_coords_raises(tmp_path):
    """A source without 2D lat/lon is rejected with a clear message."""
    ds, _ = synthetic_conus404(ntime=6)
    ds = ds.drop_vars("lat")
    path = write_conus404_store(tmp_path / "no_lat", ds, fmt="zarr")
    with pytest.raises(ValueError, match="expects a 2D latitude coordinate"):
        CONUS404Dataset(
            filename=str(path),
            start_time=datetime(2020, 6, 14, 23) + timedelta(hours=2),
            end_time=datetime(2020, 6, 14, 23) + timedelta(hours=4),
            use_dask=True,
        )


# --- fills -----------------------------------------------------------------


def test_xesmf_prefill_is_rejected(dataset):
    """The source-on-source xESMF fill needs 1D axes and cannot apply here."""
    data, _ = dataset
    data.needs_lateral_fill = True
    try:
        with pytest.raises(NotImplementedError, match="curvilinear source"):
            data.apply_prefill("inverse_dist")
    finally:
        data.needs_lateral_fill = False


def test_prefill_is_a_noop_when_source_is_nan_free(dataset, caplog):
    data, _ = dataset
    with caplog.at_level("INFO"):
        data.apply_prefill("2d_lateral_fill", prefill_was_user_set=True)
    assert "no-op" in caplog.text
