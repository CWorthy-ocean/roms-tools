"""Tests for the source-layering helpers.

These are pure-array tests: no datasets, no I/O, no regridding. Everything the
merge can get wrong -- silent index alignment, a coarse/fine shape mismatch, chunk
misalignment, an accidental compute -- is cheap to pin down here and expensive to
diagnose once it is buried in a full SurfaceForcing run.
"""

import numpy as np
import pytest
import xarray as xr

from roms_tools.blend import (
    align_fallback_time,
    coverage_fraction,
    layer_field,
    layer_fields,
)

NT, NY, NX = 4, 5, 6


def _field(value, *, times=None, with_time=True):
    """A (time, eta_rho, xi_rho) field filled with `value`."""
    if with_time:
        if times is None:
            times = np.array(
                [
                    np.datetime64("2020-06-15T00") + np.timedelta64(h, "h")
                    for h in range(NT)
                ]
            )
        data = np.full((len(times), NY, NX), float(value))
        return xr.DataArray(
            data,
            dims=("time", "eta_rho", "xi_rho"),
            coords={"time": times},
            attrs={"units": "widgets"},
        )
    data = np.full((NY, NX), float(value))
    return xr.DataArray(data, dims=("eta_rho", "xi_rho"), attrs={"units": "widgets"})


def _holed(value, hole_slice):
    """A field with `value` everywhere except NaN inside `hole_slice`."""
    da = _field(value)
    da[:, :, hole_slice] = np.nan
    return da


# --- layer_field -----------------------------------------------------------


def test_hard_edge_picks_primary_where_valid_and_fallback_in_holes():
    primary = _holed(1.0, slice(3, None))
    fallback = _field(2.0)
    out = layer_field(primary, fallback)
    assert (out.isel(xi_rho=slice(0, 3)) == 1.0).all()
    assert (out.isel(xi_rho=slice(3, None)) == 2.0).all()


def test_hard_edge_keeps_primary_attrs():
    out = layer_field(_holed(1.0, slice(3, None)), _field(2.0))
    assert out.attrs["units"] == "widgets"


def test_nan_in_fallback_propagates():
    """Documented behavior: a hole the fallback cannot fill stays a hole.

    Left to the caller's validation (`nan_check`) rather than silently patched,
    since it means neither source covers that point.
    """
    primary = _holed(1.0, slice(3, None))
    fallback = _holed(2.0, slice(5, None))
    out = layer_field(primary, fallback)
    assert out.isel(xi_rho=5).isnull().all()
    assert (out.isel(xi_rho=slice(3, 5)) == 2.0).all()


def test_all_valid_primary_is_a_noop():
    primary, fallback = _field(1.0), _field(2.0)
    xr.testing.assert_identical(layer_field(primary, fallback), primary)


# --- weights (the feathering hook) ----------------------------------------


def test_binary_weights_reproduce_the_hard_edge_exactly():
    """The weighted and unweighted paths must not be able to drift apart."""
    primary = _holed(1.0, slice(3, None))
    fallback = _field(2.0)
    ones = xr.ones_like(primary.isel(time=0, drop=True))
    np.testing.assert_array_equal(
        layer_field(primary, fallback, weights=ones).values,
        layer_field(primary, fallback).values,
    )


def test_weights_are_forced_to_zero_where_primary_is_missing():
    """A weight of 1 over a hole must still give pure fallback, not zero."""
    primary = _holed(1.0, slice(3, None))
    fallback = _field(2.0)
    ones = xr.ones_like(primary.isel(time=0, drop=True))
    out = layer_field(primary, fallback, weights=ones)
    assert (out.isel(xi_rho=slice(3, None)) == 2.0).all()


def test_half_weights_average_the_two_sources():
    primary, fallback = _field(1.0), _field(3.0)
    half = xr.full_like(primary.isel(time=0, drop=True), 0.5)
    out = layer_field(primary, fallback, weights=half)
    np.testing.assert_allclose(out.values, 2.0)


# --- dask laziness --------------------------------------------------------


def test_layer_fields_stays_lazy_and_computes_nothing():
    """The merge must build graph, not results.

    A merge that computed eagerly would materialise the full (time, eta, xi)
    output for every variable, which is exactly what the lazy pipeline exists to
    avoid.
    """
    import dask
    from dask.callbacks import Callback

    primary = {"Tair": _holed(1.0, slice(3, None)).chunk({"time": 1})}
    fallback = {"Tair": _field(2.0).chunk({"time": 2})}

    class _NoCompute(Callback):
        def __init__(self):
            self.ran = 0

        def _start(self, dsk):
            self.ran += 1

    cb = _NoCompute()
    with cb:
        out = layer_fields(primary, fallback)
    assert cb.ran == 0, "layer_fields triggered a dask compute"
    assert dask.is_dask_collection(out["Tair"].data)
    assert out["Tair"].chunks is not None


def test_fallback_is_rechunked_to_the_primary():
    primary = {"Tair": _holed(1.0, slice(3, None)).chunk({"time": 1})}
    fallback = {"Tair": _field(2.0).chunk({"time": 4})}
    out = layer_fields(primary, fallback)
    assert out["Tair"].chunksizes["time"] == primary["Tair"].chunksizes["time"]


def test_rechunk_can_be_disabled():
    primary = {"Tair": _holed(1.0, slice(3, None)).chunk({"time": 1})}
    fallback = {"Tair": _field(2.0).chunk({"time": 4})}
    out = layer_fields(primary, fallback, rechunk_to_primary=False)
    assert out["Tair"].compute().notnull().all()


# --- time alignment -------------------------------------------------------


def test_exact_alignment_selects_matching_stamps():
    times = np.array(
        [np.datetime64("2020-06-15T00") + np.timedelta64(h, "h") for h in range(NT + 3)]
    )
    fallback = _field(2.0, times=times)
    primary = _field(1.0)
    aligned = align_fallback_time(fallback, primary["time"])
    np.testing.assert_array_equal(aligned["time"].values, primary["time"].values)


def test_missing_fallback_stamps_raise_rather_than_silently_intersecting():
    """`xr.where` would quietly return the intersection; we refuse instead."""
    primary = _field(1.0)
    fallback = _field(2.0).isel(time=slice(0, 2))
    with pytest.raises(ValueError, match="missing 2 time step"):
        layer_fields({"Tair": primary}, {"Tair": fallback})


def test_nearest_alignment_tolerates_jitter():
    primary = _field(1.0)
    shifted = primary["time"].values + np.timedelta64(3, "m")
    fallback = _field(2.0, times=shifted)
    aligned = align_fallback_time(
        fallback,
        primary["time"],
        method="nearest",
        tolerance=np.timedelta64(30, "m"),
    )
    np.testing.assert_array_equal(aligned["time"].values, primary["time"].values)


def test_linear_alignment_interpolates():
    times = np.array(
        [np.datetime64("2020-06-15T00") + np.timedelta64(2 * h, "h") for h in range(NT)]
    )
    fallback = _field(2.0, times=times)
    primary = _field(1.0)
    aligned = align_fallback_time(fallback, primary["time"], method="linear")
    assert aligned.sizes["time"] == NT


def test_unknown_alignment_method_raises():
    with pytest.raises(ValueError, match="Unknown time alignment method"):
        align_fallback_time(_field(2.0), _field(1.0)["time"], method="bogus")


def test_time_free_field_passes_through():
    static = _field(2.0, with_time=False)
    out = align_fallback_time(static, _field(1.0)["time"])
    xr.testing.assert_identical(out, static)


# --- variable-set and shape checks ---------------------------------------


def test_variable_only_in_fallback_raises_by_default():
    with pytest.raises(ValueError, match="on_missing_primary_var"):
        layer_fields(
            {"Tair": _field(1.0)},
            {"Tair": _field(2.0), "rain": _field(3.0)},
        )


def test_variable_only_in_fallback_can_be_taken_wholesale():
    out = layer_fields(
        {"Tair": _holed(1.0, slice(3, None))},
        {"Tair": _field(2.0), "rain": _field(3.0)},
        on_missing_primary_var="fallback",
    )
    assert set(out) == {"Tair", "rain"}
    assert (out["rain"] == 3.0).all()


def test_variable_only_in_primary_raises():
    with pytest.raises(ValueError, match="nothing to fill its gaps with"):
        layer_fields(
            {"Tair": _field(1.0), "rain": _field(3.0)},
            {"Tair": _field(2.0)},
        )


def test_unknown_on_missing_primary_var_raises():
    with pytest.raises(ValueError, match="Unknown on_missing_primary_var"):
        layer_fields(
            {"Tair": _field(1.0)},
            {"Tair": _field(2.0)},
            on_missing_primary_var="bogus",
        )


def test_shape_mismatch_names_the_coarse_grid_cause():
    """Positional alignment on eta_rho/xi_rho would otherwise fail opaquely."""
    primary = _field(1.0)
    fallback = _field(2.0).isel(xi_rho=slice(0, 3))
    with pytest.raises(ValueError, match="use_coarse_grid"):
        layer_fields({"Tair": primary}, {"Tair": fallback})


# --- coverage diagnostic -------------------------------------------------


def test_coverage_fraction_detects_full_and_empty_coverage():
    assert coverage_fraction(_field(1.0)) == pytest.approx(1.0)
    assert coverage_fraction(_field(np.nan)) == pytest.approx(0.0)


def test_coverage_fraction_is_partial_for_a_holed_field():
    # 3 of 6 xi_rho columns are NaN.
    assert coverage_fraction(_holed(1.0, slice(3, None))) == pytest.approx(0.5)


def test_coverage_fraction_respects_the_wet_mask():
    primary = _holed(1.0, slice(3, None))
    # Mask out exactly the columns the primary is missing: coverage is then full.
    mask = xr.DataArray(
        np.repeat([[1, 1, 1, 0, 0, 0]], NY, axis=0), dims=("eta_rho", "xi_rho")
    )
    assert coverage_fraction(primary, mask) == pytest.approx(1.0)
