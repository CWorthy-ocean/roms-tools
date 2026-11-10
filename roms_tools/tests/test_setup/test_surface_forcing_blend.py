"""Tests for ``roms_tools.setup.surface_forcing_blend``.

Synthetic arrays and small datasets only: no source data, no regridding. Everything
the merge can get wrong -- silent index alignment, a coarse/fine shape mismatch,
chunk misalignment, an accidental compute, a NaN left at an ocean point, the
encodings of the written files -- is cheap to pin down here and expensive to
diagnose once it is buried in a full SurfaceForcing run.
"""

import logging

import numpy as np
import pytest
import xarray as xr
import yaml

from roms_tools.setup.surface_forcing_blend import (
    _interp_time_linear,
    align_fallback_time,
    blend_surface_forcing,
    blend_surface_forcing_from_yaml,
    blend_to_yaml,
    compute_blend_weights,
    coverage_fraction,
    layer_field,
    layer_fields,
    save_blended_surface_forcing,
)
from roms_tools.setup.utils import RECIPE_ATTR, yaml_text

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

    ``layer_field`` leaves it NaN rather than patching it, since it means neither
    source covers that point; ``blend_surface_forcing`` then raises if it is at an
    ocean point (see ``test_ocean_nan_left_after_blending_raises``).
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


def test_nearest_alignment_requires_a_tolerance():
    """An unlimited "nearest" would match a fallback from any period."""
    primary = _field(1.0)
    with pytest.raises(ValueError, match="needs a `time_tolerance`"):
        align_fallback_time(_field(2.0), primary["time"], method="nearest")


def test_nearest_alignment_rejects_a_fallback_from_another_period():
    primary = _field(1.0)
    a_year_later = primary["time"].values + np.timedelta64(365, "D")
    fallback = _field(2.0, times=a_year_later)
    with pytest.raises(ValueError, match="no record within the time tolerance"):
        align_fallback_time(
            fallback,
            primary["time"],
            method="nearest",
            tolerance=np.timedelta64(30, "m"),
        )


def test_exact_alignment_error_does_not_recommend_an_unlimited_nearest():
    primary = _field(1.0)
    fallback = _field(2.0).isel(time=slice(0, 2))
    with pytest.raises(ValueError, match="time_tolerance"):
        align_fallback_time(fallback, primary["time"])


def test_linear_alignment_interpolates():
    times = np.array(
        [np.datetime64("2020-06-15T00") + np.timedelta64(2 * h, "h") for h in range(NT)]
    )
    fallback = _field(2.0, times=times)
    primary = _field(1.0)
    aligned = align_fallback_time(fallback, primary["time"], method="linear")
    assert aligned.sizes["time"] == NT


def _ramp(times):
    """A (time, eta_rho, xi_rho) float32 field that varies in time and space."""
    n = len(times)
    data = (np.arange(n)[:, None, None] ** 2 + np.arange(NY * NX).reshape(NY, NX)) * 1.0
    return xr.DataArray(
        data.astype(np.float32),
        dims=("time", "eta_rho", "xi_rho"),
        coords={"time": times},
        name="Tair",
        attrs={"units": "widgets"},
    )


@pytest.mark.parametrize("axis", ["datetime", "days"])
def test_linear_alignment_matches_xarray_interp(axis):
    """Same values as `interp`, NaN outside the record, dtype and attrs kept."""
    if axis == "datetime":
        start, hour = np.datetime64("2020-06-15T00", "ns"), np.timedelta64(1, "h")
    else:
        start, hour = 0.0, 1.0 / 24.0
    src_times = np.array([start + hour * (2 * k) for k in range(6)])
    # On a sample, between samples, on the last sample, and past either end.
    target = np.array([start + hour * h for h in (-1, 0, 1, 2.5, 7, 10, 11)])
    fallback = _ramp(src_times)

    aligned = align_fallback_time(
        fallback, xr.DataArray(target, dims="time"), method="linear"
    )

    expected = fallback.interp(time=target)
    np.testing.assert_allclose(aligned.values, expected.values, rtol=1e-6)
    assert bool(aligned.isel(time=[0, -1]).isnull().all())
    assert aligned.dtype == np.float32
    assert aligned.dims == fallback.dims
    assert aligned.attrs == fallback.attrs
    np.testing.assert_array_equal(aligned["time"].values, target)


def test_linear_alignment_on_a_sample_ignores_a_nan_neighbour():
    """A stamp that is one of the samples takes that sample alone."""
    times = np.arange(5) / 24.0
    fallback = _ramp(times)
    fallback[1] = np.nan
    fallback[4] = np.nan
    target = xr.DataArray(times[[0, 2, 3]], dims="time")

    aligned = align_fallback_time(fallback, target, method="linear")

    np.testing.assert_array_equal(aligned.values, fallback.values[[0, 2, 3]])


def test_linear_alignment_of_a_single_record():
    """One record serves its own stamp; any other stamp is outside it."""
    times = np.array([0.5])
    fallback = _ramp(times)
    target = xr.DataArray(np.array([0.25, 0.5, 0.75]), dims="time")

    aligned = align_fallback_time(fallback, target, method="linear")

    np.testing.assert_array_equal(aligned.values[1], fallback.values[0])
    assert bool(aligned.isel(time=[0, 2]).isnull().all())


def test_time_interpolation_holds_the_end_value_only_within_the_hold():
    times = np.arange(4) / 24.0
    field = _ramp(times)
    hour = 1.0 / 24.0
    target = np.array([-2 * hour, -0.5 * hour, 3.5 * hour, 5 * hour])

    out = _interp_time_linear(field, target, hold=hour)

    np.testing.assert_array_equal(out.values[1], field.values[0])
    np.testing.assert_array_equal(out.values[2], field.values[-1])
    assert bool(out.isel(time=[0, 3]).isnull().all())


def _counting_blocks(field, counter):
    """Chunk `field` by day and record which time block each read touches."""
    import dask.array as dsa

    def _read(block, block_info=None):
        counter.append(block_info[0]["chunk-location"][0])
        return block

    data = dsa.from_array(field.values, chunks=(24, NY, NX)).map_blocks(
        _read, dtype=field.dtype
    )
    return field.copy(data=data)


def test_linear_alignment_reads_only_the_bracketing_blocks():
    """One day of output must not read the fallback's whole record.

    `xarray.interp` along a chunked dimension builds each output block from every
    input block; on a year of hourly forcing that is tens of GB per variable.
    """
    hours = np.arange(10 * 24)
    src_times = np.datetime64("2020-06-15T00", "ns") + hours * np.timedelta64(1, "h")
    blocks_read: list[int] = []
    fallback = _counting_blocks(_ramp(src_times), blocks_read)
    target = xr.DataArray(src_times[:-1] + np.timedelta64(30, "m"), dims="time")

    aligned = align_fallback_time(fallback, target, method="linear")
    assert aligned.chunks is not None
    assert blocks_read == [], "interpolating in time triggered a compute"

    aligned.isel(time=slice(0, 24)).compute()
    # The last stamp of day 0 sits between day 0's last record and day 1's first.
    assert set(blocks_read) == {0, 1}


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


# --- dataset-level API -----------------------------------------------------


def _sf_dataset(value, *, hole_slice=None, rad_offset_h=-0.5, source="X", nt=NT):
    """A SurfaceForcing-like Dataset: time in days, rad_time, ROMS-named vars."""
    days = np.arange(nt) / 24.0
    ds = xr.Dataset(
        {
            name: xr.DataArray(
                np.full((nt, NY, NX), float(value)),
                dims=("time", "eta_rho", "xi_rho"),
                attrs={"units": "u"},
            )
            for name in ("Tair", "swrad")
        },
        coords={"time": days},
        attrs={"source": source, "use_coarse_grid": "False"},
    )
    ds = ds.assign_coords(rad_time=("time", days + rad_offset_h / 24.0))
    if hole_slice is not None:
        for name in ds.data_vars:
            ds[name][:, :, hole_slice] = np.nan
    return ds


def _grid_ds():
    return xr.Dataset(
        {
            "pm": (("eta_rho", "xi_rho"), np.full((NY, NX), 1 / 1000.0)),
            "pn": (("eta_rho", "xi_rho"), np.full((NY, NX), 1 / 1000.0)),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((NY, NX), dtype=np.int32)),
        }
    )


def test_dataset_blend_hard_edge_fills_the_hole():
    out = blend_surface_forcing(
        _sf_dataset(1.0, hole_slice=slice(3, None)),
        _sf_dataset(2.0, source="Y"),
        grid=_grid_ds(),
    )
    assert (out["Tair"].isel(xi_rho=slice(0, 3)) == 1.0).all()
    assert (out["Tair"].isel(xi_rho=slice(3, None)) == 2.0).all()
    assert out.attrs["blend_method"] == "hard_edge"
    assert out.attrs["fallback_source"] == "Y"
    assert "rad_time" in out.coords


def test_dataset_blend_feathered_is_monotone_across_the_band():
    out = blend_surface_forcing(
        _sf_dataset(1.0, hole_slice=slice(3, None)),
        _sf_dataset(2.0, source="Y"),
        grid=_grid_ds(),
        blend_width_km=2.0,
    )
    row = out["Tair"].isel(time=0, eta_rho=0).values
    assert row[0] == 1.0 and row[-1] == 2.0
    assert (np.diff(row) >= -1e-12).all()
    assert out.attrs["blend_method"] == "feathered"


def test_feathered_blend_rejects_an_empty_common_footprint():
    """One variable all NaN in the first record must not zero every weight.

    The footprint is where every primary variable has data at the first time
    step. With `swrad` missing there it is empty, and the feathered blend would
    silently be the fallback alone for all variables at all times.
    """
    primary = _sf_dataset(1.0, hole_slice=slice(3, None))
    primary["swrad"][0] = np.nan
    fallback = _sf_dataset(2.0, source="Y")

    with pytest.raises(ValueError, match=r"share no footprint.*\['swrad'\]"):
        blend_surface_forcing(primary, fallback, grid=_grid_ds(), blend_width_km=2.0)

    # The hard edge needs no footprint and still uses the primary.
    out = blend_surface_forcing(primary, fallback, grid=_grid_ds())
    assert (out["Tair"].isel(xi_rho=slice(0, 3)) == 1.0).all()


def test_feathered_blend_of_an_empty_primary_is_the_fallback():
    """A primary with no data at all is still allowed: the result is the fallback."""
    primary = _sf_dataset(1.0, hole_slice=slice(0, None))
    out = blend_surface_forcing(
        primary, _sf_dataset(2.0, source="Y"), grid=_grid_ds(), blend_width_km=2.0
    )
    assert (out["Tair"] == 2.0).all()


def test_a_grid_is_required_without_a_surface_forcing_input():
    """Without a grid there is no land mask, so leftover NaNs cannot be judged."""
    with pytest.raises(ValueError, match="Pass `grid=`"):
        blend_surface_forcing(
            _sf_dataset(1.0, hole_slice=slice(3, None)), _sf_dataset(2.0)
        )


def test_a_grid_without_a_land_mask_is_rejected():
    with pytest.raises(ValueError, match="has no `mask_rho`"):
        blend_surface_forcing(
            _sf_dataset(1.0, hole_slice=slice(3, None)),
            _sf_dataset(2.0),
            grid=_grid_ds().drop_vars("mask_rho"),
        )


def test_compute_blend_weights_full_coverage_is_all_ones():
    w = compute_blend_weights(
        xr.DataArray(np.ones((NY, NX), bool), dims=("eta_rho", "xi_rho")),
        cell_size_m=(1000.0, 1000.0),
        width_km=3.0,
    )
    assert (w == 1.0).all()


def test_compute_blend_weights_rejects_bad_profile():
    fp = xr.DataArray(np.ones((NY, NX), bool), dims=("eta_rho", "xi_rho"))
    with pytest.raises(ValueError, match="blend_profile"):
        compute_blend_weights(fp, cell_size_m=(1.0, 1.0), width_km=1.0, profile="bogus")


def test_unit_mismatch_is_caught():
    f = _sf_dataset(2.0)
    f["Tair"].attrs["units"] = "other"
    with pytest.raises(ValueError, match="units"):
        blend_surface_forcing(
            _sf_dataset(1.0, hole_slice=slice(3, None)), f, grid=_grid_ds()
        )


def test_coarse_grid_mismatch_is_caught():
    f = _sf_dataset(2.0)
    f.attrs["use_coarse_grid"] = "True"
    with pytest.raises(ValueError, match="use_coarse_grid"):
        blend_surface_forcing(_sf_dataset(1.0), f, grid=_grid_ds())


def test_differing_model_reference_dates_are_caught():
    """Days since different dates are not comparable, whatever `time_align` is."""
    p = _sf_dataset(1.0, hole_slice=slice(3, None))
    f = _sf_dataset(2.0, source="Y")
    p.attrs["model_reference_date"] = "2000-01-01 00:00:00"
    f.attrs["model_reference_date"] = "2001-01-01 00:00:00"
    with pytest.raises(ValueError, match="model_reference_date"):
        blend_surface_forcing(
            p,
            f,
            grid=_grid_ds(),
            time_align="nearest",
            time_tolerance=np.timedelta64(1, "m"),
        )


def test_differing_rad_time_offsets_shift_the_fallback_radiation():
    primary = _sf_dataset(1.0, hole_slice=slice(3, None), rad_offset_h=-0.5)
    fallback = _sf_dataset(2.0, rad_offset_h=0.0)
    # Ramp the fallback's radiation in time so a shift is observable.
    fallback["swrad"] = fallback["swrad"] + xr.DataArray(
        np.arange(NT) * 10.0, dims="time"
    )
    out = blend_surface_forcing(primary, fallback, grid=_grid_ds())
    got = out["swrad"].isel(xi_rho=-1, eta_rho=0).values
    # Shifted by half an hour: interior steps land half-way between samples.
    np.testing.assert_allclose(got[1:-1], 2.0 + np.arange(1, NT - 1) * 10.0 - 5.0)
    # The first stamp falls half an hour before the fallback's record: it holds the
    # first value rather than becoming a NaN record.
    assert got[0] == 2.0
    assert not bool(out["swrad"].isnull().any())


def test_retimed_radiation_survives_stamps_that_differ():
    """Re-timed radiation must not be reindexed onto the fallback's own stamps.

    With a second of jitter no fallback stamp equals a primary one, so radiation
    that was re-timed and then aligned on the fallback's index came out all NaN.
    """
    primary = _sf_dataset(1.0, hole_slice=slice(3, None), rad_offset_h=-0.5)
    fallback = _sf_dataset(2.0, rad_offset_h=0.0)
    fallback["swrad"] = fallback["swrad"] + xr.DataArray(
        np.arange(NT) * 10.0, dims="time"
    )
    jitter = 1.0 / 86400.0
    fallback = fallback.assign_coords(
        time=fallback["time"] + jitter, rad_time=fallback["rad_time"] + jitter
    )
    out = blend_surface_forcing(
        primary,
        fallback,
        grid=_grid_ds(),
        time_align="nearest",
        time_tolerance=np.timedelta64(1, "m"),
        fill_value=None,
    )
    got = out["swrad"].isel(xi_rho=-1, eta_rho=0).values
    # Half an hour (less the jitter) before each fallback sample.
    np.testing.assert_allclose(got[1:], 2.0 + np.arange(1, NT) * 10.0 - 5.0, atol=0.01)
    # The first stamp precedes the fallback's record by the shift plus the jitter.
    assert got[0] == 2.0
    assert not bool(out["swrad"].isnull().any())


def test_radiation_retiming_reads_only_the_bracketing_blocks():
    """Re-timing radiation must stay chunk-local, like the rest of the blend."""
    nt = 10 * 24
    primary = _sf_dataset(1.0, hole_slice=slice(3, None), rad_offset_h=-0.5, nt=nt)
    fallback = _sf_dataset(2.0, rad_offset_h=0.0, nt=nt)
    blocks_read: list[int] = []
    fallback["swrad"] = _counting_blocks(fallback["swrad"], blocks_read)

    out = blend_surface_forcing(primary, fallback, grid=_grid_ds(), fill_value=None)
    assert blocks_read == [], "the blend triggered a compute"

    out["swrad"].isel(time=slice(24, 48)).compute()
    # Day 1 of the output, shifted back half an hour, needs days 0 and 1.
    assert set(blocks_read) == {0, 1}


def test_paths_are_accepted(tmp_path):
    p, f = tmp_path / "p.nc", tmp_path / "f.nc"
    _sf_dataset(1.0, hole_slice=slice(3, None)).to_netcdf(p)
    _sf_dataset(2.0, source="Y").to_netcdf(f)
    out = blend_surface_forcing(p, f, grid=_grid_ds())
    assert (out["Tair"].isel(xi_rho=slice(3, None)) == 2.0).all()


def test_feathered_blend_keeps_float32():
    """The float64 weights must not promote float32 fields (doubling file size)."""
    p = _sf_dataset(1.0, hole_slice=slice(3, None)).astype("float32")
    f = _sf_dataset(2.0, source="Y").astype("float32")
    out = blend_surface_forcing(p, f, grid=_grid_ds(), blend_width_km=2.0)
    for name in ("Tair", "swrad"):
        assert out[name].dtype == np.float32


def test_save_blended_surface_forcing_writes_roms_safe_cdf5(tmp_path):
    """A datetime abs_time is written as float64, not int64, in CDF5 month files."""
    import netCDF4

    p = _sf_dataset(1.0, hole_slice=slice(3, None))
    f = _sf_dataset(2.0, source="Y")
    abs_time = np.datetime64("2020-06-30T22") + np.arange(NT) * np.timedelta64(1, "h")
    p = p.assign_coords(abs_time=("time", abs_time))
    out = blend_surface_forcing(p, f, grid=_grid_ds())

    files = save_blended_surface_forcing(
        out, tmp_path / "blended.nc", format="NETCDF3_64BIT_DATA"
    )

    # NT=4 hourly records from 06-30 22:00 span two months.
    assert [x.name for x in files] == ["blended_202006.nc", "blended_202007.nc"]
    for path in files:
        with netCDF4.Dataset(path) as nc:
            assert nc.data_model == "NETCDF3_64BIT_DATA"
            assert nc["abs_time"].dtype == np.float64
            assert not [v for v in nc.variables if nc[v].dtype == np.int64]


def _grid_ds_with_mask(land_cols=()):
    g = _grid_ds()
    mask = np.ones((NY, NX), dtype=np.int32)
    for c in land_cols:
        mask[:, c] = 0
    return g.assign(mask_rho=(("eta_rho", "xi_rho"), mask))


def test_full_weight_primary_survives_a_fallback_nan():
    """At weight 1 the primary stands alone, as on the hard edge."""
    primary = _holed(1.0, slice(3, None))
    fallback = _holed(2.0, slice(0, 1))  # NaN deep inside the primary's footprint
    weights = xr.DataArray(
        np.where(np.arange(NX) < 3, 1.0, 0.0)[None, :].repeat(NY, 0),
        dims=("eta_rho", "xi_rho"),
    )
    out = layer_field(primary, fallback, weights=weights)
    xr.testing.assert_identical(out, layer_field(primary, fallback))


def test_nearest_tolerance_accepts_timedelta_on_a_days_axis():
    """SurfaceForcing time is float days; a timedelta tolerance must still work."""
    p = _sf_dataset(1.0, hole_slice=slice(3, None))
    f = _sf_dataset(2.0, source="Y")
    f = f.assign_coords(
        time=f["time"] + 1.0 / 86400.0, rad_time=f["rad_time"] + 1.0 / 86400.0
    )  # one second of jitter
    out = blend_surface_forcing(
        p,
        f,
        grid=_grid_ds(),
        time_align="nearest",
        time_tolerance=np.timedelta64(1, "m"),
    )
    assert (out["Tair"].isel(xi_rho=slice(3, None)) == 2.0).all()


def test_fallback_only_variable_is_time_aligned():
    """A variable taken wholly from the fallback goes through the same alignment."""
    p = _sf_dataset(1.0, hole_slice=slice(3, None)).drop_vars("swrad")
    f = _sf_dataset(2.0, source="Y")
    f = f.assign_coords(
        time=f["time"] + 1.0 / 86400.0, rad_time=f["rad_time"] + 1.0 / 86400.0
    )
    out = blend_surface_forcing(
        p,
        f,
        grid=_grid_ds(),
        time_align="nearest",
        time_tolerance=np.timedelta64(1, "m"),
        on_missing_primary_var="fallback",
        fill_value=None,
    )
    assert (out["swrad"] == 2.0).all()


def test_ocean_nan_left_after_blending_raises():
    """With a grid, land is filled but a NaN at an ocean point is an error."""
    p = _sf_dataset(1.0, hole_slice=slice(3, None))
    f = _sf_dataset(2.0, source="Y", hole_slice=slice(5, None))
    # Column 5 is uncovered by both products: fine on land, an error at sea.
    out = blend_surface_forcing(p, f, grid=_grid_ds_with_mask(land_cols=[5]))
    assert (out["Tair"].isel(xi_rho=5) == 0.0).all()
    with pytest.raises(ValueError, match="NaN value") as err:
        blend_surface_forcing(p, f, grid=_grid_ds_with_mask())
    # The way out named is the blend's own, not SurfaceForcing's.
    assert "fill_value=None" in str(err.value)
    assert "bypass_validation" not in str(err.value)


def test_missing_grid_is_reported_before_any_file_is_opened(tmp_path):
    with pytest.raises(ValueError, match="Pass `grid=`"):
        blend_surface_forcing(
            tmp_path / "missing_primary.nc", tmp_path / "missing_fallback.nc"
        )


def test_blend_attrs_describe_the_result():
    p = _sf_dataset(1.0, hole_slice=slice(3, None))
    p.attrs["nan_outside_source_footprint"] = "True"
    f = _sf_dataset(2.0, source="Y")
    f.attrs["correct_radiation"] = "True"
    out = blend_surface_forcing(p, f, grid=_grid_ds(), blend_width_km=2)
    assert "nan_outside_source_footprint" not in out.attrs
    assert "layered" not in out.attrs
    assert "fallback_correct_radiation" not in out.attrs
    assert out.attrs["fallback_source"] == "Y"
    assert out.attrs["blend_method"] == "feathered"
    assert out.attrs["blend_width_km"] == "2.0"
    assert out.attrs["blend_profile"] == "smoothstep"
    assert "primary_coverage_fraction" in out.attrs
    assert RECIPE_ATTR in out.attrs


# --- recipe ------------------------------------------------------------------


GRID_SPEC = {"nx": 5, "ny": 4, "size_x": 100.0, "size_y": 80.0, "rot": 0.0}


def _with_recipe(ds, *, grid=GRID_SPEC, **spec):
    payload = {"SurfaceForcing": {"start_time": "2020-06-15T00:00:00", **spec}}
    if grid is not None:
        payload = {"Grid": grid, **payload}
    ds.attrs[RECIPE_ATTR] = yaml_text(payload)
    return ds


def _payload(ds):
    return list(yaml.safe_load_all(ds.attrs[RECIPE_ATTR]))[-1]


def test_recipe_records_both_layers_and_every_argument():
    p = _with_recipe(_sf_dataset(1.0, hole_slice=slice(3, None)), type="physics")
    f = _with_recipe(_sf_dataset(2.0, source="Y"), type="physics", extra=1)
    out = blend_surface_forcing(
        p,
        f,
        grid=_grid_ds(),
        blend_width_km=2,
        blend_profile="linear",
        time_align="nearest",
        time_tolerance=np.timedelta64(12, "h"),
        on_missing_primary_var="fallback",
        fill_value=None,
    )
    payload = _payload(out)
    assert payload["Grid"] == GRID_SPEC
    blend = payload["SurfaceForcingBlend"]
    assert blend["layers"] == [
        {"SurfaceForcing": {"start_time": "2020-06-15T00:00:00", "type": "physics"}},
        {
            "SurfaceForcing": {
                "start_time": "2020-06-15T00:00:00",
                "type": "physics",
                "extra": 1,
            }
        },
    ]
    assert {k: v for k, v in blend.items() if k != "layers"} == {
        "blend_width_km": 2.0,
        "blend_profile": "linear",
        "time_align": "nearest",
        "time_tolerance_days": 0.5,
        "on_missing_primary_var": "fallback",
        "fill_value": None,
    }


def test_recipe_records_the_defaults():
    out = blend_surface_forcing(
        _with_recipe(_sf_dataset(1.0, hole_slice=slice(3, None))),
        _with_recipe(_sf_dataset(2.0, source="Y")),
        grid=_grid_ds(),
    )
    assert _payload(out)["SurfaceForcingBlend"] | {"layers": None} == {
        "layers": None,
        "blend_width_km": 0.0,
        "blend_profile": "smoothstep",
        "time_align": "exact",
        "time_tolerance_days": None,
        "on_missing_primary_var": "error",
        "fill_value": 0.0,
    }


def test_input_without_a_recipe_is_recorded_as_such():
    p = _sf_dataset(1.0, hole_slice=slice(3, None))
    f = _with_recipe(_sf_dataset(2.0, source="Y"))
    out = blend_surface_forcing(p, f, grid=_grid_ds())
    payload = _payload(out)
    assert "Grid" in payload
    primary, fallback = payload["SurfaceForcingBlend"]["layers"]
    assert primary == {
        "no_recipe": "xarray.Dataset without a roms_tools_yaml attribute"
    }
    assert "SurfaceForcing" in fallback


def test_path_input_without_a_recipe_names_the_path(tmp_path):
    path = tmp_path / "no_recipe.nc"
    _sf_dataset(1.0, hole_slice=slice(3, None)).to_netcdf(path)
    out = blend_surface_forcing(
        path,
        _sf_dataset(2.0, source="Y"),
        grid=_grid_ds(),
        time_align="nearest",
        time_tolerance=1.0,
    )
    layers = _payload(out)["SurfaceForcingBlend"]["layers"]
    assert str(path) in layers[0]["no_recipe"]
    assert "Grid" not in _payload(out)


def test_recipe_attribute_round_trips_through_a_path_input(tmp_path):
    path = tmp_path / "with_recipe.nc"
    _with_recipe(_sf_dataset(1.0, hole_slice=slice(3, None))).to_netcdf(path)
    out = blend_surface_forcing(
        path, _with_recipe(_sf_dataset(2.0, source="Y")), grid=_grid_ds()
    )
    layers = _payload(out)["SurfaceForcingBlend"]["layers"]
    assert "SurfaceForcing" in layers[0]


def test_differing_grids_raise():
    p = _with_recipe(_sf_dataset(1.0, hole_slice=slice(3, None)))
    f = _with_recipe(_sf_dataset(2.0, source="Y"), grid={**GRID_SPEC, "nx": 6})
    with pytest.raises(ValueError, match="different grids"):
        blend_surface_forcing(p, f, grid=_grid_ds())


def test_grid_object_is_checked_against_the_layers_and_hoisted():
    from roms_tools import Grid
    from roms_tools.setup.utils import serialize_grid

    grid = Grid(nx=4, ny=3, size_x=100, size_y=100, center_lon=0, center_lat=0, rot=0)
    spec = yaml.safe_load(yaml.dump(serialize_grid(grid)))
    # Two products that carry no recipe: the grid object supplies the spec.
    p = _sf_dataset(1.0, hole_slice=slice(3, None))
    f = _sf_dataset(2.0, source="Y")
    out = blend_surface_forcing(p, f, grid=grid)
    assert _payload(out)["Grid"] == spec
    # A recipe on a different grid conflicts with it.
    with pytest.raises(ValueError, match="`grid` differs"):
        blend_surface_forcing(_with_recipe(p.copy()), _with_recipe(f.copy()), grid=grid)


def test_blend_used_as_a_layer_nests_and_hoists_the_grid():
    inner = blend_surface_forcing(
        _with_recipe(_sf_dataset(1.0, hole_slice=slice(3, None))),
        _with_recipe(_sf_dataset(2.0, source="Y", nt=NT)),
        grid=_grid_ds(),
        blend_width_km=3,
    )
    outer = blend_surface_forcing(
        _sf_dataset(0.0, hole_slice=slice(4, None)), inner, grid=_grid_ds()
    )
    payload = _payload(outer)
    assert payload["Grid"] == GRID_SPEC
    primary, fallback = payload["SurfaceForcingBlend"]["layers"]
    assert "no_recipe" in primary
    assert list(fallback) == ["SurfaceForcingBlend"]
    assert "Grid" not in fallback["SurfaceForcingBlend"]
    assert fallback["SurfaceForcingBlend"]["blend_width_km"] == 3.0


def test_blend_to_yaml_writes_the_attribute(tmp_path):
    out = blend_surface_forcing(
        _with_recipe(_sf_dataset(1.0, hole_slice=slice(3, None))),
        _with_recipe(_sf_dataset(2.0, source="Y")),
        grid=_grid_ds(),
    )
    blend_to_yaml(out, tmp_path / "blend.yaml")
    assert (tmp_path / "blend.yaml").read_text() == out.attrs[RECIPE_ATTR]


def test_blend_to_yaml_needs_a_blend(tmp_path):
    with pytest.raises(ValueError, match="not the output of `blend_surface_forcing`"):
        blend_to_yaml(_sf_dataset(1.0), tmp_path / "blend.yaml")
    assert not (tmp_path / "blend.yaml").exists()


def test_from_yaml_refuses_a_layer_without_a_recipe(tmp_path):
    out = blend_surface_forcing(
        _sf_dataset(1.0, hole_slice=slice(3, None)),
        _with_recipe(_sf_dataset(2.0, source="Y")),
        grid=_grid_ds(),
    )
    blend_to_yaml(out, tmp_path / "blend.yaml")
    with pytest.raises(ValueError, match="primary.*without a roms_tools_yaml"):
        blend_surface_forcing_from_yaml(tmp_path / "blend.yaml")


def test_from_yaml_refuses_a_file_without_a_blend(tmp_path):
    path = tmp_path / "plain.yaml"
    path.write_text(yaml_text({"Grid": GRID_SPEC}))
    with pytest.raises(ValueError, match="No SurfaceForcingBlend section"):
        blend_surface_forcing_from_yaml(path)


def test_recipe_survives_saving_into_the_files(tmp_path):
    out = blend_surface_forcing(
        _with_recipe(_sf_dataset(1.0, hole_slice=slice(3, None))),
        _with_recipe(_sf_dataset(2.0, source="Y")),
        grid=_grid_ds(),
    )
    (path,) = save_blended_surface_forcing(out, tmp_path / "blended", group=False)
    with xr.open_dataset(path, decode_times=False) as ds:
        assert ds.attrs[RECIPE_ATTR] == out.attrs[RECIPE_ATTR]


def test_radiation_shift_is_recorded_only_when_the_offsets_differ():
    grid = _grid_ds()
    same = blend_surface_forcing(
        _sf_dataset(1.0, hole_slice=slice(3, None), rad_offset_h=-0.5),
        _sf_dataset(2.0, rad_offset_h=-0.5),
        grid=grid,
    )
    assert "fallback_radiation_shift_s" not in same.attrs
    shifted = blend_surface_forcing(
        _sf_dataset(1.0, hole_slice=slice(3, None), rad_offset_h=-0.5),
        _sf_dataset(2.0, rad_offset_h=0.0),
        grid=grid,
    )
    assert shifted.attrs["fallback_radiation_shift_s"] == "1800.0"


def test_files_blended_from_disk_keep_time_unlimited(tmp_path):
    """A product opened from files has lost the encoding that marks `time`."""
    import netCDF4

    p, f = tmp_path / "p.nc", tmp_path / "f.nc"
    _sf_dataset(1.0, hole_slice=slice(3, None)).to_netcdf(p)
    _sf_dataset(2.0, source="Y").to_netcdf(f)
    out = blend_surface_forcing(p, f, grid=_grid_ds())

    (written,) = save_blended_surface_forcing(out, tmp_path / "blended", group=False)

    with netCDF4.Dataset(written) as nc:
        assert nc.dimensions["time"].isunlimited()


def test_saving_in_groups_without_abs_time_says_what_is_missing(tmp_path):
    out = blend_surface_forcing(
        _sf_dataset(1.0, hole_slice=slice(3, None)), _sf_dataset(2.0), grid=_grid_ds()
    )
    assert "abs_time" not in out.variables
    with pytest.raises(ValueError, match="abs_time.*group=False"):
        save_blended_surface_forcing(out, tmp_path / "blended")


def test_float_tolerance_on_a_datetime_axis_is_rejected():
    """A number is days; on a datetime axis pandas would read it as nanoseconds."""
    primary = _field(1.0)
    with pytest.raises(TypeError, match="np.timedelta64"):
        align_fallback_time(
            _field(2.0), primary["time"], method="nearest", tolerance=0.5
        )


def test_coverage_counts_points_where_every_variable_has_data():
    primary = _sf_dataset(1.0, hole_slice=slice(3, None))
    primary["swrad"][:, :, 2] = np.nan  # swrad covers one column less than Tair
    out = blend_surface_forcing(primary, _sf_dataset(2.0), grid=_grid_ds())
    assert out.attrs["primary_coverage_fraction"] == f"{2 / NX:.4f}"


def test_hard_edge_warns_about_a_variable_empty_in_the_first_record(caplog):
    """The coverage must not read as "no data" because one variable starts empty."""
    primary = _sf_dataset(1.0, hole_slice=slice(3, None))
    primary["Tair"][0] = np.nan  # the first variable, in the first record only

    with caplog.at_level(logging.WARNING):
        out = blend_surface_forcing(primary, _sf_dataset(2.0), grid=_grid_ds())

    assert "no data at the first time step for ['Tair']" in caplog.text
    assert "the result is the fallback alone" not in caplog.text
    assert out.attrs["primary_coverage_fraction"] == f"{3 / NX:.4f}"
    assert (out["Tair"].isel(time=1, xi_rho=slice(0, 3)) == 1.0).all()
