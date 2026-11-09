"""Tests for ``roms_tools.scripts.blend_surface_forcing``.

Synthetic arrays and small datasets only: no source data, no regridding. Everything
the merge can get wrong -- silent index alignment, a coarse/fine shape mismatch,
chunk misalignment, an accidental compute, a NaN left at an ocean point, the
encodings of the written files -- is cheap to pin down here and expensive to
diagnose once it is buried in a full SurfaceForcing run.
"""

import numpy as np
import pytest
import xarray as xr

from roms_tools.scripts.blend_surface_forcing import (
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


# --- dataset-level API -----------------------------------------------------

from roms_tools.scripts.blend_surface_forcing import (  # noqa: E402
    blend_surface_forcing,
    compute_blend_weights,
)


def _sf_dataset(value, *, hole_slice=None, rad_offset_h=-0.5, source="X"):
    """A SurfaceForcing-like Dataset: time in days, rad_time, ROMS-named vars."""
    days = np.arange(NT) / 24.0
    ds = xr.Dataset(
        {
            name: xr.DataArray(
                np.full((NT, NY, NX), float(value)),
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

    from roms_tools.scripts.blend_surface_forcing import save_blended_surface_forcing

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
    assert out.attrs["fallback_correct_radiation"] == "True"
    assert out.attrs["blend_width_km"] == "2.0"
