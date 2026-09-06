"""Tests for the salinity-band merge (:mod:`roms_tools.setup.salinity_merge`)."""

import numpy as np
import pytest
import xarray as xr

from roms_tools.setup.salinity_merge import (
    align_partner_time,
    apply_salinity_based_merge,
    raised_cosine_weight,
    resolve_merge_spec,
)


class _FakeSource:
    """Minimal stand-in for a bgc-only forcing object: a source dict and a dataset."""

    def __init__(self, name, data):
        self.source = {"name": name}
        self.ds = xr.Dataset(data)
        self.variable_info_bgc = {k: {} for k in data}


def _salinity(values):
    return xr.DataArray(np.asarray(values, dtype=float), dims="x")


# ---------------------------------------------------------------- the weight


def test_weight_is_zero_below_and_one_above_the_band():
    w = raised_cosine_weight(_salinity([20.0, 31.0, 34.0, 36.0]), 31.0, 34.0).values
    assert w[0] == 0.0 and w[1] == 0.0  # at or below `low`: partner outright
    assert w[2] == 1.0 and w[3] == 1.0  # at or above `high`: primary outright


def test_weight_is_one_half_at_the_band_centre():
    w = raised_cosine_weight(_salinity([32.5]), 31.0, 34.0).values
    assert w[0] == pytest.approx(0.5)


def test_weight_is_monotonic_and_bounded():
    s = np.linspace(28.0, 37.0, 400)
    w = raised_cosine_weight(_salinity(s), 31.0, 34.0).values
    assert np.all((w >= 0.0) & (w <= 1.0))
    assert np.all(np.diff(w) >= -1e-15)


def test_weight_has_a_continuous_derivative_at_the_band_edges():
    """The reason for a raised cosine rather than a linear ramp.

    A linear ramp's slope jumps from 0 to 1/(high-low) at each edge; this one's slope
    goes to zero there, so the blended field has no kink in its gradient.
    """
    eps = 1e-4
    for edge in (31.0, 34.0):
        s = np.array([edge - eps, edge, edge + eps])
        w = raised_cosine_weight(_salinity(s), 31.0, 34.0).values
        slope_in = abs(w[1] - w[0]) / eps
        slope_out = abs(w[2] - w[1]) / eps
        assert min(slope_in, slope_out) < 1e-3


def test_weight_propagates_nan():
    w = raised_cosine_weight(_salinity([np.nan, 35.0]), 31.0, 34.0).values
    assert np.isnan(w[0]) and w[1] == 1.0


def test_weight_rejects_an_inverted_band():
    with pytest.raises(ValueError, match="high > low"):
        raised_cosine_weight(_salinity([33.0]), 34.0, 31.0)


# ---------------------------------------------------------------- the spec

ESPER_ITEM = {"source": {"name": "ESPER"}}


def test_spec_defaults_on_for_esper_when_a_partner_exists():
    spec = resolve_merge_spec(ESPER_ITEM, {"WOA"})
    assert spec is not None
    assert spec["with"] == "WOA"
    assert spec["range"] == (31.0, 34.0)


def test_default_variables_are_the_three_nutrients_only():
    """ALK/DIC are within 3% of WOA in brackish water and are never clipped; O2's
    clipped cells are all at S >= 33, so a salinity band cannot reach them.
    """
    spec = resolve_merge_spec(ESPER_ITEM, {"WOA"})
    assert set(spec["variables"]) == {"NO3", "PO4", "SiO3"}


def test_spec_is_off_for_a_non_esper_source():
    assert resolve_merge_spec({"source": {"name": "UNIFIED"}}, {"WOA"}) is None


def test_spec_is_off_and_warns_when_esper_has_no_partner(caplog):
    """The merge never conjures a source the caller did not configure -- WOA would
    self-download -- so it warns and stands down instead.
    """
    with caplog.at_level("WARNING"):
        assert resolve_merge_spec(ESPER_ITEM, {"UNIFIED"}) is None
    assert "salinity-based merge cannot run" in caplog.text


def test_spec_false_is_off_and_silent(caplog):
    item = {**ESPER_ITEM, "salinity_based_merge": False}
    with caplog.at_level("WARNING"):
        assert resolve_merge_spec(item, set()) is None
    assert caplog.text == ""


def test_explicit_spec_fills_missing_fields_from_the_defaults():
    item = {**ESPER_ITEM, "salinity_based_merge": {"range": (30.0, 35.0)}}
    spec = resolve_merge_spec(item, {"WOA"})
    assert spec["with"] == "WOA"
    assert set(spec["variables"]) == {"NO3", "PO4", "SiO3"}
    assert spec["range"] == (30.0, 35.0)


def test_explicit_spec_overrides_every_default():
    item = {
        **ESPER_ITEM,
        "salinity_based_merge": {
            "with": "GLODAP",
            "variables": ["ALK"],
            "range": (28.0, 33.0),
        },
    }
    spec = resolve_merge_spec(item, {"GLODAP"})
    assert spec == {"with": "GLODAP", "variables": ["ALK"], "range": (28.0, 33.0)}


def test_spec_rejects_unknown_keys_and_bad_bands():
    with pytest.raises(ValueError, match="unknown key"):
        resolve_merge_spec(
            {**ESPER_ITEM, "salinity_based_merge": {"rnage": (31, 34)}}, {"WOA"}
        )
    with pytest.raises(ValueError, match="high > low"):
        resolve_merge_spec(
            {**ESPER_ITEM, "salinity_based_merge": {"range": (34, 31)}}, {"WOA"}
        )


def test_spec_rejects_a_non_dict():
    with pytest.raises(ValueError, match="must be a dict or False"):
        resolve_merge_spec({**ESPER_ITEM, "salinity_based_merge": "yes"}, {"WOA"})


# ---------------------------------------------------------------- the blend


def _pair(salinity_values):
    n = len(salinity_values)
    esper = _FakeSource(
        "ESPER", {"NO3": ("x", np.full(n, 10.0)), "ALK": ("x", np.full(n, 2300.0))}
    )
    woa = _FakeSource("WOA", {"NO3": ("x", np.zeros(n))})
    items = [
        {
            "source": {"name": "ESPER"},
            "salinity_based_merge": {
                "with": "WOA",
                "variables": ["NO3"],
                "range": (31.0, 34.0),
            },
        },
        {"source": {"name": "WOA"}},
    ]
    salt = _salinity(salinity_values)
    return [esper, woa], items, (lambda _n, s=salt: s)


def test_blend_takes_partner_below_and_primary_above():
    objs, items, salt_for = _pair([20.0, 34.0, 36.0])
    apply_salinity_based_merge(objs, items, salt_for, verbose=False)
    got = objs[0].ds["NO3"].values
    assert got[0] == 0.0  # wholly WOA
    assert got[1] == 10.0 and got[2] == 10.0  # wholly ESPER


def test_blend_is_the_weighted_mean_inside_the_band():
    objs, items, salt_for = _pair([32.5])
    apply_salinity_based_merge(objs, items, salt_for, verbose=False)
    assert objs[0].ds["NO3"].values[0] == pytest.approx(5.0)


def test_blend_leaves_unlisted_variables_untouched():
    objs, items, salt_for = _pair([20.0])
    apply_salinity_based_merge(objs, items, salt_for, verbose=False)
    assert objs[0].ds["ALK"].values[0] == 2300.0


def test_blend_removes_the_overlap_from_the_partner():
    """process_bgc_fields and merge() both assume one source per variable."""
    objs, items, salt_for = _pair([32.0])
    apply_salinity_based_merge(objs, items, salt_for, verbose=False)
    assert "NO3" not in objs[1].ds
    assert "NO3" not in objs[1].variable_info_bgc


def test_blend_records_its_provenance():
    objs, items, salt_for = _pair([32.0])
    apply_salinity_based_merge(objs, items, salt_for, verbose=False)
    attrs = objs[0].ds["NO3"].attrs
    assert attrs["merge_partner"] == "WOA"
    assert attrs["merge_band_psu"] == "31-34"
    assert "not an optimal combination" in attrs["merge_weight"]


def test_blend_handles_per_direction_boundary_names():
    """Boundary forcing names variables NO3_south etc.; one spec covers them all."""
    esper = _FakeSource(
        "ESPER",
        {"NO3_south": ("x", np.full(3, 10.0)), "NO3_west": ("x", np.full(3, 10.0))},
    )
    woa = _FakeSource(
        "WOA", {"NO3_south": ("x", np.zeros(3)), "NO3_west": ("x", np.zeros(3))}
    )
    items = [
        {
            "source": {"name": "ESPER"},
            "salinity_based_merge": {"with": "WOA", "variables": ["NO3"]},
        },
        {"source": {"name": "WOA"}},
    ]
    salt = _salinity([20.0, 32.5, 36.0])
    apply_salinity_based_merge([esper, woa], items, lambda _n, s=salt: s, verbose=False)
    for name in ("NO3_south", "NO3_west"):
        got = esper.ds[name].values
        assert got[0] == 0.0
        assert got[1] == pytest.approx(5.0)
        assert got[2] == 10.0
        assert name not in woa.ds


def test_blend_errors_when_the_partner_is_missing():
    objs, items, salt_for = _pair([32.0])
    items[0]["salinity_based_merge"]["with"] = "GLODAP"
    with pytest.raises(ValueError, match="not among the bgc_sources"):
        apply_salinity_based_merge(objs, items, salt_for, verbose=False)


def test_blend_errors_when_the_partner_lacks_the_variable():
    objs, items, salt_for = _pair([32.0])
    objs[1].ds = objs[1].ds.drop_vars("NO3")
    with pytest.raises(ValueError, match="Add it to that source's use_vars"):
        apply_salinity_based_merge(objs, items, salt_for, verbose=False)


def test_blend_errors_when_the_variable_is_not_in_the_primary():
    objs, items, salt_for = _pair([32.0])
    items[0]["salinity_based_merge"]["variables"] = ["SiO3"]
    with pytest.raises(ValueError, match="not in the 'ESPER' source"):
        apply_salinity_based_merge(objs, items, salt_for, verbose=False)


# ---------------------------------------------------------------- the default, end to end


def _default_pair(salinity_values):
    """An ESPER source beside a WOA source, with no merge spec written anywhere."""
    n = len(salinity_values)
    esper = _FakeSource(
        "ESPER",
        {
            v: ("x", np.full(n, 10.0))
            for v in ("NO3", "PO4", "SiO3", "ALK", "DIC", "O2")
        },
    )
    woa = _FakeSource(
        "WOA", {v: ("x", np.zeros(n)) for v in ("NO3", "PO4", "SiO3", "O2")}
    )
    items = [{"source": {"name": "ESPER"}}, {"source": {"name": "WOA"}}]
    salt = _salinity(salinity_values)
    return [esper, woa], items, (lambda _n, s=salt: s)


def test_default_merges_the_nutrients_with_no_spec_written():
    objs, items, salt_for = _default_pair([20.0])
    apply_salinity_based_merge(objs, items, salt_for, verbose=False)
    for v in ("NO3", "PO4", "SiO3"):
        assert objs[0].ds[v].values[0] == 0.0, f"{v} was not merged by default"
        assert v not in objs[1].ds


def test_default_leaves_alk_dic_and_o2_alone():
    """The three ESPER gets right in brackish water; O2's clipped cells are all at
    S >= 33, where a salinity band has no reach.
    """
    objs, items, salt_for = _default_pair([20.0])
    apply_salinity_based_merge(objs, items, salt_for, verbose=False)
    for v in ("ALK", "DIC", "O2"):
        assert objs[0].ds[v].values[0] == 10.0, f"{v} should not be merged by default"
    assert "O2" in objs[1].ds, "O2 should stay with the partner, unmerged"


def test_default_is_off_without_a_partner_and_changes_nothing(caplog):
    esper = _FakeSource("ESPER", {"NO3": ("x", np.full(2, 10.0))})
    unified = _FakeSource("UNIFIED", {"Fe": ("x", np.zeros(2))})
    items = [{"source": {"name": "ESPER"}}, {"source": {"name": "UNIFIED"}}]
    salt = _salinity([20.0, 36.0])
    with caplog.at_level("WARNING"):
        apply_salinity_based_merge(
            [esper, unified], items, lambda _n, s=salt: s, verbose=False
        )
    assert np.all(esper.ds["NO3"].values == 10.0)
    assert "salinity-based merge cannot run" in caplog.text


def test_default_skips_a_variable_the_partner_does_not_carry():
    """A defaulted spec is permissive; an explicit one is not (see the error tests)."""
    objs, items, salt_for = _default_pair([20.0])
    objs[1].ds = objs[1].ds.drop_vars("SiO3")
    apply_salinity_based_merge(objs, items, salt_for, verbose=False)
    assert objs[0].ds["SiO3"].values[0] == 10.0  # untouched, no error
    assert objs[0].ds["NO3"].values[0] == 0.0  # the others still merged


def test_default_can_be_declined():
    objs, items, salt_for = _default_pair([20.0])
    items[0]["salinity_based_merge"] = False
    apply_salinity_based_merge(objs, items, salt_for, verbose=False)
    assert objs[0].ds["NO3"].values[0] == 10.0
    assert "NO3" in objs[1].ds


# ---------------------------------------------------------------- time alignment


def _indexed(values, times, *, cycle=None, dim="bry_time"):
    da = xr.DataArray(np.asarray(values, float), dims=dim, coords={dim: list(times)})
    if cycle is not None:
        da[dim].attrs["cycle_length"] = cycle
    return da


CLIM_DAYS = [15.0, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]


def test_identical_axes_pass_through_untouched():
    primary = _indexed([1.0, 2.0], [3653.0, 3654.0])
    partner = _indexed([9.0, 8.0], [3653.0, 3654.0])
    assert align_partner_time(partner, primary) is partner


def test_climatology_is_cycled_onto_the_primary_axis():
    """March 8 2010 falls 22/29 of the way from the mid-Feb to the mid-Mar record."""
    primary = _indexed([0.0], [3719.0])  # 2010-03-08
    monthly = np.arange(12, dtype=float)  # Feb = 1, Mar = 2
    partner = _indexed(monthly, CLIM_DAYS, cycle=365.25)
    got = align_partner_time(partner, primary)
    day_of_year = 3719.0 % 365.25  # 66.5
    f = (day_of_year - 45.0) / (74.0 - 45.0)
    assert got.values[0] == pytest.approx(1.0 + f * (2.0 - 1.0))
    # and it now shares the primary's coordinate, so the arithmetic aligns
    assert got["bry_time"].values.tolist() == [3719.0]


def test_climatology_wraps_december_into_january():
    """Day 3 of the year sits between the mid-December and mid-January records."""
    primary = _indexed([0.0], [3652.0 + 3.0])
    partner = _indexed([10.0] + [0.0] * 10 + [20.0], CLIM_DAYS, cycle=365.25)
    got = float(align_partner_time(partner, primary).values[0])
    assert 10.0 < got < 20.0, "did not interpolate across the year boundary"


def test_mismatched_axes_without_a_cycle_raise_rather_than_silently_emptying():
    """Xarray would outer-join these to an all-NaN result of length zero."""
    primary = _indexed([1.0, 2.0], [3653.0, 3654.0])
    partner = _indexed(np.zeros(12), CLIM_DAYS)  # no cycle_length
    with pytest.raises(ValueError, match="no 'cycle_length'"):
        align_partner_time(partner, primary)


def test_merge_with_a_climatological_partner_produces_finite_values():
    """End to end: the trap this alignment exists to prevent."""
    times = [3653.0, 3654.0, 3655.0]
    esper = _FakeSource("ESPER", {})
    esper.ds = xr.Dataset({"NO3_south": _indexed([10.0] * 3, times)})
    esper.variable_info_bgc = {"NO3_south": {}}
    woa = _FakeSource("WOA", {})
    woa.ds = xr.Dataset({"NO3_south": _indexed(np.zeros(12), CLIM_DAYS, cycle=365.25)})
    woa.variable_info_bgc = {"NO3_south": {}}
    items = [{"source": {"name": "ESPER"}}, {"source": {"name": "WOA"}}]
    salt = _indexed([20.0, 32.0, 36.0], times)  # 100% WOA, 25/75, 100% ESPER
    apply_salinity_based_merge([esper, woa], items, lambda _n, s=salt: s, verbose=False)
    got = esper.ds["NO3_south"].values
    assert np.all(np.isfinite(got)), "climatology partner produced NaNs"
    assert got[0] == pytest.approx(0.0)
    assert got[1] == pytest.approx(2.5)
    assert got[2] == pytest.approx(10.0)
