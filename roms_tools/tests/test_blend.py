"""Tests for the salinity-band blend (:mod:`roms_tools.setup.blend`)."""

import numpy as np
import pytest
import xarray as xr

from roms_tools.setup.blend import (
    apply_salinity_blends,
    parse_blend_spec,
    raised_cosine_weight,
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


def test_spec_absent_is_none():
    assert parse_blend_spec({"source": {"name": "ESPER"}}) is None


def test_spec_requires_a_partner_and_variables():
    with pytest.raises(ValueError, match="'with'"):
        parse_blend_spec({"salinity_blend": {"variables": ["NO3"]}})
    with pytest.raises(ValueError, match="'variables'"):
        parse_blend_spec({"salinity_blend": {"with": "WOA"}})


def test_spec_rejects_unknown_keys_and_bad_bands():
    with pytest.raises(ValueError, match="unknown key"):
        parse_blend_spec(
            {"salinity_blend": {"with": "WOA", "variables": ["NO3"], "rnage": (31, 34)}}
        )
    with pytest.raises(ValueError, match="high > low"):
        parse_blend_spec(
            {"salinity_blend": {"with": "WOA", "variables": ["NO3"], "range": (34, 31)}}
        )


def test_spec_defaults_the_band():
    spec = parse_blend_spec({"salinity_blend": {"with": "WOA", "variables": ["NO3"]}})
    assert spec["range"] == (31.0, 34.0)


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
            "salinity_blend": {
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
    apply_salinity_blends(objs, items, salt_for, verbose=False)
    got = objs[0].ds["NO3"].values
    assert got[0] == 0.0  # wholly WOA
    assert got[1] == 10.0 and got[2] == 10.0  # wholly ESPER


def test_blend_is_the_weighted_mean_inside_the_band():
    objs, items, salt_for = _pair([32.5])
    apply_salinity_blends(objs, items, salt_for, verbose=False)
    assert objs[0].ds["NO3"].values[0] == pytest.approx(5.0)


def test_blend_leaves_unlisted_variables_untouched():
    objs, items, salt_for = _pair([20.0])
    apply_salinity_blends(objs, items, salt_for, verbose=False)
    assert objs[0].ds["ALK"].values[0] == 2300.0


def test_blend_removes_the_overlap_from_the_partner():
    """process_bgc_fields and merge() both assume one source per variable."""
    objs, items, salt_for = _pair([32.0])
    apply_salinity_blends(objs, items, salt_for, verbose=False)
    assert "NO3" not in objs[1].ds
    assert "NO3" not in objs[1].variable_info_bgc


def test_blend_records_its_provenance():
    objs, items, salt_for = _pair([32.0])
    apply_salinity_blends(objs, items, salt_for, verbose=False)
    attrs = objs[0].ds["NO3"].attrs
    assert attrs["blend_partner"] == "WOA"
    assert attrs["blend_band_psu"] == "31-34"
    assert "not an optimal combination" in attrs["blend_weight"]


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
            "salinity_blend": {"with": "WOA", "variables": ["NO3"]},
        },
        {"source": {"name": "WOA"}},
    ]
    salt = _salinity([20.0, 32.5, 36.0])
    apply_salinity_blends([esper, woa], items, lambda _n, s=salt: s, verbose=False)
    for name in ("NO3_south", "NO3_west"):
        got = esper.ds[name].values
        assert got[0] == 0.0
        assert got[1] == pytest.approx(5.0)
        assert got[2] == 10.0
        assert name not in woa.ds


def test_blend_errors_when_the_partner_is_missing():
    objs, items, salt_for = _pair([32.0])
    items[0]["salinity_blend"]["with"] = "GLODAP"
    with pytest.raises(ValueError, match="not among the bgc_sources"):
        apply_salinity_blends(objs, items, salt_for, verbose=False)


def test_blend_errors_when_the_partner_lacks_the_variable():
    objs, items, salt_for = _pair([32.0])
    objs[1].ds = objs[1].ds.drop_vars("NO3")
    with pytest.raises(ValueError, match="Add it to that source's use_vars"):
        apply_salinity_blends(objs, items, salt_for, verbose=False)


def test_blend_errors_when_the_variable_is_not_in_the_primary():
    objs, items, salt_for = _pair([32.0])
    items[0]["salinity_blend"]["variables"] = ["SiO3"]
    with pytest.raises(ValueError, match="not in the 'ESPER' source"):
        apply_salinity_blends(objs, items, salt_for, verbose=False)
