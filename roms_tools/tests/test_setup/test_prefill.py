"""Unit tests for the shared prefill helpers (no I/O, no source data)."""

import pytest

from roms_tools.regrid import _xesmf_extrap_kwargs
from roms_tools.setup.utils import (
    PREFILL_ALLOWED_KWARGS,
    XESMF_PREFILL_METHODS,
    resolve_regrid_engine,
    validate_extrap,
    validate_prefill,
)

# A representative "BoundaryForcing-like" allowed set.
ALLOWED = frozenset(
    {None, "2d_lateral_fill", "inverse_dist", "nearest_s2d", "nearest_neighbor"}
)


class TestXesmfExtrapKwargs:
    def test_none_method_returns_empty(self):
        assert _xesmf_extrap_kwargs(None, {"num_levels": 5}) == {}

    def test_inverse_dist_maps_keys(self):
        out = _xesmf_extrap_kwargs(
            "inverse_dist", {"num_src_pnts": 4, "dist_exponent": 1.5}
        )
        assert out == {"extrap_num_src_pnts": 4, "extrap_dist_exponent": 1.5}

    def test_inverse_dist_skips_none_values(self):
        out = _xesmf_extrap_kwargs(
            "inverse_dist", {"num_src_pnts": None, "dist_exponent": None}
        )
        assert out == {}

    def test_creep_fill_maps_num_levels(self):
        assert _xesmf_extrap_kwargs("creep_fill", {"num_levels": 50}) == {
            "extrap_num_levels": 50
        }

    def test_nearest_s2d_takes_no_kwargs(self):
        assert _xesmf_extrap_kwargs("nearest_s2d", {"num_src_pnts": 4}) == {}

    def test_empty_kwargs(self):
        assert _xesmf_extrap_kwargs("inverse_dist", None) == {}


class TestValidatePrefill:
    def test_none_is_allowed(self):
        validate_prefill(None, None, ALLOWED, xesmf_available=True)

    def test_none_with_kwargs_raises(self):
        with pytest.raises(ValueError, match="prefill is None"):
            validate_prefill(None, {"num_src_pnts": 4}, ALLOWED, xesmf_available=True)

    def test_unsupported_value_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            validate_prefill("bogus", None, ALLOWED, xesmf_available=True)

    def test_restricted_allowed_set_rejects_new_values(self):
        restricted = frozenset({"2d_lateral_fill"})
        with pytest.raises(ValueError, match="not supported"):
            validate_prefill(
                "inverse_dist", None, restricted, xesmf_available=True
            )
        # the one allowed value passes
        validate_prefill("2d_lateral_fill", None, restricted, xesmf_available=True)

    @pytest.mark.parametrize("method", sorted(XESMF_PREFILL_METHODS))
    def test_xesmf_only_methods_require_xesmf(self, method):
        with pytest.raises(ImportError, match="xESMF"):
            validate_prefill(method, None, ALLOWED | {method}, xesmf_available=False)

    def test_non_xesmf_methods_ok_without_xesmf(self):
        validate_prefill("2d_lateral_fill", None, ALLOWED, xesmf_available=False)
        validate_prefill("nearest_neighbor", None, ALLOWED, xesmf_available=False)

    def test_invalid_kwarg_key_raises(self):
        with pytest.raises(ValueError, match="not valid for prefill"):
            validate_prefill(
                "inverse_dist", {"bogus_key": 1}, ALLOWED, xesmf_available=True
            )

    def test_valid_inverse_dist_kwargs_pass(self):
        validate_prefill(
            "inverse_dist",
            {"num_src_pnts": 8, "dist_exponent": 2.0},
            ALLOWED,
            xesmf_available=True,
        )

    def test_methods_that_take_no_kwargs_reject_kwargs(self):
        with pytest.raises(ValueError, match="not valid for prefill"):
            validate_prefill(
                "2d_lateral_fill", {"num_levels": 5}, ALLOWED, xesmf_available=True
            )

    def test_allowed_kwargs_table_matches_methods(self):
        # Every xESMF-only method has an entry in the kwargs table.
        for method in XESMF_PREFILL_METHODS:
            assert method in PREFILL_ALLOWED_KWARGS


class TestResolveRegridEngine:
    @pytest.mark.parametrize("method", [None, "auto"])
    def test_auto_follows_availability(self, method):
        assert resolve_regrid_engine(method, xesmf_available=True) is True
        assert resolve_regrid_engine(method, xesmf_available=False) is False

    def test_scipy_always_false(self):
        assert resolve_regrid_engine("scipy", xesmf_available=True) is False
        assert resolve_regrid_engine("scipy", xesmf_available=False) is False

    def test_xesmf_forces_true_when_available(self):
        assert resolve_regrid_engine("xesmf", xesmf_available=True) is True

    def test_xesmf_raises_when_unavailable(self):
        with pytest.raises(ImportError, match="xESMF"):
            resolve_regrid_engine("xesmf", xesmf_available=False)

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            resolve_regrid_engine("bogus", xesmf_available=True)


class TestValidateExtrap:
    @pytest.mark.parametrize("method", [None, "inverse_dist", "nearest_s2d", "creep_fill"])
    def test_valid_methods(self, method):
        validate_extrap(method, None)

    def test_none_defaults_to_inverse_dist_kwargs(self):
        # extrap_method=None is treated as inverse_dist, which accepts these keys
        validate_extrap(None, {"num_src_pnts": 8, "dist_exponent": 2.0})

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            validate_extrap("bogus", None)

    def test_invalid_kwarg_key_raises(self):
        with pytest.raises(ValueError, match="not valid for extrap_method"):
            validate_extrap("inverse_dist", {"bogus": 1})

    def test_nearest_s2d_rejects_kwargs(self):
        with pytest.raises(ValueError, match="not valid for extrap_method"):
            validate_extrap("nearest_s2d", {"num_src_pnts": 4})
