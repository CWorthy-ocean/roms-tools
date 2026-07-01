"""Unit tests for the processing-method registry, enums, and RegridConfig.

These pin the refactor's invariants: the registry is the single source of truth
the derived constants come from, the locked ``creep_fill`` decision, the StrEnum
string behavior the YAML round-trip relies on, and the resolved state / preserved
error contract of :class:`RegridConfig`.
"""

import logging

import pytest
import yaml
from pydantic import ValidationError

from roms_tools.processing_methods import (
    BGC_INTERPOLATION_METHODS,
    EXTRAP_METHODS,
    METHOD_META,
    PREFILL_ALLOWED_KWARGS,
    REGRID_METHODS,
    XESMF_PREFILL_METHODS,
    BgcInterpMethod,
    ExtrapMethod,
    PrefillMethod,
    RegridConfig,
    RegridEngine,
    resolve_bgc_interp_method,
)


class TestRegistryIsSingleSourceOfTruth:
    """The previously-scattered constants are now derived from ``METHOD_META``."""

    def test_prefill_allowed_kwargs_matches_registry(self):
        assert PREFILL_ALLOWED_KWARGS == {
            str(m): spec.allowed_kwargs for m, spec in METHOD_META.items()
        }

    def test_xesmf_methods_are_the_requires_xesmf_methods(self):
        assert XESMF_PREFILL_METHODS == frozenset(
            str(m) for m, spec in METHOD_META.items() if spec.requires_xesmf
        )

    def test_extrap_methods_match_enum(self):
        assert EXTRAP_METHODS == frozenset(str(m) for m in ExtrapMethod)

    def test_regrid_methods_match_enum(self):
        assert REGRID_METHODS == frozenset(str(m) for m in RegridEngine)

    def test_bgc_methods_match_enum(self):
        assert BGC_INTERPOLATION_METHODS == tuple(str(m) for m in BgcInterpMethod)

    def test_derived_values_unchanged_from_pre_refactor(self):
        # Pin the exact pre-refactor values so a registry edit is a conscious change.
        assert dict(PREFILL_ALLOWED_KWARGS) == {
            "2d_lateral_fill": frozenset(),
            "nearest_neighbor": frozenset(),
            "nearest_s2d": frozenset(),
            "inverse_dist": frozenset({"num_src_pnts", "dist_exponent"}),
            "creep_fill": frozenset({"num_levels"}),
        }
        assert XESMF_PREFILL_METHODS == {"inverse_dist", "nearest_s2d", "creep_fill"}
        assert EXTRAP_METHODS == {"inverse_dist", "nearest_s2d", "creep_fill"}
        assert REGRID_METHODS == {"auto", "xesmf", "scipy"}
        assert BGC_INTERPOLATION_METHODS == ("depth", "density", "density_mld")


class TestCreepFill:
    """``creep_fill`` is valid as both a prefill and an extrapolation method.

    It is a (near-)passthrough that is not in released xESMF yet but is wired up
    for once a supporting xESMF is installed, so it requires xESMF.
    """

    def test_creep_fill_is_a_prefill_method(self):
        assert "creep_fill" in {str(m) for m in PrefillMethod}
        assert PrefillMethod("creep_fill") is PrefillMethod.creep_fill

    def test_creep_fill_is_an_extrap_method(self):
        assert ExtrapMethod("creep_fill") is ExtrapMethod.creep_fill

    def test_regrid_config_accepts_creep_fill_prefill(self):
        cfg = RegridConfig(prefill="creep_fill", xesmf_available=True)
        assert cfg.prefill is PrefillMethod.creep_fill

    def test_creep_fill_prefill_requires_xesmf(self):
        with pytest.raises(ImportError, match="xESMF"):
            RegridConfig(prefill="creep_fill", xesmf_available=False)

    def test_creep_fill_prefill_accepts_num_levels(self):
        cfg = RegridConfig(
            prefill="creep_fill",
            prefill_kwargs={"num_levels": 50},
            xesmf_available=True,
        )
        assert cfg.prefill_kwargs == {"num_levels": 50}

    def test_regrid_config_accepts_creep_fill_extrap(self):
        cfg = RegridConfig(extrap_method="creep_fill", xesmf_available=True)
        assert cfg.extrap_method is ExtrapMethod.creep_fill


class TestStrEnumStringBehavior:
    """The YAML round-trip relies on ``str(member)`` being the bare value, and on
    StrEnum members serializing as plain strings under the project's dumper.
    """

    @pytest.mark.parametrize(
        "member,expected",
        [
            (PrefillMethod.inverse_dist, "inverse_dist"),
            (PrefillMethod.lateral_fill_2d, "2d_lateral_fill"),
            (ExtrapMethod.creep_fill, "creep_fill"),
            (RegridEngine.auto, "auto"),
            (BgcInterpMethod.density_mld, "density_mld"),
        ],
    )
    def test_str_is_bare_value(self, member, expected):
        assert str(member) == expected
        assert member == expected  # StrEnum compares equal to its string value

    def test_member_yaml_dumps_as_plain_string(self):
        # roms_tools.setup.utils registers a StrEnum->str representer on import.
        import roms_tools.setup.utils  # noqa: F401

        class _Dumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True

        out = yaml.dump(
            {"prefill": PrefillMethod.inverse_dist}, Dumper=_Dumper, sort_keys=False
        )
        assert out.strip() == "prefill: inverse_dist"
        assert yaml.safe_load(out) == {"prefill": "inverse_dist"}


class TestRegridConfigResolvedState:
    def test_default_path_xesmf_extrap_active(self):
        cfg = RegridConfig(xesmf_available=True)
        assert cfg.use_xesmf is True
        assert cfg.effective_extrap is ExtrapMethod.inverse_dist
        assert cfg.extrap_is_active is True
        assert cfg.regrid_extrap_method == "inverse_dist"  # plain str for xESMF
        assert cfg.regrid_extrap_kwargs is None

    def test_scipy_engine_extrap_inactive(self):
        cfg = RegridConfig(regrid_engine="scipy", xesmf_available=True)
        assert cfg.use_xesmf is False
        assert cfg.extrap_is_active is False

    def test_auto_follows_availability(self):
        assert RegridConfig(xesmf_available=False).use_xesmf is False
        assert RegridConfig(xesmf_available=True).use_xesmf is True

    def test_prefill_set_disables_extrap(self):
        cfg = RegridConfig(
            prefill="inverse_dist",
            extrap_method="nearest_s2d",
            xesmf_available=True,
        )
        assert cfg.extrap_is_active is False
        assert cfg.regrid_extrap_method is None
        assert cfg.regrid_extrap_kwargs is None
        assert cfg.user_set_extrap is True

    def test_user_set_extrap_distinction(self):
        assert RegridConfig(xesmf_available=True).user_set_extrap is False
        assert (
            RegridConfig(
                extrap_method="nearest_s2d", xesmf_available=True
            ).user_set_extrap
            is True
        )
        # Preserved behavior: a non-None extrap_kwargs (even {}) counts as user-set.
        assert (
            RegridConfig(extrap_kwargs={}, xesmf_available=True).user_set_extrap is True
        )

    def test_string_inputs_coerced_to_enums(self):
        cfg = RegridConfig(
            prefill="inverse_dist", regrid_engine="scipy", xesmf_available=True
        )
        assert isinstance(cfg.prefill, PrefillMethod)
        assert isinstance(cfg.regrid_engine, RegridEngine)


class TestRegridConfigErrorContract:
    """Error types/messages are preserved through the pydantic model."""

    def test_invalid_regrid_method_reports_public_name(self):
        with pytest.raises(ValueError, match="regrid_method"):
            RegridConfig(regrid_engine="bogus", xesmf_available=True)

    def test_invalid_extrap_method(self):
        with pytest.raises(ValueError, match="extrap_method"):
            RegridConfig(extrap_method="bogus", xesmf_available=True)

    def test_invalid_extrap_kwargs(self):
        with pytest.raises(ValueError, match="extrap_kwargs"):
            RegridConfig(
                extrap_method="inverse_dist",
                extrap_kwargs={"bogus": 1},
                xesmf_available=True,
            )

    def test_validation_error_is_value_error(self):
        # ValidationError subclasses ValueError, so pytest.raises(ValueError) works.
        assert issubclass(ValidationError, ValueError)

    def test_xesmf_prefill_without_xesmf_raises_importerror(self):
        with pytest.raises(ImportError, match="xESMF"):
            RegridConfig(prefill="inverse_dist", xesmf_available=False)

    def test_xesmf_engine_without_xesmf_raises_importerror(self):
        with pytest.raises(ImportError, match="xESMF"):
            RegridConfig(regrid_engine="xesmf", xesmf_available=False)


class TestFromOptions:
    """`RegridConfig.from_options` owns the deprecated-flag mapping."""

    def _opts(self, **over):
        base = dict(
            prefill=None,
            prefill_kwargs=None,
            regrid_method=None,
            extrap_method=None,
            extrap_kwargs=None,
            xesmf_available=True,
        )
        base.update(over)
        return base

    def test_plain_options_passthrough(self):
        cfg = RegridConfig.from_options(**self._opts(prefill="nearest_neighbor"))
        assert cfg.prefill is PrefillMethod.nearest_neighbor

    def test_deprecated_true_maps_to_2d_lateral_fill(self):
        with pytest.warns(DeprecationWarning):
            cfg = RegridConfig.from_options(**self._opts(apply_2d_horizontal_fill=True))
        assert cfg.prefill is PrefillMethod.lateral_fill_2d

    def test_deprecated_false_maps_to_no_prefill(self):
        with pytest.warns(DeprecationWarning):
            cfg = RegridConfig.from_options(
                **self._opts(apply_2d_horizontal_fill=False)
            )
        assert cfg.prefill is None

    def test_deprecated_conflicts_with_explicit_prefill(self):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match="not both"):
                RegridConfig.from_options(
                    **self._opts(prefill="inverse_dist", apply_2d_horizontal_fill=True)
                )

    def test_regrid_method_none_means_auto(self):
        cfg = RegridConfig.from_options(**self._opts(regrid_method=None))
        assert cfg.regrid_engine is RegridEngine.auto


class TestIgnoredExtrapNotice:
    """The "extrap ignored" decision is logged at config build time, not returned."""

    def test_silent_when_extrap_not_user_set(self, caplog):
        with caplog.at_level(logging.INFO):
            RegridConfig(xesmf_available=True)
        assert "ignored" not in caplog.text

    def test_silent_when_extrap_is_honored(self, caplog):
        with caplog.at_level(logging.INFO):
            RegridConfig(extrap_method="nearest_s2d", xesmf_available=True)
        assert "ignored" not in caplog.text

    def test_logs_when_prefill_set(self, caplog):
        with caplog.at_level(logging.INFO):
            RegridConfig(
                prefill="inverse_dist",
                extrap_method="nearest_s2d",
                xesmf_available=True,
            )
        assert "ignored because prefill" in caplog.text

    def test_logs_when_scipy_engine(self, caplog):
        with caplog.at_level(logging.INFO):
            RegridConfig(
                regrid_engine="scipy",
                extrap_method="nearest_s2d",
                xesmf_available=True,
            )
        assert "scipy regrid" in caplog.text


class TestResolveBgcInterpMethod:
    def test_depth_passes_through(self):
        assert (
            resolve_bgc_interp_method(
                "depth", has_physics_forcing=False, has_source_ts=False
            )
            is BgcInterpMethod.depth
        )

    def test_density_resolves_when_inputs_present(self, caplog):
        with caplog.at_level(logging.INFO):
            method = resolve_bgc_interp_method(
                "density", has_physics_forcing=True, has_source_ts=True
            )
        assert method is BgcInterpMethod.density
        assert caplog.text == "" or "falling back" not in caplog.text

    def test_falls_back_without_physics_forcing(self, caplog):
        with caplog.at_level(logging.INFO):
            method = resolve_bgc_interp_method(
                "density_mld",
                has_physics_forcing=False,
                has_source_ts=True,
                where="south boundary",
            )
        assert method is BgcInterpMethod.depth
        assert "no physics_forcing provided" in caplog.text
        assert "south boundary" in caplog.text

    def test_falls_back_without_source_ts(self, caplog):
        with caplog.at_level(logging.INFO):
            method = resolve_bgc_interp_method(
                "density", has_physics_forcing=True, has_source_ts=False
            )
        assert method is BgcInterpMethod.depth
        assert "temperature/salinity" in caplog.text

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            resolve_bgc_interp_method(
                "bogus", has_physics_forcing=True, has_source_ts=True
            )
