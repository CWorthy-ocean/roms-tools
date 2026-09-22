"""Unit tests for the BGC model abstraction (BGCModel / BGCMarbl).

These use lightweight synthetic forcing stand-ins (objects exposing ``ds`` and,
for the boundary layout, ``boundaries``) so the completion logic of
:meth:`BGCMarbl.process_bgc_fields` can be exercised deterministically and
without building real forcing datasets.

``process_bgc_fields`` enforces (rather than merely assumes) that variables do
not overlap across sources: it derives tracers from each object's own key
fields (an explicit tracer from one source suppresses derivation of the same
tracer from another), raises if any tracer is present in more than one object
after derivation, fills whatever is still missing across the union into the
first object able to supply a fill template, and warns if anything remains
absent.
"""

import numpy as np
import pytest
import xarray as xr

from roms_tools import BGCMarbl, BGCModel
from roms_tools.setup.utils import build_bgc_companions


class _FakeBoundaryForcing:
    """Minimal BoundaryForcing-like object: per-direction variable suffixes."""

    def __init__(self, ds, boundaries, source_name="fake_bgc", use_vars=None):
        self.ds = ds
        self.boundaries = boundaries
        self.source = {"name": source_name}
        self.use_vars = use_vars


class _FakeInitialConditions:
    """Minimal InitialConditions-like object: bare variable names."""

    def __init__(self, ds, source_name="fake_bgc", use_vars=None):
        self.ds = ds
        self.source = {"name": source_name}
        self.use_vars = use_vars


def _grid_da(value, dims=("s_rho", "xi_rho"), shape=(3, 4)):
    return xr.DataArray(np.full(shape, value, dtype="float32"), dims=dims)


def _boundary_ds(values, direction="south"):
    return xr.Dataset({f"{k}_{direction}": _grid_da(v) for k, v in values.items()})


def _ic_ds(values):
    return xr.Dataset({k: _grid_da(v) for k, v in values.items()})


def test_tracer_and_known_vars():
    m = BGCMarbl()
    assert "CHL" not in m.tracer_vars()
    assert "CHL" in m.known_vars()
    assert m.known_vars() - m.tracer_vars() == frozenset({"CHL"})
    for v in ("PO4", "NO3", "ALK", "spChl", "diatFe", "zooC", "Lig"):
        assert v in m.tracer_vars()


def test_base_model_cannot_be_instantiated():
    """`BGCModel` is an ABC: `process_bgc_fields` is abstract, so instantiating the
    base class fails up front (TypeError) rather than at the first call.
    """
    with pytest.raises(TypeError, match="process_bgc_fields"):
        BGCModel()


class TestBGCModelSubclassContract:
    """`__init_subclass__` + ABC: what a concrete model must declare."""

    def test_concrete_subclass_without_tracers_is_rejected_at_definition(self):
        with pytest.raises(TypeError, match="_TRACER_VARS"):

            class Empty(BGCModel):
                def process_bgc_fields(self, forcings):
                    return forcings

    def test_abstract_intermediate_subclass_may_leave_tracers_empty(self):
        class Intermediate(BGCModel):  # still abstract: no process_bgc_fields
            name = "intermediate"

        with pytest.raises(TypeError):
            Intermediate()

        class Concrete(Intermediate):
            _TRACER_VARS = frozenset({"NO3"})

            def process_bgc_fields(self, forcings):
                return forcings

        assert Concrete().tracer_vars() == frozenset({"NO3"})

    def test_marbl_subclass_override_still_works(self):
        class DoubleLigand(BGCMarbl):
            _FE_TO_LIG = 6.0

        assert DoubleLigand().tracer_vars() == BGCMarbl().tracer_vars()


class TestValidateBgcModel:
    def test_accepts_concrete_class(self):
        from roms_tools.setup.bgc_model import validate_bgc_model

        assert validate_bgc_model(BGCMarbl) is BGCMarbl

    def test_rejects_instance_with_pointer_to_class(self):
        from roms_tools.setup.bgc_model import validate_bgc_model

        with pytest.raises(ValueError, match=r"bgc_model=BGCMarbl`, not"):
            validate_bgc_model(BGCMarbl())

    def test_rejects_unrelated_class_and_non_class(self):
        from roms_tools.setup.bgc_model import validate_bgc_model

        with pytest.raises(ValueError, match="BGCModel subclass"):
            validate_bgc_model(dict)
        with pytest.raises(ValueError, match="BGCModel subclass"):
            validate_bgc_model("BGCMarbl")

    def test_rejects_abstract_base(self):
        from roms_tools.setup.bgc_model import validate_bgc_model

        with pytest.raises(ValueError, match="abstract"):
            validate_bgc_model(BGCModel)

    def test_to_name_validates_too(self):
        from roms_tools.setup.bgc_model import bgc_model_to_name

        with pytest.raises(ValueError, match="not an instance"):
            bgc_model_to_name(BGCMarbl())


def test_chl_expansion_and_drop():
    """CHL is expanded into per-PFT tracers and then dropped."""
    ic = _FakeInitialConditions(_ic_ds({"PO4": 1.0, "ALK": 2300.0, "CHL": 2.0}))
    BGCMarbl().process_bgc_fields(ic)

    assert "CHL" not in ic.ds
    assert np.allclose(ic.ds["spChl"].values, 2.0 * 0.675)
    assert np.allclose(ic.ds["diatChl"].values, 2.0 * 0.0675)


def test_fe_to_lig_and_alt_co2_derivations():
    ic = _FakeInitialConditions(_ic_ds({"Fe": 1.0, "DIC": 2000.0, "ALK": 2300.0}))
    BGCMarbl().process_bgc_fields(ic)
    assert np.allclose(ic.ds["Lig"].values, 3.0)  # Fe * 3
    assert np.allclose(ic.ds["DIC_ALT_CO2"].values, 2000.0)
    assert np.allclose(ic.ds["ALK_ALT_CO2"].values, 2300.0)


def test_source_provided_tracer_not_overwritten_by_derivation():
    """A tracer already present from the source is not clobbered by derivation."""
    ic = _FakeInitialConditions(_ic_ds({"Fe": 1.0, "Lig": 42.0}))
    BGCMarbl().process_bgc_fields(ic)
    assert np.allclose(ic.ds["Lig"].values, 42.0)  # kept, not 3*Fe


def test_missing_tracers_filled_with_ocean_background():
    ic = _FakeInitialConditions(_ic_ds({"Fe": 1.0, "ALK": 2300.0}))
    BGCMarbl().process_bgc_fields(ic)
    # DOCr has no source and is filled with its open-ocean background constant.
    assert "DOCr" in ic.ds
    assert np.allclose(ic.ds["DOCr"].values, BGCMarbl._OCEAN_FILL["DOCr"])
    # A phytoplankton tracer has no CHL source and no ocean background defined,
    # so it falls back to zero.
    assert "spChl" in ic.ds
    assert np.allclose(ic.ds["spChl"].values, 0.0)


def test_ocean_fill_mirrors_main_compute_missing_bgc_variables():
    """The constant fills must match the ``(None, factor)`` entries of the
    ``compute_missing_bgc_variables`` table this class replaced.

    Regression guard: these were briefly taken from ``get_tracer_defaults()``,
    which reads ``river_tracer_defaults.nc`` -- river-mouth concentrations. That
    put DOC at 460.476 mmol/m3 uniformly through the ocean interior instead of
    1e-6, and zeroed the refractory DOM pools.
    """
    assert BGCMarbl._OCEAN_FILL == {
        "NH4": 1e-6,
        "DOC": 1e-6,
        "DON": 1.0,
        "DOP": 0.1,
        "DOCr": 1e-6,
        "DONr": 0.8,
        "DOPr": 0.003,
    }


def test_fill_does_not_use_river_tracer_defaults():
    """No river-mouth value may reach an ocean field."""
    from roms_tools.setup.utils import get_tracer_defaults

    river = get_tracer_defaults()
    ic = _FakeInitialConditions(_ic_ds({"Fe": 1.0, "ALK": 2300.0}))
    BGCMarbl().process_bgc_fields(ic)
    for var in ("DOC", "DON", "DOP", "DONr", "DOPr"):
        assert not np.allclose(ic.ds[var].values, river[var]), (
            f"{var} was filled with its river default ({river[var]}) "
            "instead of its ocean background"
        )
    assert np.allclose(ic.ds["DOC"].values, 1e-6)


def test_unsourced_tracer_without_background_warns(caplog):
    """A tracer set to zero for want of any source is reported, not silent."""
    ic = _FakeInitialConditions(_ic_ds({"Fe": 1.0, "ALK": 2300.0}))
    with caplog.at_level("WARNING"):
        BGCMarbl().process_bgc_fields(ic)
    assert "no ocean background value" in caplog.text
    assert "spChl" in caplog.text


def test_derivation_is_per_object_no_prioritization():
    """Each object derives from its OWN key fields; no cross-object priority.

    CHL in object A → phytoplankton in A; Fe in object B → Lig in B.
    """
    a = _FakeInitialConditions(_ic_ds({"ALK": 2300.0, "CHL": 2.0}), source_name="a")
    b = _FakeInitialConditions(_ic_ds({"Fe": 1.0, "NO3": 24.0}), source_name="b")
    BGCMarbl().process_bgc_fields([a, b])

    assert "spChl" in a.ds and "CHL" not in a.ds
    assert "Lig" in b.ds
    assert np.allclose(b.ds["Lig"].values, 3.0)
    assert np.allclose(b.ds["NO3"].values, 24.0)


def test_explicit_tracer_in_one_source_suppresses_derivation_in_another():
    """An explicit Lig in source B suppresses Fe->Lig derivation in source A.

    Regression: derivation used to be decided per object in isolation, so a
    source that supplied Fe would derive its own Lig even when another source
    already supplied an explicit Lig -- producing an overlap that the old code
    never checked for.
    """
    a = _FakeInitialConditions(_ic_ds({"Fe": 1.0}), source_name="a")
    b = _FakeInitialConditions(_ic_ds({"Lig": 42.0}), source_name="b")
    BGCMarbl().process_bgc_fields([a, b])

    assert "Lig" not in a.ds
    assert np.allclose(b.ds["Lig"].values, 42.0)


def test_missing_filled_into_first_object():
    """Union-missing tracers are filled into the first object only."""
    a = _FakeInitialConditions(_ic_ds({"PO4": 1.0}), source_name="a")
    b = _FakeInitialConditions(_ic_ds({"NO3": 24.0}), source_name="b")
    BGCMarbl().process_bgc_fields([a, b])
    # DOCr is missing from both; it lands in the first object, not the second.
    assert "DOCr" in a.ds
    assert "DOCr" not in b.ds


def test_constant_fill_uses_first_adapter_with_a_template():
    """When the first object has no BGC tracer at all to use as a fill
    template, the constant fill lands in the first object that does, rather
    than being silently dropped.

    Regression: the old code always wrote into ``adapters[0]``, whose own
    ``assign_const`` silently no-opped when that object had no template --
    the fill was reported as done (in ``filled``) but never written anywhere.
    """
    a = _FakeInitialConditions(xr.Dataset(), source_name="a")  # no BGC vars at all
    b = _FakeInitialConditions(_ic_ds({"Fe": 1.0, "ALK": 2300.0}), source_name="b")
    BGCMarbl().process_bgc_fields([a, b])

    assert "DOCr" not in a.ds
    assert "DOCr" in b.ds
    assert np.allclose(b.ds["DOCr"].values, BGCMarbl._OCEAN_FILL["DOCr"])


def test_boundary_layout_only_touches_active_directions():
    ds = xr.merge(
        [
            _boundary_ds({"ALK": 2300.0, "CHL": 2.0}, "south"),
            _boundary_ds({"ALK": 2300.0, "CHL": 2.0}, "north"),
        ]
    )
    bf = _FakeBoundaryForcing(ds, {"south": True, "north": False})
    BGCMarbl().process_bgc_fields(bf)

    assert "spChl_south" in bf.ds
    assert "CHL_south" not in bf.ds
    # north is inactive: untouched (still has raw CHL, no derived tracers)
    assert "spChl_north" not in bf.ds
    assert "CHL_north" in bf.ds


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least one forcing object"):
        BGCMarbl().process_bgc_fields([])


class TestOverlapDetection:
    """``process_bgc_fields`` must raise, not silently pick a winner, when more
    than one source supplies the same tracer (after derivation).
    """

    def test_overlap_raises_for_ic_layout(self):
        a = _FakeInitialConditions(_ic_ds({"DIC": 2000.0}), source_name="source_a")
        b = _FakeInitialConditions(_ic_ds({"DIC": 1900.0}), source_name="source_b")
        with pytest.raises(ValueError, match="DIC") as excinfo:
            BGCMarbl().process_bgc_fields([a, b])
        assert "source_a" in str(excinfo.value)
        assert "source_b" in str(excinfo.value)

    def test_overlap_raises_for_boundary_layout(self):
        ds = xr.merge(
            [
                _boundary_ds({"DIC": 2000.0}, "south"),
            ]
        )
        a = _FakeBoundaryForcing(
            ds.copy(deep=True), {"south": True}, source_name="source_a"
        )
        b = _FakeBoundaryForcing(
            ds.copy(deep=True), {"south": True}, source_name="source_b"
        )
        with pytest.raises(ValueError, match="DIC") as excinfo:
            BGCMarbl().process_bgc_fields([a, b])
        assert "source_a" in str(excinfo.value)
        assert "source_b" in str(excinfo.value)

    def test_two_sources_both_with_dic_raise(self):
        a = _FakeInitialConditions(_ic_ds({"DIC": 2000.0}), source_name="a")
        b = _FakeInitialConditions(_ic_ds({"DIC": 1950.0}), source_name="b")
        with pytest.raises(ValueError, match="use_vars"):
            BGCMarbl().process_bgc_fields([a, b])

    def test_overlap_message_includes_use_vars_when_set(self):
        a = _FakeInitialConditions(
            _ic_ds({"DIC": 2000.0}), source_name="a", use_vars=["DIC"]
        )
        b = _FakeInitialConditions(_ic_ds({"DIC": 1950.0}), source_name="b")
        with pytest.raises(ValueError) as excinfo:
            BGCMarbl().process_bgc_fields([a, b])
        assert "use_vars=DIC" in str(excinfo.value)


class TestBuildBgcCompanionsValidation:
    """``build_bgc_companions`` rejects unknown per-item keys before ever
    constructing a source object.
    """

    def test_unknown_item_key_raises(self):
        def _never_call(**_kwargs):
            raise AssertionError(
                "source_cls must not be constructed when an item key is invalid"
            )

        with pytest.raises(ValueError, match="typo_key"):
            build_bgc_companions(
                _never_call,
                grid=object(),
                physics_obj=object(),
                bgc_sources=[{"source": {"name": "constants"}, "typo_key": 1}],
                shared_kwargs={},
            )

    def test_empty_use_vars_list_treated_as_unset(self):
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return object()

        build_bgc_companions(
            _capture,
            grid=object(),
            physics_obj=object(),
            bgc_sources=[{"source": {"name": "constants"}, "use_vars": []}],
            shared_kwargs={},
        )
        assert "use_vars" not in captured


class TestDerivationRulesAreTheSingleSourceOfTruth:
    """The public ``derive_*`` helpers and ``process_bgc_fields`` must agree.

    Regression: both carried independent copies of the same expressions -- the helpers
    as methods, the pipeline as inline lambdas -- so the two could silently diverge and
    only the pipeline copy was ever exercised. Both now read ``derivation_rules()``.
    """

    def test_rules_cover_every_public_helper(self):
        from roms_tools import BGCMarbl

        rules = BGCMarbl.derivation_rules()
        targets = {t for t, _s, _fn in rules}
        assert "Lig" in targets
        assert {"DIC_ALT_CO2", "ALK_ALT_CO2"} <= targets
        assert set(BGCMarbl._CHL_FACTORS) <= targets
        # Every rule is single-source and every source is a real input tracer.
        assert {s for _t, s, _fn in rules} == {"Fe", "DIC", "ALK", "CHL"}
        # No rule derives from a derived tracer (order would then matter).
        assert not ({s for _t, s, _fn in rules} & targets)

    def test_helpers_match_the_rule_table(self):
        import numpy as np
        import xarray as xr

        from roms_tools import BGCMarbl

        src = xr.DataArray(np.linspace(0.1, 2.0, 6), dims=("x",))
        rules = {t: fn for t, _s, fn in BGCMarbl.derivation_rules()}

        xr.testing.assert_identical(
            BGCMarbl.derive_ligand_from_iron(src), rules["Lig"](src)
        )
        alt = BGCMarbl.derive_alt_co2(src, src)
        xr.testing.assert_identical(alt["DIC_ALT_CO2"], rules["DIC_ALT_CO2"](src))
        xr.testing.assert_identical(alt["ALK_ALT_CO2"], rules["ALK_ALT_CO2"](src))

        phyto = BGCMarbl.derive_phytoplankton_from_chl(src)
        assert set(phyto) == set(BGCMarbl._CHL_FACTORS)
        for var, factor in BGCMarbl._CHL_FACTORS.items():
            xr.testing.assert_identical(phyto[var], src * factor)

    def test_helpers_reflect_a_subclass_override(self):
        """A subclass changing a factor must change both paths, not just one."""
        import numpy as np
        import xarray as xr

        from roms_tools import BGCMarbl

        class DoubleLigand(BGCMarbl):
            _FE_TO_LIG = 6.0

        fe = xr.DataArray(np.array([1.0, 2.0]), dims=("x",))
        xr.testing.assert_identical(DoubleLigand.derive_ligand_from_iron(fe), fe * 6.0)
        rule = {t: fn for t, _s, fn in DoubleLigand.derivation_rules()}["Lig"]
        xr.testing.assert_identical(rule(fe), fe * 6.0)
