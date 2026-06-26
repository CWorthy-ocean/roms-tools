"""Unit tests for the BGC model abstraction (BGCModel / BGCMarbl).

These use lightweight synthetic forcing stand-ins (objects exposing ``ds`` and,
for the boundary layout, ``boundaries``) so the cross-object completion logic of
:meth:`BGCMarbl.process_bgc_fields` can be exercised deterministically and
without building real forcing datasets.
"""

import numpy as np
import pytest
import xarray as xr

from roms_tools import BGCMarbl, BGCModel


class _FakeBoundaryForcing:
    """Minimal BoundaryForcing-like object: per-direction variable suffixes."""

    def __init__(self, ds, boundaries):
        self.ds = ds
        self.boundaries = boundaries
        self.saved_to = None

    def save(self, filepath):
        self.saved_to = filepath


class _FakeInitialConditions:
    """Minimal InitialConditions-like object: bare variable names."""

    def __init__(self, ds):
        self.ds = ds
        self.saved_to = None

    def save(self, filepath):
        self.saved_to = filepath


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
    # A representative sampling of the MARBL output set.
    for v in ("PO4", "NO3", "ALK", "spChl", "diatFe", "zooC", "Lig"):
        assert v in m.tracer_vars()


def test_base_model_compute_missing_is_abstract():
    with pytest.raises(NotImplementedError):
        BGCModel().process_bgc_fields([])


def test_chl_expansion_and_drop_initial_conditions():
    """CHL is expanded into per-PFT tracers and then dropped."""
    ic = _FakeInitialConditions(_ic_ds({"PO4": 1.0, "ALK": 2300.0, "CHL": 2.0}))
    BGCMarbl().process_bgc_fields(ic)

    assert "CHL" not in ic.ds
    # spChl = CHL * 0.675
    assert np.allclose(ic.ds["spChl"].values, 2.0 * 0.675)
    # diatChl = CHL * 0.0675
    assert np.allclose(ic.ds["diatChl"].values, 2.0 * 0.0675)


def test_fe_to_lig_and_alt_co2_derivations():
    ic = _FakeInitialConditions(
        _ic_ds({"Fe": 1.0, "DIC": 2000.0, "ALK": 2300.0})
    )
    BGCMarbl().process_bgc_fields(ic)
    assert np.allclose(ic.ds["Lig"].values, 3.0)  # Fe * 3
    assert np.allclose(ic.ds["DIC_ALT_CO2"].values, 2000.0)
    assert np.allclose(ic.ds["ALK_ALT_CO2"].values, 2300.0)


def test_defaults_filled_into_primary():
    ic = _FakeInitialConditions(_ic_ds({"Fe": 1.0, "ALK": 2300.0}))
    BGCMarbl().process_bgc_fields(ic)
    # DOCr has no source and is filled with its constant default.
    assert "DOCr" in ic.ds
    assert np.allclose(ic.ds["DOCr"].values, BGCMarbl()._DEFAULTS["DOCr"])
    # constant field (spatially uniform)
    assert np.allclose(ic.ds["DOCr"].std(), 0.0, atol=1e-10)


def test_collective_union_across_objects():
    """A tracer present in one object is not re-derived/duplicated into another.

    CHL lives in object A; Fe lives in object B.  Phytoplankton tracers are
    written into A (the CHL holder); Lig into B (the Fe holder).  A tracer that
    object B already supplies (NO3) is left untouched everywhere.
    """
    a = _FakeInitialConditions(_ic_ds({"ALK": 2300.0, "CHL": 2.0}))
    b = _FakeInitialConditions(_ic_ds({"Fe": 1.0, "NO3": 24.0, "PO4": 1.0}))

    BGCMarbl().process_bgc_fields([a, b])

    # CHL expansion landed in A, not B.
    assert "spChl" in a.ds and "spChl" not in b.ds
    assert "CHL" not in a.ds
    # Lig derived into B (the Fe holder).
    assert "Lig" in b.ds and "Lig" not in a.ds
    # NO3 was supplied by B and must keep its source value.
    assert np.allclose(b.ds["NO3"].values, 24.0)


def test_boundary_layout_only_touches_active_directions():
    ds = xr.merge(
        [
            _boundary_ds({"ALK": 2300.0, "CHL": 2.0}, "south"),
            _boundary_ds({"ALK": 2300.0, "CHL": 2.0}, "north"),
        ]
    )
    # 'north' is disabled → it must be ignored.
    bf = _FakeBoundaryForcing(ds, {"south": True, "north": False})
    BGCMarbl().process_bgc_fields(bf)

    assert "spChl_south" in bf.ds
    assert "CHL_south" not in bf.ds
    # north was inactive: untouched (still has raw CHL, no derived tracers)
    assert "spChl_north" not in bf.ds
    assert "CHL_north" in bf.ds


def test_save_called_with_filepath(tmp_path):
    a = _FakeInitialConditions(_ic_ds({"Fe": 1.0, "ALK": 2300.0}))
    b = _FakeInitialConditions(_ic_ds({"PO4": 1.0, "NO3": 24.0}))
    p1, p2 = tmp_path / "a.nc", tmp_path / "b.nc"
    BGCMarbl().process_bgc_fields([a, b], filepath=[p1, p2])
    assert a.saved_to == p1
    assert b.saved_to == p2


def test_save_filepath_length_mismatch_raises(tmp_path):
    a = _FakeInitialConditions(_ic_ds({"Fe": 1.0}))
    b = _FakeInitialConditions(_ic_ds({"PO4": 1.0}))
    with pytest.raises(ValueError, match="one path per forcing object"):
        BGCMarbl().process_bgc_fields([a, b], filepath=[tmp_path / "only.nc"])


def test_single_object_save(tmp_path):
    ic = _FakeInitialConditions(_ic_ds({"Fe": 1.0, "ALK": 2300.0}))
    p = tmp_path / "ic.nc"
    result = BGCMarbl().process_bgc_fields(ic, filepath=p)
    assert result is ic
    assert ic.saved_to == p


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least one forcing object"):
        BGCMarbl().process_bgc_fields([])


def test_first_in_list_wins_for_unclaimed_overlap():
    """A tracer in two objects is kept only in the first; dropped from the rest."""
    a = _FakeInitialConditions(_ic_ds({"NO3": 24.0, "ALK": 2300.0}))
    b = _FakeInitialConditions(_ic_ds({"NO3": 99.0, "PO4": 1.0}))
    BGCMarbl().process_bgc_fields([a, b])
    assert "NO3" in a.ds
    assert "NO3" not in b.ds  # dropped from the later object
    assert np.allclose(a.ds["NO3"].values, 24.0)


def test_prefer_overrides_list_order():
    """An (object, [fields]) claim keeps the field there and drops it elsewhere."""
    a = _FakeInitialConditions(_ic_ds({"NO3": 24.0, "ALK": 2300.0}))
    b = _FakeInitialConditions(_ic_ds({"NO3": 99.0, "PO4": 1.0}))
    # Prefer b for NO3 even though a is earlier in the list.
    BGCMarbl().process_bgc_fields([a, (b, ["NO3"])])
    assert "NO3" not in a.ds
    assert "NO3" in b.ds
    assert np.allclose(b.ds["NO3"].values, 99.0)


def test_prefer_conflicting_claims_raise():
    a = _FakeInitialConditions(_ic_ds({"NO3": 1.0}))
    b = _FakeInitialConditions(_ic_ds({"NO3": 2.0}))
    with pytest.raises(ValueError, match="claimed via `prefer` by more than one"):
        BGCMarbl().process_bgc_fields([(a, ["NO3"]), (b, ["NO3"])])


def test_prefer_field_not_present_raises():
    a = _FakeInitialConditions(_ic_ds({"ALK": 2300.0}))
    b = _FakeInitialConditions(_ic_ds({"Fe": 1.0}))
    with pytest.raises(ValueError, match="does not contain 'NO3'"):
        BGCMarbl().process_bgc_fields([(a, ["NO3"]), b])


def test_invalid_entry_raises():
    a = _FakeInitialConditions(_ic_ds({"ALK": 2300.0}))
    with pytest.raises(ValueError, match="forcing object or a"):
        BGCMarbl().process_bgc_fields([a, "not_a_forcing"])


def test_prefer_save_paths_positional(tmp_path):
    """filepath stays positional: one path per entry, including tuple entries."""
    a = _FakeInitialConditions(_ic_ds({"NO3": 24.0, "ALK": 2300.0}))
    b = _FakeInitialConditions(_ic_ds({"NO3": 99.0, "Fe": 1.0}))
    pa, pb = tmp_path / "a.nc", tmp_path / "b.nc"
    BGCMarbl().process_bgc_fields([a, (b, ["NO3"])], filepath=[pa, pb])
    assert a.saved_to == pa and b.saved_to == pb
    assert "NO3" in b.ds and "NO3" not in a.ds


def test_inconsistent_spatial_dims_raise():
    """Objects describing different grids (here: different xi_rho) are rejected."""
    a = _FakeInitialConditions(_ic_ds({"ALK": 2300.0}))  # xi_rho=4
    b = _FakeInitialConditions(
        xr.Dataset({"Fe": _grid_da(1.0, dims=("s_rho", "xi_rho"), shape=(3, 5))})
    )  # xi_rho=5
    with pytest.raises(ValueError, match="Inconsistent spatial dimension 'xi_rho'"):
        BGCMarbl().process_bgc_fields([a, b])


def test_differing_time_axes_allowed():
    """Time axes may differ across objects (ROMS interpolates each file)."""
    a = _FakeInitialConditions(
        xr.Dataset({"CHL": _grid_da(2.0, dims=("time", "s_rho", "xi_rho"), shape=(12, 3, 4))})
    )
    b = _FakeInitialConditions(
        xr.Dataset({"Fe": _grid_da(1.0, dims=("time", "s_rho", "xi_rho"), shape=(2, 3, 4))})
    )
    # Must not raise despite time=12 vs time=2.
    BGCMarbl().process_bgc_fields([a, b])
    assert "spChl" in a.ds and "Lig" in b.ds
