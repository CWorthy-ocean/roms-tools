"""Unit tests for the BGC model abstraction (BGCModel / BGCMarbl).

These use lightweight synthetic forcing stand-ins (objects exposing ``ds`` and,
for the boundary layout, ``boundaries``) so the completion logic of
:meth:`BGCMarbl.process_bgc_fields` can be exercised deterministically and
without building real forcing datasets.

``process_bgc_fields`` makes no prioritization decisions: it derives tracers from
each object's own key fields, fills whatever is still missing across the union
into the *first* object, and warns if anything remains absent.
"""

import numpy as np
import pytest
import xarray as xr

from roms_tools import BGCMarbl, BGCModel
from roms_tools.setup.utils import get_tracer_defaults


class _FakeBoundaryForcing:
    """Minimal BoundaryForcing-like object: per-direction variable suffixes."""

    def __init__(self, ds, boundaries):
        self.ds = ds
        self.boundaries = boundaries
        self.saved_to = None

    def save(self, filepath, serialize_dask=None):
        self.saved_to = filepath


class _FakeInitialConditions:
    """Minimal InitialConditions-like object: bare variable names."""

    def __init__(self, ds):
        self.ds = ds
        self.saved_to = None

    def save(self, filepath, serialize_dask=None):
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
    for v in ("PO4", "NO3", "ALK", "spChl", "diatFe", "zooC", "Lig"):
        assert v in m.tracer_vars()


def test_base_model_process_is_abstract():
    with pytest.raises(NotImplementedError):
        BGCModel().process_bgc_fields([])


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


def test_missing_tracers_filled_with_defaults():
    ic = _FakeInitialConditions(_ic_ds({"Fe": 1.0, "ALK": 2300.0}))
    BGCMarbl().process_bgc_fields(ic)
    defaults = get_tracer_defaults()
    # DOCr has no source and is filled with its MARBL default constant.
    assert "DOCr" in ic.ds
    assert np.allclose(ic.ds["DOCr"].values, defaults["DOCr"])
    # A phytoplankton tracer with no CHL source is also default-filled.
    assert "spChl" in ic.ds
    assert np.allclose(ic.ds["spChl"].values, defaults["spChl"])


def test_derivation_is_per_object_no_prioritization():
    """Each object derives from its OWN key fields; no cross-object priority.

    CHL in object A → phytoplankton in A; Fe in object B → Lig in B.
    """
    a = _FakeInitialConditions(_ic_ds({"ALK": 2300.0, "CHL": 2.0}))
    b = _FakeInitialConditions(_ic_ds({"Fe": 1.0, "NO3": 24.0}))
    BGCMarbl().process_bgc_fields([a, b])

    assert "spChl" in a.ds and "CHL" not in a.ds
    assert "Lig" in b.ds
    assert np.allclose(b.ds["Lig"].values, 3.0)
    assert np.allclose(b.ds["NO3"].values, 24.0)


def test_missing_filled_into_first_object():
    """Union-missing tracers are filled into the first object only."""
    a = _FakeInitialConditions(_ic_ds({"PO4": 1.0}))
    b = _FakeInitialConditions(_ic_ds({"NO3": 24.0}))
    BGCMarbl().process_bgc_fields([a, b])
    # DOCr is missing from both; it lands in the first object, not the second.
    assert "DOCr" in a.ds
    assert "DOCr" not in b.ds


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
