"""Tests for the ESPER (PyESPER-derived) BGC source.

The estimation tests need both the PyESPER package and its on-disk data
(``Mat_fullgrid/`` + ``NeuralNetworks/``); they are skipped unless a PyESPER directory is
available. Set ``ROMS_TOOLS_PYESPER_PATH`` to point at it (defaults to a local checkout).
The pure input-validation tests run everywhere (they don't import PyESPER).
"""

import copy
import os
from datetime import datetime
from pathlib import Path

import dask
import numpy as np
import pytest
import xarray as xr

from roms_tools import (
    BGCMarbl,
    BoundaryForcingSource,
    Grid,
    InitialConditions,
    InitialConditionsSource,
)
from roms_tools.datasets.download import download_test_data
from roms_tools.setup.esper import (
    ESPER_SUPPORTED_VARS,
    _MAX_POINTS_PER_CHUNK,
    _pyesper_chunk_plan,
    estimate_bgc_fields,
    validate_esper_source,
)

_PYESPER_PATH = os.environ.get(
    "ROMS_TOOLS_PYESPER_PATH", "/Users/blsaenz/Projects/git/PyESPER"
)


def _pyesper_available() -> bool:
    if not Path(_PYESPER_PATH, "Mat_fullgrid").is_dir():
        return False
    import sys

    if _PYESPER_PATH not in sys.path:
        sys.path.insert(0, _PYESPER_PATH)
    try:
        import PyESPER  # noqa: F401
        from PyESPER import nn_xr  # noqa: F401
    except ImportError:
        return False
    return True


needs_pyesper = pytest.mark.skipif(
    not _pyesper_available(),
    reason="PyESPER package and its data directory are required for ESPER tests",
)


def _small_grid() -> Grid:
    return Grid(
        nx=2,
        ny=2,
        size_x=500,
        size_y=1000,
        center_lon=0,
        center_lat=55,
        rot=10,
        N=3,
        theta_s=5.0,
        theta_b=2.0,
        hc=250.0,
    )


# --------------------------------------------------------------------------------------
# Input validation (no PyESPER / data needed)
# --------------------------------------------------------------------------------------
def test_validate_esper_source_requires_path():
    with pytest.raises(ValueError, match="requires a 'path'"):
        validate_esper_source({"name": "ESPER"})


def test_validate_esper_source_bad_method():
    with pytest.raises(ValueError, match="'method' must be one of"):
        validate_esper_source({"name": "ESPER", "path": "/x", "method": "bogus"})


def test_validate_esper_source_bad_equation():
    with pytest.raises(ValueError, match="'equation' must be 8"):
        validate_esper_source({"name": "ESPER", "path": "/x", "equation": 1})


def test_pyesper_chunk_plan_caps_chunk_count_at_production_grid_scale():
    """Regression: PyESPER's *_xr methods reload a neural-network/regression
    model on EVERY dask block, a multi-second fixed cost per call regardless of
    block size. Without a chunk-count cap, whatever fine-grained chunking the
    upstream regrid pipeline left `temp`/`salt` with turns into that many
    redundant reloads -- at production grid scale (e.g. a 4km/100-level domain,
    ~90M points) this ran for over an hour at 2% progress (a many-hour
    projected total) before this fix. `_pyesper_chunk_plan` must collapse a
    heavily over-chunked input down to a small, bounded chunk count.
    """
    import dask.array as da

    # Mirrors a real 4km CCS-scale IC volume (672 x 1344 x 100) arriving with a
    # naive, fine-grained chunking from an upstream regrid pipeline.
    arr = da.random.random((100, 1344, 672), chunks=(1, 50, 50))
    temp = xr.DataArray(arr, dims=("s_rho", "eta_rho", "xi_rho"))

    original_chunk_count = 1
    for c in temp.data.chunks:
        original_chunk_count *= len(c)
    assert original_chunk_count > 1000, (
        "fixture should start heavily over-chunked, to exercise the cap"
    )

    plan = _pyesper_chunk_plan(temp)
    rechunked = temp.chunk(plan)
    new_chunk_count = 1
    for c in rechunked.data.chunks:
        new_chunk_count *= len(c)

    # Bounded, low-tens chunk count -- not hundreds/thousands -- regardless of
    # how finely the input arrived chunked.
    assert new_chunk_count <= 32
    assert new_chunk_count < original_chunk_count
    # No data lost/reshaped -- chunking is purely a dask block-boundary change.
    assert rechunked.shape == temp.shape

    # Each resulting chunk stays in the right ballpark of the point-count
    # target (some slack is expected/fine since whole dims are collapsed).
    max_chunk_points = max(
        s0 * s1 * s2
        for s0 in rechunked.data.chunks[0]
        for s1 in rechunked.data.chunks[1]
        for s2 in rechunked.data.chunks[2]
    )
    assert max_chunk_points <= _MAX_POINTS_PER_CHUNK * 2  # generous slack


def test_pyesper_chunk_plan_noop_for_small_array():
    """An array already under the cap round-trips to a single chunk per dim
    (not needlessly split), and a plain in-memory (non-dask) array is handled
    the same way by `.chunk()` when the caller applies the plan unconditionally.
    """
    small = xr.DataArray(np.zeros((3, 4, 5)), dims=("s_rho", "eta_rho", "xi_rho"))
    plan = _pyesper_chunk_plan(small.chunk())
    assert plan == {"s_rho": -1, "eta_rho": -1, "xi_rho": -1}


def test_ic_esper_missing_path_raises():
    with pytest.raises(ValueError, match="requires a 'path'"):
        InitialConditionsSource(
            grid=_small_grid(),
            ini_time=datetime(2021, 6, 29),
            type="bgc",
            source={"name": "ESPER"},
            physics_forcing=_small_physics_ic(),
            use_dask=False,
        )


def test_bf_esper_requires_physics_forcing():
    with pytest.raises(ValueError, match="requires `physics_forcing`"):
        BoundaryForcingSource(
            grid=_small_grid(),
            start_time=datetime(2021, 6, 29),
            end_time=datetime(2021, 6, 30),
            type="bgc",
            source={"name": "ESPER", "path": _PYESPER_PATH},
            use_dask=False,
        )


def test_ic_source_required_without_physics_forcing():
    with pytest.raises(ValueError, match="`source` is required"):
        InitialConditionsSource(
            grid=_small_grid(),
            ini_time=datetime(2021, 6, 29),
            type="bgc",
            physics_forcing=_small_physics_ic(),
            use_dask=False,
        )


def _small_physics_ic(use_dask: bool = False) -> InitialConditionsSource:
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    return InitialConditionsSource(
        grid=_small_grid(),
        ini_time=datetime(2021, 6, 29),
        source={"name": "GLORYS", "path": fname},
        use_dask=use_dask,
    )


def test_ic_physics_forcing_only_applies_to_bgc_type():
    phys = _small_physics_ic()
    with pytest.raises(ValueError, match="only applies when `type='bgc'`"):
        InitialConditionsSource(
            grid=_small_grid(),
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": "physics.nc"},
            physics_forcing=phys,
            use_dask=False,
        )


def test_ic_physics_forcing_requires_source():
    phys = _small_physics_ic()
    with pytest.raises(ValueError, match="`source` is required"):
        InitialConditionsSource(
            grid=_small_grid(),
            ini_time=datetime(2021, 6, 29),
            type="bgc",
            physics_forcing=phys,
            use_dask=False,
        )


# --------------------------------------------------------------------------------------
# Estimation (require PyESPER + data)
# --------------------------------------------------------------------------------------
@needs_pyesper
@pytest.mark.parametrize("method", ["nn", "lir"])
def test_estimate_bgc_fields_units_and_laziness(method):
    import xarray as xr

    ny, nz = 4, 3
    rng = np.random.default_rng(0)
    temp = xr.DataArray(rng.uniform(2, 20, (nz, ny)), dims=("s", "y")).chunk({"y": 2})
    salt = xr.DataArray(rng.uniform(34, 36, (nz, ny)), dims=("s", "y")).chunk({"y": 2})
    lon = xr.DataArray(rng.uniform(-40, 0, ny), dims=("y",))
    lat = xr.DataArray(rng.uniform(40, 60, ny), dims=("y",))
    depth = xr.DataArray(np.linspace(0, 1000, nz), dims=("s",))

    out = estimate_bgc_fields(
        temp,
        salt,
        lon,
        lat,
        depth,
        source={"name": "ESPER", "path": _PYESPER_PATH, "method": method},
        roms_variables=["NO3", "ALK", "DIC", "O2"],
        est_dates=2020.0,
    )
    assert set(out) == {"NO3", "ALK", "DIC", "O2"}
    # lazy until computed
    assert hasattr(out["NO3"].data, "dask")
    with dask.config.set(scheduler="synchronous"):
        no3 = out["NO3"].compute()
        alk = out["ALK"].compute()
        o2 = out["O2"].compute()
    # mmol/m^3 magnitudes; non-negative; ALK ~ 2000-2600; O2 ~ 0-400 (OMZ to supersaturated)
    assert float(no3.min()) >= 0.0
    assert 1500.0 < float(alk.mean()) < 3000.0
    assert float(o2.min()) >= 0.0
    assert 0.0 < float(o2.mean()) < 400.0
    assert no3.dims == ("s", "y")
    assert o2.attrs.get("long_name") == "dissolved oxygen"
    assert o2.attrs.get("units") == "mmol/m^3"


@needs_pyesper
def test_estimate_bgc_fields_call_count_independent_of_variable_count(monkeypatch):
    """Regression test: `est`'s per-variable DataArrays all share ONE underlying
    per-chunk `nn()` call (`apply_ufunc` is called once, with N output_core_dims).
    Materialising them with N *separate* `.compute()` calls (one per variable)
    instead of one combined `dask.compute()` does NOT share that upstream task --
    verified empirically to silently multiply every chunk's real PyESPER
    invocation count by the number of requested variables. `estimate_bgc_fields`
    itself stays lazy (see its docstring), so this test does the materialising a
    real caller would eventually do -- one combined `dask.compute()` over every
    requested variable together -- and checks the one property that actually
    matters: real `nn()` call count for the same array must be the same whether 2
    or 4 variables are requested. A regression (materialising separately) would
    multiply it by variable count instead. Rather than pin down an exact expected
    chunk count (which depends on two layers of rechunking -- roms-tools' own
    plan, then PyESPER's own further "auto" rechunk downstream of it -- and is
    liable to shift for reasons unrelated to this bug), only the ratio between
    the two variable counts is checked.
    """
    import sys

    import PyESPER.nn  # noqa: F401 -- ensures the submodule is in sys.modules

    # `PyESPER/__init__.py` does `from .nn import nn`, which rebinds the
    # *package*-level `PyESPER.nn` attribute to the function -- shadowing the
    # submodule there. `xr_methods._method_fn`'s own `from PyESPER.nn import nn`
    # reads straight off the submodule object in `sys.modules`, bypassing that
    # shadowing, so the patch target must be the same: `sys.modules`, not
    # `PyESPER.nn` attribute access (which would silently patch the function
    # object itself, not the module's `nn` attribute the real code looks up).
    pyesper_nn_module = sys.modules["PyESPER.nn"]
    real_nn = pyesper_nn_module.nn

    def _run(roms_variables) -> int:
        call_count = {"n": 0}

        def counting_nn(*args, **kwargs):
            call_count["n"] += 1
            return real_nn(*args, **kwargs)

        monkeypatch.setattr(pyesper_nn_module, "nn", counting_nn)

        ny, nz = 4, 3
        rng = np.random.default_rng(0)
        temp = xr.DataArray(rng.uniform(2, 20, (nz, ny)), dims=("s", "y")).chunk(
            {"y": 2}
        )
        salt = xr.DataArray(rng.uniform(34, 36, (nz, ny)), dims=("s", "y")).chunk(
            {"y": 2}
        )
        lon = xr.DataArray(rng.uniform(-40, 0, ny), dims=("y",))
        lat = xr.DataArray(rng.uniform(40, 60, ny), dims=("y",))
        depth = xr.DataArray(np.linspace(0, 1000, nz), dims=("s",))

        out = estimate_bgc_fields(
            temp,
            salt,
            lon,
            lat,
            depth,
            source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
            roms_variables=roms_variables,
            est_dates=2020.0,
        )
        # `estimate_bgc_fields` itself is lazy -- materialise every requested
        # variable together in one combined call, the way a real caller
        # eventually does at write time (see the module docstring).
        dask.compute(*out.values(), scheduler="synchronous")
        return call_count["n"]

    # (Two variables, not one: a single-`DesiredVariables` request hits an
    # unrelated pre-existing shape issue elsewhere in PyESPER, orthogonal to
    # what this test checks.)
    calls_for_two_vars = _run(["NO3", "ALK"])
    calls_for_four_vars = _run(["NO3", "ALK", "DIC", "O2"])
    assert calls_for_two_vars > 0
    assert calls_for_four_vars == calls_for_two_vars, (
        f"real nn() call count must not depend on how many variables are "
        f"requested (one chunk -> one call, regardless of output count): got "
        f"{calls_for_two_vars} call(s) for 2 variables vs "
        f"{calls_for_four_vars} call(s) for 4 -- the latter being ~2x the "
        "former would mean the per-variable-separate-.compute() bug has "
        "regressed."
    )


@needs_pyesper
def test_initial_conditions_esper(use_dask):
    grid = _small_grid()
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with dask.config.set(scheduler="synchronous"):
        ic = InitialConditions(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            bgc_source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
            bgc_model=BGCMarbl,
            use_dask=use_dask,
        )
        known = BGCMarbl().known_vars()
        present = {str(v) for v in ic.ds.data_vars if str(v) in known}
        assert set(ESPER_SUPPORTED_VARS) <= present
        assert present <= known
        # units/magnitudes
        assert 1500.0 < float(ic.ds["ALK"].mean()) < 3000.0
        assert float(ic.ds["NO3"].min()) >= 0.0

        # process_bgc_fields completes the MARBL set
        ic2 = copy.copy(ic)
        ic2.ds = ic.ds.copy(deep=True)
        BGCMarbl().process_bgc_fields(ic2)
        for var in BGCMarbl().tracer_vars():
            assert var in ic2.ds, f"{var} missing"


@needs_pyesper
def test_initial_conditions_esper_use_vars(use_dask):
    """use_vars down-selection holds on a raw (not auto-completed) bgc object.

    Built directly at the ``InitialConditionsSource`` level: the
    ``InitialConditions`` wrapper's ``bgc_model`` auto-completion would
    legitimately re-fill an excluded "required" tracer like NO3/PO4 from a
    default, which is unrelated to what's under test here (down-selection
    itself).
    """
    grid = _small_grid()
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with dask.config.set(scheduler="synchronous"):
        phys = InitialConditionsSource(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            use_dask=use_dask,
        )
        ic = InitialConditionsSource(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            type="bgc",
            source={"name": "ESPER", "path": _PYESPER_PATH, "method": "lir"},
            physics_forcing=phys,
            use_vars=["ALK", "DIC"],
            use_dask=use_dask,
        )
        assert "ALK" in ic.ds and "DIC" in ic.ds
        assert "NO3" not in ic.ds and "PO4" not in ic.ds


@needs_pyesper
def test_initial_conditions_esper_with_physics_forcing_matches_combined(use_dask):
    """A `physics_forcing`-driven, BGC-only IC object must produce ESPER tracers
    matching the old combined (redundant-regrid) path to float32 precision, while
    carrying none of the physics variables in its own dataset.

    Not bit-exact: `combined`'s bgc pass consumes its own in-memory (float64) T/S,
    whereas `split` consumes `phys.ds["temp"/"salt"]` -- already written out and
    downcast to float32 (see `_write_into_dataset`). This is the same precision
    characteristic BoundaryForcing's ESPER-via-physics_forcing path already has.
    """
    grid = _small_grid()
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with dask.config.set(scheduler="synchronous"):
        combined = InitialConditions(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            bgc_source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
            bgc_model=BGCMarbl,
            use_dask=use_dask,
        )

        # `physics_forcing` is a `InitialConditionsSource`-only mechanism (the
        # `InitialConditions` wrapper always builds its own physics companion
        # internally), so `phys`/`split` are built directly at that level here.
        phys = InitialConditionsSource(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            use_dask=use_dask,
        )
        split = InitialConditionsSource(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            type="bgc",
            source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
            physics_forcing=phys,
            use_dask=use_dask,
        )

        # The split (bgc-only) object carries no physics fields of its own.
        for var in ("u", "v", "zeta", "w", "ubar", "vbar", "temp", "salt"):
            assert var not in split.ds

        # But every physics var IS present on the companion physics object.
        for var in ("u", "v", "zeta", "w", "ubar", "vbar", "temp", "salt"):
            assert var in phys.ds

        for var in ESPER_SUPPORTED_VARS:
            xr.testing.assert_allclose(
                split.ds[var], combined.ds[var], rtol=1e-3, atol=1e-3
            )


@needs_pyesper
def test_initial_conditions_physics_forcing_yaml_roundtrip(tmp_path, use_dask):
    """A `physics_forcing`-driven IC object must round-trip through YAML with its
    companion physics object intact.
    """
    grid = _small_grid()
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with dask.config.set(scheduler="synchronous"):
        phys = InitialConditionsSource(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            use_dask=use_dask,
        )
        split = InitialConditionsSource(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            type="bgc",
            source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
            physics_forcing=phys,
            use_dask=use_dask,
        )

        filepath = tmp_path / "esper_ic.yaml"
        split.to_yaml(filepath)
        reloaded = InitialConditionsSource.from_yaml(filepath, use_dask=use_dask)

        assert reloaded.physics_forcing is not None
        assert reloaded.source["name"] == "ESPER"
        assert reloaded.physics_forcing.grid is reloaded.grid

        for var in ESPER_SUPPORTED_VARS:
            xr.testing.assert_allclose(reloaded.ds[var], split.ds[var])


@needs_pyesper
def test_initial_conditions_merge_combines_physics_and_bgc(use_dask):
    """`InitialConditionsSource.merge()` combines a physics object with N bgc-only
    objects into one dataset, matching a manual `xr.merge` of the same pieces.
    """
    grid = _small_grid()
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with dask.config.set(scheduler="synchronous"):
        phys = InitialConditionsSource(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            use_dask=use_dask,
        )
        esper = InitialConditionsSource(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            type="bgc",
            source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
            physics_forcing=phys,
            use_dask=use_dask,
        )

        merged = InitialConditionsSource.merge(phys, esper)
        expected = xr.merge(
            [phys.ds, esper.ds], compat="override", combine_attrs="override"
        )
        xr.testing.assert_identical(merged, expected)

        # Accepts a bare object, not just a list.
        merged_from_list = InitialConditionsSource.merge(phys, [esper])
        xr.testing.assert_identical(merged, merged_from_list)

        for var in ("u", "v", "zeta", "temp", "salt"):
            assert var in merged
        for var in ESPER_SUPPORTED_VARS:
            assert var in merged


def test_initial_conditions_merge_rejects_empty_bgc_list():
    phys = _small_physics_ic()
    with pytest.raises(ValueError, match="at least one"):
        InitialConditionsSource.merge(phys, [])


def test_initial_conditions_merge_rejects_mismatched_physics_forcing():
    phys = _small_physics_ic()
    other_phys = _small_physics_ic()
    bgc = InitialConditionsSource(
        grid=_small_grid(),
        ini_time=datetime(2021, 6, 29),
        type="bgc",
        source={"name": "constants", "constants": {"ALK": 2350.0}},
        physics_forcing=other_phys,
        use_dask=False,
    )
    with pytest.raises(ValueError, match="physics_forcing.*is not `physics`"):
        InitialConditionsSource.merge(phys, bgc)


@needs_pyesper
def test_initial_conditions_merge_with_filepath_saves(tmp_path, use_dask):
    grid = _small_grid()
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with dask.config.set(scheduler="synchronous"):
        phys = InitialConditionsSource(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            use_dask=use_dask,
        )
        esper = InitialConditionsSource(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            type="bgc",
            source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
            physics_forcing=phys,
            use_dask=use_dask,
        )

        saved = InitialConditionsSource.merge(
            phys, esper, filepath=tmp_path / "merged_ic.nc"
        )
        assert len(saved) == 1
        reopened = xr.open_dataset(saved[0])
        for var in ("u", "v", "zeta", "temp", "salt", *ESPER_SUPPORTED_VARS):
            assert var in reopened


@needs_pyesper
def test_boundary_forcing_esper(use_dask):
    grid = _small_grid()
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with dask.config.set(scheduler="synchronous"):
        phys = BoundaryForcingSource(
            grid=grid,
            start_time=datetime(2021, 6, 29),
            end_time=datetime(2021, 6, 30),
            type="physics",
            source={"name": "GLORYS", "path": fname},
            use_dask=use_dask,
        )
        bf = BoundaryForcingSource(
            grid=grid,
            start_time=datetime(2021, 6, 29),
            end_time=datetime(2021, 6, 30),
            type="bgc",
            source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
            physics_forcing=phys,
            use_dask=use_dask,
        )
        active = [d for d, on in bf.boundaries.items() if on]
        known = BGCMarbl().known_vars()
        for d in active:
            present = {
                str(v)[: -(len(d) + 1)]
                for v in bf.ds.data_vars
                if str(v).endswith(f"_{d}")
            }
            assert set(ESPER_SUPPORTED_VARS) <= present
            assert {p for p in present if p in known} <= known
        assert "bry_time" in bf.ds.dims

        bf2 = copy.copy(bf)
        bf2.ds = bf.ds.copy(deep=True)
        BGCMarbl().process_bgc_fields(bf2)
        for d in active:
            for var in BGCMarbl().tracer_vars():
                assert f"{var}_{d}" in bf2.ds, f"{var}_{d} missing"


@needs_pyesper
def test_esper_construction_triggers_serialize_dask_via_validate(monkeypatch):
    """Regression test for a real production crash (confirmed via a full
    thread-stack dump): an ESPER source's own ``_validate()`` -- called from
    ``__post_init__``, i.e. at OBJECT CONSTRUCTION time, well before any
    ``.save()`` -- runs ``nan_check_batch``, which does a real
    ``dask.compute()``. That ``dask.compute()`` was NOT covered by
    ``HIGH_MEMORY_METHOD``-based serialization at all (only ``.save()`` was),
    so it ran under whatever the ambient (often fully concurrent, several
    dask workers) scheduler happened to be -- multiplying ESPER's own
    already-large per-chunk memory cost by however many workers ran
    concurrently. Confirmed via a kernel OOM-kill log to exhaust all memory on
    a 251 GB machine. Fixed by threading ``serialize_dask=self.HIGH_MEMORY_METHOD``
    into ``nan_check_batch`` from both ``InitialConditionsSource._validate``
    and ``BoundaryForcingSource._validate``. This test checks that
    construction of an ESPER-backed source actually invokes
    ``serialize_dask_and_boost_threads(True)`` -- not just that
    ``HIGH_MEMORY_METHOD`` is ``True`` in isolation, which would not have
    caught this bug (the property existed and was correct; it just wasn't
    being *consulted* at construction time).
    """
    import roms_tools.setup.utils as setup_utils_mod
    import roms_tools.utils as utils_mod

    calls = []
    real = utils_mod.serialize_dask_and_boost_threads

    def spy(serialize):
        calls.append(serialize)
        return real(serialize)

    # Patch both the defining module and the name `setup/utils.py` imported,
    # since the latter is what `nan_check_batch` actually calls.
    monkeypatch.setattr(utils_mod, "serialize_dask_and_boost_threads", spy)
    monkeypatch.setattr(setup_utils_mod, "serialize_dask_and_boost_threads", spy)

    phys = _small_physics_ic()
    calls.clear()  # ignore whatever physics's own (non-ESPER) construction triggered

    esper = InitialConditionsSource(
        grid=_small_grid(),
        ini_time=datetime(2021, 6, 29),
        type="bgc",
        source={"name": "ESPER", "path": _PYESPER_PATH},
        physics_forcing=phys,
        use_dask=False,
    )
    assert esper.HIGH_MEMORY_METHOD is True
    assert True in calls, (
        "constructing an ESPER-backed InitialConditionsSource must trigger "
        "serialize_dask_and_boost_threads(True) via _validate()'s "
        "nan_check_batch call -- if this is empty or only contains False, the "
        "construction-time compute is running unprotected again."
    )
