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

from roms_tools import BGCMarbl, BoundaryForcing, Grid, InitialConditions
from roms_tools.datasets.download import download_test_data
from roms_tools.setup.esper import (
    ESPER_SUPPORTED_VARS,
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


def test_ic_esper_missing_path_raises():
    with pytest.raises(ValueError, match="requires a 'path'"):
        InitialConditions(
            grid=_small_grid(),
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": "physics.nc"},
            bgc_source={"name": "ESPER"},
            use_dask=False,
        )


def test_bf_esper_requires_physics_forcing():
    with pytest.raises(ValueError, match="requires `physics_forcing`"):
        BoundaryForcing(
            grid=_small_grid(),
            start_time=datetime(2021, 6, 29),
            end_time=datetime(2021, 6, 30),
            type="bgc",
            source={"name": "ESPER", "path": _PYESPER_PATH},
            use_dask=False,
        )


def test_ic_source_required_without_physics_forcing():
    with pytest.raises(ValueError, match="`source` is required"):
        InitialConditions(
            grid=_small_grid(),
            ini_time=datetime(2021, 6, 29),
            bgc_source={"name": "ESPER", "path": _PYESPER_PATH},
            use_dask=False,
        )


def _small_physics_ic(use_dask: bool = False) -> InitialConditions:
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    return InitialConditions(
        grid=_small_grid(),
        ini_time=datetime(2021, 6, 29),
        source={"name": "GLORYS", "path": fname},
        use_dask=use_dask,
    )


def test_ic_physics_forcing_and_source_are_mutually_exclusive():
    phys = _small_physics_ic()
    with pytest.raises(ValueError, match="must not be provided"):
        InitialConditions(
            grid=_small_grid(),
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": "physics.nc"},
            bgc_source={"name": "ESPER", "path": _PYESPER_PATH},
            physics_forcing=phys,
            use_dask=False,
        )


def test_ic_physics_forcing_requires_bgc_source():
    phys = _small_physics_ic()
    with pytest.raises(ValueError, match="requires `bgc_source`"):
        InitialConditions(
            grid=_small_grid(),
            ini_time=datetime(2021, 6, 29),
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
def test_initial_conditions_esper(use_dask):
    grid = _small_grid()
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with dask.config.set(scheduler="synchronous"):
        ic = InitialConditions(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            bgc_source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
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
    grid = _small_grid()
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with dask.config.set(scheduler="synchronous"):
        ic = InitialConditions(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            bgc_source={"name": "ESPER", "path": _PYESPER_PATH, "method": "lir"},
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
            use_dask=use_dask,
        )

        phys = InitialConditions(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            use_dask=use_dask,
        )
        split = InitialConditions(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            bgc_source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
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
        phys = InitialConditions(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            use_dask=use_dask,
        )
        split = InitialConditions(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            bgc_source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
            physics_forcing=phys,
            use_dask=use_dask,
        )

        filepath = tmp_path / "esper_ic.yaml"
        split.to_yaml(filepath)
        reloaded = InitialConditions.from_yaml(filepath, use_dask=use_dask)

        assert reloaded.physics_forcing is not None
        assert reloaded.source is None
        assert reloaded.physics_forcing.grid is reloaded.grid

        for var in ESPER_SUPPORTED_VARS:
            xr.testing.assert_allclose(reloaded.ds[var], split.ds[var])


@needs_pyesper
def test_initial_conditions_merge_combines_physics_and_bgc(use_dask):
    """`InitialConditions.merge()` combines a physics object with N bgc-only
    objects into one dataset, matching a manual `xr.merge` of the same pieces.
    """
    grid = _small_grid()
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with dask.config.set(scheduler="synchronous"):
        phys = InitialConditions(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            use_dask=use_dask,
        )
        esper = InitialConditions(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            bgc_source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
            physics_forcing=phys,
            use_dask=use_dask,
        )

        merged = InitialConditions.merge(phys, esper)
        expected = xr.merge(
            [phys.ds, esper.ds], compat="override", combine_attrs="override"
        )
        xr.testing.assert_identical(merged, expected)

        # Accepts a bare object, not just a list.
        merged_from_list = InitialConditions.merge(phys, [esper])
        xr.testing.assert_identical(merged, merged_from_list)

        for var in ("u", "v", "zeta", "temp", "salt"):
            assert var in merged
        for var in ESPER_SUPPORTED_VARS:
            assert var in merged


def test_initial_conditions_merge_rejects_empty_bgc_list():
    phys = _small_physics_ic()
    with pytest.raises(ValueError, match="at least one"):
        InitialConditions.merge(phys, [])


def test_initial_conditions_merge_rejects_mismatched_physics_forcing():
    phys = _small_physics_ic()
    other_phys = _small_physics_ic()
    bgc = InitialConditions(
        grid=_small_grid(),
        ini_time=datetime(2021, 6, 29),
        bgc_source={"name": "constants", "constants": {"ALK": 2350.0}},
        physics_forcing=other_phys,
        use_dask=False,
    )
    with pytest.raises(ValueError, match="physics_forcing.*is not `physics`"):
        InitialConditions.merge(phys, bgc)


@needs_pyesper
def test_initial_conditions_merge_with_filepath_saves(tmp_path, use_dask):
    grid = _small_grid()
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with dask.config.set(scheduler="synchronous"):
        phys = InitialConditions(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            use_dask=use_dask,
        )
        esper = InitialConditions(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            bgc_source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
            physics_forcing=phys,
            use_dask=use_dask,
        )

        saved = InitialConditions.merge(phys, esper, filepath=tmp_path / "merged_ic.nc")
        assert len(saved) == 1
        reopened = xr.open_dataset(saved[0])
        for var in ("u", "v", "zeta", "temp", "salt", *ESPER_SUPPORTED_VARS):
            assert var in reopened


@needs_pyesper
def test_boundary_forcing_esper(use_dask):
    grid = _small_grid()
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with dask.config.set(scheduler="synchronous"):
        phys = BoundaryForcing(
            grid=grid,
            start_time=datetime(2021, 6, 29),
            end_time=datetime(2021, 6, 30),
            type="physics",
            source={"name": "GLORYS", "path": fname},
            use_dask=use_dask,
        )
        bf = BoundaryForcing(
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
