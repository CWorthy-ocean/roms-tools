"""Tests for the ESPER (PyESPER-derived) BGC source.

The estimation tests need both the PyESPER package and its on-disk data
(``Mat_fullgrid/`` + ``NeuralNetworks/``); they are skipped unless a PyESPER directory is
available. Set ``ROMS_TOOLS_PYESPER_PATH`` to point at it (defaults to a local checkout).

The dict-shape validation tests run everywhere -- they trip on `method`/`equation`
before `validate_esper_source` reaches its import check. Anything that expects a
*successful* validation is marked ``needs_pyesper``, because that check now makes a
missing package an error in its own right; the two tests that assert on the guidance
message stub the import out instead, so they run bare.
"""

import copy
import itertools
import os
import sys
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
from roms_tools.setup import esper as esper_module
from roms_tools.setup.esper import (
    _MAX_POINTS_PER_CHUNK,
    ESPER_SUPPORTED_VARS,
    _apply_chunk_plan,
    _decimal_year,
    _pyesper_chunk_plan,
    _time_dim,
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
@needs_pyesper
def test_validate_esper_source_accepts_missing_path():
    """`path` is optional: without one, PyESPER must simply be importable from the
    environment (e.g. pip install -e), and it locates its own data directories.

    Needs PyESPER present: validation ends with the import check, so "no path" only
    passes when the environment supplies the package.
    """
    validate_esper_source({"name": "ESPER"})  # must not raise


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


def _daily_bry_temp(start, end, spatial_len, spatial_dim="xi_rho", n_levels=100):
    """A boundary-shaped `temp`: (time, s_rho, <spatial>) on the daily datetime
    axis -- including the bracketing days -- that BoundaryForcing hands the ESPER
    path, arriving per-step chunked as the upstream regrid leaves it.
    """
    import dask.array as dsa
    import pandas as pd

    times = pd.date_range(
        pd.Timestamp(start) - pd.Timedelta(days=1),
        pd.Timestamp(end) + pd.Timedelta(days=1),
        freq="D",
    )
    arr = dsa.zeros(
        (len(times), n_levels, spatial_len),
        chunks=(1, n_levels, spatial_len),
        dtype="f8",
    )
    return xr.DataArray(
        arr, dims=("time", "s_rho", spatial_dim), coords={"time": times}
    )


def test_pyesper_chunk_plan_cuts_time_not_space_for_a_long_boundary_run():
    """Regression (OOM kill): a multi-month boundary run must be cut along TIME.

    Cutting a boundary slab spatially leaves every chunk needing the *whole* time
    range of the upstream regrid behind it, so nothing streams and a year-long
    run holds a year-long GLORYS slab per chunk. That OOM-killed a 251 GB machine
    on the 12-month, 100-level Pacific 12km domain reproduced here (367 daily
    steps x 100 levels x xi_rho 1858), while the physics write of the very same
    boundaries -- which keeps its natural per-time chunking -- streamed fine.
    """
    temp = _daily_bry_temp("2010-01-01", "2010-12-31", 1858)
    plan = _pyesper_chunk_plan(temp)

    # Time is the axis that gets cut; the spatial dims stay whole.
    assert plan["s_rho"] == -1
    assert plan["xi_rho"] == -1
    assert plan["time"] != -1

    rechunked = _apply_chunk_plan(temp, plan)
    assert rechunked.shape == temp.shape  # pure block-boundary change
    time_blocks = rechunked.chunks[rechunked.dims.index("time")]
    assert sum(time_blocks) == temp.sizes["time"]
    assert max(time_blocks) < temp.sizes["time"], "the time axis must really split"

    # No block exceeds the budget PyESPER's per-chunk memory bound assumes.
    for block in itertools.product(*rechunked.chunks):
        assert np.prod(block) <= _MAX_POINTS_PER_CHUNK


def test_pyesper_chunk_plan_time_blocks_land_on_month_boundaries():
    """Every time-block edge is a calendar-month edge -- the same partition
    `group_by_month` uses to split the saved files -- so a chunk always covers
    whole output files and dask can retire it once those are written, rather than
    holding it for a file it only partly feeds.
    """
    temp = _daily_bry_temp("2010-01-01", "2010-12-31", 1858)
    blocks = _pyesper_chunk_plan(temp)["time"]
    assert isinstance(blocks, tuple)

    index = temp["time"].to_index()
    month_edges = {0, len(index)}
    for i in range(1, len(index)):
        if (index[i].year, index[i].month) != (index[i - 1].year, index[i - 1].month):
            month_edges.add(i)

    edge = 0
    for block in blocks:
        edge += block
        assert edge in month_edges, (
            f"block edge at time index {edge} falls mid-month; the chunk would "
            "straddle a partial output file"
        )


def test_pyesper_chunk_plan_gives_one_block_per_month_never_more():
    """A block is recomputed once per output file it feeds, because
    `save_mfdataset(compute=True)` issues a separate `dask.compute` per file. So
    months must never be bundled, even when two of them would fit the point
    budget -- a two-month block means every monthly write recomputes both months.

    Measured on this axis (367 daily steps -> 14 monthly files), as multiples of
    the useful work: one block per month 2.8x, two months per block 4.8x, time
    collapsed to a single block 14.0x.
    """
    index = _daily_bry_temp("2010-01-01", "2010-12-31", 1858)["time"].to_index()
    expected = tuple(
        sum(1 for _ in group)
        for _, group in itertools.groupby(zip(index.year, index.month, strict=True))
    )

    for spatial_len, spatial_dim in ((1858, "xi_rho"), (962, "eta_rho")):
        temp = _daily_bry_temp(
            "2010-01-01", "2010-12-31", spatial_len, spatial_dim=spatial_dim
        )
        blocks = _pyesper_chunk_plan(temp)["time"]
        assert blocks == expected, (
            f"{spatial_dim}: expected one block per calendar month, got {blocks}"
        )

    # The east/west slab is small enough that two months would still fit the
    # budget -- exactly the case that must NOT be bundled.
    assert 2 * 31 * 100 * 962 < _MAX_POINTS_PER_CHUNK


def test_pyesper_chunk_plan_leaves_a_short_run_as_one_chunk():
    """No regression for the configuration this pipeline was validated on: the
    CCS 4km / 2-month boundary (62 daily steps x 100 levels x xi_rho 674) is
    under the point budget, so it stays a single chunk and pays no extra
    per-chunk PyESPER setup cost.
    """
    temp = _daily_bry_temp("2013-01-01", "2013-03-01", 674)
    assert temp.size <= _MAX_POINTS_PER_CHUNK
    assert _pyesper_chunk_plan(temp) == {"time": -1, "s_rho": -1, "xi_rho": -1}


def test_pyesper_chunk_plan_uniform_time_blocks_without_a_datetime_coord():
    """With no datetime coordinate there are no months to align to, so the time
    axis is cut into uniform blocks instead -- still time, still under budget.
    """
    import dask.array as dsa

    temp = xr.DataArray(
        dsa.zeros((367, 100, 1858), chunks=(1, 100, 1858), dtype="f8"),
        dims=("time", "s_rho", "xi_rho"),
    )
    plan = _pyesper_chunk_plan(temp)
    assert isinstance(plan["time"], int)
    assert 0 < plan["time"] < temp.sizes["time"]
    assert plan["s_rho"] == -1 and plan["xi_rho"] == -1
    assert 100 * 1858 * plan["time"] <= _MAX_POINTS_PER_CHUNK


def test_pyesper_chunk_plan_splits_space_only_when_one_step_busts_the_budget():
    """A single time level over the budget is the one case that still needs a
    spatial cut -- but time drops to one step first, so a chunk never pulls more
    than one step of upstream regrid.
    """
    import dask.array as dsa
    import pandas as pd

    temp = xr.DataArray(
        dsa.zeros((40, 100, 80_000), chunks=(1, 100, 80_000), dtype="f8"),
        dims=("time", "s_rho", "xi_rho"),
        coords={"time": pd.date_range("2010-01-01", periods=40, freq="D")},
    )
    assert 100 * 80_000 > _MAX_POINTS_PER_CHUNK  # one step alone busts it
    plan = _pyesper_chunk_plan(temp)
    assert plan["time"] == 1
    assert plan["xi_rho"] < 80_000
    assert 100 * plan["xi_rho"] <= _MAX_POINTS_PER_CHUNK


def test_pyesper_chunk_plan_ic_volume_unchanged_without_a_time_dim():
    """Initial conditions are one instant with no time axis, so the plan keeps
    its original behaviour: collapse every dim, split the single largest one.
    """
    import dask.array as dsa

    temp = xr.DataArray(
        dsa.zeros((100, 1344, 672), chunks=(1, 50, 50), dtype="f8"),
        dims=("s_rho", "eta_rho", "xi_rho"),
    )
    assert _time_dim(temp) is None
    plan = _pyesper_chunk_plan(temp)
    assert plan["s_rho"] == -1 and plan["xi_rho"] == -1
    assert 0 < plan["eta_rho"] < 1344  # the largest dim is the one that splits


def test_apply_chunk_plan_filters_entries_the_input_cannot_take():
    """`DataArray.chunk` raises on a mapping key that isn't one of the array's own
    dims, so a plan derived from `temp` has to be filtered per input: 2D lon/lat
    against a 3D temp, and a boundary `depth` carrying no time dim at all.
    """
    import dask.array as dsa
    import pandas as pd

    plan = {"time": (3, 3), "s_rho": -1, "xi_rho": -1}

    lon = xr.DataArray(dsa.zeros((8,), chunks=(8,), dtype="f8"), dims=("xi_rho",))
    assert _apply_chunk_plan(lon, plan).chunks == ((8,),)

    depth = xr.DataArray(
        dsa.zeros((4, 8), chunks=(4, 8), dtype="f8"), dims=("s_rho", "xi_rho")
    )
    assert _apply_chunk_plan(depth, plan).chunks == ((4,), (8,))

    temp = xr.DataArray(
        dsa.zeros((6, 4, 8), chunks=(1, 4, 8), dtype="f8"),
        dims=("time", "s_rho", "xi_rho"),
        coords={"time": pd.date_range("2010-01-01", periods=6, freq="D")},
    )
    assert _apply_chunk_plan(temp, plan).chunks == ((3, 3), (4,), (8,))

    # A tuple that doesn't sum to this input's own length along the dim falls
    # back to a single chunk there rather than raising.
    assert _apply_chunk_plan(temp.isel(time=slice(0, 5)), plan).chunks == (
        (5,),
        (4,),
        (8,),
    )


@needs_pyesper
def test_esper_estimates_are_identical_under_time_chunking(monkeypatch):
    """Cutting the time axis is a pure blocking change: PyESPER's kernel is
    point-wise, so time-chunked estimates must match the single-chunk result
    exactly (NaNs in the same places included).
    """
    import dask.array as dsa
    import pandas as pd

    n_time, n_lev, n_x = 6, 3, 5
    shape = (n_time, n_lev, n_x)
    dims = ("time", "s_rho", "xi_rho")
    times = pd.date_range("2010-01-31", periods=n_time, freq="ME")
    rng = np.random.default_rng(0)

    salt_v = 34.0 + rng.random(shape)
    temp_v = 10.0 + 5.0 * rng.random(shape)
    lon_v = np.broadcast_to(np.linspace(200.0, 210.0, n_x), shape).copy()
    lat_v = np.broadcast_to(np.linspace(30.0, 40.0, n_x), shape).copy()
    depth_v = np.broadcast_to(
        np.linspace(10.0, 500.0, n_lev)[None, :, None], shape
    ).copy()
    source = {"name": "ESPER", "path": _PYESPER_PATH, "method": "nn", "equation": 8}

    def run(cap):
        # Drive _pyesper_chunk_plan's decision from a stubbed budget: the real
        # 6M-point cap would leave this 90-point fixture as one chunk either way.
        monkeypatch.setattr(esper_module, "_MAX_POINTS_PER_CHUNK", cap)

        def wrap(values):
            return xr.DataArray(
                dsa.from_array(values, chunks=shape), dims=dims, coords={"time": times}
            )

        fields = estimate_bgc_fields(
            wrap(temp_v),
            wrap(salt_v),
            wrap(lon_v),
            wrap(lat_v),
            wrap(depth_v),
            source=source,
            roms_variables=("ALK", "DIC"),
            est_dates=_decimal_year(xr.DataArray(times, dims=("time",))),
        )
        return {k: np.asarray(v.values) for k, v in fields.items()}

    whole = run(10**9)  # comfortably one chunk
    chunked = run(30)  # forces a cut along time

    # The stubbed budget really did split the time axis.
    monkeypatch.setattr(esper_module, "_MAX_POINTS_PER_CHUNK", 30)
    split_plan = _pyesper_chunk_plan(
        xr.DataArray(
            dsa.from_array(temp_v, chunks=shape), dims=dims, coords={"time": times}
        )
    )
    assert split_plan["time"] != -1

    assert np.isfinite(whole["ALK"]).any(), "fixture should produce real estimates"
    for name in ("ALK", "DIC"):
        np.testing.assert_array_equal(chunked[name], whole[name])


@needs_pyesper
def test_ic_esper_without_path_uses_the_importable_pyesper():
    """An ESPER source with no `path` works when PyESPER is importable.

    The test session imports PyESPER off ``_PYESPER_PATH`` (see module top), which
    mirrors an installed environment: `_ensure_pyesper(None)` finds the already-
    importable package, and PyESPER's own ``paths.data_root()`` auto-detects the
    data directories next to it. This is the "pip install -e, no path in the
    blueprint" configuration.
    """
    ic = InitialConditionsSource(
        grid=_small_grid(),
        ini_time=datetime(2021, 6, 29),
        type="bgc",
        source={"name": "ESPER"},
        physics_forcing=_small_physics_ic(),
        use_dask=False,
    )
    assert ic._is_esper_source is True
    for var_name in ("ALK", "DIC", "NO3", "PO4", "SiO3", "O2"):
        assert var_name in ic.ds
        assert np.isfinite(np.asarray(ic.ds[var_name].values)).any()


@needs_pyesper
def test_bf_esper_requires_physics_forcing():
    """Needs PyESPER present: `_input_checks` validates the ESPER source (import
    check included) before it gets to the `physics_forcing` requirement, so without
    the package this raises ImportError instead of the ValueError under test.
    """
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
def test_esper_construction_materializes_without_serializing(monkeypatch):
    """Construction-time behaviour after the removal of HIGH_MEMORY_METHOD.

    History: an ESPER source's ``_validate()`` (run from ``__post_init__``, i.e.
    at OBJECT CONSTRUCTION time) does a real ``dask.compute()``. Against a
    pre-kernel-lock PyESPER that compute had to run serialized -- a production
    crash (thread-stack dump + kernel OOM-kill log on a 251 GB machine) was
    traced to exactly this compute running under the ambient concurrent
    scheduler. PyESPER now serialises its own kernels, so the automatic
    caller-side protection was removed; ``serialize_dask`` survives only as a
    manual kwarg.

    Two properties must hold, and this test checks both:

    1. Construction never *automatically* enters
       ``serialize_dask_and_boost_threads(True)`` -- everything runs under the
       ambient scheduler.
    2. Construction still *materialises* every ESPER variable into ``.ds`` (the
       compute-once cache that stops ``.save()`` from recomputing the whole
       PyESPER graph).
    """
    import roms_tools.setup.utils as setup_utils_mod
    import roms_tools.utils as utils_mod

    calls = []
    real = utils_mod.serialize_dask_and_boost_threads

    def spy(serialize):
        calls.append(serialize)
        return real(serialize)

    # Patch both the defining module and the name `setup/utils.py` imported,
    # since the latter is what nan_check_batch/materialize_before_check call.
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
    assert esper._is_esper_source is True
    assert calls, "construction never reached the compute paths under test"
    assert True not in calls, (
        "construction entered serialize_dask_and_boost_threads(True) -- nothing "
        "should turn the serialized regime on automatically any more"
    )
    for var_name in esper.ds.data_vars:
        assert not hasattr(esper.ds[var_name].data, "dask"), (
            f"{var_name} is still lazy after construction -- "
            "materialize_before_check stopped caching"
        )


@needs_pyesper
def test_serialize_dask_kwarg_still_forces_the_serialized_write(monkeypatch, tmp_path):
    """The manual escape hatch: ``save(serialize_dask=True)`` must still route
    the write through ``serialize_dask_and_boost_threads(True)`` -- kept for
    low-memory machines (it bounds peak memory to one task's footprint, which
    the plain threaded scheduler cannot guarantee) and for troubleshooting.
    """
    import roms_tools.utils as utils_mod

    calls = []
    real = utils_mod.serialize_dask_and_boost_threads

    def spy(serialize):
        calls.append(serialize)
        return real(serialize)

    monkeypatch.setattr(utils_mod, "serialize_dask_and_boost_threads", spy)

    phys = _small_physics_ic(use_dask=True)

    # Default: concurrent write, the serialized context is never entered as True.
    calls.clear()
    phys.save(tmp_path / "default")
    assert True not in calls

    # Manual request: the serialized context must be entered.
    calls.clear()
    phys.save(tmp_path / "serialized", serialize_dask=True)
    assert True in calls, (
        "save(serialize_dask=True) no longer reaches "
        "serialize_dask_and_boost_threads(True) -- the manual escape hatch is broken"
    )


@needs_pyesper
def test_estimate_bgc_fields_threaded_scheduler_matches_synchronous():
    """ESPER estimation under dask's *threaded* scheduler must complete and agree
    with the synchronous result.

    This is the roms-tools-level counterpart of PyESPER's own
    ``test_nn_xr_completes_under_dasks_threaded_scheduler``: with the capability
    gate flipping ``HIGH_MEMORY_METHOD`` off, production saves now compute these
    graphs under the ambient threaded scheduler, so that path must be exercised
    here too -- against the legacy PyESPER this exact configuration deadlocked.
    """
    import time as _time

    import xarray as xr

    ny, nz = 8, 4
    rng = np.random.default_rng(3)
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
        source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
        roms_variables=["NO3", "ALK"],
        est_dates=2020.0,
    )

    started = _time.monotonic()
    with dask.config.set(scheduler="threads", num_workers=4):
        threaded = dask.compute(*(out[v] for v in ("NO3", "ALK")))
    assert _time.monotonic() - started < 300, "threaded compute took implausibly long"

    with dask.config.set(scheduler="synchronous"):
        serial = dask.compute(*(out[v] for v in ("NO3", "ALK")))

    for name, a, b in zip(("NO3", "ALK"), threaded, serial):
        np.testing.assert_allclose(
            a.values,
            b.values,
            rtol=1e-12,
            atol=0.0,
            err_msg=f"{name}: threaded and synchronous schedulers disagree",
        )


@needs_pyesper
def test_esper_validate_does_not_recompute_at_save_time(monkeypatch, tmp_path):
    """Regression test for a real production double-compute: construction-time
    `_validate()` and the later `.save()` each independently forced PyESPER's
    expensive per-chunk neural-net evaluation (`nn()`), because `_validate()`'s
    `nan_check_batch` computed-and-discarded the checked field instead of
    caching it -- confirmed via a production log showing ~16 "PyESPER_NN
    took..." lines before "Writing the following NetCDF files:" and ~16 more
    after, roughly doubling wall-clock time. Fixed via
    `materialize_before_check` (`roms_tools/setup/utils.py`), called from
    `_validate()` before any NaN-check view is built.

    This test checks BOTH of the two ways a narrower fix could still be wrong:
    1. Total real `nn()` call count must not increase between construction
       (validate has run) and `.save()` -- proves `.save()` reused validate's
       realized values rather than recomputing.
    2. EVERY one of the source's `use_vars` (not just `ALK`, the only variable
       `bgc_variable_info()` actually flags `validate=True` for) must already
       be concrete (non-dask-backed) in `.ds` right after construction --
       proves the fix caches every variable sharing ALK's underlying PyESPER
       call, not just ALK itself. A fix that only cached the validated
       variable would still pass check 1 by accident in some cases while
       silently failing to help DIC/NO3/PO4/SiO3/O2 at all.
    """
    import sys

    import PyESPER.nn  # noqa: F401 -- ensures the submodule is in sys.modules

    # See test_estimate_bgc_fields_call_count_independent_of_variable_count's
    # comment above for why this must patch `sys.modules["PyESPER.nn"].nn`,
    # not `PyESPER.nn` attribute access (shadowed by `PyESPER/__init__.py`).
    pyesper_nn_module = sys.modules["PyESPER.nn"]
    real_nn = pyesper_nn_module.nn
    call_count = {"n": 0}

    def counting_nn(*args, **kwargs):
        call_count["n"] += 1
        return real_nn(*args, **kwargs)

    monkeypatch.setattr(pyesper_nn_module, "nn", counting_nn)

    with dask.config.set(scheduler="synchronous"):
        phys = _small_physics_ic(use_dask=True)
        esper = InitialConditionsSource(
            grid=_small_grid(),
            ini_time=datetime(2021, 6, 29),
            type="bgc",
            source={"name": "ESPER", "path": _PYESPER_PATH, "method": "nn"},
            use_vars=list(ESPER_SUPPORTED_VARS),
            physics_forcing=phys,
            use_dask=True,
        )

        assert call_count["n"] > 0, (
            "construction (_validate()) should have already triggered at "
            "least one real PyESPER nn() call"
        )
        calls_after_construction = call_count["n"]

        for var in ESPER_SUPPORTED_VARS:
            assert esper.ds[var].chunks is None, (
                f"{var} is still dask-backed right after construction -- "
                "materialize_before_check should have realized every "
                "use_var sharing ALK's PyESPER call, not just ALK itself"
            )

        esper.save(tmp_path / "esper_ic.nc")

        assert call_count["n"] == calls_after_construction, (
            f"PyESPER nn() was called {call_count['n'] - calls_after_construction} "
            "more time(s) during .save() -- validate-time results were not "
            "reused, reproducing the original double-compute bug"
        )


# ---------------------------------------------------------------------------
# Missing / wrong PyESPER: the message has to name the fork and how to get it
# ---------------------------------------------------------------------------


class _BlockPyESPER:
    """Import hook that makes PyESPER unimportable, whatever is installed."""

    def find_spec(self, name, path=None, target=None):
        if name == "PyESPER" or name.startswith("PyESPER."):
            raise ImportError(f"No module named {name!r}")


def _drop_pyesper_modules(monkeypatch):
    for mod in [
        m for m in list(sys.modules) if m == "PyESPER" or m.startswith("PyESPER.")
    ]:
        monkeypatch.delitem(sys.modules, mod, raising=False)


def test_missing_pyesper_points_at_the_cworthy_fork(monkeypatch):
    """No PyESPER at all: say which fork, and give the install steps verbatim.

    The upstream project shares the name, so "install PyESPER" is ambiguous advice --
    the message has to carry the CWorthy URL and the editable-install command.
    """
    from roms_tools.setup.esper import validate_esper_source

    _drop_pyesper_modules(monkeypatch)
    monkeypatch.setattr(sys, "meta_path", [_BlockPyESPER(), *sys.meta_path])

    with pytest.raises(ImportError) as excinfo:
        validate_esper_source({"name": "ESPER"})

    msg = str(excinfo.value)
    assert "https://github.com/CWorthy-ocean/PyESPER" in msg
    assert "pip install -e ." in msg
    assert "No PyESPER is importable" in msg


def test_upstream_pyesper_is_named_as_the_wrong_project(monkeypatch):
    """An importable PyESPER without the `*_xr` methods is upstream, not the fork.

    This is the confusing failure: the package imports fine and only the `*_xr` names
    are missing, which reads like a version skew rather than the wrong project. The
    message must say so, and point at where the offending package was found.
    """
    import types

    from roms_tools.setup.esper import validate_esper_source

    _drop_pyesper_modules(monkeypatch)
    stub = types.ModuleType("PyESPER")
    stub.__file__ = "/somewhere/site-packages/PyESPER/__init__.py"
    monkeypatch.setitem(sys.modules, "PyESPER", stub)

    with pytest.raises(ImportError) as excinfo:
        validate_esper_source({"name": "ESPER"})

    msg = str(excinfo.value)
    assert "upstream PyESPER, not" in msg
    assert "/somewhere/site-packages/PyESPER/__init__.py" in msg
    assert "https://github.com/CWorthy-ocean/PyESPER" in msg


def test_validate_esper_source_checks_the_import_before_any_regridding():
    """The import check lives in `validate_esper_source`, which `_input_checks` calls.

    That places it at the start of the ESPER object's construction rather than inside
    `estimate_bgc_fields`, so a missing PyESPER is reported before the source's own
    depth-coordinate and derivation setup runs.
    """
    import inspect

    from roms_tools.setup import esper

    assert "_ensure_pyesper" in inspect.getsource(esper.validate_esper_source)
