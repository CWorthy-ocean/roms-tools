"""Adapter for deriving BGC tracers from physics T/S via PyESPER (the ESPER source).

Unlike the gridded BGC sources (CESM/UNIFIED/GLODAP), which regrid a dataset onto the
ROMS grid, the ``ESPER`` source *derives* biogeochemical variables from the physics
temperature/salinity that are already on the ROMS grid, using the PyESPER empirical
routines. It is therefore handled like the ``constants`` source: no dataset load, no
lateral/vertical regridding.

Requires CWorthy's PyESPER fork (https://github.com/CWorthy-ocean/PyESPER), not
upstream (https://github.com/LarissaMDias/PyESPER): only the fork provides the
dask-native ``*_xr`` methods called here. See ``_PYESPER_INSTALL_HINT``.

The heavy lifting -- dask-lazy, chunked, uncertainty-free estimation -- lives in PyESPER's
``lir_xr``/``nn_xr``/``mixed_xr`` (a fork addition). This module only:

* lazily imports PyESPER (an optional dependency). When the source dict carries a
  ``path``, that directory is put on ``sys.path`` first (a repository checkout);
  without one, PyESPER must simply be importable -- e.g. installed into the
  environment with ``pip install -e <checkout>``, which also lets PyESPER find its
  own data directories (see ``PyESPER.paths.data_root``);
* maps ROMS/MARBL tracer names to/from ESPER names;
* converts ESPER's µmol/kg output to MARBL's mmol/m³ using in-situ density
  (TEOS-10 gsw.rho with pressure from depth);
* clamps physically non-negative tracers at 0.

The returned DataArrays are lazy (dask) when the inputs are, same as any other
lazy result in a forcing dataset -- callers materialise them (along with
everything else in the dataset) at write time.

Concurrency and memory: PyESPER (this project's fused-kernel fork -- the only
PyESPER whose ``*_xr`` methods exist, so the only one that can run here) serialises
its own kernel entry with a module-level lock (``PyESPER.concurrency.kernel_lock``),
budgets its own per-chunk memory internally (~0.5-1.2 KB/point), and does not use
BLAS on the hot path. Its chunks are therefore safe to compute under any ambient
dask scheduler, threaded included; no caller-side scheduler intervention is needed.
For genuinely memory-starved machines or scheduler troubleshooting there is still a
manual one-task-at-a-time write: :func:`roms_tools.utils.save_datasets`'s
``serialize_dask`` (and the ``serialize_dask`` kwarg on every ``save``).

(Historical note: an earlier PyESPER cost ~10 KB/point per chunk -- multiplied by
the dask worker count this OOM-killed a 251 GB machine, and its numba kernel could
deadlock under concurrent entry. The automatic caller-side protection built for it,
``HIGH_MEMORY_METHOD``, was removed once no such PyESPER could be encountered; the
manual ``serialize_dask`` escape hatch is what remains of it.)
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import xarray as xr

from roms_tools.setup.utils import compute_in_situ_density, get_variable_metadata

# Fallback ceiling on ESPER chunk size, used only when PyESPER does not expose its
# own budget helper. Equal to the value PyESPER itself hard-coded before it made the
# cap dynamic, so behaviour on an older fork is unchanged.
_MAX_POINTS_PER_CHUNK = 6_000_000


def _pyesper_point_budget(method: str, n_variables: int) -> int:
    """Points per chunk to target, taken from PyESPER's own per-chunk budget.

    Ask PyESPER rather than keeping a twin constant here. PyESPER adds a rechunk
    layer only when the largest block it receives *exceeds* its budget, and that
    budget is derived from the method and the variable count -- ``mixed`` is
    charged for both the LIR and the NN path, so its cap is roughly a third of
    ``nn``'s. Building our plan from the same number keeps every block inside the
    budget, so PyESPER short-circuits instead of re-blocking the array.

    Matching it matters far more than it looks. A plan even slightly *above*
    PyESPER's cap makes it rechunk, and that severs the block alignment between
    the ESPER estimate and the upstream regrid feeding it, so the upstream chain
    is recomputed per output block. A pac-12km initial condition
    (962 x 1858 x 100) hit exactly that: our flat 6,000,000 sat 1.9% above
    ``mixed``'s 5,883,516 cap, and one call ran for 10 hours -- against 4.4
    seconds for the slowest call of the same grid under ``nn``, which cleared the
    cap and so was never rechunked -- while writing 1.5 TiB into a 10 GiB file.

    Falls back to :data:`_MAX_POINTS_PER_CHUNK` when the helper is absent (an
    older fork) or does not recognise the method.
    """
    try:
        from PyESPER.xr_methods import _max_points_per_chunk
    except ImportError:
        return _MAX_POINTS_PER_CHUNK
    try:
        return int(_max_points_per_chunk(method, n_variables))
    except KeyError:
        # Unknown method: validate_esper_source already rejects those, so this is
        # only reachable if PyESPER's method table and ours drift apart.
        return _MAX_POINTS_PER_CHUNK


# Dim names treated as the time axis when no datetime coordinate identifies one
# (see _time_dim). Ordered by how specific they are to a real time axis.
_TIME_DIM_NAMES = ("time", "bry_time", "abs_time")

# ROMS/MARBL tracer name -> PyESPER "DesiredVariable" name.
ROMS_TO_ESPER = {
    "NO3": "nitrate",
    "PO4": "phosphate",
    "SiO3": "silicate",
    "ALK": "TA",
    "DIC": "DIC",
    "O2": "oxygen",
}
ESPER_TO_ROMS = {v: k for k, v in ROMS_TO_ESPER.items()}

#: ROMS/MARBL tracers the ESPER source can derive from T/S.
ESPER_SUPPORTED_VARS = tuple(ROMS_TO_ESPER)

_VALID_METHODS = ("lir", "nn", "mixed")

#: Shared install guidance. ``{problem}`` names which of the two failure modes was hit
#: -- nothing importable at all, versus an importable PyESPER that turns out to be
#: upstream rather than the fork. The second is the confusing one: the package imports
#: fine and only fails on the ``*_xr`` names, which reads like a version skew rather
#: than the wrong project.
_PYESPER_INSTALL_HINT = (
    "The ESPER BGC source requires CWorthy's PyESPER fork. PyESPER support is "
    "considered experimental at this time.\n"
    "\n"
    "{problem}\n"
    "\n"
    "Install it from source:\n"
    "\n"
    "    git clone https://github.com/CWorthy-ocean/PyESPER\n"
    "    cd PyESPER\n"
    "    pip install -e .\n"
    "\n"
    "The editable install also lets PyESPER locate its own data directories "
    "(`Mat_fullgrid/` and the top-level `NeuralNetworks/`) automatically, and brings "
    "in its runtime dependencies (numpy, scipy, pandas, numba, PyCO2SYS, seawater). "
    "Alternatively, leave it uninstalled and point `source['path']` at the checkout."
)


def _ensure_pyesper(path=None):
    """Import PyESPER's xarray methods, optionally making ``path`` importable first.

    ``path``, when given, is a PyESPER repository checkout (holding ``Mat_fullgrid/``
    and the top-level ``NeuralNetworks/`` package); it is inserted on ``sys.path`` so
    both packages import from there. When omitted, PyESPER must already be importable
    from the environment -- the recommended setup is an editable install
    (``pip install -e <checkout>``), which also lets PyESPER locate its own data
    directories automatically (``PyESPER.paths.data_root``).

    The ``sys.path`` insertion -- and any ``PyESPER`` module the failed attempt
    cached in ``sys.modules`` -- are rolled back if this raises: a failed attempt
    (no PyESPER importable there, or an upstream-not-fork PyESPER) must not leave
    a stale entry for a later, corrected call in the same process to trip over.
    """
    path = str(path or "")
    inserted = bool(path) and path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    success = False
    try:
        try:
            import PyESPER
        except ImportError as exc:
            where = f" (nothing importable at `source['path']`: {path})" if path else ""
            raise ImportError(
                _PYESPER_INSTALL_HINT.format(
                    problem=f"No PyESPER is importable in this environment{where}."
                )
            ) from exc
        try:
            from PyESPER import lir_xr, mixed_xr, nn_xr
        except ImportError as exc:
            raise ImportError(
                _PYESPER_INSTALL_HINT.format(
                    problem=(
                        "A PyESPER was found at "
                        f"{getattr(PyESPER, '__file__', 'an unknown location')}, but it has "
                        "no `lir_xr`/`nn_xr`/`mixed_xr` -- so it is upstream PyESPER, not "
                        "the CWorthy fork."
                    )
                )
            ) from exc
        success = True
        return {"lir": lir_xr, "nn": nn_xr, "mixed": mixed_xr}
    finally:
        if inserted and not success:
            sys.path.remove(path)
            # `import PyESPER` above may have cached a wrong/upstream module in
            # sys.modules; without evicting it a retry with a corrected
            # `source['path']` in the same process would silently reuse it and
            # fail again with the same message.
            for name in [
                m for m in sys.modules if m == "PyESPER" or m.startswith("PyESPER.")
            ]:
                del sys.modules[name]


def _time_dim(da: xr.DataArray) -> str | None:
    """Name of ``da``'s time dimension, or None when it has no usable one.

    Prefers a dim carrying a datetime64 coordinate -- what the boundary path
    hands us, since ``BoundaryForcing._process_bgc_esper`` renames ``abs_time``
    to ``time`` precisely so the datetime view *is* the dim -- and falls back to
    a conventional name for a bare dim with no coordinate. Initial conditions
    have no time dim at all (one instant, one level-set), so this returns None
    there and :func:`_pyesper_chunk_plan` keeps its spatial-split behaviour.
    """
    for dim in da.dims:
        coord = da.coords.get(dim)
        if coord is not None and np.issubdtype(coord.dtype, np.datetime64):
            return str(dim)
    for name in _TIME_DIM_NAMES:
        if name in da.dims:
            return name
    return None


def _month_aligned_time_chunks(
    da: xr.DataArray, time_dim: str, max_steps: int
) -> tuple[int, ...] | int:
    """Chunk lengths along ``time_dim``: exactly one block per calendar month, or
    a uniform ``max_steps`` when the coordinate isn't datetimes (or one month
    alone would exceed ``max_steps``, e.g. sub-daily forcing on a large grid).

    One block per month -- never several months bundled into one, even when they
    would fit ``max_steps`` -- because a block is recomputed once per output file
    it feeds. ``xarray.save_mfdataset(compute=True)`` issues a *separate*
    ``dask.compute`` per file (``writes = [w.sync(compute=compute) for w in
    writers]``), so nothing is shared between files: a block spanning N monthly
    files is computed in full N times. Measured on the 12-month Pacific boundary
    axis (367 daily steps, 14 monthly files), counting the time steps each file's
    graph forces:

        one block per month      2.8x the useful work
        two months per block     4.8x
        time collapsed to one    14.0x   (every block spans every file)

    Undersized blocks only cost PyESPER's fixed per-call setup (~2.4 s measured),
    which is cheap next to recomputing whole months; oversized ones cost real
    duplicated estimation. So this errs small. For data coarse enough that
    ``group_dataset`` writes yearly rather than monthly files, monthly blocks are
    finer than the file partition -- a few extra calls, still no recompute, since
    each block feeds exactly one file.
    """
    coord = da.coords.get(time_dim)
    if coord is None or not np.issubdtype(coord.dtype, np.datetime64):
        return max_steps
    index = coord.to_index()
    # Run lengths of consecutive (year, month). `group_by_month` groups by the
    # (year, month) *value*; for the monotonic time axes ROMS forcing carries
    # that is the same partition, and run lengths are what `.chunk()` needs --
    # dask blocks have to be contiguous.
    months = [
        sum(1 for _ in group)
        for _, group in itertools.groupby(zip(index.year, index.month, strict=True))
    ]
    if not months or max(months) > max_steps:
        return max_steps
    return tuple(months)


def _pyesper_chunk_plan(
    da: xr.DataArray, max_points: int | None = None
) -> dict[str, int | tuple[int, ...]]:
    """Compute a ``max_points``-ish chunk plan from ``da``'s own shape, to apply
    uniformly to every PyESPER input (see :func:`_pyesper_point_budget` for where
    the ceiling comes from and why matching it matters) -- one plan, not each
    input rechunked independently, so they stay consistently blocked with each
    other. Apply it with
    :func:`_apply_chunk_plan`, which drops the entries a given input can't take.

    Cuts along **time** whenever there is a time axis to cut (see
    :func:`_time_dim`), leaving the spatial dims whole. That is a memory choice,
    not a chunk-count one. Cutting a boundary slab spatially leaves every chunk
    needing the *entire* time range of the upstream regrid behind it, so a
    year-long run has to hold a year-long GLORYS slab per chunk and nothing can
    stream -- a 12-month, 100-level Pacific boundary run (367 daily steps,
    xi_rho 1858) OOM-killed a 251 GB machine that way, while the physics write
    of the very same boundaries, which keeps its natural per-time chunking,
    streamed fine in 75 minutes. Cutting along time instead bounds each chunk's
    upstream to that chunk's own time span.

    Falls back to the original "collapse everything, split the single largest
    dim" behaviour when there is no time axis -- initial conditions, a single
    instant, where a spatial split is the only option and implies no upstream
    time range anyway.

    ``max_points`` is the per-chunk ceiling, normally
    :func:`_pyesper_point_budget` for the request being made; it defaults to
    :data:`_MAX_POINTS_PER_CHUNK` so the plan can be exercised on its own.
    """
    cap = _MAX_POINTS_PER_CHUNK if max_points is None else max(int(max_points), 1)
    if da.size <= cap:
        return {d: -1 for d in da.dims}

    plan: dict[str, int | tuple[int, ...]] = {d: -1 for d in da.dims}
    time_dim = _time_dim(da)

    if time_dim is not None and da.sizes[time_dim] > 1:
        per_step = max(1, da.size // da.sizes[time_dim])
        if per_step <= cap:
            # The ordinary case: one time step's spatial slab fits the budget, so
            # time is the only axis that needs cutting.
            plan[time_dim] = _month_aligned_time_chunks(
                da, time_dim, max(1, cap // per_step)
            )
            return plan
        # A single time step already busts the budget (a very large grid): take
        # time down to one step and split the largest spatial dim for the rest.
        # Still time-cut, so still one step of upstream per chunk.
        plan[time_dim] = 1
        spatial = [d for d in da.dims if d != time_dim]
        if spatial:
            chunk_dim = max(spatial, key=lambda d: da.sizes[d])
            per_slice = max(1, per_step // da.sizes[chunk_dim])
            plan[chunk_dim] = max(1, cap // per_slice)
        return plan

    # No time axis: collapse every dim to one chunk, then split the largest dim
    # just enough to bring each block down to ~`cap` points -- one big axis is
    # enough to cap the block count without fragmenting them all.
    chunk_dim = max(da.dims, key=lambda d: da.sizes[d])
    per_slice = max(1, da.size // da.sizes[chunk_dim])
    plan[chunk_dim] = max(1, cap // per_slice)
    return plan


def _apply_chunk_plan(
    da: xr.DataArray, plan: dict[str, int | tuple[int, ...]]
) -> xr.DataArray:
    """Apply as much of ``plan`` as ``da`` can take, and return the result.

    ``DataArray.chunk`` raises on a mapping key that isn't one of the array's own
    dims, so a plan derived from ``temp`` has to be filtered per input: 2D
    ``lon``/``lat`` against a 3D ``temp``, and a boundary ``depth`` that need not
    carry the time dim at all. A tuple entry additionally has to sum to that
    input's own length along the dim; where it doesn't, fall back to a single
    chunk for that dim rather than raising -- the inputs are broadcast against
    each other downstream regardless.
    """
    filtered: dict[str, int | tuple[int, ...]] = {}
    for dim, spec in plan.items():
        if dim not in da.dims:
            continue
        if isinstance(spec, tuple) and sum(spec) != da.sizes[dim]:
            spec = -1
        filtered[dim] = spec
    return da.chunk(filtered) if filtered else da


def _decimal_year(time_da: xr.DataArray) -> xr.DataArray:
    """Convert a datetime64 DataArray to decimal years (for PyESPER ``EstDates``)."""
    year = time_da.dt.year
    doy = time_da.dt.dayofyear
    return (year + (doy - 1) / 365.25).astype("float64")


def validate_esper_source(source: dict) -> None:
    """Validate an ESPER ``source``/``bgc_source`` dict (raises ``ValueError``).

    Also checks that the required PyESPER is importable. This runs from
    ``_input_checks``, so a missing or wrong PyESPER is reported before the grid and
    physics regrid are built rather than partway through construction -- on a
    production grid that is the difference between failing immediately and failing
    minutes in.
    """
    method = str(source.get("method", "nn")).lower()
    if method not in _VALID_METHODS:
        raise ValueError(
            f"ESPER source 'method' must be one of {list(_VALID_METHODS)}, got {method!r}."
        )
    equation = source.get("equation", 8)
    if equation not in (8, 16):
        raise ValueError(
            "ESPER source 'equation' must be 8 (salinity+temperature) or 16 "
            f"(salinity only), got {equation!r}."
        )
    _ensure_pyesper(source.get("path"))


def estimate_bgc_fields(
    temp: xr.DataArray,
    salt: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    depth: xr.DataArray,
    *,
    source: dict,
    roms_variables=ESPER_SUPPORTED_VARS,
    est_dates=None,
) -> dict[str, xr.DataArray]:
    """Derive MARBL BGC tracers from physics T/S on the ROMS grid via PyESPER.

    Parameters
    ----------
    temp, salt : xarray.DataArray
        In-situ temperature (°C) and practical salinity on the ROMS grid (may be
        dask-backed; results are then lazy -- see Returns).
    lon, lat : xarray.DataArray
        Longitude (°E) and latitude (°N) of the same points (broadcastable to ``temp``).
    depth : xarray.DataArray
        Depth of the points (metres); sign-agnostic (absolute value is used).
    source : dict
        The ESPER source dict; uses ``path`` (optional -- see below), ``method``
        (default ``"nn"``), ``equation`` (default 8). ``path`` points at a PyESPER
        repository checkout; omit it when PyESPER is installed in the environment
        (``pip install -e <checkout>``), in which case PyESPER finds its own data.
    roms_variables : sequence of str
        ROMS/MARBL tracer names to derive (subset of :data:`ESPER_SUPPORTED_VARS`).
    est_dates : float or xarray.DataArray, optional
        Decimal year(s); only affects DIC. Defaults to 2002.0.

    Returns
    -------
    dict[str, xarray.DataArray]
        ``{roms_name: DataArray}`` in mmol/m³, dims/coords matching the broadcast
        inputs, lazy when the inputs are -- not materialised here. Safe to compute
        under any dask scheduler; see this module's docstring.
    """
    validate_esper_source(source)
    method = str(source.get("method", "nn")).lower()
    equation = source.get("equation", 8)
    path = source.get("path") or ""

    unknown = [v for v in roms_variables if v not in ROMS_TO_ESPER]
    if unknown:
        raise ValueError(
            f"ESPER cannot derive {unknown}; supported: {list(ESPER_SUPPORTED_VARS)}."
        )
    esper_vars = [ROMS_TO_ESPER[v] for v in roms_variables]

    xr_methods = _ensure_pyesper(path)
    estimate = xr_methods[method]

    # ESPER expects depth positive-downward in metres; ROMS layer depths are negative.
    depth_pos = np.abs(depth)

    # Defensive rechunk before the PyESPER call -- see _pyesper_point_budget's
    # docstring. temp/salt carry whatever chunking the upstream regrid pipeline
    # left them with (often many small chunks); one plan derived from temp's own
    # shape and time axis is applied to every input so they stay consistently
    # blocked. The ceiling comes from PyESPER's own budget for *this* method and
    # variable count, so our blocks land inside it and PyESPER does not re-block
    # them behind our back.
    if hasattr(temp.data, "chunks"):
        chunk_plan = _pyesper_chunk_plan(
            temp, _pyesper_point_budget(method, len(esper_vars))
        )
        salt = _apply_chunk_plan(salt, chunk_plan)
        temp = _apply_chunk_plan(temp, chunk_plan)
        lon = _apply_chunk_plan(lon, chunk_plan) if hasattr(lon.data, "chunks") else lon
        lat = _apply_chunk_plan(lat, chunk_plan) if hasattr(lat.data, "chunks") else lat
        depth_pos = (
            _apply_chunk_plan(depth_pos, chunk_plan)
            if hasattr(depth_pos.data, "chunks")
            else depth_pos
        )

    est = estimate(
        salt,
        temp,
        lon,
        lat,
        depth_pos,
        variables=esper_vars,
        path=str(path),
        equation=equation,
        est_dates=est_dates,
    )

    # Results are deliberately left lazy: PyESPER's own kernel lock keeps its
    # chunks one-at-a-time internally, so there is no reason to materialise them
    # here. They join whatever single `dask.compute()`/`.store()` call the caller
    # eventually makes at write time, which also lets every requested variable
    # share this call's per-chunk upstream work (they are all one graph).

    # µmol/kg -> mmol/m³ via in-situ density (TEOS-10 gsw.rho with pressure from
    # depth), matching the GLODAP/WOA adapters' convention; then clamp
    # physically non-negative tracers at 0.
    density = compute_in_situ_density(temp, salt, depth_pos, lat)
    factor = density / 1000.0
    d_meta = get_variable_metadata()

    out: dict[str, xr.DataArray] = {}
    for roms_name in roms_variables:
        da = est[ROMS_TO_ESPER[roms_name]] * factor
        da = da.clip(min=0.0)
        meta = d_meta.get(roms_name, {})
        if "long_name" in meta:
            da.attrs["long_name"] = meta["long_name"]
        if "units" in meta:
            da.attrs["units"] = meta["units"]
        out[roms_name] = da
    return out
