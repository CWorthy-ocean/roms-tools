"""Adapter for deriving BGC tracers from physics T/S via PyESPER (the ESPER source).

Unlike the gridded BGC sources (CESM/UNIFIED/GLODAP), which regrid a dataset onto the
ROMS grid, the ``ESPER`` source *derives* biogeochemical variables from the physics
temperature/salinity that are already on the ROMS grid, using the PyESPER empirical
routines (https://github.com/LarissaMDias/PyESPER). It is therefore handled like the
``constants`` source: no dataset load, no lateral/vertical regridding.

The heavy lifting -- dask-lazy, chunked, uncertainty-free estimation -- lives in PyESPER's
``lir_xr``/``nn_xr``/``mixed_xr`` (a fork addition). This module only:

* lazily imports PyESPER (an optional dependency) and puts its directory on ``sys.path`` so
  the bundled ``NeuralNetworks`` package is importable for the NN methods;
* maps ROMS/MARBL tracer names to/from ESPER names;
* converts ESPER's µmol/kg output to MARBL's mmol/m³ using potential density;
* clamps physically non-negative tracers at 0.

The returned DataArrays are lazy (dask) when the inputs are, same as any other
lazy result in a forcing dataset -- callers materialise them (along with
everything else in the dataset) at write time.

Concurrency and memory: this depends on WHICH PyESPER is at ``source['path']``.

* Current PyESPER (the fused-kernel rework) serialises its own kernel entry with a
  module-level lock (``PyESPER.concurrency.kernel_lock``), budgets its own per-chunk
  memory internally, and no longer uses BLAS on the hot path -- so its chunks are
  safe to compute under any ambient dask scheduler, threaded included. Detected via
  :func:`pyesper_self_serializing`; when detected, ``HIGH_MEMORY_METHOD`` on the
  source classes reports ``False`` and everything runs under the ordinary
  concurrent scheduler.
* Legacy PyESPER (no ``PyESPER.concurrency``) has a ~10 KB/point per-chunk cost
  that multiplies by the dask worker count (confirmed via a kernel OOM-kill log to
  exhaust a 251 GB machine) and a numba kernel that deadlocks under concurrent
  entry. For that case the caller-side mitigation still applies -- see
  ``InitialConditionsSource.HIGH_MEMORY_METHOD`` /
  ``BoundaryForcingSource.HIGH_MEMORY_METHOD`` and
  :func:`roms_tools.utils.save_datasets`'s ``serialize_dask``.
"""

from __future__ import annotations

import functools
import importlib
import sys

import numpy as np
import xarray as xr

from roms_tools.setup.utils import compute_potential_density, get_variable_metadata

# Target ESPER chunk size. PyESPER's estimation chunks are effectively serial
# either way -- current PyESPER holds its own kernel lock so only one chunk is
# ever inside the kernels (see `pyesper_self_serializing`), and legacy PyESPER is
# run under a serialized scheduler by the caller -- so many small chunks buy no
# ESPER-side parallelism, they just repeat the fixed per-chunk overhead
# (defaults/iterations/polygon lookup); observed pre-fix as thousands of chunks
# each doing a few seconds of real work on a 90M-point domain. This targets a
# chunk *count* in the low tens instead (split along one outer dimension only:
# PyESPER's own "auto" rechunk would fragment every dimension to hit its byte
# budget, multiplying chunk count far more than a naive divide suggests).
#
# Under the ambient threaded scheduler (current PyESPER), low-tens chunks also
# pipeline well: while one worker holds PyESPER's kernel lock, the others run
# the physics/regrid/write tasks. Current PyESPER only ever *shrinks* chunks
# (its internal budget is larger than this), so this constant is the effective
# ESPER chunk size in practice.
_MAX_POINTS_PER_CHUNK = 6_000_000

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


def _ensure_pyesper(path):
    """Import PyESPER's xarray methods, making ``path`` importable first.

    ``path`` is the PyESPER directory (holding ``Mat_fullgrid/`` and the top-level
    ``NeuralNetworks/`` package). Adding it to ``sys.path`` makes both ``PyESPER`` and
    ``NeuralNetworks`` importable regardless of whether PyESPER was pip-installed.
    """
    path = str(path)
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        from PyESPER import lir_xr, mixed_xr, nn_xr
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "The ESPER BGC source requires the PyESPER package (with the dask-native "
            "`*_xr` methods) and its runtime deps (numpy, scipy, pandas, numba, "
            "PyCO2SYS, seawater). Install PyESPER and point `source['path']` at its "
            "directory (containing Mat_fullgrid/ and NeuralNetworks/)."
        ) from exc
    return {"lir": lir_xr, "nn": nn_xr, "mixed": mixed_xr}


@functools.lru_cache(maxsize=None)
def pyesper_self_serializing(path: str) -> bool:
    """True if the PyESPER at ``path`` serialises its own kernel entry.

    PyESPER's fused-kernel rework ships ``PyESPER.concurrency`` with a module-level
    ``kernel_lock``: exactly one dask chunk is ever inside its numba kernels no
    matter which scheduler the caller uses, per-chunk memory is budgeted internally
    (~0.5-1.2 KB/point instead of the legacy ~10 KB/point), and the hot path no
    longer calls BLAS. With such a PyESPER, the caller-side serialized regime
    (``HIGH_MEMORY_METHOD`` -> ``serialize_dask``) is unnecessary -- and harmful,
    since it also serialises every non-ESPER task sharing the graph.

    Detection is by importing ``PyESPER.concurrency`` from ``path`` (same sys.path
    mechanism as :func:`_ensure_pyesper`) and checking for ``kernel_lock``. Any
    failure -> ``False``, i.e. the conservative legacy treatment. Cached per path
    string: the answer cannot change within a process (imports are sticky).
    """
    try:
        path = str(path)
        if path and path not in sys.path:
            sys.path.insert(0, path)
        concurrency = importlib.import_module("PyESPER.concurrency")
    except Exception:
        return False
    return callable(getattr(concurrency, "kernel_lock", None))


def _pyesper_chunk_plan(da: xr.DataArray) -> dict[str, int]:
    """Compute a large, ``_MAX_POINTS_PER_CHUNK``-ish chunk plan from ``da``'s own
    shape, to apply uniformly to every PyESPER input (see that constant's
    docstring for why) -- one plan, not each input rechunked independently, so
    they stay consistently blocked with each other (``DataArray.chunk`` ignores
    any dim in the mapping it doesn't have, so this is safe to apply even to an
    input with fewer dims, e.g. a 2D ``lon``/``lat`` against 3D ``temp``).
    """
    if da.size <= _MAX_POINTS_PER_CHUNK:
        return {d: -1 for d in da.dims}
    # Collapse every dim to one chunk, then split the largest dim just enough to
    # bring each block down to ~_MAX_POINTS_PER_CHUNK points -- one big axis to
    # chunk along is enough to cap block count without fragmenting every dim.
    chunk_dim = max(da.dims, key=lambda d: da.sizes[d])
    per_slice = max(1, da.size // da.sizes[chunk_dim])
    target_len = max(1, _MAX_POINTS_PER_CHUNK // per_slice)
    chunks = {d: -1 for d in da.dims}
    chunks[chunk_dim] = target_len
    return chunks


def _decimal_year(time_da: xr.DataArray) -> xr.DataArray:
    """Convert a datetime64 DataArray to decimal years (for PyESPER ``EstDates``)."""
    year = time_da.dt.year
    doy = time_da.dt.dayofyear
    return (year + (doy - 1) / 365.25).astype("float64")


def validate_esper_source(source: dict) -> None:
    """Validate an ESPER ``source``/``bgc_source`` dict (raises ``ValueError``)."""
    if "path" not in source:
        raise ValueError(
            "An ESPER BGC source requires a 'path' to the PyESPER directory "
            "(containing Mat_fullgrid/ and NeuralNetworks/)."
        )
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
        The ESPER source dict; uses ``path`` (required), ``method`` (default ``"nn"``),
        ``equation`` (default 8).
    roms_variables : sequence of str
        ROMS/MARBL tracer names to derive (subset of :data:`ESPER_SUPPORTED_VARS`).
    est_dates : float or xarray.DataArray, optional
        Decimal year(s); only affects DIC. Defaults to 2002.0.

    Returns
    -------
    dict[str, xarray.DataArray]
        ``{roms_name: DataArray}`` in mmol/m³, dims/coords matching the broadcast
        inputs, lazy when the inputs are -- not materialised here. See this
        module's own docstring for the oversubscription hazard this implies for
        callers computing several chunks concurrently, and how it's mitigated
        (at the caller level, not inside this function).
    """
    validate_esper_source(source)
    method = str(source.get("method", "nn")).lower()
    equation = source.get("equation", 8)
    path = source["path"]

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

    # Defensive rechunk before the PyESPER call -- see _MAX_POINTS_PER_CHUNK's
    # docstring. temp/salt carry whatever chunking the upstream regrid pipeline
    # left them with (often many small chunks); one plan derived from temp's own
    # shape is applied to every input so they stay consistently blocked.
    if hasattr(temp.data, "chunks"):
        chunk_plan = _pyesper_chunk_plan(temp)
        salt = salt.chunk(chunk_plan)
        temp = temp.chunk(chunk_plan)
        lon = lon.chunk(chunk_plan) if hasattr(lon.data, "chunks") else lon
        lat = lat.chunk(chunk_plan) if hasattr(lat.data, "chunks") else lat
        depth_pos = (
            depth_pos.chunk(chunk_plan)
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

    # `est`'s DataArrays are left lazy here, same as any other dask-backed
    # result -- deliberately NOT forced to materialise eagerly. An earlier
    # version of this function did force eager materialisation (one
    # `dask.compute()` per call, covering every requested variable together),
    # to work around PyESPER's neural-net path needing dask chunks run one at a
    # time rather than concurrently (see PyESPER/run_nets.py's own docstring
    # for the mechanism). That mitigation is no longer applied here: splitting
    # available cores between dask's own worker count and each worker's
    # internal BLAS/numba thread count (see cstar-forge's `input_data.py`)
    # avoids the same oversubscription hazard while keeping cross-chunk
    # parallelism, so ESPER's results can go back to being computed the same
    # way as every other lazy result -- together with the rest of the forcing
    # dataset, in whatever single `dask.compute()`/`.store()` call the caller
    # eventually does at write time (naturally sharing this call's own shared
    # per-chunk upstream work across every requested variable, since they're
    # all part of the same graph passed to that one call).

    # µmol/kg -> mmol/m³ via potential density (sigma0 + 1000), matching the GLODAP
    # adapter's convention; then clamp physically non-negative tracers at 0.
    density = compute_potential_density(temp, salt) + 1000.0
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
