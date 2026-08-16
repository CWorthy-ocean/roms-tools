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

The returned DataArrays are lazy (dask) when the inputs are, so callers can assign them to
a forcing dataset and only materialise them at write time.
"""

from __future__ import annotations

import sys

import numpy as np
import xarray as xr

from roms_tools.setup.utils import compute_potential_density, get_variable_metadata

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
        dask-backed; results are then lazy).
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
        ``{roms_name: DataArray}`` in mmol/m³, dims/coords matching the broadcast inputs,
        lazy when the inputs are.
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
