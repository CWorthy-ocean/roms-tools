"""Shared helpers for tests that need a synthetic GLODAP source."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import xarray as xr


def write_synthetic_glodap(
    directory: Path,
    include_ts: bool = True,
    *,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    field_fn: Callable[[str, np.ndarray], np.ndarray] | None = None,
) -> Path:
    """A minimal GLODAP-shaped source: one file per variable, lat/lon/depth, NO time.

    GLODAP's real layout is exactly this -- ``{dir}/GLODAPv2.2016b.{file_var}.nc``, each
    holding one variable on a 1 degree grid with no time dimension -- so a synthetic one
    exercises the static-source path without needing the multi-GB download.

    ``include_ts`` writes temperature/salinity files alongside the six BGC tracers
    (uniform T=2 degC, S=35 PSU), exercising the real in-situ-density unit
    conversion; set it False to exercise the T/S-missing 1025 kg/m3 fallback path.

    By default the grid comfortably covers a ROMS grid centred at 0E/55N and every
    variable is uniform in space. ``lat``, ``lon`` and ``depth`` replace the grid, and
    ``field_fn(file_var, values)`` receives each variable's uniform
    ``(depth, lat, lon)`` float32 array and returns the array to write, so a test can
    shape the columns (shallow ones, NaN gaps, a NaN temperature at one cell, ...).
    """
    directory.mkdir(parents=True, exist_ok=True)
    if lat is None:
        lat = np.arange(35.0, 76.0, 1.0)
    if lon is None:
        lon = np.arange(-30.0, 31.0, 1.0)
    if depth is None:
        depth = np.array([0.0, 50.0, 200.0, 1000.0])
    variables = [
        ("TAlk", 2300.0),
        ("TCO2", 2100.0),
        ("PO4", 1.0),
        ("NO3", 15.0),
        ("silicate", 10.0),
        ("oxygen", 280.0),
    ]
    if include_ts:
        variables += [("temperature", 2.0), ("salinity", 35.0)]
    for file_var, value in variables:
        values = np.full((depth.size, lat.size, lon.size), value, dtype="f4")
        if field_fn is not None:
            values = field_fn(file_var, values)
        # Mirrors the real files: `Depth` is a 1-D *data variable* on an unlabelled
        # `depth_surface` dimension, which `GLODAPv2Dataset.clean_up` promotes to a
        # dimension coordinate. Writing it as a plain coord would not exercise that.
        xr.Dataset(
            {
                file_var: (("depth_surface", "lat", "lon"), values),
                "Depth": (("depth_surface",), depth),
            },
            coords={"lat": lat, "lon": lon},
        ).to_netcdf(directory / f"GLODAPv2.2016b.{file_var}.nc")
    return directory
