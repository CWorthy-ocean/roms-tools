"""Shared helpers for GloFAS-related tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def write_glofas_file(
    path: str | Path,
    lats: np.ndarray,
    lons: np.ndarray,
    flow: np.ndarray,
    river_names: list[str],
    times: np.ndarray,
    *,
    ratio: np.ndarray | None = None,
    vol: np.ndarray | None = None,
) -> None:
    """Write a minimal synthetic GloFAS-format NetCDF file for tests."""
    n_stations = len(lats)
    if ratio is None:
        ratio = np.ones(n_stations, dtype=np.float32)
    data_vars = {
        "lat_mou": (["station"], lats),
        "lon_mou": (["station"], lons),
        "FLOW": (["time", "station"], flow),
        "ratio_m2s": (["station"], ratio),
        "riv_name": (["station"], river_names),
    }
    if vol is not None:
        data_vars["vol_stn"] = (["station"], vol)
    ds = xr.Dataset(
        data_vars,
        coords={"time": times, "station": np.arange(n_stations)},
    )
    ds.to_netcdf(path)
