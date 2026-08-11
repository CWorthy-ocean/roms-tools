"""Shared helpers for river dataset tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def _write_river_file(
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
    """Write a minimal synthetic river dataset NetCDF file for tests."""
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
    """Write a minimal synthetic GloFAS-format NetCDF file for tests.

    Time should be datetime64 values, as GloFAS uses CF-compliant datetime encoding.
    """
    _write_river_file(path, lats, lons, flow, river_names, times, ratio=ratio, vol=vol)


def write_dai_file(
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
    """Write a minimal synthetic Dai & Trenberth-format NetCDF file for tests.

    Time should be numeric YYYYMM integer values (e.g. 199801 for January 1998),
    matching the format expected by DaiRiverDataset.add_time_info.
    """
    _write_river_file(path, lats, lons, flow, river_names, times, ratio=ratio, vol=vol)
