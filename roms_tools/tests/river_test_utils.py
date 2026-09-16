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


def write_glofas_file_with_rivr2o(
    path: str | Path,
    lats: np.ndarray,
    lons: np.ndarray,
    flow: np.ndarray,
    river_names: list[str],
    times: np.ndarray,
    *,
    years: np.ndarray,
    rivr2o_concentrations: dict[str, np.ndarray],
    ratio: np.ndarray | None = None,
    vol: np.ndarray | None = None,
) -> None:
    """Write a synthetic GloFAS file enriched with per-station RIVR2O
    concentrations, mimicking ``glofas_v4_rivers_daily_w_rivr2o.nc``.

    ``rivr2o_concentrations`` maps tracer name (e.g. ``"DIC"``) to an array
    of shape ``(len(years), n_stations)`` -- the "total_discharge" mode's
    ``Rivr2oRiverBGCDataset.extract_station_concentrations`` reads these
    directly, on a ``(year, station)`` dims, independent of the file's
    ``FLOW`` time axis.
    """
    _write_river_file(path, lats, lons, flow, river_names, times, ratio=ratio, vol=vol)
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    for tracer_name, values in rivr2o_concentrations.items():
        ds[tracer_name] = (["year", "station"], np.asarray(values, dtype=np.float64))
    ds = ds.assign_coords(year=np.asarray(years))
    ds.to_netcdf(path)


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
