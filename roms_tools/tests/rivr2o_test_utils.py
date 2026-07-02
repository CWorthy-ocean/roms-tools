"""Shared helpers for RIVR2O-related tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xarray as xr

from roms_tools.datasets.river_datasets import RIVR2O_FILL_VALUE


def write_rivr2o_file(
    path: str | Path,
    lat: np.ndarray,
    lon: np.ndarray,
    tracer_values: Mapping[str, np.ndarray],
) -> None:
    """Write a minimal synthetic RIVR2O annual NetCDF file for tests."""
    time = np.array([np.nan], dtype=np.float32)
    if "POC" not in tracer_values:
        tracer_values = {
            **tracer_values,
            "POC": np.full((len(lat), len(lon)), RIVR2O_FILL_VALUE),
        }
    data_vars = {
        name: (["time", "lat", "lon"], values[np.newaxis, :, :])
        for name, values in tracer_values.items()
    }
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"lat": lat, "lon": lon, "time": time},
    )
    ds.to_netcdf(path)
