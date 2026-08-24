"""Shared helpers for CONUS404 / curvilinear dataset tests.

Builds a small synthetic CONUS404-shaped dataset analytically, rather than
committing a slice of the real store. Every property worth testing here -- the
radiation differencing and its float32 quantization, the wind rotation, the unit
conversions, the footprint-edge behavior -- is both easier and stricter to assert
against a field whose exact answer is known by construction. A real slice could
only assert "matches the last run".
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

# Magnitude the real ACSWDNB accumulations have reached by 2020. Adding it before
# the float32 cast reproduces the store's actual quantization: one float32 ULP at
# this magnitude is 16384 J/m^2, i.e. 4.55 W/m^2 over an hour.
RADIATION_ACCUM_OFFSET_J_M2 = 2.38e11

# CONUS404's native grid spacing, in metres.
GRID_SPACING_M = 4000.0

# Albedo used to derive the synthetic upwelling shortwave from the downwelling
# one, so that net shortwave has a known closed form.
ALBEDO = 0.1

METRES_PER_DEGREE = 111_320.0


def synthetic_conus404(
    ny: int = 40,
    nx: int = 50,
    ntime: int = 30,
    start: datetime = datetime(2020, 6, 14, 23),
    center_lon: float = -124.0,
    center_lat: float = 36.0,
    rot_deg: float = 15.0,
    sw_peak: float = 1000.0,
    lw_value: float = 350.0,
) -> tuple[xr.Dataset, dict[str, np.ndarray]]:
    """Build a synthetic CONUS404-shaped dataset plus the exact expected outputs.

    The horizontal grid is a regular ``GRID_SPACING_M`` patch in projected space,
    rotated by ``rot_deg`` before being mapped to latitude/longitude, so ``lat``
    and ``lon`` are genuinely 2D and the map rotation is a known constant.

    Parameters
    ----------
    ny, nx : int
        Grid size.
    ntime : int
        Number of hourly records. The first is consumed by the radiation
        differencing, so the processed dataset has ``ntime - 1``.
    start : datetime
        Time stamp of the first record.
    center_lon, center_lat : float
        Geographic centre of the patch.
    rot_deg : float
        Rotation of the projected grid relative to east/north, in degrees.
    sw_peak : float
        Peak downwelling shortwave flux, W/m^2.
    lw_value : float
        Constant downwelling longwave flux, W/m^2.

    Returns
    -------
    ds : xr.Dataset
        A dataset with CONUS404's variable names, units and dimension layout.
    expected : dict[str, np.ndarray]
        Exact expected values of the processed fields, on the *post-differencing*
        time axis (i.e. records ``1:``): ``swrad`` (net, W/m^2), ``lwrad``
        (W/m^2), ``Tair`` (degC), ``rain`` (cm/day), ``uwnd``/``vwnd``
        (earth-relative m/s), plus ``cos_alpha``/``sin_alpha``/``rot_deg``.
    """
    rot = np.deg2rad(rot_deg)
    cos_a, sin_a = np.cos(rot), np.sin(rot)

    # Projected coordinates, metres, origin at the patch centre.
    x = (np.arange(nx) - nx / 2) * GRID_SPACING_M
    y = (np.arange(ny) - ny / 2) * GRID_SPACING_M
    xx, yy = np.meshgrid(x, y)

    # Rotate the projected axes into east/north, then map to lat/lon on a local
    # tangent plane. This is what makes lat/lon 2D.
    east = xx * cos_a - yy * sin_a
    north = xx * sin_a + yy * cos_a
    lat = center_lat + north / METRES_PER_DEGREE
    lon = center_lon + east / (METRES_PER_DEGREE * np.cos(np.deg2rad(center_lat)))

    times = np.array(
        [np.datetime64(start, "ns") + np.timedelta64(h, "h") for h in range(ntime)]
    )
    hours = np.array([(start.hour + h) % 24 for h in range(ntime)], dtype=float)

    # --- radiation: choose the flux first, then integrate it --------------
    # A clean diurnal cycle, exactly zero at night so the test can assert that
    # night-time differences come back as exactly 0.0 (as they do on the real
    # store).
    diurnal = np.clip(np.sin(2 * np.pi * (hours - 6.0) / 24.0), 0.0, None)
    # Mild spatial structure so a constant field cannot mask an indexing bug.
    spatial = 1.0 + 0.1 * np.cos(np.deg2rad(lat) * 40.0)
    sw_down_true = sw_peak * diurnal[:, None, None] * spatial[None, :, :]
    sw_up_true = ALBEDO * sw_down_true
    lw_down_true = np.full_like(sw_down_true, lw_value) * spatial[None, :, :]

    def accumulate(flux_true: np.ndarray) -> np.ndarray:
        """Integrate a flux into a since-model-start accumulation.

        Built so that ``diff(accum)[k] / 3600 == flux_true[k + 1]``, matching how
        ``post_process`` recovers the flux (``diff`` labels at the interval end).
        The float32 cast happens once at the end, exactly as the store does it, so
        the quantization the test sees is the real one.
        """
        steps = np.concatenate(
            [np.zeros((1,) + flux_true.shape[1:]), flux_true[1:] * 3600.0]
        )
        return (RADIATION_ACCUM_OFFSET_J_M2 + np.cumsum(steps, axis=0)).astype(
            np.float32
        )

    acswdnb = accumulate(sw_down_true)
    acswupb = accumulate(sw_up_true)
    aclwdnb = accumulate(lw_down_true)

    # --- other fields -----------------------------------------------------
    t2 = 288.0 + 10.0 * diurnal[:, None, None] + 0.0 * spatial[None, :, :]
    t2 = np.broadcast_to(t2, sw_down_true.shape).copy()
    td2 = t2 - 5.0
    psfc = np.full_like(t2, 101_300.0)
    q2 = np.full_like(t2, 0.008)
    # mm accumulated over the prior hour, i.e. mm/hr
    prec = np.abs(np.sin(hours / 3.0))[:, None, None] * np.ones_like(t2)

    # Grid-relative winds, deliberately constant so the rotated answer is a
    # single pair of numbers.
    u10_grid = np.full_like(t2, 5.0)
    v10_grid = np.full_like(t2, 2.0)

    landmask = np.zeros((ny, nx), dtype=np.float32)
    landmask[:, : nx // 2] = 1.0  # left half land, right half water

    def _v(data, dims, **attrs):
        return xr.DataArray(data.astype(np.float32), dims=dims, attrs=attrs)

    ds = xr.Dataset(
        {
            "ACSWDNB": _v(acswdnb, ("time", "y", "x"), units="J m-2"),
            "ACSWUPB": _v(acswupb, ("time", "y", "x"), units="J m-2"),
            "ACLWDNB": _v(aclwdnb, ("time", "y", "x"), units="J m-2"),
            "T2": _v(t2, ("time", "y", "x"), units="K"),
            "TD2": _v(td2, ("time", "y", "x"), units="K"),
            "PSFC": _v(psfc, ("time", "y", "x"), units="Pa"),
            "Q2": _v(q2, ("time", "y", "x"), units="kg kg-1"),
            "PREC_ACC_NC": _v(prec, ("time", "y", "x"), units="mm"),
            "U10": _v(u10_grid, ("time", "y", "x"), units="m s-1"),
            "V10": _v(v10_grid, ("time", "y", "x"), units="m s-1"),
            "COSALPHA": _v(np.full((ny, nx), cos_a), ("y", "x")),
            "SINALPHA": _v(np.full((ny, nx), sin_a), ("y", "x")),
            "LANDMASK": _v(landmask, ("y", "x")),
        },
        coords={
            "lat": xr.DataArray(
                lat.astype(np.float32), dims=("y", "x"), attrs={"units": "degree_north"}
            ),
            "lon": xr.DataArray(
                lon.astype(np.float32), dims=("y", "x"), attrs={"units": "degree_east"}
            ),
            # Projected axes in metres, so `compute_minimal_grid_spacing` can read
            # the exact spacing off them.
            "x": xr.DataArray(x, dims="x", attrs={"units": "m"}),
            "y": xr.DataArray(y, dims="y", attrs={"units": "m"}),
            "time": times,
        },
    )
    ds["time"].encoding = {
        "units": "hours since 1979-10-01 00:00:00",
        "calendar": "proleptic_gregorian",
        "dtype": "int64",
    }

    expected = {
        "swrad": ((sw_down_true - sw_up_true)[1:]).astype(np.float64),
        "lwrad": (lw_down_true[1:]).astype(np.float64),
        "Tair": (t2[1:] - 273.15).astype(np.float64),
        "rain": (prec[1:] * 2.4).astype(np.float64),
        "uwnd": np.full_like(t2[1:], 5.0 * cos_a - 2.0 * sin_a, dtype=np.float64),
        "vwnd": np.full_like(t2[1:], 2.0 * cos_a + 5.0 * sin_a, dtype=np.float64),
        "cos_alpha": cos_a,
        "sin_alpha": sin_a,
        "rot_deg": rot_deg,
        "times": times,
        "lat": lat,
        "lon": lon,
    }
    return ds, expected


def write_conus404_store(
    path: str | Path, ds: xr.Dataset, fmt: str = "zarr"
) -> Path:
    """Write a synthetic dataset to disk as a zarr store or a NetCDF file.

    ``fmt="zarr"`` exercises the ``read_zarr`` code path (which requires dask);
    ``fmt="netcdf"`` allows the eager path to be tested too.
    """
    path = Path(path)
    if fmt == "zarr":
        store = path.with_suffix(".zarr")
        ds.to_zarr(store, mode="w", consolidated=True)
        return store
    store = path.with_suffix(".nc")
    ds.to_netcdf(store)
    return store
