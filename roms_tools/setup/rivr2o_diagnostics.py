"""Notebook helpers to debug RIVR2O sampling and DIC concentration for ``RiverForcing``."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from roms_tools.datasets.river_datasets import (
    SECONDS_PER_YEAR,
    Rivr2oRiverBGCDataset,
    clamp_rivr2o_time,
    rivr2o_coerce_time,
)
from roms_tools.setup.utils import gc_dist

if TYPE_CHECKING:
    from roms_tools.setup.river_forcing import RiverForcing

_DEBUG_LOG = "/Users/ullaheede/roms-tools/.cursor/debug-406116.log"
_MOLAR_MASS_C = 12.011


def _agent_log(
    message: str,
    data: dict[str, Any],
    *,
    hypothesis_id: str,
    run_id: str = "notebook",
) -> None:
    # region agent log
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "sessionId": "406116",
                        "timestamp": int(time.time() * 1000),
                        "location": "rivr2o_diagnostics.py",
                        "message": message,
                        "data": data,
                        "hypothesisId": hypothesis_id,
                        "runId": run_id,
                    }
                )
                + "\n"
            )
    except OSError:
        pass
    # endregion


def rivr2o_export_to_mmol_m3(
    export: float | np.ndarray,
    river_volume_m3_s: float | np.ndarray,
    *,
    molar_mass_g: float = _MOLAR_MASS_C,
) -> float | np.ndarray:
    """Same concentration formula as ``RiverForcing._rivr2o_export_to_concentration``."""
    export = np.asarray(export, dtype=np.float64)
    river_volume_m3_s = np.asarray(river_volume_m3_s, dtype=np.float64)
    mass_flux_g_s = export * 1e6 / SECONDS_PER_YEAR
    mmol_flux = mass_flux_g_s / molar_mass_g * 1000.0
    return (mmol_flux / river_volume_m3_s).astype(np.float64)


def expected_mmol_m3_range(
    export_1e6_g_yr: float, river_volume_m3_s: float = 400.0
) -> float:
    """Hand-check concentration for one export value and discharge."""
    return float(rivr2o_export_to_mmol_m3(export_1e6_g_yr, river_volume_m3_s))


def _iceland_bbox_from_grid(river_forcing: RiverForcing) -> dict[str, float]:
    lon = river_forcing.grid.ds.lon_rho.values
    lat = river_forcing.grid.ds.lat_rho.values
    return {
        "lat_min": float(np.nanmin(lat)),
        "lat_max": float(np.nanmax(lat)),
        "lon_min": float(np.nanmin(lon)),
        "lon_max": float(np.nanmax(lon)),
    }


def _rivr2o_field_stats_in_bbox(
    bgc: Rivr2oRiverBGCDataset,
    bbox: dict[str, float],
    *,
    time_index: int = 0,
) -> dict[str, float]:
    lat_dim = bgc.dim_names["latitude"]
    lon_dim = bgc.dim_names["longitude"]
    time_dim = bgc.dim_names["time"]

    lats = bgc.ds[lat_dim]
    lons = bgc.ds[lon_dim]
    lat_mask = (lats >= bbox["lat_min"]) & (lats <= bbox["lat_max"])
    lon_min = bbox["lon_min"]
    lon_max = bbox["lon_max"]
    if float(lons.max()) <= 180 and lon_min > 180:
        lon_min, lon_max = lon_min - 360, lon_max - 360
    elif float(lons.max()) > 180 and lon_max < 0:
        lon_min, lon_max = lon_min + 360, lon_max + 360
    lon_mask = (lons >= lon_min) & (lons <= lon_max)

    sub = (
        bgc.ds.isel({time_dim: time_index})
        .where(lat_mask, drop=True)
        .where(lon_mask, drop=True)
    )
    dic = sub["DIC"].values
    valid = np.isfinite(dic) & (dic > 0)
    if not valid.any():
        return {"dic_min": np.nan, "dic_max": np.nan, "dic_mean": np.nan}
    return {
        "dic_min": float(np.min(dic[valid])),
        "dic_max": float(np.max(dic[valid])),
        "dic_mean": float(np.mean(dic[valid])),
    }


def _nearest_nonzero_cell_info(
    bgc: Rivr2oRiverBGCDataset,
    lon: float,
    lat: float,
    *,
    straddle: bool,
    time: datetime | np.datetime64 | None = None,
) -> dict[str, float | int | None]:
    """Mirror ``Rivr2oRiverBGCDataset.sample_at_points`` cell selection."""
    lat_dim = bgc.dim_names["latitude"]
    lon_dim = bgc.dim_names["longitude"]

    query_lon = bgc._adjust_lon_to_grid(np.array([lon]), straddle=straddle)[0]
    time_index = bgc._nearest_time_index(time) if time is not None else None
    lat_indices, lon_indices, grid_lats, grid_lons = bgc._valid_export_cell_indices(
        time_index=time_index
    )
    dist_m = gc_dist(query_lon, lat, grid_lons, grid_lats)
    nearest = int(np.argmin(dist_m))
    return {
        "query_lon": float(query_lon),
        "query_lat": float(lat),
        "cell_lat": float(bgc.ds[lat_dim].values[lat_indices[nearest]]),
        "cell_lon": float(bgc.ds[lon_dim].values[lon_indices[nearest]]),
        "dist_m": float(dist_m[nearest]),
        "dist_km": float(dist_m[nearest]) / 1000.0,
        "lat_index": int(lat_indices[nearest]),
        "lon_index": int(lon_indices[nearest]),
        "rivr2o_time_index": time_index,
    }


def _interp_at_mouth(
    bgc: Rivr2oRiverBGCDataset,
    lon: float,
    lat: float,
    *,
    straddle: bool,
    time: datetime | np.datetime64,
) -> dict[str, float]:
    """Plain xarray nearest interp at mouth (may hit zero/fill cells)."""
    lon_dim = bgc.dim_names["longitude"]
    query_lon = float(bgc._adjust_lon_to_grid(np.array([lon]), straddle=straddle)[0])
    time_val = np.datetime64(time)
    point = bgc.ds.interp(
        {
            lon_dim: query_lon,
            bgc.dim_names["latitude"]: lat,
            bgc.dim_names["time"]: time_val,
        },
        method="nearest",
    )
    return {
        "interp_dic": float(point["DIC"].values),
        "interp_doc_l": float(point["DOC_l"].values),
    }


def diagnose_rivr2o_river_forcing(
    river_forcing: RiverForcing,
    *,
    time_index: int = 0,
    iceland_bbox: dict[str, float] | None = None,
    reference_q_m3_s: float = 400.0,
    write_debug_log: bool = True,
    run_id: str = "notebook",
) -> pd.DataFrame:
    """Build a per-river table comparing RIVR2O exports, Q, and DIC concentrations.

    Run in a notebook after constructing ``river_forcing`` with ``bgc_source`` RIVR2O.

    Parameters
    ----------
    river_forcing
        Constructed ``RiverForcing`` with ``include_bgc=True`` and RIVR2O ``bgc_source``.
    time_index
        Index along ``river_time`` / ``abs_time`` for snapshots (default first step).
    iceland_bbox
        Optional ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max`` for field stats.
        Defaults to the ROMS grid extent.
    reference_q_m3_s
        Discharge used for the hand-check columns at 10k and 50k export (default 400).
    write_debug_log
        Append one NDJSON line per river to the debug log (session 406116).
    run_id
        Tag written to the debug log.

    Returns
    -------
    pandas.DataFrame
        One row per river with sampled exports, volumes, and concentration checks.
    """
    if (
        river_forcing.bgc_source is None
        or river_forcing.bgc_source.get("name") != "RIVR2O"
    ):
        raise ValueError("river_forcing must use bgc_source name 'RIVR2O'.")

    ds = river_forcing.ds
    bgc = river_forcing._get_bgc_data()
    river_names = [str(n) for n in ds.river_name.values]
    mouth_lons, mouth_lats = river_forcing._get_river_sample_coords(river_names)

    abs_time = ds["abs_time"].isel(river_time=time_index).values
    abs_time_scalar = pd.Timestamp(abs_time).to_pydatetime()
    anchor_time = rivr2o_coerce_time(
        clamp_rivr2o_time(ds["abs_time"].isel(river_time=time_index)).values
    )
    rivr2o_time_index = bgc._nearest_time_index(anchor_time)
    rivr2o_time_used = pd.Timestamp(
        bgc.ds[bgc.dim_names["time"]].values[rivr2o_time_index]
    )

    sampled = bgc.sample_at_points(
        lon=mouth_lons,
        lat=mouth_lats,
        straddle=river_forcing.grid.straddle,
        time=anchor_time,
    )

    bbox = iceland_bbox or _iceland_bbox_from_grid(river_forcing)
    field_stats = _rivr2o_field_stats_in_bbox(bgc, bbox, time_index=rivr2o_time_index)

    dic_idx = int(np.where(ds.tracer_name.values == "DIC")[0][0])
    rows: list[dict[str, Any]] = []

    for i, name in enumerate(river_names):
        cell = _nearest_nonzero_cell_info(
            bgc,
            float(mouth_lons[i]),
            float(mouth_lats[i]),
            straddle=river_forcing.grid.straddle,
            time=anchor_time,
        )
        interp = _interp_at_mouth(
            bgc,
            float(mouth_lons[i]),
            float(mouth_lats[i]),
            straddle=river_forcing.grid.straddle,
            time=abs_time_scalar,
        )

        target_time = clamp_rivr2o_time(ds["abs_time"].isel(river_time=time_index))
        export_dic = float(
            sampled["DIC"]
            .interp(time=target_time, method="nearest")
            .isel(points=i)
            .values
        )
        export_doc_l = float(
            sampled["DOC_l"]
            .interp(time=target_time, method="nearest")
            .isel(points=i)
            .values
        )
        q = float(ds["river_volume"].isel(nriver=i, river_time=time_index).values)
        c_dic_file = float(rivr2o_export_to_mmol_m3(export_dic, q))
        c_doc_l = float(rivr2o_export_to_mmol_m3(export_doc_l, q))
        c_dic_forcing_manual = c_dic_file + c_doc_l
        c_dic_actual = float(
            ds["river_tracer"]
            .isel(ntracers=dic_idx, nriver=i, river_time=time_index)
            .values
        )

        row = {
            "river": name,
            "mouth_lon": float(mouth_lons[i]),
            "mouth_lat": float(mouth_lats[i]),
            "sample_cell_lon": cell["cell_lon"],
            "sample_cell_lat": cell["cell_lat"],
            "dist_to_cell_km": cell["dist_km"],
            "rivr2o_time_used": str(rivr2o_time_used),
            "abs_time": str(abs_time_scalar),
            "DIC_export_1e6g_yr": export_dic,
            "DOC_l_export_1e6g_yr": export_doc_l,
            "interp_DIC_at_mouth": interp["interp_dic"],
            "river_volume_m3_s": q,
            "mmol_m3_from_DIC": c_dic_file,
            "mmol_m3_from_DIC_plus_DOC_l": c_dic_forcing_manual,
            "mmol_m3_river_tracer_DIC": c_dic_actual,
            "manual_matches_forcing": np.isclose(
                c_dic_forcing_manual, c_dic_actual, rtol=1e-4, atol=1e-3
            ),
            "expected_at_10k_export_Q400": expected_mmol_m3_range(
                10_000.0, reference_q_m3_s
            ),
            "expected_at_50k_export_Q400": expected_mmol_m3_range(
                50_000.0, reference_q_m3_s
            ),
            "bbox_dic_min": field_stats["dic_min"],
            "bbox_dic_max": field_stats["dic_max"],
        }
        rows.append(row)

        if write_debug_log:
            _agent_log(
                "rivr2o river diagnostic row",
                {
                    "river": name,
                    "export_dic": export_dic,
                    "export_doc_l": export_doc_l,
                    "river_volume_m3_s": q,
                    "c_dic_forcing_manual": c_dic_forcing_manual,
                    "c_dic_actual": c_dic_actual,
                    "cell_lon": cell["cell_lon"],
                    "cell_lat": cell["cell_lat"],
                    "dist_km": cell["dist_km"],
                    "interp_dic": interp["interp_dic"],
                    "bbox_dic_max": field_stats["dic_max"],
                },
                hypothesis_id="A" if export_dic < 1000 else "B",
                run_id=run_id,
            )

    df = pd.DataFrame(rows)
    print("RIVR2O field DIC in grid bbox (10^6 g C yr-1 per cell):")
    print(f"  min={field_stats['dic_min']:.4g}  max={field_stats['dic_max']:.4g}")
    print(
        f"Hand-check at Q={reference_q_m3_s} m3/s: "
        f"E=10000 -> {expected_mmol_m3_range(10_000, reference_q_m3_s):.2f} mmol/m3; "
        f"E=50000 -> {expected_mmol_m3_range(50_000, reference_q_m3_s):.2f} mmol/m3"
    )
    print(
        "DIC in river_forcing = mmol/m3 from (DIC + DOC_l) export / river_volume "
        "(see roms_tools/setup/river_forcing.py)."
    )
    print(f"RIVR2O anchor year for cell search: {rivr2o_time_used}")
    return df


def print_usage() -> None:
    """Print how to run diagnostics from a Jupyter notebook."""
    print(
        "rivr2o_diagnostics.py defines helpers only; running it does not analyze data.\n"
        "\n"
        "In your notebook (after building river_forcing), use either:\n"
        "\n"
        "  from roms_tools.setup.rivr2o_diagnostics import diagnose_rivr2o_river_forcing\n"
        "  diag = diagnose_rivr2o_river_forcing(river_forcing)\n"
        "  display(diag)\n"
        "\n"
        "Or keep names in the notebook namespace with:\n"
        "\n"
        "  %run -i path/to/rivr2o_diagnostics.py\n"
        "  diag = diagnose_rivr2o_river_forcing(river_forcing)\n"
        "\n"
        "Plain %run without -i removes functions after the cell finishes.\n"
        "\n"
        "Hand-check (E in 10^6 g C/yr, Q in m3/s) at Q=400:\n"
        f"  E=10000  -> {expected_mmol_m3_range(10_000, 400):.2f} mmol/m3\n"
        f"  E=50000  -> {expected_mmol_m3_range(50_000, 400):.2f} mmol/m3\n"
    )


if __name__ == "__main__":
    print_usage()
