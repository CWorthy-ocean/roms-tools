from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from roms_tools import Grid, get_glorys_bounds

try:
    import copernicusmarine  # type: ignore
except ImportError:
    copernicusmarine = None


def download_regional_and_bigger(
    tmp_path: Path,
    grid: Grid,
    start_time: datetime,
    variables: list[str] = ["thetao", "so", "uo", "vo", "zos"],
) -> tuple[Path, Path]:
    """
    Helper: download minimal and slightly bigger GLORYS subsets.

    Parameters
    ----------
    tmp_path : Path
        Directory to store the downloaded NetCDF files.
    grid : Grid
        ROMS-Tools Grid object defining the target domain.
    start_time : datetime
        Start time of the requested subset.
    variables : list[str]
        What variables to download.

    Returns
    -------
    Tuple[Path, Path]
        Paths to the minimal and slightly bigger GLORYS subset files.
    """
    bounds = get_glorys_bounds(grid)

    # minimal dataset
    regional_file = tmp_path / "regional_GLORYS.nc"
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=variables,
        **bounds,
        start_datetime=start_time,
        end_datetime=start_time,
        coordinates_selection_method="outside",
        output_filename=str(regional_file),
    )

    # slightly bigger dataset
    for key, delta in {
        "minimum_latitude": -1,
        "minimum_longitude": -1,
        "maximum_latitude": +1,
        "maximum_longitude": +1,
    }.items():
        bounds[key] += delta

    bigger_regional_file = tmp_path / "bigger_regional_GLORYS.nc"
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=variables,
        **bounds,
        start_datetime=start_time,
        end_datetime=start_time,
        coordinates_selection_method="outside",
        output_filename=str(bigger_regional_file),
    )

    return regional_file, bigger_regional_file


def write_synthetic_glodap(
    directory: Path,
    include_ts: bool = True,
    *,
    field_fn: Callable[[str, np.ndarray], np.ndarray] | None = None,
) -> Path:
    """A minimal GLODAP-shaped source: one file per variable, lat/lon/depth, NO time.

    GLODAP's real layout is exactly this -- ``{dir}/GLODAPv2.2016b.{file_var}.nc``, each
    holding one variable on a 1 degree grid with no time dimension -- so a synthetic one
    exercises the static-source path without needing the multi-GB download.

    ``include_ts`` writes temperature/salinity files alongside the six BGC tracers
    (uniform T=2 degC, S=35 PSU), exercising the real in-situ-density unit
    conversion; set it False to exercise the T/S-missing 1025 kg/m3 fallback path.

    Every variable is uniform in space. ``field_fn(file_var, values)`` receives each
    variable's uniform ``(depth, lat, lon)`` float32 array and returns the array to
    write, so a test can put NaN gaps in chosen variables.
    """
    directory.mkdir(parents=True, exist_ok=True)
    # Comfortably covers a test grid (centred at 0E/55N) plus the regrid margin.
    lat = np.arange(35.0, 76.0, 1.0)
    lon = np.arange(-30.0, 31.0, 1.0)
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
