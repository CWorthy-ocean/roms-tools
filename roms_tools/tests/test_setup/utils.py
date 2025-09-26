from datetime import datetime
from pathlib import Path

from roms_tools import Grid, get_glorys_bounds

try:
    import copernicusmarine  # type: ignore
except ImportError:
    copernicusmarine = None


def download_regional_and_bigger(
    tmp_path: Path, grid: Grid, start_time: datetime
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

    Returns
    -------
    Tuple[Path, Path]
        Paths to the minimal and slightly bigger GLORYS subset files.
    """
    bounds = get_glorys_bounds(grid_ds=grid.ds)

    # minimal dataset
    regional_file = tmp_path / "regional_GLORYS.nc"
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao", "so", "uo", "vo", "zos"],
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
        variables=["thetao", "so", "uo", "vo", "zos"],
        **bounds,
        start_datetime=start_time,
        end_datetime=start_time,
        coordinates_selection_method="outside",
        output_filename=str(bigger_regional_file),
    )

    return regional_file, bigger_regional_file
