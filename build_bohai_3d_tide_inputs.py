#!/usr/bin/env python3
"""
Generate UCLA-ROMS input files for a 3-D Bohai Sea tidal simulation.

Simulation period:
    2025-06-01 00:00:00 to 2025-07-01 00:00:00

Generated files:
    bohai_grd.nc       ROMS grid and vertical coordinates
    bohai_ini.nc       3-D initial conditions from GLORYS
    bohai_bry.nc       3-D open-boundary forcing from GLORYS
    bohai_tides.nc     TPXO tidal elevation, transport and potential
    bohai_frc.nc       ERA5 atmospheric forcing, only when USE_ERA5=True
    cppdefs.opt        compile-time switches for this case
    bohai_3d_tide.nml  runtime namelist based on UCLA-ROMS src/namelist.nml

The script does not overwrite the UCLA-ROMS source templates.
Edit the paths in the USER CONFIGURATION section before running.
"""

from __future__ import annotations

import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from roms_tools import (
    BoundaryForcing,
    Grid,
    InitialConditions,
    SurfaceForcing,
    TidalForcing,
)


# =============================================================================
# USER CONFIGURATION
# =============================================================================

UCLA_ROMS_ROOT = Path("/home/zyjworkspace/ucla-roms")

CASE_ROOT = Path(r"D:\era5data\bohai_3d_tide_202506")
INPUT_DIR = CASE_ROOT / "input"
OUTPUT_DIR = CASE_ROOT / "output"

# Existing topography file.
topography_source ={
  'name':"ETOPO2022"  
#   path: Path(r"D:\era5data\topography\SRTM15_plus_v2.0.nc"),
}

# GLORYS must contain at least one record at/near 2025-06-01 for initial
# conditions, and records bracketing 2025-06-01 through 2025-07-01 for
# boundary forcing. A wildcard string is accepted.
# GLORYS_PATH = (
#     "/home/zyjworkspace/PythonProject/roms-tides/data/GLORYS/"
#     "bohai_glorys_20250531_20250702*.nc"
# )

# # TPXO10v2a example filenames. Change them to match your downloaded version.
# TPXO_DIR = Path(
#     "/home/zyjworkspace/PythonProject/roms-tides/data/TPXO10v2a"
# )
# TPXO_FILES = {
#     "grid": TPXO_DIR / "grid_tpxo10v2a.nc",
#     "h": TPXO_DIR / "h_tpxo10.v2a.nc",
#     "u": TPXO_DIR / "u_tpxo10.v2a.nc",
# }

# False: pure tidal experiment with zero analytical surface fluxes.
# True:  tides + realistic ERA5 atmospheric forcing.
USE_ERA5 = False

# When USE_ERA5=True:
#   None means stream ERA5 from the ROMS-Tools-supported cloud source.
#   Otherwise set a local NetCDF path, wildcard, or list-compatible path.
ERA5_PATH: str | None = None

START_TIME = datetime(2025, 6, 1, 0, 0, 0)
END_TIME = datetime(2025, 7, 1, 0, 0, 0)

# Setting the model reference date equal to START_TIME makes model time zero
# correspond to 2025-06-01 00:00:00 in all generated forcing files.
MODEL_REFERENCE_DATE = START_TIME

# Bohai Sea plus northern Yellow Sea buffer.
# The southern and eastern edges are treated as open boundaries.
NX = 240
NY = 207
SIZE_X_KM = 720.0
SIZE_Y_KM = 620.0
CENTER_LON = 120.75
CENTER_LAT = 38.50
ROTATION_DEG = 0.0

# Terrain-following vertical coordinate.
N_LEVELS = 20
THETA_S = 5.0
THETA_B = 2.0
HC_M = 10.0
HMIN_M = 5.0

# Runtime decomposition and time stepping.
NP_XI = 4
NP_ETA = 3
DT_SECONDS = 120
NDTFAST = 30

# TPXO constituents. ROMS-Tools supports up to 15.
NTIDES = 15

# Geographic source-data subset used by Dask before interpolation.
SOURCE_BOUNDS = {
    "longitude": (115.0, 126.5),
    "latitude": (34.0, 43.0),
}


# =============================================================================
# HELPERS
# =============================================================================

def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_glob(pattern: str, label: str) -> list[str]:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"{label} matched no files.\nPattern: {pattern}"
        )
    return matches


def normalize_saved_paths(result, requested_path: Path) -> list[Path]:
    """Normalize ROMS-Tools save() return values across package versions."""
    if result is None:
        return [requested_path]
    if isinstance(result, (str, Path)):
        return [Path(result)]
    return [Path(item) for item in result]


def set_cpp_switch(text: str, switch: str, enabled: bool) -> str:
    """Set an existing #define/#undef switch, or insert it before final include."""
    replacement = f"#define {switch}" if enabled else f"#undef {switch}"
    pattern = re.compile(
        rf"^[ \t]*#(?:define|undef)[ \t]+{re.escape(switch)}(?:[ \t].*)?$",
        flags=re.MULTILINE,
    )
    updated, count = pattern.subn(replacement, text)

    if count:
        return updated

    marker = '#include "set_global_definitions.h"'
    if marker not in text:
        raise RuntimeError(
            f"Cannot insert CPP switch {switch}: final include marker not found."
        )
    return updated.replace(marker, replacement + "\n" + marker)


def patch_cppdefs(template_path: Path, target_path: Path) -> None:
    """Create a hydrostatic, physical 3-D tidal cppdefs.opt."""
    require_file(template_path, "UCLA-ROMS cppdefs.opt template")
    text = template_path.read_text(encoding="utf-8")

    enabled = {
        "NONLIN_EOS",
        "SALINITY",
        "SOLVE3D",
        "SPLIT_EOS",
        "TIDES",
        "UV_ADV",
        "UV_COR",
        "CURVGRID",
        "MASKING",
        "SPHERICAL",
        "SPONGE",
        "M2_FRC_BRY",
        "M3_FRC_BRY",
        "OBC_EAST",
        "OBC_SOUTH",
        "OBC_M2FLATHER",
        "OBC_M3ORLANSKI",
        "OBC_TORLANSKI",
        "T_FRC_BRY",
        "Z_FRC_BRY",
        "LMD_BKPP",
        "LMD_KPP",
        "LMD_MIXING",
        "LMD_NONLOCAL",
        "LMD_RIMIX",
        "TS_DIF2",
        "UV_VIS2",
    }

    disabled = {
        "EXACT_RESTARTS",
        "NHMG",
        "NHMGDIAG",
        "DIAGNOSTICS_NHMG",
        "OBC_NORTH",
        "OBC_WEST",
        "ANA_BRY",
        "OBC_M2ORLANSKI",
        "OBC_M3SPECIFIED",
        "OBC_TSPECIFIED",
        "SPONGE_TUNE",
        "MARBL",
        "MARBL_DIAGS",
        "NHY_FORCING",
        "NOX_FORCING",
        "PCO2AIR_FORCING",
        "BIOLOGY_BEC2",
        "BEC2_DIAG",
        "CDR_FORCING",
        "CDR_TRACER",
        "UPSCALING",
        "ANA_INITIAL",
    }

    if USE_ERA5:
        enabled.add("BULK_FRC")
        disabled.update(
            {"ANA_SMFLUX", "ANA_SRFLUX", "ANA_SSFLUX", "ANA_STFLUX"}
        )
    else:
        disabled.add("BULK_FRC")
        enabled.update(
            {"ANA_SMFLUX", "ANA_SRFLUX", "ANA_SSFLUX", "ANA_STFLUX"}
        )

    for switch in sorted(enabled):
        text = set_cpp_switch(text, switch, True)
    for switch in sorted(disabled):
        text = set_cpp_switch(text, switch, False)

    target_path.write_text(text, encoding="utf-8")


def set_namelist_value(
    text: str,
    key: str,
    value: str,
    *,
    required: bool = True,
) -> str:
    """Replace a scalar/list value in the official UCLA-ROMS namelist."""
    pattern = re.compile(
        rf"^([ \t]*{re.escape(key)}[ \t]*=).*$",
        flags=re.MULTILINE | re.IGNORECASE,
    )
    updated, count = pattern.subn(
        lambda match: f"{match.group(1)} {value}",
        text,
        count=1,
    )
    if required and count == 0:
        raise KeyError(f"Namelist key not found in template: {key}")
    return updated


def replace_forcing_group(text: str, forcing_files: Iterable[Path]) -> str:
    lines = ",\n".join(f"'{path.resolve()}'" for path in forcing_files)
    replacement = (
        "&FORCING_FILES\n"
        "! Generated by build_bohai_3d_tide_inputs.py\n"
        "frcfiles =\n"
        f"{lines},\n"
        "/"
    )
    pattern = re.compile(
        r"^&FORCING_FILES\b.*?^/",
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("Could not replace &FORCING_FILES namelist group.")
    return updated


def patch_namelist(
    template_path: Path,
    target_path: Path,
    grid_file: Path,
    initial_file: Path,
    forcing_files: list[Path],
) -> None:
    """Patch the complete official namelist instead of creating a partial one."""
    require_file(template_path, "UCLA-ROMS namelist template")
    text = template_path.read_text(encoding="utf-8")

    run_seconds = int((END_TIME - START_TIME).total_seconds())
    if run_seconds % DT_SECONDS != 0:
        raise ValueError("Simulation duration must be divisible by DT_SECONDS.")
    ntimes = run_seconds // DT_SECONDS

    values = {
        "output_root_name": f'"{(OUTPUT_DIR / "bohai_3d_tide").resolve()}"',
        "title": '"Bohai 3-D tidal simulation: 2025-06-01 to 2025-07-01"',
        "ntimes": str(ntimes),
        "dt": str(DT_SECONDS),
        "ndtfast": str(NDTFAST),
        "ninfo": "100",
        "reference_date": "2025, 6, 1",
        "grdname": f'"{grid_file.resolve()}"',
        "theta_s": f"{THETA_S}D0",
        "theta_b": f"{THETA_B}D0",
        "hc": f"{HC_M}D0",
        "NP_XI": str(NP_XI),
        "NP_ETA": str(NP_ETA),
        "LLm": str(NX),
        "MMm": str(NY),
        "nz": str(N_LEVELS),
        "nt_passive": "0",
        "nt_cdr_oae": "0",
        "nt_cdr_dor": "0",
        "nt_bgc": "0",
        "inifile": f'"{initial_file.resolve()}"',
        "interp_bulk_frc": ".false.",
        "check_bulk_frc_units": ".true.",
        "interp_flux_frc": ".false.",
        "river_source": ".false.",
        "river_analytical": ".false.",
        "bry_tides": ".true.",
        "pot_tides": ".true.",
        "ana_tides": ".false.",
        "ntides": str(NTIDES),
        "wrt_file_his": ".true.",
        "output_period_his": "3600",
        "nrpf_his": "24",
        "wrt_Z": ".true.",
        "wrt_Ub": ".true.",
        "wrt_Vb": ".true.",
        "wrt_U": ".true.",
        "wrt_V": ".true.",
        "wrt_R": ".false.",
        "wrt_O": ".false.",
        "wrt_W": ".false.",
        "wrt_Akv": ".false.",
        "wrt_Akt": ".false.",
        "wrt_Aks": ".false.",
        "wrt_Hbls": ".false.",
        "wrt_Hbbl": ".false.",
        "wrt_file_avg": ".false.",
        "wrt_file_rst": ".true.",
        "monthly_restarts": ".false.",
        "output_period_rst": "86400",
        "nrpf_rst": "1",
        "wrt_temp": ".true.",
        "wrt_salt": ".true.",
        "wrt_temp_dia": ".false.",
        "wrt_salt_dia": ".false.",
        "calc_pflx": ".false.",
        "rho0": "1025.0",
        "tnu2": "5.0, 5.0",
        "rdrg": "0.0",
        "rdrg2": "2.5e-3",
        "zob": "1.0e-2",
        "akv_bak": "1.0e-5",
        "akt_bak": "1.0e-6, 1.0e-6",
        "visc2": "10.0",
        "ubind": "0.1",
        "v_sponge": "50.0",
    }

    # Some optional keys can vary between UCLA-ROMS revisions.
    optional_keys = {
        "nt_cdr_oae",
        "nt_cdr_dor",
        "check_bulk_frc_units",
        "wrt_R",
        "wrt_O",
        "wrt_W",
        "wrt_Akv",
        "wrt_Akt",
        "wrt_Aks",
        "wrt_Hbls",
        "wrt_Hbbl",
        "calc_pflx",
    }

    for key, value in values.items():
        text = set_namelist_value(
            text,
            key,
            value,
            required=key not in optional_keys,
        )

    text = replace_forcing_group(text, forcing_files)
    target_path.write_text(text, encoding="utf-8")


def source_dict(name: str, path: str | None) -> dict[str, str]:
    source = {"name": name}
    if path:
        source["path"] = path
    return source


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # require_file(SRTM15_PATH, "SRTM15 topography")
    # require_glob(GLORYS_PATH, "GLORYS source")
    # for key, path in TPXO_FILES.items():
    #     require_file(path, f"TPXO {key} file")

    if NX % NP_XI != 0 or NY % NP_ETA != 0:
        raise ValueError(
            "For this case, choose NP_XI and NP_ETA that divide NX and NY "
            "exactly. Current values are inconsistent."
        )

    print("1/6 Creating ROMS grid...")
    grid = Grid(
        nx=NX,
        ny=NY,
        size_x=SIZE_X_KM,
        size_y=SIZE_Y_KM,
        center_lon=CENTER_LON,
        center_lat=CENTER_LAT,
        rot=ROTATION_DEG,
        N=N_LEVELS,
        theta_s=THETA_S,
        theta_b=THETA_B,
        hc=HC_M,
        hmin=HMIN_M,
        topography_source=topography_source,
        close_narrow_channels=False,
        verbose=True,
    )
    grid_path = INPUT_DIR / "bohai_grd.nc"
    grid_saved = normalize_saved_paths(grid.save(grid_path), grid_path)
    grid.to_yaml(INPUT_DIR / "bohai_grd.yaml")

    print("2/6 Creating 3-D initial conditions from GLORYS...")
    initial = InitialConditions(
        grid=grid,
        ini_time=START_TIME,
        source={"name": "GLORYS", "path": GLORYS_PATH},
        model_reference_date=MODEL_REFERENCE_DATE,
        use_dask=True,
        initial_slice_bounds=SOURCE_BOUNDS,
    )
    ini_path = INPUT_DIR / "bohai_ini.nc"
    ini_saved = normalize_saved_paths(initial.save(ini_path), ini_path)
    initial.to_yaml(INPUT_DIR / "bohai_ini.yaml")

    print("3/6 Creating 3-D south/east boundary forcing from GLORYS...")
    boundary = BoundaryForcing(
        grid=grid,
        start_time=START_TIME,
        end_time=END_TIME,
        boundaries={
            "south": True,
            "east": True,
            "north": False,
            "west": False,
        },
        source={"name": "GLORYS", "path": GLORYS_PATH},
        type="physics",
        model_reference_date=MODEL_REFERENCE_DATE,
        apply_2d_horizontal_fill=True,
        use_dask=True,
        initial_slice_bounds=SOURCE_BOUNDS,
    )
    bry_path = INPUT_DIR / "bohai_bry.nc"
    bry_saved = normalize_saved_paths(
        boundary.save(bry_path, group=False),
        bry_path,
    )
    boundary.to_yaml(INPUT_DIR / "bohai_bry.yaml")

    print("4/6 Creating TPXO tidal forcing...")
    tidal = TidalForcing(
        grid=grid,
        source={
            "name": "TPXO",
            "path": {key: str(value) for key, value in TPXO_FILES.items()},
        },
        ntides=NTIDES,
        model_reference_date=MODEL_REFERENCE_DATE,
        use_dask=True,
    )
    tide_path = INPUT_DIR / "bohai_tides.nc"
    tide_saved = normalize_saved_paths(tidal.save(tide_path), tide_path)
    tidal.to_yaml(INPUT_DIR / "bohai_tides.yaml")

    forcing_files = [tide_saved[0], bry_saved[0]]

    if USE_ERA5:
        print("5/6 Creating ERA5 atmospheric forcing...")
        if ERA5_PATH is not None:
            require_glob(ERA5_PATH, "ERA5 source")

        surface = SurfaceForcing(
            grid=grid,
            start_time=START_TIME,
            end_time=END_TIME,
            source=source_dict("ERA5", ERA5_PATH),
            type="physics",
            coarse_grid_mode="never",
            correct_radiation=True,
            model_reference_date=MODEL_REFERENCE_DATE,
            use_dask=True,
            initial_slice_bounds=SOURCE_BOUNDS,
        )
        frc_path = INPUT_DIR / "bohai_frc.nc"
        frc_saved = normalize_saved_paths(
            surface.save(frc_path, group=False),
            frc_path,
        )
        surface.to_yaml(INPUT_DIR / "bohai_frc.yaml")
        forcing_files.append(frc_saved[0])
    else:
        print("5/6 ERA5 disabled: using analytical zero surface fluxes.")

    print("6/6 Writing UCLA-ROMS compile-time and runtime configuration...")
    cppdefs_template = UCLA_ROMS_ROOT / "src" / "cppdefs.opt"
    namelist_template = UCLA_ROMS_ROOT / "src" / "namelist.nml"

    cppdefs_target = CASE_ROOT / "cppdefs.opt"
    namelist_target = CASE_ROOT / "bohai_3d_tide.nml"

    patch_cppdefs(cppdefs_template, cppdefs_target)
    patch_namelist(
        namelist_template,
        namelist_target,
        grid_file=grid_saved[0],
        initial_file=ini_saved[0],
        forcing_files=forcing_files,
    )

    print("\nGenerated case:")
    print(f"  case directory : {CASE_ROOT}")
    print(f"  grid           : {grid_saved[0]}")
    print(f"  initial        : {ini_saved[0]}")
    print(f"  boundary       : {bry_saved[0]}")
    print(f"  tides          : {tide_saved[0]}")
    if USE_ERA5:
        print(f"  surface        : {forcing_files[-1]}")
    print(f"  cppdefs        : {cppdefs_target}")
    print(f"  namelist       : {namelist_target}")

    print("\nNext commands:")
    print(f"  cp {cppdefs_target} {UCLA_ROMS_ROOT}/cppdefs.opt")
    print(f"  cd {UCLA_ROMS_ROOT}")
    print("  make clean")
    print("  make COMPILER=gnu MPI_WRAPPER=mpifort")
    print(f"  mpirun -np {NP_XI * NP_ETA} ./roms {namelist_target}")


if __name__ == "__main__":
    main()
