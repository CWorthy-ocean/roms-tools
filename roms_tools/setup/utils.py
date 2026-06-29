import importlib.metadata
import logging
import time
import typing
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeAlias

import gsw
import numba as nb
import numpy as np
import pandas as pd
import xarray as xr
import xgcm
import yaml
from pydantic import BaseModel

from roms_tools.constants import R_EARTH
from roms_tools.utils import transpose_dimensions

if typing.TYPE_CHECKING:
    from roms_tools.setup.grid import Grid

yaml.SafeDumper.add_multi_representer(
    StrEnum,
    yaml.representer.SafeRepresenter.represent_str,
)

HEADER_SZ = 96
HEADER_CHAR = "="

RawDataSource: TypeAlias = dict[str, str | Path | list[str | Path] | bool]


def log_the_separator() -> None:
    """Log a separator line using HEADER_CHAR repeated HEADER_SZ times."""
    logging.info(HEADER_CHAR * HEADER_SZ)


class Timed:
    """Context manager to time a block and log messages."""

    def __init__(self, message: str = "", verbose: bool = True) -> None:
        """
        Initialize the context manager.

        Parameters
        ----------
        message : str, optional
            A log message printed at the start of the block (default: "").
        verbose : bool, optional
            Whether to log timing information (default: True).
        """
        self.message = message
        self.verbose = verbose
        self.start: float | None = None

    def __enter__(self) -> "Timed":
        if self.verbose:
            self.start = time.time()
            if self.message:
                logging.info(self.message)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.verbose and self.start is not None:
            logging.info(f"Total time: {time.time() - self.start:.3f} seconds")
            log_the_separator()


def nan_check(field, mask, error_message=None) -> None:
    """Checks for NaN values at wet points in the field.

    This function examines the interpolated input field for NaN values at positions indicated as wet points by the mask.
    If any NaN values are found at these wet points, a ValueError is raised.

    Parameters
    ----------
    field : array-like
        The data array to be checked for NaN values. This is typically an xarray.DataArray or numpy array.

    mask : array-like
        A boolean mask or data array with the same shape as `field`. The wet points (usually ocean points)
        are indicated by `1` or `True`, and land points by `0` or `False`.

    error_message : str, optional
        A custom error message to be included in the ValueError if NaN values are detected. If not provided,
        a default message will explain the potential cause and suggest ensuring the dataset's coverage.

    Raises
    ------
    ValueError
        If the field contains NaN values at any of the wet points indicated by the mask.
        The error message will explain the potential cause and suggest ensuring the dataset's coverage.
    """
    # Replace values in field with 0 where mask is not 1
    da = xr.where(mask == 1, field, 0)
    if error_message is None:
        error_message = (
            "NaN values found in regridded field. This likely occurs because the ROMS grid, including "
            "a small safety margin for interpolation, is not fully contained within the dataset's longitude/latitude range. Please ensure that the "
            "dataset covers the entire area required by the ROMS grid."
        )
    # Check if any NaN values exist in the modified field
    if da.isnull().any().values:
        raise ValueError(error_message)


def substitute_nans_by_fillvalue(field, fill_value=0.0) -> xr.DataArray:
    """Replace NaN values in the field with a specified fill value.

    This function replaces any NaN values in the input field with the provided fill value.

    Parameters
    ----------
    field : xr.DataArray
        The data array in which NaN values need to be replaced. This is typically an xarray.DataArray.
    fill_value : scalar, optional
        The value to use for replacing NaNs. Default is 0.0.

    Returns
    -------
    xr.DataArray
        The data array with NaN values replaced by the specified fill value.
    """
    return field.fillna(fill_value)


def assign_dates_to_climatology(ds: xr.Dataset, time_dim: str) -> xr.Dataset:
    """Assigns climatology dates to the dataset's time dimension.

    This function updates the dataset's time coordinates to reflect climatological dates.
    It defines fixed day increments for each month and assigns these to the specified time dimension.
    The increments represent the cumulative days at mid-month for each month.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset to which climatological dates will be assigned.
    time_dim : str
        The name of the time dimension in the dataset that will be updated with climatological dates.

    Returns
    -------
    xr.Dataset
        The updated xarray Dataset with climatological dates assigned to the specified time dimension.
    """
    # Define the days in each month and convert to timedelta
    increments = [15, 30, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30]
    days = np.cumsum(increments)
    timedelta_ns = np.array(days, dtype="timedelta64[D]").astype("timedelta64[ns]")
    time = xr.DataArray(timedelta_ns, dims=[time_dim])
    ds = ds.assign_coords({"time": time})
    return ds


def calendar_midmonth_dates(start_time: datetime, end_time: datetime) -> list[datetime]:
    """Return 15th-of-month dates between ``start_time`` and ``end_time``."""
    dates: list[datetime] = []
    year, month = start_time.year, start_time.month
    while (year, month) <= (end_time.year, end_time.month):
        candidate = datetime(year, month, 15)
        if start_time <= candidate <= end_time:
            dates.append(candidate)
        month += 1
        if month > 12:
            month = 1
            year += 1
    if not dates:
        # Window is shorter than a month and skips every 15th; fall back to a
        # single representative mid-month date clamped into the window so the
        # correct climatology month is still selected.
        candidate = datetime(start_time.year, start_time.month, 15)
        dates.append(min(max(candidate, start_time), end_time))
    return dates


def month_to_time_index(month_coord: np.ndarray | xr.DataArray) -> dict[int, int]:
    """Map calendar month (1-12) to index along a monthly climatology axis."""
    values = np.asarray(month_coord).astype(int)
    return {int(month): idx for idx, month in enumerate(values)}


def tile_monthly_climatology_on_calendar(
    ds: xr.Dataset,
    calendar_dates: Sequence[datetime],
    *,
    month_coord: str = "month",
    time_dim: str = "river_time",
) -> xr.Dataset:
    """Repeat monthly climatology fields along a calendar date axis."""
    if month_coord not in ds:
        raise ValueError(f"Dataset must contain coordinate '{month_coord}'.")

    month_to_index = month_to_time_index(ds[month_coord])
    ds_base = ds.drop_vars(month_coord, errors="ignore")
    pieces = [
        ds_base.isel({time_dim: month_to_index[dt.month]}) for dt in calendar_dates
    ]
    ds_tiled = xr.concat(pieces, dim=time_dim)
    return ds_tiled.assign_coords(
        {time_dim: np.array(calendar_dates, dtype="datetime64[ns]")}
    )


def expand_monthly_climatology_time_axis(
    ds: xr.Dataset,
    start_time: datetime,
    end_time: datetime,
    model_reference_date: datetime | np.datetime64,
    *,
    time_dim: str = "river_time",
    month_coord: str = "month",
    discharge_climatology_attr: str = "discharge_climatology",
) -> xr.Dataset:
    """Expand a monthly climatology dataset to calendar mid-month times."""
    calendar_dates = calendar_midmonth_dates(start_time, end_time)
    ds_expanded = tile_monthly_climatology_on_calendar(
        ds,
        calendar_dates,
        month_coord=month_coord,
        time_dim=time_dim,
    )
    ds_expanded.attrs.pop("climatology", None)
    ds_expanded.attrs[discharge_climatology_attr] = "True"
    ds_expanded, time = add_time_info_to_ds(
        ds_expanded,
        model_reference_date,
        climatology=False,
        time_name=time_dim,
    )
    logging.info(
        "Repeated 12-month discharge climatology on %d calendar river_time steps "
        "(%s to %s).",
        len(calendar_dates),
        calendar_dates[0].date(),
        calendar_dates[-1].date(),
    )
    return ds_expanded.assign_coords({time_dim: time})


def interpolate_dynamic_bgc_by_calendar_year(
    dynamic: dict[str, xr.DataArray],
    abs_time: xr.DataArray,
    *,
    time_dim: str = "river_time",
) -> dict[str, xr.DataArray]:
    """Linearly interpolate interior gap years in dynamic BGC concentrations.

    Each tracer is collapsed to one value per calendar year (mean of finite
    ``river_time`` steps in that year), missing years are inserted, interior NaN
    years are linearly interpolated along the year axis, and the result is
    broadcast back to every ``river_time`` step. Leading and trailing NaN years
    are left unchanged.
    """
    calendar_years = abs_time.dt.year
    if time_dim not in calendar_years.dims:
        raise ValueError(
            f"abs_time must have dimension {time_dim!r}, got dims {calendar_years.dims}."
        )

    start_year = int(calendar_years.min())
    end_year = int(calendar_years.max())
    full_years = np.arange(start_year, end_year + 1)
    year_at_steps = calendar_years.values

    interpolated: dict[str, xr.DataArray] = {}
    for tracer_name, values in dynamic.items():
        if time_dim not in values.dims:
            raise ValueError(
                f"Dynamic tracer {tracer_name!r} must have dimension {time_dim!r}."
            )

        tagged = values.assign_coords(calendar_year=calendar_years)
        annual = tagged.groupby("calendar_year").mean(dim=time_dim, skipna=True)
        annual = annual.reindex(calendar_year=full_years)
        annual = annual.interpolate_na(dim="calendar_year", method="linear")

        year_indexer = xr.DataArray(
            year_at_steps,
            dims=[time_dim],
            coords={time_dim: values.coords[time_dim]},
        )
        filled = annual.sel(calendar_year=year_indexer, drop=True)
        dims = [d for d in (time_dim, "nriver") if d in values.dims]
        interpolated[tracer_name] = filled.transpose(*dims).astype(np.float32)

    return interpolated


def get_variable_metadata():
    """Retrieves metadata for commonly used variables in the dataset.

    This function returns a dictionary containing the metadata for various variables, including long names
    and units for each variable.

    Returns
    -------
    dict of str: dict
        Dictionary where keys are variable names and values are dictionaries with "long_name" and "units" keys.
    """
    d = {
        "ssh_Re": {"long_name": "Tidal elevation, real part", "units": "m"},
        "ssh_Im": {"long_name": "Tidal elevation, complex part", "units": "m"},
        "pot_Re": {"long_name": "Tidal potential, real part", "units": "m"},
        "pot_Im": {"long_name": "Tidal potential, complex part", "units": "m"},
        "u_Re": {
            "long_name": "Tidal velocity in x-direction, real part",
            "units": "m/s",
        },
        "u_Im": {
            "long_name": "Tidal velocity in x-direction, complex part",
            "units": "m/s",
        },
        "v_Re": {
            "long_name": "Tidal velocity in y-direction, real part",
            "units": "m/s",
        },
        "v_Im": {
            "long_name": "Tidal velocity in y-direction, complex part",
            "units": "m/s",
        },
        "uwnd": {"long_name": "10 meter wind in x-direction", "units": "m/s"},
        "vwnd": {"long_name": "10 meter wind in y-direction", "units": "m/s"},
        "swrad": {
            "long_name": "downward short-wave (solar) radiation",
            "units": "W/m^2",
        },
        "lwrad": {
            "long_name": "downward long-wave (thermal) radiation",
            "units": "W/m^2",
        },
        "Tair": {"long_name": "air temperature at 2m", "units": "degrees Celsius"},
        "qair": {"long_name": "absolute humidity at 2m", "units": "kg/kg"},
        "rain": {"long_name": "total precipitation", "units": "cm/day"},
        "temp": {
            "long_name": "potential temperature",
            "units": "degrees Celsius",
            "flux_units": "degrees Celsius/s",
        },
        "salt": {"long_name": "salinity", "units": "PSU", "flux_units": "PSU/s"},
        "sss": {"long_name": "sea surface salinity", "units": "PSU"},
        "sDIC": {"long_name": "sea surface DIC", "units": "mmol/m3"},
        "sALK": {"long_name": "sea surface ALK", "units": "mmol/m3"},
        "zeta": {"long_name": "sea surface height", "units": "m"},
        "u": {"long_name": "u-flux component", "units": "m/s"},
        "v": {"long_name": "v-flux component", "units": "m/s"},
        "w": {"long_name": "w-flux component", "units": "m/s"},
        "ubar": {
            "long_name": "vertically integrated u-flux component",
            "units": "m/s",
        },
        "vbar": {
            "long_name": "vertically integrated v-flux component",
            "units": "m/s",
        },
        "PO4": {
            "long_name": "dissolved inorganic phosphate",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "NO3": {
            "long_name": "dissolved inorganic nitrate",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "SiO3": {
            "long_name": "dissolved inorganic silicate",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "NH4": {
            "long_name": "dissolved ammonia",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "Fe": {
            "long_name": "dissolved inorganic iron",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "Lig": {
            "long_name": "iron binding ligand",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "O2": {
            "long_name": "dissolved oxygen",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "DIC": {
            "long_name": "dissolved inorganic carbon",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "DIC_ALT_CO2": {
            "long_name": "dissolved inorganic carbon, alternative CO2",
            "units": "mmol/m^3",
            "flux_units": "meq/s",
            "integrated_units": "meq",
        },
        "ALK": {
            "long_name": "alkalinity",
            "units": "meq/m^3",
            "flux_units": "meq/s",
            "integrated_units": "meq",
        },
        "ALK_ALT_CO2": {
            "long_name": "alkalinity, alternative CO2",
            "units": "meq/m^3",
            "flux_units": "meq/s",
            "integrated_units": "meq",
        },
        "DOC": {
            "long_name": "dissolved organic carbon",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "DON": {
            "long_name": "dissolved organic nitrogen",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "DOP": {
            "long_name": "dissolved organic phosphorus",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "DOCr": {
            "long_name": "refractory dissolved organic carbon",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "DONr": {
            "long_name": "refractory dissolved organic nitrogen",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "DOPr": {
            "long_name": "refractory dissolved organic phosphorus",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "zooC": {
            "long_name": "zooplankton carbon",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "spChl": {
            "long_name": "small phytoplankton chlorophyll",
            "units": "mg/m^3",
            "flux_units": "mg/s",
            "integrated_units": "mg",
        },
        "spC": {
            "long_name": "small phytoplankton carbon",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "spP": {
            "long_name": "small phytoplankton phosphorous",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "spFe": {
            "long_name": "small phytoplankton iron",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "spCaCO3": {
            "long_name": "small phytoplankton CaCO3",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "diatChl": {
            "long_name": "diatom chloropyll",
            "units": "mg/m^3",
            "flux_units": "mg/s",
            "integrated_units": "mg",
        },
        "diatC": {
            "long_name": "diatom carbon",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "diatP": {
            "long_name": "diatom phosphorus",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "diatFe": {
            "long_name": "diatom iron",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "diatSi": {
            "long_name": "diatom silicate",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "diazChl": {
            "long_name": "diazotroph chloropyll",
            "units": "mg/m^3",
            "flux_units": "mg/s",
            "integrated_units": "mg",
        },
        "diazC": {
            "long_name": "diazotroph carbon",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "diazP": {
            "long_name": "diazotroph phosphorus",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "diazFe": {
            "long_name": "diazotroph iron",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
            "integrated_units": "mmol",
        },
        "xco2_air": {"long_name": "CO2, Marine Boundary Layer", "units": "µmol mol⁻¹"},
        "xco2_air_alt": {
            "long_name": "CO2, Marine Boundary Layer; alternative CO2",
            "units": "µmol mol⁻¹",
        },
        "iron": {"long_name": "iron decomposition", "units": "nmol/cm^2/s"},
        "dust": {"long_name": "dust decomposition", "units": "kg/m^2/s"},
        "nox": {"long_name": "NOx decomposition", "units": "kg/m^2/s"},
        "nhy": {"long_name": "NHy decomposition", "units": "kg/m^2/s"},
    }
    return d


def compute_missing_bgc_variables(bgc_data):
    """Fills in missing biogeochemical (BGC) variables in the input dictionary.

    This function checks for missing BGC variables in the provided dictionary and
    computes them based on predefined relationships with existing variables. The
    relationships specify either a multiplication factor applied to an existing
    variable or a constant value if no related variable is available. The resulting
    variables are added to the dictionary.

    Parameters
    ----------
    bgc_data : dict
        A dictionary containing biogeochemical variables as xarray DataArrays.
        Missing variables are computed and added to this dictionary.

        Assumptions:
        - If `Fe` is part of the input dictionary, it is in units of mmol m-3.
        - If `CHL` is part of the input dictionary, it is in units of mg m-3.
        - If `ALK` is part of the input dictionary, it is in units of meq m-3 = mmol m-3.
        - If `DIC` is part of the input dictionary, it is in units of mmol m-3.

    Returns
    -------
    dict
        The updated dictionary with missing BGC variables filled in.

    Notes
    -----
    - If `NH4`, `DOC`, `DON`, `DOP`, `DOCr`, `DONr`, and `DOPr` are not part of the input
      dictionary, they are filled with constant values.
    - `CHL` is removed from the dictionary after the necessary calculations.
    """
    # Define the relationships for missing variables
    variable_relations = {
        "NH4": (None, 10**-6),  # mmol m-3
        "Lig": ("Fe", 3),  # mmol m-3
        "DIC_ALT_CO2": ("DIC", 1),  # mmol m-3
        "ALK_ALT_CO2": ("ALK", 1),  # meq m-3 = mmol m-3
        "DOC": (None, 10**-6),  # mmol m-3
        "DON": (None, 1.0),  # mmol m-3
        "DOP": (None, 0.1),  # mmol m-3
        "DOCr": (None, 10**-6),  # mmol m-3
        "DONr": (None, 0.8),  # mmol m-3
        "DOPr": (None, 0.003),  # mmol m-3
        "zooC": ("CHL", 1.35),  # mmol m-3
        "spChl": ("CHL", 0.675),  # mg m-3
        "spC": ("CHL", 3.375),  # mmol m-3
        "spP": ("CHL", 0.03),  # mmol m-3
        "spFe": ("CHL", 1.35e-5),  # mmol m-3
        "spCaCO3": ("CHL", 0.0675),  # mmol m-3
        "diatChl": ("CHL", 0.0675),  # mg m-3
        "diatC": ("CHL", 0.2025),  # mmol m-3
        "diatP": ("CHL", 0.02),  # mmol m-3
        "diatFe": ("CHL", 1.35e-6),  # mmol m-3
        "diatSi": ("CHL", 0.0675),  # mmol m-3
        "diazChl": ("CHL", 0.0075),  # mg m-3
        "diazC": ("CHL", 0.0375),  # mmol m-3
        "diazP": ("CHL", 0.01),  # mmol m-3
        "diazFe": ("CHL", 7.5e-7),  # mmol m-3
    }

    # Fill in missing variables using the defined relationships
    for var_name, (base_var, factor) in variable_relations.items():
        if var_name not in bgc_data:
            if base_var:
                bgc_data[var_name] = bgc_data[base_var] * factor
            else:
                bgc_data[var_name] = factor * xr.ones_like(bgc_data["ALK"])

    bgc_data.pop("CHL", None)

    return bgc_data


def compute_potential_density(
    temp: "xr.DataArray", salt: "xr.DataArray"
) -> "xr.DataArray":
    """Compute sigma-0 potential density anomaly (kg/m³ - 1000) via TEOS-10 (gsw).

    Wraps gsw.sigma0 with apply_ufunc for dask compatibility. Treats practical
    salinity as Absolute Salinity and in-situ temperature as Conservative
    Temperature — an approximation sufficient for density-coordinate interpolation.

    Parameters
    ----------
    temp : xr.DataArray
        In-situ temperature (°C).
    salt : xr.DataArray
        Practical salinity (PSU).

    Returns
    -------
    xr.DataArray
        Potential density anomaly sigma-0 (kg/m³ - 1000).
    """
    density = xr.apply_ufunc(
        gsw.sigma0,
        salt,
        temp,
        dask="parallelized",
        output_dtypes=[temp.dtype],
    )
    # apply_ufunc preserves the input dim order, but normalize to the package's
    # canonical order so this public function returns a predictable layout to
    # users who call it directly.
    density = transpose_dimensions(density)
    density.name = "sigma0"
    density.attrs["long_name"] = "potential density anomaly"
    density.attrs["units"] = "kg/m^3 - 1000"
    return density


# Internal variable-name keys for the single source temperature/salinity pair used to
# build the BGC density coordinate. A BGC dataset declares these keys in its
# ``opt_var_names`` (mapping them to whatever the file calls the fields, e.g.
# ``temp_WOA``/``salt_WOA``); the density-space interpolation in
# ``InitialConditions``/``BoundaryForcing`` detects, uses, and then drops them. The keys
# are deliberately NOT ``temp``/``salt`` so they cannot collide with the physics model
# T/S that share ``processed_fields`` in ``InitialConditions``.
BGC_SOURCE_TEMP = "temp_bgc"
BGC_SOURCE_SALT = "salt_bgc"


def _compute_density_coord(
    temp: "xr.DataArray",
    salt: "xr.DataArray",
    depth_dim: str,
) -> "xr.DataArray":
    """Build a strictly monotonic potential-density coordinate for density-space
    interpolation.

    Computes sigma-0 from ``temp``/``salt`` and adds a tiny depth-index perturbation
    (matching the reference MATLAB implementation) so the profile is strictly
    increasing along ``depth_dim`` — required by the ``xgcm`` transform that consumes
    it as a coordinate. The result is single-chunked along ``depth_dim``, which
    ``xgcm.transform`` also requires.

    The same helper builds both the *source* coordinate (``depth_dim`` = the BGC
    source depth dimension) and the *target* coordinate (``depth_dim`` = the ROMS
    ``s_rho`` dimension); the inputs differ (BGC's own T/S for the source, the model's
    T/S for the target), but the construction is identical.

    Parameters
    ----------
    temp : xr.DataArray
        Temperature (°C) on the grid whose density coordinate is wanted.
    salt : xr.DataArray
        Practical salinity (PSU) on the same grid.
    depth_dim : str
        Name of the vertical dimension along which the coordinate must be monotonic.

    Returns
    -------
    xr.DataArray
        Potential density anomaly sigma-0 plus monotonicity perturbation, single-
        chunked along ``depth_dim``.
    """
    density = compute_potential_density(temp, salt)
    n_depth = density.sizes[depth_dim]
    density = density + xr.DataArray(np.arange(n_depth) * 1e-7, dims=[depth_dim])
    # xgcm.transform requires a single chunk along the dim being transformed.
    return density.chunk({depth_dim: -1})


# Available BGC vertical-interpolation methods (selected via
# ``bgc_interpolation_method`` on ``InitialConditions``/``BoundaryForcing``):
#   "depth"       — linear interpolation in depth (the conservative default).
#   "density"     — linear interpolation in potential-density (isopycnal) space.
#   "density_mld" — linear interpolation in depth within two segments split at the
#                   mixed layer depth (MLD), with the MLD matched between source and
#                   target. Avoids the surface degeneracy of pure density space.
BGC_INTERPOLATION_METHODS = ("depth", "density", "density_mld")

# Mixed-layer-depth detection defaults (density-threshold criterion, de Boyer
# Montégut et al. 2004): the MLD is the depth at which potential density first exceeds
# the value at ``MLD_REFERENCE_DEPTH`` by ``MLD_DENSITY_THRESHOLD``.
MLD_DENSITY_THRESHOLD = 0.03  # kg/m^3
MLD_REFERENCE_DEPTH = 10.0  # m


def compute_mld(
    sigma0: "xr.DataArray",
    depth: "xr.DataArray",
    depth_dim: str,
    reference_depth: float = MLD_REFERENCE_DEPTH,
    threshold: float = MLD_DENSITY_THRESHOLD,
) -> "xr.DataArray":
    """Compute the mixed layer depth (MLD) from a potential-density field.

    Density-threshold criterion (de Boyer Montégut et al. 2004): the MLD is the
    (positive) depth at which sigma-0 first exceeds its value at ``reference_depth`` by
    ``threshold`` kg/m³, found by linear interpolation to the crossing. Fully mixed
    columns (no crossing) return the full water-column depth; all-NaN (land) columns
    return NaN. Works per horizontal column, so both 1D and spatially varying (3D)
    ``depth`` coordinates and either vertical orientation (surface-first or
    surface-last) are supported.

    The crossing is found with ``xgcm.transform`` — the same engine as xroms'
    ``mld``/``isoslice`` (so results are consistent with xroms; pass
    ``reference_depth=0`` for the surface-referenced xroms convention). Like all
    ``xgcm`` linear transforms it assumes a monotonic profile and does an endpoint
    flip rather than a full sort, so for a non-monotonic upper-ocean profile (a density
    inversion / barrier layer) it returns the linear-search crossing rather than
    guaranteeing the shallowest one.

    Parameters
    ----------
    sigma0 : xr.DataArray
        Potential density anomaly (e.g. from :func:`compute_potential_density`) with
        vertical dimension ``depth_dim``.
    depth : xr.DataArray
        Positive-down depth (m) aligned with ``sigma0`` along ``depth_dim``. May be
        1D (broadcast over the horizontal) or share ``sigma0``'s horizontal dims.
    depth_dim : str
        Name of the vertical dimension.
    reference_depth : float, optional
        Depth (m) at which the reference density is taken. Default 10 m (de Boyer
        Montégut). Columns shallower than ``reference_depth`` fall back to the
        shallowest level. Pass 0 for the surface-referenced xroms convention.
    threshold : float, optional
        Density excess (kg/m³) defining the base of the mixed layer. Default 0.03.

    Returns
    -------
    xr.DataArray
        MLD as positive depth (m), with ``depth_dim`` removed.

    References
    ----------
    de Boyer Montégut, C., Madec, G., Fischer, A. S., Lazar, A., & Iudicone, D. (2004).
    Mixed layer depth over the global ocean. J. Geophys. Res. Oceans, 109(C12).
    See also xroms ``roms_seawater.mld`` and the NCL ``mixed_layer_depth`` routine.
    """
    absz = np.abs(depth).broadcast_like(sigma0)
    if sigma0.chunks is not None:
        sigma0 = sigma0.chunk({depth_dim: -1})
        absz = absz.chunk({depth_dim: -1})

    grid = xgcm.Grid(
        sigma0.to_dataset(name="sigma0"),
        coords={depth_dim: {"center": depth_dim}},
        periodic=False,
        autoparse_metadata=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="xgcm")
        # Reference density: sigma-0 interpolated to ``reference_depth`` (NaN where the
        # column does not span it), with a shallowest-level fallback for shallow columns.
        sig_ref = grid.transform(
            sigma0,
            depth_dim,
            target=np.array([float(reference_depth)]),
            target_data=absz.rename("zref"),
            method="linear",
        ).isel(zref=0, drop=True)
        surface = sigma0.where(absz == absz.min(depth_dim)).max(depth_dim)
        sig_ref = sig_ref.where(sig_ref.notnull(), surface)

        # MLD: the depth at which (sigma0 - sig_ref) crosses ``threshold``.
        iso = (sigma0 - sig_ref).rename("iso")
        mld = grid.transform(
            absz,
            depth_dim,
            target=np.array([float(threshold)]),
            target_data=iso,
            method="linear",
        ).isel(iso=0, drop=True)

    # No crossing (fully mixed) -> full water-column depth; land (all-NaN) -> NaN.
    bottom = absz.where(sigma0.notnull()).max(depth_dim)
    mld = mld.where(mld.notnull(), bottom)
    mld = mld.where(~sigma0.isnull().all(depth_dim))
    mld = np.abs(mld)
    mld.name = "mld"
    mld.attrs["long_name"] = "mixed layer depth"
    mld.attrs["units"] = "m"
    return mld


def _compute_mld_warp(
    temp: "xr.DataArray",
    salt: "xr.DataArray",
    depth: "xr.DataArray",
    depth_dim: str,
    target_mld: "xr.DataArray",
    target_H: "xr.DataArray | None" = None,
    reference_depth: float = MLD_REFERENCE_DEPTH,
    threshold: float = MLD_DENSITY_THRESHOLD,
) -> "xr.DataArray":
    """Build the warped *source* depth coordinate for MLD-anchored interpolation.

    Maps each source depth into the target's depth space so that the source mixed
    layer depth aligns with the target's:

    - mixed layer ``[surface, MLD_src]`` maps to ``[surface, MLD_tgt]`` (scaled so the
      source MLD lands on the target MLD), and
    - below the MLD the map is **1:1 in depth** (``d_warp = MLD_tgt + (|z| - MLD_src)``),
      preserving the *absolute* depth of sub-mixed-layer features below the MLD.

    Paired in :func:`build_bgc_vertical_coords` with the *real* target depth, this makes
    ``VerticalRegrid.apply`` interpolate linearly in depth within each segment with the
    MLD matched. The 1:1 lower segment (rather than stretching the source bottom onto
    the target bottom) is deliberate: when the target is much shallower than the source
    it would otherwise compress the entire deep source column into the thin target
    layer. With 1:1, source water below the target floor is simply edge-clamped/unused.

    Columns lacking a resolved mixed layer — fully mixed source, fully mixed target,
    NaN MLD, or a degenerate (≈0) source/target MLD — fall back to the identity map
    (``d_warp = |z|``), i.e. plain depth interpolation for that column. Because
    ``xgcm.transform`` is per-column independent, stratified and degenerate columns
    coexist in one call.

    Returns the warped depth plus the same monotonicity perturbation as
    :func:`_compute_density_coord`, single-chunked along ``depth_dim``.
    """
    # ``depth`` is assumed ordered shallow->deep along ``depth_dim`` (surface first),
    # as for BGC source depth levels; the warp is then co-monotonic with the index-based
    # perturbation below (same orientation convention as ``_compute_density_coord``).
    sigma0 = compute_potential_density(temp, salt)
    absz = np.abs(depth)
    mld_src = compute_mld(
        sigma0, depth, depth_dim, reference_depth=reference_depth, threshold=threshold
    )
    H_src = absz.where(sigma0.notnull()).max(depth_dim)

    # A resolved mixed layer on both sides is required for a strictly monotonic warp;
    # otherwise fall back to the identity (depth) map. A fully mixed source has
    # mld_src == H_src; a fully mixed target has target_mld == target_H.
    eps = 1e-6
    can_warp = (
        (H_src - mld_src > eps)
        & (mld_src > eps)
        & (target_mld > eps)
        & mld_src.notnull()
        & target_mld.notnull()
    )
    if target_H is not None:
        can_warp = can_warp & (
            target_H - target_mld > eps
        )  # fully-mixed target -> identity

    ml = absz <= mld_src
    warp_mixed = absz * (target_mld / mld_src)
    warp_below = target_mld + (absz - mld_src)  # 1:1 in depth below the MLD
    warped = xr.where(ml, warp_mixed, warp_below)
    d_warp = xr.where(can_warp, warped, absz)

    n_depth = d_warp.sizes[depth_dim]
    d_warp = d_warp + xr.DataArray(np.arange(n_depth) * 1e-7, dims=[depth_dim])
    return d_warp.chunk({depth_dim: -1})


def build_bgc_vertical_coords(
    method: str,
    *,
    source_temp: "xr.DataArray",
    source_salt: "xr.DataArray",
    source_depth: "xr.DataArray",
    source_depth_dim: str,
    target_temp: "xr.DataArray",
    target_salt: "xr.DataArray",
    target_depth: "xr.DataArray",
    target_depth_dim: str,
) -> "tuple[xr.DataArray, xr.DataArray]":
    """Build the ``(source, target)`` vertical coordinate pair fed to
    ``VerticalRegrid.apply`` for non-depth BGC interpolation.

    For ``"density"`` both coordinates are potential-density coordinates (the depth
    arguments are unused). For ``"density_mld"`` the source is a warped depth that
    aligns the source MLD with the target MLD and the target is its real depth, so the
    transform interpolates linearly in depth within the two MLD segments.
    """
    if method == "density":
        return (
            _compute_density_coord(source_temp, source_salt, source_depth_dim),
            _compute_density_coord(target_temp, target_salt, target_depth_dim),
        )
    if method == "density_mld":
        target_sigma0 = compute_potential_density(target_temp, target_salt)
        target_mld = compute_mld(target_sigma0, target_depth, target_depth_dim)
        target_H = (
            np.abs(target_depth).where(target_sigma0.notnull()).max(target_depth_dim)
        )
        source_coord = _compute_mld_warp(
            source_temp,
            source_salt,
            source_depth,
            source_depth_dim,
            target_mld,
            target_H=target_H,
        )
        target_coord = np.abs(target_depth).chunk({target_depth_dim: -1})
        return source_coord, target_coord
    raise ValueError(
        f"Unknown BGC interpolation method {method!r}; "
        f"expected one of {BGC_INTERPOLATION_METHODS}."
    )


def compute_missing_surface_bgc_variables(bgc_data):
    """Fills in missing surface biogeochemical (BGC) variables in the input dictionary.

    This function checks for missing surface BGC variables in the provided dictionary and
    computes them based on predefined relationships with existing variables. The relationships
    specify either a multiplication factor applied to an existing variable or a constant value
    if no related variable is available. The resulting variables are added to the dictionary.

    Parameters
    ----------
    bgc_data : dict
        A dictionary containing surface biogeochemical variables as xarray DataArrays.
        Missing variables are computed and added to this dictionary.

    Returns
    -------
    dict
        The updated dictionary with missing surface BGC variables filled in.

    Notes
    -----
    - If `nox` and `nhy` are not part of the input dictionary, the are assigned constant values.
    """
    # Define the relationships for missing variables
    variable_relations = {
        "nox": (None, 1e-12),  # kg/m2/s
        "nhy": (None, 5e-12),  # kg/m2/s
    }

    # Fill in missing variables using the defined relationships
    for var_name, (base_var, factor) in variable_relations.items():
        if var_name not in bgc_data:
            if base_var:
                bgc_data[var_name] = bgc_data[base_var] * factor
            else:
                bgc_data[var_name] = factor * xr.ones_like(bgc_data["dust"])

    return bgc_data


# Canonical ROMS-MARBL tracer list for river forcing and related setup code.
# Defined here (not in river_datasets) so RiverForcing, BGC dataset classes, and
# tracer metadata share one schema without circular imports between setup and datasets.
MARBL_TRACER_NAMES = (
    "temp",
    "salt",
    "PO4",
    "NO3",
    "SiO3",
    "NH4",
    "Fe",
    "Lig",
    "O2",
    "DIC",
    "DIC_ALT_CO2",
    "ALK",
    "ALK_ALT_CO2",
    "DOC",
    "DON",
    "DOP",
    "DOPr",
    "DONr",
    "DOCr",
    "zooC",
    "spChl",
    "spC",
    "spP",
    "spFe",
    "spCaCO3",
    "diatChl",
    "diatC",
    "diatP",
    "diatFe",
    "diatSi",
    "diazChl",
    "diazC",
    "diazP",
    "diazFe",
)


def get_tracer_metadata_dict(
    include_bgc: bool = True,
    unit_type: Literal["concentration", "flux", "integrated"] = "concentration",
):
    """Generate a dictionary containing metadata for model tracers.

    The returned dictionary maps tracer names to their associated units and long names.
    Optionally includes biogeochemical tracers and can toggle between concentration and flux units.

    Parameters
    ----------
    include_bgc : bool, optional
        If True (default), includes biogeochemical tracers in the output.
        If False, returns only physical tracers (e.g., temperature, salinity).

    unit_type : str
        One of "concentration" (default), "flux", or "integrated".

    Returns
    -------
    dict
        A dictionary where keys are tracer names and values are dictionaries
        containing 'units' and 'long_name' for each tracer.
    """
    if include_bgc:
        tracer_names = list(MARBL_TRACER_NAMES)
    else:
        tracer_names = ["temp", "salt"]

    metadata = get_variable_metadata()

    tracer_dict = {}
    for tracer in tracer_names:
        if unit_type == "flux":
            unit = metadata[tracer]["flux_units"]
        elif unit_type == "integrated":
            unit = metadata[tracer].get("integrated_units", None)
        else:  # default to concentration units
            unit = metadata[tracer]["units"]

        tracer_dict[tracer] = {
            "units": unit,
            "long_name": metadata[tracer]["long_name"],
        }

    return tracer_dict


def add_tracer_metadata_to_ds(ds, include_bgc=True, with_flux_units=False):
    """Adds tracer metadata to a dataset.

    This function adds tracer metadata (name, unit, long name) as coordinates to
    the provided dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to which tracer metadata will be added.
    include_bgc : bool, optional
        If True (default), includes biogeochemical tracers in the output.
        If False, returns only physical tracers (e.g., temperature, salinity).
    with_flux_units : bool, optional
        If True, uses units appropriate for tracer fluxes (e.g., mmol/s).
        If False (default), uses units appropriate for tracer concentrations (e.g., mmol/m³).

    Returns
    -------
    xarray.Dataset
        The dataset with added tracer metadata.
    """
    unit_type = "flux" if with_flux_units else "concentration"
    tracer_dict = get_tracer_metadata_dict(include_bgc, unit_type=unit_type)

    tracer_names = list(tracer_dict.keys())
    tracer_units = [tracer_dict[tracer]["units"] for tracer in tracer_names]
    tracer_long_names = [tracer_dict[tracer]["long_name"] for tracer in tracer_names]

    ds = ds.assign_coords(
        tracer_name=("ntracers", tracer_names, {"long_name": "Tracer name"}),
        tracer_unit=(
            "ntracers",
            tracer_units,
            {
                "long_name": "Tracer flux unit"
                if with_flux_units
                else "Tracer concentration unit"
            },
        ),
        tracer_long_name=(
            "ntracers",
            tracer_long_names,
            {"long_name": "Tracer long name"},
        ),
    )

    return ds


def get_tracer_defaults() -> dict[str, float]:
    """Return constant default tracer concentrations for ROMS-MARBL.

    Values are read from ``river_tracer_defaults.nc`` (recommended values at
    ``value_option`` index 0) from the roms-tools-data repository.

    This accessor lives in ``setup.utils`` rather than ``river_datasets`` so
    ``RiverForcing``, fill sources, and other setup code can reuse the same
    defaults without pulling in dataset implementations at import time. The
    dataset class is loaded lazily in :func:`_load_tracer_defaults` to avoid a
    circular import: ``river_datasets`` imports :data:`MARBL_TRACER_NAMES` from
    here for schema validation.

    Returns
    -------
    dict
        Dictionary of tracer names and their default concentrations.
    """
    return _load_tracer_defaults()


@lru_cache(maxsize=1)
def _load_tracer_defaults() -> dict[str, float]:
    """Load and cache default tracer concentrations from ``river_tracer_defaults.nc``.

    ``RiverTracerDefaultsDataset`` is imported inside this function so
    ``river_datasets`` can import :data:`MARBL_TRACER_NAMES` from this module
    without a circular dependency at module load time.
    """
    from roms_tools.datasets.river_datasets import RiverTracerDefaultsDataset

    return RiverTracerDefaultsDataset().defaults


def extract_single_value(data):
    """Extracts a single value from an xarray.DataArray or numpy array.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        The data from which to extract the single value.

    Returns
    -------
    scalar
        The single value contained in the array.

    Raises
    ------
    ValueError
        If the data contains more than one element or is not a recognized type.
    """
    # Convert xarray.DataArray to numpy array if necessary
    if isinstance(data, xr.DataArray):
        data = data.values

    # Check that the data is a numpy array and contains only one element
    if isinstance(data, np.ndarray) and data.size == 1:
        return data.item()
    else:
        raise ValueError("Data must be a single-element array or DataArray.")


def group_dataset(ds, filepath):
    """Group the dataset into monthly or yearly subsets based on the frequency of the
    data.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be grouped.
    filepath : str
        The base filename for the output files.

    Returns
    -------
    tuple
        A tuple containing the list of grouped datasets and corresponding output filenames.
    """
    if hasattr(ds, "climatology"):
        output_filename = f"{filepath}_clim"
        output_filenames = [output_filename]
        dataset_list = [ds]
    else:
        if len(ds["abs_time"]) > 2:
            # Determine the frequency of the data
            abs_time_freq = pd.infer_freq(ds["abs_time"].to_index())
            if abs_time_freq:
                if abs_time_freq.lower() in [
                    "d",
                    "h",
                    "t",
                    "s",
                ]:  # Daily or higher frequency
                    dataset_list, output_filenames = group_by_month(ds, filepath)
                else:
                    dataset_list, output_filenames = group_by_year(ds, filepath)
            # If no regular spacing, default to year grouping
            else:
                dataset_list, output_filenames = group_by_year(ds, filepath)
        else:
            # Convert time index to datetime if not already
            abs_time_index = ds["abs_time"].to_index()
            # Determine if the entries are in the same month
            first_entry = abs_time_index[0]
            last_entry = abs_time_index[-1]

            if (
                first_entry.year == last_entry.year
                and first_entry.month == last_entry.month
            ):
                # Same month
                dataset_list, output_filenames = group_by_month(ds, filepath)
            else:
                # Different months, group by year
                dataset_list, output_filenames = group_by_year(ds, filepath)

    return dataset_list, output_filenames


def group_by_month(ds, filepath):
    """Group the dataset by month and generate filenames with 'YYYYMM' format.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be grouped.
    filepath : str
        The base filename for the output files.

    Returns
    -------
    tuple
        A tuple containing the list of monthly datasets and corresponding output filenames.
    """
    dataset_list = []
    output_filenames = []

    # Group dataset by year
    grouped_by_year = ds.groupby("abs_time.year")

    for year, yearly_dataset in grouped_by_year:
        # Further group each yearly group by month
        grouped_by_month = yearly_dataset.groupby("abs_time.month")

        for month, monthly_dataset in grouped_by_month:
            dataset_list.append(monthly_dataset)

            # Format: "filepath_YYYYMM.nc"
            year_month_str = f"{year}{month:02}"
            output_filename = f"{filepath}_{year_month_str}"
            output_filenames.append(output_filename)

    return dataset_list, output_filenames


def group_by_year(ds, filepath):
    """Group the dataset by year and generate filenames with 'YYYY' format.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be grouped.
    filepath : str
        The base filename for the output files.

    Returns
    -------
    tuple
        A tuple containing the list of yearly datasets and corresponding output filenames.
    """
    dataset_list = []
    output_filenames = []

    # Group dataset by year
    grouped_by_year = ds.groupby("abs_time.year")

    for year, yearly_dataset in grouped_by_year:
        dataset_list.append(yearly_dataset)

        # Format: "filepath_YYYY.nc"
        year_str = f"{year}"
        output_filename = f"{filepath}_{year_str}"
        output_filenames.append(output_filename)

    return dataset_list, output_filenames


def get_target_coords(
    grid: "Grid", use_coarse_grid: bool = False
) -> dict[str, xr.DataArray | bool | None]:
    """
    Retrieve longitude, latitude, and auxiliary grid coordinates, adjusting for
    longitude ranges and coarse grid usage.

    Parameters
    ----------
    grid : Grid
        Grid object.
    use_coarse_grid : bool, optional
        If True, use the coarse grid variables (`lat_coarse`, `lon_coarse`, etc.)
        instead of the native grid. Defaults to False.

    Returns
    -------
    dict[str, xr.DataArray | bool | None]
        Dictionary containing the following keys:

        - `"lat"` : xr.DataArray
            Latitude at rho points.
        - `"lon"` : xr.DataArray
            Longitude at rho points, adjusted to -180 to 180 or 0 to 360 range.
        - `"lat_psi"` : xr.DataArray | None
            Latitude at psi points, if available.
        - `"lon_psi"` : xr.DataArray | None
            Longitude at psi points, if available.
        - `"angle"` : xr.DataArray
            Grid rotation angle.
        - `"mask"` : xr.DataArray | None
            Land/sea mask at rho points.
        - `"straddle"` : bool
            True if the grid crosses the Greenwich meridian, False otherwise.

    Notes
    -----
    - If `grid.straddle` is False and the ROMS domain lies more than 5° from
      the Greenwich meridian, longitudes are adjusted to 0-360 range.
    - Renaming of coarse grid dimensions is applied to match the rho-point
      naming convention (`eta_rho`, `xi_rho`) for compatibility with ROMS-Tools.
    """
    # Select grid variables based on whether the coarse grid is used
    if use_coarse_grid:
        lat = grid.ds.lat_coarse.rename(
            {"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"}
        )
        lon = grid.ds.lon_coarse.rename(
            {"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"}
        )
        angle = grid.ds.angle_coarse.rename(
            {"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"}
        )
        mask = grid.ds.get("mask_coarse")
        if mask is not None:
            mask = mask.rename({"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"})

        lat_psi = grid.ds.get("lat_psi_coarse")
        lon_psi = grid.ds.get("lon_psi_coarse")

    else:
        lat = grid.ds.lat_rho
        lon = grid.ds.lon_rho
        angle = grid.ds.angle
        mask = grid.ds.get("mask_rho")
        lat_psi = grid.ds.get("lat_psi")
        lon_psi = grid.ds.get("lon_psi")

    # Operate on longitudes between -180 and 180 unless ROMS domain lies at least 5 degrees in lontitude away from Greenwich meridian
    lon = xr.where(lon > 180, lon - 360, lon)
    if lon_psi is not None:
        lon_psi = xr.where(lon_psi > 180, lon_psi - 360, lon_psi)

    straddle = True
    if not grid.straddle and abs(lon).min() > 5:
        lon = xr.where(lon < 0, lon + 360, lon)
        if lon_psi is not None:
            lon_psi = xr.where(lon_psi < 0, lon_psi + 360, lon_psi)
        straddle = False

    target_coords = {
        "lat": lat,
        "lon": lon,
        "lat_psi": lat_psi,
        "lon_psi": lon_psi,
        "angle": angle,
        "mask": mask,
        "straddle": straddle,
    }

    return target_coords


def compute_barotropic_velocity(
    vel: xr.DataArray, interface_depth: xr.DataArray
) -> xr.DataArray:
    """Compute barotropic (depth-averaged) velocity from 3D velocity.

    Assumes `vel` and `interface_depth` are at the same horizontal grid location.

    Parameters
    ----------
    vel : xarray.DataArray
        Velocity components (zonal and meridional) at u- and v-points.
    interface_depth : xarray.DataArray
        Depth values for computing layer thickness.

    Returns
    -------
    xarray.DataArray
        Depth-averaged velocity (`vel_bar`).

    Notes
    -----
    Computed as:
      - `vel_bar` = sum(dz * vel) / sum(dz)
    """
    # Layer thickness
    dz = -interface_depth.diff(dim="s_w")
    dz = dz.rename({"s_w": "s_rho"})

    vel_bar = (dz * vel).sum(dim="s_rho") / dz.sum(dim="s_rho")

    return vel_bar


def get_vector_pairs(variable_info):
    """Extracts all unique vector pairs from the variable_info dictionary.

    Parameters
    ----------
    variable_info : dict
        Dictionary containing variable information, including location,
        whether it's a vector, and its vector pair.

    Returns
    -------
    list of tuples
        List of unique vector pairs, where each tuple contains the names of
        the two vector components (e.g., ("u", "v")).
    """
    vector_pairs = []
    processed = set()  # Track variables that have already been paired

    for var_name, var_info in variable_info.items():
        if var_info["is_vector"] and var_name not in processed:
            vector_pair = var_info["vector_pair"]

            # Ensure the vector_pair exists in the dictionary and has not been processed
            if vector_pair and vector_pair in variable_info:
                vector_pairs.append((var_name, vector_pair))
                # Mark both the variable and its pair as processed
                processed.update([var_name, vector_pair])

    return vector_pairs


def gc_dist(lon1, lat1, lon2, lat2, input_in_degrees=True):
    """Calculate the great circle distance between two points on the Earth's surface
    using the Haversine formula.

    Latitude and longitude are assumed to be in degrees by default. If `input_in_degrees` is set to `False`,
    the input is assumed to already be in radians.

    This function is a wrapper for two numba-vectorized versions of the function, one each for degrees and radians.
    The wrapper is additionally needed to be able to use kwargs.

    Parameters
    ----------
    lon1, lat1 : float
        Longitude and latitude of the first point.
    lon2, lat2 : float
        Longitude and latitude of the second point.
    input_in_degrees : bool, optional
        If True (default), the input coordinates are assumed to be in degrees and will be converted to radians.
        If False, the input is assumed to be in radians and no conversion is applied.

    Returns
    -------
    dist : float
        The great circle distance between the two points in meters.

    Notes
    -----
    The radius of the Earth is taken to be 6371315 meters.
    """
    if input_in_degrees:
        return _gc_dist_degrees(lon1, lat1, lon2, lat2)
    return _gc_dist_radians(lon1, lat1, lon2, lat2)


@nb.vectorize(
    [nb.float64(nb.float64, nb.float64, nb.float64, nb.float64)], nopython=True
)
def _gc_dist_degrees(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance, given lat and lon in degrees.

    Returns
    -------
    Great circle distance in meters
    """
    # Convert degrees to radians
    d2r = np.pi / 180
    lon1 = lon1 * d2r
    lat1 = lat1 * d2r
    lon2 = lon2 * d2r
    lat2 = lat2 * d2r

    # Difference in latitudes and longitudes
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    dang = 2 * np.arcsin(
        np.sqrt(
            np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
    )

    # Distance in meters
    dis = R_EARTH * dang

    return dis


@nb.vectorize(
    [nb.float64(nb.float64, nb.float64, nb.float64, nb.float64)], nopython=True
)
def _gc_dist_radians(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance, given lat and lon in radians.

    Returns
    -------
    Great circle distance in meters
    """
    # Difference in latitudes and longitudes
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    dang = 2 * np.arcsin(
        np.sqrt(
            np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
    )

    # Distance in meters
    dis = R_EARTH * dang

    return dis


@nb.njit(
    [
        nb.float64[:, :](
            nb.float64[:, :],
            nb.float64[:, :],
            nb.int32[:, :],
        )
    ],
    parallel=True,
)
def min_dist_to_land(
    lon: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
):
    """Calculate the distance between one set of points (lon1, lat1) to the closest of
    another set of points (lon2, lat2).

    Parameters
    ----------
    lon : np.ndarray
        2-D Array of longitudes (in degrees) for all points on the grid
    lat : np.ndarray
        2-D Arrays of latitudes (in degrees) for all points on the grid
    mask: np.ndarray
        2-D integer array where ocean points have value 1 and land points are 0

    Returns
    -------
    2-D Array of the same shape as lon and lat, which will be filled with the resulting distance values
    to the nearest non-nan lon2, lat2 point
    """
    # get flattened ocean/land indices
    ocean = (mask == 1).ravel()
    land = (mask == 0).ravel()

    # get flattened and separate lon/lat arrays for ocean and land
    ocean_lon = lon.ravel()[ocean]
    ocean_lat = lat.ravel()[ocean]
    land_lon = lon.ravel()[land]
    land_lat = lat.ravel()[land]

    # keep track of the alignment between the full 2-D grid and the 1-D ocean indices
    # (nonzero() returns a tuple of the i, j indices where mask is 1)
    ocean_indices = mask.nonzero()

    # create a results array that will hold the distances from each ocean point to the nearest land point
    # initially fill arrays with zeros, as we will not do this calculation for land points, and land points
    # have zero distance to land by definition.
    result = np.zeros_like(lon)

    # iterate in parallel and do the distance calculation, taking the min for each ocean point without needing to
    # allocate a huge array for the entire calculation space
    for i in nb.prange(ocean_lon.shape[0]):
        result[ocean_indices[0][i], ocean_indices[1][i]] = np.min(
            _gc_dist_degrees(ocean_lon[i], ocean_lat[i], land_lon, land_lat)
        )

    return result


def convert_to_relative_days(
    times: Sequence[datetime] | np.ndarray,
    model_reference_date: datetime | np.datetime64,
) -> np.ndarray:
    """Convert absolute datetimes to model-relative time in days.

    Parameters
    ----------
    times : sequence of datetime or np.ndarray
        Absolute times to convert.
    model_reference_date : datetime or np.datetime64
        Reference date from which to compute relative days.

    Returns
    -------
    np.ndarray
        Times relative to the reference date, in days.
    """
    times = np.array(times, dtype="datetime64[ns]")
    ref = np.datetime64(model_reference_date, "ns")
    rel_times = (times - ref) / np.timedelta64(1, "D")

    return rel_times


def add_time_info_to_ds(
    ds: xr.Dataset,
    model_reference_date: datetime | np.datetime64,
    climatology: bool,
    time_name: str = "time",
) -> tuple[xr.Dataset, xr.DataArray]:
    """Add relative and absolute time coordinates to a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to which time information will be added.
    model_reference_date : datetime or np.datetime64
        The reference date for computing relative time.
    climatology : bool
        Whether the time data is climatological (cyclical over the year).
    time_name : str
        Name of the time coordinate in the dataset.

    Returns
    -------
    tuple[xr.Dataset, xr.DataArray]
        Updated dataset with time information and the relative time array.
    """
    if climatology:
        ds.attrs["climatology"] = str(True)
        month = xr.DataArray(range(1, 13), dims=time_name)
        month.attrs["long_name"] = "Month index (1-12)"
        ds = ds.assign_coords({"month": month})

        # Absolute time (for readability only)
        abs_time = np.datetime64(model_reference_date) + ds[time_name]

        # Custom relative time logic for climatology
        timedelta_index = pd.to_timedelta(ds[time_name].values)
        start_of_year = datetime(model_reference_date.year, 1, 1)
        offset = model_reference_date - start_of_year

        time = xr.DataArray(
            (timedelta_index - offset).view("int64") / 3600 / 24 * 1e-9,
            dims=time_name,
        )
        time.attrs["cycle_length"] = 365.25

    else:
        abs_time = ds[time_name]

        time = xr.DataArray(
            convert_to_relative_days(ds[time_name].values, model_reference_date),
            dims=time_name,
        )

    # Clean up and assign attributes
    abs_time.attrs.clear()
    abs_time.attrs["long_name"] = "absolute time"
    ds = ds.assign_coords({"abs_time": abs_time})

    time.attrs["long_name"] = f"relative time: days since {model_reference_date!s}"
    time.encoding["units"] = "days"
    time.attrs["units"] = "days"
    ds.encoding["unlimited_dims"] = time_name

    return ds, time


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def write_to_yaml(yaml_data, filepath: str | Path) -> None:
    """Write pre-serialized YAML data and additional metadata to a YAML file.

    This function writes the provided pre-serialized YAML data along with metadata, such as the version
    of the `roms-tools` package, to the specified file. The metadata header is written first, followed by
    the provided YAML data.

    Parameters
    ----------
    yaml_data : dict or str
        The pre-serialized YAML data to be written to the file. This data may include the forcing object and grid.
    filepath : Union[str, Path]
        The path (as a string or Path object) where the serialized YAML file will be saved.

    Returns
    -------
    None
        This function does not return anything. It writes the provided YAML data directly to the specified file.
    """
    # Convert the filepath to a Path object
    filepath = Path(filepath)

    # Create YAML header with version information
    try:
        roms_tools_version = importlib.metadata.version("roms-tools")
    except importlib.metadata.PackageNotFoundError:
        roms_tools_version = "unknown"

    header = f"---\nroms_tools_version: {roms_tools_version}\n---\n"

    # Write to YAML file
    with filepath.open("w") as file:
        # Write the header first
        file.write(header)
        # Write the serialized YAML data
        yaml.dump(
            yaml_data,
            file,
            Dumper=NoAliasDumper,
            default_flow_style=False,
            sort_keys=False,
        )


def serialize_paths(value: Any) -> Any:
    """Recursively convert Path objects to strings."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [serialize_paths(v) for v in value]
    if isinstance(value, dict):
        return {k: serialize_paths(v) for k, v in value.items()}
    return value


def normalize_paths(value: Any) -> Any:
    """Recursively convert path-like strings back to Path objects.

    Heuristic: strings containing '/' or ending with '.nc' are treated as paths.
    """
    if isinstance(value, str):
        # if the path looks like a URL, don't make it a PosixPath, or it will strip out the double //
        if "://" in value:
            return value
        return Path(value) if "/" in value or value.endswith(".nc") else value
    if isinstance(value, list):
        return [normalize_paths(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_paths(v) for k, v in value.items()}
    return value


def serialize_datetime(value: datetime | list[datetime] | Any) -> Any:
    """Convert datetime or list of datetimes to ISO 8601 strings."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list) and all(isinstance(v, datetime) for v in value):
        return [v.isoformat() for v in value]
    return value


def deserialize_datetime(
    value: str | list[str] | datetime | Any,
) -> datetime | list[datetime] | Any:
    """Convert ISO 8601 string(s) to datetime object(s).

    Returns:
        datetime if input is string,
        list of datetime if input is list of strings,
        original value if parsing fails or input is already datetime.
    """
    if isinstance(value, list):
        result: list[datetime | Any] = []
        for v in value:
            try:
                result.append(datetime.fromisoformat(str(v)))
            except ValueError:
                result.append(v)
        return result

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return value

    return value


def serialize_source_dict(
    src: dict[str, Any] | BaseModel | None,
) -> dict[str, Any] | None:
    """Serialize a source or BGC source dictionary for YAML or JSON output.

    This function performs the following transformations:
    - Converts any `Path` objects (including nested lists or dicts) to strings.
    - Serializes any nested `Grid` objects using `serialize_grid`.
    - Creates a deep copy of the input dictionary to avoid modifying the original.

    Parameters
    ----------
    src : dict[str, Any] | None
        The source or BGC source dictionary to serialize. Keys typically include:
        - "path": path(s) to files
        - "grid": a Grid object

    Returns
    -------
    dict[str, Any] | None
        A serialized dictionary suitable for saving to YAML or JSON, with:
        - Paths converted to strings
        - Nested Grid objects serialized
        Returns `None` if input `src` is `None`.
    """
    if src is None:
        return None

    if isinstance(src, BaseModel):
        src = src.model_dump(mode="python")
    else:
        src = deepcopy(src)

    # Serialize paths
    if "path" in src:
        src["path"] = serialize_paths(src["path"])

    # Serialize nested grid
    if "grid" in src and src["grid"] is not None:
        src["grid"] = serialize_grid(src["grid"])

    return src


def deserialize_source_dict(src: dict[str, Any] | None) -> dict[str, Any] | None:
    """Deserialize a source / bgc_source dictionary.

    Converts string paths back to Path objects.

    Parameters
    ----------
    src : dict[str, Any] | None
        Serialized source or bgc_source dictionary.

    Returns
    -------
    dict[str, Any] | None
        Dictionary with paths converted to Path objects.
    """
    if src is None:
        return None

    src = deepcopy(src)

    # Deserialize paths
    if "path" in src:
        src["path"] = normalize_paths(src["path"])

    return src


def serialize_grid(grid_obj: Any) -> dict[str, Any]:
    """Serialize a Grid object to a dictionary, excluding non-serializable attributes."""
    return pop_grid_data(asdict(grid_obj))


def pop_grid_data(grid_data: dict[str, Any]) -> dict[str, Any]:
    """Remove non-serializable or unnecessary keys from a Grid dictionary.

    Removes 'ds', 'straddle', and 'verbose' keys if present.

    Parameters
    ----------
    grid_data : dict
        Dictionary representation of a Grid object.

    Returns
    -------
    dict
        Cleaned dictionary suitable for serialization.
    """
    for key in ("ds", "straddle", "verbose"):
        grid_data.pop(key, None)
    return grid_data


def to_dict(forcing_object, exclude: list[str] | None = None) -> dict:
    """Serialize a forcing object (including its grid) into a dictionary.

    This function serializes a forcing object (dataclass or pydantic model),
    including its associated grid(s), into a dictionary suitable for YAML output.

    - Top-level grids (`grid`, `parent_grid`) are serialized consistently
    - Nested grids inside `source` and `bgc_source` are also serialized
    - Datetime objects are converted to ISO strings
    - Path objects are converted to strings

    Parameters
    ----------
    forcing_object : object
        A dataclass or pydantic model representing a forcing configuration.
    exclude : list[str], optional
        List of field names to exclude from serialization. The fields
        "grid", "parent_grid", and "ds" are always excluded.

    Returns
    -------
    dict
        Serialized representation of the forcing object.
    """
    exclude_list = exclude or []
    exclude_set: set[str] = {"grid", "parent_grid", "ds", "_bgc_dataset", *exclude_list}

    # --- Serialize top-level grid(s) ---
    yaml_data = {}

    if hasattr(forcing_object, "grid") and forcing_object.grid is not None:
        yaml_data["Grid"] = serialize_grid(forcing_object.grid)

    if (
        hasattr(forcing_object, "parent_grid")
        and forcing_object.parent_grid is not None
    ):
        yaml_data["ParentGrid"] = serialize_grid(forcing_object.parent_grid)

    # --- Collect forcing fields ---
    if isinstance(forcing_object, BaseModel):
        field_names = forcing_object.model_fields.keys()
    elif is_dataclass(forcing_object):
        field_names = [f.name for f in fields(forcing_object)]
    else:
        raise TypeError("Forcing object must be a dataclass or pydantic model")

    forcing_data = {}

    for name in field_names:
        if name in exclude_set:
            continue

        value = getattr(forcing_object, name)

        if name in {"source", "bgc_source"}:
            forcing_data[name] = serialize_source_dict(value)
            continue

        value = serialize_datetime(value)
        value = serialize_paths(value)

        forcing_data[name] = value

    # --- Final YAML structure ---
    yaml_data[forcing_object.__class__.__name__] = forcing_data

    return yaml_data


def from_yaml(forcing_object: type, filepath: str | Path) -> dict[str, Any]:
    """Load configuration for a forcing object from a YAML file.

    Searches for a dictionary keyed by the class name of `forcing_object` and
    returns it, converting:
    - ISO-format date strings to `datetime` objects
    - Path-like strings back to `Path` objects
    - `source` and `bgc_source` nested dictionaries back to proper Grid objects

    Parameters
    ----------
    forcing_object : type
        The class type whose configuration to load (e.g., `TidalForcing`).
    filepath : str | Path
        Path to the YAML file containing the configuration.

    Returns
    -------
    dict[str, Any]
        Dictionary of configuration parameters with dates, paths, and nested grids restored.

    Raises
    ------
    ValueError
        If no configuration for the specified class is found in the YAML file.
    """
    filepath = Path(filepath)
    with filepath.open("r") as f:
        documents = list(yaml.safe_load_all(f))

    forcing_data = None
    forcing_object_name = forcing_object.__name__

    for doc in documents:
        if doc is None:
            continue
        if forcing_object_name in doc:
            forcing_data = doc[forcing_object_name]
            break

    if forcing_data is None:
        raise ValueError(
            f"No {forcing_object_name} configuration found in the YAML file."
        )

    return deserialize_forcing_data(forcing_data)


def deserialize_forcing_data(forcing_data: dict[str, Any]) -> dict[str, Any]:
    """Restore datetimes, paths, and source/bgc_source dicts in a forcing-data block.

    Converts ISO date strings to ``datetime`` objects, path-like strings back to
    ``Path`` objects, and ``source``/``bgc_source`` nested dictionaries back to their
    proper form. Used for both the top-level forcing block and nested forcing blocks
    (e.g. an embedded ``physics_forcing``).
    """
    # Convert ISO date strings to datetime objects
    for key, value in forcing_data.items():
        forcing_data[key] = deserialize_datetime(value)

    # Convert path-like strings back to Path objects
    forcing_data = normalize_paths(forcing_data)

    # Deserialize source and bgc_source nested dictionaries
    for key in ["source", "bgc_source"]:
        if key in forcing_data:
            forcing_data[key] = deserialize_source_dict(forcing_data[key])

    return forcing_data


def handle_boundaries(field):
    """Adjust the boundaries of a 2D field by copying values from adjacent cells.

    Parameters
    ----------
    field : numpy.ndarray or xarray.DataArray
        A 2D array representing a field (e.g., topography or mask) whose boundary values
        need to be adjusted.

    Returns
    -------
    field : numpy.ndarray or xarray.DataArray
        The input field with adjusted boundary values.
    """
    field[0, :] = field[1, :]
    field[-1, :] = field[-2, :]
    field[:, 0] = field[:, 1]
    field[:, -1] = field[:, -2]

    return field


def get_boundary_coords():
    """This function determines the boundary points for the grid variables by specifying
    the indices for the south, east, north, and west boundaries.

    Returns
    -------
    dict
        A dictionary containing the boundary coordinates for different variable types.
        The dictionary has the following structure:
        - Keys: Variable types ("rho", "u", "v", "vector").
        - Values: Nested dictionaries that map each direction ("south", "east", "north", "west")
          to another dictionary specifying the boundary coordinates, represented by grid indices
          for the respective variable types. For example:
          - "rho" variables (e.g., `eta_rho`, `xi_rho`)
          - "u" variables (e.g., `xi_u`)
          - "v" variables (e.g., `eta_v`)
          - "vector" variables with lists of indices for multiple grid points (e.g., `eta_rho`, `xi_rho`).
    """
    bdry_coords = {
        "rho": {
            "south": {"eta_rho": 0},
            "east": {"xi_rho": -1},
            "north": {"eta_rho": -1},
            "west": {"xi_rho": 0},
        },
        "u": {
            "south": {"eta_rho": 0},
            "east": {"xi_u": -1},
            "north": {"eta_rho": -1},
            "west": {"xi_u": 0},
        },
        "v": {
            "south": {"eta_v": 0},
            "east": {"xi_rho": -1},
            "north": {"eta_v": -1},
            "west": {"xi_rho": 0},
        },
        "vector": {
            "south": {"eta_rho": [0, 1]},
            "east": {"xi_rho": [-2, -1]},
            "north": {"eta_rho": [-2, -1]},
            "west": {"xi_rho": [0, 1]},
        },
    }

    return bdry_coords


def to_float(val):
    """Convert a value or list of values to float.

    Parameters
    ----------
    val : float, int, or list of float/int
        A numeric value or a list of numeric values.

    Returns
    -------
    float or list of float
        The input value(s) converted to float type.
    """
    if isinstance(val, list):
        return [float(v) for v in val]
    return float(val)


def validate_names(
    names: list[str] | str,
    valid_names: list[str],
    include_all_sentinel: str,
    max_to_plot: int,
    label: str = "item",
) -> list[str]:
    """
    Generic validation and filtering for a list of names.

    Parameters
    ----------
    names : list of str or sentinel
        Names to validate, or sentinel value to include all valid names.
    valid_names : list of str
        List of valid names to check against.
    include_all_sentinel : str
        Sentinel value to indicate all names should be included.
    max_to_plot : int
        Maximum number of names to return.
    label : str, default "item"
        Label to use in error and warning messages.

    Returns
    -------
    list of str
        Validated and possibly truncated list of names.

    Raises
    ------
    ValueError
        If any names are invalid or input is not a list of strings.
    """
    if names == include_all_sentinel:
        names = valid_names

    if isinstance(names, list):
        if not all(isinstance(n, str) for n in names):
            raise ValueError(f"All elements in `{label}_names` must be strings.")
    else:
        raise ValueError(f"`{label}_names` should be a list of strings.")

    invalid = [n for n in names if n not in valid_names]
    if invalid:
        raise ValueError(f"Invalid {label}s: {', '.join(invalid)}")

    if len(names) > max_to_plot:
        logging.warning(
            f"Only the first {max_to_plot} {label}s will be plotted "
            f"(received {len(names)})."
        )
        names = names[:max_to_plot]

    return names


def check_and_set_boundaries(
    boundaries: dict[str, bool] | None,
    mask: xr.DataArray,
) -> dict[str, bool]:
    """
    Validate and finalize the `boundaries` dictionary.

    Parameters
    ----------
    boundaries : dict[str, bool] or None
        User-supplied dictionary controlling which boundaries are active.
        Keys may include any subset of {"south", "east", "north", "west"}.
        Missing keys will be filled from mask-based defaults.
        If None, all boundaries are inferred from the land mask.

    mask : xr.DataArray
        2D land/sea mask on rho-points. Used to determine which boundaries
        contain at least one ocean point.

    Returns
    -------
    dict[str, bool]
        Completed and validated boundary configuration.
    """
    valid_keys = {"south", "east", "north", "west"}

    # --------------------------------------------
    # Case 1: boundaries not provided → infer them
    # --------------------------------------------
    if boundaries is None:
        inferred = _infer_valid_boundaries_from_mask(mask)
        logging.info(f"No `boundaries` provided. Using mask-based defaults: {inferred}")
        return inferred

    # --------------------------------------------
    # Case 2: boundaries provided → validate
    # --------------------------------------------
    if not isinstance(boundaries, dict):
        raise TypeError(
            "`boundaries` must be a dict mapping boundary names to booleans."
        )

    # Unknown keys?
    unknown_keys = set(boundaries) - valid_keys
    if unknown_keys:
        raise ValueError(
            f"`boundaries` contains invalid keys: {unknown_keys}. "
            "Allowed keys are: 'south', 'east', 'north', 'west'."
        )

    # Type-check provided values
    for key, val in boundaries.items():
        if not isinstance(val, bool):
            raise TypeError(f"Boundary '{key}' must be a boolean.")

    # Fill missing boundaries using defaults
    inferred_defaults = _infer_valid_boundaries_from_mask(mask)
    completed = boundaries.copy()

    for key in valid_keys:
        if key not in completed:
            completed[key] = inferred_defaults[key]
            logging.info(
                f"`boundaries[{key!r}]` not provided — defaulting to "
                f"{inferred_defaults[key]}"
            )

    logging.info(f"Using boundary configuration: {completed}")
    return completed


def _infer_valid_boundaries_from_mask(mask: xr.DataArray) -> dict[str, bool]:
    """
    Determine which grid boundaries contain at least one ocean point.

    Any boundary consisting entirely of land is considered inactive.

    Parameters
    ----------
    mask : xr.DataArray
        2D mask array on rho-points where 1 = ocean, 0 = land.

    Returns
    -------
    dict[str, bool]
        Boolean availability for {south, east, north, west}.
    """
    bdry_coords = get_boundary_coords()
    boundaries = {}

    for direction in ["south", "east", "north", "west"]:
        coords = bdry_coords["rho"][direction]
        bdry_mask = mask.isel(**coords)

        # Boundary is valid if ANY ocean point exists
        boundaries[direction] = bool(bdry_mask.values.any())

    return boundaries
