import importlib.metadata
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Sequence, Type, Union

import cftime
import numba as nb
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pydantic import BaseModel

from roms_tools.constants import R_EARTH
from roms_tools.utils import interpolate_from_rho_to_u, interpolate_from_rho_to_v

yaml.SafeDumper.add_multi_representer(
    StrEnum,
    yaml.representer.SafeRepresenter.represent_str,
)


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


def one_dim_fill(da: xr.DataArray, dim: str, direction="forward") -> xr.DataArray:
    """Fill NaN values in a DataArray along a specified dimension.

    Parameters
    ----------
    da : xr.DataArray
        The input DataArray with NaN values to be filled, which must include the specified dimension.
    dim : str
        The name of the dimension along which to fill NaN values (e.g., 'depth' or 'time').
    direction : str, optional
        The filling direction; either "forward" to propagate non-NaN values downward or "backward" to propagate them upward.
        Defaults to "forward".

    Returns
    -------
    xr.DataArray
        A new DataArray with NaN values filled in the specified direction, leaving the original data unchanged.
    """
    if dim in da.dims:
        if direction == "forward":
            return da.ffill(dim=dim)
        elif direction == "backward":
            return da.bfill(dim=dim)
    return da


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


def interpolate_cyclic_time(
    data_array: xr.DataArray,
    time_dim_name: str,
    day_of_year: Union[
        int, float, np.ndarray, xr.DataArray, Sequence[Union[int, float]]
    ],
) -> xr.DataArray:
    """Interpolates a DataArray cyclically across the start and end of the year.

    This function extends the data cyclically by appending the last time step
    (shifted back by one year) at the beginning and the first time step
    (shifted forward by one year) at the end. It then performs linear interpolation
    to match the specified `day_of_year` values.

    Parameters
    ----------
    data_array : xr.DataArray
        The input data array containing a time-like dimension.
    time_dim_name : str
        The name of the time dimension in the dataset.
    day_of_year : Union[int, float, np.ndarray, xr.DataArray, Sequence[Union[int, float]]]
        The target day(s) of the year for interpolation. This can be:
        - A single integer or float representing the day of the year.
        - A NumPy array or xarray DataArray containing multiple days.
        - A list or tuple of integers or floats for multiple target days.

    Returns
    -------
    xr.DataArray
        The interpolated DataArray, ensuring cyclic continuity across year boundaries.

    Notes
    -----
    - This function is useful for interpolating climatological data, where the time axis
      represents a repeating annual cycle.
    - The `day_of_year` values should be within the range [1, 365] or [1, 366] for leap years.
    """
    # Concatenate across the beginning and end of the year
    time_concat = xr.concat(
        [
            data_array[time_dim_name][-1] - 365.25,  # Shift last time backward
            data_array[time_dim_name],
            data_array[time_dim_name][0] + 365.25,  # Shift first time forward
        ],
        dim=time_dim_name,
    )

    data_array_concat = xr.concat(
        [
            data_array.isel(
                **{time_dim_name: -1}
            ),  # Append last value at the beginning
            data_array,
            data_array.isel(**{time_dim_name: 0}),  # Append first value at the end
        ],
        dim=time_dim_name,
    )
    data_array_concat[time_dim_name] = time_concat

    # Interpolate to specified times
    data_array_interpolated = data_array_concat.interp(
        **{time_dim_name: day_of_year}, method="linear"
    )

    return data_array_interpolated


def interpolate_from_climatology(
    field: Union[xr.DataArray, xr.Dataset],
    time_dim_name: str,
    time: Union[xr.DataArray, pd.DatetimeIndex],
) -> Union[xr.DataArray, xr.Dataset]:
    """Interpolates a climatological field to specified time points.

    This function interpolates the input `field` based on `day_of_year` values
    extracted from the provided `time` points. If `field` is an `xarray.Dataset`,
    interpolation is applied to all its data variables individually.

    Parameters
    ----------
    field : xarray.DataArray or xarray.Dataset
        The input field to be interpolated.
        - If `field` is an `xarray.DataArray`, it must have a time dimension identified by `time_dim_name`.
        - If `field` is an `xarray.Dataset`, all variables within the dataset are interpolated along `time_dim_name`.
        The time dimension is assumed to represent `day_of_year` for climatological purposes.
    time_dim_name : str
        The name of the time dimension in `field`. This dimension is used for interpolation.
    time : xarray.DataArray or pandas.DatetimeIndex
        The target time points for interpolation. These are internally converted to `day_of_year`
        before performing interpolation.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The interpolated field, maintaining the same type (`xarray.DataArray` or `xarray.Dataset`)
        but aligned to the specified `time` values.

    Notes
    -----
    - This function assumes that `field` represents a climatological dataset, where time is expressed as `day_of_year` (1-365).
    - The `time` input is automatically converted to `day_of_year`, so manual conversion is not required before calling this function.
    """

    def interpolate_single_field(data_array: xr.DataArray) -> xr.DataArray:
        if isinstance(time, xr.DataArray):
            # Extract day of year from xarray.DataArray
            day_of_year = time.dt.dayofyear
        else:
            if np.size(time) == 1:
                # Convert single datetime64 object to pandas.Timestamp
                date = pd.Timestamp(time)
                day_of_year = (
                    date.dayofyear
                    + (date.hour / 24)
                    + (date.minute / 1440)
                    + (date.second / 86400)
                )
            else:
                # Convert each datetime64 object in the array to pandas.Timestamp and compute fractional day of year
                day_of_year = np.array(
                    [
                        pd.Timestamp(t).dayofyear
                        + (pd.Timestamp(t).hour / 24)
                        + (pd.Timestamp(t).minute / 1440)
                        + (pd.Timestamp(t).second / 86400)
                        for t in time
                    ]
                )

        data_array_interpolated = interpolate_cyclic_time(
            data_array, time_dim_name, day_of_year
        )

        if np.size(time) == 1:
            data_array_interpolated = data_array_interpolated.expand_dims(
                {time_dim_name: 1}
            )
        return data_array_interpolated

    if isinstance(field, xr.DataArray):
        return interpolate_single_field(field)
    elif isinstance(field, xr.Dataset):
        interpolated_data_vars = {
            var: interpolate_single_field(data_array)
            for var, data_array in field.data_vars.items()
        }
        return xr.Dataset(interpolated_data_vars, attrs=field.attrs)

    else:
        raise TypeError("Input 'field' must be an xarray.DataArray or xarray.Dataset.")


def get_time_type(data_array: xr.DataArray) -> str:
    """Determines the type of time values in the xarray DataArray.

    Parameters
    ----------
    data_array : xr.DataArray
        The xarray DataArray to be checked for time data types.

    Returns
    -------
    str
        A string indicating the type of the time data: 'cftime', 'datetime', or 'int'.

    Raises
    ------
    TypeError
        If the values in the DataArray are not of type numpy.ndarray or list.
    """
    # List of cftime datetime types
    cftime_types = (
        cftime.DatetimeNoLeap,
        cftime.DatetimeJulian,
        cftime.DatetimeGregorian,
        cftime.Datetime360Day,
        cftime.DatetimeProlepticGregorian,
    )

    # Check if any of the coordinate values are of cftime, datetime, or integer type
    if isinstance(data_array.values, (np.ndarray, list)):
        # Check if the data type is numpy datetime64, indicating standard datetime objects
        if data_array.values.dtype == "datetime64[ns]":
            return "datetime"

        # Check if any values in the array are instances of cftime types
        if any(isinstance(value, cftime_types) for value in data_array.values):
            return "cftime"

        # Check if all values are of integer type (e.g., for indices or time steps)
        if np.issubdtype(data_array.values.dtype, np.integer):
            return "int"

        # If none of the above conditions are met, raise a ValueError
        raise ValueError("Unsupported data type for time values in input dataset.")

    # Handle unexpected types
    raise TypeError("DataArray values must be of type numpy.ndarray or list.")


def convert_cftime_to_datetime(data_array: np.ndarray) -> np.ndarray:
    """Converts cftime datetime objects to numpy datetime64 objects in a numpy ndarray.

    Parameters
    ----------
    data_array : np.ndarray
        The numpy ndarray containing cftime datetime objects to be converted.

    Returns
    -------
    np.ndarray
        The ndarray with cftime datetimes converted to numpy datetime64 objects.

    Notes
    -----
    This function is intended to be used with numpy ndarrays. If you need to convert
    cftime datetime objects in an xarray.DataArray, please use the appropriate function
    to handle xarray.DataArray conversions.
    """
    # List of cftime datetime types
    cftime_types = (
        cftime.DatetimeNoLeap,
        cftime.DatetimeJulian,
        cftime.DatetimeGregorian,
    )

    # Define a conversion function for cftime to numpy datetime64
    def convert_datetime(dt):
        if isinstance(dt, cftime_types):
            # Convert to ISO format and then to nanosecond precision
            return np.datetime64(dt.isoformat(), "ns")
        return np.datetime64(dt, "ns")

    return np.vectorize(convert_datetime)(data_array)


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
        },
        "NO3": {
            "long_name": "dissolved inorganic nitrate",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "SiO3": {
            "long_name": "dissolved inorganic silicate",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "NH4": {
            "long_name": "dissolved ammonia",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "Fe": {
            "long_name": "dissolved inorganic iron",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "Lig": {
            "long_name": "iron binding ligand",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "O2": {
            "long_name": "dissolved oxygen",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "DIC": {
            "long_name": "dissolved inorganic carbon",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "DIC_ALT_CO2": {
            "long_name": "dissolved inorganic carbon, alternative CO2",
            "units": "mmol/m^3",
            "flux_units": "meq/s",
        },
        "ALK": {"long_name": "alkalinity", "units": "meq/m^3", "flux_units": "meq/s"},
        "ALK_ALT_CO2": {
            "long_name": "alkalinity, alternative CO2",
            "units": "meq/m^3",
            "flux_units": "meq/s",
        },
        "DOC": {
            "long_name": "dissolved organic carbon",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "DON": {
            "long_name": "dissolved organic nitrogen",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "DOP": {
            "long_name": "dissolved organic phosphorus",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "DOCr": {
            "long_name": "refractory dissolved organic carbon",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "DONr": {
            "long_name": "refractory dissolved organic nitrogen",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "DOPr": {
            "long_name": "refractory dissolved organic phosphorus",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "zooC": {
            "long_name": "zooplankton carbon",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "spChl": {
            "long_name": "small phytoplankton chlorophyll",
            "units": "mg/m^3",
            "flux_units": "mg/s",
        },
        "spC": {
            "long_name": "small phytoplankton carbon",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "spP": {
            "long_name": "small phytoplankton phosphorous",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "spFe": {
            "long_name": "small phytoplankton iron",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "spCaCO3": {
            "long_name": "small phytoplankton CaCO3",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "diatChl": {
            "long_name": "diatom chloropyll",
            "units": "mg/m^3",
            "flux_units": "mg/s",
        },
        "diatC": {
            "long_name": "diatom carbon",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "diatP": {
            "long_name": "diatom phosphorus",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "diatFe": {
            "long_name": "diatom iron",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "diatSi": {
            "long_name": "diatom silicate",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "diazChl": {
            "long_name": "diazotroph chloropyll",
            "units": "mg/m^3",
            "flux_units": "mg/s",
        },
        "diazC": {
            "long_name": "diazotroph carbon",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "diazP": {
            "long_name": "diazotroph phosphorus",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "diazFe": {
            "long_name": "diazotroph iron",
            "units": "mmol/m^3",
            "flux_units": "mmol/s",
        },
        "pco2_air": {"long_name": "atmospheric pCO2", "units": "ppmv"},
        "pco2_air_alt": {
            "long_name": "atmospheric pCO2, alternative CO2",
            "units": "ppmv",
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

        Assumptions:
        - If `pco2_air` is part of the input dictionary, it is in units of ppmv.

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
        "pco2_air_alt": ("pco2_air", 1.0),
        "nox": (None, 1e-12),  # kg/m2/s
        "nhy": (None, 5e-12),  # kg/m2/s
    }

    # Fill in missing variables using the defined relationships
    for var_name, (base_var, factor) in variable_relations.items():
        if var_name not in bgc_data:
            if base_var:
                bgc_data[var_name] = bgc_data[base_var] * factor
            else:
                bgc_data[var_name] = factor * xr.ones_like(bgc_data["pco2_air"])

    return bgc_data


def get_tracer_metadata_dict(include_bgc=True, with_flux_units=False):
    """Generate a dictionary containing metadata for model tracers.

    The returned dictionary maps tracer names to their associated units and long names.
    Optionally includes biogeochemical tracers and can toggle between concentration and flux units.

    Parameters
    ----------
    include_bgc : bool, optional
        If True (default), includes biogeochemical tracers in the output.
        If False, returns only physical tracers (e.g., temperature, salinity).

    with_flux_units : bool, optional
        If True, uses units appropriate for tracer fluxes (e.g., mmol/s).
        If False (default), uses units appropriate for tracer concentrations (e.g., mmol/m³).

    Returns
    -------
    dict
        A dictionary where keys are tracer names and values are dictionaries
        containing 'units' and 'long_name' for each tracer.
    """
    if include_bgc:
        tracer_names = [
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
        ]
    else:
        tracer_names = ["temp", "salt"]

    metadata = get_variable_metadata()

    tracer_dict = {
        tracer: {
            "units": metadata[tracer]["flux_units"]
            if with_flux_units
            else metadata[tracer]["units"],
            "long_name": metadata[tracer]["long_name"],
        }
        for tracer in tracer_names
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
    tracer_dict = get_tracer_metadata_dict(include_bgc, with_flux_units)

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
    """Returns constant default tracer concentrations for ROMS-MARBL.

    These values represent typical physical and biogeochemical tracer levels
    (e.g., temperature, salinity, nutrients, carbon) in freshwater.

    Returns
    -------
    dict
        Dictionary of tracer names and their default concentrations
    """
    return {
        "temp": 17.0,  # degrees C
        "salt": 1.0,  # psu
        "PO4": 2.7,  # mmol m-3
        "NO3": 24.2,  # mmol m-3
        "SiO3": 13.2,  # mmol m-3
        "NH4": 2.2,  # mmol m-3
        "Fe": 1.79,  # mmol m-3
        "Lig": 3 * 1.79,  # mmol m-3, inferred from Fe
        "O2": 187.5,  # mmol m-3
        "DIC": 2370.0,  # mmol m-3
        "DIC_ALT_CO2": 2370.0,  # mmol m-3
        "ALK": 2310.0,  # meq m-3
        "ALK_ALT_CO2": 2310.0,  # meq m-3
        "DOC": 1e-4,  # mmol m-3
        "DON": 1.0,  # mmol m-3
        "DOP": 0.1,  # mmol m-3
        "DOPr": 0.003,  # mmol m-3
        "DONr": 0.8,  # mmol m-3
        "DOCr": 1e-6,  # mmol m-3
        "zooC": 2.7,  # mmol m-3
        "spChl": 1.35,  # mg m-3
        "spC": 6.75,  # mmol m-3
        "spP": 1.5 * 0.03,  # mmol m-3, inferred from ?
        "spFe": 2.7e-5,  # mmol m-3
        "spCaCO3": 0.135,  # mmol m-3
        "diatChl": 0.135,  # mg m-3
        "diatC": 0.405,  # mmol m-3
        "diatP": 1.5 * 0.02,  # mmol m-3, inferred from ?
        "diatFe": 2.7e-6,  # mmol m-3
        "diatSi": 0.135,  # mmol m-3
        "diazChl": 0.015,  # mg m-3
        "diazC": 0.075,  # mmol m-3
        "diazP": 1.5 * 0.01,  # mmol m-3, inferred from ?
        "diazFe": 1.5e-6,  # mmol m-3
    }


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
            if abs_time_freq.lower() in [
                "d",
                "h",
                "t",
                "s",
            ]:  # Daily or higher frequency
                dataset_list, output_filenames = group_by_month(ds, filepath)
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


def get_target_coords(grid, use_coarse_grid=False):
    """Retrieves longitude and latitude coordinates from the grid, adjusting them based
    on longitude range.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information used for the model.
    use_coarse_grid : bool, optional
        Use coarse grid data if True. Defaults to False.

    Returns
    -------
    dict
        Dictionary containing the longitude, latitude, angle and mask arrays, along with a boolean indicating
        if the grid straddles the meridian.
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


def rotate_velocities(
    u: xr.DataArray, v: xr.DataArray, angle: xr.DataArray, interpolate: bool = True
) -> tuple[xr.DataArray, xr.DataArray]:
    """Rotate and optionally interpolate velocity components to align with grid
    orientation.

    Parameters
    ----------
    u : xarray.DataArray
        Zonal (east-west) velocity component at u-points.
    v : xarray.DataArray
        Meridional (north-south) velocity component at v-points.
    angle : xarray.DataArray
        Grid angle values for rotation.
    interpolate : bool, optional
        If True, interpolates rotated velocities to grid points (default is True).

    Returns
    -------
    tuple of xarray.DataArray
        Rotated velocity components (u_rot, v_rot).

    Notes
    -----
    - Rotation formulas:
      - u_rot = u * cos(angle) + v * sin(angle)
      - v_rot = v * cos(angle) - u * sin(angle)
    """
    # Rotate velocities to grid orientation
    u_rot = u * np.cos(angle) + v * np.sin(angle)
    v_rot = v * np.cos(angle) - u * np.sin(angle)

    # Interpolate to u- and v-points
    if interpolate:
        u_rot = interpolate_from_rho_to_u(u_rot)
        v_rot = interpolate_from_rho_to_v(v_rot)

    return u_rot, v_rot


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
    times: Union[Sequence[datetime], np.ndarray],
    model_reference_date: Union[datetime, np.datetime64],
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
    model_reference_date: Union[datetime, np.datetime64],
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

    time.attrs["long_name"] = f"relative time: days since {str(model_reference_date)}"
    time.encoding["units"] = "days"
    time.attrs["units"] = "days"
    ds.encoding["unlimited_dims"] = time_name

    return ds, time


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def write_to_yaml(yaml_data, filepath: Union[str, Path]) -> None:
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


def to_dict(forcing_object, exclude: list[str] | None = None) -> dict:
    """Serialize a forcing object (including its grid) into a dictionary.

    This function serializes a dataclass object (forcing_object) and its associated
    `grid` attribute into a dictionary. It omits fields like `grid` and `ds`
    that are not serializable or meant to be excluded.

    The function also converts datetime fields to ISO format strings for proper
    serialization.

    Parameters
    ----------
    forcing_object : object
        The object that contains the forcing data, typically a dataclass with attributes
        such as `grid`, `start_time`, `end_time`, etc.
    exclude : list[str], optional
        List of keys to exclude from the serialized output. Defaults to empty list. The field "ds" is always excluded by default.

    Returns
    -------
    dict
    """
    # Serialize Grid data
    if hasattr(forcing_object, "grid") and forcing_object.grid is not None:
        grid_data = asdict(forcing_object.grid)
        grid_yaml_data = {"Grid": pop_grid_data(grid_data)}
    elif hasattr(forcing_object, "parent_grid"):
        grid_data = asdict(forcing_object.parent_grid)
        grid_yaml_data = {"ParentGrid": pop_grid_data(grid_data)}

    # Ensure Paths are Strings
    def ensure_paths_are_strings(obj, key):
        attr = getattr(obj, key, None)
        if attr is not None and "path" in attr:
            paths = attr["path"]
            if isinstance(paths, list):
                attr["path"] = [str(p) if isinstance(p, Path) else p for p in paths]
            elif isinstance(paths, Path):
                attr["path"] = str(paths)
            elif isinstance(paths, dict):
                for key, path in paths.items():
                    attr["path"][key] = str(path)

    ensure_paths_are_strings(forcing_object, "source")
    ensure_paths_are_strings(forcing_object, "bgc_source")

    # Prepare Forcing Data
    forcing_data = {}
    if isinstance(forcing_object, BaseModel):
        field_names = forcing_object.model_fields
    elif is_dataclass(forcing_object):
        field_names = [field.name for field in fields(forcing_object)]
    else:
        raise TypeError("Forcing object is not a dataclass or pydantic model")

    if exclude is None:
        exclude = []
    exclude = ["grid", "parent_grid", "ds"] + exclude

    filtered_field_names = [param for param in field_names if param not in exclude]

    for field_name in filtered_field_names:
        # Retrieve the value of each field using getattr
        value = getattr(forcing_object, field_name)

        # If the field is a datetime object, convert it to ISO format
        if isinstance(value, datetime):
            value = value.isoformat()
        # Convert list of datetimes to list of ISO strings
        elif isinstance(value, list) and all(isinstance(v, datetime) for v in value):
            value = [v.isoformat() for v in value]

        # Add the field and its value to the forcing_data dictionary
        forcing_data[field_name] = value

    # Combine Grid and Forcing Data into a single dictionary for the final YAML content
    yaml_data = {
        **grid_yaml_data,  # Add the grid data to the final YAML structure
        forcing_object.__class__.__name__: forcing_data,  # Include the serialized forcing object data
    }

    return yaml_data


def pop_grid_data(grid_data):
    grid_data.pop("ds", None)  # Remove 'ds' attribute (non-serializable)
    grid_data.pop("straddle", None)
    grid_data.pop("verbose", None)

    return grid_data


def from_yaml(forcing_object: Type, filepath: Union[str, Path]) -> Dict[str, Any]:
    """Extract the configuration data for a given forcing object from a YAML file.

    This function reads a YAML file, searches for the configuration data associated
    with the class name of the forcing object, and returns the configuration data
    as a dictionary. The dictionary contains the forcing parameters extracted from
    the YAML file, with any date fields converted from ISO format.

    Parameters
    ----------
    filepath : Union[str, Path]
        The path to the YAML file from which the parameters will be read.
    forcing_object : Type
        The class type (e.g., TidalForcing) whose configuration data is to be loaded
        from the YAML file. The class name is used to locate the relevant data in
        the YAML structure.

    Returns
    -------
    dict
        A dictionary containing the forcing parameters extracted from the YAML file.
        This dictionary contains key-value pairs where the keys are the parameter
        names, and the values are the corresponding values from the YAML file.
        Any date fields are converted from ISO format if necessary.

    Raises
    ------
    ValueError
        If no configuration for the specified class name is found in the YAML file.
    """
    # Ensure filepath is a Path object
    filepath = Path(filepath)

    # Read the entire file content
    with filepath.open("r") as file:
        file_content = file.read()

    # Split the content into YAML documents
    documents = list(yaml.safe_load_all(file_content))

    forcing_data = None
    forcing_object_name = forcing_object.__name__

    # Process the YAML documents to find the forcing data for the given object
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

    # Convert any date fields from ISO format if necessary
    for key, value in forcing_data.items():
        forcing_data[key] = _convert_from_iso_format(value)

    # Return the forcing data as a dictionary
    return forcing_data


def _convert_from_iso_format(value):
    try:
        # Return the parsed datetime object if successful
        return datetime.fromisoformat(str(value))
    except ValueError:
        # Return None or raise an exception if parsing fails
        return value


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


def wrap_longitudes(grid_ds, straddle):
    """Adjusts longitude values in a dataset to handle dateline crossing.

    Parameters
    ----------
    grid_ds : xr.Dataset
        The dataset containing longitude variables to adjust.
    straddle : bool
        If True, adjusts longitudes to the range [-180, 180] for datasets
        that straddle the dateline. If False, adjusts longitudes to the
        range [0, 360].

    Returns
    -------
    xr.Dataset
        The dataset with adjusted longitude values.
    """
    for lon_dim in ["lon_rho", "lon_u", "lon_v"]:
        if straddle:
            grid_ds[lon_dim] = xr.where(
                grid_ds[lon_dim] > 180,
                grid_ds[lon_dim] - 360,
                grid_ds[lon_dim],
            )
        else:
            grid_ds[lon_dim] = xr.where(
                grid_ds[lon_dim] < 0, grid_ds[lon_dim] + 360, grid_ds[lon_dim]
            )

    return grid_ds


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
