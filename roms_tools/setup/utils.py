import xarray as xr
import numpy as np
from typing import Union
import pandas as pd
import cftime


def nan_check(field, mask) -> None:
    """
    Checks for NaN values at wet points in the field.

    This function examines the interpolated input field for NaN values at positions indicated as wet points by the mask.
    If any NaN values are found at these wet points, a ValueError is raised.

    Parameters
    ----------
    field : array-like
        The data array to be checked for NaN values. This is typically an xarray.DataArray or numpy array.

    mask : array-like
        A boolean mask or data array with the same shape as `field`. The wet points (usually ocean points)
        are indicated by `1` or `True`, and land points by `0` or `False`.

    Raises
    ------
    ValueError
        If the field contains NaN values at any of the wet points indicated by the mask.
        The error message will explain the potential cause and suggest ensuring the dataset's coverage.

    """

    # Replace values in field with 0 where mask is not 1
    da = xr.where(mask == 1, field, 0)

    # Check if any NaN values exist in the modified field
    if da.isnull().any().values:
        raise ValueError(
            "NaN values found in interpolated field. This likely occurs because the ROMS grid, including "
            "a small safety margin for interpolation, is not fully contained within the dataset's longitude/latitude range. Please ensure that the "
            "dataset covers the entire area required by the ROMS grid."
        )


def substitute_nans_by_fillvalue(field, fill_value=0.0) -> xr.DataArray:
    """
    Replace NaN values in the field with a specified fill value.

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


def interpolate_from_rho_to_u(field, method="additive"):

    """
    Interpolates the given field from rho points to u points.

    This function performs an interpolation from the rho grid (cell centers) to the u grid
    (cell edges in the xi direction). Depending on the chosen method, it either averages
    (additive) or multiplies (multiplicative) the field values between adjacent rho points
    along the xi dimension. It also handles the removal of unnecessary coordinate variables
    and updates the dimensions accordingly.

    Parameters
    ----------
    field : xr.DataArray
        The input data array on the rho grid to be interpolated. It is assumed to have a dimension
        named "xi_rho".

    method : str, optional, default='additive'
        The method to use for interpolation. Options are:
        - 'additive': Average the field values between adjacent rho points.
        - 'multiplicative': Multiply the field values between adjacent rho points. Appropriate for
          binary masks.

    Returns
    -------
    field_interpolated : xr.DataArray
        The interpolated data array on the u grid with the dimension "xi_u".
    """

    if method == "additive":
        field_interpolated = 0.5 * (field + field.shift(xi_rho=1)).isel(
            xi_rho=slice(1, None)
        )
    elif method == "multiplicative":
        field_interpolated = (field * field.shift(xi_rho=1)).isel(xi_rho=slice(1, None))
    else:
        raise NotImplementedError(f"Unsupported method '{method}' specified.")

    if "lat_rho" in field_interpolated.coords:
        field_interpolated.drop_vars(["lat_rho"])
    if "lon_rho" in field_interpolated.coords:
        field_interpolated.drop_vars(["lon_rho"])

    field_interpolated = field_interpolated.swap_dims({"xi_rho": "xi_u"})

    return field_interpolated


def interpolate_from_rho_to_v(field, method="additive"):

    """
    Interpolates the given field from rho points to v points.

    This function performs an interpolation from the rho grid (cell centers) to the v grid
    (cell edges in the eta direction). Depending on the chosen method, it either averages
    (additive) or multiplies (multiplicative) the field values between adjacent rho points
    along the eta dimension. It also handles the removal of unnecessary coordinate variables
    and updates the dimensions accordingly.

    Parameters
    ----------
    field : xr.DataArray
        The input data array on the rho grid to be interpolated. It is assumed to have a dimension
        named "eta_rho".

    method : str, optional, default='additive'
        The method to use for interpolation. Options are:
        - 'additive': Average the field values between adjacent rho points.
        - 'multiplicative': Multiply the field values between adjacent rho points. Appropriate for
          binary masks.

    Returns
    -------
    field_interpolated : xr.DataArray
        The interpolated data array on the v grid with the dimension "eta_v".
    """

    if method == "additive":
        field_interpolated = 0.5 * (field + field.shift(eta_rho=1)).isel(
            eta_rho=slice(1, None)
        )
    elif method == "multiplicative":
        field_interpolated = (field * field.shift(eta_rho=1)).isel(
            eta_rho=slice(1, None)
        )
    else:
        raise NotImplementedError(f"Unsupported method '{method}' specified.")

    if "lat_rho" in field_interpolated.coords:
        field_interpolated.drop_vars(["lat_rho"])
    if "lon_rho" in field_interpolated.coords:
        field_interpolated.drop_vars(["lon_rho"])

    field_interpolated = field_interpolated.swap_dims({"eta_rho": "eta_v"})

    return field_interpolated


def extrapolate_deepest_to_bottom(field: xr.DataArray, dim: str) -> xr.DataArray:
    """
    Extrapolate the deepest non-NaN values to the bottom along a specified dimension.

    Parameters
    ----------
    field : xr.DataArray
        The input data array containing NaN values that need to be filled. This array
        should have at least one dimension named by `dim`.
    dim : str
        The name of the dimension along which to perform the interpolation and extrapolation.
        Typically, this would be a vertical dimension such as 'depth' or 's_rho'.

    Returns
    -------
    field_interpolated : xr.DataArray
        A new data array with NaN values along the specified dimension filled by nearest
        neighbor interpolation and extrapolation to the bottom. The original data array is not modified.

    """
    field_interpolated = field.interpolate_na(
        dim=dim, method="nearest", fill_value="extrapolate"
    )

    return field_interpolated


def assign_dates_to_climatology(ds: xr.Dataset, time_dim: str) -> xr.Dataset:
    """
    Assigns climatology dates to the dataset's time dimension.

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


def interpolate_from_climatology(
    field: Union[xr.DataArray, xr.Dataset],
    time_dim_name: str,
    time: Union[xr.DataArray, pd.DatetimeIndex],
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Interpolates the given field temporally based on the specified time points.

    If `field` is an xarray.Dataset, this function applies the interpolation to all data variables in the dataset.

    Parameters
    ----------
    field : xarray.DataArray or xarray.Dataset
        The field data to be interpolated. Can be a single DataArray or a Dataset.
    time_dim_name : str
        The name of the dimension in `field` that represents time.
    time : xarray.DataArray or pandas.DatetimeIndex
        The target time points for interpolation.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The field values interpolated to the specified time points. The type matches the input type.
    """

    def interpolate_single_field(data_array: xr.DataArray) -> xr.DataArray:

        if isinstance(time, xr.DataArray):
            # Extract day of year from xarray.DataArray
            day_of_year = time.dt.dayofyear
        else:
            if np.size(time) == 1:
                day_of_year = time.timetuple().tm_yday
            else:
                day_of_year = np.array([t.timetuple().tm_yday for t in time])

        data_array[time_dim_name] = data_array[time_dim_name].dt.days

        # Concatenate across the beginning and end of the year
        time_concat = xr.concat(
            [
                data_array[time_dim_name][-1] - 365.25,
                data_array[time_dim_name],
                365.25 + data_array[time_dim_name][0],
            ],
            dim=time_dim_name,
        )
        data_array_concat = xr.concat(
            [
                data_array.isel(**{time_dim_name: -1}),
                data_array,
                data_array.isel(**{time_dim_name: 0}),
            ],
            dim=time_dim_name,
        )
        data_array_concat[time_dim_name] = time_concat

        # Interpolate to specified times
        data_array_interpolated = data_array_concat.interp(
            **{time_dim_name: day_of_year}, method="linear"
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


def is_cftime_datetime(data_array: xr.DataArray) -> bool:
    """
    Checks if the xarray DataArray contains cftime datetime objects.

    Parameters
    ----------
    data_array : xr.DataArray
        The xarray DataArray to be checked for cftime datetime objects.

    Returns
    -------
    bool
        True if the DataArray contains cftime datetime objects, False otherwise.

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
    )

    # Check if any of the coordinate values are of cftime type
    if isinstance(data_array.values, (np.ndarray, list)):
        # Check the dtype of the array; numpy datetime64 indicates it's not cftime
        if data_array.values.dtype == "datetime64[ns]":
            return False

        # Check if any of the values in the array are instances of cftime types
        return any(isinstance(value, cftime_types) for value in data_array.values)

    # Handle unexpected types
    raise TypeError("DataArray values must be of type numpy.ndarray or list.")


def convert_cftime_to_datetime(data_array: np.ndarray) -> np.ndarray:
    """
    Converts cftime datetime objects to numpy datetime64 objects in a numpy ndarray.

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
    """
    Retrieves metadata for commonly used variables in the dataset.

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
        "temp": {"long_name": "potential temperature", "units": "degrees Celsius"},
        "salt": {"long_name": "salinity", "units": "PSU"},
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
        "PO4": {"long_name": "dissolved inorganic phosphate", "units": "mmol/m^3"},
        "NO3": {"long_name": "dissolved inorganic nitrate", "units": "mmol/m^3"},
        "SiO3": {"long_name": "dissolved inorganic silicate", "units": "mmol/m^3"},
        "NH4": {"long_name": "dissolved ammonia", "units": "mmol/m^3"},
        "Fe": {"long_name": "dissolved inorganic iron", "units": "mmol/m^3"},
        "Lig": {"long_name": "iron binding ligand", "units": "mmol/m^3"},
        "O2": {"long_name": "dissolved oxygen", "units": "mmol/m^3"},
        "DIC": {"long_name": "dissolved inorganic carbon", "units": "mmol/m^3"},
        "DIC_ALT_CO2": {
            "long_name": "dissolved inorganic carbon, alternative CO2",
            "units": "mmol/m^3",
        },
        "ALK": {"long_name": "alkalinity", "units": "meq/m^3"},
        "ALK_ALT_CO2": {
            "long_name": "alkalinity, alternative CO2",
            "units": "meq/m^3",
        },
        "DOC": {"long_name": "dissolved organic carbon", "units": "mmol/m^3"},
        "DON": {"long_name": "dissolved organic nitrogen", "units": "mmol/m^3"},
        "DOP": {"long_name": "dissolved organic phosphorus", "units": "mmol/m^3"},
        "DOCr": {
            "long_name": "refractory dissolved organic carbon",
            "units": "mmol/m^3",
        },
        "DONr": {
            "long_name": "refractory dissolved organic nitrogen",
            "units": "mmol/m^3",
        },
        "DOPr": {
            "long_name": "refractory dissolved organic phosphorus",
            "units": "mmol/m^3",
        },
        "zooC": {"long_name": "zooplankton carbon", "units": "mmol/m^3"},
        "spChl": {
            "long_name": "small phytoplankton chlorophyll",
            "units": "mg/m^3",
        },
        "spC": {"long_name": "small phytoplankton carbon", "units": "mmol/m^3"},
        "spP": {
            "long_name": "small phytoplankton phosphorous",
            "units": "mmol/m^3",
        },
        "spFe": {"long_name": "small phytoplankton iron", "units": "mmol/m^3"},
        "spCaCO3": {"long_name": "small phytoplankton CaCO3", "units": "mmol/m^3"},
        "diatChl": {"long_name": "diatom chloropyll", "units": "mg/m^3"},
        "diatC": {"long_name": "diatom carbon", "units": "mmol/m^3"},
        "diatP": {"long_name": "diatom phosphorus", "units": "mmol/m^3"},
        "diatFe": {"long_name": "diatom iron", "units": "mmol/m^3"},
        "diatSi": {"long_name": "diatom silicate", "units": "mmol/m^3"},
        "diazChl": {"long_name": "diazotroph chloropyll", "units": "mg/m^3"},
        "diazC": {"long_name": "diazotroph carbon", "units": "mmol/m^3"},
        "diazP": {"long_name": "diazotroph phosphorus", "units": "mmol/m^3"},
        "diazFe": {"long_name": "diazotroph iron", "units": "mmol/m^3"},
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


def get_boundary_info():

    """
    This function provides information about the boundary points for the rho, u, and v
    variables on the grid, specifying the indices for the south, east, north, and west
    boundaries.

    Returns
    -------
    dict
        A dictionary where keys are variable types ("rho", "u", "v"), and values
        are nested dictionaries mapping directions ("south", "east", "north", "west")
        to the corresponding boundary coordinates.
    """

    # Boundary coordinates
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
    }

    return bdry_coords


def extract_single_value(data):
    """
    Extracts a single value from an xarray.DataArray or numpy array.

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
