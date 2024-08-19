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
