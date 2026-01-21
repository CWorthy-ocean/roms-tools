import logging
from datetime import datetime, timedelta

import cftime
import numpy as np
import xarray as xr

from roms_tools.fill import one_dim_fill
from roms_tools.utils import interpolate_from_climatology


def extrapolate_deepest_to_bottom(ds: xr.Dataset, depth_dim: str) -> xr.Dataset:
    """Extrapolate the deepest non-NaN values downward along a depth dimension.

    For each variable in the dataset that includes the specified depth dimension,
    missing values at the bottom are filled by propagating the deepest available
    data downward.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing variables with a depth dimension.
    depth_dim : str
        Name of the depth dimension (e.g., 's_rho') along which to extrapolate.

    Returns
    -------
    xr.Dataset
        Dataset with bottom NaNs filled along the specified depth dimension.
    """
    for var_name in ds.data_vars:
        if depth_dim in ds[var_name].dims:
            ds[var_name] = one_dim_fill(ds[var_name], depth_dim, direction="forward")

    return ds


def convert_to_float64(ds: xr.Dataset) -> xr.Dataset:
    """Convert all non-mask data variables to float64.

    Variables whose names start with ``"mask_"`` are left unchanged.
    """
    dtype_map = {
        name: ("float64" if not name.startswith("mask_") else var.dtype)
        for name, var in ds.data_vars.items()
    }

    return ds.astype(dtype_map)


def check_dataset(
    ds: xr.Dataset,
    dim_names: dict[str, str] | None = None,
    var_names: dict[str, str] | None = None,
    opt_var_names: dict[str, str] | None = None,
) -> None:
    """Check if the dataset contains the specified variables and dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset to check.
    dim_names: dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
    var_names: dict[str, str], optional
        Dictionary of variable names that are required in the dataset.
    opt_var_names : dict[str, str], optional
        Dictionary of optional variable names.
        These variables are not strictly required, and the function will not raise an error if they are missing.
        Default is None, meaning no optional variables are considered.


    Raises
    ------
    ValueError
        If the dataset does not contain the specified variables or dimensions.
    """
    if dim_names:
        missing_dims = [dim for dim in dim_names.values() if dim not in ds.dims]
        if missing_dims:
            raise ValueError(
                f"Dataset does not contain all required dimensions. The following dimensions are missing: {missing_dims}"
            )

    if var_names:
        missing_vars = [var for var in var_names.values() if var not in ds.data_vars]
        if missing_vars:
            raise ValueError(
                f"Dataset does not contain all required variables. The following variables are missing: {missing_vars}"
            )

    if opt_var_names:
        missing_optional_vars = [
            var for var in opt_var_names.values() if var not in ds.data_vars
        ]
        if missing_optional_vars:
            logging.warning(
                f"Optional variables missing (but not critical): {missing_optional_vars}"
            )


def validate_start_end_time(
    start_time: datetime | None = None, end_time: datetime | None = None
) -> None:
    """
    Validate the provided start and end times.

    Parameters
    ----------
    start_time : datetime or None
        Start of the time interval. Must be a `datetime` object if provided.
    end_time : datetime or None
        End of the time interval. Must be a `datetime` object if provided.

    Raises
    ------
    TypeError
        If `start_time` or `end_time` is provided but is not a `datetime`.
    ValueError
        If both `start_time` and `end_time` are provided and
        `end_time` occurs before `start_time`.
    """
    if start_time is not None and not isinstance(start_time, datetime):
        raise TypeError(
            f"`start_time` must be a datetime object or None, "
            f"but got {type(start_time).__name__}."
        )

    if end_time is not None and not isinstance(end_time, datetime):
        raise TypeError(
            f"`end_time` must be a datetime object or None, "
            f"but got {type(end_time).__name__}."
        )

    if start_time is not None and end_time is not None:
        if end_time < start_time:
            raise ValueError(
                f"`end_time` ({end_time}) cannot be earlier than "
                f"`start_time` ({start_time})."
            )


def select_relevant_fields(ds: xr.Dataset, var_names: list[str]) -> xr.Dataset:
    """
    Return a subset of the dataset containing only the specified variables.

    All data variables not listed in ``var_names`` are removed, except for the
    special variable ``"mask"``, which is always retained if present.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset from which variables will be selected.
    var_names : list of str
        Names of variables that should be kept in the resulting dataset.

    Returns
    -------
    xr.Dataset
        A new dataset containing only the variables in ``var_names`` and
        ``"mask"`` (if it exists in the input dataset).
    """
    vars_to_keep = set(var_names)
    vars_to_drop = [
        var for var in ds.data_vars if var not in vars_to_keep and var != "mask"
    ]

    if vars_to_drop:
        ds = ds.drop_vars(vars_to_drop)

    return ds


def select_relevant_times(
    ds: xr.Dataset,
    time_dim: str,
    time_coord: str,
    start_time: datetime,
    end_time: datetime | None = None,
    climatology: bool = False,
    allow_flex_time: bool = False,
) -> xr.Dataset:
    """
    Select a subset of the dataset based on time constraints.

    This function supports two main use cases:

    1. **Time range selection (start_time + end_time provided):**
       - Returns all records strictly between `start_time` and `end_time`.
       - Ensures at least one record at or before `start_time` and one record at or
         after `end_time` are included, even if they fall outside the strict range.

    2. **Initial condition selection (start_time provided, end_time=None):**
       - Delegates to `_select_initial_time`, which reduces the dataset to exactly one
         time entry.
       - If `allow_flex_time=True`, a +24-hour buffer around `start_time` is allowed,
         and the closest timestamp is chosen.
       - If `allow_flex_time=False`, requires an exact timestamp match.

    Additional behavior:
    - If `climatology=True`, the dataset must contain exactly 12 time steps. If valid,
      the climatology dataset is returned without further filtering.
    - If the dataset uses `cftime` datetime objects, these are converted to
      `np.datetime64` before filtering.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to filter. Must contain a valid time dimension.
    time_dim : str
        Name of the time dimension in `ds`.
    time_coord : str
        Name of the time coordinate in `ds`.
    start_time : datetime
        Start time for filtering.
    end_time : datetime or None
        End time for filtering. If `None`, the function assumes an initial condition
        use case and selects exactly one timestamp.
    climatology : bool, optional
        If True, requires exactly 12 time steps and bypasses normal filtering.
        Defaults to False.
    allow_flex_time : bool, optional
        Whether to allow a +24h search window after `start_time` when `end_time`
        is None. If False (default), requires an exact match.

    Returns
    -------
    xr.Dataset
        A filtered dataset containing only the selected time entries.

    Raises
    ------
    ValueError
        - If `climatology=True` but the dataset does not contain exactly 12 time steps.
        - If `climatology=False` and the dataset contains integer time values.
        - If no valid records are found within the requested range or window.

    Warns
    -----
    UserWarning
        - If no records exist at or before `start_time` or at or after `end_time`.
        - If the specified time dimension does not exist in the dataset.

    Notes
    -----
    - For initial conditions (end_time=None), see `_select_initial_time` for details
      on strict vs. flexible selection behavior.
    - Logs warnings instead of failing hard when boundary records are missing, and
      defaults to using the earliest or latest available time in such cases.
    """
    if time_dim not in ds.dims:
        logging.warning(
            f"Dataset does not contain time dimension '{time_dim}'. "
            "Please check variable naming or dataset structure."
        )
        return ds

    if time_coord not in ds.variables:
        logging.warning(
            f"Dataset does not contain time coordinate '{time_coord}'. "
            "Please check variable naming or dataset structure."
        )
        return ds

    time_type = get_time_type(ds[time_coord])

    if climatology:
        if len(ds[time_coord]) != 12:
            raise ValueError(
                f"The dataset contains {len(ds[time_coord])} time steps, but the climatology flag is set to True, which requires exactly 12 time steps."
            )
    else:
        if time_type == "int":
            raise ValueError(
                "The dataset contains integer time values, which are only supported when the climatology flag is set to True. However, your climatology flag is set to False."
            )
    if time_type == "cftime":
        ds = ds.assign_coords({time_dim: convert_cftime_to_datetime(ds[time_coord])})

    if not end_time:
        # Assume we are looking for exactly one time record for initial conditions
        return _select_initial_time(
            ds, time_dim, time_coord, start_time, climatology, allow_flex_time
        )

    if climatology:
        return ds

    # Identify records before or at start_time
    before_start = ds[time_coord] <= np.datetime64(start_time)
    if before_start.any():
        closest_before_start = ds[time_coord].where(before_start, drop=True)[-1]
    else:
        logging.warning(f"No records found at or before the start_time: {start_time}.")
        closest_before_start = ds[time_coord][0]

    # Identify records after or at end_time
    after_end = ds[time_coord] >= np.datetime64(end_time)
    if after_end.any():
        closest_after_end = ds[time_coord].where(after_end, drop=True).min()
    else:
        logging.warning(f"No records found at or after the end_time: {end_time}.")
        closest_after_end = ds[time_coord].max()

    # Select records within the time range and add the closest before/after
    within_range = (ds[time_coord] > np.datetime64(start_time)) & (
        ds[time_coord] < np.datetime64(end_time)
    )
    selected_times = ds[time_coord].where(
        within_range
        | (ds[time_coord] == closest_before_start)
        | (ds[time_coord] == closest_after_end),
        drop=True,
    )
    ds = ds.sel({time_dim: selected_times})

    return ds


def _select_initial_time(
    ds: xr.Dataset,
    time_dim: str,
    time_coord: str,
    ini_time: datetime,
    climatology: bool,
    allow_flex_time: bool = False,
) -> xr.Dataset:
    """Select exactly one initial time from dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset with a time dimension.
    time_dim : str
        Name of the time dimension.
    time_coord : str
        Name of the time coordinate.
    ini_time : datetime
        The desired initial time.
    allow_flex_time : bool
        - If True: allow a +24h window and pick the closest available timestamp.
        - If False (default): require an exact match, otherwise raise ValueError.

    Returns
    -------
    xr.Dataset
        Dataset reduced to exactly one timestamp.

    Raises
    ------
    ValueError
        If no matching time is found (when `allow_flex_time=False`), or no entries are
        available within the +24h window (when `allow_flex_time=True`).
    """
    if climatology:
        # Convert from timedelta64[ns] to fractional days
        ds["time"] = ds["time"] / np.timedelta64(1, "D")
        # Interpolate from climatology for initial conditions
        return interpolate_from_climatology(ds, time_dim, time_coord, ini_time)

    if allow_flex_time:
        # Look in time range [ini_time, ini_time + 24h)
        end_time = ini_time + timedelta(days=1)
        times = (np.datetime64(ini_time) <= ds[time_coord]) & (
            ds[time_coord] < np.datetime64(end_time)
        )

        if np.all(~times):
            raise ValueError(
                f"No time entries found between {ini_time} and {end_time}."
            )

        ds = ds.where(times, drop=True)
        if ds.sizes[time_dim] > 1:
            # Pick the time closest to start_time
            ds = ds.isel({time_dim: 0})

        logging.warning(
            f"Selected time entry closest to the specified start_time in +24 hour range: {ds[time_coord].values}"
        )

    else:
        # Strict match required
        if not (ds[time_coord].values == np.datetime64(ini_time)).any():
            raise ValueError(
                f"No exact match found for initial time {ini_time}. Consider setting allow_flex_time to True."
            )

        ds = ds.sel({time_coord: np.datetime64(ini_time)})

    if time_dim not in ds.dims:
        ds = ds.expand_dims(time_dim)

    return ds


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

    values = data_array.values

    print(values)
    # numpy datetime64
    if values.dtype == "datetime64[ns]":
        return "datetime"

    # cftime objects (stored as object dtype)
    if any(isinstance(value, cftime_types) for value in values):
        return "cftime"

    # integer time axis
    if np.issubdtype(values.dtype, np.integer):
        return "int"

    raise ValueError("Unsupported data type for time values in input dataset.")


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
