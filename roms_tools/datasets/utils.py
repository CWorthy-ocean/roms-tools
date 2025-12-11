from datetime import datetime

import xarray as xr

from roms_tools.fill import one_dim_fill


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
    """Convert all data variables in the dataset to float64.

    This method updates the dataset by converting all of its data variables to the
    `float64` data type, ensuring consistency for numerical operations that require
    high precision.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset

    Returns
    -------
    xr.Dataset:
        Input dataset with data variables converted to double precision.
    """
    return ds.astype({var: "float64" for var in ds.data_vars})


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
