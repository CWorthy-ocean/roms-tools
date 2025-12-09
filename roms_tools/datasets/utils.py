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
