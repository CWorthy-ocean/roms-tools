import xarray as xr


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
