import xarray as xr

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

