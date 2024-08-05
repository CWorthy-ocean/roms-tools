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


def interpolate_from_climatology(field, time_dim_name, time):
    """
    Interpolates the given field temporally based on the specified time points.

    Parameters
    ----------
    field : xarray.DataArray
        The field data to be interpolated.
    time : xarray.DataArray or pandas.DatetimeIndex
        The target time points for interpolation.

    Returns
    -------
    xr.DataArray
        The field values interpolated to the specified time points.


    """
    field["time"] = field[time_dim_name].dt.days
    day_of_year = time.dt.dayofyear

    # Concatenate across the beginning and end of the year
    time_concat = xr.concat(
        [
            field["time"][-1] - 365.25,
            field["time"],
            365.25 + field["time"][0],
        ],
        dim="time",
    )
    field_concat = xr.concat(
        [
            field.isel(time=-1),
            field,
            field.isel(time=0),
        ],
        dim="time",
    )
    field_concat["time"] = time_concat
    # Interpolate to specified times
    field_interpolated = field_concat.interp(time=day_of_year, method="linear")

    return field_interpolated
