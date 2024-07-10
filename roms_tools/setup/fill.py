import numpy as np
import xarray as xr
from numba import jit


def fill_and_interpolate(
    field, mask, fill_dims, coords, method="linear", fillvalue=0.0
):
    """
    Propagate ocean values into land and interpolate using specified coordinates and a given method.

    Parameters
    ----------
    field : xr.DataArray
        The data array to be interpolated. This array typically contains oceanographic or atmospheric data
        with dimensions including latitude, longitude, and depth.

    mask : xr.DataArray
        A data array with the same spatial dimensions as `field`, where `1` indicates wet (ocean) points
        and `0` indicates land points. This mask is used to identify land and ocean areas in the dataset.

    fill_dims : list of str
        A list specifying the dimensions along which to perform the lateral fill. Typically, this would be
        the horizontal dimensions such as latitude and longitude, e.g., ["latitude", "longitude"].

    coords : dict
        A dictionary specifying the target coordinates for interpolation. The keys should match the dimensions
        of `field` (e.g., {"longitude": lon_values, "latitude": lat_values, "depth": depth_values}). This
        dictionary provides the new coordinates onto which the data array will be interpolated.

    method : str, optional, default='linear'
        The interpolation method to use. Valid options are those supported by `xarray.DataArray.interp`,
        such as 'linear' or 'nearest'.

    fillvalue : float, optional, default=0.0
        Value to use if an entire data slice along the fill_dims contains only NaNs.

    Returns
    -------
    xr.DataArray
        The interpolated data array. This array has the same dimensions as the input `field` but with values
        interpolated to the new coordinates specified in `coords`.

    Notes
    -----
    This method performs the following steps:
    1. Sets land values to NaN based on the provided mask to ensure that interpolation does not cross
       the land-ocean boundary.
    2. Uses the `lateral_fill` function to propagate ocean values into land interior, which helps to fill
       gaps in the dataset.
    3. Interpolates the filled data array over the specified coordinates using the selected interpolation method.

    Example
    -------
    >>> import xarray as xr
    >>> field = xr.DataArray(...)
    >>> mask = xr.DataArray(...)
    >>> fill_dims = ["latitude", "longitude"]
    >>> coords = {"latitude": new_lat_values, "longitude": new_lon_values}
    >>> interpolated_field = fill_and_interpolate(
    ...     field, mask, fill_dims, coords, method="linear"
    ... )
    >>> print(interpolated_field)
    """
    if not isinstance(field, xr.DataArray):
        raise TypeError("field must be an xarray.DataArray")
    if not isinstance(mask, xr.DataArray):
        raise TypeError("mask must be an xarray.DataArray")
    if not isinstance(coords, dict):
        raise TypeError("coords must be a dictionary")
    if not all(dim in field.dims for dim in coords.keys()):
        raise ValueError("All keys in coords must match dimensions of field")
    if method not in ["linear", "nearest"]:
        raise ValueError(
            "Unsupported interpolation method. Choose from 'linear', 'nearest'"
        )

    # Set land values to NaN
    field = field.where(mask)

    # Propagate ocean values into land interior before interpolation
    field = lateral_fill(field, 1 - mask, fill_dims, fillvalue)

    # Interpolate
    if len(field.squeeze().dims) == 2:
        # don't extrapolate if we deal with only 2d interpolation in horizontal direction because we want to identify missing data with nan_check
        fill_value = np.nan
    else:
        # but do extrapolate with we deal with 3d interpolation in horizontal + vertical direction because we want to extrapolate in depth
        fill_value = None
    field_interpolated = field.interp(
        coords, method=method, kwargs={"fill_value": fill_value}
    ).drop_vars(list(coords.keys()))

    return field_interpolated


def determine_fillvalue(field, dims):
    """
    Determine fill value by computing the spatial mean for each horizontal slice and selecting the
    spatial mean for the deepest slice that is not NaN.

    Parameters
    ----------
    field : xr.DataArray
        The data array for which fill values are to be determined. This array is typically
        three-dimensional (3D) or four-dimensional (4D), with dimensions including latitude,
        longitude, and depth.

    dims : dict
        A dictionary specifying the names of the dimensions for latitude, longitude, and depth.
        Example: {"latitude": "lat", "longitude": "lon", "depth": "depth"}

    Returns
    -------
    fill_value : float
        The fill value derived from the deepest non-NaN horizontal slice of the spatial mean.

    Notes
    -----
    This function is particularly useful for handling the bottom levels of the data array,
    where entire horizontal slices might be NaNs due to missing data. By computing the
    spatial mean and selecting the deepest non-NaN slice, this function provides a fill
    value that is representative of the bottom layer of the data.
    """
    # Compute spatial mean in the horizontal (latitude and longitude)
    horizontal_mean = field.mean(
        dim=[dims["latitude"], dims["longitude"]], skipna=True
    ).dropna(dims["depth"])

    # Find the depth index with the maximum absolute depth value (this will work whether depth is positive or negative)
    index = np.abs(horizontal_mean[dims["depth"]]).argmax()

    # Extract the corresponding fill value
    fillvalue = horizontal_mean.isel({dims["depth"]: index}).data.squeeze().compute()

    if type(fillvalue) == np.ndarray:
        fillvalue = fillvalue.item()

    return fillvalue


def lateral_fill(var, land_mask, dims=["latitude", "longitude"], fillvalue=0.0):
=======
def lateral_fill(var, land_mask, dims=["latitude", "longitude"]):
>>>>>>> 9908bdedcd2aadf014049fac7a68fcc99d73f35f
    """
    Perform lateral fill on an xarray DataArray using a land mask.

    Parameters
    ----------
    var : xarray.DataArray
        DataArray on which to fill NaNs. The fill is performed on the dimensions specified
        in `dims`.

    land_mask : xarray.DataArray
        Boolean DataArray indicating valid values: `True` where data should be filled. Must have the
        same shape as `var` for the specified dimensions.

    dims : list of str, optional, default=['latitude', 'longitude']
        Dimensions along which to perform the fill. The default is ['latitude', 'longitude'].

<<<<<<< HEAD
    fillvalue : float, optional, default=0.0
        Value to use if an entire data slice along the dims contains only NaNs.

=======
>>>>>>> 9908bdedcd2aadf014049fac7a68fcc99d73f35f
    Returns
    -------
    var_filled : xarray.DataArray
        DataArray with NaNs filled by iterative smoothing, except for the regions
        specified by `land_mask` where NaNs are preserved.

    """
<<<<<<< HEAD

    var_filled = xr.apply_ufunc(
        _lateral_fill_np_array,
        # var, land_mask, fillvalue,
        # input_core_dims=[dims, dims, []],
=======
    var_filled = xr.apply_ufunc(
        _lateral_fill_np_array,
>>>>>>> 9908bdedcd2aadf014049fac7a68fcc99d73f35f
        var,
        land_mask,
        input_core_dims=[dims, dims],
        output_core_dims=[dims],
<<<<<<< HEAD
        output_dtypes=[var.dtype],
        dask="parallelized",
        vectorize=True,
        kwargs={"fillvalue": fillvalue},
=======
        dask="parallelized",
        output_dtypes=[var.dtype],
        vectorize=True,
>>>>>>> 9908bdedcd2aadf014049fac7a68fcc99d73f35f
    )

    return var_filled


<<<<<<< HEAD
def _lateral_fill_np_array(
    var, isvalid_mask, fillvalue=0.0, tol=1.0e-4, rc=1.8, max_iter=10000
):
=======
def _lateral_fill_np_array(var, isvalid_mask, tol=1.0e-4, rc=1.8, max_iter=10000):
>>>>>>> 9908bdedcd2aadf014049fac7a68fcc99d73f35f
    """
    Perform lateral fill on a numpy array.

    Parameters
    ----------
    var : numpy.array
        Two-dimensional array on which to fill NaNs.Only NaNs where `isvalid_mask` is
        True will be filled.

    isvalid_mask : numpy.array, boolean
        Valid values mask: `True` where data should be filled. Must have same shape
        as `var`.

<<<<<<< HEAD
    fillvalue: float
        Value to use if the full field `var` contains only  NaNs. Default is 0.0.

=======
>>>>>>> 9908bdedcd2aadf014049fac7a68fcc99d73f35f
    tol : float, optional, default=1.0e-4
        Convergence criteria: stop filling when the value change is less than
        or equal to `tol * var`, i.e., `delta <= tol * np.abs(var[j, i])`.

    rc : float, optional, default=1.8
        Over-relaxation coefficient to use in the Successive Over-Relaxation (SOR)
        fill algorithm. Larger arrays (or extent of region to be filled if not global)
        typically converge faster with larger coefficients. For completely
        land-filling a 1-degree grid (360x180), a coefficient in the range 1.85-1.9
        is near optimal. Valid bounds are (1.0, 2.0).

    max_iter : int, optional, default=10000
        Maximum number of iterations to perform before giving up if the tolerance
        is not reached.

    Returns
    -------
    var : numpy.array
        Array with NaNs filled by iterative smoothing, except for the regions
        specified by `isvalid_mask` where NaNs are preserved.


    Example
    -------
    >>> import numpy as np
    >>> var = np.array([[1, 2, np.nan], [4, np.nan, 6]])
    >>> isvalid_mask = np.array([[True, True, True], [True, True, True]])
    >>> filled_var = lateral_fill_np_array(var, isvalid_mask)
    >>> print(filled_var)
    """
    nlat, nlon = var.shape[-2:]
    var = var.copy()

    fillmask = np.isnan(var)  # Fill all NaNs
    keepNaNs = ~isvalid_mask & np.isnan(var)
<<<<<<< HEAD
    var = _iterative_fill_sor(nlat, nlon, var, fillmask, tol, rc, max_iter, fillvalue)
=======
    var = _iterative_fill_sor(nlat, nlon, var, fillmask, tol, rc, max_iter)
>>>>>>> 9908bdedcd2aadf014049fac7a68fcc99d73f35f
    var[keepNaNs] = np.nan  # Replace NaNs in areas not designated for filling

    return var


@jit(nopython=True, parallel=True)
<<<<<<< HEAD
def _iterative_fill_sor(nlat, nlon, var, fillmask, tol, rc, max_iter, fillvalue=0.0):
=======
def _iterative_fill_sor(nlat, nlon, var, fillmask, tol, rc, max_iter):
>>>>>>> 9908bdedcd2aadf014049fac7a68fcc99d73f35f
    """
    Perform an iterative land fill algorithm using the Successive Over-Relaxation (SOR)
    solution of the Laplace Equation.

    Parameters
    ----------
    nlat : int
        Number of latitude points in the array.

    nlon : int
        Number of longitude points in the array.

    var : numpy.array
        Two-dimensional array on which to fill NaNs.

    fillmask : numpy.array, boolean
        Mask indicating positions to be filled: `True` where data should be filled.

    tol : float
        Convergence criterion: the iterative process stops when the maximum residual change
        is less than or equal to `tol`.

    rc : float
        Over-relaxation coefficient used in the SOR algorithm. Must be between 1.0 and 2.0.

    max_iter : int
        Maximum number of iterations allowed before the process is terminated.

<<<<<<< HEAD
    fillvalue: float
        Value to use if the full field is NaNs. Default is 0.0.

=======
>>>>>>> 9908bdedcd2aadf014049fac7a68fcc99d73f35f
    Returns
    -------
    None
        The input array `var` is modified in-place with the NaN values filled.

    Notes
    -----
    This function performs the following steps:
    1. Computes a zonal mean to use as an initial guess for the fill.
    2. Replaces missing values in the input array with the computed zonal average.
    3. Iteratively fills the missing values using the SOR algorithm until the specified
       tolerance `tol` is reached or the maximum number of iterations `max_iter` is exceeded.

    Example
    -------
    >>> nlat, nlon = 180, 360
    >>> var = np.random.rand(nlat, nlon)
    >>> fillmask = np.isnan(var)
    >>> tol = 1.0e-4
    >>> rc = 1.8
    >>> max_iter = 10000
    >>> _iterative_fill_sor(nlat, nlon, var, fillmask, tol, rc, max_iter)
    """

    # If field consists only of zeros, fill NaNs in with zeros and all done
    # Note: this will happen for shortwave downward radiation at night time
    if np.max(np.fabs(var)) == 0.0:
        var = np.zeros_like(var)
        return var
<<<<<<< HEAD
    # If field consists only of NaNs, fill NaNs with fill value
    if np.isnan(var).all():
        var = fillvalue * np.ones_like(var)
        return var
=======
>>>>>>> 9908bdedcd2aadf014049fac7a68fcc99d73f35f

    # Compute a zonal mean to use as a first guess
    zoncnt = np.zeros(nlat)
    zonavg = np.zeros(nlat)
    for j in range(0, nlat):
        zoncnt[j] = np.sum(np.where(fillmask[j, :], 0, 1))
        zonavg[j] = np.sum(np.where(fillmask[j, :], 0, var[j, :]))
        if zoncnt[j] != 0:
            zonavg[j] = zonavg[j] / zoncnt[j]

    # Fill missing zonal averages for rows that are entirely land
    for j in range(0, nlat - 1):  # northward pass
        if zoncnt[j] > 0 and zoncnt[j + 1] == 0:
            zoncnt[j + 1] = 1
            zonavg[j + 1] = zonavg[j]
    for j in range(nlat - 1, 0, -1):  # southward pass
        if zoncnt[j] > 0 and zoncnt[j - 1] == 0:
            zoncnt[j - 1] = 1
            zonavg[j - 1] = zonavg[j]

    # Replace the input array missing values with zonal average as first guess
    for j in range(0, nlat):
        for i in range(0, nlon):
            if fillmask[j, i]:
                var[j, i] = zonavg[j]

    # Now do the iterative 2D fill
    res = np.zeros((nlat, nlon))  # work array hold residuals
    res_max = tol
    iter_cnt = 0
    while iter_cnt < max_iter and res_max >= tol:
        res[:] = 0.0  # reset the residual to zero for this iteration

        for j in range(1, nlat - 1):
            jm1 = j - 1
            jp1 = j + 1

            for i in range(1, nlon - 1):
                if fillmask[j, i]:
                    im1 = i - 1
                    ip1 = i + 1

                    # this is SOR
                    res[j, i] = (
                        var[j, ip1]
                        + var[j, im1]
                        + var[jm1, i]
                        + var[jp1, i]
                        - 4.0 * var[j, i]
                    )
                    var[j, i] = var[j, i] + rc * 0.25 * res[j, i]

        # do 1D smooth on top and bottom row if there is some valid data there in the input
        # otherwise leave it set to zonal average
        for j in [0, nlat - 1]:
            if zoncnt[j] > 1:

                for i in range(1, nlon - 1):
                    if fillmask[j, i]:
                        im1 = i - 1
                        ip1 = i + 1

                        res[j, i] = var[j, ip1] + var[j, im1] - 2.0 * var[j, i]
                        var[j, i] = var[j, i] + rc * 0.5 * res[j, i]

        # do 1D smooth in the vertical on left and right column
        for i in [0, nlon - 1]:

            for j in range(1, nlat - 1):
                if fillmask[j, i]:
                    jm1 = j - 1
                    jp1 = j + 1

                    res[j, i] = var[jp1, i] + var[jm1, i] - 2.0 * var[j, i]
                    var[j, i] = var[j, i] + rc * 0.5 * res[j, i]

        # four corners
        for j in [0, nlat - 1]:
            if j == 0:
                jp1 = j + 1
                jm1 = j
            elif j == nlat - 1:
                jp1 = j
                jm1 = j - 1

            for i in [0, nlon - 1]:
                if i == 0:
                    ip1 = i + 1
                    im1 = i
                elif i == nlon - 1:
                    ip1 = i
                    im1 = i - 1

                res[j, i] = (
                    var[j, ip1]
                    + var[j, im1]
                    + var[jm1, i]
                    + var[jp1, i]
                    - 4.0 * var[j, i]
                )
                var[j, i] = var[j, i] + rc * 0.25 * res[j, i]

        res_max = np.max(np.fabs(res)) / np.max(np.fabs(var))
        iter_cnt += 1

    return var


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
