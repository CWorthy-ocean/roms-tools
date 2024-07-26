import numpy as np
import xarray as xr
from numba import jit


def fill_and_interpolate(
    field,
    mask,
    fill_dims,
    coords,
    method="linear",
    fillvalue_fill=0.0,
    fillvalue_interp=np.nan,
):
    """
    Propagates ocean values into land areas and interpolates the data to specified coordinates using a given method.

    Parameters
    ----------
    field : xr.DataArray
        The data array to be interpolated, typically containing oceanographic or atmospheric data
        with dimensions such as latitude and longitude.

    mask : xr.DataArray
        A data array with the same spatial dimensions as `field`, where `1` indicates ocean points
        and `0` indicates land points. This mask is used to identify land and ocean areas in the dataset.

    fill_dims : list of str
        List specifying the dimensions along which to perform the lateral fill, typically the horizontal
        dimensions such as latitude and longitude, e.g., ["latitude", "longitude"].

    coords : dict
        Dictionary specifying the target coordinates for interpolation. The keys should match the dimensions
        of `field` (e.g., {"longitude": lon_values, "latitude": lat_values, "depth": depth_values}).
        This dictionary provides the new coordinates onto which the data array will be interpolated.

    method : str, optional, default='linear'
        The interpolation method to use. Valid options are those supported by `xarray.DataArray.interp`,
        such as 'linear' or 'nearest'.

    fillvalue_fill : float, optional, default=0.0
        Value to use in the fill step if an entire data slice along the fill dimensions contains only NaNs.

    fillvalue_interp : float, optional, default=np.nan
        Value to use in the interpolation step. `np.nan` means that no extrapolation is applied.
        `None` means that extrapolation is applied, which often makes sense when interpolating in the
        vertical direction to avoid NaNs at the surface if the lowest depth is greater than zero.

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
    2. Uses the `lateral_fill` function to propagate ocean values into the land interior, helping to fill
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
    field = lateral_fill(field, 1 - mask, fill_dims, fillvalue_fill)

    field_interpolated = field.interp(
        coords, method=method, kwargs={"fill_value": fillvalue_interp}
    ).drop_vars(list(coords.keys()))

    return field_interpolated


def lateral_fill(var, land_mask, dims=["latitude", "longitude"], fillvalue=0.0):
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

    fillvalue : float, optional, default=0.0
        Value to use if an entire data slice along the dims contains only NaNs.

    Returns
    -------
    var_filled : xarray.DataArray
        DataArray with NaNs filled by iterative smoothing, except for the regions
        specified by `land_mask` where NaNs are preserved.

    """

    var_filled = xr.apply_ufunc(
        _lateral_fill_np_array,
        var,
        land_mask,
        input_core_dims=[dims, dims],
        output_core_dims=[dims],
        output_dtypes=[var.dtype],
        dask="parallelized",
        vectorize=True,
        kwargs={"fillvalue": fillvalue},
    )

    return var_filled


def _lateral_fill_np_array(
    var, isvalid_mask, fillvalue=0.0, tol=1.0e-4, rc=1.8, max_iter=10000
):
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

    fillvalue: float
        Value to use if the full field `var` contains only  NaNs. Default is 0.0.

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
    var = _iterative_fill_sor(nlat, nlon, var, fillmask, tol, rc, max_iter, fillvalue)
    var[keepNaNs] = np.nan  # Replace NaNs in areas not designated for filling

    return var


@jit(nopython=True, parallel=True)
def _iterative_fill_sor(nlat, nlon, var, fillmask, tol, rc, max_iter, fillvalue=0.0):
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

    fillvalue: float
        Value to use if the full field is NaNs. Default is 0.0.

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
    # If field consists only of NaNs, fill NaNs with fill value
    if np.isnan(var).all():
        var = fillvalue * np.ones_like(var)
        return var

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
