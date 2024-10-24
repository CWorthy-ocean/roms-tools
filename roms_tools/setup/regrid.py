import xarray as xr


class LateralRegrid:
    """Applies lateral fill and regridding to data.

    This class fills missing values in ocean data and interpolates it onto a new grid
    defined by the provided longitude and latitude.

    Parameters
    ----------
    data : DataContainer
        Container with variables to be interpolated, including a `mask` and dimension names.
    lon : xarray.DataArray
        Target longitude coordinates.
    lat : xarray.DataArray
        Target latitude coordinates.
    """

    def __init__(self, data, lon, lat):
        """Initializes the lateral fill and target grid coordinates.

        Parameters
        ----------
        data : DataContainer
            Data with dimensions and mask for filling.
        lon : xarray.DataArray
            Longitude for new grid.
        lat : xarray.DataArray
            Latitude for new grid.
        """

        self.coords = {
            data.dim_names["latitude"]: lat,
            data.dim_names["longitude"]: lon,
        }

    def apply(self, var):
        """Fills missing values and regrids the variable.

        Parameters
        ----------
        var : xarray.DataArray
            Input data to fill and regrid.

        Returns
        -------
        xarray.DataArray
            Regridded data with filled values.
        """
        regridded = var.interp(self.coords, method="linear").drop_vars(
            list(self.coords.keys())
        )
        return regridded


class VerticalRegrid:
    """Performs vertical interpolation of data onto new depth coordinates.

    Parameters
    ----------
    data : DataContainer
        Container holding the data to be regridded, with relevant dimension names.
    target_depth : xarray.DataArray
        Target depth coordinates for interpolation.
    """

    def __init__(self, data, target_depth):
        """Initializes vertical regridding with specified depth coordinates.

        Parameters
        ----------
        data : DataContainer
            Container holding the data to be regridded, with relevant dimension names.
        target_depth : xarray.DataArray
            Target depth coordinates for interpolation.
        """

        self.depth_dim = data.dim_names["depth"]
        dims = {"dim": self.depth_dim}

        dlev = data.ds[data.dim_names["depth"]] - target_depth
        is_below = dlev == dlev.where(dlev >= 0).min(**dims)
        is_above = dlev == dlev.where(dlev <= 0).max(**dims)
        p_below = dlev.where(is_below).sum(**dims)
        p_above = -dlev.where(is_above).sum(**dims)
        denominator = p_below + p_above
        denominator = denominator.where(denominator > 1e-6, 1e-6)
        factor = p_below / denominator

        upper_mask = is_above.sum(**dims) > 0
        lower_mask = is_below.sum(**dims) > 0

        self.coeff = xr.Dataset(
            {
                "is_below": is_below,
                "is_above": is_above,
                "upper_mask": upper_mask,
                "lower_mask": lower_mask,
                "factor": factor,
            }
        )

    def apply(self, var, fill_nans=True):
        """Interpolates the variable onto the new depth grid using precomputed
        coefficients for linear interpolation between layers.

        Parameters
        ----------
        var : xarray.DataArray
            The input data to be regridded along the depth dimension. This should be
            an array with the same depth coordinates as the original grid.
        fill_nans : bool, optional
            Whether to fill NaN values in the regridded data. If True (default),
            forward-fill and backward-fill are applied along the 's_rho' dimension to
            ensure there are no NaNs after interpolation.

        Returns
        -------
        xarray.DataArray
            The regridded data array, interpolated onto the new depth grid. NaN values
            are replaced if `fill_nans=True`, with extrapolation allowed at the surface
            and bottom layers to minimize gaps.
        """

        dims = {"dim": self.depth_dim}

        var_below = var.where(self.coeff["is_below"]).sum(**dims)
        var_above = var.where(self.coeff["is_above"]).sum(**dims)

        result = var_below + (var_above - var_below) * self.coeff["factor"]
        if fill_nans:
            result = result.where(self.coeff["upper_mask"], var.isel({dims["dim"]: 0}))
            result = result.where(self.coeff["lower_mask"], var.isel({dims["dim"]: -1}))
        else:
            result = result.where(self.coeff["upper_mask"]).where(
                self.coeff["lower_mask"]
            )

        return result


def _lateral_regrid(data, lon, lat, data_vars, var_names):
    """Laterally regrid specified variables onto new latitude and longitude coordinates.

    Parameters
    ----------
    data : Dataset
        Input data containing the information about source dimensions.
    lon : xarray.DataArray
        Target longitude coordinates.
    lat : xarray.DataArray
        Target latitude coordinates.
    data_vars : dict of str : xarray.DataArray
        Dictionary of variables to regrid.
    var_names : list of str
        Names of variables to regrid.

    Returns
    -------
    dict of str : xarray.DataArray
        Updated data_vars with regridded variables.
    """
    lateral_regrid = LateralRegrid(data, lon, lat)

    for var_name in var_names:
        if var_name in data_vars:
            data_vars[var_name] = lateral_regrid.apply(data_vars[var_name])

    return data_vars


def _vertical_regrid(data, target_depth, data_vars, var_names):
    """Vertically regrid specified variables onto new depth coordinates.

    Parameters
    ----------
    data : Dataset
        Input dataset containing the variables and source depth information.
    target_depth : xarray.DataArray
        Target depth coordinates for regridding.
    data_vars : dict of str : xarray.DataArray
        Dictionary of variables to be regridded.
    var_names : list of str
        Names of variables to regrid.

    Returns
    -------
    dict of str : xarray.DataArray
        Updated data_vars with variables regridded onto the target depth coordinates.
    """
    vertical_regrid = VerticalRegrid(data, target_depth)

    for var_name in var_names:
        if var_name in data_vars:
            data_vars[var_name] = vertical_regrid.apply(data_vars[var_name])

    return data_vars
