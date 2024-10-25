import xarray as xr


class LateralRegrid:
    """Handles lateral regridding of data onto a new spatial grid."""

    def __init__(self, target_coords, source_dim_names):
        """Initialize target grid coordinates and names for grid dimensions.

        Parameters
        ----------
        target_coords : dict
            Dictionary containing 'lon' and 'lat' as xarray.DataArrays representing
            the longitude and latitude values of the target grid.
        source_dim_names : dict
            Dictionary specifying names for the latitude and longitude dimensions,
            typically using keys like "latitude" and "longitude" to align with the dataset conventions.

        Attributes
        ----------
        coords : dict
            Maps the dimension names to the corresponding latitude and longitude
            DataArrays, providing easy access to target grid coordinates.
        """

        self.coords = {
            source_dim_names["latitude"]: target_coords["lat"],
            source_dim_names["longitude"]: target_coords["lon"],
        }

    def apply(self, da):
        """Fills missing values and regrids the variable.

        Parameters
        ----------
        da : xarray.DataArray
            Input data to fill and regrid.

        Returns
        -------
        xarray.DataArray
            Regridded data with filled values.
        """
        regridded = da.interp(self.coords, method="linear").drop_vars(
            list(self.coords.keys())
        )
        return regridded


class VerticalRegrid:
    """Interpolates data onto new vertical (depth) coordinates.

    Parameters
    ----------
    target_depth_coords : xarray.DataArray
        Depth coordinates for the target grid.
    source_depth_coords : xarray.DataArray
        Depth coordinates for the source grid.
    """

    def __init__(self, target_depth_coords, source_depth_coords):
        """Initialize regridding factors for interpolation.

        Parameters
        ----------
        target_depth_coords : xarray.DataArray
            Depth coordinates for the target grid.
        source_depth_coords : xarray.DataArray
            Depth coordinates for the source grid.

        Attributes
        ----------
        coeff : xarray.Dataset
            Dataset containing:
            - `is_below` : Boolean mask for depths just below target.
            - `is_above` : Boolean mask for depths just above target.
            - `upper_mask`, `lower_mask` : Masks for valid interpolation bounds.
            - `factor` : Weight for blending values between levels.
        """

        self.depth_dim = source_depth_coords.dims[0]
        source_depth = source_depth_coords
        dims = {"dim": self.depth_dim}

        dlev = source_depth - target_depth_coords
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


def _lateral_regrid(target_coords, source_dim_names, data_vars, var_names):
    """Laterally regrid specified variables onto new latitude and longitude coordinates.

    Parameters
    ----------
    target_coords : dict
        Dictionary containing 'lon' and 'lat' as xarray.DataArrays representing
        the longitude and latitude values of the target grid.
    source_dim_names : dict
        Dictionary specifying names for the latitude and longitude dimensions,
        typically using keys like "latitude" and "longitude" to align with the dataset conventions.
    data_vars : dict of str : xarray.DataArray
        Dictionary of variables to regrid.
    var_names : list of str
        Names of variables to regrid.

    Returns
    -------
    dict of str : xarray.DataArray
        Updated data_vars with regridded variables.
    """
    lateral_regrid = LateralRegrid(target_coords, source_dim_names)
    for var_name in var_names:
        if var_name in data_vars:
            data_vars[var_name] = lateral_regrid.apply(data_vars[var_name])

    return data_vars


def _vertical_regrid(target_depth_coords, source_depth_coords, data_vars, var_names):
    """Vertically regrid specified variables onto new depth coordinates.

    Parameters
    ----------
    target_depth_coords : xarray.DataArray
        Depth coordinates to interpolate onto.
    source_depth_coords : xarray.DataArray
        Depth coordinates to interpolate from.
    data_vars : dict of str : xarray.DataArray
        Dictionary of variables to be regridded.
    var_names : list of str
        Names of variables to regrid.

    Returns
    -------
    dict of str : xarray.DataArray
        Updated data_vars with variables regridded onto the target depth coordinates.
    """
    vertical_regrid = VerticalRegrid(target_depth_coords, source_depth_coords)

    for var_name in var_names:
        if var_name in data_vars:
            data_vars[var_name] = vertical_regrid.apply(data_vars[var_name])

    return data_vars
