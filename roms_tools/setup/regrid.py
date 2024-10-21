from dataclasses import dataclass
from roms_tools.setup.grid import Grid
from roms_tools.setup.fill import LateralFill
from roms_tools.setup.utils import (
    extrapolate_deepest_to_bottom,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
)
import xarray as xr
import numpy as np


class LateralRegrid:
    """
    Applies lateral fill and regridding to data.

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
        """
        Initializes the lateral fill and target grid coordinates.

        Parameters
        ----------
        data : DataContainer
            Data with dimensions and mask for filling.
        lon : xarray.DataArray
            Longitude for new grid.
        lat : xarray.DataArray
            Latitude for new grid.
        """
        
        # Set up solver that does lateral fill before regridding
        self.lateral_fill = LateralFill(
            data.ds["mask"],
            [data.dim_names["latitude"], data.dim_names["longitude"]],
        )
        self.coords = {data.dim_names["latitude"]: lat, data.dim_names["longitude"]: lon}

    def apply(self, var):
        """
        Fills missing values and regrids the variable.

        Parameters
        ----------
        var : xarray.DataArray
            Input data to fill and regrid.

        Returns
        -------
        xarray.DataArray
            Regridded data with filled values.
        """
    
        # Propagate ocean values into land via lateral fill
        filled = self.lateral_fill.apply(var.astype(np.float64))
    
        # Regrid
        regridded = filled.interp(self.coords, method="linear").drop_vars(list(self.coords.keys()))
 
        return regridded

class VerticalRegrid:
    """
    Performs vertical interpolation of data onto new depth coordinates.

    Parameters
    ----------
    data : DataContainer
        Container holding the data to be regridded, with relevant dimension names.
    depth : xarray.DataArray
        Target depth coordinates for interpolation.
    """

    def __init__(self, data, grid):
        """
        Initializes vertical regridding with specified depth coordinates.

        Parameters
        ----------
        data : DataContainer
            Data with dimension names required for regridding.
        grid : Grid Object
        """

        self.depth_dim = data.dim_names["depth"]
        dims = {"dim": self.depth_dim}

        dlev = data.ds[data.dim_names["depth"]] - grid.ds.layer_depth_rho
        is_below = dlev == dlev.where(dlev>=0).min(**dims)
        is_above = dlev == dlev.where(dlev<=0).max(**dims)
        p_below = dlev.where(is_below).sum(**dims)
        p_above = -dlev.where(is_above).sum(**dims)
        denominator = p_below + p_above
        denominator = denominator.where(denominator > 1e-6, 1e-6)
        factor = p_below / denominator

        self.coeff = xr.Dataset({
            "is_below": is_below,
            "is_above": is_above,
            "factor": factor,
        })


    def apply(self, var):
        """
        Interpolates the variable onto the new depth grid.

        Parameters
        ----------
        var : xarray.DataArray
            Input data to be regridded.

        Returns
        -------
        xarray.DataArray
            Regridded data with extrapolation allowed to avoid NaNs at the surface.
        """    
        
        dims = {"dim": self.depth_dim}

        var_below = var.where(self.coeff["is_below"]).sum(**dims)
        var_above = var.where(self.coeff["is_above"]).sum(**dims)

        result = var_below + (var_above - var_below) * self.coeff["factor"]
        mask = (self.coeff["is_above"].sum(**dims) > 0) & (self.coeff["is_below"].sum(**dims) > 0)

        result = result.where(mask).ffill(**{"dim": "s_rho"}).bfill(**{"dim": "s_rho"})

        return result
 
