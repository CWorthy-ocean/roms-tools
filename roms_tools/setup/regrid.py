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

    def __init__(self, data, depth):
        """
        Initializes vertical regridding with specified depth coordinates.

        Parameters
        ----------
        data : DataContainer
            Data with dimension names required for regridding.
        depth : xarray.DataArray
            New depth coordinates for interpolation.
        """
        self.coords = {data.dim_names["depth"]: depth}

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
        return var.interp(self.coords, method="linear", kwargs={"fill_value": None}).drop_vars(list(self.coords.keys()))
 
