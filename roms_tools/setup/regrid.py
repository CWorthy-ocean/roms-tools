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
    def __init__(self, data, lon, lat):
        """
        Initializes the LateralRegrid class.
    
        Parameters
        ----------
        data : DataContainer
            The container holding the variables to be interpolated. It must include attributes such as
            `dim_names` and `var_names`.
        lon : xarray.DataArray
            Longitude coordinates for interpolation.
        lat : xarray.DataArray
            Latitude coordinates for interpolation.
        """
        
        # Set up solver that does lateral fill before regridding
        self.lateral_fill = LateralFill(
            data.ds["mask"],
            [data.dim_names["latitude"], data.dim_names["longitude"]],
        )
        self.data = data
        self.coords = {data.dim_names["latitude"]: lat, data.dim_names["longitude"]: lon}
        self.target_lon = lon
        self.target_lat = lat


    def apply(self, var):
    
        """
        Interpolates data onto the desired grid.
    
        This method interpolates the specified variable onto a new grid defined by the provided
        longitude and latitude coordinates. 
    
        Parameters
        ----------
        var : xarray.DataArray
            Input DataArray to be regridded.
    
        """
    
        # Propagate ocean values into land via lateral fill
        filled = lateral_fill.apply(
            self.data.ds[self.data.var_names[var]].astype(np.float64)
        )
    
        # Regrid
        regridded = filled.interp(coords, method="linear").drop_vars(list(coords.keys()))
 
        return regridded

        if vars_3d:
        # extrapolate deepest value all the way to bottom
        for var in vars_3d:
            data.ds[data.var_names[var]] = extrapolate_deepest_to_bottom(
                data.ds[data.var_names[var]], data.dim_names["depth"]
            )
            # Propagate ocean values into land via lateral fill
            data.ds[data.var_names[var]] = lateral_fill.apply(
                data.ds[data.var_names[var]].astype(np.float64)
            )
    
            # Regrid
            # setting the fill value to None means that we allow extrapolation in the
            # interpolation step to avoid NaNs at the surface if the lowest depth in original
            # data is greater than zero
            data_vars[var] = (
                data.ds[data.var_names[var]]
                .interp(coords, method="linear", kwargs={"fill_value": None})
                .drop_vars(list(self.coords.keys()))
            )
    
            if data.dim_names["time"] != "time":
                data_vars[var] = data_vars[var].rename({data.dim_names["time"]: "time"})
    
            # transpose to correct order (time, s_rho, eta_rho, xi_rho)
            data_vars[var] = data_vars[var].transpose(
                "time", "s_rho", "eta_rho", "xi_rho"
            )
    
        return data_vars

#class VerticalRegrid:
#            # 3d interpolation
#            coords = {
#                data.dim_names["depth"]: grid.ds["layer_depth_rho"],
#                data.dim_names["latitude"]: lat,
#                data.dim_names["longitude"]: lon,
#            }

