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


def regrid_data(grid, data, vars_2d, vars_3d, lon, lat):

    """
    Interpolates data onto the desired grid and processes it for 2D and 3D variables.

    This method interpolates the specified 2D and 3D variables onto a new grid defined by the provided
    longitude and latitude coordinates. It handles both 2D and 3D data, performing extrapolation for 3D
    variables to fill values up to the bottom of the depth range.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information used for the model.
    data : DataContainer
        The container holding the variables to be interpolated. It must include attributes such as
        `dim_names` and `var_names`.
    vars_2d : list of str
        List of 2D variable names that should be interpolated.
    vars_3d : list of str
        List of 3D variable names that should be interpolated.
    lon : xarray.DataArray
        Longitude coordinates for interpolation.
    lat : xarray.DataArray
        Latitude coordinates for interpolation.

    Returns
    -------
    dict of str: xarray.DataArray
        A dictionary where keys are variable names and values are the interpolated DataArrays.

    Notes
    -----
    - 2D interpolation is performed using linear interpolation on the provided latitude and longitude coordinates.
    - For 3D variables, the method extrapolates the deepest values to the bottom of the depth range and interpolates
      using the specified depth coordinates.
    - The method assumes the presence of `dim_names` and `var_names` attributes in the `data` object.
    """

    # interpolate onto desired grid
    data_vars = {}

    # Set up solver that does lateral fill
    lateral_fill = LateralFill(
        data.ds["mask"],
        [data.dim_names["latitude"], data.dim_names["longitude"]],
    )

    # 2d interpolation
    coords = {data.dim_names["latitude"]: lat, data.dim_names["longitude"]: lon}
    for var in vars_2d:
        # Propagate ocean values into land via lateral fill
        data.ds[data.var_names[var]] = lateral_fill.apply(
            data.ds[data.var_names[var]].astype(np.float64)
        )

        # Regrid
        data_vars[var] = (
            data.ds[data.var_names[var]]
            .interp(coords, method="linear")
            .drop_vars(list(coords.keys()))
        )

    if vars_3d:
        # 3d interpolation
        coords = {
            data.dim_names["depth"]: grid.ds["layer_depth_rho"],
            data.dim_names["latitude"]: lat,
            data.dim_names["longitude"]: lon,
        }
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
            .drop_vars(list(coords.keys()))
        )

        if data.dim_names["time"] != "time":
            data_vars[var] = data_vars[var].rename({data.dim_names["time"]: "time"})

        # transpose to correct order (time, s_rho, eta_rho, xi_rho)
        data_vars[var] = data_vars[var].transpose(
            "time", "s_rho", "eta_rho", "xi_rho"
        )

    return data_vars

