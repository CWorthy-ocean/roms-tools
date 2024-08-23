from dataclasses import dataclass
from roms_tools.setup.grid import Grid
from roms_tools.setup.fill import fill_and_interpolate
from roms_tools.setup.utils import (
    extrapolate_deepest_to_bottom,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
)
import xarray as xr
import numpy as np


@dataclass(frozen=True, kw_only=True)
class ROMSToolsMixins:
    """
    Represents a mixin tool for ROMS-Tools with capabilities shared by the various
    ROMS-Tools dataclasses.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information used for the model.

    """

    grid: Grid

    def get_target_lon_lat(self, use_coarse_grid=False):
        """
        Retrieves the longitude and latitude arrays from the grid and adjusts them based on the grid's orientation.

        This method provides longitude and latitude coordinates, with options for using a coarse grid
        if specified. It also handles longitudes to ensure they are between -180 and 180 degrees and adjusts
        based on whether the grid straddles the Greenwich meridian.

        Parameters
        ----------
        use_coarse_grid : bool, optional
            If True, uses the coarse grid data for longitude and latitude. Defaults to False.

        Returns
        -------
        tuple of (xarray.DataArray, xarray.DataArray, xarray.DataArray, bool)
            The longitude latitude, and angle arrays, and a boolean indicating whether the grid straddles the meridian.

        Raises
        ------
        ValueError
            If the coarse grid data has not been generated yet.
        """

        if use_coarse_grid:
            lon = self.grid.ds.lon_coarse
            lat = self.grid.ds.lat_coarse
            angle = self.grid.ds.angle_coarse
        else:
            lon = self.grid.ds.lon_rho
            lat = self.grid.ds.lat_rho
            angle = self.grid.ds.angle

        # operate on longitudes between -180 and 180 unless ROMS domain lies at least 5 degrees in lontitude away from Greenwich meridian
        lon = xr.where(lon > 180, lon - 360, lon)
        straddle = True
        if not self.grid.straddle and abs(lon).min() > 5:
            lon = xr.where(lon < 0, lon + 360, lon)
            straddle = False

        return lon, lat, angle, straddle

    def regrid_data(self, data, vars_2d, vars_3d, lon, lat):

        """
        Interpolates data onto the desired grid and processes it for 2D and 3D variables.

        This method interpolates the specified 2D and 3D variables onto a new grid defined by the provided
        longitude and latitude coordinates. It handles both 2D and 3D data, performing extrapolation for 3D
        variables to fill values up to the bottom of the depth range.

        Parameters
        ----------
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

        # 2d interpolation
        fill_dims = [data.dim_names["latitude"], data.dim_names["longitude"]]
        coords = {data.dim_names["latitude"]: lat, data.dim_names["longitude"]: lon}
        for var in vars_2d:
            mask = xr.where(data.ds[data.var_names[var]].isel(time=0).isnull(), 0, 1)

            data_vars[var] = fill_and_interpolate(
                data.ds[data.var_names[var]].astype(np.float64),
                mask,
                fill_dims=fill_dims,
                coords=coords,
                method="linear",
            )

        if vars_3d:
            # 3d interpolation
            coords = {
                data.dim_names["depth"]: self.grid.ds["layer_depth_rho"],
                data.dim_names["latitude"]: lat,
                data.dim_names["longitude"]: lon,
            }
        # extrapolate deepest value all the way to bottom ("flooding")
        for var in vars_3d:
            data.ds[data.var_names[var]] = extrapolate_deepest_to_bottom(
                data.ds[data.var_names[var]], data.dim_names["depth"]
            )
            mask = xr.where(data.ds[data.var_names[var]].isel(time=0).isnull(), 0, 1)

            # setting fillvalue_interp to None means that we allow extrapolation in the
            # interpolation step to avoid NaNs at the surface if the lowest depth in original
            # data is greater than zero

            data_vars[var] = fill_and_interpolate(
                data.ds[data.var_names[var]].astype(np.float64),
                mask,
                fill_dims=fill_dims,
                coords=coords,
                method="linear",
                fillvalue_interp=None,
            )
            if data.dim_names["time"] != "time":
                data_vars[var] = data_vars[var].rename({data.dim_names["time"]: "time"})

            # transpose to correct order (time, s_rho, eta_rho, xi_rho)
            data_vars[var] = data_vars[var].transpose(
                "time", "s_rho", "eta_rho", "xi_rho"
            )

        return data_vars

    def process_velocities(self, data_vars, angle, uname, vname, interpolate=True):
        """
        Process and rotate velocity components to align with the grid orientation and optionally interpolate
        them to the appropriate grid points.

        This method performs the following steps:

        1. **Rotation**: Rotates the velocity components (e.g., `u`, `v`) to align with the grid orientation
           using the provided angle data.
        2. **Interpolation**: Optionally interpolates the rotated velocities from rho-points to u- and v-points
           of the grid.
        3. **Barotropic Velocity Calculation**: If the velocity components are 3D (with vertical coordinates),
           computes the barotropic (depth-averaged) velocities.

        Parameters
        ----------
        data_vars : dict of str: xarray.DataArray
            Dictionary containing the velocity components to be processed. The dictionary should include keys
            corresponding to the velocity component names (e.g., `uname`, `vname`).
        angle : xarray.DataArray
            DataArray containing the grid angle values used to rotate the velocity components to the correct
            orientation on the grid.
        uname : str
            The key corresponding to the zonal (east-west) velocity component in `data_vars`.
        vname : str
            The key corresponding to the meridional (north-south) velocity component in `data_vars`.
        interpolate : bool, optional
            If True, interpolates the rotated velocity components to the u- and v-points of the grid.
            Defaults to True.

        Returns
        -------
        dict of str: xarray.DataArray
            A dictionary of the processed velocity components. The returned dictionary includes the rotated and,
            if applicable, interpolated velocity components. If the input velocities are 3D (having a vertical
            dimension), the dictionary also includes the barotropic (depth-averaged) velocities (`ubar` and `vbar`).
        """

        # Rotate velocities to grid orientation
        u_rot = data_vars[uname] * np.cos(angle) + data_vars[vname] * np.sin(angle)
        v_rot = data_vars[vname] * np.cos(angle) - data_vars[uname] * np.sin(angle)

        # Interpolate to u- and v-points
        if interpolate:
            data_vars[uname] = interpolate_from_rho_to_u(u_rot)
            data_vars[vname] = interpolate_from_rho_to_v(v_rot)
        else:
            data_vars[uname] = u_rot
            data_vars[vname] = v_rot

        if "s_rho" in data_vars[uname].dims and "s_rho" in data_vars[vname].dims:
            # Compute barotropic velocity
            dz = -self.grid.ds["interface_depth_rho"].diff(dim="s_w")
            dz = dz.rename({"s_w": "s_rho"})
            dzu = interpolate_from_rho_to_u(dz)
            dzv = interpolate_from_rho_to_v(dz)

            data_vars["ubar"] = (
                (dzu * data_vars[uname]).sum(dim="s_rho") / dzu.sum(dim="s_rho")
            ).transpose("time", "eta_rho", "xi_u")
            data_vars["vbar"] = (
                (dzv * data_vars[vname]).sum(dim="s_rho") / dzv.sum(dim="s_rho")
            ).transpose("time", "eta_v", "xi_rho")

        return data_vars
