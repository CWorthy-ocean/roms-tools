import xarray as xr
import dask
import numpy as np
from dataclasses import dataclass, field
from roms_tools.setup.grid import Grid
from datetime import datetime
from typing import Optional, Dict, Union
from roms_tools.setup.datasets import Dataset
from roms_tools.setup.vertical_coordinate import compute_depth, sigma_stretch
from roms_tools.setup.fill import fill_and_interpolate, determine_fillvalue, interpolate_from_rho_to_u, interpolate_from_rho_to_v
from roms_tools.setup.utils import nan_check

@dataclass(frozen=True, kw_only=True)
class InitialConditions:
    """
    Represents initial conditions for ROMS.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
    ini_time : datetime
        Time of the desired initialization data.
    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.
    source : str, optional
        Source of the initial condition data. Default is "glorys".
    filename: str
        Path to the atmospheric forcing source data file. Can contain wildcards.
    N : int
        The number of vertical levels.
    theta_s : float
        The surface control parameter.
    theta_b : float
        The bottom control parameter.
    hc : float
        The critical depth.

    Attributes
    ----------
    ds : xr.Dataset
        Xarray Dataset containing the atmospheric forcing data.

    Notes
    -----
    This class represents atmospheric forcing data used in ocean modeling. It provides a convenient
    interface to work with forcing data including shortwave radiation correction and river forcing.

    Examples
    --------
    >>> grid_info = Grid(...)
    >>> start_time = datetime(2000, 1, 1)
    >>> end_time = datetime(2000, 1, 2)
    >>> atm_forcing = AtmosphericForcing(grid=grid_info, start_time=start_time, end_time=end_time, source='era5', filename='atmospheric_data_*.nc', swr_correction=swr_correction)
    """

    grid: Grid
    ini_time: datetime
    model_reference_date: datetime = datetime(2000, 1, 1)
    source: str = "glorys"
    filename: str
    N: int
    theta_s: float
    theta_b: float
    hc: float
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        lon = self.grid.ds.lon_rho
        lat = self.grid.ds.lat_rho
        angle = self.grid.ds.angle
        h = self.grid.ds.h

        if self.source == "glorys":
            dims = {"longitude": "longitude", "latitude": "latitude", "depth": "depth", "time": "time"}

        data = Dataset(filename=self.filename, start_time=self.ini_time, dim_names=dims)

        # operate on longitudes between -180 and 180 unless ROMS domain lies at least 5 degrees in lontitude away from Greenwich meridian
        lon = xr.where(lon > 180, lon - 360, lon)
        straddle = True
        if not self.grid.straddle and abs(lon).min() > 5:
            lon = xr.where(lon < 0, lon + 360, lon)
            straddle = False

        # Step 1: Choose subdomain of forcing data including safety margin for interpolation, and Step 2: Convert to the proper longitude range.
        # Step 1 is necessary to avoid discontinuous longitudes that could be introduced by Step 2.
        # Discontinuous longitudes can lead to artifacts in the interpolation process. Specifically, if there is a data gap,
        # discontinuous longitudes could result in values that appear to come from a distant location instead of producing NaNs.
        # These NaNs are important as they can be identified and handled appropriately by the nan_check function.
        # Note that no error is thrown if the data does not have the full safety margin available, as long as interpolation does not give any NaNs over the ocean.
        data.choose_subdomain(latitude_range=[lat.min().values, lat.max().values], longitude_range=[lon.min().values, lon.max().values], margin=2, straddle=straddle)

        data.convert_to_negative_depth()

        # interpolate onto desired grid
        if self.source == "glorys":
            varnames = {
                "temp": "thetao",
                "salt": "so",
                "u": "uo",
                "v": "vo",
                "ssh": "zos"
            }

        fill_dims=[dims["latitude"], dims["longitude"]]

        # 2d interpolation
        mask = xr.where(data.ds[varnames["ssh"]].isel(time=0).isnull(), 0, 1)
        coords={dims["latitude"]: lat, dims["longitude"]: lon}

        ssh = fill_and_interpolate(data.ds[varnames["ssh"]], mask, fill_dims=fill_dims, coords=coords, method='linear')

        # 3d interpolation
        cs_r, sigma_r = sigma_stretch(self.theta_s, self.theta_b, self.N, 'r')
        zr = compute_depth(h*0, h, self.hc, cs_r, sigma_r)

        mask = xr.where(data.ds[varnames["temp"]].isel(time=0).isnull(), 0, 1)
        coords={dims["latitude"]: lat, dims["longitude"]: lon, dims["depth"]: zr}

        # the following computes a fill value to be used if an entire horizontal slice consists of NaNs (often the case
        # for the deepest levels); otherwise NaNs are filled in by horizontal diffusion from non-NaN regions
        fillvalue_temp = determine_fillvalue(data.ds[varnames["temp"]], dims)
        fillvalue_salt = determine_fillvalue(data.ds[varnames["salt"]], dims)

        temp = fill_and_interpolate(data.ds[varnames["temp"]], mask, fill_dims=fill_dims, coords=coords, method='linear', fillvalue=fillvalue_temp)
        salt = fill_and_interpolate(data.ds[varnames["salt"]], mask, fill_dims=fill_dims, coords=coords, method='linear', fillvalue=fillvalue_salt)
        u = fill_and_interpolate(data.ds[varnames["u"]], mask, fill_dims=fill_dims, coords=coords, method='linear')  # use default fill value of 0
        v = fill_and_interpolate(data.ds[varnames["v"]], mask, fill_dims=fill_dims, coords=coords, method='linear')  # use default fill value of 0

        # rotate to grid orientation
        u_rot = u * np.cos(angle) + v * np.sin(angle)
        v_rot = v * np.cos(angle) - u * np.sin(angle)

        # interpolate to u- and v-points
        u = interpolate_from_rho_to_u(u_rot)
        v = interpolate_from_rho_to_v(v_rot)

        # 3d masks for ROMS domain
        umask = self.grid.ds.mask_u.expand_dims({'s_r': self.N})
        vmask = self.grid.ds.mask_v.expand_dims({'s_r': self.N})

        u = u * umask
        v = v * vmask

        # Compute barotropic velocity
        cs_w, sigma_w = sigma_stretch(self.theta_s, self.theta_b,self. N, 'w')
        zw = compute_depth(h*0, h, self.hc, cs_w, sigma_w)
        # thicknesses
        dz = zw.diff(dim='s_w')
        dz = dz.rename({"s_w": "s_r"})
        # thicknesses at u- and v-points
        dzu = interpolate_from_rho_to_u(dz)
        dzv = interpolate_from_rho_to_v(dz)

        ubar = (dzu * u).sum(dim="s_r") / dzu.sum(dim="s_r")
        vbar = (dzv * v).sum(dim="s_r")/ dzv.sum(dim="s_r")

        # save in new dataset
        ds = xr.Dataset()

        ds["temp"] =  temp.astype(np.float32)
        ds["temp"].attrs["long_name"] = "Potential temperature"
        ds["temp"].attrs["units"] = "Celsius"

        ds["salt"] =  salt.astype(np.float32)
        ds["salt"].attrs["long_name"] = "Salinity"
        ds["salt"].attrs["units"] = "PSU"

        ds["zeta"] =  ssh.astype(np.float32)
        ds["zeta"].attrs["long_name"] = "Free surface"
        ds["zeta"].attrs["units"] = "m"

        ds["u"] = u.astype(np.float32)
        ds["u"].attrs["long_name"] = "u-flux component"
        ds["u"].attrs["units"] = "m/s"

        ds["v"] =  v.astype(np.float32)
        ds["v"].attrs["long_name"] = "v-flux component"
        ds["v"].attrs["units"] = "m/s"

        ds["ubar"] = ubar.transpose('time', 'eta_rho', 'xi_u').astype(np.float32)
        ds["ubar"].attrs["long_name"] = "vertically integrated u-flux component"
        ds["ubar"].attrs["units"] = "m/s"

        ds["vbar"] =  vbar.transpose('time', 'eta_v', 'xi_rho').astype(np.float32)
        ds["vbar"].attrs["long_name"] = "vertically integrated v-flux component"
        ds["vbar"].attrs["units"] = "m/s"

        # initialize vertical velocity to zero
        ds['w'] = xr.zeros_like(zw.expand_dims(time=ds['time']))

        depth = -zr
        depth.attrs["long_name"] = "Depth"
        depth.attrs["units"] = "m"
        ds = ds.assign_coords({"depth": depth})

        ds.attrs["Title"] = "ROMS initial file produced by roms-tools"

        if dims["time"] != "time":
            ds = ds.rename({dims["time"]: "time"})

        # Preserve the original time coordinate for readability
        ds = ds.assign_coords({"Time": ds["time"]})

        # Translate the time coordinate to days since the model reference date
        model_reference_date = np.datetime64(self.model_reference_date)

        # Convert the time coordinate to the format expected by ROMS (days since model reference date)
        ocean_time = (ds["time"] - model_reference_date).astype('float64') / 3600 / 24 * 1e-9
        ocean_time.attrs["long_name"] = "time since initialization"
        ds = ds.assign_coords({"ocean_time": ocean_time})

        ds["tstart"] = 1.0
        ds["tstart"].attrs["long_name"] = "Start processing day"
        ds["tstart"].attrs["units"] = "day"

        ds["tend"] = 1.0
        ds["tend"].attrs["long_name"] = "End processing day"
        ds["tend"].attrs["units"] = "day"

        ds["theta_s"] = self.theta_s
        ds["theta_s"].attrs["long_name"] = "S-coordinate surface control parameter"
        ds["theta_s"].attrs["units"] = "nondimensional"

        ds["theta_b"] = self.theta_b
        ds["theta_b"].attrs["long_name"] = "S-coordinate bottom control parameter"
        ds["theta_b"].attrs["units"] = "nondimensional"

        ds["Tcline"] = self.hc
        ds["Tcline"].attrs["long_name"] = "S-coordinate surface/bottom layer width"
        ds["Tcline"].attrs["units"] = "m"

        ds["hc"] = self.hc
        ds["hc"].attrs["long_name"] = "S-coordinate parameter critical depth"
        ds["hc"].attrs["units"] = "m"

        ds["sc_r"] = sigma_r
        ds["sc_r"].attrs["long_name"] = "S-coordinate at rho-point"
        ds["sc_r"].attrs["units"] = "-"

        ds["Cs_r"] = cs_r
        ds["Cs_r"].attrs["long_name"] = "S-coordinate stretching curves at RHO-points"
        ds["Cs_r"].attrs["units"] = "-"

        object.__setattr__(self, "ds", ds)

        nan_check(ds["zeta"].squeeze(), self.grid.ds.mask_rho)



