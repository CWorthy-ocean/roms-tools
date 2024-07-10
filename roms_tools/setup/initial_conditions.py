import xarray as xr
import numpy as np
from dataclasses import dataclass, field
from roms_tools.setup.grid import Grid
from datetime import datetime
from roms_tools.setup.datasets import Dataset
from roms_tools.setup.vertical_coordinate import compute_depth, sigma_stretch
from roms_tools.setup.fill import (
    fill_and_interpolate,
    determine_fillvalue,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
)
from roms_tools.setup.utils import nan_check
from roms_tools.setup.plot import _plot, _section_plot, _profile_plot, _line_plot
import matplotlib.pyplot as plt


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
    >>> atm_forcing = AtmosphericForcing(
    ...     grid=grid_info,
    ...     start_time=start_time,
    ...     end_time=end_time,
    ...     source="era5",
    ...     filename="atmospheric_data_*.nc",
    ...     swr_correction=swr_correction,
    ... )
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
            dims = {
                "longitude": "longitude",
                "latitude": "latitude",
                "depth": "depth",
                "time": "time",
            }

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
        data.choose_subdomain(
            latitude_range=[lat.min().values, lat.max().values],
            longitude_range=[lon.min().values, lon.max().values],
            margin=2,
            straddle=straddle,
        )

        data.convert_to_negative_depth()

        # interpolate onto desired grid
        if self.source == "glorys":
            varnames = {
                "temp": "thetao",
                "salt": "so",
                "u": "uo",
                "v": "vo",
                "ssh": "zos",
            }

        fill_dims = [dims["latitude"], dims["longitude"]]

        # 2d interpolation
        mask = xr.where(data.ds[varnames["ssh"]].isel(time=0).isnull(), 0, 1)
        coords = {dims["latitude"]: lat, dims["longitude"]: lon}

        ssh = fill_and_interpolate(
            data.ds[varnames["ssh"]],
            mask,
            fill_dims=fill_dims,
            coords=coords,
            method="linear",
        )

        # 3d interpolation
        cs_r, sigma_r = sigma_stretch(self.theta_s, self.theta_b, self.N, "r")
        zr = compute_depth(h * 0, h, self.hc, cs_r, sigma_r)

        mask = xr.where(data.ds[varnames["temp"]].isel(time=0).isnull(), 0, 1)
        coords = {dims["latitude"]: lat, dims["longitude"]: lon, dims["depth"]: zr}

        # the following computes a fill value to be used if an entire horizontal slice consists of NaNs (often the case
        # for the deepest levels); otherwise NaNs are filled in by horizontal diffusion from non-NaN regions
        fillvalue_temp = determine_fillvalue(data.ds[varnames["temp"]], dims)
        fillvalue_salt = determine_fillvalue(data.ds[varnames["salt"]], dims)

        temp = fill_and_interpolate(
            data.ds[varnames["temp"]],
            mask,
            fill_dims=fill_dims,
            coords=coords,
            method="linear",
            fillvalue=fillvalue_temp,
        )
        salt = fill_and_interpolate(
            data.ds[varnames["salt"]],
            mask,
            fill_dims=fill_dims,
            coords=coords,
            method="linear",
            fillvalue=fillvalue_salt,
        )
        u = fill_and_interpolate(
            data.ds[varnames["u"]],
            mask,
            fill_dims=fill_dims,
            coords=coords,
            method="linear",
        )  # use default fill value of 0
        v = fill_and_interpolate(
            data.ds[varnames["v"]],
            mask,
            fill_dims=fill_dims,
            coords=coords,
            method="linear",
        )  # use default fill value of 0

        # rotate to grid orientation
        u_rot = u * np.cos(angle) + v * np.sin(angle)
        v_rot = v * np.cos(angle) - u * np.sin(angle)

        # interpolate to u- and v-points
        u = interpolate_from_rho_to_u(u_rot)
        v = interpolate_from_rho_to_v(v_rot)

        # 3d masks for ROMS domain
        umask = self.grid.ds.mask_u.expand_dims({"s_rho": self.N})
        vmask = self.grid.ds.mask_v.expand_dims({"s_rho": self.N})

        u = u * umask
        v = v * vmask

        # Compute barotropic velocity
        cs_w, sigma_w = sigma_stretch(self.theta_s, self.theta_b, self.N, "w")
        zw = compute_depth(h * 0, h, self.hc, cs_w, sigma_w)
        # thicknesses
        dz = zw.diff(dim="s_w")
        dz = dz.rename({"s_w": "s_rho"})
        # thicknesses at u- and v-points
        dzu = interpolate_from_rho_to_u(dz)
        dzv = interpolate_from_rho_to_v(dz)

        ubar = (dzu * u).sum(dim="s_rho") / dzu.sum(dim="s_rho")
        vbar = (dzv * v).sum(dim="s_rho") / dzv.sum(dim="s_rho")

        # save in new dataset
        ds = xr.Dataset()

        ds["temp"] = temp.astype(np.float32)
        ds["temp"].attrs["long_name"] = "Potential temperature"
        ds["temp"].attrs["units"] = "Celsius"

        ds["salt"] = salt.astype(np.float32)
        ds["salt"].attrs["long_name"] = "Salinity"
        ds["salt"].attrs["units"] = "PSU"

        ds["zeta"] = ssh.astype(np.float32)
        ds["zeta"].attrs["long_name"] = "Free surface"
        ds["zeta"].attrs["units"] = "m"

        ds["u"] = u.astype(np.float32)
        ds["u"].attrs["long_name"] = "u-flux component"
        ds["u"].attrs["units"] = "m/s"

        ds["v"] = v.astype(np.float32)
        ds["v"].attrs["long_name"] = "v-flux component"
        ds["v"].attrs["units"] = "m/s"

        # initialize vertical velocity to zero
        ds["w"] = xr.zeros_like(zw.expand_dims(time=ds["time"])).astype(np.float32)
        ds["w"].attrs["long_name"] = "w-flux component"
        ds["w"].attrs["units"] = "m/s"

        ds["ubar"] = ubar.transpose("time", "eta_rho", "xi_u").astype(np.float32)
        ds["ubar"].attrs["long_name"] = "vertically integrated u-flux component"
        ds["ubar"].attrs["units"] = "m/s"

        ds["vbar"] = vbar.transpose("time", "eta_v", "xi_rho").astype(np.float32)
        ds["vbar"].attrs["long_name"] = "vertically integrated v-flux component"
        ds["vbar"].attrs["units"] = "m/s"

        depth = -zr
        depth.attrs["long_name"] = "Layer depth at rho-points"
        depth.attrs["units"] = "m"
        ds = ds.assign_coords({"depth_rho": depth})

        depth_u = interpolate_from_rho_to_u(depth)
        depth_u.attrs["long_name"] = "Layer depth at u-points"
        depth_u.attrs["units"] = "m"
        ds = ds.assign_coords({"depth_u": depth_u})

        depth_v = interpolate_from_rho_to_v(depth)
        depth_v.attrs["long_name"] = "Layer depth at v-points"
        depth_v.attrs["units"] = "m"
        ds = ds.assign_coords({"depth_v": depth_v})

        ds.attrs["Title"] = "ROMS initial file produced by roms-tools"

        if dims["time"] != "time":
            ds = ds.rename({dims["time"]: "time"})

        # Translate the time coordinate to days since the model reference date
        model_reference_date = np.datetime64(self.model_reference_date)

        # Convert the time coordinate to the format expected by ROMS (days since model reference date)
        ocean_time = (
            (ds["time"] - model_reference_date).astype("float64") * 1e-9
        )
        ocean_time.attrs[
            "long_name"
        ] = f"time since {np.datetime_as_string(model_reference_date, unit='D')}"
        ocean_time.attrs[
            "units"
        ] = "seconds"
        ds = ds.assign_coords({"ocean_time": ocean_time})

        ds = ds.drop_vars(["eta_rho", "xi_rho"])

        ds["theta_s"] = np.float32(self.theta_s)
        ds["theta_s"].attrs["long_name"] = "S-coordinate surface control parameter"
        ds["theta_s"].attrs["units"] = "nondimensional"

        ds["theta_b"] = np.float32(self.theta_b)
        ds["theta_b"].attrs["long_name"] = "S-coordinate bottom control parameter"
        ds["theta_b"].attrs["units"] = "nondimensional"

        ds["Tcline"] = np.float32(self.hc)
        ds["Tcline"].attrs["long_name"] = "S-coordinate surface/bottom layer width"
        ds["Tcline"].attrs["units"] = "m"

        ds["hc"] = np.float32(self.hc)
        ds["hc"].attrs["long_name"] = "S-coordinate parameter critical depth"
        ds["hc"].attrs["units"] = "m"

        ds["sc_r"] = sigma_r.astype(np.float32)
        ds["sc_r"].attrs["long_name"] = "S-coordinate at rho-point"
        ds["sc_r"].attrs["units"] = "-"

        ds["Cs_r"] = cs_r.astype(np.float32)
        ds["Cs_r"].attrs["long_name"] = "S-coordinate stretching curves at RHO-points"
        ds["Cs_r"].attrs["units"] = "-"

        object.__setattr__(self, "ds", ds)

        nan_check(ds["zeta"].squeeze(), self.grid.ds.mask_rho)

    def plot(
        self,
        varname,
        s_rho=None,
        eta=None,
        xi=None,
        depth_contours=False,
        layer_contours=False,
    ) -> None:
        """
        Plot the initial conditions field for a given eta-, xi-, or s_rho-slice.

        Parameters
        ----------
        varname : str
            The name of the initial conditions field to plot. Options include:
            - "temp": Potential temperature.
            - "salt": Salinity.
            - "zeta": Free surface.
            - "u": u-flux component.
            - "v": v-flux component.
            - "w": w-flux component.
            - "ubar": Vertically integrated u-flux component.
            - "vbar": Vertically integrated v-flux component.
            - "depth": Depth of layer.
        s_rho : int, optional
            The index of the vertical layer to plot. Default is None.
        eta : int, optional
            The eta-index to plot. Default is None.
        xi : int, optional
            The xi-index to plot. Default is None.
        depth_contours : bool, optional
            Whether to include depth contours in the plot. Default is False.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        Raises
        ------
        ValueError
            If the specified varname is not one of the valid options.
            If field is 3D and none of s_rho, eta, xi are specified.
            If field is 2D and both eta and xi are specified.
        """

        if len(self.ds[varname].squeeze().dims) == 3 and not any(
            [s_rho is not None, eta is not None, xi is not None]
        ):
            raise ValueError(
                "For 3D fields, at least one of s_rho, eta, or xi must be specified."
            )

        if len(self.ds[varname].squeeze().dims) == 2 and all(
            [eta is not None, xi is not None]
        ):
            raise ValueError("For 2D fields, specify either eta or xi, not both.")

        self.ds[varname].load()
        field = self.ds[varname].squeeze()

        # slice the field as desired
        title = ""
        if s_rho is not None:
            title = title + f"s_rho = {field.s_rho[s_rho].item()}"
            field = field.isel(s_rho=s_rho)
        else:
            depth_contours = False

        if eta is not None:
            if "eta_rho" in field.dims:
                title = title + f"eta_rho = {field.eta_rho[eta].item()} "
                field = field.isel(eta_rho=eta)
            elif "eta_v" in field.dims:
                title = title + f"eta_v = {field.eta_v[eta].item()} "
                field = field.isel(eta_v=eta)
            else:
                raise ValueError(
                    f"None of the expected dimensions (eta_rho, eta_v) found in ds[{varname}]."
                )
        if xi is not None:
            if "xi_rho" in field.dims:
                title = title + f"xi_rho = {field.xi_rho[xi].item()} "
                field = field.isel(xi_rho=xi)
            elif "xi_u" in field.dims:
                title = title + f"xi_u = {field.xi_u[xi].item()} "
                field = field.isel(xi_u=xi)
            else:
                raise ValueError(
                    f"None of the expected dimensions (xi_rho, xi_u) found in ds[{varname}]."
                )

        # chose colorbar
        if varname in ["u", "v", "w", "ubar", "vbar", "zeta"]:
            vmax = max(field.max().values, -field.min().values)
            vmin = -vmax
            cmap = plt.colormaps.get_cmap("RdBu_r")
        else:
            vmax = field.max().values
            vmin = field.min().values
            cmap = plt.colormaps.get_cmap("YlOrRd")
        cmap.set_bad(color="gray")
        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

        if eta is None and xi is None:
            _plot(
                self.grid.ds,
                field=field,
                straddle=self.grid.straddle,
                depth_contours=depth_contours,
                title=title,
                kwargs=kwargs,
                c="g",
            )
        else:

            if len(field.dims) == 2:
                _section_plot(
                    field, layer_contours=layer_contours, title=title, kwargs=kwargs
                )
            else:
                if "s_rho" in field.dims:
                    _profile_plot(field, title=title)
                else:
                    _line_plot(field, title=title)
