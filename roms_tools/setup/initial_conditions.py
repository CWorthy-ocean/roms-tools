import xarray as xr
import numpy as np
import yaml
import importlib.metadata
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Union
from roms_tools.setup.grid import Grid
from roms_tools.setup.vertical_coordinate import VerticalCoordinate
from datetime import datetime
from roms_tools.setup.datasets import GLORYSDataset, CESMBGCDataset
from roms_tools.setup.fill import fill_and_interpolate
from roms_tools.setup.utils import (
    nan_check,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    extrapolate_deepest_to_bottom,
)
from roms_tools.setup.plot import _plot, _section_plot, _profile_plot, _line_plot
import matplotlib.pyplot as plt


@dataclass(frozen=True, kw_only=True)
class InitialConditions:
    """
    Represents initial conditions for ROMS, including physical and biogeochemical data.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information used for the model.
    vertical_coordinate : VerticalCoordinate
        Object representing the vertical coordinate system.
    ini_time : datetime
        The date and time at which the initial conditions are set.
    physics_source : Dict[str, Union[str, None]]
        Dictionary specifying the source of the physical initial condition data:
        - "name" (str): Name of the data source (e.g., "GLORYS").
        - "path" (str): Path to the physical data file. Can contain wildcards.
        - "climatology" (bool): Indicates if the physical data is climatology data. Defaults to False.
    bgc_source : Optional[Dict[str, Union[str, None]]]
        Dictionary specifying the source of the biogeochemical (BGC) initial condition data:
        - "name" (str): Name of the BGC data source (e.g., "CESM_REGRIDDED").
        - "path" (str): Path to the BGC data file. Can contain wildcards.
        - "climatology" (bool): Indicates if the BGC data is climatology data. Defaults to False.
    model_reference_date : datetime, optional
        The reference date for the model. Defaults to January 1, 2000.

    Attributes
    ----------
    ds : xr.Dataset
        Xarray Dataset containing the initial condition data loaded from the specified files.

    Examples
    --------
    >>> initial_conditions = InitialConditions(
    ...     grid=grid,
    ...     vertical_coordinate=vertical_coordinate,
    ...     ini_time=datetime(2022, 1, 1),
    ...     physics_source={"name": "GLORYS", "path": "physics_data.nc"},
    ...     bgc_source={"name": "CESM_REGRIDDED", "path": "bgc_data.nc"},
    ... )
    """

    grid: Grid
    vertical_coordinate: VerticalCoordinate
    ini_time: datetime
    physics_source: Dict[str, Union[str, None]]
    bgc_source: Optional[Dict[str, Union[str, None]]] = None
    model_reference_date: datetime = datetime(2000, 1, 1)

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        if "name" not in self.physics_source.keys():
            raise ValueError("`physics_source` must include a 'name'.")
        if "path" not in self.physics_source.keys():
            raise ValueError("`physics_source` must include a 'path'.")
        # set self.physics_source["climatology"] to False if not provided
        object.__setattr__(
            self,
            "physics_source",
            {
                **self.physics_source,
                "climatology": self.physics_source.get("climatology", False),
            },
        )
        if self.physics_source["name"] == "GLORYS":
            data = GLORYSDataset(
                filename=self.physics_source["path"],
                start_time=self.ini_time,
                climatology=self.physics_source["climatology"],
            )
        else:
            raise ValueError(
                'Only "GLORYS" is a valid option for physics_source["name"].'
            )
        if self.bgc_source is not None:
            if "name" not in self.bgc_source.keys():
                raise ValueError(
                    "`bgc_source` must include a 'name' if it is provided."
                )
            if "path" not in self.bgc_source.keys():
                raise ValueError(
                    "`bgc_source` must include a 'path' if it is provided."
                )
            # set self.physics_source["climatology"] to False if not provided
            object.__setattr__(
                self,
                "bgc_source",
                {
                    **self.bgc_source,
                    "climatology": self.bgc_source.get("climatology", False),
                },
            )

            if self.bgc_source["name"] == "CESM_REGRIDDED":

                bgc_data = CESMBGCDataset(
                    filename=self.bgc_source["path"],
                    start_time=self.ini_time,
                    climatology=self.bgc_source["climatology"],
                )
                bgc_data.post_process()
            else:
                raise ValueError(
                    'Only "CESM_REGRIDDED" is a valid option for bgc_source["name"].'
                )

        lon = self.grid.ds.lon_rho
        lat = self.grid.ds.lat_rho
        angle = self.grid.ds.angle

        # operate on longitudes between -180 and 180 unless ROMS domain lies at least 5 degrees in lontitude away from Greenwich meridian
        lon = xr.where(lon > 180, lon - 360, lon)
        straddle = True
        if not self.grid.straddle and abs(lon).min() > 5:
            lon = xr.where(lon < 0, lon + 360, lon)
            straddle = False

        # Restrict data to relevant subdomain to achieve better performance and to avoid discontinuous longitudes introduced by converting
        # to a different longitude range (+- 360 degrees). Discontinues longitudes can lead to artifacts in the interpolation process that
        # would not be detected by the nan_check function.
        data.choose_subdomain(
            latitude_range=[lat.min().values, lat.max().values],
            longitude_range=[lon.min().values, lon.max().values],
            margin=2,
            straddle=straddle,
        )
        if self.bgc_source is not None:
            bgc_data.choose_subdomain(
                latitude_range=[lat.min().values, lat.max().values],
                longitude_range=[lon.min().values, lon.max().values],
                margin=2,
                straddle=straddle,
            )

        # interpolate onto desired grid
        fill_dims = [data.dim_names["latitude"], data.dim_names["longitude"]]

        # 2d interpolation
        mask = xr.where(data.ds[data.var_names["ssh"]].isel(time=0).isnull(), 0, 1)
        coords = {data.dim_names["latitude"]: lat, data.dim_names["longitude"]: lon}

        ssh = fill_and_interpolate(
            data.ds[data.var_names["ssh"]].astype(np.float64),
            mask,
            fill_dims=fill_dims,
            coords=coords,
            method="linear",
        )

        # 3d interpolation

        # extrapolate deepest value all the way to bottom ("flooding")
        for var in ["temp", "salt", "u", "v"]:
            data.ds[data.var_names[var]] = extrapolate_deepest_to_bottom(
                data.ds[data.var_names[var]], data.dim_names["depth"]
            )
        mask = xr.where(data.ds[data.var_names["temp"]].isel(time=0).isnull(), 0, 1)
        coords = {
            data.dim_names["latitude"]: lat,
            data.dim_names["longitude"]: lon,
            data.dim_names["depth"]: self.vertical_coordinate.ds["layer_depth_rho"],
        }

        # setting fillvalue_interp to None means that we allow extrapolation in the
        # interpolation step to avoid NaNs at the surface if the lowest depth in original
        # data is greater than zero
        data_vars = {}

        for var in ["temp", "salt", "u", "v"]:
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

        # do the same for the BGC variables if present
        if self.bgc_source is not None:
            fill_dims = [
                bgc_data.dim_names["latitude"],
                bgc_data.dim_names["longitude"],
            ]

            for var in bgc_data.var_names.values():
                bgc_data.ds[var] = extrapolate_deepest_to_bottom(
                    bgc_data.ds[var], bgc_data.dim_names["depth"]
                )
            mask = xr.where(
                bgc_data.ds[bgc_data.var_names["PO4"]].isel(time=0).isnull(), 0, 1
            )
            coords = {
                bgc_data.dim_names["latitude"]: lat,
                bgc_data.dim_names["longitude"]: lon,
                bgc_data.dim_names["depth"]: self.vertical_coordinate.ds[
                    "layer_depth_rho"
                ],
            }
            for var in bgc_data.var_names.keys():
                data_vars[var] = fill_and_interpolate(
                    bgc_data.ds[bgc_data.var_names[var]].astype(np.float64),
                    mask,
                    fill_dims=fill_dims,
                    coords=coords,
                    method="linear",
                    fillvalue_interp=None,
                )
                if bgc_data.dim_names["time"] != "time":
                    data_vars[var] = data_vars[var].rename(
                        {bgc_data.dim_names["time"]: "time"}
                    )
                if self.bgc_source["climatology"]:
                    # make sure time coordinate coincides, otherwise BGC variables are written into .ds as NaNs
                    data_vars[var] = data_vars[var].assign_coords(
                        {"time": data_vars["temp"]["time"]}
                    )

        # rotate velocities to grid orientation
        u_rot = data_vars["u"] * np.cos(angle) + data_vars["v"] * np.sin(angle)
        v_rot = data_vars["v"] * np.cos(angle) - data_vars["u"] * np.sin(angle)

        # interpolate to u- and v-points
        u = interpolate_from_rho_to_u(u_rot)
        v = interpolate_from_rho_to_v(v_rot)

        # 3d masks for ROMS domain
        umask = self.grid.ds.mask_u.expand_dims({"s_rho": u.s_rho})
        vmask = self.grid.ds.mask_v.expand_dims({"s_rho": v.s_rho})

        u = u * umask
        v = v * vmask

        # Compute barotropic velocity
        # thicknesses
        dz = -self.vertical_coordinate.ds["interface_depth_rho"].diff(dim="s_w")
        dz = dz.rename({"s_w": "s_rho"})
        # thicknesses at u- and v-points
        dzu = interpolate_from_rho_to_u(dz)
        dzv = interpolate_from_rho_to_v(dz)

        ubar = (dzu * u).sum(dim="s_rho") / dzu.sum(dim="s_rho")
        vbar = (dzv * v).sum(dim="s_rho") / dzv.sum(dim="s_rho")

        # save in new dataset
        ds = xr.Dataset()

        ds["temp"] = data_vars["temp"].astype(np.float32)
        ds["temp"].attrs["long_name"] = "Potential temperature"
        ds["temp"].attrs["units"] = "Celsius"

        ds["salt"] = data_vars["salt"].astype(np.float32)
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
        ds["w"] = xr.zeros_like(
            self.vertical_coordinate.ds["interface_depth_rho"].expand_dims(
                time=ds[data.dim_names["time"]]
            )
        ).astype(np.float32)
        ds["w"].attrs["long_name"] = "w-flux component"
        ds["w"].attrs["units"] = "m/s"

        ds["ubar"] = ubar.transpose(data.dim_names["time"], "eta_rho", "xi_u").astype(
            np.float32
        )
        ds["ubar"].attrs["long_name"] = "vertically integrated u-flux component"
        ds["ubar"].attrs["units"] = "m/s"

        ds["vbar"] = vbar.transpose(data.dim_names["time"], "eta_v", "xi_rho").astype(
            np.float32
        )
        ds["vbar"].attrs["long_name"] = "vertically integrated v-flux component"
        ds["vbar"].attrs["units"] = "m/s"

        if self.bgc_source is not None:
            ds["PO4"] = data_vars["PO4"].astype(np.float32)
            ds["PO4"].attrs["long_name"] = "Dissolved Inorganic Phosphate"
            ds["PO4"].attrs["units"] = "mmol/m^3"

            ds["NO3"] = data_vars["NO3"].astype(np.float32)
            ds["NO3"].attrs["long_name"] = "Dissolved Inorganic Nitrate"
            ds["NO3"].attrs["units"] = "mmol/m^3"

            ds["SiO3"] = data_vars["SiO3"].astype(np.float32)
            ds["SiO3"].attrs["long_name"] = "Dissolved Inorganic Silicate"
            ds["SiO3"].attrs["units"] = "mmol/m^3"

            ds["NH4"] = data_vars["NH4"].astype(np.float32)
            ds["NH4"].attrs["long_name"] = "Dissolved Ammonia"
            ds["NH4"].attrs["units"] = "mmol/m^3"

            ds["Fe"] = data_vars["Fe"].astype(np.float32)
            ds["Fe"].attrs["long_name"] = "Dissolved Inorganic Iron"
            ds["Fe"].attrs["units"] = "mmol/m^3"

            ds["Lig"] = data_vars["Lig"].astype(np.float32)
            ds["Lig"].attrs["long_name"] = "Iron Binding Ligand"
            ds["Lig"].attrs["units"] = "mmol/m^3"

            ds["O2"] = data_vars["O2"].astype(np.float32)
            ds["O2"].attrs["long_name"] = "Dissolved Oxygen"
            ds["O2"].attrs["units"] = "mmol/m^3"

            ds["DIC"] = data_vars["DIC"].astype(np.float32)
            ds["DIC"].attrs["long_name"] = "Dissolved Inorganic Carbon"
            ds["DIC"].attrs["units"] = "mmol/m^3"

            ds["DIC_ALT_CO2"] = data_vars["DIC_ALT_CO2"].astype(np.float32)
            ds["DIC_ALT_CO2"].attrs[
                "long_name"
            ] = "Dissolved Inorganic Carbon, Alternative CO2"
            ds["DIC_ALT_CO2"].attrs["units"] = "mmol/m^3"

            ds["ALK"] = data_vars["ALK"].astype(np.float32)
            ds["ALK"].attrs["long_name"] = "Alkalinity"
            ds["ALK"].attrs["units"] = "meq/m^3"

            ds["ALK_ALT_CO2"] = data_vars["ALK_ALT_CO2"].astype(np.float32)
            ds["ALK_ALT_CO2"].attrs["long_name"] = "Alkalinity, Alternative CO2"
            ds["ALK_ALT_CO2"].attrs["units"] = "meq/m^3"

            ds["DOC"] = data_vars["DOC"].astype(np.float32)
            ds["DOC"].attrs["long_name"] = "Dissolved Organic Carbon"
            ds["DOC"].attrs["units"] = "mmol/m^3"

            ds["DON"] = data_vars["DON"].astype(np.float32)
            ds["DON"].attrs["long_name"] = "Dissolved Organic Nitrogen"
            ds["DON"].attrs["units"] = "mmol/m^3"

            ds["DOP"] = data_vars["DOP"].astype(np.float32)
            ds["DOP"].attrs["long_name"] = "Dissolved Organic Phosphorus"
            ds["DOP"].attrs["units"] = "mmol/m^3"

            ds["DOPr"] = data_vars["DOPr"].astype(np.float32)
            ds["DOPr"].attrs["long_name"] = "Refractory Dissolved Organic Phosphorus"
            ds["DOPr"].attrs["units"] = "mmol/m^3"

            ds["DONr"] = data_vars["DONr"].astype(np.float32)
            ds["DONr"].attrs["long_name"] = "Refractory Dissolved Organic Nitrogen"
            ds["DONr"].attrs["units"] = "mmol/m^3"

            ds["DOCr"] = data_vars["DOCr"].astype(np.float32)
            ds["DOCr"].attrs["long_name"] = "Refractory Dissolved Organic Carbon"
            ds["DOCr"].attrs["units"] = "mmol/m^3"

            ds["zooC"] = data_vars["zooC"].astype(np.float32)
            ds["zooC"].attrs["long_name"] = "Zooplankton Carbon"
            ds["zooC"].attrs["units"] = "mmol/m^3"

            ds["spChl"] = data_vars["spChl"].astype(np.float32)
            ds["spChl"].attrs["long_name"] = "Small Phytoplankton Chlorophyll"
            ds["spChl"].attrs["units"] = "mg/m^3"

            ds["spC"] = data_vars["spC"].astype(np.float32)
            ds["spC"].attrs["long_name"] = "Small Phytoplankton Carbon"
            ds["spC"].attrs["units"] = "mmol/m^3"

            ds["spC"] = data_vars["spC"].astype(np.float32)
            ds["spC"].attrs["long_name"] = "Small Phytoplankton Carbon"
            ds["spC"].attrs["units"] = "mmol/m^3"

            ds["spP"] = data_vars["spP"].astype(np.float32)
            ds["spP"].attrs["long_name"] = "Small Phytoplankton Phosphorous"
            ds["spP"].attrs["units"] = "mmol/m^3"

            ds["spFe"] = data_vars["spFe"].astype(np.float32)
            ds["spFe"].attrs["long_name"] = "Small Phytoplankton Iron"
            ds["spFe"].attrs["units"] = "mmol/m^3"

            ds["spCaCO3"] = data_vars["spCaCO3"].astype(np.float32)
            ds["spCaCO3"].attrs["long_name"] = "Small Phytoplankton CaCO3"
            ds["spCaCO3"].attrs["units"] = "mmol/m^3"

            ds["diatChl"] = data_vars["diatChl"].astype(np.float32)
            ds["diatChl"].attrs["long_name"] = "Diatom Chlorophyll"
            ds["diatChl"].attrs["units"] = "mg/m^3"

            ds["diatC"] = data_vars["diatC"].astype(np.float32)
            ds["diatC"].attrs["long_name"] = "Diatom Carbon"
            ds["diatC"].attrs["units"] = "mmol/m^3"

            ds["diatP"] = data_vars["diatP"].astype(np.float32)
            ds["diatP"].attrs["long_name"] = "Diatom Phosphorus"
            ds["diatP"].attrs["units"] = "mmol/m^3"

            ds["diatFe"] = data_vars["diatFe"].astype(np.float32)
            ds["diatFe"].attrs["long_name"] = "Diatom Iron"
            ds["diatFe"].attrs["units"] = "mmol/m^3"

            ds["diatSi"] = data_vars["diatSi"].astype(np.float32)
            ds["diatSi"].attrs["long_name"] = "Diatom Silicate"
            ds["diatSi"].attrs["units"] = "mmol/m^3"

            ds["diazChl"] = data_vars["diazChl"].astype(np.float32)
            ds["diazChl"].attrs["long_name"] = "Diazotroph Chlorophyll"
            ds["diazChl"].attrs["units"] = "mg/m^3"

            ds["diazC"] = data_vars["diazC"].astype(np.float32)
            ds["diazC"].attrs["long_name"] = "Diazotroph Carbon"
            ds["diazC"].attrs["units"] = "mmol/m^3"

            ds["diazP"] = data_vars["diazP"].astype(np.float32)
            ds["diazP"].attrs["long_name"] = "Diazotroph Phosphorus"
            ds["diazP"].attrs["units"] = "mmol/m^3"

            ds["diazFe"] = data_vars["diazFe"].astype(np.float32)
            ds["diazFe"].attrs["long_name"] = "Diazotroph Iron"
            ds["diazFe"].attrs["units"] = "mmol/m^3"

        ds = ds.assign_coords(
            {
                "layer_depth_u": self.vertical_coordinate.ds["layer_depth_u"],
                "layer_depth_v": self.vertical_coordinate.ds["layer_depth_v"],
                "interface_depth_u": self.vertical_coordinate.ds["interface_depth_u"],
                "interface_depth_v": self.vertical_coordinate.ds["interface_depth_v"],
            }
        )

        ds.attrs["title"] = "ROMS initial conditions file created by ROMS-Tools"
        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"
        ds.attrs["roms_tools_version"] = roms_tools_version
        ds.attrs["ini_time"] = str(self.ini_time)
        ds.attrs["model_reference_date"] = str(self.model_reference_date)
        ds.attrs["physical_source"] = self.physics_source["name"]
        if self.bgc_source is not None:
            ds.attrs["bgc_source"] = self.bgc_source["name"]

        # Translate the time coordinate to days since the model reference date
        model_reference_date = np.datetime64(self.model_reference_date)

        # Convert the time coordinate to the format expected by ROMS (days since model reference date)
        ocean_time = (ds["time"] - model_reference_date).astype("float64") * 1e-9
        ds = ds.assign_coords(ocean_time=("time", np.float32(ocean_time)))
        ds["ocean_time"].attrs[
            "long_name"
        ] = f"time since {np.datetime_as_string(model_reference_date, unit='D')}"
        ds["ocean_time"].attrs["units"] = "seconds"

        ds["theta_s"] = self.vertical_coordinate.ds["theta_s"]
        ds["theta_b"] = self.vertical_coordinate.ds["theta_b"]
        ds["Tcline"] = self.vertical_coordinate.ds["Tcline"]
        ds["hc"] = self.vertical_coordinate.ds["hc"]
        ds["sc_r"] = self.vertical_coordinate.ds["sc_r"]
        ds["Cs_r"] = self.vertical_coordinate.ds["Cs_r"]

        ds = ds.drop_vars(["s_rho"])

        object.__setattr__(self, "ds", ds)

        ds["zeta"].load()
        nan_check(ds["zeta"].squeeze(), self.grid.ds.mask_rho)

    def plot(
        self,
        varname,
        s=None,
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
            - "PO4": Dissolved Inorganic Phosphate (mmol/m³).
            - "NO3": Dissolved Inorganic Nitrate (mmol/m³).
            - "SiO3": Dissolved Inorganic Silicate (mmol/m³).
            - "NH4": Dissolved Ammonia (mmol/m³).
            - "Fe": Dissolved Inorganic Iron (mmol/m³).
            - "Lig": Iron Binding Ligand (mmol/m³).
            - "O2": Dissolved Oxygen (mmol/m³).
            - "DIC": Dissolved Inorganic Carbon (mmol/m³).
            - "DIC_ALT_CO2": Dissolved Inorganic Carbon, Alternative CO2 (mmol/m³).
            - "ALK": Alkalinity (meq/m³).
            - "ALK_ALT_CO2": Alkalinity, Alternative CO2 (meq/m³).
            - "DOC": Dissolved Organic Carbon (mmol/m³).
            - "DON": Dissolved Organic Nitrogen (mmol/m³).
            - "DOP": Dissolved Organic Phosphorus (mmol/m³).
            - "DOPr": Refractory Dissolved Organic Phosphorus (mmol/m³).
            - "DONr": Refractory Dissolved Organic Nitrogen (mmol/m³).
            - "DOCr": Refractory Dissolved Organic Carbon (mmol/m³).
            - "zooC": Zooplankton Carbon (mmol/m³).
            - "spChl": Small Phytoplankton Chlorophyll (mg/m³).
            - "spC": Small Phytoplankton Carbon (mmol/m³).
            - "spP": Small Phytoplankton Phosphorous (mmol/m³).
            - "spFe": Small Phytoplankton Iron (mmol/m³).
            - "spCaCO3": Small Phytoplankton CaCO3 (mmol/m³).
            - "diatChl": Diatom Chlorophyll (mg/m³).
            - "diatC": Diatom Carbon (mmol/m³).
            - "diatP": Diatom Phosphorus (mmol/m³).
            - "diatFe": Diatom Iron (mmol/m³).
            - "diatSi": Diatom Silicate (mmol/m³).
            - "diazChl": Diazotroph Chlorophyll (mg/m³).
            - "diazC": Diazotroph Carbon (mmol/m³).
            - "diazP": Diazotroph Phosphorus (mmol/m³).
            - "diazFe": Diazotroph Iron (mmol/m³).
        s : int, optional
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
            If the specified `varname` is not one of the valid options.
            If the field specified by `varname` is 3D and none of `s`, `eta`, or `xi` are specified.
            If the field specified by `varname` is 2D and both `eta` and `xi` are specified.
        """

        if len(self.ds[varname].squeeze().dims) == 3 and not any(
            [s is not None, eta is not None, xi is not None]
        ):
            raise ValueError(
                "For 3D fields, at least one of s, eta, or xi must be specified."
            )

        if len(self.ds[varname].squeeze().dims) == 2 and all(
            [eta is not None, xi is not None]
        ):
            raise ValueError("For 2D fields, specify either eta or xi, not both.")

        self.ds[varname].load()
        field = self.ds[varname].squeeze()

        if all(dim in field.dims for dim in ["eta_rho", "xi_rho"]):
            interface_depth = self.ds.interface_depth_rho
        elif all(dim in field.dims for dim in ["eta_rho", "xi_u"]):
            interface_depth = self.ds.interface_depth_u
        elif all(dim in field.dims for dim in ["eta_v", "xi_rho"]):
            interface_depth = self.ds.interface_depth_v

        # slice the field as desired
        title = field.long_name
        if s is not None:
            title = title + f", s_rho = {field.s_rho[s].item()}"
            field = field.isel(s_rho=s)
        else:
            depth_contours = False

        if eta is not None:
            if "eta_rho" in field.dims:
                title = title + f", eta_rho = {field.eta_rho[eta].item()}"
                field = field.isel(eta_rho=eta)
                interface_depth = interface_depth.isel(eta_rho=eta)
            elif "eta_v" in field.dims:
                title = title + f", eta_v = {field.eta_v[eta].item()}"
                field = field.isel(eta_v=eta)
                interface_depth = interface_depth.isel(eta_v=eta)
            else:
                raise ValueError(
                    f"None of the expected dimensions (eta_rho, eta_v) found in ds[{varname}]."
                )
        if xi is not None:
            if "xi_rho" in field.dims:
                title = title + f", xi_rho = {field.xi_rho[xi].item()}"
                field = field.isel(xi_rho=xi)
                interface_depth = interface_depth.isel(xi_rho=xi)
            elif "xi_u" in field.dims:
                title = title + f", xi_u = {field.xi_u[xi].item()}"
                field = field.isel(xi_u=xi)
                interface_depth = interface_depth.isel(xi_u=xi)
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
            if varname in ["temp", "salt"]:
                cmap = plt.colormaps.get_cmap("YlOrRd")
            else:
                cmap = plt.colormaps.get_cmap("YlGn")
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
            if not layer_contours:
                interface_depth = None
            else:
                # restrict number of layer_contours to 10 for the sake of plot clearity
                nr_layers = len(interface_depth["s_w"])
                selected_layers = np.linspace(
                    0, nr_layers - 1, min(nr_layers, 10), dtype=int
                )
                interface_depth = interface_depth.isel(s_w=selected_layers)

            if len(field.dims) == 2:
                _section_plot(
                    field, interface_depth=interface_depth, title=title, kwargs=kwargs
                )
            else:
                if "s_rho" in field.dims:
                    _profile_plot(field, title=title)
                else:
                    _line_plot(field, title=title)

    def save(self, filepath: str) -> None:
        """
        Save the initial conditions information to a netCDF4 file.

        Parameters
        ----------
        filepath
        """
        self.ds.to_netcdf(filepath)

    def to_yaml(self, filepath: str) -> None:
        """
        Export the parameters of the class to a YAML file, including the version of roms-tools.

        Parameters
        ----------
        filepath : str
            The path to the YAML file where the parameters will be saved.
        """
        # Serialize Grid data
        grid_data = asdict(self.grid)
        grid_data.pop("ds", None)  # Exclude non-serializable fields
        grid_data.pop("straddle", None)

        # Serialize VerticalCoordinate data
        vertical_coordinate_data = asdict(self.vertical_coordinate)
        vertical_coordinate_data.pop("ds", None)  # Exclude non-serializable fields
        vertical_coordinate_data.pop("grid", None)  # Exclude non-serializable fields

        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"

        # Create header
        header = f"---\nroms_tools_version: {roms_tools_version}\n---\n"

        grid_yaml_data = {"Grid": grid_data}
        vertical_coordinate_yaml_data = {"VerticalCoordinate": vertical_coordinate_data}

        initial_conditions_data = {
            "InitialConditions": {
                "physics_source": self.physics_source,
                "ini_time": self.ini_time.isoformat(),
                "model_reference_date": self.model_reference_date.isoformat(),
            }
        }
        # Include bgc_source if it's not None
        if self.bgc_source is not None:
            initial_conditions_data["InitialConditions"]["bgc_source"] = self.bgc_source

        yaml_data = {
            **grid_yaml_data,
            **vertical_coordinate_yaml_data,
            **initial_conditions_data,
        }

        with open(filepath, "w") as file:
            # Write header
            file.write(header)
            # Write YAML data
            yaml.dump(yaml_data, file, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filepath: str) -> "InitialConditions":
        """
        Create an instance of the InitialConditions class from a YAML file.

        Parameters
        ----------
        filepath : str
            The path to the YAML file from which the parameters will be read.

        Returns
        -------
        InitialConditions
            An instance of the InitialConditions class.
        """
        # Read the entire file content
        with open(filepath, "r") as file:
            file_content = file.read()

        # Split the content into YAML documents
        documents = list(yaml.safe_load_all(file_content))

        initial_conditions_data = None

        # Process the YAML documents
        for doc in documents:
            if doc is None:
                continue
            if "InitialConditions" in doc:
                initial_conditions_data = doc["InitialConditions"]
                break

        if initial_conditions_data is None:
            raise ValueError(
                "No InitialConditions configuration found in the YAML file."
            )

        # Convert from string to datetime
        for date_string in ["model_reference_date", "ini_time"]:
            initial_conditions_data[date_string] = datetime.fromisoformat(
                initial_conditions_data[date_string]
            )

        # Create VerticalCoordinate instance from the YAML file
        vertical_coordinate = VerticalCoordinate.from_yaml(filepath)
        grid = vertical_coordinate.grid

        # Create and return an instance of InitialConditions
        return cls(
            grid=grid,
            vertical_coordinate=vertical_coordinate,
            **initial_conditions_data,
        )
