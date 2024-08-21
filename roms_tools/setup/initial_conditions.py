import xarray as xr
import numpy as np
import yaml
import importlib.metadata
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Union
from roms_tools.setup.grid import Grid
from datetime import datetime
from roms_tools.setup.datasets import GLORYSDataset, CESMBGCDataset
from roms_tools.setup.utils import (
    nan_check,
)
from roms_tools.setup.mixins import ROMSToolsMixins
from roms_tools.setup.plot import _plot, _section_plot, _profile_plot, _line_plot
import matplotlib.pyplot as plt


@dataclass(frozen=True, kw_only=True)
class InitialConditions(ROMSToolsMixins):
    """
    Represents initial conditions for ROMS, including physical and biogeochemical data.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information used for the model.
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
        - "climatology" (bool): Indicates if the BGC data is climatology data. Defaults to True.
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
    ...     ini_time=datetime(2022, 1, 1),
    ...     physics_source={"name": "GLORYS", "path": "physics_data.nc"},
    ...     bgc_source={
    ...         "name": "CESM_REGRIDDED",
    ...         "path": "bgc_data.nc",
    ...         "climatology": True,
    ...     },
    ... )
    """

    grid: Grid
    ini_time: datetime
    physics_source: Dict[str, Union[str, None]]
    bgc_source: Optional[Dict[str, Union[str, None]]] = None
    model_reference_date: datetime = datetime(2000, 1, 1)

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        self._input_checks()
        lon, lat, angle, straddle = super().get_target_lon_lat()

        data = self._get_data()
        data.choose_subdomain(
            latitude_range=[lat.min().values, lat.max().values],
            longitude_range=[lon.min().values, lon.max().values],
            margin=2,
            straddle=straddle,
        )

        vars_2d = ["zeta"]
        vars_3d = ["temp", "salt", "u", "v"]
        data_vars = super().regrid_data(data, vars_2d, vars_3d, lon, lat)
        data_vars = super().process_velocities(data_vars, angle, "u", "v")

        if self.bgc_source is not None:
            bgc_data = self._get_bgc_data()
            bgc_data.choose_subdomain(
                latitude_range=[lat.min().values, lat.max().values],
                longitude_range=[lon.min().values, lon.max().values],
                margin=2,
                straddle=straddle,
            )

            vars_2d = []
            vars_3d = bgc_data.var_names.keys()
            bgc_data_vars = super().regrid_data(bgc_data, vars_2d, vars_3d, lon, lat)

            # Ensure time coordinate matches if climatology is applied in one case but not the other
            if (
                not self.physics_source["climatology"]
                and self.bgc_source["climatology"]
            ):
                for var in bgc_data_vars.keys():
                    bgc_data_vars[var] = bgc_data_vars[var].assign_coords(
                        {"time": data_vars["temp"]["time"]}
                    )

            # Combine data variables from physical and biogeochemical sources
            data_vars.update(bgc_data_vars)

        d_meta = super().get_variable_metadata()
        ds = self._write_into_dataset(data_vars, d_meta)

        ds = self._add_global_metadata(ds)

        ds["zeta"].load()
        nan_check(ds["zeta"].squeeze(), self.grid.ds.mask_rho)

        object.__setattr__(self, "ds", ds)

    def _input_checks(self):

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
        if self.bgc_source is not None:
            if "name" not in self.bgc_source.keys():
                raise ValueError(
                    "`bgc_source` must include a 'name' if it is provided."
                )
            if "path" not in self.bgc_source.keys():
                raise ValueError(
                    "`bgc_source` must include a 'path' if it is provided."
                )
            # set self.bgc_source["climatology"] to True if not provided
            object.__setattr__(
                self,
                "bgc_source",
                {
                    **self.bgc_source,
                    "climatology": self.bgc_source.get("climatology", True),
                },
            )

    def _get_data(self):

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
        return data

    def _get_bgc_data(self):

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

        return bgc_data

    def _write_into_dataset(self, data_vars, d_meta):

        # save in new dataset
        ds = xr.Dataset()

        for var in data_vars.keys():
            ds[var] = data_vars[var].astype(np.float32)
            ds[var].attrs["long_name"] = d_meta[var]["long_name"]
            ds[var].attrs["units"] = d_meta[var]["units"]

        # initialize vertical velocity to zero
        ds["w"] = xr.zeros_like(
            self.grid.ds["interface_depth_rho"].expand_dims(time=data_vars["u"].time)
        ).astype(np.float32)
        ds["w"].attrs["long_name"] = d_meta["w"]["long_name"]
        ds["w"].attrs["units"] = d_meta["w"]["units"]

        variables_to_drop = [
            "s_rho",
            "lat_rho",
            "lon_rho",
            "layer_depth_rho",
            "interface_depth_rho",
            "lat_u",
            "lon_u",
            "lat_v",
            "lon_v",
        ]
        existing_vars = [var for var in variables_to_drop if var in ds]
        ds = ds.drop_vars(existing_vars)

        ds["sc_r"] = self.grid.ds["sc_r"]
        ds["Cs_r"] = self.grid.ds["Cs_r"]

        # Preserve absolute time coordinate for readability
        ds = ds.assign_coords({"abs_time": ds["time"]})

        # Translate the time coordinate to days since the model reference date
        model_reference_date = np.datetime64(self.model_reference_date)

        # Convert the time coordinate to the format expected by ROMS (days since model reference date)
        ocean_time = (ds["time"] - model_reference_date).astype("float64") * 1e-9
        ds = ds.assign_coords(ocean_time=("time", np.float32(ocean_time)))
        ds["ocean_time"].attrs[
            "long_name"
        ] = f"seconds since {np.datetime_as_string(model_reference_date, unit='s')}"
        ds["ocean_time"].attrs["units"] = "seconds"
        ds = ds.swap_dims({"time": "ocean_time"})
        ds = ds.drop_vars("time")

        return ds

    def _add_global_metadata(self, ds):

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

        ds.attrs["theta_s"] = self.grid.ds.attrs["theta_s"]
        ds.attrs["theta_b"] = self.grid.ds.attrs["theta_b"]
        ds.attrs["hc"] = self.grid.ds.attrs["hc"]

        return ds

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
            interface_depth = self.grid.ds.interface_depth_rho
            layer_depth = self.grid.ds.layer_depth_rho
        elif all(dim in field.dims for dim in ["eta_rho", "xi_u"]):
            interface_depth = self.grid.ds.interface_depth_u
            layer_depth = self.grid.ds.layer_depth_u
        elif all(dim in field.dims for dim in ["eta_v", "xi_rho"]):
            interface_depth = self.grid.ds.interface_depth_v
            layer_depth = self.grid.ds.layer_depth_v

        # slice the field as desired
        title = field.long_name
        if s is not None:
            title = title + f", s_rho = {field.s_rho[s].item()}"
            field = field.isel(s_rho=s)
            layer_depth = layer_depth.isel(s_rho=s)
            field = field.assign_coords({"layer_depth": layer_depth})
        else:
            depth_contours = False

        if eta is not None:
            if "eta_rho" in field.dims:
                title = title + f", eta_rho = {field.eta_rho[eta].item()}"
                field = field.isel(eta_rho=eta)
                layer_depth = layer_depth.isel(eta_rho=eta)
                field = field.assign_coords({"layer_depth": layer_depth})
                interface_depth = interface_depth.isel(eta_rho=eta)
            elif "eta_v" in field.dims:
                title = title + f", eta_v = {field.eta_v[eta].item()}"
                field = field.isel(eta_v=eta)
                layer_depth = layer_depth.isel(eta_v=eta)
                field = field.assign_coords({"layer_depth": layer_depth})
                interface_depth = interface_depth.isel(eta_v=eta)
            else:
                raise ValueError(
                    f"None of the expected dimensions (eta_rho, eta_v) found in ds[{varname}]."
                )
        if xi is not None:
            if "xi_rho" in field.dims:
                title = title + f", xi_rho = {field.xi_rho[xi].item()}"
                field = field.isel(xi_rho=xi)
                layer_depth = layer_depth.isel(xi_rho=xi)
                field = field.assign_coords({"layer_depth": layer_depth})
                interface_depth = interface_depth.isel(xi_rho=xi)
            elif "xi_u" in field.dims:
                title = title + f", xi_u = {field.xi_u[xi].item()}"
                field = field.isel(xi_u=xi)
                layer_depth = layer_depth.isel(xi_u=xi)
                field = field.assign_coords({"layer_depth": layer_depth})
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

        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"

        # Create header
        header = f"---\nroms_tools_version: {roms_tools_version}\n---\n"

        grid_yaml_data = {"Grid": grid_data}

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

        grid = Grid.from_yaml(filepath)

        # Create and return an instance of InitialConditions
        return cls(
            grid=grid,
            **initial_conditions_data,
        )
