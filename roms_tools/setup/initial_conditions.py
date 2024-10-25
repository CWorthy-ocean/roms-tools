import xarray as xr
import numpy as np
import yaml
import importlib.metadata
from dataclasses import dataclass, field, asdict
from typing import Dict, Union, List, Optional
from roms_tools.setup.grid import Grid
from datetime import datetime
from roms_tools.setup.datasets import GLORYSDataset, CESMBGCDataset
from roms_tools.setup.utils import (
    nan_check,
    substitute_nans_by_fillvalue,
    get_variable_metadata,
    save_datasets,
    get_target_coords,
    rotate_velocities,
    compute_barotropic_velocity,
    _extrapolate_deepest_to_bottom,
    transpose_dimensions,
)
from roms_tools.setup.fill import _lateral_fill
from roms_tools.setup.regrid import _lateral_regrid, _vertical_regrid
from roms_tools.setup.plot import _plot, _section_plot, _profile_plot, _line_plot
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class InitialConditions:
    """Represents initial conditions for ROMS, including physical and biogeochemical
    data.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information used for the model.
    ini_time : datetime
        The date and time at which the initial conditions are set.
        If no exact match is found, the closest time entry to `ini_time` within the time range [ini_time, ini_time + 24 hours] is selected.
    source : Dict[str, Union[str, Path, List[Union[str, Path]]], bool]

        Dictionary specifying the source of the physical initial condition data. Keys include:

          - "name" (str): Name of the data source (e.g., "GLORYS").
          - "path" (Union[str, Path, List[Union[str, Path]]]): The path to the raw data file(s). This can be:

            - A single string (with or without wildcards).
            - A single Path object.
            - A list of strings or Path objects containing multiple files.
          - "climatology" (bool): Indicates if the data is climatology data. Defaults to False.

    bgc_source : Dict[str, Union[str, Path, List[Union[str, Path]]], bool]
        Dictionary specifying the source of the biogeochemical (BGC) initial condition data. Keys include:

          - "name" (str): Name of the data source (e.g., "CESM_REGRIDDED").
          - "path" (Union[str, Path, List[Union[str, Path]]]): The path to the raw data file(s). This can be:

            - A single string (with or without wildcards).
            - A single Path object.
            - A list of strings or Path objects containing multiple files.
          - "climatology" (bool): Indicates if the data is climatology data. Defaults to False.

    model_reference_date : datetime, optional
        The reference date for the model. Defaults to January 1, 2000.
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.

    Examples
    --------
    >>> initial_conditions = InitialConditions(
    ...     grid=grid,
    ...     ini_time=datetime(2022, 1, 1),
    ...     source={"name": "GLORYS", "path": "physics_data.nc"},
    ...     bgc_source={
    ...         "name": "CESM_REGRIDDED",
    ...         "path": "bgc_data.nc",
    ...         "climatology": False,
    ...     },
    ... )
    """

    grid: Grid
    ini_time: datetime
    source: Dict[str, Union[str, Path, List[Union[str, Path]]]]
    bgc_source: Optional[Dict[str, Union[str, Path, List[Union[str, Path]]]]] = None
    model_reference_date: datetime = datetime(2000, 1, 1)
    use_dask: bool = False

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        self._input_checks()

        data_vars = {}
        data_vars = self._process_data(data_vars, type="physics")

        if self.bgc_source is not None:
            data_vars = self._process_data(data_vars, type="bgc")

        for var in data_vars.keys():
            data_vars[var] = transpose_dimensions(data_vars[var])

        d_meta = get_variable_metadata()
        ds = self._write_into_dataset(data_vars, d_meta)

        ds = self._add_global_metadata(ds)

        ds["zeta"].load()
        # NaN values at wet points indicate that the raw data did not cover the domain, and the following will raise a ValueError
        nan_check(ds["zeta"].squeeze(), self.grid.ds.mask_rho)

        # substitute NaNs over land by a fill value to avoid blow-up of ROMS
        for var in ds.data_vars:
            ds[var] = substitute_nans_by_fillvalue(ds[var])

        object.__setattr__(self, "ds", ds)

    def _process_data(self, data_vars, type="physics"):

        target_coords = get_target_coords(self.grid)

        if type == "physics":
            data = self._get_data()
        else:
            data = self._get_bgc_data()

        data.choose_subdomain(
            target_coords,
            buffer_points=20,  # lateral fill needs good buffer from data margin
        )

        variable_info = self._set_variable_info(data, type=type)

        data_vars = _extrapolate_deepest_to_bottom(data_vars, data)

        data_vars = _lateral_fill(data_vars, data)

        # lateral regridding
        var_names = variable_info.keys()
        data_vars = _lateral_regrid(
            data, target_coords["lon"], target_coords["lat"], data_vars, var_names
        )

        # rotation of velocities and interpolation to u/v points
        if "u" in variable_info and "v" in variable_info:
            (data_vars["u"], data_vars["v"],) = rotate_velocities(
                data_vars["u"],
                data_vars["v"],
                target_coords["angle"],
                interpolate=True,
            )

        # vertical regridding
        for location in ["rho", "u", "v"]:
            var_names = [
                name
                for name, info in variable_info.items()
                if info["location"] == location and info["is_3d"]
            ]
            if len(var_names) > 0:
                data_vars = _vertical_regrid(
                    data,
                    self.grid.ds[f"layer_depth_{location}"],
                    data_vars,
                    var_names,
                )

        # compute barotropic velocities
        if "u" in variable_info and "v" in variable_info:
            for var in ["u", "v"]:
                data_vars[f"{var}bar"] = compute_barotropic_velocity(
                    data_vars[var], self.grid.ds[f"interface_depth_{var}"]
                )

        if type == "bgc":
            # Ensure time coordinate matches that of physical variables
            for var in variable_info.keys():
                data_vars[var] = data_vars[var].assign_coords(
                    {"time": data_vars["temp"]["time"]}
                )

        return data_vars

    def _input_checks(self):

        if "name" not in self.source.keys():
            raise ValueError("`source` must include a 'name'.")
        if "path" not in self.source.keys():
            raise ValueError("`source` must include a 'path'.")
        # set self.source["climatology"] to False if not provided
        object.__setattr__(
            self,
            "source",
            {
                **self.source,
                "climatology": self.source.get("climatology", False),
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
            # set self.bgc_source["climatology"] to False if not provided
            object.__setattr__(
                self,
                "bgc_source",
                {
                    **self.bgc_source,
                    "climatology": self.bgc_source.get("climatology", False),
                },
            )

    def _get_data(self):

        if self.source["name"] == "GLORYS":
            data = GLORYSDataset(
                filename=self.source["path"],
                start_time=self.ini_time,
                climatology=self.source["climatology"],
                use_dask=self.use_dask,
            )
        else:
            raise ValueError('Only "GLORYS" is a valid option for source["name"].')
        return data

    def _get_bgc_data(self):

        if self.bgc_source["name"] == "CESM_REGRIDDED":

            data = CESMBGCDataset(
                filename=self.bgc_source["path"],
                start_time=self.ini_time,
                climatology=self.bgc_source["climatology"],
                use_dask=self.use_dask,
            )
        else:
            raise ValueError(
                'Only "CESM_REGRIDDED" is a valid option for bgc_source["name"].'
            )

        return data

    def _set_variable_info(self, data, type="physics"):
        """Sets up a dictionary with metadata for variables based on the type.

        The dictionary contains the following information:
        - `location`: Where the variable resides in the grid (e.g., rho, u, or v points).
        - `is_vector`: Whether the variable is part of a vector (True for velocity components like 'u' and 'v').
        - `vector_pair`: For vector variables, this indicates the associated variable that forms the vector (e.g., 'u' and 'v').
        - `is_3d`: Indicates whether the variable is 3D (True for variables like 'temp' and 'salt') or 2D (False for 'zeta').

        Returns
        -------
        dict
            A dictionary where the keys are variable names and the values are dictionaries of metadata
            about each variable, including 'location', 'is_vector', 'vector_pair', and 'is_3d'.
        """
        default_info = {
            "location": "rho",
            "is_vector": False,
            "vector_pair": None,
            "is_3d": True,
        }

        # Define a dictionary for variable names and their associated information
        if type == "physics":
            variable_info = {
                "zeta": {
                    "location": "rho",
                    "is_vector": False,
                    "vector_pair": None,
                    "is_3d": False,
                },
                "temp": default_info,
                "salt": default_info,
                "u": {
                    "location": "u",
                    "is_vector": True,
                    "vector_pair": "v",
                    "is_3d": True,
                },
                "v": {
                    "location": "v",
                    "is_vector": True,
                    "vector_pair": "u",
                    "is_3d": True,
                },
                "ubar": {
                    "location": "u",
                    "is_vector": True,
                    "vector_pair": "vbar",
                    "is_3d": False,
                },
                "vbar": {
                    "location": "v",
                    "is_vector": True,
                    "vector_pair": "ubar",
                    "is_3d": False,
                },
            }
        elif type == "bgc":
            variable_info = {}
            for var in data.var_names.keys():
                variable_info[var] = default_info

        return variable_info

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
            "lat_u",
            "lon_u",
            "lat_v",
            "lon_v",
            "layer_depth_rho",
            "interface_depth_rho",
            "layer_depth_u",
            "interface_depth_u",
            "layer_depth_v",
            "interface_depth_v",
        ]
        existing_vars = [var for var in variables_to_drop if var in ds]
        ds = ds.drop_vars(existing_vars)

        ds["Cs_r"] = self.grid.ds["Cs_r"]
        ds["Cs_w"] = self.grid.ds["Cs_w"]

        # Preserve absolute time coordinate for readability
        ds = ds.assign_coords({"abs_time": ds["time"]})

        # Translate the time coordinate to days since the model reference date
        model_reference_date = np.datetime64(self.model_reference_date)

        # Convert the time coordinate to the format expected by ROMS (days since model reference date)
        ocean_time = (ds["time"] - model_reference_date).astype("float64") * 1e-9
        ds = ds.assign_coords(ocean_time=("time", ocean_time.data.astype("float64")))
        ds["ocean_time"].attrs[
            "long_name"
        ] = f"seconds since {str(self.model_reference_date)}"
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
        ds.attrs["source"] = self.source["name"]
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
        """Plot the initial conditions field for a given eta-, xi-, or s_rho- slice.

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
            The index of the vertical layer (`s_rho`) to plot. If not specified, the plot
            will represent a horizontal slice (eta- or xi- plane). Default is None.
        eta : int, optional
            The eta-index to plot. Used for vertical sections or horizontal slices.
            Default is None.
        xi : int, optional
            The xi-index to plot. Used for vertical sections or horizontal slices.
            Default is None.
        depth_contours : bool, optional
            If True, depth contours will be overlaid on the plot, showing lines of constant
            depth. This is typically used for plots that show a single vertical layer.
            Default is False.
        layer_contours : bool, optional
            If True, contour lines representing the boundaries between vertical layers will
            be added to the plot. This is particularly useful in vertical sections to
            visualize the layering of the water column. For clarity, the number of layer
            contours displayed is limited to a maximum of 10. Default is False.

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
            mask = self.grid.ds.mask_rho
            field = field.assign_coords(
                {"lon": self.grid.ds.lon_rho, "lat": self.grid.ds.lat_rho}
            )

        elif all(dim in field.dims for dim in ["eta_rho", "xi_u"]):
            interface_depth = self.grid.ds.interface_depth_u
            layer_depth = self.grid.ds.layer_depth_u
            mask = self.grid.ds.mask_u
            field = field.assign_coords(
                {"lon": self.grid.ds.lon_u, "lat": self.grid.ds.lat_u}
            )

        elif all(dim in field.dims for dim in ["eta_v", "xi_rho"]):
            interface_depth = self.grid.ds.interface_depth_v
            layer_depth = self.grid.ds.layer_depth_v
            mask = self.grid.ds.mask_v
            field = field.assign_coords(
                {"lon": self.grid.ds.lon_v, "lat": self.grid.ds.lat_v}
            )
        else:
            ValueError("provided field does not have two horizontal dimension")

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
                interface_depth = interface_depth.isel(eta_rho=eta)
                if "s_rho" in field.dims:
                    field = field.assign_coords({"layer_depth": layer_depth})
            elif "eta_v" in field.dims:
                title = title + f", eta_v = {field.eta_v[eta].item()}"
                field = field.isel(eta_v=eta)
                layer_depth = layer_depth.isel(eta_v=eta)
                interface_depth = interface_depth.isel(eta_v=eta)
                if "s_rho" in field.dims:
                    field = field.assign_coords({"layer_depth": layer_depth})
            else:
                raise ValueError(
                    f"None of the expected dimensions (eta_rho, eta_v) found in ds[{varname}]."
                )
        if xi is not None:
            if "xi_rho" in field.dims:
                title = title + f", xi_rho = {field.xi_rho[xi].item()}"
                field = field.isel(xi_rho=xi)
                layer_depth = layer_depth.isel(xi_rho=xi)
                interface_depth = interface_depth.isel(xi_rho=xi)
                if "s_rho" in field.dims:
                    field = field.assign_coords({"layer_depth": layer_depth})
            elif "xi_u" in field.dims:
                title = title + f", xi_u = {field.xi_u[xi].item()}"
                field = field.isel(xi_u=xi)
                layer_depth = layer_depth.isel(xi_u=xi)
                interface_depth = interface_depth.isel(xi_u=xi)
                if "s_rho" in field.dims:
                    field = field.assign_coords({"layer_depth": layer_depth})
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
                field=field.where(mask),
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

    def save(
        self, filepath: Union[str, Path], np_eta: int = None, np_xi: int = None
    ) -> None:
        """Save the initial conditions information to a netCDF4 file.

        This method supports saving the dataset in two modes:

          1. **Single File Mode (default)**:

            If both `np_eta` and `np_xi` are `None`, the entire dataset is saved as a single netCDF4 file
            with the base filename specified by `filepath.nc`.

          2. **Partitioned Mode**:

            - If either `np_eta` or `np_xi` is specified, the dataset is divided into spatial tiles along the eta-axis and xi-axis.
            - Each spatial tile is saved as a separate netCDF4 file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The base path or filename where the dataset should be saved.
        np_eta : int, optional
            The number of partitions along the `eta` direction. If `None`, no spatial partitioning is performed.
        np_xi : int, optional
            The number of partitions along the `xi` direction. If `None`, no spatial partitioning is performed.

        Returns
        -------
        List[Path]
            A list of Path objects for the filenames that were saved.
        """

        # Ensure filepath is a Path object
        filepath = Path(filepath)

        # Remove ".nc" suffix if present
        if filepath.suffix == ".nc":
            filepath = filepath.with_suffix("")

        dataset_list = [self.ds.load()]
        output_filenames = [str(filepath)]

        saved_filenames = save_datasets(
            dataset_list, output_filenames, np_eta=np_eta, np_xi=np_xi
        )

        return saved_filenames

    def to_yaml(self, filepath: Union[str, Path]) -> None:
        """Export the parameters of the class to a YAML file, including the version of
        roms-tools.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file where the parameters will be saved.
        """
        filepath = Path(filepath)

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
                "source": self.source,
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

        with filepath.open("w") as file:
            # Write header
            file.write(header)
            # Write YAML data
            yaml.dump(yaml_data, file, default_flow_style=False)

    @classmethod
    def from_yaml(
        cls, filepath: Union[str, Path], use_dask: bool = False
    ) -> "InitialConditions":
        """Create an instance of the InitialConditions class from a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file from which the parameters will be read.
        use_dask: bool, optional
            Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.

        Returns
        -------
        InitialConditions
            An instance of the InitialConditions class.
        """
        filepath = Path(filepath)
        # Read the entire file content
        with filepath.open("r") as file:
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
        return cls(grid=grid, **initial_conditions_data, use_dask=use_dask)
