import xarray as xr
import numpy as np
import importlib.metadata
from dataclasses import dataclass, field
from typing import Dict, Union, List, Optional
from roms_tools.setup.grid import Grid
from datetime import datetime
from roms_tools.setup.datasets import GLORYSDataset, CESMBGCDataset
from roms_tools.setup.vertical_coordinate import compute_depth
from roms_tools.setup.utils import (
    nan_check,
    substitute_nans_by_fillvalue,
    get_variable_metadata,
    save_datasets,
    get_target_coords,
    rotate_velocities,
    compute_barotropic_velocity,
    transpose_dimensions,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    _to_yaml,
    _from_yaml,
)
from roms_tools.setup.regrid import LateralRegrid, VerticalRegrid
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

        processed_fields = {}
        processed_fields = self._process_data(processed_fields, type="physics")

        if self.bgc_source is not None:
            processed_fields = self._process_data(processed_fields, type="bgc")

        d_meta = get_variable_metadata()
        ds = self._write_into_dataset(processed_fields, d_meta)

        ds = self._add_global_metadata(ds)

        self._validate(ds)

        # substitute NaNs over land by a fill value to avoid blow-up of ROMS
        for var_name in ds.data_vars:
            ds[var_name] = substitute_nans_by_fillvalue(ds[var_name])

        object.__setattr__(self, "ds", ds)

    def _process_data(self, processed_fields, type="physics"):

        target_coords = get_target_coords(self.grid)

        if type == "physics":
            data = self._get_data()
        else:
            data = self._get_bgc_data()

        data.choose_subdomain(
            target_coords,
            buffer_points=20,  # lateral fill needs good buffer from data margin
        )

        data.extrapolate_deepest_to_bottom()
        data.apply_lateral_fill()

        self._set_variable_info(data, type=type)
        attr_name = f"variable_info_{type}"
        variable_info = getattr(self, attr_name)
        var_names = variable_info.keys()

        # lateral regridding
        lateral_regrid = LateralRegrid(target_coords, data.dim_names)
        for var_name in var_names:
            if var_name in data.var_names.keys():
                processed_fields[var_name] = lateral_regrid.apply(
                    data.ds[data.var_names[var_name]]
                )

        # rotation of velocities and interpolation to u/v points
        if "u" in variable_info and "v" in variable_info:
            (processed_fields["u"], processed_fields["v"],) = rotate_velocities(
                processed_fields["u"],
                processed_fields["v"],
                target_coords["angle"],
                interpolate=True,
            )

        var_names_dict = {}
        for location in ["rho", "u", "v"]:
            var_names_dict[location] = [
                name
                for name, info in variable_info.items()
                if info["location"] == location and info["is_3d"]
            ]

        # compute layer depth coordinates
        if len(var_names_dict["u"]) > 0 or len(var_names_dict["v"]) > 0:
            self._get_vertical_coordinates(
                type="layer",
                additional_locations=["u", "v"],
            )
        else:
            if len(var_names_dict["rho"]) > 0:
                self._get_vertical_coordinates(type="layer", additional_locations=[])
        # vertical regridding
        for location in ["rho", "u", "v"]:
            if len(var_names_dict[location]) > 0:
                vertical_regrid = VerticalRegrid(
                    self.grid.ds[f"layer_depth_{location}"],
                    data.ds[data.dim_names["depth"]],
                )
                for var_name in var_names_dict[location]:
                    if var_name in processed_fields:
                        processed_fields[var_name] = vertical_regrid.apply(
                            processed_fields[var_name]
                        )

        # compute barotropic velocities
        if "u" in variable_info and "v" in variable_info:
            self._get_vertical_coordinates(
                type="interface",
                additional_locations=["u", "v"],
            )
            for location in ["u", "v"]:
                processed_fields[f"{location}bar"] = compute_barotropic_velocity(
                    processed_fields[location],
                    self.grid.ds[f"interface_depth_{location}"],
                )

        if type == "bgc":
            # Ensure time coordinate matches that of physical variables
            for var_name in variable_info.keys():
                processed_fields[var_name] = processed_fields[var_name].assign_coords(
                    {"time": processed_fields["temp"]["time"]}
                )

        for var_name in processed_fields.keys():
            processed_fields[var_name] = transpose_dimensions(
                processed_fields[var_name]
            )

        return processed_fields

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

        Parameters
        ----------
        data : object
            The data object which contains variable names for the "bgc" type variables.

        type : str, optional, default="physics"
            The type of variable metadata to return. Can be one of:
            - "physics": for physical variables such as temperature, salinity, and velocity components.
            - "bgc": for biogeochemical variables (like ALK).

        Returns
        -------
        dict
            A dictionary where the keys are variable names and the values are dictionaries of metadata
            about each variable, including 'location', 'is_vector', 'vector_pair', 'is_3d', and 'validate'.
        """
        default_info = {
            "location": "rho",
            "is_vector": False,
            "vector_pair": None,
            "is_3d": True,
        }

        if type == "physics":
            variable_info = {
                "zeta": {
                    "location": "rho",
                    "is_vector": False,
                    "vector_pair": None,
                    "is_3d": False,
                    "validate": True,
                },
                "temp": {**default_info, "validate": False},
                "salt": {**default_info, "validate": False},
                "u": {
                    "location": "u",
                    "is_vector": True,
                    "vector_pair": "v",
                    "is_3d": True,
                    "validate": False,
                },
                "v": {
                    "location": "v",
                    "is_vector": True,
                    "vector_pair": "u",
                    "is_3d": True,
                    "validate": False,
                },
                "ubar": {
                    "location": "u",
                    "is_vector": True,
                    "vector_pair": "vbar",
                    "is_3d": False,
                    "validate": False,
                },
                "vbar": {
                    "location": "v",
                    "is_vector": True,
                    "vector_pair": "ubar",
                    "is_3d": False,
                    "validate": False,
                },
                "w": {
                    "location": "rho",
                    "is_vector": False,
                    "vector_pair": None,
                    "is_3d": True,
                    "validate": False,
                },
            }

        if type == "bgc":
            variable_info = {}
            for var_name in data.var_names.keys():
                if var_name == "ALK":
                    variable_info[var_name] = {**default_info, "validate": True}
                else:
                    variable_info[var_name] = {**default_info, "validate": False}

        object.__setattr__(self, f"variable_info_{type}", variable_info)

    def _get_vertical_coordinates(self, type, additional_locations=["u", "v"]):
        """Retrieve layer and interface depth coordinates.

        This method computes and updates the layer and interface depth coordinates. It handles depth calculations for rho points and
        additional specified locations (u and v).

        Parameters
        ----------
        type : str
            The type of depth coordinate to retrieve. Valid options are:
            - "layer": Retrieves layer depth coordinates.
            - "interface": Retrieves interface depth coordinates.

        additional_locations : list of str, optional
            Specifies additional locations to compute depth coordinates for. Default is ["u", "v"].
            Valid options include:
            - "u": Computes depth coordinates for u points.
            - "v": Computes depth coordinates for v points.

        Updates
        -------
        self.grid.ds : xarray.Dataset
            The dataset is updated with the following vertical depth coordinates:
            - f"{type}_depth_rho": Depth coordinates at rho points.
            - f"{type}_depth_u": Depth coordinates at u points (if applicable).
            - f"{type}_depth_v": Depth coordinates at v points (if applicable).
        """

        layer_vars = []
        for location in ["rho"] + additional_locations:
            layer_vars.append(f"{type}_depth_{location}")

        if all(layer_var in self.grid.ds for layer_var in layer_vars):
            # Vertical coordinate data already exists
            pass

        elif f"{type}_depth_rho" in self.grid.ds:
            depth = self.grid.ds[f"{type}_depth_rho"]

            if "u" in additional_locations or "v" in additional_locations:
                # interpolation
                if "u" in additional_locations:
                    depth_u = interpolate_from_rho_to_u(depth)
                    depth_u.attrs["long_name"] = f"{type} depth at u-points"
                    depth_u.attrs["units"] = "m"
                    self.grid.ds[f"{type}_depth_u"] = depth_u
                if "v" in additional_locations:
                    depth_v = interpolate_from_rho_to_v(depth)
                    depth_v.attrs["long_name"] = f"{type} depth at v-points"
                    depth_v.attrs["units"] = "m"
                    self.grid.ds[f"{type}_depth_v"] = depth_v
        else:
            h = self.grid.ds["h"]
            if type == "layer":
                depth = compute_depth(
                    0, h, self.grid.hc, self.grid.ds.Cs_r, self.grid.ds.sigma_r
                )
            else:
                depth = compute_depth(
                    0, h, self.grid.hc, self.grid.ds.Cs_w, self.grid.ds.sigma_w
                )

            depth.attrs["long_name"] = f"{type} depth at rho-points"
            depth.attrs["units"] = "m"
            self.grid.ds[f"{type}_depth_rho"] = depth

            if "u" in additional_locations or "v" in additional_locations:
                # interpolation
                depth_u = interpolate_from_rho_to_u(depth)
                depth_u.attrs["long_name"] = f"{type} depth at u-points"
                depth_u.attrs["units"] = "m"
                depth_v = interpolate_from_rho_to_v(depth)
                depth_v.attrs["long_name"] = f"{type} depth at v-points"
                depth_v.attrs["units"] = "m"
                self.grid.ds[f"{type}_depth_u"] = depth_u
                self.grid.ds[f"{type}_depth_v"] = depth_v

    def _write_into_dataset(self, processed_fields, d_meta):

        # save in new dataset
        ds = xr.Dataset()

        for var_name in processed_fields.keys():
            ds[var_name] = processed_fields[var_name].astype(np.float32)
            ds[var_name].attrs["long_name"] = d_meta[var_name]["long_name"]
            ds[var_name].attrs["units"] = d_meta[var_name]["units"]

        # initialize vertical velocity to zero
        ds["w"] = xr.zeros_like(
            self.grid.ds["interface_depth_rho"].expand_dims(
                time=processed_fields["u"].time
            )
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
        existing_vars = [var_name for var_name in variables_to_drop if var_name in ds]
        ds = ds.drop_vars(existing_vars)

        ds["Cs_r"] = self.grid.ds["Cs_r"]
        ds["Cs_w"] = self.grid.ds["Cs_w"]

        # Preserve absolute time coordinate for readability
        ds = ds.assign_coords({"abs_time": ds["time"]})

        # Translate the time coordinate to days since the model reference date
        model_reference_date = np.datetime64(self.model_reference_date)

        # Convert the time coordinate to the format expected by ROMS (seconds since model reference date)
        ocean_time = (ds["time"] - model_reference_date).astype("float64") * 1e-9
        ds = ds.assign_coords(ocean_time=("time", ocean_time.data.astype("float64")))
        ds["ocean_time"].attrs[
            "long_name"
        ] = f"seconds since {str(self.model_reference_date)}"
        ds["ocean_time"].attrs["units"] = "seconds"
        ds = ds.swap_dims({"time": "ocean_time"})
        ds = ds.drop_vars("time")

        return ds

    def _validate(self, ds):
        """Validates the dataset by checking for NaN values in SSH at wet points, which
        would indicate missing raw data coverage over the target domain.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to validate.

        Raises
        ------
        ValueError
            If NaN values are found in any of the specified variables at wet points,
            indicating incomplete data coverage.

        Notes
        -----
        This check is only applied to the 2D variable SSH to improve performance.
        """
        if self.bgc_source is not None:
            variable_info = {**self.variable_info_physics, **self.variable_info_bgc}
        else:
            variable_info = self.variable_info_physics

        for var_name in variable_info:
            # Only validate variables based on "validate" flag if use_dask is False
            if not self.use_dask or variable_info[var_name]["validate"]:
                ds[var_name].load()
                nan_check(ds[var_name].squeeze(), self.grid.ds.mask_rho)

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
        var_name,
        s=None,
        eta=None,
        xi=None,
        depth_contours=False,
        layer_contours=False,
        ax=None,
    ) -> None:
        """Plot the initial conditions field for a given eta-, xi-, or s_rho- slice.

        Parameters
        ----------
        var_name : str
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
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure is created. Note that this argument does not work for horizontal plots that display the eta- and xi-dimensions at the same time.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        Raises
        ------
        ValueError
            If the specified `var_name` is not one of the valid options.
            If the field specified by `var_name` is 3D and none of `s`, `eta`, or `xi` are specified.
            If the field specified by `var_name` is 2D and both `eta` and `xi` are specified.
        """

        if len(self.ds[var_name].squeeze().dims) == 3 and not any(
            [s is not None, eta is not None, xi is not None]
        ):
            raise ValueError(
                "For 3D fields, at least one of s, eta, or xi must be specified."
            )

        if len(self.ds[var_name].squeeze().dims) == 2 and all(
            [eta is not None, xi is not None]
        ):
            raise ValueError("For 2D fields, specify either eta or xi, not both.")

        if self.use_dask:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                self.ds[var_name].load()

        field = self.ds[var_name].squeeze()
        if s is not None:
            layer_contours = False

        if all(dim in field.dims for dim in ["eta_rho", "xi_rho"]):
            if layer_contours:
                if "interface_depth_rho" in self.grid.ds:
                    interface_depth = self.grid.ds.interface_depth_rho
                else:
                    self.get_vertical_coordinates(
                        type="interface", additional_locations=[]
                    )
            layer_depth = self.grid.ds.layer_depth_rho
            mask = self.grid.ds.mask_rho
            field = field.assign_coords(
                {"lon": self.grid.ds.lon_rho, "lat": self.grid.ds.lat_rho}
            )

        elif all(dim in field.dims for dim in ["eta_rho", "xi_u"]):
            if layer_contours:
                if "interface_depth_u" in self.grid.ds:
                    interface_depth = self.grid.ds.interface_depth_u
                else:
                    self.get_vertical_coordinates(
                        type="interface", additional_locations=["u", "v"]
                    )
            layer_depth = self.grid.ds.layer_depth_u
            mask = self.grid.ds.mask_u
            field = field.assign_coords(
                {"lon": self.grid.ds.lon_u, "lat": self.grid.ds.lat_u}
            )

        elif all(dim in field.dims for dim in ["eta_v", "xi_rho"]):
            if layer_contours:
                if "interface_depth_v" in self.grid.ds:
                    interface_depth = self.grid.ds.interface_depth_v
                else:
                    self.get_vertical_coordinates(
                        type="interface", additional_locations=["u", "v"]
                    )
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
                if layer_contours:
                    interface_depth = interface_depth.isel(eta_rho=eta)
                if "s_rho" in field.dims:
                    field = field.assign_coords({"layer_depth": layer_depth})
            elif "eta_v" in field.dims:
                title = title + f", eta_v = {field.eta_v[eta].item()}"
                field = field.isel(eta_v=eta)
                layer_depth = layer_depth.isel(eta_v=eta)
                if layer_contours:
                    interface_depth = interface_depth.isel(eta_v=eta)
                if "s_rho" in field.dims:
                    field = field.assign_coords({"layer_depth": layer_depth})
            else:
                raise ValueError(
                    f"None of the expected dimensions (eta_rho, eta_v) found in ds[{var_name}]."
                )
        if xi is not None:
            if "xi_rho" in field.dims:
                title = title + f", xi_rho = {field.xi_rho[xi].item()}"
                field = field.isel(xi_rho=xi)
                layer_depth = layer_depth.isel(xi_rho=xi)
                if layer_contours:
                    interface_depth = interface_depth.isel(xi_rho=xi)
                if "s_rho" in field.dims:
                    field = field.assign_coords({"layer_depth": layer_depth})
            elif "xi_u" in field.dims:
                title = title + f", xi_u = {field.xi_u[xi].item()}"
                field = field.isel(xi_u=xi)
                layer_depth = layer_depth.isel(xi_u=xi)
                if layer_contours:
                    interface_depth = interface_depth.isel(xi_u=xi)
                if "s_rho" in field.dims:
                    field = field.assign_coords({"layer_depth": layer_depth})
            else:
                raise ValueError(
                    f"None of the expected dimensions (xi_rho, xi_u) found in ds[{var_name}]."
                )

        # chose colorbar
        if var_name in ["u", "v", "w", "ubar", "vbar", "zeta"]:
            vmax = max(field.max().values, -field.min().values)
            vmin = -vmax
            cmap = plt.colormaps.get_cmap("RdBu_r")
        else:
            vmax = field.max().values
            vmin = field.min().values
            if var_name in ["temp", "salt"]:
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
                    field,
                    interface_depth=interface_depth,
                    title=title,
                    kwargs=kwargs,
                    ax=ax,
                )
            else:
                if "s_rho" in field.dims:
                    _profile_plot(field, title=title, ax=ax)
                else:
                    _line_plot(field, title=title, ax=ax)

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

        if self.use_dask:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                self.ds.load()

        dataset_list = [self.ds]
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

        _to_yaml(self, filepath)

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

        grid = Grid.from_yaml(filepath)
        initial_conditions_params = _from_yaml(cls, filepath)
        return cls(grid=grid, **initial_conditions_params, use_dask=use_dask)
