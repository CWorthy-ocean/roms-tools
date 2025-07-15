import importlib.metadata
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from roms_tools import Grid
from roms_tools.plot import plot
from roms_tools.regrid import LateralRegridToROMS, VerticalRegridToROMS
from roms_tools.setup.datasets import CESMBGCDataset, GLORYSDataset, UnifiedBGCDataset
from roms_tools.setup.utils import (
    _from_yaml,
    _to_dict,
    _write_to_yaml,
    compute_barotropic_velocity,
    compute_missing_bgc_variables,
    get_target_coords,
    get_variable_metadata,
    nan_check,
    rotate_velocities,
    substitute_nans_by_fillvalue,
)
from roms_tools.utils import (
    get_dask_chunks,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    save_datasets,
    transpose_dimensions,
)
from roms_tools.vertical_coordinate import (
    compute_depth,
)


@dataclass(kw_only=True)
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

    adjust_depth_for_sea_surface_height : bool, optional
        Whether to account for sea surface height variations when computing depth coordinates.
        Defaults to `False`.
    model_reference_date : datetime, optional
        The reference date for the model. Defaults to January 1, 2000.
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.
    horizontal_chunk_size : int, optional
        The chunk size used for horizontal partitioning for the vertical regridding when `use_dask = True`. Defaults to 50.
        A larger number results in a bigger memory footprint but faster computations.
        A smaller number results in a smaller memory footprint but slower computations.
    bypass_validation: bool, optional
        Indicates whether to skip validation checks in the processed data. When set to True,
        the validation process that ensures no NaN values exist at wet points
        in the processed dataset is bypassed. Defaults to False.

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
    adjust_depth_for_sea_surface_height: bool = False
    use_dask: bool = False
    horizontal_chunk_size: int = 50
    bypass_validation: bool = False

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        self._input_checks()
        # Dataset for depth coordinates
        self.ds_depth_coords = xr.Dataset()

        processed_fields = {}
        processed_fields = self._process_data(processed_fields, type="physics")

        if self.bgc_source is not None:
            processed_fields = self._process_data(processed_fields, type="bgc")
            processed_fields = compute_missing_bgc_variables(processed_fields)

        for var_name in processed_fields:
            processed_fields[var_name] = transpose_dimensions(
                processed_fields[var_name]
            )

        d_meta = get_variable_metadata()
        ds = self._write_into_dataset(processed_fields, d_meta)

        ds = self._add_global_metadata(ds)

        if not self.bypass_validation:
            self._validate(ds)

        # substitute NaNs over land by a fill value to avoid blow-up of ROMS
        for var_name in ds.data_vars:
            ds[var_name] = substitute_nans_by_fillvalue(ds[var_name])

        self.ds = ds

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
        # Enforce double precision to ensure reproducibility
        data.convert_to_float64()
        data.extrapolate_deepest_to_bottom()
        data.apply_lateral_fill()

        self._set_variable_info(data, type=type)
        attr_name = f"variable_info_{type}"
        variable_info = getattr(self, attr_name)

        # Create the var_names dictionary, associating each variable with its location
        # Avoid looping over processed_fields.keys() directly, as they may already contain
        # finalized physics variables. This is especially important when transitioning
        # to processing biogeochemical (BGC) variables, ensuring that only relevant
        # variables are processed.
        var_names = {
            var: {
                "name": data.var_names[var],
                "location": variable_info[var]["location"],
            }
            for var in data.var_names.keys()
            if data.var_names[var] in data.ds.data_vars
        }
        # Update the dictionary with optional variables and their locations
        var_names.update(
            {
                var: {
                    "name": data.opt_var_names[var],
                    "location": variable_info[var]["location"],
                }
                for var in data.opt_var_names.keys()
                if data.opt_var_names[var] in data.ds.data_vars
            }
        )

        # lateral regridding
        lateral_regrid = LateralRegridToROMS(target_coords, data.dim_names)

        for var_name in var_names:
            processed_fields[var_name] = lateral_regrid.apply(
                data.ds[var_names[var_name]["name"]]
            )

        # rotation of velocities and interpolation to u/v points
        if "u" in var_names and "v" in var_names:
            processed_fields["u"], processed_fields["v"] = rotate_velocities(
                processed_fields["u"],
                processed_fields["v"],
                target_coords["angle"],
                interpolate=True,
            )

        if type == "bgc":
            # Ensure time coordinate matches that of physical variables
            for var_name in var_names:
                processed_fields[var_name] = processed_fields[var_name].assign_coords(
                    {"time": processed_fields["temp"]["time"]}
                )

        # Get depth coordinates
        zeta = (
            processed_fields["zeta"] if self.adjust_depth_for_sea_surface_height else 0
        )

        for location in ["rho", "u", "v"]:
            # Filter var_names by location and check for 3D variables
            filtered_vars = [
                var_name
                for var_name, info in var_names.items()
                if info["location"] == location and variable_info[var_name]["is_3d"]
            ]

            if filtered_vars:
                # Handle depth coordinates
                self._get_depth_coordinates(zeta, location, "layer")

                # Vertical regridding
                vertical_regrid = VerticalRegridToROMS(
                    self.ds_depth_coords[f"layer_depth_{location}"],
                    data.ds[data.dim_names["depth"]],
                )
                for var_name in filtered_vars:
                    if var_name in processed_fields:
                        field = processed_fields[var_name]
                        if self.use_dask:
                            field = field.chunk(
                                get_dask_chunks(location, self.horizontal_chunk_size)
                            )
                        processed_fields[var_name] = vertical_regrid.apply(field)

        # Compute barotropic velocities
        if "u" in var_names and "v" in var_names:
            for location in ["u", "v"]:
                self._get_depth_coordinates(zeta, location, "interface")
                processed_fields[f"{location}bar"] = compute_barotropic_velocity(
                    processed_fields[location],
                    self.ds_depth_coords[f"interface_depth_{location}"],
                )

        return processed_fields

    def _input_checks(self):
        # Check that ini_time is not None
        if self.ini_time is None:
            raise ValueError(
                "`ini_time` must be a valid datetime object and cannot be None."
            )

        if "name" not in self.source.keys():
            raise ValueError("`source` must include a 'name'.")
        if "path" not in self.source.keys():
            raise ValueError("`source` must include a 'path'.")
        # set self.source["climatology"] to False if not provided
        self.source = {
            **self.source,
            "climatology": self.source.get("climatology", False),
        }
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
            self.bgc_source = {
                **self.bgc_source,
                "climatology": self.bgc_source.get("climatology", False),
            }
        if self.adjust_depth_for_sea_surface_height:
            logging.info("Sea surface height will be used to adjust depth coordinates.")
        else:
            logging.info(
                "Sea surface height will NOT be used to adjust depth coordinates."
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
        elif self.bgc_source["name"] == "UNIFIED":
            data = UnifiedBGCDataset(
                filename=self.bgc_source["path"],
                start_time=self.ini_time,
                climatology=self.bgc_source["climatology"],
                use_dask=self.use_dask,
            )

        else:
            raise ValueError(
                'Only "CESM_REGRIDDED" and "UNIFIED" are valid options for bgc_source["name"].'
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

            for var_name in list(data.var_names.keys()) + list(
                data.opt_var_names.keys()
            ):
                if var_name == "ALK":
                    variable_info[var_name] = {**default_info, "validate": True}
                else:
                    variable_info[var_name] = {**default_info, "validate": False}

        object.__setattr__(self, f"variable_info_{type}", variable_info)

    def _get_depth_coordinates(
        self, zeta: xr.DataArray | float, location: str, depth_type: str = "layer"
    ) -> None:
        """Ensure depth coordinates are computed and stored for a given location and
        depth type.

        Parameters
        ----------
        zeta : xr.DataArray or float
            Free-surface elevation (can be a scalar or a DataArray).
        location : str
            Grid location for depth computation ("rho", "u", or "v").
        depth_type : str, optional
            Type of depth coordinates to compute, by default "layer".

        Notes
        ------
        Rather than calling compute_depth_coordinates from the vertical_coordinate.py module,
        this method computes the depth coordinates from scratch because of optional chunking.
        """
        key = f"{depth_type}_depth_{location}"

        if key not in self.ds_depth_coords:
            # Select the appropriate depth computation parameters
            if depth_type == "layer":
                Cs = self.grid.ds["Cs_r"]
                sigma = self.grid.ds["sigma_r"]
            elif depth_type == "interface":
                Cs = self.grid.ds["Cs_w"]
                sigma = self.grid.ds["sigma_w"]
            else:
                raise ValueError(
                    f"Invalid depth_type: {depth_type}. Choose 'layer' or 'interface'."
                )

            h = self.grid.ds["h"]

            # Interpolate h and zeta to the specified location
            if location == "u":
                h = interpolate_from_rho_to_u(h)
                if isinstance(zeta, xr.DataArray):
                    zeta = interpolate_from_rho_to_u(zeta)
            elif location == "v":
                h = interpolate_from_rho_to_v(h)
                if isinstance(zeta, xr.DataArray):
                    zeta = interpolate_from_rho_to_v(zeta)

            if self.use_dask:
                h = h.chunk(get_dask_chunks(location, self.horizontal_chunk_size))
                if self.adjust_depth_for_sea_surface_height:
                    zeta = zeta.chunk(
                        get_dask_chunks(location, self.horizontal_chunk_size)
                    )
            depth = compute_depth(zeta, h, self.grid.ds.attrs["hc"], Cs, sigma)
            self.ds_depth_coords[key] = depth

    def _write_into_dataset(self, processed_fields, d_meta):

        # save in new dataset
        ds = xr.Dataset()

        for var_name in processed_fields:
            if var_name in d_meta:

                # drop auxiliary variables
                ds[var_name] = processed_fields[var_name].astype(np.float32)
                ds[var_name].attrs["long_name"] = d_meta[var_name]["long_name"]
                ds[var_name].attrs["units"] = d_meta[var_name]["units"]

        # initialize vertical velocity to zero
        ds["w"] = xr.zeros_like(
            (self.grid.ds["Cs_w"] * self.grid.ds["h"]).expand_dims(
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
        abs_time = ds["time"]
        attrs = [key for key in abs_time.attrs]
        for attr in attrs:
            del abs_time.attrs[attr]
        abs_time.attrs["long_name"] = "absolute time"
        ds = ds.assign_coords({"abs_time": abs_time})

        # Translate the time coordinate to days since the model reference date
        model_reference_date = np.datetime64(self.model_reference_date)

        # Convert the time coordinate to the format expected by ROMS (seconds since model reference date)
        ocean_time = (ds["time"] - model_reference_date).astype("float64") * 1e-9
        ds = ds.assign_coords(ocean_time=("time", ocean_time.data.astype("float64")))
        ds["ocean_time"].attrs[
            "long_name"
        ] = f"relative time: seconds since {str(self.model_reference_date)}"
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
            if variable_info[var_name]["validate"]:
                if variable_info[var_name]["location"] == "rho":
                    mask = self.grid.ds.mask_rho
                elif variable_info[var_name]["location"] == "u":
                    mask = self.grid.ds.mask_u
                elif variable_info[var_name]["location"] == "v":
                    mask = self.grid.ds.mask_v
                ds[var_name].load()
                nan_check(ds[var_name].squeeze(), mask)

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
        ds.attrs["adjust_depth_for_sea_surface_height"] = str(
            self.adjust_depth_for_sea_surface_height
        )
        ds.attrs["source"] = self.source["name"]
        if self.bgc_source is not None:
            ds.attrs["bgc_source"] = self.bgc_source["name"]

        ds.attrs["theta_s"] = self.grid.ds.attrs["theta_s"]
        ds.attrs["theta_b"] = self.grid.ds.attrs["theta_b"]
        ds.attrs["hc"] = self.grid.ds.attrs["hc"]

        return ds

    def plot(
        self,
        var_name: str,
        s: int | None = None,
        eta: int | None = None,
        xi: int | None = None,
        depth_contours: bool = False,
        layer_contours: bool = False,
        ax: Axes | None = None,
        save_path: str | None = None,
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
            The axes to plot on. If None, a new figure is created. Note that this argument is ignored for 2D horizontal plots. Default is None.
        save_path : str, optional
            Path to save the generated plot. If None, the plot is shown interactively.
            Default is None.

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
        if var_name not in self.ds:
            raise ValueError(f"Variable '{var_name}' is not found in the dataset.")

        # Load the data
        if self.use_dask:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                self.ds[var_name].load()

        if self.adjust_depth_for_sea_surface_height:
            zeta = self.ds.zeta.squeeze().load()
        else:
            zeta = 0

        field = self.ds[var_name].squeeze()

        if var_name in ["u", "v", "w", "ubar", "vbar", "zeta"]:
            cmap_name = "RdBu_r"
        elif var_name in ["temp", "salt"]:
            cmap_name = "YlOrRd"
        else:
            cmap_name = "YlGn"

        plot(
            field=field,
            grid_ds=self.grid.ds,
            zeta=zeta,
            s=s,
            eta=eta,
            xi=xi,
            depth_contours=depth_contours,
            layer_contours=layer_contours,
            ax=ax,
            save_path=save_path,
            cmap_name=cmap_name,
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """Save the initial conditions information to one netCDF4 file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The base path or filename where the dataset should be saved.

        Returns
        -------
        Path
            A `Path` object representing the location of the saved file.
        """

        # Ensure filepath is a Path object
        filepath = Path(filepath)

        # Remove ".nc" suffix if present
        if filepath.suffix == ".nc":
            filepath = filepath.with_suffix("")

        dataset_list = [self.ds]
        output_filenames = [str(filepath)]

        saved_filenames = save_datasets(
            dataset_list, output_filenames, use_dask=self.use_dask
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

        forcing_dict = _to_dict(self)
        _write_to_yaml(forcing_dict, filepath)

    @classmethod
    def from_yaml(
        cls,
        filepath: Union[str, Path],
        use_dask: bool = False,
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
        return cls(
            grid=grid,
            **initial_conditions_params,
            use_dask=use_dask,
        )
