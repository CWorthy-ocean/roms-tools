import xarray as xr
import numpy as np
from scipy.ndimage import label
import logging
import importlib.metadata
from typing import Dict, Union, List
from dataclasses import dataclass, field
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from roms_tools import Grid
from roms_tools.regrid import LateralRegrid, VerticalRegrid
from roms_tools.vertical_coordinate import compute_depth
from roms_tools.plot import _section_plot, _line_plot
from roms_tools.utils import (
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    transpose_dimensions,
)
from roms_tools.setup.datasets import GLORYSDataset, CESMBGCDataset
from roms_tools.setup.utils import (
    get_variable_metadata,
    group_dataset,
    save_datasets,
    get_target_coords,
    rotate_velocities,
    compute_barotropic_velocity,
    one_dim_fill,
    nan_check,
    substitute_nans_by_fillvalue,
    convert_to_roms_time,
    get_boundary_coords,
    _to_yaml,
    _from_yaml,
)


@dataclass(frozen=True, kw_only=True)
class BoundaryForcing:
    """Represents boundary forcing input data for ROMS.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
    start_time : datetime
        Start time of the desired boundary forcing data.
    end_time : datetime
        End time of the desired boundary forcing data.
    boundaries : Dict[str, bool], optional
        Dictionary specifying which boundaries are forced (south, east, north, west). Default is all True.
    source : Dict[str, Union[str, Path, List[Union[str, Path]]], bool]
        Dictionary specifying the source of the boundary forcing data. Keys include:

          - "name" (str): Name of the data source (e.g., "GLORYS").
          - "path" (Union[str, Path, List[Union[str, Path]]]): The path to the raw data file(s). This can be:

            - A single string (with or without wildcards).
            - A single Path object.
            - A list of strings or Path objects containing multiple files.
          - "climatology" (bool): Indicates if the data is climatology data. Defaults to False.

    type : str
        Specifies the type of forcing data. Options are:

          - "physics": for physical atmospheric forcing.
          - "bgc": for biogeochemical forcing.

    apply_2d_horizontal_fill: bool, optional
        Indicates whether to perform a two-dimensional horizontal fill on the source data prior to regridding to boundaries.
        If `False`, a one-dimensional horizontal fill is performed separately on each of the four regridded boundaries.
        Defaults to `False`.
    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.
    bypass_validation: bool, optional
        Indicates whether to skip validation checks in the processed data. When set to True,
        the validation process that ensures no NaN values exist at wet points
        in the processed dataset is bypassed. Defaults to False.

    Examples
    --------
    >>> boundary_forcing = BoundaryForcing(
    ...     grid=grid,
    ...     boundaries={"south": True, "east": True, "north": False, "west": True},
    ...     start_time=datetime(2022, 1, 1),
    ...     end_time=datetime(2022, 1, 2),
    ...     source={"name": "GLORYS", "path": "glorys_data.nc"},
    ...     type="physics",
    ... )
    """

    grid: Grid
    start_time: datetime
    end_time: datetime
    boundaries: Dict[str, bool] = field(
        default_factory=lambda: {
            "south": True,
            "east": True,
            "north": True,
            "west": True,
        }
    )
    source: Dict[str, Union[str, Path, List[Union[str, Path]]]]
    type: str = "physics"
    apply_2d_horizontal_fill: bool = False
    model_reference_date: datetime = datetime(2000, 1, 1)
    use_dask: bool = False
    bypass_validation: bool = False

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        self._input_checks()
        target_coords = get_target_coords(self.grid)

        data = self._get_data()

        if self.apply_2d_horizontal_fill:
            data.choose_subdomain(
                target_coords,
                buffer_points=20,  # lateral fill needs good buffer from data margin
            )
            data.extrapolate_deepest_to_bottom()
            data.apply_lateral_fill()

        self._set_variable_info(data)
        self._set_boundary_info()
        ds = xr.Dataset()

        for direction in ["south", "east", "north", "west"]:
            if self.boundaries[direction]:

                bdry_target_coords = {
                    "lat": target_coords["lat"].isel(
                        **self.bdry_coords["vector"][direction]
                    ),
                    "lon": target_coords["lon"].isel(
                        **self.bdry_coords["vector"][direction]
                    ),
                    "straddle": target_coords["straddle"],
                }

                bdry_data = data.choose_subdomain(
                    bdry_target_coords,
                    buffer_points=3,
                    return_copy=True,
                )

                if not self.apply_2d_horizontal_fill:
                    bdry_data.extrapolate_deepest_to_bottom()

                processed_fields = {}

                # lateral regridding of vector fields
                vector_var_names = [
                    name
                    for name, info in self.variable_info.items()
                    if info["is_vector"]
                ]
                if len(vector_var_names) > 0:
                    lon = target_coords["lon"].isel(
                        **self.bdry_coords["vector"][direction]
                    )
                    lat = target_coords["lat"].isel(
                        **self.bdry_coords["vector"][direction]
                    )
                    lateral_regrid = LateralRegrid(
                        {"lat": lat, "lon": lon}, bdry_data.dim_names
                    )
                    for var_name in vector_var_names:
                        if var_name in bdry_data.var_names.keys():
                            processed_fields[var_name] = lateral_regrid.apply(
                                bdry_data.ds[bdry_data.var_names[var_name]]
                            )

                # lateral regridding of tracer fields
                tracer_var_names = [
                    name
                    for name, info in self.variable_info.items()
                    if not info["is_vector"]
                ]
                if len(tracer_var_names) > 0:
                    lon = target_coords["lon"].isel(
                        **self.bdry_coords["rho"][direction]
                    )
                    lat = target_coords["lat"].isel(
                        **self.bdry_coords["rho"][direction]
                    )
                    lateral_regrid = LateralRegrid(
                        {"lat": lat, "lon": lon}, bdry_data.dim_names
                    )
                    for var_name in tracer_var_names:
                        if var_name in bdry_data.var_names.keys():
                            processed_fields[var_name] = lateral_regrid.apply(
                                bdry_data.ds[bdry_data.var_names[var_name]]
                            )

                # rotation of velocities and interpolation to u/v points
                if "u" in self.variable_info and "v" in self.variable_info:
                    angle = target_coords["angle"].isel(
                        **self.bdry_coords["vector"][direction]
                    )
                    (processed_fields["u"], processed_fields["v"],) = rotate_velocities(
                        processed_fields["u"],
                        processed_fields["v"],
                        angle,
                        interpolate=True,
                    )

                # selection of outermost margin for u/v variables
                for var_name in self.variable_info.keys():
                    if var_name in processed_fields:
                        location = self.variable_info[var_name]["location"]
                        if location in ["u", "v"]:
                            processed_fields[var_name] = processed_fields[
                                var_name
                            ].isel(**self.bdry_coords[location][direction])

                if not self.apply_2d_horizontal_fill:
                    self._validate_1d_fill(
                        processed_fields,
                        direction,
                        bdry_data.dim_names["depth"],
                    )
                    processed_fields = apply_1d_horizontal_fill(processed_fields)

                var_names_dict = {}
                for location in ["rho", "u", "v"]:
                    var_names_dict[location] = [
                        name
                        for name, info in self.variable_info.items()
                        if info["location"] == location and info["is_3d"]
                    ]
                # compute layer depth coordinates
                if len(var_names_dict["u"]) > 0 or len(var_names_dict["v"]) > 0:
                    self._get_vertical_coordinates(
                        type="layer",
                        direction=direction,
                        additional_locations=["u", "v"],
                    )
                else:
                    if len(var_names_dict["rho"]) > 0:
                        self._get_vertical_coordinates(
                            type="layer", direction=direction, additional_locations=[]
                        )

                # vertical regridding
                for location in ["rho", "u", "v"]:
                    if len(var_names_dict[location]) > 0:
                        vertical_regrid = VerticalRegrid(
                            self.grid.ds[f"layer_depth_{location}_{direction}"],
                            bdry_data.ds[bdry_data.dim_names["depth"]],
                        )
                        for var_name in var_names_dict[location]:
                            if var_name in processed_fields:
                                processed_fields[var_name] = vertical_regrid.apply(
                                    processed_fields[var_name]
                                )

                # compute barotropic velocities
                if "u" in self.variable_info and "v" in self.variable_info:
                    self._get_vertical_coordinates(
                        type="interface",
                        direction=direction,
                        additional_locations=["u", "v"],
                    )
                    for location in ["u", "v"]:
                        processed_fields[
                            f"{location}bar"
                        ] = compute_barotropic_velocity(
                            processed_fields[location],
                            self.grid.ds[f"interface_depth_{location}_{direction}"],
                        )

                # Reorder dimensions
                for var_name in processed_fields.keys():
                    processed_fields[var_name] = transpose_dimensions(
                        processed_fields[var_name]
                    )

                # Write the boundary data into dataset
                ds = self._write_into_dataset(direction, processed_fields, ds)

        # Add global information
        ds = self._add_global_metadata(data, ds)

        if not self.bypass_validation:
            self._validate(ds)

        # substitute NaNs over land by a fill value to avoid blow-up of ROMS
        for var_name in ds.data_vars:
            ds[var_name] = substitute_nans_by_fillvalue(ds[var_name])

        object.__setattr__(self, "ds", ds)

    def _input_checks(self):
        # Validate the 'type' parameter
        if self.type not in ["physics", "bgc"]:
            raise ValueError("`type` must be either 'physics' or 'bgc'.")

        # Ensure 'source' dictionary contains required keys
        if "name" not in self.source:
            raise ValueError("`source` must include a 'name'.")
        if "path" not in self.source:
            raise ValueError("`source` must include a 'path'.")

        # Set 'climatology' to False if not provided in 'source'
        object.__setattr__(
            self,
            "source",
            {**self.source, "climatology": self.source.get("climatology", False)},
        )

    def _get_data(self):

        data_dict = {
            "filename": self.source["path"],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "climatology": self.source["climatology"],
            "use_dask": self.use_dask,
        }

        if self.type == "physics":
            if self.source["name"] == "GLORYS":
                data = GLORYSDataset(**data_dict)
            else:
                raise ValueError(
                    'Only "GLORYS" is a valid option for source["name"] when type is "physics".'
                )

        elif self.type == "bgc":
            if self.source["name"] == "CESM_REGRIDDED":

                data = CESMBGCDataset(**data_dict)
            else:
                raise ValueError(
                    'Only "CESM_REGRIDDED" is a valid option for source["name"] when type is "bgc".'
                )

        return data

    def _set_variable_info(self, data):
        """Sets up a dictionary with metadata for variables based on the type of data
        (physics or BGC).

        The dictionary contains the following information:
        - `location`: Where the variable resides in the grid (e.g., rho, u, or v points).
        - `is_vector`: Whether the variable is part of a vector (True for velocity components like 'u' and 'v').
        - `vector_pair`: For vector variables, this indicates the associated variable that forms the vector (e.g., 'u' and 'v').
        - `is_3d`: Indicates whether the variable is 3D (True for variables like 'temp' and 'salt') or 2D (False for 'zeta').

        Parameters
        ----------
        data : object
            An object that contains variable names for the data being processed. This is used to set variable information for biogeochemical data.

        Returns
        -------
        None
            This method updates the instance attribute `variable_info` with the metadata dictionary for the variables.
        """
        default_info = {
            "location": "rho",
            "is_vector": False,
            "vector_pair": None,
            "is_3d": True,
        }

        # Define a dictionary for variable names and their associated information
        if self.type == "physics":
            variable_info = {
                "zeta": {
                    "location": "rho",
                    "is_vector": False,
                    "vector_pair": None,
                    "is_3d": False,
                    "validate": True,
                },
                "temp": {**default_info, "validate": True},
                "salt": {**default_info, "validate": False},
                "u": {
                    "location": "u",
                    "is_vector": True,
                    "vector_pair": "v",
                    "is_3d": True,
                    "validate": True,
                },
                "v": {
                    "location": "v",
                    "is_vector": True,
                    "vector_pair": "u",
                    "is_3d": True,
                    "validate": True,
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
            }
        elif self.type == "bgc":
            variable_info = {}
            for var_name in data.var_names.keys():
                if var_name == "ALK":
                    variable_info[var_name] = {**default_info, "validate": True}
                else:
                    variable_info[var_name] = {**default_info, "validate": False}

        object.__setattr__(self, "variable_info", variable_info)

    def _write_into_dataset(self, direction, processed_fields, ds=None):
        if ds is None:
            ds = xr.Dataset()

        d_meta = get_variable_metadata()

        for var_name in processed_fields.keys():
            ds[f"{var_name}_{direction}"] = processed_fields[var_name].astype(
                np.float32
            )

            ds[f"{var_name}_{direction}"].attrs[
                "long_name"
            ] = f"{direction}ern boundary {d_meta[var_name]['long_name']}"

            ds[f"{var_name}_{direction}"].attrs["units"] = d_meta[var_name]["units"]

        # Gracefully handle dropping variables that might not be present
        variables_to_drop = [
            "s_rho",
            "layer_depth_rho",
            "layer_depth_u",
            "layer_depth_v",
            "interface_depth_rho",
            "interface_depth_u",
            "interface_depth_v",
            "lat_rho",
            "lon_rho",
            "lat_u",
            "lon_u",
            "lat_v",
            "lon_v",
        ]
        suffixes = ["", "_south", "_east", "_north", "_west"]
        # Existing variables with suffixes
        existing_vars = []
        for var_name in variables_to_drop:
            for suffix in suffixes:
                full_var_name = f"{var_name}{suffix}"
                if full_var_name in ds:
                    existing_vars.append(full_var_name)

        ds = ds.drop_vars(existing_vars)

        return ds

    def _set_boundary_info(self):
        """Sets boundary coordinates for rho, u, and v variables on the grid.

        This method determines the boundary points for the grid variables by specifying the
        indices for the south, east, north, and west boundaries. The resulting boundary
        information is stored in the instance attribute `bdry_coords`.

        Returns
        -------
        None
            The method does not return a value. Instead, it updates the instance attribute
            `bdry_coords`, which is a dictionary structured as follows:
            - Keys: Variable types ("rho", "u", "v", "vector").
            - Values: Nested dictionaries mapping each direction ("south", "east", "north", "west")
              to their corresponding boundary coordinates. The coordinates are specified in terms of
              grid indices for the respective variable types.
        """

        bdry_coords = get_boundary_coords()

        object.__setattr__(self, "bdry_coords", bdry_coords)

    def _get_vertical_coordinates(
        self, type, direction, additional_locations=["u", "v"]
    ):
        """Retrieve layer and interface depth coordinates for a specified grid boundary.

        This method computes and updates the layer and interface depth coordinates along a specified
        boundary (north, south, east, or west). It handles depth calculations for rho points and
        additional specified locations (u and v).

        Parameters
        ----------
        type : str
            The type of depth coordinate to retrieve. Valid options are:
            - "layer": Retrieves layer depth coordinates.
            - "interface": Retrieves interface depth coordinates.

        direction : str
            The direction of the boundary to retrieve coordinates for. Valid options are:
            - "north"
            - "south"
            - "east"
            - "west"

        additional_locations : list of str, optional
            Specifies additional locations to compute depth coordinates for. Default is ["u", "v"].
            Valid options include:
            - "u": Computes depth coordinates for u points.
            - "v": Computes depth coordinates for v points.

        Updates
        -------
        self.grid.ds : xarray.Dataset
            The dataset is updated with the following vertical depth coordinates:
            - f"{type}_depth_rho_{direction}": Depth coordinates at rho points.
            - f"{type}_depth_u_{direction}": Depth coordinates at u points (if applicable).
            - f"{type}_depth_v_{direction}": Depth coordinates at v points (if applicable).
        """

        layer_vars = []
        for location in ["rho"] + additional_locations:
            layer_vars.append(f"{type}_depth_{location}_{direction}")

        if all(layer_var in self.grid.ds for layer_var in layer_vars):
            # Vertical coordinate data already exists
            pass

        elif f"{type}_depth_rho" in self.grid.ds:
            depth = self.grid.ds[f"{type}_depth_rho"]
            depth.attrs["long_name"] = f"{type} depth at rho-points"
            depth.attrs["units"] = "m"
            self.grid.ds[f"{type}_depth_rho_{direction}"] = depth.isel(
                **self.bdry_coords["rho"][direction]
            )

            if "u" in additional_locations or "v" in additional_locations:
                # selection of margin consisting of 2 grid cells
                depth = depth.isel(**self.bdry_coords["vector"][direction])
                # interpolation
                if "u" in additional_locations:
                    depth_u = interpolate_from_rho_to_u(depth)
                    depth_u.attrs["long_name"] = f"{type} depth at u-points"
                    depth_u.attrs["units"] = "m"
                    self.grid.ds[f"{type}_depth_u_{direction}"] = depth_u.isel(
                        **self.bdry_coords["u"][direction]
                    )
                if "v" in additional_locations:
                    depth_v = interpolate_from_rho_to_v(depth)
                    depth_v.attrs["long_name"] = f"{type} depth at v-points"
                    depth_v.attrs["units"] = "m"
                    self.grid.ds[f"{type}_depth_v_{direction}"] = depth_v.isel(
                        **self.bdry_coords["v"][direction]
                    )
        else:
            if "u" in additional_locations or "v" in additional_locations:
                h = self.grid.ds["h"].isel(**self.bdry_coords["vector"][direction])
            else:
                h = self.grid.ds["h"].isel(**self.bdry_coords["rho"][direction])
            if type == "layer":
                depth = compute_depth(
                    0, h, self.grid.hc, self.grid.ds.Cs_r, self.grid.ds.sigma_r
                )
            else:
                depth = compute_depth(
                    0, h, self.grid.hc, self.grid.ds.Cs_w, self.grid.ds.sigma_w
                )

            if "u" in additional_locations or "v" in additional_locations:
                depth.attrs["long_name"] = f"{type} depth at rho-points"
                depth.attrs["units"] = "m"
                self.grid.ds[f"{type}_depth_rho_{direction}"] = depth.isel(
                    **self.bdry_coords["rho"][direction]
                )
                # selection of margin consisting of 2 grid cells
                depth = depth.isel(**self.bdry_coords["vector"][direction])
                # interpolation
                depth_u = interpolate_from_rho_to_u(depth)
                depth_v = interpolate_from_rho_to_v(depth)
                # selection of outermost margin
                depth_u.attrs["long_name"] = f"{type} depth at u-points"
                depth_u.attrs["units"] = "m"
                self.grid.ds[f"{type}_depth_u_{direction}"] = depth_u.isel(
                    **self.bdry_coords["u"][direction]
                )
                depth_v.attrs["long_name"] = f"{type} depth at v-points"
                depth_v.attrs["units"] = "m"
                self.grid.ds[f"{type}_depth_v_{direction}"] = depth_v.isel(
                    **self.bdry_coords["v"][direction]
                )
            else:
                depth.attrs["long_name"] = f"{type} depth at rho-points"
                depth.attrs["units"] = "m"
                self.grid.ds[f"{type}_depth_rho_{direction}"] = depth

    def _add_global_metadata(self, data, ds=None):

        if ds is None:
            ds = xr.Dataset()
        ds.attrs["title"] = "ROMS boundary forcing file created by ROMS-Tools"
        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"
        ds.attrs["roms_tools_version"] = roms_tools_version
        ds.attrs["start_time"] = str(self.start_time)
        ds.attrs["end_time"] = str(self.end_time)
        ds.attrs["source"] = self.source["name"]
        ds.attrs["model_reference_date"] = str(self.model_reference_date)

        ds.attrs["theta_s"] = self.grid.ds.attrs["theta_s"]
        ds.attrs["theta_b"] = self.grid.ds.attrs["theta_b"]
        ds.attrs["hc"] = self.grid.ds.attrs["hc"]

        # Convert the time coordinate to the format expected by ROMS
        ds, bry_time = convert_to_roms_time(
            ds, self.model_reference_date, data.climatology
        )

        ds = ds.assign_coords({"bry_time": bry_time})
        ds = ds.swap_dims({"time": "bry_time"})
        ds = ds.drop_vars("time")

        return ds

    def _validate_1d_fill(self, processed_fields, direction, depth_dim):
        """Check if any boundary is divided by land and issue a warning if so,
        suggesting the use of 2D horizontal fill for safer regridding.

        Parameters
        ----------
        processed_fields : dict
            A dictionary where keys are variable names and values are `xarray.DataArray`
            objects representing the processed data for each variable.

        direction : str
            The boundary direction being processed (e.g., "north", "south", "east", or "west").

        depth_dim : str
            The dimension representing depth (e.g., 'z', 'depth', etc.), used when slicing 3D
            data for a specific depth level.

        Returns
        -------
        None
            If a boundary is divided by land, a warning is issued. No return value is provided.
        """

        for var_name in processed_fields.keys():
            # Only validate variables based on "validate" flag if use_dask is False
            if not self.use_dask or self.variable_info[var_name]["validate"]:
                location = self.variable_info[var_name]["location"]

                # Select the appropriate mask based on variable location
                if location == "rho":
                    mask = self.grid.ds.mask_rho
                elif location == "u":
                    mask = self.grid.ds.mask_u
                elif location == "v":
                    mask = self.grid.ds.mask_v

                mask = mask.isel(**self.bdry_coords[location][direction])

                if self.variable_info[var_name]["is_3d"]:
                    da = processed_fields[var_name].isel({depth_dim: 0, "time": 0})
                else:
                    da = processed_fields[var_name].isel({"time": 0})

                wet_nans = xr.where(da.where(mask).isnull(), 1, 0)
                # Apply label to find connected components of wet NaNs
                labeled_array, num_features = label(wet_nans)
                left_margin = labeled_array[0]
                right_margin = labeled_array[-1]
                if left_margin != 0:
                    num_features = num_features - 1
                if right_margin != 0:
                    num_features = num_features - 1
                if num_features > 0:
                    logging.warning(
                        f"For {var_name}, the {direction}ern boundary is divided by land. It would be safer (but slower) to use `apply_2d_horizontal_fill = True`."
                    )

    def _validate(self, ds):
        """Validate the dataset for NaN values at the first time step (bry_time=0) for
        specified variables. If NaN values are found at wet points, this function raises
        an error.

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
        Validation is performed on the initial boundary time step (`bry_time=0`) for each
        variable in the dataset.
        """
        for var_name in self.variable_info:
            if self.variable_info[var_name]["validate"]:
                location = self.variable_info[var_name]["location"]

                # Select the appropriate mask based on variable location
                if location == "rho":
                    mask = self.grid.ds.mask_rho
                elif location == "u":
                    mask = self.grid.ds.mask_u
                elif location == "v":
                    mask = self.grid.ds.mask_v

                for direction in ["south", "east", "north", "west"]:
                    if self.boundaries[direction]:
                        bdry_var_name = f"{var_name}_{direction}"

                        # Check for NaN values at the first time step using the nan_check function
                        if self.apply_2d_horizontal_fill:
                            error_message = None
                        else:
                            error_message = (
                                f"{bdry_var_name} consists entirely of NaNs after regridding. "
                                f"This may be due to the {direction}ern boundary being on land in the "
                                f"{self.source['name']} data, which could have a coarser resolution than the ROMS domain. "
                                f"Try setting `apply_2d_horizontal_fill = True` to resolve this issue."
                            )

                        nan_check(
                            ds[bdry_var_name].isel(bry_time=0),
                            mask.isel(**self.bdry_coords[location][direction]),
                            error_message=error_message,
                        )

    def plot(self, var_name, time=0, layer_contours=False, ax=None) -> None:
        """Plot the boundary forcing field for a given time-slice.

        Parameters
        ----------
        var_name : str
            The name of the boundary forcing field to plot. Options include:

            - "temp_{direction}": Potential temperature,
            - "salt_{direction}": Salinity,
            - "zeta_{direction}": Sea surface height,
            - "u_{direction}": u-flux component,
            - "v_{direction}": v-flux component,
            - "ubar_{direction}": Vertically integrated u-flux component,
            - "vbar_{direction}": Vertically integrated v-flux component,
            - "PO4_{direction}": Dissolved Inorganic Phosphate (mmol/m³),
            - "NO3_{direction}": Dissolved Inorganic Nitrate (mmol/m³),
            - "SiO3_{direction}": Dissolved Inorganic Silicate (mmol/m³),
            - "NH4_{direction}": Dissolved Ammonia (mmol/m³),
            - "Fe_{direction}": Dissolved Inorganic Iron (mmol/m³),
            - "Lig_{direction}": Iron Binding Ligand (mmol/m³),
            - "O2_{direction}": Dissolved Oxygen (mmol/m³),
            - "DIC_{direction}": Dissolved Inorganic Carbon (mmol/m³),
            - "DIC_ALT_CO2_{direction}": Dissolved Inorganic Carbon, Alternative CO2 (mmol/m³),
            - "ALK_{direction}": Alkalinity (meq/m³),
            - "ALK_ALT_CO2_{direction}": Alkalinity, Alternative CO2 (meq/m³),
            - "DOC_{direction}": Dissolved Organic Carbon (mmol/m³),
            - "DON_{direction}": Dissolved Organic Nitrogen (mmol/m³),
            - "DOP_{direction}": Dissolved Organic Phosphorus (mmol/m³),
            - "DOPr_{direction}": Refractory Dissolved Organic Phosphorus (mmol/m³),
            - "DONr_{direction}": Refractory Dissolved Organic Nitrogen (mmol/m³),
            - "DOCr_{direction}": Refractory Dissolved Organic Carbon (mmol/m³),
            - "zooC_{direction}": Zooplankton Carbon (mmol/m³),
            - "spChl_{direction}": Small Phytoplankton Chlorophyll (mg/m³),
            - "spC_{direction}": Small Phytoplankton Carbon (mmol/m³),
            - "spP_{direction}": Small Phytoplankton Phosphorous (mmol/m³),
            - "spFe_{direction}": Small Phytoplankton Iron (mmol/m³),
            - "spCaCO3_{direction}": Small Phytoplankton CaCO3 (mmol/m³),
            - "diatChl_{direction}": Diatom Chlorophyll (mg/m³),
            - "diatC_{direction}": Diatom Carbon (mmol/m³),
            - "diatP_{direction}": Diatom Phosphorus (mmol/m³),
            - "diatFe_{direction}": Diatom Iron (mmol/m³),
            - "diatSi_{direction}": Diatom Silicate (mmol/m³),
            - "diazChl_{direction}": Diazotroph Chlorophyll (mg/m³),
            - "diazC_{direction}": Diazotroph Carbon (mmol/m³),
            - "diazP_{direction}": Diazotroph Phosphorus (mmol/m³),
            - "diazFe_{direction}": Diazotroph Iron (mmol/m³),

            where {direction} can be one of ["south", "east", "north", "west"].

        time : int, optional
            The time index to plot. Default is 0.
        layer_contours : bool, optional
            If True, contour lines representing the boundaries between vertical layers will
            be added to the plot. For clarity, the number of layer
            contours displayed is limited to a maximum of 10. Default is False.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure is created.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        Raises
        ------
        ValueError
            If the specified var_name is not one of the valid options.
        """

        if var_name not in self.ds:
            raise ValueError(f"Variable '{var_name}' is not found in dataset.")

        field = self.ds[var_name].isel(bry_time=time)

        if self.use_dask:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                field = field.load()

        title = field.long_name
        var_name_wo_direction, direction = var_name.split("_")
        location = self.variable_info[var_name_wo_direction]["location"]

        # Find correct mask
        if location == "rho":
            mask = self.grid.ds.mask_rho
        elif location == "u":
            mask = self.grid.ds.mask_u
        elif location == "v":
            mask = self.grid.ds.mask_v

        mask = mask.isel(**self.bdry_coords[location][direction])

        if "s_rho" in field.dims:
            field = field.assign_coords(
                {"layer_depth": self.grid.ds[f"layer_depth_{location}_{direction}"]}
            )
        if var_name.startswith(("u", "v", "ubar", "vbar", "zeta")):
            vmax = max(field.max().values, -field.min().values)
            vmin = -vmax
            cmap = plt.colormaps.get_cmap("RdBu_r")
        else:
            vmax = field.max().values
            vmin = field.min().values
            if var_name.startswith(("temp", "salt")):
                cmap = plt.colormaps.get_cmap("YlOrRd")
            else:
                cmap = plt.colormaps.get_cmap("YlGn")
        cmap.set_bad(color="gray")
        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

        if len(field.dims) == 2:
            if layer_contours:
                if location in ["u", "v"]:
                    additional_locations = ["u", "v"]
                else:
                    additional_locations = []
                self._get_vertical_coordinates(
                    type="interface",
                    direction=direction,
                    additional_locations=additional_locations,
                )

                interface_depth = self.grid.ds[
                    f"interface_depth_{location}_{direction}"
                ]
                # restrict number of layer_contours to 10 for the sake of plot clearity
                nr_layers = len(interface_depth["s_w"])
                selected_layers = np.linspace(
                    0, nr_layers - 1, min(nr_layers, 10), dtype=int
                )
                interface_depth = interface_depth.isel(s_w=selected_layers)

            else:
                interface_depth = None

            _section_plot(
                field.where(mask),
                interface_depth=interface_depth,
                title=title,
                kwargs=kwargs,
                ax=ax,
            )
        else:
            _line_plot(field.where(mask), title=title, ax=ax)

    def save(
        self,
        filepath: Union[str, Path],
        group: bool = False,
    ) -> None:
        """Save the boundary forcing fields to one or more netCDF4 files.

        This method saves the dataset to disk as either a single netCDF4 file or multiple files, depending on the `group` parameter.
        If `group` is `True`, the dataset is divided into subsets (e.g., monthly or yearly) based on the temporal frequency
        of the data, and each subset is saved to a separate file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The base path and filename for the output file(s). If `group` is `True`, the filenames will include additional
            time-based information (e.g., year or month) to distinguish the subsets.
        group : bool, optional
            Whether to divide the dataset into multiple files based on temporal frequency. Defaults to `False`, meaning the
            dataset is saved as a single file.

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

        if group:
            dataset_list, output_filenames = group_dataset(self.ds, str(filepath))
        else:
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

        _to_yaml(self, filepath)

    @classmethod
    def from_yaml(
        cls,
        filepath: Union[str, Path],
        use_dask: bool = False,
        bypass_validation: bool = False,
    ) -> "BoundaryForcing":
        """Create an instance of the BoundaryForcing class from a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file from which the parameters will be read.
        use_dask: bool, optional
            Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.
        bypass_validation: bool, optional
            Indicates whether to skip validation checks in the processed data. When set to True,
            the validation process that ensures no NaN values exist at wet points
            in the processed dataset is bypassed. Defaults to False.

        Returns
        -------
        BoundaryForcing
            An instance of the BoundaryForcing class.
        """
        filepath = Path(filepath)

        grid = Grid.from_yaml(filepath)
        params = _from_yaml(cls, filepath)

        # Create and return an instance of InitialConditions
        return cls(
            grid=grid, **params, use_dask=use_dask, bypass_validation=bypass_validation
        )


def apply_1d_horizontal_fill(processed_fields: dict) -> dict:
    """Forward and backward fill NaN values in horizontal direction for open boundaries.

    Parameters
    ----------
    processed_fields : dict
        A dictionary of variables to be updated, where each value is an
        `xarray.DataArray`.

    Returns
    -------
    dict of str : xarray.DataArray
        The updated dictionary of variables, with NaN values filled.

    Raises
    ------
    ValueError
        If more than one horizontal dimension is found or none at all.
    """

    horizontal_dims = ["eta_rho", "eta_v", "xi_rho", "xi_u"]

    for var_name in processed_fields.keys():
        selected_horizontal_dim = None
        # Determine the horizontal dimension to fill
        for dim in horizontal_dims:
            if dim in processed_fields[var_name].dims:
                if selected_horizontal_dim is not None:
                    raise ValueError(
                        f"More than one horizontal dimension found in variable '{var_name}'."
                    )
                selected_horizontal_dim = dim

        if selected_horizontal_dim is None:
            raise ValueError(
                f"No valid horizontal dimension found for variable '{var_name}'."
            )

        # Forward and backward fill in the horizontal direction
        filled = one_dim_fill(
            processed_fields[var_name], selected_horizontal_dim, direction="forward"
        )
        processed_fields[var_name] = one_dim_fill(
            filled, selected_horizontal_dim, direction="backward"
        )

    return processed_fields
