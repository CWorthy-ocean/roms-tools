import importlib.metadata
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import label

from roms_tools import Grid
from roms_tools.plot import _line_plot, _section_plot
from roms_tools.regrid import LateralRegridToROMS, VerticalRegridToROMS
from roms_tools.setup.datasets import CESMBGCDataset, GLORYSDataset, UnifiedBGCDataset
from roms_tools.setup.utils import (
    add_time_info_to_ds,
    compute_barotropic_velocity,
    compute_missing_bgc_variables,
    from_yaml,
    get_boundary_coords,
    get_target_coords,
    get_variable_metadata,
    group_dataset,
    nan_check,
    one_dim_fill,
    rotate_velocities,
    substitute_nans_by_fillvalue,
    to_dict,
    write_to_yaml,
)
from roms_tools.utils import (
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    save_datasets,
    transpose_dimensions,
)
from roms_tools.vertical_coordinate import compute_depth


@dataclass(kw_only=True)
class BoundaryForcing:
    """Represents boundary forcing input data for ROMS.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
    start_time : datetime, optional
        The start time of the desired surface forcing data. This time is used to filter the dataset
        to include only records on or after this time, with a single record at or before this time.
        If no time filtering is desired, set it to None. Default is None.
    end_time : datetime, optional
        The end time of the desired surface forcing data. This time is used to filter the dataset
        to include only records on or before this time, with a single record at or after this time.
        If no time filtering is desired, set it to None. Default is None.
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

    apply_2d_horizontal_fill : bool, optional
        Indicates whether to perform a two-dimensional horizontal fill on the source data prior to regridding to boundaries.
        If `False`, a one-dimensional horizontal fill is performed separately on each of the four regridded boundaries.
        Defaults to `False`.
    adjust_depth_for_sea_surface_height : bool, optional
        Whether to account for sea surface height (`zeta`) variations when computing depth coordinates.
        This adjustment is only applicable for `type="physics"`, as for biogeochemical fields usually `zeta` is not available.
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
    """Object representing the grid information."""
    start_time: datetime | None = None
    """The start time of the desired surface forcing data."""
    end_time: datetime | None = None
    """The end time of the desired surface forcing data."""
    boundaries: dict[str, bool] = field(
        default_factory=lambda: {
            "south": True,
            "east": True,
            "north": True,
            "west": True,
        }
    )
    """Dictionary specifying which boundaries are forced (south, east, north, west)."""
    source: dict[str, str | Path | list[str | Path]]
    """Dictionary specifying the source of the boundary forcing data."""
    type: str = "physics"
    """Specifies the type of forcing data ("physics", "bgc")."""
    apply_2d_horizontal_fill: bool = False
    """Whether to perform a two-dimensional horizontal fill on the source data prior to
    regridding to boundaries."""
    adjust_depth_for_sea_surface_height: bool = False
    """Whether to account for sea surface height (`zeta`) variations when computing
    depth coordinates."""
    model_reference_date: datetime = datetime(2000, 1, 1)
    """Reference date for the model."""
    use_dask: bool = False
    """Whether to use dask for processing."""
    bypass_validation: bool = False
    """Whether to skip validation checks in the processed data."""

    ds: xr.Dataset = field(init=False, repr=False)
    """An xarray Dataset containing post-processed variables ready for input into
    ROMS."""

    def __post_init__(self):

        self._input_checks()
        # Dataset for depth coordinates
        self.ds_depth_coords = xr.Dataset()

        target_coords = get_target_coords(self.grid)

        data = self._get_data()

        if self.apply_2d_horizontal_fill:

            data.choose_subdomain(
                target_coords,
                buffer_points=20,  # lateral fill needs good buffer from data margin
            )
            # Enforce double precision to ensure reproducibility
            data.convert_to_float64()
            data.extrapolate_deepest_to_bottom()
            data.apply_lateral_fill()

        self._set_variable_info(data)
        self._set_boundary_info()
        ds = xr.Dataset()

        var_names = {
            var: {
                "name": data.var_names[var],
                "location": self.variable_info[var]["location"],
            }
            for var in data.var_names.keys()
            if data.var_names[var] in data.ds.data_vars
        }
        # Update the dictionary with optional variables and their locations
        var_names.update(
            {
                var: {
                    "name": data.opt_var_names[var],
                    "location": self.variable_info[var]["location"],
                }
                for var in data.opt_var_names.keys()
                if data.opt_var_names[var] in data.ds.data_vars
            }
        )

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
                    # Enforce double precision to ensure reproducibility
                    bdry_data.convert_to_float64()
                    bdry_data.extrapolate_deepest_to_bottom()

                processed_fields = {}

                # Filter var_names by vector fields
                filtered_vars = [
                    var_name
                    for var_name, info in var_names.items()
                    if self.variable_info[var_name]["is_vector"]
                ]

                # lateral regridding of vector fields
                if filtered_vars:
                    lon = target_coords["lon"].isel(
                        **self.bdry_coords["vector"][direction]
                    )
                    lat = target_coords["lat"].isel(
                        **self.bdry_coords["vector"][direction]
                    )
                    lateral_regrid = LateralRegridToROMS(
                        {"lat": lat, "lon": lon}, bdry_data.dim_names
                    )
                    for var_name in filtered_vars:
                        processed_fields[var_name] = lateral_regrid.apply(
                            bdry_data.ds[var_names[var_name]["name"]]
                        )

                    if self.adjust_depth_for_sea_surface_height:
                        # Regrid sea surface height ('zeta') onto a 2-cell-wide margin.
                        # This is needed to correctly infer depth coordinates at u- and v-points along the boundary.
                        zeta_vector = lateral_regrid.apply(
                            bdry_data.ds[var_names["zeta"]["name"]]
                        )

                # Filter var_names by tracer fields
                filtered_vars = [
                    var_name
                    for var_name, info in var_names.items()
                    if not self.variable_info[var_name]["is_vector"]
                ]

                # lateral regridding of tracer fields
                if filtered_vars:
                    lon = target_coords["lon"].isel(
                        **self.bdry_coords["rho"][direction]
                    )
                    lat = target_coords["lat"].isel(
                        **self.bdry_coords["rho"][direction]
                    )
                    lateral_regrid = LateralRegridToROMS(
                        {"lat": lat, "lon": lon}, bdry_data.dim_names
                    )
                    for var_name in filtered_vars:
                        processed_fields[var_name] = lateral_regrid.apply(
                            bdry_data.ds[var_names[var_name]["name"]]
                        )

                # rotation of velocities and interpolation to u/v points
                if "u" in processed_fields and "v" in processed_fields:
                    angle = target_coords["angle"].isel(
                        **self.bdry_coords["vector"][direction]
                    )
                    (processed_fields["u"], processed_fields["v"],) = rotate_velocities(
                        processed_fields["u"],
                        processed_fields["v"],
                        angle,
                        interpolate=True,
                    )
                    if self.adjust_depth_for_sea_surface_height:
                        zeta_u = interpolate_from_rho_to_u(zeta_vector)
                        zeta_v = interpolate_from_rho_to_v(zeta_vector)

                # selection of outermost margin for u/v variables
                for var_name in processed_fields:
                    location = self.variable_info[var_name]["location"]
                    if location in ["u", "v"]:
                        processed_fields[var_name] = processed_fields[var_name].isel(
                            **self.bdry_coords[location][direction]
                        )

                if self.adjust_depth_for_sea_surface_height:
                    zeta_u = zeta_u.isel(**self.bdry_coords["u"][direction])
                    zeta_v = zeta_v.isel(**self.bdry_coords["v"][direction])

                if not self.apply_2d_horizontal_fill and bdry_data.needs_lateral_fill:
                    logging.info(
                        f"Applying 1D horizontal fill to {direction}ern boundary."
                    )
                    self._validate_1d_fill(
                        processed_fields,
                        direction,
                        bdry_data.dim_names["depth"],
                    )
                    for var_name in processed_fields:
                        processed_fields[var_name] = apply_1d_horizontal_fill(
                            processed_fields[var_name]
                        )
                    if self.adjust_depth_for_sea_surface_height:
                        zeta_u = apply_1d_horizontal_fill(zeta_u)
                        zeta_v = apply_1d_horizontal_fill(zeta_v)

                if self.adjust_depth_for_sea_surface_height:
                    zeta = processed_fields["zeta"]
                else:
                    zeta = 0
                    zeta_u = 0
                    zeta_v = 0

                for location in ["rho", "u", "v"]:
                    # Filter var_names by location and check for 3D variables
                    filtered_vars = [
                        var_name
                        for var_name, info in var_names.items()
                        if info["location"] == location
                        and self.variable_info[var_name]["is_3d"]
                    ]

                    if filtered_vars:
                        # compute layer depth coordinates
                        if location == "rho":
                            self._get_depth_coordinates(zeta, direction, "rho", "layer")
                            self._get_depth_coordinates(
                                zeta, direction, "rho", "interface"
                            )  # only necessary for plotting
                        else:
                            self._get_depth_coordinates(zeta_u, direction, "u", "layer")
                            self._get_depth_coordinates(zeta_v, direction, "v", "layer")

                        # vertical regridding
                        vertical_regrid = VerticalRegridToROMS(
                            self.ds_depth_coords[f"layer_depth_{location}_{direction}"],
                            bdry_data.ds[bdry_data.dim_names["depth"]],
                        )
                        for var_name in filtered_vars:
                            if var_name in processed_fields:
                                processed_fields[var_name] = vertical_regrid.apply(
                                    processed_fields[var_name]
                                )

                # compute barotropic velocities
                if "u" in var_names and "v" in var_names:
                    self._get_depth_coordinates(zeta_u, direction, "u", "interface")
                    self._get_depth_coordinates(zeta_v, direction, "v", "interface")
                    for location in ["u", "v"]:
                        processed_fields[
                            f"{location}bar"
                        ] = compute_barotropic_velocity(
                            processed_fields[location],
                            self.ds_depth_coords[
                                f"interface_depth_{location}_{direction}"
                            ],
                        )

                # Reorder dimensions
                for var_name in processed_fields:
                    processed_fields[var_name] = transpose_dimensions(
                        processed_fields[var_name]
                    )

                if self.type == "bgc":
                    processed_fields = compute_missing_bgc_variables(processed_fields)

                # Write the boundary data into dataset
                ds = self._write_into_dataset(direction, processed_fields, ds)

        # Add global information
        ds = self._add_global_metadata(data, ds)

        if not self.bypass_validation:
            self._validate(ds)

        # substitute NaNs over land by a fill value to avoid blow-up of ROMS
        for var_name in ds.data_vars:
            ds[var_name] = substitute_nans_by_fillvalue(ds[var_name])

        self.ds = ds

    def _input_checks(self):
        # Check that start_time and end_time are both None or none of them is
        if (self.start_time is None) != (self.end_time is None):
            raise ValueError(
                "Both `start_time` and `end_time` must be provided together as datetime objects or both should be None."
            )

        # Trigger a warning if both are None
        if self.start_time is None and self.end_time is None:
            logging.warning(
                "Both `start_time` and `end_time` are None. No time filtering will be applied to the source data."
            )

        # Validate the 'type' parameter
        if self.type not in ["physics", "bgc"]:
            raise ValueError("`type` must be either 'physics' or 'bgc'.")

        # Ensure 'source' dictionary contains required keys
        if "name" not in self.source:
            raise ValueError("`source` must include a 'name'.")
        if "path" not in self.source:
            raise ValueError("`source` must include a 'path'.")

        # Set 'climatology' to False if not provided in 'source'
        self.source = {
            **self.source,
            "climatology": self.source.get("climatology", False),
        }

        # Ensure adjust_depth_for_sea_surface_height is only used with type="physics"
        if self.type == "bgc" and self.adjust_depth_for_sea_surface_height:
            logging.warning(
                "adjust_depth_for_sea_surface_height is not applicable for BGC fields. "
                "Setting it to False."
            )
            self.adjust_depth_for_sea_surface_height = False
        elif self.adjust_depth_for_sea_surface_height:
            logging.info("Sea surface height will be used to adjust depth coordinates.")
        else:
            logging.info(
                "Sea surface height will NOT be used to adjust depth coordinates."
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
            elif self.source["name"] == "UNIFIED":
                data = UnifiedBGCDataset(**data_dict)
            else:
                raise ValueError(
                    'Only "CESM_REGRIDDED" and "UNIFIED" are valid options for source["name"] when type is "bgc".'
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
            for var_name in list(data.var_names.keys()) + list(
                data.opt_var_names.keys()
            ):
                if var_name == "ALK":
                    variable_info[var_name] = {**default_info, "validate": True}
                else:
                    variable_info[var_name] = {**default_info, "validate": False}

        self.variable_info = variable_info

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

        self.bdry_coords = bdry_coords

    def _get_depth_coordinates(
        self,
        zeta: xr.DataArray | float,
        direction: str,
        location: str,
        depth_type: str = "layer",
    ) -> None:
        """Compute and store depth coordinates for a specified boundary direction, grid
        location, and depth type.

        This method efficiently computes depth coordinates along the specified boundary without
        interpolating the entire domain topography. The computed depth values are stored in
        `self.ds_depth_coords`.

        Parameters
        ----------
        zeta : xr.DataArray or float
            Free-surface elevation (`zeta`). Can be:
            - A scalar float value (constant sea surface height).
            - An `xarray.DataArray` with spatial variations. If provided as an array, it may have a
              time dimension, but must be **1D** (varying only in time).
        direction : str
            The boundary direction for which depth coordinates are computed. Must be one of:
            - "north"
            - "south"
            - "east"
            - "west"
        location : str
            Grid location at which depth is computed. Must be one of:
            - `"rho"`: Depth at scalar grid points.
            - `"u"`: Depth at U-velocity grid points.
            - `"v"`: Depth at V-velocity grid points.
        depth_type : str, optional
            Type of depth coordinate to compute, either:
            - `"layer"` (default): Depth at vertical layer midpoints.
            - `"interface"`: Depth at vertical layer interfaces.

        Notes
        -----
        - This method is optimized for boundary computations by selecting only the relevant margin
          (2 grid cells) instead of interpolating the entire domain.
        """
        key = f"{depth_type}_depth_{location}_{direction}"
        if key not in self.ds_depth_coords:
            if location in ["u", "v"]:
                # selection of margin consisting of 2 grid cells
                h = self.grid.ds["h"].isel(**self.bdry_coords["vector"][direction])
                if location == "u":
                    h = interpolate_from_rho_to_u(h)
                    h = h.isel(**self.bdry_coords["u"][direction])
                elif location == "v":
                    h = interpolate_from_rho_to_v(h)
                    h = h.isel(**self.bdry_coords["v"][direction])
            else:
                h = self.grid.ds["h"].isel(**self.bdry_coords["rho"][direction])

            if depth_type == "layer":
                depth = compute_depth(
                    zeta, h, self.grid.hc, self.grid.ds.Cs_r, self.grid.ds.sigma_r
                )
            else:
                depth = compute_depth(
                    zeta, h, self.grid.hc, self.grid.ds.Cs_w, self.grid.ds.sigma_w
                )

            # Add metadata
            depth.attrs.update(
                {
                    "long_name": f"{depth_type} depth at {location}-points along {direction}ern boundary",
                    "units": "m",
                }
            )

            self.ds_depth_coords[key] = depth

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
        ds.attrs["apply_2d_horizontal_fill"] = str(self.apply_2d_horizontal_fill)
        ds.attrs["adjust_depth_for_sea_surface_height"] = str(
            self.adjust_depth_for_sea_surface_height
        )

        ds.attrs["theta_s"] = self.grid.ds.attrs["theta_s"]
        ds.attrs["theta_b"] = self.grid.ds.attrs["theta_b"]
        ds.attrs["hc"] = self.grid.ds.attrs["hc"]

        ds, bry_time = add_time_info_to_ds(
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
            if self.variable_info[var_name]["validate"]:
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

        # Load the data
        if self.use_dask:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                field = field.load()

        if "s_rho" in field.dims:
            layer_depth = self.ds_depth_coords[f"layer_depth_{location}_{direction}"]
            if self.adjust_depth_for_sea_surface_height:
                layer_depth = layer_depth.isel(time=time).load()
            field = field.assign_coords({"layer_depth": layer_depth})
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
                interface_depth = self.ds_depth_coords[
                    f"interface_depth_{location}_{direction}"
                ]
                if self.adjust_depth_for_sea_surface_height:
                    interface_depth = interface_depth.isel(time=time)
                # restrict number of layer_contours to 10 for the sake of plot clearity
                nr_layers = len(interface_depth["s_w"])
                selected_layers = np.linspace(
                    0, nr_layers - 1, min(nr_layers, 10), dtype=int
                )
                interface_depth = interface_depth.isel(s_w=selected_layers)

            else:
                interface_depth = None

            _section_plot(
                field,
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
        group: bool = True,
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
            Whether to divide the dataset into multiple files based on temporal frequency. Defaults to `True`.

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

        forcing_dict = to_dict(self, exclude=["use_dask"])
        write_to_yaml(forcing_dict, filepath)

    @classmethod
    def from_yaml(
        cls,
        filepath: Union[str, Path],
        use_dask: bool = False,
    ) -> "BoundaryForcing":
        """Create an instance of the BoundaryForcing class from a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file from which the parameters will be read.
        use_dask: bool, optional
            Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.

        Returns
        -------
        BoundaryForcing
            An instance of the BoundaryForcing class.
        """
        filepath = Path(filepath)

        grid = Grid.from_yaml(filepath)
        params = from_yaml(cls, filepath)

        # Create and return an instance of InitialConditions
        return cls(grid=grid, **params, use_dask=use_dask)


def apply_1d_horizontal_fill(data_array: xr.DataArray) -> xr.DataArray:
    """Forward and backward fill NaN values in a single horizontal dimension for open
    boundaries.

    Parameters
    ----------
    data_array : xarray.DataArray
        The data array to be updated.

    Returns
    -------
    xarray.DataArray
        The updated data array with NaN values filled.

    Raises
    ------
    ValueError
        If more than one horizontal dimension is found or none at all.
    """

    horizontal_dims = ["eta_rho", "eta_v", "xi_rho", "xi_u"]
    selected_horizontal_dim = None

    # Determine the horizontal dimension to fill
    for dim in horizontal_dims:
        if dim in data_array.dims:
            if selected_horizontal_dim is not None:
                raise ValueError(
                    f"More than one horizontal dimension found in variable '{data_array.name}'."
                )
            selected_horizontal_dim = dim

    if selected_horizontal_dim is None:
        raise ValueError(
            f"No valid horizontal dimension found for variable '{data_array.name}'."
        )

    # Forward and backward fill in the horizontal direction
    filled = one_dim_fill(data_array, selected_horizontal_dim, direction="forward")
    return one_dim_fill(filled, selected_horizontal_dim, direction="backward")
