import xarray as xr
import numpy as np
import pandas as pd
import yaml
import importlib.metadata
import warnings
from typing import Dict, Union, List
from dataclasses import dataclass, field, asdict
from roms_tools.setup.grid import Grid
from roms_tools.setup.regrid import LateralRegrid, VerticalRegrid
from datetime import datetime
from roms_tools.setup.datasets import GLORYSDataset, CESMBGCDataset
from roms_tools.setup.utils import (
    get_variable_metadata,
    group_dataset,
    save_datasets,
    get_target_coords,
    rotate_velocities,
    compute_barotropic_velocity,
    transpose_dimensions,
    one_dim_fill,
    nan_check,
    substitute_nans_by_fillvalue,
)
from roms_tools.setup.plot import _section_plot, _line_plot
import matplotlib.pyplot as plt
from pathlib import Path


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

    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.
    apply_2d_horizontal_fill: bool, optional
        Indicates whether to perform a two-dimensional horizontal fill on the source data prior to regridding to boundaries.
        If `False`, a one-dimensional horizontal fill is performed separately on each of the four regridded boundaries.
        Defaults to `False`.
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.

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
    model_reference_date: datetime = datetime(2000, 1, 1)
    apply_2d_horizontal_fill: bool = False
    use_dask: bool = False

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

        variable_info = self._set_variable_info(data)
        bdry_coords = get_boundary_info()
        ds = xr.Dataset()

        for direction in ["south", "east", "north", "west"]:
            if self.boundaries[direction]:

                bdry_target_coords = {
                    "lat": target_coords["lat"].isel(
                        **bdry_coords["vector"][direction]
                    ),
                    "lon": target_coords["lon"].isel(
                        **bdry_coords["vector"][direction]
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
                    name for name, info in variable_info.items() if info["is_vector"]
                ]
                if len(vector_var_names) > 0:
                    lon = target_coords["lon"].isel(**bdry_coords["vector"][direction])
                    lat = target_coords["lat"].isel(**bdry_coords["vector"][direction])
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
                    for name, info in variable_info.items()
                    if not info["is_vector"]
                ]
                if len(tracer_var_names) > 0:
                    lon = target_coords["lon"].isel(**bdry_coords["rho"][direction])
                    lat = target_coords["lat"].isel(**bdry_coords["rho"][direction])
                    lateral_regrid = LateralRegrid(
                        {"lat": lat, "lon": lon}, bdry_data.dim_names
                    )
                    for var_name in tracer_var_names:
                        if var_name in bdry_data.var_names.keys():
                            processed_fields[var_name] = lateral_regrid.apply(
                                bdry_data.ds[bdry_data.var_names[var_name]]
                            )

                # rotation of velocities and interpolation to u/v points
                if "u" in variable_info and "v" in variable_info:
                    angle = target_coords["angle"].isel(
                        **bdry_coords["vector"][direction]
                    )
                    (processed_fields["u"], processed_fields["v"],) = rotate_velocities(
                        processed_fields["u"],
                        processed_fields["v"],
                        angle,
                        interpolate=True,
                    )

                # selection of outermost margin for u/v variables
                for var_name in variable_info.keys():
                    if var_name in processed_fields:
                        location = variable_info[var_name]["location"]
                        if location in ["u", "v"]:
                            processed_fields[var_name] = processed_fields[
                                var_name
                            ].isel(**bdry_coords[location][direction])

                if not self.apply_2d_horizontal_fill:
                    processed_fields = apply_1d_horizontal_fill(processed_fields)

                # vertical regridding
                for location in ["rho", "u", "v"]:
                    var_names = [
                        name
                        for name, info in variable_info.items()
                        if info["location"] == location and info["is_3d"]
                    ]
                    if len(var_names) > 0:
                        vertical_regrid = VerticalRegrid(
                            self.grid.ds[f"layer_depth_{location}"].isel(
                                **bdry_coords[location][direction]
                            ),
                            bdry_data.ds[bdry_data.dim_names["depth"]],
                        )
                        for var_name in var_names:
                            if var_name in processed_fields:
                                processed_fields[var_name] = vertical_regrid.apply(
                                    processed_fields[var_name]
                                )

                # compute barotropic velocities
                if "u" in variable_info and "v" in variable_info:
                    for var_name in ["u", "v"]:
                        processed_fields[
                            f"{var_name}bar"
                        ] = compute_barotropic_velocity(
                            processed_fields[var_name],
                            self.grid.ds[f"interface_depth_{var_name}"].isel(
                                **bdry_coords[var_name][direction]
                            ),
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

        self._validate(ds, variable_info, bdry_coords)

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
        if self.type == "physics":
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
        elif self.type == "bgc":
            variable_info = {}
            for var_name in data.var_names.keys():
                variable_info[var_name] = default_info

        return variable_info

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
        existing_vars = [var_name for var_name in variables_to_drop if var_name in ds]
        ds = ds.drop_vars(existing_vars)

        return ds

    def _get_coordinates(self, direction, point):
        """Retrieve layer and interface depth coordinates for a specified grid boundary.

        This method extracts the layer depth and interface depth coordinates along
        a specified boundary (north, south, east, or west) and for a specified point
        type (rho, u, or v) from the grid dataset.

        Parameters
        ----------
        direction : str
            The direction of the boundary to retrieve coordinates for. Valid options
            are "north", "south", "east", and "west".
        point : str
            The type of grid point to retrieve coordinates for. Valid options are
            "rho" for the grid's central points, "u" for the u-flux points, and "v"
            for the v-flux points.

        Returns
        -------
        xarray.DataArray, xarray.DataArray
            The layer depth and interface depth coordinates for the specified grid
            boundary and point type.
        """

        bdry_coords = get_boundary_info()

        layer_depth = self.grid.ds[f"layer_depth_{point}"].isel(
            **bdry_coords[point][direction]
        )
        interface_depth = self.grid.ds[f"interface_depth_{point}"].isel(
            **bdry_coords[point][direction]
        )

        return layer_depth, interface_depth

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
        if data.climatology:
            ds.attrs["climatology"] = str(True)
            # Preserve absolute time coordinate for readability
            ds = ds.assign_coords(
                {"abs_time": np.datetime64(self.model_reference_date) + ds["time"]}
            )
            # Convert to pandas TimedeltaIndex
            timedelta_index = pd.to_timedelta(ds["time"].values)

            # Determine the start of the year for the base_datetime
            start_of_year = datetime(self.model_reference_date.year, 1, 1)

            # Calculate the offset from midnight of the new year
            offset = self.model_reference_date - start_of_year

            # Convert the timedelta to nanoseconds first, then to days
            bry_time = xr.DataArray(
                (timedelta_index - offset).view("int64") / 3600 / 24 * 1e-9,
                dims="time",
            )

        else:
            # Preserve absolute time coordinate for readability
            ds = ds.assign_coords({"abs_time": ds["time"]})
            bry_time = (
                (ds["time"] - np.datetime64(self.model_reference_date)).astype(
                    "float64"
                )
                / 3600
                / 24
                * 1e-9
            )

        ds = ds.assign_coords({"bry_time": bry_time})
        ds["bry_time"].attrs[
            "long_name"
        ] = f"days since {str(self.model_reference_date)}"
        ds["bry_time"].encoding["units"] = "days"
        ds["bry_time"].attrs["units"] = "days"
        ds = ds.swap_dims({"time": "bry_time"})
        ds = ds.drop_vars("time")
        ds.encoding["unlimited_dims"] = "bry_time"

        if data.climatology:
            ds["bry_time"].attrs["cycle_length"] = 365.25

        return ds

    def _validate(self, ds, variable_info, bdry_coords):
        """Validate the dataset for NaN values at the first time step based on the fill
        method used.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to validate.

        variable_info : dict
            A dictionary containing metadata about the variables, including their locations (e.g., 'rho', 'u', 'v').

        bdry_coords : dict
            A dictionary containing the boundary coordinates for each variable location.

        Raises
        ------
        ValueError
            If NaN values are found in any of the specified variables at wet points,
            indicating incomplete data coverage.

        Notes
        -----
        Validation is performed on the initial boundary time step (`bry_time=0`) for each
        variable in the dataset. If the `apply_2d_horizontal_fill` attribute is set to False,
        a warning is issued instead of a strict NaN check, as the data may not be reliably validated.
        Conversely, if `apply_2d_horizontal_fill` is True, a strict NaN check is performed, raising
        a ValueError if any NaN values are detected.
        """
        if self.apply_2d_horizontal_fill:
            # Strict NaN check with ValueError makes sense to be applied
            for var_name in variable_info:
                location = variable_info[var_name]["location"]

                # Select the appropriate mask based on variable location
                if location == "rho":
                    mask = self.grid.ds.mask_rho
                elif location == "u":
                    mask = self.grid.ds.mask_u
                elif location == "v":
                    mask = self.grid.ds.mask_v
                else:
                    continue  # Skip if location is not recognized

                for direction in ["south", "east", "north", "west"]:
                    if self.boundaries[direction]:
                        bdry_var_name = f"{var_name}_{direction}"

                        # Check for NaN values at the first time step using the nan_check function
                        nan_check(
                            ds[bdry_var_name].isel(bry_time=0),
                            mask.isel(**bdry_coords[location][direction]),
                        )
        else:
            # Can't apply strict NaN check because land values haven't been filled before regridding step; instead warn user
            for direction in ["south", "east", "north", "west"]:
                if self.boundaries[direction]:
                    for var_name in variable_info:
                        bdry_var_name = f"{var_name}_{direction}"
                        if ds[bdry_var_name].isel(bry_time=0).isnull().any().values:
                            warnings.warn(
                                f"NaN values detected in regridded variables along the {direction}ern boundary. This may indicate that the entire boundary is on land in the source data, or that the source data does not cover this boundary.",
                                UserWarning,
                            )
                            # Break after the first warning for this direction to avoid duplicates
                            break

    def plot(
        self,
        var_name,
        time=0,
        layer_contours=False,
    ) -> None:
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

        field = self.ds[var_name].isel(bry_time=time).load()
        title = field.long_name

        if "s_rho" in field.dims:
            if var_name.startswith(("u_", "ubar_")):
                point = "u"
            elif var_name.startswith(("v_", "vbar_")):
                point = "v"
            else:
                point = "rho"
            direction = var_name.split("_")[-1]

            layer_depth, interface_depth = self._get_coordinates(direction, point)

            field = field.assign_coords({"layer_depth": layer_depth})

        # chose colorbar
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
                # restrict number of layer_contours to 10 for the sake of plot clearity
                nr_layers = len(interface_depth["s_w"])
                selected_layers = np.linspace(
                    0, nr_layers - 1, min(nr_layers, 10), dtype=int
                )
                interface_depth = interface_depth.isel(s_w=selected_layers)

            else:
                interface_depth = None

            _section_plot(
                field, interface_depth=interface_depth, title=title, kwargs=kwargs
            )
        else:
            _line_plot(field, title=title)

    def save(
            self, filepath: Union[str, Path], np_eta: int = None, np_xi: int = None, group : bool = False
    ) -> None:
        """Save the boundary forcing fields to one or more netCDF4 files.

        This method saves the dataset either as a single file or as multiple files depending on the partitioning and grouping options.
        The dataset can be saved in two modes:

        1. **Single File Mode (default)**:
            - If both `np_eta` and `np_xi` are `None`, the entire dataset is saved as a single netCDF4 file.
            - The file is named based on the `filepath`, with `.nc` automatically appended.
        
        2. **Partitioned Mode**:
            - If either `np_eta` or `np_xi` is specified, the dataset is partitioned into spatial tiles along the `eta` and `xi` axes.
            - Each tile is saved as a separate netCDF4 file, and filenames are modified with an index (e.g., `"filepath_YYYYMM.0.nc"`, `"filepath_YYYYMM.1.nc"`).

        Additionally, if `group` is set to `True`, the dataset is first grouped into temporal subsets, resulting in multiple grouped files before partitioning and saving.

        Parameters
        ----------
        filepath : Union[str, Path]
            The base path and filename for the output files. The format of the filenames depends on whether partitioning is used
            and the temporal range of the data. For partitioned datasets, files will be named with an additional index, e.g.,
            `"filepath_YYYYMM.0.nc"`, `"filepath_YYYYMM.1.nc"`, etc.
        np_eta : int, optional
            The number of partitions along the `eta` direction. If `None`, no spatial partitioning is performed.
        np_xi : int, optional
            The number of partitions along the `xi` direction. If `None`, no spatial partitioning is performed.
        group: bool, optional
            If `True`, groups the dataset into multiple files based on temporal data frequency. Defaults to `False`.

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
            dataset_list, output_filenames = group_dataset(self.ds.load(), str(filepath))
        else:
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

        boundary_forcing_data = {
            "BoundaryForcing": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "boundaries": self.boundaries,
                "source": self.source,
                "type": self.type,
                "model_reference_date": self.model_reference_date.isoformat(),
            }
        }

        yaml_data = {
            **grid_yaml_data,
            **boundary_forcing_data,
        }

        with filepath.open("w") as file:
            # Write header
            file.write(header)
            # Write YAML data
            yaml.dump(yaml_data, file, default_flow_style=False)

    @classmethod
    def from_yaml(
        cls, filepath: Union[str, Path], use_dask: bool = False
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
        # Read the entire file content
        with filepath.open("r") as file:
            file_content = file.read()

        # Split the content into YAML documents
        documents = list(yaml.safe_load_all(file_content))

        boundary_forcing_data = None

        # Process the YAML documents
        for doc in documents:
            if doc is None:
                continue
            if "BoundaryForcing" in doc:
                boundary_forcing_data = doc["BoundaryForcing"]
                break

        if boundary_forcing_data is None:
            raise ValueError("No BoundaryForcing configuration found in the YAML file.")

        # Convert from string to datetime
        for date_string in ["model_reference_date", "start_time", "end_time"]:
            boundary_forcing_data[date_string] = datetime.fromisoformat(
                boundary_forcing_data[date_string]
            )

        grid = Grid.from_yaml(filepath)

        # Create and return an instance of InitialConditions
        return cls(grid=grid, **boundary_forcing_data, use_dask=use_dask)


def get_boundary_info():
    """This function provides information about the boundary points for the rho, u, and
    v variables on the grid, specifying the indices for the south, east, north, and west
    boundaries.

    Returns
    -------
    dict
        A dictionary where keys are variable types ("rho", "u", "v"), and values
        are nested dictionaries mapping directions ("south", "east", "north", "west")
        to the corresponding boundary coordinates.
    """

    # Boundary coordinates
    bdry_coords = {
        "rho": {
            "south": {"eta_rho": 0},
            "east": {"xi_rho": -1},
            "north": {"eta_rho": -1},
            "west": {"xi_rho": 0},
        },
        "u": {
            "south": {"eta_rho": 0},
            "east": {"xi_u": -1},
            "north": {"eta_rho": -1},
            "west": {"xi_u": 0},
        },
        "v": {
            "south": {"eta_v": 0},
            "east": {"xi_rho": -1},
            "north": {"eta_v": -1},
            "west": {"xi_rho": 0},
        },
        "vector": {
            "south": {"eta_rho": [0, 1]},
            "east": {"xi_rho": [-2, -1]},
            "north": {"eta_rho": [-2, -1]},
            "west": {"xi_rho": [0, 1]},
        },
    }

    return bdry_coords


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
