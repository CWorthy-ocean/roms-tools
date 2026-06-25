import importlib.metadata
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import label

from roms_tools import Grid
from roms_tools.datasets.lat_lon_datasets import (
    GLORYSDataset,
    GLORYSDefaultDataset,
)
from roms_tools.fill import one_dim_fill
from roms_tools.plot import line_plot, section_plot
from roms_tools.regrid import LateralRegridToROMS, VerticalRegrid
from roms_tools.setup.bgc_source import (
    BGC_VARIABLE_INFO,
    BGCSource,
    instantiate_bgc_dataset,
    merge_bgc_fields,
)
from roms_tools.setup.utils import (
    RawDataSource,
    _compute_density_coord,
    add_time_info_to_ds,
    check_and_set_boundaries,
    compute_barotropic_velocity,
    compute_missing_bgc_variables,
    deserialize_forcing_data,
    from_yaml,
    get_boundary_coords,
    get_target_coords,
    get_variable_metadata,
    group_dataset,
    nan_check,
    pop_grid_data,
    substitute_nans_by_fillvalue,
    to_dict,
    write_to_yaml,
)
from roms_tools.utils import (
    interpolate_cyclic_time,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    rotate_velocities,
    save_datasets,
    transpose_dimensions,
)
from roms_tools.vertical_coordinate import compute_depth


def _broadcast_static_to_time(
    fields: dict[str, xr.DataArray],
) -> dict[str, xr.DataArray]:
    """Broadcast time-free fields to match the time dimension of time-varying fields.

    When a static (no-time) source such as GLODAPv2 is mixed with a time-varying
    source (e.g. CESM, UNIFIED), the merged dict contains DataArrays with and without
    a ``"time"`` dimension.  This function broadcasts the time-free arrays against the
    first time-varying array found, using
    :meth:`xarray.DataArray.broadcast_like` which is dask-safe — no ``.compute()``
    is triggered.

    If no time-varying field is present (pure static source), the dict is returned
    unchanged; the caller must handle the no-time case.
    """
    time_template = next(
        (da for da in fields.values() if "time" in da.dims), None
    )
    if time_template is None:
        return fields
    for var in list(fields):
        if "time" not in fields[var].dims:
            fields[var] = fields[var].broadcast_like(time_template)
    return fields


def _interpolate_phys_to_bgc_time(
    phys_da: xr.DataArray,
    time_dim: str,
    bgc_time_coord: xr.DataArray,
    bgc_climatology: bool,
) -> xr.DataArray:
    """Interpolate a physics DataArray onto the BGC time coordinate.

    For climatology BGC sources (``bgc_climatology=True``) a cyclic linear
    interpolation is performed in fractional day-of-year space.  Otherwise a
    standard ``datetime64`` linear interpolation is used.

    Parameters
    ----------
    phys_da : xr.DataArray
        Physics data with a ``datetime64`` time dimension named ``time_dim``.
    time_dim : str
        Name of the time dimension in ``phys_da``.
    bgc_time_coord : xr.DataArray
        Target time coordinate from the BGC dataset (1-D).
    bgc_climatology : bool
        Whether the BGC dataset is a climatology.

    Returns
    -------
    xr.DataArray
        ``phys_da`` interpolated to ``bgc_time_coord``.
    """
    if bgc_climatology:
        bgc_doy = (bgc_time_coord / np.timedelta64(1, "D")).values + 1.0
        phys_doy = phys_da[time_dim].dt.dayofyear.values.astype(float)
        phys_for_interp = phys_da.assign_coords(
            {time_dim: xr.DataArray(phys_doy, dims=[time_dim])}
        )
        result = interpolate_cyclic_time(phys_for_interp, time_dim, time_dim, bgc_doy)
        return result.assign_coords({time_dim: bgc_time_coord.values})
    return phys_da.interp({time_dim: bgc_time_coord}, method="linear")


def _broadcast_scalar_bgc_fields(
    fields: dict[str, xr.DataArray],
) -> dict[str, xr.DataArray]:
    """Broadcast any 0-D (scalar) DataArrays to match the first N-D field.

    Called after :func:`~roms_tools.setup.bgc_source.merge_bgc_fields` to expand
    Mode B (constant) contributions to the ROMS boundary shape.

    Uses :meth:`xarray.DataArray.broadcast_like` which is dask-safe — no
    ``.compute()`` is triggered.
    """
    template = next((da for da in fields.values() if da.ndim > 0), None)
    if template is None:
        return fields
    for var in list(fields):
        if fields[var].ndim == 0:
            fields[var] = fields[var].broadcast_like(template)
    return fields


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
    boundaries : dict[str, bool], optional
        Specifies which grid boundaries ('south', 'east', 'north', 'west') are active and to be processed.
        if not provided, valid (non-land) boundaries are enabled automatically.
    source : RawDataSource
        Dictionary specifying the source of the boundary forcing data. Keys include:

          - "name" (str): Name of the data source (e.g., "GLORYS").
          - "path" (Union[str, Path, List[Union[str, Path]]]): The path to the raw data file(s). This can be:

            - A single string (with or without wildcards).
            - A single Path object.
            - A list of strings or Path objects.
            If omitted, the data will be streamed via the Copernicus Marine Toolkit.
            Note: streaming is currently not recommended due to performance limitations.
          - "climatology" (bool): Indicates if the data is climatology data. Defaults to False.

    type : str
        Specifies the type of forcing data. Options are:

          - "physics": for physical atmospheric forcing.
          - "bgc": for biogeochemical forcing.

    apply_2d_horizontal_fill : bool, optional
        Indicates whether to perform a two-dimensional horizontal fill on the source data prior to regridding to boundaries.
        If `False`, a one-dimensional horizontal fill is performed separately on each of the four regridded boundaries.
        Defaults to `False`.
    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.
    chunks : dict[str, int], optional
        Dictionary specifying chunk sizes for dask dimensions, e.g., ``{"latitude": 100, "longitude": 100}``.
        If provided, these chunks override the default chunking scheme when ``use_dask=True``.
        Defaults to None (default chunking is used).
    initial_slice_bounds : dict, optional
        Optional horizontal subset to apply when loading with dask. Only Geographic bounds are supported:
         ``{"latitude": (min_lat, max_lat), "longitude": (min_lon, max_lon)}`` in degrees. The
         bounds are applied to the dataset before reading the underlying datasets to reduce memory usage.
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
    boundaries: dict[str, bool] | None = None
    """Dictionary specifying which boundaries are forced (south, east, north, west)."""
    source: RawDataSource | BGCSource | list[BGCSource]
    """Boundary forcing data source.  For ``type='physics'`` this is a
    :data:`~roms_tools.setup.utils.RawDataSource` dict.  For ``type='bgc'`` this is a
    :class:`~roms_tools.setup.bgc_source.BGCSource` or an ordered list of them (first
    source has highest priority)."""
    type: str = "physics"
    """Specifies the type of forcing data ("physics", "bgc")."""
    apply_2d_horizontal_fill: bool = False
    """Whether to perform a two-dimensional horizontal fill on the source data prior to
    regridding to boundaries."""
    model_reference_date: datetime = datetime(2000, 1, 1)
    """Reference date for the model."""
    use_dask: bool = False
    """Whether to use dask for processing."""
    chunks: dict[str, int] | None = None
    """Optional Dask chunk sizes for lat/lon boundary-forcing sources."""
    initial_slice_bounds: dict[str, tuple[int | float, int | float]] | None = None
    """Optional initial bounding slice when loading source data (Dask); see dataset classes."""
    bypass_validation: bool = False
    """Whether to skip validation checks in the processed data."""
    use_density_interpolation: bool = False
    """Interpolate BGC tracers in density space rather than depth space when True.

    Requires that the BGC source dataset declares ``bgc_source_ts`` (a T/S pair
    for the source density coordinate) and that ``physics_forcing`` supplies the
    model T/S for the target density coordinate.  Falls back to depth-space with
    a log message if either is unavailable.  Only applied when ``type='bgc'``.
    """
    physics_forcing: "BoundaryForcing | None" = None
    """Physics BoundaryForcing whose T/S fields supply the target density coordinate
    for density-space BGC tracer interpolation."""

    ds: xr.Dataset = field(init=False, repr=False)
    """An xarray Dataset containing post-processed variables ready for input into
    ROMS."""
    adjust_depth_for_sea_surface_height: bool = field(init=False)
    """Whether to account for sea surface height when computing depth coordinates."""
    ds_depth_coords: xr.Dataset = field(init=False, repr=False)
    """An xarray Dataset containing the depth coordinates."""

    def __post_init__(self):
        # Initialize depth coordinates
        self.adjust_depth_for_sea_surface_height = False
        self.ds_depth_coords = xr.Dataset()

        self._input_checks()

        if (
            self.type == "bgc"
            and self.use_density_interpolation
            and self.physics_forcing is None
        ):
            logging.info(
                "use_density_interpolation=True but no physics_forcing provided. "
                "BGC tracers will be interpolated in depth space instead."
            )

        # BGC has its own multi-source pipeline; physics continues below
        if self.type == "bgc":
            self.ds = self._process_bgc()
            return

        target_coords = get_target_coords(self.grid)

        data = self._get_data()

        if self.apply_2d_horizontal_fill:
            data.choose_subdomain(
                target_coords,
                unchunk_lateral_dims=True,
            )
            # Enforce double precision to ensure reproducibility
            data.convert_to_float64()
            data.extrapolate_deepest_to_bottom()
            data.apply_lateral_fill()

        self._set_variable_info()
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

        for direction, is_enabled in self.boundaries.items():
            if is_enabled:
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
                    unchunk_lateral_dims=True,
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
                    (
                        processed_fields["u"],
                        processed_fields["v"],
                    ) = rotate_velocities(
                        processed_fields["u"],
                        processed_fields["v"],
                        angle,
                        interpolate_after=True,
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
                    if not self.bypass_validation:
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
                        vertical_regrid = VerticalRegrid(
                            bdry_data.ds, source_dim=bdry_data.dim_names["depth"]
                        )
                        for var_name in filtered_vars:
                            if var_name in processed_fields:
                                processed_fields[var_name] = vertical_regrid.apply(
                                    processed_fields[var_name],
                                    source_depth_coords=bdry_data.ds[
                                        bdry_data.dim_names["depth"]
                                    ],
                                    target_depth_coords=self.ds_depth_coords[
                                        f"layer_depth_{location}_{direction}"
                                    ],
                                )

                # compute barotropic velocities
                if "u" in var_names and "v" in var_names:
                    self._get_depth_coordinates(zeta_u, direction, "u", "interface")
                    self._get_depth_coordinates(zeta_v, direction, "v", "interface")
                    for location in ["u", "v"]:
                        processed_fields[f"{location}bar"] = (
                            compute_barotropic_velocity(
                                processed_fields[location],
                                self.ds_depth_coords[
                                    f"interface_depth_{location}_{direction}"
                                ],
                            )
                        )

                # Reorder dimensions
                for var_name in processed_fields:
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

        self.ds = ds

    def _input_checks(self) -> None:
        """Validate and normalize user-provided input parameters."""
        # -------------------------------------------------------
        # Time range checks
        # -------------------------------------------------------
        if (self.start_time is None) != (self.end_time is None):
            raise ValueError(
                "Both `start_time` and `end_time` must be provided together as datetime objects or both should be None."
            )

        if self.start_time is None and self.end_time is None:
            logging.warning(
                "Both `start_time` and `end_time` are None. No time filtering will be applied to the source data."
            )

        # -------------------------------------------------------
        # Type check
        # -------------------------------------------------------
        if self.type not in {"physics", "bgc"}:
            raise ValueError("`type` must be either 'physics' or 'bgc'.")

        # -------------------------------------------------------
        # Source configuration checks
        # -------------------------------------------------------
        if self.type == "bgc":
            # Normalise to list[BGCSource]
            if isinstance(self.source, BGCSource):
                self.source = [self.source]
            elif not isinstance(self.source, list) or not all(
                isinstance(s, BGCSource) for s in self.source
            ):
                raise ValueError(
                    "For type='bgc', `source` must be a BGCSource or a list of BGCSource objects."
                )
            if not self.source:
                raise ValueError(
                    "For type='bgc', `source` must contain at least one BGCSource."
                )
        else:
            # Physics: existing dict-based validation (unchanged)
            if "name" not in self.source:
                raise ValueError("`source` must include a 'name'.")

            if "path" not in self.source:
                if self.source["name"] != "GLORYS":
                    raise ValueError("`source` must include a 'path'.")
                self.source["path"] = GLORYSDefaultDataset.dataset_name

            # Assign default value
            self.source["climatology"] = self.source.get("climatology", False)

        # -------------------------------------------------------
        # Boundary selection defaults and validation
        # -------------------------------------------------------

        self.boundaries = check_and_set_boundaries(
            self.boundaries, self.grid.ds.mask_rho
        )

        # -------------------------------------------------------
        # Depth adjustment checks
        # -------------------------------------------------------
        if self.type == "bgc" and self.adjust_depth_for_sea_surface_height:
            logging.warning(
                "adjust_depth_for_sea_surface_height is not applicable for BGC fields. "
                "Setting it to False."
            )
            self.adjust_depth_for_sea_surface_height = False

    def _get_data(self) -> GLORYSDataset | GLORYSDefaultDataset:
        """Instantiate the physics dataset.  BGC sources are handled by _process_bgc().

        Returns
        -------
        GLORYSDataset or GLORYSDefaultDataset
        """
        physics_map: dict[str, dict[str, type[GLORYSDataset | GLORYSDefaultDataset]]] = {
            "GLORYS": {
                "external": GLORYSDataset,
                "default": GLORYSDefaultDataset,
            },
        }

        source_name = str(self.source["name"])
        if source_name not in physics_map:
            raise ValueError(
                f'Valid options for source["name"] for type "physics" include: '
                f'{" and ".join(physics_map)}'
            )

        has_no_path = "path" not in self.source
        has_default_path = self.source.get("path") == GLORYSDefaultDataset.dataset_name
        variant = "default" if (has_no_path or has_default_path) else "external"
        data_type = physics_map[source_name][variant]

        if isinstance(self.source["path"], bool):
            raise ValueError('source["path"] cannot be a boolean here')

        return data_type(
            filename=self.source["path"],
            start_time=self.start_time,
            end_time=self.end_time,
            climatology=self.source["climatology"],  # type: ignore[arg-type]
            use_dask=self.use_dask,
            chunks=self.chunks,
            initial_slice_bounds=self.initial_slice_bounds,
        )

    def _set_variable_info(self):
        """Sets up a dictionary with metadata for physics variables.

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

        # Physics variable metadata (BGC is handled in _process_bgc via BGC_VARIABLE_INFO)
        self.variable_info = {
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

    def _process_bgc(self) -> xr.Dataset:
        """Full BGC processing pipeline supporting multiple :class:`BGCSource` entries.

        Called from ``__post_init__`` when ``self.type == "bgc"``.  Returns the fully
        processed :class:`xarray.Dataset` that is assigned to ``self.ds``.
        """
        target_coords = get_target_coords(self.grid)
        self.variable_info = dict(BGC_VARIABLE_INFO)
        self._set_boundary_info()

        # Pre-instantiate Mode A datasets; apply global 2-D fill upfront if requested
        source_data: list[tuple[BGCSource, object]] = []
        for src in self.source:
            if src.name is not None:  # Mode A
                raw_data = instantiate_bgc_dataset(
                    src,
                    self.start_time,
                    self.end_time,
                    self.use_dask,
                    self.chunks,
                    self.initial_slice_bounds,
                )
                if self.apply_2d_horizontal_fill:
                    raw_data.choose_subdomain(target_coords, unchunk_lateral_dims=True)
                    raw_data.convert_to_float64()
                    raw_data.extrapolate_deepest_to_bottom()
                    raw_data.apply_lateral_fill()
            else:
                raw_data = None
            source_data.append((src, raw_data))

        ds = xr.Dataset()
        for direction, is_enabled in self.boundaries.items():
            if is_enabled:
                # Pre-compute both depth types so plotting always works
                self._get_depth_coordinates(0, direction, "rho", "layer")
                self._get_depth_coordinates(0, direction, "rho", "interface")

                source_fields = []
                for src, raw_data in source_data:
                    partial = self._extract_bgc_source_fields(
                        src, raw_data, direction, target_coords
                    )
                    source_fields.append((src, partial))

                merged = merge_bgc_fields(source_fields)
                merged = _broadcast_scalar_bgc_fields(merged)
                merged = _broadcast_static_to_time(merged)

                for var_name in merged:
                    merged[var_name] = transpose_dimensions(merged[var_name])

                merged = compute_missing_bgc_variables(merged)
                ds = self._write_into_dataset(direction, merged, ds)

        # Determine climatology flag and source name from Mode A sources
        primary_lat_lon = next(
            (src for src, _ in source_data if src.name is not None), None
        )
        climatology = primary_lat_lon.climatology if primary_lat_lon is not None else False
        source_names = ", ".join(
            src.name for src, _ in source_data if src.name is not None
        ) or "constants"

        # Pure-static sources produce no time dimension; add a single time step
        # so _add_global_metadata / ROMS can handle the output consistently.
        if "time" not in ds.dims:
            ds = ds.expand_dims(
                {"time": [np.datetime64(self.start_time, "ns")]}, axis=0
            )

        ds = self._add_global_metadata(
            None, ds, source_name=source_names, climatology=climatology
        )

        if not self.bypass_validation:
            self._validate(ds)

        for var_name in ds.data_vars:
            ds[var_name] = substitute_nans_by_fillvalue(ds[var_name])

        return ds

    def _extract_bgc_source_fields(
        self,
        src: BGCSource,
        raw_data,
        direction: str,
        target_coords: dict,
    ) -> dict[str, xr.DataArray]:
        """Regrid / extract BGC fields from one source for one boundary direction.

        Handles all three :class:`BGCSource` modes:

        * **Mode B** (constants): returns 0-D scalar :class:`xarray.DataArray` objects;
          broadcasting to boundary shape happens later in ``_process_bgc``.
        * **Mode C** (algorithm): delegates to ``src.algorithm(physics_ds, direction)``;
          result must already be on the ROMS boundary grid.
        * **Mode A** (lat/lon dataset): performs lateral then vertical regridding.

        All paths are dask-safe.
        """
        # --- Mode B: constants ---
        if src.constants is not None:
            return {var: xr.DataArray(float(val)) for var, val in src.constants.items()}

        # --- Mode C: physics-derived algorithm ---
        if src.physics_forcing is not None:
            return src.algorithm(src.physics_forcing.ds, direction)

        # --- Mode A: lat/lon dataset ---
        bdry_target_coords = {
            "lat": target_coords["lat"].isel(**self.bdry_coords["rho"][direction]),
            "lon": target_coords["lon"].isel(**self.bdry_coords["rho"][direction]),
            "straddle": target_coords["straddle"],
        }

        bdry_data = raw_data.choose_subdomain(
            bdry_target_coords,
            buffer_points=3,
            return_copy=True,
            unchunk_lateral_dims=True,
        )

        if not self.apply_2d_horizontal_fill:
            bdry_data.convert_to_float64()
            bdry_data.extrapolate_deepest_to_bottom()

        # Collect variables provided by this source (required + optional present)
        src_var_names = {
            var: raw_data.var_names[var]
            for var in raw_data.var_names
            if raw_data.var_names[var] in raw_data.ds.data_vars
        }
        src_var_names.update({
            var: raw_data.opt_var_names[var]
            for var in raw_data.opt_var_names
            if raw_data.opt_var_names[var] in raw_data.ds.data_vars
        })

        # Lateral regrid — all BGC vars are rho-point tracers (no vector handling)
        lon = target_coords["lon"].isel(**self.bdry_coords["rho"][direction])
        lat = target_coords["lat"].isel(**self.bdry_coords["rho"][direction])
        lateral_regrid = LateralRegridToROMS(
            {"lat": lat, "lon": lon}, bdry_data.dim_names
        )

        partial_fields: dict[str, xr.DataArray] = {}
        for var_name, src_name in src_var_names.items():
            partial_fields[var_name] = lateral_regrid.apply(bdry_data.ds[src_name])

        # 1-D lateral fill per boundary direction
        if not self.apply_2d_horizontal_fill and bdry_data.needs_lateral_fill:
            if not self.bypass_validation:
                self._validate_1d_fill(
                    partial_fields, direction, bdry_data.dim_names["depth"]
                )
            for var_name in partial_fields:
                partial_fields[var_name] = apply_1d_horizontal_fill(
                    partial_fields[var_name]
                )

        # Vertical regrid — all BGC vars are 3-D at rho-points
        # _get_depth_coordinates caches by key so repeated calls are no-ops
        self._get_depth_coordinates(0, direction, "rho", "layer")
        vertical_regrid = VerticalRegrid(
            bdry_data.ds, source_dim=bdry_data.dim_names["depth"]
        )

        # Determine if density-space interpolation is available for this source.
        ts_keys = tuple(getattr(bdry_data, "bgc_source_ts", ()))
        aux_ts_vars = [v for v in ts_keys if v in partial_fields]
        has_source_ts = len(aux_ts_vars) == 2
        use_density = (
            self.use_density_interpolation
            and self.physics_forcing is not None
            and has_source_ts
        )
        if self.use_density_interpolation and not use_density:
            reason = (
                "no physics_forcing provided"
                if self.physics_forcing is None
                else "this BGC source has no temperature/salinity"
            )
            logging.info(
                f"Density-space interpolation requested but {reason}; "
                f"falling back to depth-space ({direction} boundary)."
            )

        source_density = target_density = None
        if use_density:
            source_density, target_density = self._compute_bgc_density_coords(
                direction, bdry_data, partial_fields
            )

        tracer_vars = [v for v in partial_fields if v not in aux_ts_vars]
        for var_name in tracer_vars:
            if use_density:
                partial_fields[var_name] = vertical_regrid.apply(
                    partial_fields[var_name],
                    source_depth_coords=source_density,
                    target_depth_coords=target_density,
                )
            else:
                partial_fields[var_name] = vertical_regrid.apply(
                    partial_fields[var_name],
                    source_depth_coords=bdry_data.ds[bdry_data.dim_names["depth"]],
                    target_depth_coords=self.ds_depth_coords[
                        f"layer_depth_rho_{direction}"
                    ],
                )

        # Drop auxiliary T/S — not ROMS output variables.
        for v in aux_ts_vars:
            partial_fields.pop(v, None)

        return partial_fields

    def _compute_bgc_density_coords(
        self,
        direction: str,
        bdry_data,
        partial_fields: dict,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Build source and target density coordinates for density-space BGC interpolation.

        Source density comes from the BGC dataset's own T/S pair (``bgc_source_ts``),
        already laterally regridded to this boundary.  Target density comes from
        ``physics_forcing`` sigma-level T/S, interpolated to the BGC time axis.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            ``(source_density, target_density)``
        """
        assert self.physics_forcing is not None
        temp_key, salt_key = bdry_data.bgc_source_ts
        bgc_depth_dim = bdry_data.dim_names["depth"]

        source_density = _compute_density_coord(
            partial_fields[temp_key],
            partial_fields[salt_key],
            bgc_depth_dim,
        )

        # Determine BGC time coordinate for physics T/S alignment
        bgc_time_dim = bdry_data.dim_names.get("time")
        bgc_time_coord = None
        if bgc_time_dim and bgc_time_dim in partial_fields[temp_key].dims:
            bgc_time_coord = partial_fields[temp_key][bgc_time_dim]

        # Detect whether source is a climatology (drives cyclic time interpolation)
        primary_src = next(
            (s for s in self.source if s.name is not None), None
        )
        bgc_climatology = primary_src.climatology if primary_src is not None else False

        def _align_time(da: xr.DataArray, time_dim: str) -> xr.DataArray:
            if time_dim not in da.dims:
                return da
            if bgc_time_coord is not None:
                return _interpolate_phys_to_bgc_time(
                    da, time_dim, bgc_time_coord, bgc_climatology
                )
            return da.mean(time_dim)

        temp_sigma = self.physics_forcing.ds[f"temp_{direction}"]
        salt_sigma = self.physics_forcing.ds[f"salt_{direction}"]
        if "abs_time" in temp_sigma.coords:
            temp_sigma = temp_sigma.swap_dims({"bry_time": "abs_time"}).rename(
                {"abs_time": "time"}
            )
            salt_sigma = salt_sigma.swap_dims({"bry_time": "abs_time"}).rename(
                {"abs_time": "time"}
            )
            temp_sigma = _align_time(temp_sigma, "time")
            salt_sigma = _align_time(salt_sigma, "time")
        else:
            temp_sigma = _align_time(temp_sigma, "bry_time")
            salt_sigma = _align_time(salt_sigma, "bry_time")

        s_dim = next(d for d in temp_sigma.dims if d.startswith("s_"))
        target_density = _compute_density_coord(temp_sigma, salt_sigma, s_dim)

        return source_density, target_density

    def _write_into_dataset(self, direction, processed_fields, ds=None):
        if ds is None:
            ds = xr.Dataset()

        d_meta = get_variable_metadata()

        for var_name in processed_fields.keys():
            ds[f"{var_name}_{direction}"] = processed_fields[var_name].astype(
                np.float32
            )

            ds[f"{var_name}_{direction}"].attrs["long_name"] = (
                f"{direction}ern boundary {d_meta[var_name]['long_name']}"
            )

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

    def _add_global_metadata(
        self,
        data,
        ds=None,
        source_name: str | None = None,
        climatology: bool | None = None,
    ):
        """Add global attributes and time coordinates to the output dataset.

        Parameters
        ----------
        data : dataset object or None
            Physics dataset (provides ``data.climatology``).  Pass ``None`` for the
            BGC path and supply ``climatology`` explicitly instead.
        ds : xr.Dataset, optional
        source_name : str, optional
            Override for the ``source`` attribute (BGC multi-source path).
        climatology : bool, optional
            Override for the climatology flag (BGC multi-source path).
        """
        if ds is None:
            ds = xr.Dataset()
        ds.attrs["title"] = "ROMS boundary forcing file created by ROMS-Tools"
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"
        ds.attrs["roms_tools_version"] = roms_tools_version
        ds.attrs["start_time"] = str(self.start_time)
        ds.attrs["end_time"] = str(self.end_time)
        ds.attrs["source"] = source_name if source_name is not None else self.source["name"]
        ds.attrs["model_reference_date"] = str(self.model_reference_date)
        ds.attrs["apply_2d_horizontal_fill"] = str(self.apply_2d_horizontal_fill)
        ds.attrs["adjust_depth_for_sea_surface_height"] = str(
            self.adjust_depth_for_sea_surface_height
        )

        ds.attrs["theta_s"] = self.grid.ds.attrs["theta_s"]
        ds.attrs["theta_b"] = self.grid.ds.attrs["theta_b"]
        ds.attrs["hc"] = self.grid.ds.attrs["hc"]

        _climatology = climatology if climatology is not None else data.climatology
        ds, bry_time = add_time_info_to_ds(
            ds, self.model_reference_date, _climatology
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
        if not hasattr(self, "_warned_directions"):
            self._warned_directions = set()

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

                if num_features > 0 and direction not in self._warned_directions:
                    logging.warning(
                        f"The {direction}ern boundary is divided by land. "
                        "It would be safer (but slower and more memory-intensive) to use `apply_2d_horizontal_fill = True`."
                    )
                    self._warned_directions.add(direction)

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

                for direction, is_enabled in self.boundaries.items():
                    if is_enabled:
                        bdry_var_name = f"{var_name}_{direction}"

                        # Check for NaN values at the first time step using the nan_check function
                        if self.apply_2d_horizontal_fill:
                            error_message = None
                        else:
                            if isinstance(self.source, list):
                                src_name = ", ".join(
                                    s.name or "constants" for s in self.source
                                )
                            else:
                                src_name = self.source.get("name", "unknown")
                            error_message = (
                                f"{bdry_var_name} consists entirely of NaNs after regridding. "
                                f"This may be due to the {direction}ern boundary being on land in the "
                                f"{src_name} data, which could have a coarser resolution than the ROMS domain. "
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

            section_plot(
                field,
                interface_depth=interface_depth,
                title=title,
                kwargs=kwargs,
                ax=ax,
            )
        else:
            line_plot(field.where(mask), title=title, ax=ax)

    def save(
        self,
        filepath: str | Path,
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

    def to_yaml(self, filepath: str | Path) -> None:
        """Export the parameters of the class to a YAML file, including the version of
        roms-tools.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file where the parameters will be saved.
        """
        forcing_dict = to_dict(
            self,
            exclude=[
                "ds_depth_coords",
                "adjust_depth_for_sea_surface_height",
                "use_dask",
                "physics_forcing",
            ],
        )
        # Embed the companion physics BoundaryForcing as an optional sub-item so
        # the density-space BGC interpolation survives a YAML round-trip.
        if self.physics_forcing is not None:
            physics_dict = to_dict(
                self.physics_forcing,
                exclude=[
                    "ds_depth_coords",
                    "adjust_depth_for_sea_surface_height",
                    "use_dask",
                    "physics_forcing",
                ],
            )
            forcing_dict["BoundaryForcing"]["physics_forcing"] = physics_dict[
                "BoundaryForcing"
            ]
        write_to_yaml(forcing_dict, filepath)

    @classmethod
    def from_yaml(
        cls,
        filepath: str | Path,
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

        # Reconstruct an optional embedded physics BoundaryForcing, reusing the
        # shared grid.  The generic `from_yaml` only deserializes the top-level
        # block, so the nested block's datetimes/paths/source are restored here.
        physics_data = params.pop("physics_forcing", None)
        physics_forcing = None
        if physics_data is not None:
            physics_data = deserialize_forcing_data(physics_data)
            for name in ["source", "bgc_source"]:
                src_dict = physics_data.get(name)
                if src_dict and isinstance(src_dict, dict) and src_dict.get("grid") is not None:
                    src_dict["grid"] = Grid(**pop_grid_data(src_dict["grid"]))
            physics_forcing = cls(grid=grid, **physics_data, use_dask=use_dask)

        return cls(
            grid=grid,
            **params,
            physics_forcing=physics_forcing,
            use_dask=use_dask,
        )


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
