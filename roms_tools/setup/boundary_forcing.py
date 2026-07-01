import importlib.metadata
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from roms_tools import Grid
from roms_tools.datasets.lat_lon_datasets import (
    CESMBGCDataset,
    GLORYSDataset,
    GLORYSDefaultDataset,
    UnifiedBGCDataset,
)
from roms_tools.plot import line_plot, section_plot
from roms_tools.processing_methods import (
    BGC_INTERPOLATION_METHODS,
    BgcInterpMethod,
    RegridConfig,
    _xesmf_available,
    resolve_bgc_interp_method,
)
from roms_tools.regrid import (
    LateralRegridToROMS,
    VerticalRegrid,
)
from roms_tools.setup.utils import (
    RawDataSource,
    add_time_info_to_ds,
    build_bgc_vertical_coords,
    check_and_set_boundaries,
    compute_barotropic_velocity,
    compute_missing_bgc_variables,
    deserialize_forcing_data,
    from_yaml,
    get_boundary_coords,
    get_target_coords,
    get_variable_metadata,
    group_dataset,
    nan_check_batch,
    pop_grid_data,
    substitute_nans_by_fillvalue,
    to_dict,
    write_to_yaml,
)
from roms_tools.utils import (
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    rotate_velocities,
    save_datasets,
    transpose_dimensions,
)
from roms_tools.vertical_coordinate import compute_depth


def _interpolate_phys_to_bgc_time(
    phys_da: xr.DataArray,
    time_dim: str,
    bgc_time_coord: xr.DataArray,
    bgc_climatology: bool,
) -> xr.DataArray:
    """Sample a physics DataArray at the BGC times using nearest-time selection.

    Parameters
    ----------
    phys_da : xr.DataArray
        Physics data with a ``datetime64`` time dimension named ``time_dim``.
    time_dim : str
        Name of the time dimension in ``phys_da``.
    bgc_time_coord : xr.DataArray
        Target time coordinate from the BGC dataset (1-D).
    bgc_climatology : bool
        Whether the BGC dataset is a climatology. If True, ``bgc_time_coord``
        is expected to be ``timedelta64`` from the start of the year (as set by
        ``assign_dates_to_climatology``), and the nearest neighbour is taken
        cyclically in fractional day-of-year space (so an early-January target can
        match late-December physics). If False, nearest selection is performed in
        ``datetime64`` space.

    Returns
    -------
    xr.DataArray
        ``phys_da`` sampled at ``bgc_time_coord``, with time dimension still named
        ``time_dim`` and coordinate set to ``bgc_time_coord``.

    Notes
    -----
    The BGC boundary output is typically a 12-step climatology, and ROMS linearly
    interpolates boundary records in time at runtime, so sub-monthly precision in the
    physics T/S used only as the density/MLD anchor is washed out. Nearest-time
    selection is therefore sufficient and, unlike ``xr.interp``, requires no rechunk of
    the time axis (which would otherwise pull the entire physics time series into a
    single in-memory chunk); only the selected slices are read.
    """
    if bgc_climatology:
        # Circular nearest neighbour in fractional day-of-year space.
        bgc_doy = (bgc_time_coord / np.timedelta64(1, "D")).values + 1.0
        phys_doy = phys_da[time_dim].dt.dayofyear.values.astype(float)
        period = 365.25
        diff = np.abs(phys_doy[None, :] - np.asarray(bgc_doy)[:, None])
        nearest = np.minimum(diff, period - diff).argmin(axis=1)
        result = phys_da.isel({time_dim: nearest})
        return result.assign_coords({time_dim: bgc_time_coord.values})

    # Non-climatology: nearest selection in datetime64 space.
    return phys_da.sel({time_dim: bgc_time_coord}, method="nearest")


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

    prefill : str or None, optional
        How to fill NaN (land/void) cells in the *source* before regridding. The
        default (``None``) applies **no** source prefill: with xESMF installed,
        masked bilinear interpolation plus destination extrapolation
        (``extrap_method``) produces NaN-free boundaries directly; without xESMF,
        the source is automatically pre-filled with a cheap nearest-neighbor fill
        before scipy interpolation. Set ``prefill`` to fill the whole-domain
        source first (the regrid is then plain bilinear and ``extrap_method`` is
        ignored). Options:

          - ``"2d_lateral_fill"`` -- legacy AMG Poisson fill (smoothest, slow;
            no xESMF required). This is the modern spelling of the deprecated
            ``apply_2d_horizontal_fill=True``.
          - ``"inverse_dist"`` -- xESMF inverse-distance-weighted source fill
            (tunable via ``prefill_kwargs``; requires xESMF).
          - ``"nearest_s2d"`` -- xESMF nearest-source fill (requires xESMF).
          - ``"nearest_neighbor"`` -- cheap distance-transform fill (no xESMF;
            also the automatic fallback when xESMF is unavailable). Use for
            cross-platform reproducibility or when xESMF is unavailable and the
            AMG fill is too slow; not recommended when xESMF is available.
          - ``"creep_fill"`` -- xESMF truncated Laplace-style diffusion source
            fill (tunable via ``prefill_kwargs``; requires xESMF). **Not available
            in current released xESMF** -- requires a newer/unreleased xESMF +
            ESMF; provided for use once a supporting xESMF is installed.

        Defaults to ``None``.
    prefill_kwargs : dict, optional
        Method-specific options for ``prefill``: ``num_src_pnts`` /
        ``dist_exponent`` for ``"inverse_dist"``; ``num_levels`` for
        ``"creep_fill"``. Ignored by the other methods. Defaults to ``None``.
    regrid_method : str or None, optional
        Horizontal regrid engine, chosen independently of ``prefill``:

          - ``None`` / ``"auto"`` (default) -- use xESMF if it is installed
            (lazy, weight-reused, faster on large grids), otherwise scipy.
          - ``"xesmf"`` -- force the xESMF regridder (raises if xESMF is absent).
          - ``"scipy"`` -- force scipy ``interp``. Byte-reproducible with pre-v4
            outputs; when ``prefill`` is ``None`` a nearest-neighbor source
            pre-fill is applied automatically so scipy cannot propagate NaNs.

        Note that ``inverse_dist`` / ``nearest_s2d`` *prefills* still require xESMF
        for the fill step regardless of ``regrid_method``. Defaults to ``None``.
    extrap_method : str or None, optional
        xESMF *destination* extrapolation used on the default path
        (``prefill is None``) to fill boundary points whose source neighbors are
        all land/out of range, guaranteeing NaN-free output. ``"inverse_dist"``
        (the effective default) gives an inverse-distance-weighted average of the
        nearest source points (smoothly varying); ``"nearest_s2d"`` uses the
        single nearest source point. Ignored when ``prefill`` is set. Defaults to
        ``None`` (treated as ``"inverse_dist"``).
    extrap_kwargs : dict, optional
        Method-specific options for ``extrap_method``: ``num_src_pnts`` /
        ``dist_exponent`` for ``"inverse_dist"``. Defaults to ``None``.
    apply_2d_horizontal_fill : bool, optional
        **Deprecated** -- use ``prefill`` instead. ``True`` maps to
        ``prefill="2d_lateral_fill"`` and ``False`` to ``prefill=None``; setting
        it emits a ``DeprecationWarning``. Cannot be combined with an explicit
        ``prefill``. Defaults to ``None`` (unset).
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
    bgc_interpolation_method : str, optional
        Vertical interpolation method for BGC tracers (only used when ``type='bgc'``).
        One of:

        - ``"depth"`` (default): linear interpolation in depth.
        - ``"density"``: linear interpolation in potential-density (isopycnal) space,
          preserving water-mass properties. Density is computed via TEOS-10 sigma-0 from
          the BGC source's own T/S (source coordinate) and the physics T/S supplied by
          ``physics_forcing`` (target coordinate).
        - ``"density_mld"``: the mixed layer depth (MLD) is found in the source and target
          density fields; the source mixed layer is scaled so its MLD matches the target's,
          and below the MLD the tracer is interpolated 1:1 in depth. This keeps the mixed
          layers aligned while preserving the absolute depth of sub-mixed-layer features,
          and avoids the surface degeneracy of pure density space.

        ``"density"`` and ``"density_mld"`` require ``physics_forcing`` and a BGC source
        carrying temperature/salinity; otherwise interpolation falls back to depth space.
        Interpolation uses ``xgcm.Grid.transform`` with the linear method inside the
        source range and edge-value extrapolation outside (``mask_edges=False``).
    physics_forcing : BoundaryForcing, optional
        A physics ``BoundaryForcing`` object (``type='physics'``) whose T/S fields
        supply the target density coordinate for BGC tracer interpolation. When None and
        a density method is requested, falls back to depth-based interpolation.


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
    source: RawDataSource
    """Dictionary specifying the source of the boundary forcing data."""
    type: str = "physics"
    """Specifies the type of forcing data ("physics", "bgc")."""
    prefill: str | None = None
    """Source-side fill applied before regridding (``None`` = no prefill, the default
    NaN-aware masked-bilinear path). See the class docstring for the available methods."""
    prefill_kwargs: dict | None = None
    """Method-specific options for ``prefill`` (e.g. ``num_src_pnts``/``dist_exponent``)."""
    regrid_method: str | None = None
    """Horizontal regrid engine, independent of ``prefill``. ``None``/``"auto"`` uses xESMF
    when installed (faster on large grids) and scipy otherwise; ``"xesmf"`` forces xESMF;
    ``"scipy"`` forces scipy ``interp`` (byte-reproducible with pre-v4 outputs)."""
    extrap_method: str | None = None
    """xESMF destination extrapolation for the default path (``prefill is None``). ``None`` is
    treated as ``"inverse_dist"``. Ignored when ``prefill`` is set."""
    extrap_kwargs: dict | None = None
    """Method-specific options for ``extrap_method`` (e.g. ``num_src_pnts``/``dist_exponent``)."""
    apply_2d_horizontal_fill: bool | None = None
    """Deprecated alias for ``prefill`` (sentinel ``None`` = unset). ``True`` ->
    ``prefill="2d_lateral_fill"``, ``False`` -> ``prefill=None``; emits a DeprecationWarning."""

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
    bgc_interpolation_method: str = "depth"
    """Vertical interpolation method for BGC tracers: ``"depth"``, ``"density"``, or
    ``"density_mld"``."""
    physics_forcing: "BoundaryForcing | None" = None
    """Physics BoundaryForcing object supplying T/S for density-based BGC interpolation."""

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

        self._resolve_prefill_options()
        self._input_checks()

        target_coords = get_target_coords(self.grid)

        data = self._get_data()

        # Regrid engine is chosen independently of the prefill via the resolved
        # ``RegridConfig`` (built in _resolve_prefill_options):
        #   - prefill is None + xESMF      : masked xESMF bilinear regrid + extrap (no fill)
        #   - prefill is None + scipy      : nearest-neighbor pre-fill + scipy interp
        #   - prefill set + xESMF          : whole-domain source fill, then plain
        #                                    xESMF bilinear regrid (lazy, faster on large grids)
        #   - prefill set + scipy          : whole-domain source fill, then scipy interp
        # On a prefilled (NaN-free) source no mask or extrapolation is needed, so
        # the xESMF regrid is plain bilinear.
        regrid = self._regrid
        prefill = self.prefill
        use_xesmf = regrid.use_xesmf

        if prefill is not None:
            # Whole-domain source fill (parallels the legacy AMG path): gives the
            # fill the same ocean context across the full footprint, not a thin
            # per-boundary strip. After this the source is NaN-free, so each
            # boundary is regridded with plain bilinear (no extrapolation).
            data.choose_subdomain(
                target_coords,
                unchunk_lateral_dims=True,
            )
            # Enforce double precision to ensure reproducibility
            data.convert_to_float64()
            data.extrapolate_deepest_to_bottom()
            data.apply_prefill(
                prefill,
                prefill_kwargs=self.prefill_kwargs,
                prefill_was_user_set=True,
            )

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

        for direction, is_enabled in self.boundaries.items():
            if not is_enabled:
                continue

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
                # TODO: make per-boundary buffer_points configurable.
                buffer_points=3,
                return_copy=True,
                unchunk_lateral_dims=True,
            )

            if prefill is None:
                # Default (no source prefill) path. Prep happens per boundary.
                # Enforce double precision to ensure reproducibility
                bdry_data.convert_to_float64()
                bdry_data.extrapolate_deepest_to_bottom()
                if not use_xesmf:
                    # xESMF unavailable: nearest-neighbor pre-fill the source so the
                    # subsequent scipy interpolation cannot propagate NaNs.
                    bdry_data.apply_nearest_neighbor_fill()
            # When prefill is set, the whole-domain source was already filled
            # (float64 + deepest-to-bottom + fill) before this loop.

            # Precomputed static source masks for the xESMF masked-bilinear
            # path, matched to the field type: ``mask`` (tracer validity) for
            # tracers and ``zeta``, ``mask_vel`` (velocity validity) for u/v.
            # Reusing these stored 2D fields avoids recomputing a mask from the
            # full (lazy) source series. ``None`` means the source is already
            # NaN-free (e.g. the pre-filled UNIFIED BGC dataset, which carries
            # no mask, or a whole-domain prefill above) so the regridder uses
            # plain bilinear; irrelevant on the scipy path.
            if use_xesmf and prefill is None:
                tracer_mask = (
                    bdry_data.ds["mask"] if "mask" in bdry_data.ds.data_vars else None
                )
                vector_mask = (
                    bdry_data.ds["mask_vel"]
                    if "mask_vel" in bdry_data.ds.data_vars
                    else tracer_mask
                )
            else:
                tracer_mask = None
                vector_mask = None

            # With a prefilled (NaN-free) source, no regrid-time extrapolation
            # is needed; use plain bilinear (the config returns ``None`` then).
            regrid_extrap_method = regrid.regrid_extrap_method
            regrid_extrap_kwargs = regrid.regrid_extrap_kwargs

            processed_fields = {}

            # Filter var_names by vector fields
            filtered_vars = [
                var_name
                for var_name, info in var_names.items()
                if self.variable_info[var_name]["is_vector"]
            ]

            # lateral regridding of vector fields

            if filtered_vars:
                lon = target_coords["lon"].isel(**self.bdry_coords["vector"][direction])
                lat = target_coords["lat"].isel(**self.bdry_coords["vector"][direction])
                lateral_regrid_vector = LateralRegridToROMS(
                    {"lat": lat, "lon": lon},
                    bdry_data.dim_names,
                    source_ds=bdry_data.ds,
                    use_xesmf=use_xesmf,
                    source_mask=vector_mask,
                    extrap_method=regrid_extrap_method,
                    extrap_kwargs=regrid_extrap_kwargs,
                )
                for var_name in filtered_vars:
                    processed_fields[var_name] = lateral_regrid_vector.apply(
                        bdry_data.ds[var_names[var_name]["name"]]
                    )

                if self.adjust_depth_for_sea_surface_height:
                    # Regrid sea surface height ('zeta') onto a 2-cell-wide margin.
                    # This is needed to correctly infer depth coordinates at u- and v-points along the boundary.
                    # 'zeta' is a scalar, so it uses the tracer mask (not the
                    # velocity mask of the vector regridder); build a dedicated
                    # regridder on the same vector-margin target.
                    zeta_vector_regrid = LateralRegridToROMS(
                        {"lat": lat, "lon": lon},
                        bdry_data.dim_names,
                        source_ds=bdry_data.ds,
                        use_xesmf=use_xesmf,
                        source_mask=tracer_mask,
                        extrap_method=regrid_extrap_method,
                        extrap_kwargs=regrid_extrap_kwargs,
                    )
                    zeta_vector = zeta_vector_regrid.apply(
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
                lon = target_coords["lon"].isel(**self.bdry_coords["rho"][direction])
                lat = target_coords["lat"].isel(**self.bdry_coords["rho"][direction])
                lateral_regrid = LateralRegridToROMS(
                    {"lat": lat, "lon": lon},
                    bdry_data.dim_names,
                    source_ds=bdry_data.ds,
                    use_xesmf=use_xesmf,
                    source_mask=tracer_mask,
                    extrap_method=regrid_extrap_method,
                    extrap_kwargs=regrid_extrap_kwargs,
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

                    # The BGC dataset declares its own source T/S pair
                    # (``bgc_source_ts``, e.g. ``temp_bgc``/``salt_bgc``) that defines
                    # the source density coordinate; it is not written to output, so it
                    # is handled separately from the tracers and dropped afterwards.
                    ts_keys = tuple(getattr(bdry_data, "bgc_source_ts", ()))
                    aux_ts_vars = [
                        v
                        for v in ts_keys
                        if v in filtered_vars and v in processed_fields
                    ]
                    tracer_vars = [v for v in filtered_vars if v not in aux_ts_vars]

                    has_source_ts = len(aux_ts_vars) == 2
                    # Resolve the requested method against availability of the
                    # physics target T/S and the BGC source T/S (falls back to
                    # depth, logging the reason, when either is missing).
                    method = BgcInterpMethod.depth
                    if self.type == "bgc" and location == "rho":
                        method = resolve_bgc_interp_method(
                            self.bgc_interpolation_method,
                            has_physics_forcing=self.physics_forcing is not None,
                            has_source_ts=has_source_ts,
                            where=f"{direction} boundary",
                        )

                    source_coord = None
                    target_coord = None
                    if method != BgcInterpMethod.depth:
                        source_coord, target_coord = self._compute_bgc_vertical_coords(
                            method, direction, bdry_data, processed_fields
                        )

                    for var_name in tracer_vars:
                        if var_name not in processed_fields:
                            continue
                        if method != BgcInterpMethod.depth:
                            processed_fields[var_name] = vertical_regrid.apply(
                                processed_fields[var_name],
                                source_depth_coords=source_coord,
                                target_depth_coords=target_coord,
                            )
                        else:
                            processed_fields[var_name] = vertical_regrid.apply(
                                processed_fields[var_name],
                                source_depth_coords=bdry_data.ds[
                                    bdry_data.dim_names["depth"]
                                ],
                                target_depth_coords=self.ds_depth_coords[
                                    f"layer_depth_{location}_{direction}"
                                ],
                            )

                    # Drop the auxiliary source T/S; not ROMS output variables.
                    for v in aux_ts_vars:
                        processed_fields.pop(v, None)

            # compute barotropic velocities
            if "u" in var_names and "v" in var_names:
                self._get_depth_coordinates(zeta_u, direction, "u", "interface")
                self._get_depth_coordinates(zeta_v, direction, "v", "interface")
                for location in ["u", "v"]:
                    processed_fields[f"{location}bar"] = compute_barotropic_velocity(
                        processed_fields[location],
                        self.ds_depth_coords[f"interface_depth_{location}_{direction}"],
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

    def _resolve_prefill_options(self) -> None:
        """Build the validated :class:`RegridConfig` from the public options.

        Delegates the deprecated-flag mapping and all prefill/extrap/regrid
        validation to :meth:`RegridConfig.from_options`, then writes the resolved
        ``prefill`` back to the public field (and clears the deprecated alias) so
        the YAML round-trip emits the plain ``prefill`` string. Derived state
        (``use_xesmf``, ``effective_extrap``, ...) is read off ``self._regrid``.
        """
        # ``RegridConfig.from_options`` owns the deprecation mapping and all
        # prefill/extrap/regrid validation (``allowed_prefill`` defaults to every
        # ``PrefillMethod`` member, which is exactly the set this class accepts).
        self._regrid = RegridConfig.from_options(
            prefill=self.prefill,
            prefill_kwargs=self.prefill_kwargs,
            regrid_method=self.regrid_method,
            extrap_method=self.extrap_method,
            extrap_kwargs=self.extrap_kwargs,
            apply_2d_horizontal_fill=self.apply_2d_horizontal_fill,
            xesmf_available=_xesmf_available(),
        )
        # Persist the resolved prefill (deprecated alias mapped to a plain string)
        # so the YAML round-trip emits ``prefill`` and never the deprecated flag.
        self.prefill = (
            None if self._regrid.prefill is None else str(self._regrid.prefill)
        )
        self.apply_2d_horizontal_fill = None

    def _compute_bgc_vertical_coords(
        self,
        method: str,
        direction: str,
        bdry_data,
        processed_fields: dict,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Build source and target vertical coordinates for non-depth BGC
        interpolation (``"density"`` or ``"density_mld"``) at one boundary.

        The source T/S comes from the BGC dataset's OWN pair (``temp_bgc``/``salt_bgc``,
        carried at the boundary on the BGC depth and time grid). No regridding or time
        alignment is needed: it shares the tracers' grid and time axis.

        The target T/S comes from the model's (physics) sigma-level fields supplied by
        ``physics_forcing``, interpolated onto the BGC time axis. The actual coordinate
        construction (density vs. MLD-warped depth) is delegated to
        :func:`build_bgc_vertical_coords`.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            ``(source_coord, target_coord)``.
        """
        assert self.physics_forcing is not None
        bgc_climatology = bool(self.source["climatology"])
        bgc_depth_dim = bdry_data.dim_names["depth"]
        temp_key, salt_key = bdry_data.bgc_source_ts

        # BGC time axis (shared with the tracers) — taken from the source T/S.
        bgc_time_dim = bdry_data.dim_names.get("time")
        bgc_time_coord = None
        src_temp = processed_fields[temp_key]
        if bgc_time_dim is not None and bgc_time_dim in src_temp.dims:
            bgc_time_coord = src_temp[bgc_time_dim]

        def _align_time(da: xr.DataArray, time_dim: str) -> xr.DataArray:
            """Align ``da``'s ``time_dim`` to the BGC time axis, or collapse it."""
            if time_dim not in da.dims:
                return da
            if bgc_time_coord is not None:
                return _interpolate_phys_to_bgc_time(
                    da, time_dim, bgc_time_coord, bgc_climatology
                )
            return da.mean(time_dim)

        # --- Target density: physics (model) sigma-level T/S, aligned to BGC time ---
        # Physics BC dataset uses "bry_time" as the time dim with an "abs_time"
        # datetime64 companion coord. Swap to the datetime view before time-aligning.
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

        return build_bgc_vertical_coords(
            method,
            source_temp=processed_fields[temp_key],
            source_salt=processed_fields[salt_key],
            source_depth=bdry_data.ds[bgc_depth_dim],
            source_depth_dim=bgc_depth_dim,
            target_temp=temp_sigma,
            target_salt=salt_sigma,
            target_depth=self.ds_depth_coords[f"layer_depth_rho_{direction}"],
            target_depth_dim=s_dim,
        )

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

        if self.bgc_interpolation_method not in BGC_INTERPOLATION_METHODS:
            raise ValueError(
                f"`bgc_interpolation_method` must be one of "
                f"{BGC_INTERPOLATION_METHODS}, got {self.bgc_interpolation_method!r}."
            )

        # -------------------------------------------------------
        # Source configuration checks
        # -------------------------------------------------------
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

    def _get_data(
        self,
    ) -> GLORYSDataset | GLORYSDefaultDataset | CESMBGCDataset | UnifiedBGCDataset:
        """Determine the correct `Dataset` type and return an instance.

        Returns
        -------
        Dataset
            The `Dataset` instance

        """
        dataset_map: dict[
            str,
            dict[
                str,
                dict[
                    str,
                    type[
                        GLORYSDataset
                        | GLORYSDefaultDataset
                        | CESMBGCDataset
                        | UnifiedBGCDataset
                    ],
                ],
            ],
        ] = {
            "physics": {
                "GLORYS": {
                    "external": GLORYSDataset,
                    "default": GLORYSDefaultDataset,
                },
            },
            "bgc": {
                "CESM_REGRIDDED": defaultdict(lambda: CESMBGCDataset),
                "UNIFIED": defaultdict(lambda: UnifiedBGCDataset),
            },
        }

        source_name = str(self.source["name"])
        if source_name not in dataset_map[self.type]:
            tpl = 'Valid options for source["name"] for type {} include: {}'
            msg = tpl.format(self.type, " and ".join(dataset_map[self.type].keys()))
            raise ValueError(msg)

        has_no_path = "path" not in self.source
        has_default_path = self.source.get("path") == GLORYSDefaultDataset.dataset_name
        use_default = has_no_path or has_default_path

        variant = "default" if use_default else "external"

        data_type = dataset_map[self.type][source_name][variant]

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
        ds.attrs["prefill"] = str(self.prefill)
        ds.attrs["regrid_method"] = "xesmf" if self._regrid.use_xesmf else "scipy"
        ds.attrs["extrap_method"] = str(self._regrid.effective_extrap)
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
        # Build the NaN checks lazily and evaluate them in a single computation so a
        # lazy subgraph shared across variables (e.g. the density/MLD interpolation
        # coordinate reused across BGC tracers) is computed once, not once per variable.
        checks = []
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
                        error_message = (
                            f"{bdry_var_name} consists entirely of NaNs after regridding. "
                            f"This may be due to the {direction}ern boundary being entirely on land in the "
                            f"{self.source['name']} data, which could have a coarser resolution than the ROMS domain. "
                            f"Try setting a `prefill` method (e.g. 'inverse_dist', 'nearest_neighbor', or "
                            f"'2d_lateral_fill') to fill the source before regridding; see "
                            f"https://roms-tools.readthedocs.io/en/latest/boundary_forcing.html for details."
                        )

                        checks.append(
                            (
                                ds[bdry_var_name].isel(bry_time=0),
                                mask.isel(**self.bdry_coords[location][direction]),
                                error_message,
                            )
                        )

        nan_check_batch(checks)

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
                # Deprecated alias: superseded by ``prefill``. Emit only ``prefill``
                # going forward (old YAML setting it still loads via __init__).
                "apply_2d_horizontal_fill",
                "physics_forcing",
            ],
        )
        # Embed the companion physics BoundaryForcing (used as the target density
        # coordinate for density-space BGC interpolation) as an optional sub-item of
        # the BGC block, mirroring how Grids are embedded. The shared "Grid" is
        # dropped since the physics forcing reuses the same grid on reconstruction.
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

        # Reconstruct an optional embedded physics BoundaryForcing, reusing the shared
        # grid. The generic `from_yaml` only deserializes the top-level block, so the
        # nested block's datetimes/paths/source are restored here.
        physics_data = params.pop("physics_forcing", None)
        physics_forcing = None
        if physics_data is not None:
            physics_data = deserialize_forcing_data(physics_data)
            for name in ["source", "bgc_source"]:
                src_dict = physics_data.get(name)
                if src_dict and src_dict.get("grid") is not None:
                    src_dict["grid"] = Grid(**pop_grid_data(src_dict["grid"]))
            physics_forcing = cls(grid=grid, **physics_data, use_dask=use_dask)

        return cls(
            grid=grid,
            **params,
            physics_forcing=physics_forcing,
            use_dask=use_dask,
        )
