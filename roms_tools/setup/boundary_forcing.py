import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from roms_tools import Grid
from roms_tools.datasets.lat_lon_datasets import (
    CESMBGCDataset,
    GLODAPv2BGCDataset,
    GLORYSDataset,
    GLORYSDefaultDataset,
    UnifiedBGCDataset,
    WOABGCDataset,
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
    VerticalRegrid,
    build_lateral_regridder,
    select_source_mask,
)
from roms_tools.setup.bgc_model import (
    BGCModel,
    bgc_model_from_name,
    bgc_model_to_name,
    bgc_variable_info,
    validate_bgc_model,
)
from roms_tools.setup.utils import (
    _CLIMATOLOGY_ONLY_BGC,
    _SELF_DOWNLOADING_BGC,
    RawDataSource,
    add_time_info_to_ds,
    bgc_source_extra_kwargs,
    build_bgc_companions,
    build_bgc_vertical_coords,
    check_and_set_boundaries,
    compute_barotropic_velocity,
    deserialize_forcing_data,
    forwardable_fields,
    from_yaml,
    get_boundary_coords,
    get_roms_tools_version_info,
    get_target_coords,
    get_variable_metadata,
    group_dataset,
    materialize_before_check,
    nan_check_batch,
    pop_grid_data,
    substitute_nans_by_fillvalue,
    to_dict,
    write_to_yaml,
)
from roms_tools.utils import (
    DEFAULT_NETCDF_FORMAT,
    NetCDFFormat,
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


#: Which dataset class implements each ``source["name"]``, per forcing type. This is the
#: single source of truth for *which source names BoundaryForcing supports*: both
#: ``_input_checks`` (at construction) and ``_get_data`` (at load) read it, so the two
#: cannot drift apart and admit a name that later fails to load.
#:
#: Note there is no ``"ROMS"`` entry, for either forcing type. Boundary data taken from a
#: parent ROMS run is the nesting workflow (:mod:`roms_tools.setup.nesting`), not a source
#: here -- it needs the parent's boundary segments mapped onto the child grid rather than
#: a lateral regrid. ``InitialConditions`` *does* accept a ``"ROMS"`` source (a restart
#: file regridded onto the new grid), which is why its own map differs.
_DATASET_MAP: dict[str, dict[str, dict[str, type]]] = {
    "physics": {
        "GLORYS": {
            "external": GLORYSDataset,
            "default": GLORYSDefaultDataset,
        },
    },
    "bgc": {
        "CESM_REGRIDDED": defaultdict(lambda: CESMBGCDataset),
        "UNIFIED": defaultdict(lambda: UnifiedBGCDataset),
        "GLODAP": defaultdict(lambda: GLODAPv2BGCDataset),
        "WOA": defaultdict(lambda: WOABGCDataset),
    },
}

#: BGC source names ``BoundaryForcing`` accepts: the dataset-backed ones above, plus the
#: two derived pseudo-sources that load no dataset at all.
_BGC_SOURCE_NAMES: frozenset[str] = frozenset(_DATASET_MAP["bgc"]) | {
    "constants",
    "ESPER",
}


@dataclass(kw_only=True)
class BoundaryForcingSource:
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

          - "physics": for physical ocean boundary data (T/S/u/v/zeta).
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
    physics_forcing : BoundaryForcingSource, optional
        A physics ``BoundaryForcingSource`` object (``type='physics'``) whose T/S fields
        supply the target density coordinate for BGC tracer interpolation. When None and
        a density method is requested, falls back to depth-based interpolation.
        **Required for an ``ESPER`` bgc source**, where the physics T/S are the
        estimator's own inputs rather than merely a target coordinate; omitting it
        there raises ``ValueError``.


    Examples
    --------
    >>> boundary_forcing = BoundaryForcingSource(
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
    start_time_pad: bool = True
    """If True (default), include one dataset record before start_time so ROMS can
    interpolate at the exact simulation start boundary. If False, select only records
    at or after start_time."""
    end_time_pad: bool = True
    """If True (default), include one dataset record after end_time so ROMS can
    interpolate at the exact simulation end boundary. If False, select only records
    at or before end_time."""
    bgc_interpolation_method: str = "depth"
    """Vertical interpolation method for BGC tracers: ``"depth"``, ``"density"``, or
    ``"density_mld"``."""
    physics_forcing: "BoundaryForcingSource | None" = None
    """Physics BoundaryForcingSource object supplying T/S for density-based BGC interpolation."""
    use_vars: list[str] | None = None
    """Optional down-selection of the BGC variables written from ``source`` (only
    applies when ``type="bgc"``). When set, only these variables are kept (presence-only
    check — a ``ValueError`` is raised if any requested variable is not provided by the
    source). No MARBL derivation is performed here; call
    :meth:`BGCMarbl.process_bgc_fields` on the finished object(s) to complete the set."""

    # `compare=False`: `xr.Dataset.__eq__` returns an elementwise Dataset (truthy in
    # a way that makes `==` comparisons on this dataclass vacuously true/broken)
    # rather than a bool, so equality between two objects of this class is defined
    # over their (comparable) parameters, not their computed datasets.
    ds: xr.Dataset = field(init=False, repr=False, compare=False)
    """An xarray Dataset containing post-processed variables ready for input into
    ROMS."""
    adjust_depth_for_sea_surface_height: bool = field(init=False)
    """Whether to account for sea surface height when computing depth coordinates."""
    ds_depth_coords: xr.Dataset = field(init=False, repr=False, compare=False)
    """An xarray Dataset containing the depth coordinates."""

    def __post_init__(self):
        # Initialize depth coordinates
        self.adjust_depth_for_sea_surface_height = False
        self.ds_depth_coords = xr.Dataset()

        self._resolve_prefill_options()
        self._input_checks()

        target_coords = get_target_coords(self.grid)

        # BGC "constants" source: no dataset to load/regrid. Broadcast the user-supplied
        # values onto each active boundary's rho grid and finish early.
        if self.type == "bgc" and self.source["name"] == "constants":
            self._reject_climatology_on_static_source()
            self.ds = self._process_bgc_constants(target_coords)
            return

        # BGC "ESPER" source: derive tracers from physics_forcing T/S via PyESPER.
        if self.type == "bgc" and self.source["name"] == "ESPER":
            self.ds = self._process_bgc_esper(target_coords)
            return

        data = self._get_data()

        if self.type == "bgc" and "time" not in data.ds.dims:
            self._reject_climatology_on_static_source()

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
            # path, matched to the field type: ``mask`` (scalar-field validity)
            # for tracers and ``zeta``, ``mask_vel`` (velocity validity) for u/v.
            # Reusing these stored 2D fields avoids recomputing a mask from the
            # full (lazy) source series. ``None`` means the source is already
            # NaN-free (e.g. the pre-filled UNIFIED BGC dataset, which carries
            # no mask, or a whole-domain prefill above) so the regridder uses
            # plain bilinear; irrelevant on the scipy path.
            scalar_mask = select_source_mask(
                bdry_data.ds, is_vector=False, use_xesmf=use_xesmf, prefill=prefill
            )
            vector_mask = select_source_mask(
                bdry_data.ds, is_vector=True, use_xesmf=use_xesmf, prefill=prefill
            )

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
                lateral_regrid_vector = build_lateral_regridder(
                    {"lat": lat, "lon": lon}, bdry_data, regrid, vector_mask
                )
                for var_name in filtered_vars:
                    processed_fields[var_name] = lateral_regrid_vector.apply(
                        bdry_data.ds[var_names[var_name]["name"]]
                    )

                if self.adjust_depth_for_sea_surface_height:
                    # Regrid sea surface height ('zeta') onto a 2-cell-wide margin.
                    # This is needed to correctly infer depth coordinates at u- and v-points along the boundary.
                    # 'zeta' is a scalar, so it uses the scalar mask (not the
                    # velocity mask of the vector regridder); build a dedicated
                    # regridder on the same vector-margin target.
                    zeta_vector_regrid = build_lateral_regridder(
                        {"lat": lat, "lon": lon}, bdry_data, regrid, scalar_mask
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
                lateral_regrid = build_lateral_regridder(
                    {"lat": lat, "lon": lon}, bdry_data, regrid, scalar_mask
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
                # Keep only the source's own (raw) tracers, optionally down-selected via
                # ``use_vars``. MARBL derivation/fill is done later by
                # ``BGCMarbl().process_bgc_fields()`` on the finished object(s).
                processed_fields = self._apply_use_vars(processed_fields)

            # Write the boundary data into dataset
            ds = self._write_into_dataset(direction, processed_fields, ds)

        if self.type == "bgc" and "time" not in ds.dims:
            # Static BGC dataset source (no time axis, e.g. GLODAP climatology).
            ds = self._bracket_static_time(ds)

        # Add global information
        ds = self._add_global_metadata(data, ds)

        if self.type == "bgc":
            # Describe the BGC variables actually present model-agnostically (only ALK is
            # NaN-validated), so validation does not trip over ``use_vars``-dropped vars.
            self.variable_info = bgc_variable_info(self._present_bgc_bare_names(ds))

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

    def _process_bgc_constants(self, target_coords) -> xr.Dataset:
        """Build a BGC boundary dataset from a ``constants`` source.

        Broadcasts each ``source["constants"]`` value onto every active boundary's
        rho grid (depth x horizontal). Static sources carry no time dimension, so the
        result is expanded to two records bracketing ``start_time``/``end_time`` (one
        if they coincide) — ROMS interpolates linearly between boundary records, so
        this yields a constant-in-time boundary condition. MARBL derivation/fill is not
        performed here; call :meth:`BGCMarbl.process_bgc_fields` on the finished object.
        """
        if self.start_time is None or self.end_time is None:
            raise ValueError(
                "A 'constants' BGC source requires `start_time` and `end_time` so the "
                "static field can be bracketed with boundary time records."
            )
        self._set_boundary_info()

        ds = xr.Dataset()
        for direction, is_enabled in self._active_boundaries.items():
            if not is_enabled:
                continue
            # zeta = 0: SSH is not applicable to BGC fields. Precompute both depth
            # types so plotting always works.
            self._get_depth_coordinates(0, direction, "rho", "layer")
            self._get_depth_coordinates(0, direction, "rho", "interface")

            template = self.ds_depth_coords[f"layer_depth_rho_{direction}"]
            # Validated as a mapping in __post_init__; `RawDataSource`'s value
            # union is too wide to express that, so narrow it here.
            constants = cast(dict[str, float], self.source["constants"])
            fields = {
                var: xr.full_like(template, float(val))
                for var, val in constants.items()
            }
            fields = self._apply_use_vars(fields)
            for var_name in fields:
                fields[var_name] = transpose_dimensions(fields[var_name])
            ds = self._write_into_dataset(direction, fields, ds)

        ds = self._bracket_static_time(ds)

        ds = self._add_global_metadata(None, ds, climatology=False)

        self.variable_info = bgc_variable_info(self._present_bgc_bare_names(ds))

        if not self.bypass_validation:
            self._validate(ds)

        for var_name in ds.data_vars:
            ds[var_name] = substitute_nans_by_fillvalue(ds[var_name])

        return ds

    def _reject_climatology_on_static_source(self) -> None:
        """Reject ``climatology=True`` on a BGC source that carries no time axis.

        A static source (``constants``, or an observational climatology such as GLODAP
        that ships no time dimension) is given the two bracketing records ROMS needs by
        :meth:`_bracket_static_time` -- it is never cycled over a twelve-month axis.
        Claiming ``climatology=True`` therefore describes an axis that does not exist,
        and used to fail far downstream: ``add_time_info_to_ds`` builds a 12-element
        ``month`` coordinate and ``assign_coords`` raises an ``AlignmentError`` about
        "conflicting dimension sizes: {2, 12}", naming neither the source nor the flag.

        The field is constant in time either way, so dropping the flag loses nothing.
        """
        if not self.source.get("climatology"):
            return
        raise ValueError(
            f"BGC source {self.source['name']!r} carries no time axis, so it cannot "
            "also be a climatology: it is bracketed to `start_time`/`end_time` rather "
            "than cycled over twelve months. Remove `'climatology': True` from this "
            "source -- the field is constant in time either way."
        )

    def _bracket_static_time(self, ds: xr.Dataset) -> xr.Dataset:
        """Give a static (no-time) BGC dataset the two bracketing time records ROMS needs.

        Static sources (a ``constants`` source, or an observational climatology such as
        GLODAP that carries no time axis) produce no ``time`` dimension. ROMS interpolates
        linearly between boundary records, so repeating the field at ``start_time`` and
        ``end_time`` (one record if they coincide) yields a constant-in-time boundary
        condition. A dataset that already has a ``time`` dim is returned unchanged.
        """
        if "time" in ds.dims:
            return ds
        if self.start_time is None or self.end_time is None:
            raise ValueError(
                "A static BGC source (constants or a no-time climatology) requires "
                "`start_time` and `end_time` so the field can be bracketed with "
                "boundary time records."
            )
        t0 = np.datetime64(self.start_time, "ns")
        t1 = np.datetime64(self.end_time, "ns")
        time_vals = [t0] if t0 == t1 else [t0, t1]
        return ds.expand_dims({"time": time_vals}, axis=0)

    def _process_bgc_esper(self, target_coords) -> xr.Dataset:
        """Build a BGC boundary dataset from an ``ESPER`` source.

        Derives BGC tracers from the companion physics ``physics_forcing`` T/S (already on
        the ROMS grid) via PyESPER -- no dataset load, no lateral/vertical regridding. The
        estimates inherit ``physics_forcing``'s time axis, so the result is a genuine
        time-varying boundary condition (not a static bracket). Lazy when the physics T/S
        are dask-backed. MARBL derivation/fill is not performed here; call
        :meth:`BGCMarbl.process_bgc_fields` on the finished object.
        """
        from roms_tools.setup.esper import (
            ESPER_SUPPORTED_VARS,
            _decimal_year,
            estimate_bgc_fields,
        )

        pf = self.physics_forcing
        if pf is None:  # pragma: no cover - enforced in __post_init__
            raise ValueError(
                "An ESPER BGC BoundaryForcingSource requires `physics_forcing`."
            )
        self._set_boundary_info()
        climatology = bool(self.source.get("climatology", False))

        # Only ask PyESPER to estimate what was actually requested; unsupported
        # requested variables are still caught below by `_apply_use_vars`'s
        # presence check (which runs against the full, unfiltered `use_vars`).
        roms_variables = (
            [v for v in self.use_vars if v in ESPER_SUPPORTED_VARS]
            if self.use_vars is not None
            else ESPER_SUPPORTED_VARS
        )

        ds = xr.Dataset()
        for direction, is_enabled in self._active_boundaries.items():
            if not is_enabled:
                continue
            # zeta = 0: SSH is not applicable to BGC fields.
            self._get_depth_coordinates(0, direction, "rho", "layer")
            self._get_depth_coordinates(0, direction, "rho", "interface")
            depth = self.ds_depth_coords[f"layer_depth_rho_{direction}"]

            temp = pf.ds[f"temp_{direction}"]
            salt = pf.ds[f"salt_{direction}"]
            # Physics BC vars carry a "bry_time" dim with an "abs_time" datetime coord.
            # Swap to the datetime view named "time" so the shared _add_global_metadata
            # can rebuild bry_time from it (as the dataset/constants paths do).
            if "abs_time" in temp.coords:
                temp = temp.swap_dims({"bry_time": "abs_time"}).rename(
                    {"abs_time": "time"}
                )
                salt = salt.swap_dims({"bry_time": "abs_time"}).rename(
                    {"abs_time": "time"}
                )
            est_dates = _decimal_year(temp["time"]) if "time" in temp.dims else None

            lon = target_coords["lon"].isel(**self.bdry_coords["rho"][direction])
            lat = target_coords["lat"].isel(**self.bdry_coords["rho"][direction])

            fields = estimate_bgc_fields(
                temp,
                salt,
                lon,
                lat,
                depth,
                source=self.source,
                roms_variables=roms_variables,
                est_dates=est_dates,
            )
            fields = self._apply_use_vars(fields)
            for var_name in fields:
                fields[var_name] = transpose_dimensions(fields[var_name])
            ds = self._write_into_dataset(direction, fields, ds)

        # ds carries a "time" dim inherited from physics_forcing; convert to bry_time.
        ds = self._add_global_metadata(None, ds, climatology=climatology)

        self.variable_info = bgc_variable_info(self._present_bgc_bare_names(ds))

        if not self.bypass_validation:
            self._validate(ds)

        for var_name in ds.data_vars:
            ds[var_name] = substitute_nans_by_fillvalue(ds[var_name])

        return ds

    def _apply_use_vars(self, fields: dict) -> dict:
        """Down-select ``fields`` (bare BGC var name -> DataArray) to ``self.use_vars``.

        Presence-only check: raises ``ValueError`` if any requested variable is not
        present in this source's own (regridded) fields.
        """
        if self.use_vars is None:
            return fields
        requested = list(self.use_vars)
        missing = [v for v in requested if v not in fields]
        if missing:
            raise ValueError(
                f"use_vars requested variable(s) not present in the "
                f"'{self.source['name']}' BGC source: {sorted(missing)}. "
                f"Available here: {sorted(fields)}."
            )
        return {v: fields[v] for v in requested}

    def _present_bgc_bare_names(self, ds) -> set[str]:
        """Bare BGC variable names present in ``ds`` across active boundaries.

        ``ds`` stores variables suffixed by direction (``PO4_south``); this strips the
        suffix of each active boundary so the model-agnostic metadata can be built.
        """
        active = [d for d, on in self._active_boundaries.items() if on]
        bare: set[str] = set()
        for v in ds.data_vars:
            name = str(v)
            for d in active:
                if name.endswith(f"_{d}"):
                    bare.add(name[: -(len(d) + 1)])
                    break
        return bare

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

        if self.type == "physics" and self.use_vars is not None:
            raise ValueError("`use_vars` only applies when `type='bgc'`.")

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

        name = self.source["name"]
        if self.type == "bgc":
            if name == "constants":
                if not self.source.get("constants"):
                    raise ValueError(
                        "For source={'name': 'constants', ...} you must provide a "
                        "non-empty 'constants' mapping."
                    )
            elif name == "ESPER":
                from roms_tools.setup.esper import validate_esper_source

                validate_esper_source(self.source)
                if self.physics_forcing is None:
                    raise ValueError(
                        "An ESPER BGC BoundaryForcingSource requires `physics_forcing` (a "
                        "physics BoundaryForcingSource supplying T/S on the ROMS grid)."
                    )
            elif name not in _BGC_SOURCE_NAMES:
                raise ValueError(
                    f"Unknown BGC source name '{name}' for boundary forcing. Valid "
                    f"options: {sorted(_BGC_SOURCE_NAMES)}. (Boundary data from a "
                    "parent ROMS run is the nesting workflow, not a BGC source; see "
                    "roms_tools.setup.nesting.)"
                )
            elif "path" not in self.source and name not in _SELF_DOWNLOADING_BGC:
                raise ValueError("`source` must include a 'path'.")
        else:
            if "path" not in self.source:
                if name != "GLORYS":
                    raise ValueError("`source` must include a 'path'.")
                self.source["path"] = GLORYSDefaultDataset.dataset_name

        # Assign default value. Sources that only ever exist as a 12-month
        # climatology default to True, since False fails later with a confusing
        # message about integer time values.
        self.source["climatology"] = self.source.get(
            "climatology", name in _CLIMATOLOGY_ONLY_BGC
        )

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
        dataset_map = _DATASET_MAP

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

        if isinstance(self.source.get("path"), bool):
            raise ValueError('source["path"] cannot be a boolean here')

        return data_type(
            # A self-downloading source (see _SELF_DOWNLOADING_BGC) may carry no
            # "path"; it fetches its own data when handed a falsy filename.
            filename=self.source.get("path", ""),
            start_time=self.start_time,
            end_time=self.end_time,
            climatology=self.source["climatology"],  # type: ignore[arg-type]
            use_dask=self.use_dask,
            chunks=self.chunks,
            initial_slice_bounds=self.initial_slice_bounds,
            start_time_pad=self.start_time_pad,
            end_time_pad=self.end_time_pad,
            **bgc_source_extra_kwargs(self.source),
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

    def _add_global_metadata(self, data, ds=None, climatology=None):
        if ds is None:
            ds = xr.Dataset()
        ds.attrs["title"] = "ROMS boundary forcing file created by ROMS-Tools"
        # Include the version of roms-tools -- both the semantic version (which can
        # go stale relative to an editable install's actual source, see
        # get_roms_tools_version_info's docstring) and, when available, the exact
        # git commit that produced this file.
        version_info = get_roms_tools_version_info()
        ds.attrs["roms_tools_version"] = version_info["roms_tools_version"]
        # netCDF attrs cannot hold `None`; omit the attr entirely rather than
        # writing the misleading string "None" when no git commit is known (e.g.
        # a real pip/conda install with no `.git` directory).
        if version_info["roms_tools_git_commit"] is not None:
            ds.attrs["roms_tools_git_commit"] = version_info["roms_tools_git_commit"]
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

        # ``data`` is None for a "constants" source; fall back to the explicit flag.
        clim = data.climatology if data is not None else bool(climatology)
        ds, bry_time = add_time_info_to_ds(ds, self.model_reference_date, clim)

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
        # Materialize every variable (every direction) sharing this source's expensive
        # computation -- not just ALK, which is all bgc_variable_info() actually flags
        # for the NaN check below -- so a later .save() on this same `ds` reuses these
        # values instead of recomputing them. Must happen before any check view (e.g.
        # `.isel(bry_time=0)`) is built from `ds`; see materialize_before_check's
        # docstring for why, and for how the cost trades off -- the ESPER chunk plan
        # cuts multi-month boundary runs along time, so on those this buys a single
        # compute of the full series at the price of holding it resident, rather
        # than being close to free as it is when a dimension collapses to one chunk.
        materialize_before_check(
            ds,
            [
                f"{var_name}_{direction}"
                for var_name in self.variable_info
                for direction, is_enabled in self.boundaries.items()
                if is_enabled
            ],
            materialize=self._is_esper_source,
        )

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
            The name of the boundary forcing field to plot. Format:

            "{base_var_name}_{direction}" ,

            where {base_var_name} is a physical, BGC, or other boundary tracer name,
            and {direction} is one of ["south", "east", "north", "west"].

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

    @property
    def _active_boundaries(self) -> dict[str, bool]:
        """``boundaries`` narrowed to non-optional.

        The field is declared optional so callers may omit it, but ``__post_init__``
        always replaces it with the result of
        :func:`roms_tools.setup.utils.check_and_set_boundaries`, which never returns
        ``None``. Annotated methods go through this accessor so that invariant is
        visible to a type checker.
        """
        if self.boundaries is None:  # pragma: no cover - set in __post_init__
            raise ValueError(
                "`boundaries` is unset; it is populated during __post_init__."
            )
        return self.boundaries

    @property
    def _is_esper_source(self) -> bool:
        """True when this source derives its tracers via PyESPER (the ESPER source).

        All of an ESPER source's ``use_vars`` come out of one shared, expensive lazy
        computation per chunk, which is why validation materialises them once and
        caches (see :func:`roms_tools.setup.utils.materialize_before_check`).
        """
        return self.type == "bgc" and self.source.get("name") == "ESPER"

    def save(
        self,
        filepath: str | Path,
        group: bool = True,
        format: NetCDFFormat = DEFAULT_NETCDF_FORMAT,
        serialize_dask: bool | None = None,
    ) -> list[Path]:
        """Save the boundary forcing fields to one or more NetCDF files.

        This method saves the dataset to disk as either a single NetCDF file or multiple files, depending on the `group` parameter.
        If `group` is `True`, the dataset is divided into subsets (e.g., monthly or yearly) based on the temporal frequency
        of the data, and each subset is saved to a separate file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The base path and filename for the output file(s). If `group` is `True`, the filenames will include additional
            time-based information (e.g., year or month) to distinguish the subsets.
        group : bool, optional
            Whether to divide the dataset into multiple files based on temporal frequency. Defaults to `True`.
        format : {"NETCDF4", "NETCDF3_CLASSIC", "NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA"}, optional
            NetCDF file format. Defaults to ``"NETCDF4"``.
        serialize_dask : bool, optional
            See :func:`roms_tools.utils.save_datasets`. Defaults to ``None``,
            which resolves to ``False`` -- the ordinary concurrent write under
            the ambient dask scheduler. Pass ``True`` to force the serialized,
            one-task-at-a-time write instead: a manual tool for low-memory
            machines (it bounds peak memory to a single chunk's footprint,
            which plain dask's threaded scheduler cannot guarantee) and for
            troubleshooting scheduler-dependent failures.

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

        if serialize_dask is None:
            serialize_dask = False

        saved_filenames = save_datasets(
            dataset_list,
            output_filenames,
            use_dask=self.use_dask,
            format=format,
            serialize_dask=serialize_dask,
            # ESPER's chunks are few, large and uneven, and it prints per chunk --
            # the dask bar is misleading and collides with those prints. Cosmetic
            # only; scheduling is untouched. See save_datasets' `show_progress`.
            show_progress=not self._is_esper_source,
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
        # Embed the companion physics BoundaryForcingSource (used as the target density
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
            forcing_dict["BoundaryForcingSource"]["physics_forcing"] = physics_dict[
                "BoundaryForcingSource"
            ]
        write_to_yaml(forcing_dict, filepath)

    @classmethod
    def from_yaml(
        cls,
        filepath: str | Path,
        use_dask: bool = False,
    ) -> "BoundaryForcingSource":
        """Create an instance of the BoundaryForcingSource class from a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file from which the parameters will be read.
        use_dask: bool, optional
            Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.

        Returns
        -------
        BoundaryForcingSource
            An instance of the BoundaryForcingSource class.
        """
        filepath = Path(filepath)

        grid = Grid.from_yaml(filepath)
        params = from_yaml(cls, filepath)

        # Reconstruct an optional embedded physics BoundaryForcingSource, reusing the shared
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


@dataclass(kw_only=True)
class BoundaryForcing:
    """Wrapper class that can initialize and process multiple constituent
    :class:`BoundaryForcingSource` objects.  This class is the intended
    interface for generating and writing ROMS boundary forcing files, and
    its use is fully supported by the ``to_yaml()``/``from_yaml()``
    conventions.

    Internally builds one ``type="physics"`` :class:`BoundaryForcingSource` plus one
    ``type="bgc"`` :class:`BoundaryForcingSource` per ``bgc_sources`` item (each
    wired with ``physics_forcing=`` to reuse the physics object's T/S -- see
    :class:`BoundaryForcingSource`'s own docstring for that mechanism), and
    completes the BGC tracer set via ``bgc_model().process_bgc_fields()``.

    Unlike :class:`InitialConditions`, there is **no merge into one dataset** here: ROMS's
    ``frcfiles`` namelist key accepts a list, so each BGC source is written to its
    own file, like the physics object.

    The constituent `BoundaryForcingSource` objects are  public and
    documented.  They can be accessed as ``.physics`` and ``.bgc[i]``,
    each a :class:`BoundaryForcingSource` carrying its own ``.ds``
    xarray DataSet and ``.plot()`` capability.

    Parameters
    ----------
    grid, start_time, end_time, boundaries, prefill, prefill_kwargs, regrid_method,
    extrap_method, extrap_kwargs, apply_2d_horizontal_fill, model_reference_date,
    use_dask, chunks, initial_slice_bounds, bypass_validation, start_time_pad,
    end_time_pad
        Forwarded to the internal physics :class:`BoundaryForcingSource` (and, for
        the fields that apply to a lat/lon bgc source too, to each bgc companion).
        See :class:`BoundaryForcingSource` for the full description of each.
    source : RawDataSource
        The physical boundary-forcing dataset. Required.
    bgc_sources : list[dict], optional
        Zero or more BGC sources, one dict per source:
        ``{"source": RawDataSource, "use_vars": list[str] | None,
        "bgc_interpolation_method": str | None}``. A per-item
        ``bgc_interpolation_method`` overrides the wrapper-level default below.
    bgc_model : type[BGCModel], optional
        The :class:`~roms_tools.setup.bgc_model.BGCModel` subclass (e.g.
        :class:`~roms_tools.setup.bgc_model.BGCMarbl`) used to complete the tracer
        set -- passed as the class itself, since every ``BGCModel`` is instantiated
        with zero arguments. Required whenever ``bgc_sources`` is given.
    bgc_interpolation_method : str, optional
        Wrapper-level default vertical interpolation method for BGC tracers
        (``"depth"``, ``"density"``, or ``"density_mld"``); see
        :class:`BoundaryForcingSource` for the full description. Overridden
        per-source by that source's own ``bgc_interpolation_method`` entry.

    Examples
    --------
    >>> bf = BoundaryForcing(
    ...     grid=grid,
    ...     start_time=datetime(2013, 1, 1),
    ...     end_time=datetime(2013, 2, 1),
    ...     source={"name": "GLORYS", "path": glorys_path},
    ...     bgc_sources=[
    ...         {
    ...             "source": {"name": "ESPER", "path": pyesper_path},
    ...             "use_vars": ["NO3", "PO4", "SiO3", "ALK", "DIC", "O2"],
    ...         },
    ...         {
    ...             "source": {
    ...                 "name": "UNIFIED",
    ...                 "path": unified_path,
    ...                 "climatology": True,
    ...             },
    ...             "use_vars": ["Fe", "CHL"],
    ...             "bgc_interpolation_method": "density_mld",
    ...         },
    ...     ],
    ...     bgc_model=BGCMarbl,
    ... )
    >>> physics_paths, bgc_paths = bf.save(
    ...     "boundary-physics.nc", ["boundary-bgc-esper.nc", "boundary-bgc-unified.nc"]
    ... )
    """

    grid: Grid
    """Object representing the grid information."""
    start_time: datetime | None = None
    """The start time of the desired boundary forcing data."""
    end_time: datetime | None = None
    """The end time of the desired boundary forcing data."""
    boundaries: dict[str, bool] | None = None
    """Dictionary specifying which boundaries are forced (south, east, north, west)."""
    source: RawDataSource
    """Dictionary specifying the source of the physical boundary forcing data."""
    bgc_sources: list[dict] | None = None
    """Zero or more BGC sources; see the class docstring for the per-item shape."""
    bgc_model: type[BGCModel] | None = None
    """The BGCModel subclass (e.g. BGCMarbl) used to complete the tracer set.
    Required whenever ``bgc_sources`` is given."""
    prefill: str | None = None
    """Source-side fill applied before regridding; see :class:`BoundaryForcingSource`
    for the full description."""
    prefill_kwargs: dict | None = None
    """Method-specific options for ``prefill``."""
    regrid_method: str | None = None
    """Horizontal regrid engine, chosen independently of ``prefill``."""
    extrap_method: str | None = None
    """xESMF destination extrapolation used on the default no-prefill path."""
    extrap_kwargs: dict | None = None
    """Method-specific options for ``extrap_method``."""
    apply_2d_horizontal_fill: bool | None = None
    """Deprecated alias for ``prefill``; see :class:`BoundaryForcingSource`."""
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
    start_time_pad: bool = True
    """If True (default), include one dataset record before start_time so ROMS can
    interpolate at the exact simulation start boundary."""
    end_time_pad: bool = True
    """If True (default), include one dataset record after end_time so ROMS can
    interpolate at the exact simulation end boundary."""
    bgc_interpolation_method: str = "depth"
    """Wrapper-level default vertical interpolation method for BGC tracers;
    overridden per-source by that source's own entry in ``bgc_sources``."""

    # `compare=False`: these hold constituent `BoundaryForcingSource` objects,
    # each carrying a computed `xr.Dataset` -- see the `ds` field's own comment
    # in `BoundaryForcingSource` for why dataclass equality must exclude them.
    physics: BoundaryForcingSource = field(init=False, repr=False, compare=False)
    """The internal physics-only BoundaryForcingSource object. Exposed for advanced
    use (e.g. ``bf.physics.plot(...)``)."""
    bgc: list[BoundaryForcingSource] = field(init=False, repr=False, compare=False)
    """The internal bgc-only BoundaryForcingSource objects, one per ``bgc_sources``
    item, in order. Exposed for advanced use."""

    def __post_init__(self):
        if self.bgc_interpolation_method not in BGC_INTERPOLATION_METHODS:
            raise ValueError(
                f"`bgc_interpolation_method` must be one of "
                f"{BGC_INTERPOLATION_METHODS}, got {self.bgc_interpolation_method!r}."
            )
        # Validate `bgc_model` BEFORE building the physics source: a wrong value
        # (None, an instance, an unrelated class) should fail here, not after
        # the physics regrid has already been paid for.
        bgc_sources = self.bgc_sources or []
        if bgc_sources:
            if self.bgc_model is None:
                raise ValueError(
                    "`bgc_model` is required when `bgc_sources` is provided "
                    "(e.g. `bgc_model=rt.BGCMarbl`)."
                )
            validate_bgc_model(self.bgc_model)
        # Every constructor argument this wrapper shares by name with
        # BoundaryForcingSource is forwarded to the physics object and to each bgc
        # companion -- derived from the two dataclasses (see forwardable_fields)
        # instead of hand-listed, so a new shared field cannot be silently
        # dropped. Nothing needs excluding here: `source` is overridden per
        # companion by build_bgc_companions, and `bgc_interpolation_method` is
        # validated-but-unused on a physics source.
        shared_kwargs = {
            name: getattr(self, name)
            for name in forwardable_fields(type(self), BoundaryForcingSource)
        }
        self.physics = BoundaryForcingSource(**{**shared_kwargs, "type": "physics"})

        if bgc_sources:
            self.bgc = build_bgc_companions(
                BoundaryForcingSource,
                self.grid,
                self.physics,
                bgc_sources,
                shared_kwargs,
                type_="bgc",
            )
            self.bgc_model().process_bgc_fields(self.bgc)
        else:
            self.bgc = []

    def save(
        self,
        physics_filepath: str | Path,
        bgc_filepaths: list[str | Path] | None = None,
        group: bool = True,
        format: NetCDFFormat = DEFAULT_NETCDF_FORMAT,
        serialize_dask: bool | None = None,
    ) -> tuple[list[Path], list[list[Path]]]:
        """Save the physics object and every BGC source to its own file.

        BGC tracer completion (derivation/overlap-check/constant-fill) already
        happened once, in ``__post_init__`` via ``bgc_model().process_bgc_fields()``
        -- this method only writes the already-processed objects to disk, honouring
        ``group``/``format`` exactly as :meth:`BoundaryForcingSource.save` does for
        the physics object.

        Parameters
        ----------
        physics_filepath : Union[str, Path]
            Base path/filename for the physics boundary forcing file(s).
        bgc_filepaths : list[str or Path], optional
            One filename per entry in ``self.bgc`` (i.e. per ``bgc_sources`` item),
            in the same order. Required (and length-checked) whenever ``bgc_sources``
            was provided; omit when there are no BGC sources.
        group : bool, optional
            Whether to divide each dataset into multiple files based on temporal
            frequency (e.g. monthly); see :meth:`BoundaryForcingSource.save`.
            Defaults to `True`.
        format : {"NETCDF4", "NETCDF3_CLASSIC", "NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA"}, optional
            NetCDF file format. Defaults to ``"NETCDF4"``.
        serialize_dask : bool, optional
            See :func:`roms_tools.utils.save_datasets`. Defaults to ``None``,
            which resolves to ``False`` on every constituent's own save (the
            ordinary concurrent write). Pass ``True`` here to force the
            serialized, one-task-at-a-time write onto every one of them -- a
            manual low-memory / troubleshooting tool.

        Returns
        -------
        tuple[list[Path], list[list[Path]]]
            ``(physics_paths, bgc_paths)``. ``physics_paths`` is the actual saved
            path(s) for the physics file. ``bgc_paths`` is the actual saved path(s)
            for each bgc source, in the same order as ``self.bgc`` -- one
            ``list[Path]`` per source (as returned by its own
            :meth:`BoundaryForcingSource.save`), so a ``group=True`` split into
            multiple files is reflected directly rather than needing rediscovery on
            disk.
        """
        physics_paths = self.physics.save(
            physics_filepath,
            group=group,
            format=format,
            serialize_dask=serialize_dask,
        )

        requested_paths: list[str | Path] = list(bgc_filepaths) if bgc_filepaths else []
        bgc_paths: list[list[Path]] = []
        if self.bgc:
            if len(requested_paths) != len(self.bgc):
                raise ValueError(
                    "`bgc_filepaths` must provide one path per bgc source "
                    f"(got {len(requested_paths)} path(s) for {len(self.bgc)} "
                    "source(s))."
                )
            bgc_paths = [
                b.save(p, group=group, format=format, serialize_dask=serialize_dask)
                for b, p in zip(self.bgc, requested_paths)
            ]
        elif requested_paths:
            raise ValueError(
                "`bgc_filepaths` was given but this object has no bgc sources."
            )

        return physics_paths, bgc_paths

    def to_yaml(self, filepath: str | Path) -> None:
        """Export the parameters of the class to a YAML file, including the version
        of roms-tools.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file where the parameters will be saved.
        """
        forcing_dict = to_dict(
            self, exclude=["physics", "bgc", "bgc_model", "use_dask"]
        )
        # `bgc_model` is a class, not a plain value -- to_dict's generic
        # serialization has no way to make that YAML-safe, so it's excluded above
        # and handled explicitly via the name registry (see bgc_model.py).
        forcing_dict["BoundaryForcing"]["bgc_model"] = bgc_model_to_name(self.bgc_model)
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
        use_dask : bool, optional
            Indicates whether to use dask for processing. Defaults to False.

        Returns
        -------
        BoundaryForcing
            An instance of the BoundaryForcing class.

        Raises
        ------
        ValueError
            If the YAML file was saved by the pre-5.0 single-source
            `BoundaryForcingSource` class (identifiable by a `type` or
            `physics_forcing` key in the loaded block) rather than this wrapper
            class.
        """
        filepath = Path(filepath)
        grid = Grid.from_yaml(filepath)
        params = from_yaml(cls, filepath)

        # `BoundaryForcing`'s own YAML block never carries `type`/`physics_forcing`
        # -- those are `BoundaryForcingSource` (the pre-5.0, single-source class)
        # fields. A file saved by that legacy class also keys its block
        # "BoundaryForcing" (the class was renamed when this wrapper was
        # introduced), so `from_yaml(cls, filepath)` above finds it and these
        # keys land in `params` -- which would otherwise fail deep inside the
        # dataclass constructor as an opaque `TypeError` for an unexpected
        # keyword argument. Catch it here with a message that names the cause.
        legacy_keys = {"type", "physics_forcing"} & set(params)
        if legacy_keys:
            raise ValueError(
                f"This YAML file was saved by `BoundaryForcingSource` (the "
                f"pre-5.0 single-source class), not `BoundaryForcing` -- found "
                f"legacy key(s) {sorted(legacy_keys)}. Load it with "
                "`BoundaryForcingSource.from_yaml(...)` instead, or re-save it "
                "with the current `BoundaryForcing` wrapper class."
            )

        params["bgc_model"] = bgc_model_from_name(params.get("bgc_model"))

        # Deserialize nested grids: the top-level source, and each
        # bgc_sources[i]["source"] (e.g. a "ROMS" restart bgc source).
        src_dict = params.get("source")
        if src_dict and src_dict.get("grid") is not None:
            src_dict["grid"] = Grid(**pop_grid_data(src_dict["grid"]))
        for item in params.get("bgc_sources") or []:
            src_dict = item.get("source")
            if src_dict and src_dict.get("grid") is not None:
                src_dict["grid"] = Grid(**pop_grid_data(src_dict["grid"]))

        return cls(grid=grid, **params, use_dask=use_dask)
