import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from roms_tools import Grid
from roms_tools.datasets.lat_lon_datasets import (
    CESMBGCDataset,
    GLODAPv2BGCDataset,
    GLORYSDataset,
    GLORYSDefaultDataset,
    LatLonDataset,
    UnifiedBGCDataset,
    WOABGCDataset,
)
from roms_tools.datasets.roms_dataset import ROMSDataset, choose_subdomain
from roms_tools.plot import plot
from roms_tools.processing_methods import RegridConfig, _xesmf_available
from roms_tools.regrid import (
    LateralRegridFromROMS,
    VerticalRegrid,
    build_lateral_regridder,
    select_source_mask,
)
from roms_tools.setup.bgc_model import (
    BGCModel,
    bgc_model_from_name,
    bgc_model_to_name,
    bgc_variable_info,
)
from roms_tools.setup.utils import (
    _CLIMATOLOGY_ONLY_BGC,
    _SELF_DOWNLOADING_BGC,
    BGC_INTERPOLATION_METHODS,
    RawDataSource,
    apply_scipy_fallback_fill,
    apply_source_prefill,
    bgc_source_extra_kwargs,
    build_bgc_companions,
    build_bgc_vertical_coords,
    check_source_coverage,
    compute_barotropic_velocity,
    deserialize_forcing_data,
    from_yaml,
    get_roms_tools_version_info,
    get_target_coords,
    get_variable_metadata,
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
from roms_tools.vertical_coordinate import (
    compute_depth,
)

#: Which dataset class implements each ``source["name"]``, per forcing type. Single
#: source of truth for *which source names InitialConditions supports*: both
#: ``_input_checks`` (at construction) and ``_get_data`` (at load) read it, so the two
#: cannot drift apart and admit a name that later fails to load.
#:
#: Unlike ``BoundaryForcing``, ``"ROMS"`` is valid here for both forcing types -- a
#: restart file from another run, regridded onto the new grid.
_DATASET_MAP: dict[str, dict[str, dict[str, type[LatLonDataset | ROMSDataset]]]] = {
    "physics": {
        "GLORYS": {
            "external": GLORYSDataset,
            "default": GLORYSDefaultDataset,
        },
        "ROMS": defaultdict(lambda: ROMSDataset),
    },
    "bgc": {
        "CESM_REGRIDDED": defaultdict(lambda: CESMBGCDataset),
        "UNIFIED": defaultdict(lambda: UnifiedBGCDataset),
        "GLODAP": defaultdict(lambda: GLODAPv2BGCDataset),
        "WOA": defaultdict(lambda: WOABGCDataset),
        "ROMS": defaultdict(lambda: ROMSDataset),
    },
}

#: BGC source names ``InitialConditions`` accepts: the dataset-backed ones above, plus
#: the two derived pseudo-sources that load no dataset at all.
_BGC_SOURCE_NAMES: frozenset[str] = frozenset(_DATASET_MAP["bgc"]) | {
    "constants",
    "ESPER",
}


@dataclass(kw_only=True)
class InitialConditionsSource:
    """Represents initial conditions for ROMS, including physical and biogeochemical
    data. This class will not typically be called by a user.  Instead, multiple
    InitialConditionsSource objects are created when interacting with the
    user-facing InitialConditions class, from which a final  ROMS input file
    can be saved.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information used for the model.
    ini_time : datetime
        The date and time at which the initial conditions are set.
        If no exact match is found, the closest time entry to `ini_time` within the time range [ini_time, ini_time + 24 hours] is selected.
    type : {"physics", "bgc"}, optional
        Whether this object processes the physical initial-condition dataset
        (``"physics"``, the default) or a biogeochemical dataset (``"bgc"``).
        Mirrors :class:`~roms_tools.setup.boundary_forcing.BoundaryForcingSource`'s
        ``type`` exactly. ``type="bgc"`` requires ``physics_forcing`` -- unlike
        boundary forcing, there is no standalone bgc-only mode here (this object's
        own bgc processing always aligns its output onto the physics time
        coordinate, so it needs a physics companion regardless of whether the
        chosen ``bgc_interpolation_method`` itself needs T/S).
    source : RawDataSource, optional

        Dictionary specifying the source: the physical initial-condition dataset
        when ``type="physics"``, or the BGC dataset when ``type="bgc"``. Keys
        include:

          - "name" (str): Name of the data source (e.g., "GLORYS" for physics;
            "CESM_REGRIDDED", "UNIFIED", "GLODAP", "constants", or "ESPER" for bgc).
          - "path" (Union[str, Path, List[Union[str, Path]]]): The path to the raw data file(s). This can be:

            - A single string (with or without wildcards).
            - A single Path object.
            - A list of strings or Path objects.
            If omitted (physics/GLORYS only), the data will be streamed via the
            Copernicus Marine Toolkit. Note: streaming is currently not recommended
            due to performance limitations.
          - "climatology" (bool): Indicates if the data is climatology data. Defaults to False.

    physics_forcing : InitialConditionsSource, optional
        Required when ``type="bgc"``: an already-built, ``type="physics"``
        ``InitialConditionsSource`` object supplying temperature/salinity for this
        object's BGC processing.

          - This object's own physics regridding (u, v, zeta, w, barotropic
            velocities, temp, salt, ...) is skipped entirely, so multiple BGC
            sources for one initial-condition snapshot no longer each pay for a
            redundant full-physics regrid.
          - ``ESPER`` derivation and ``"density"``/``"density_mld"`` vertical
            interpolation read ``physics_forcing.ds["temp"]``/``["salt"]`` and
            ``physics_forcing.ds_depth_coords`` directly, lazily (no ``.load()``/
            ``.compute()``), instead of this object's own (nonexistent) physics pass.
          - The resulting ``ds`` carries only this object's own BGC tracers -- combine
            it with ``physics_forcing.ds`` (e.g. via ``xr.merge``/``merge()``) to get
            a complete initial-conditions dataset.

        Mirrors :class:`~roms_tools.setup.boundary_forcing.BoundaryForcingSource`'s
        ``physics_forcing`` pattern.
    model_reference_date : datetime, optional
        The reference date for the model. Defaults to January 1, 2000.
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.
    chunks : dict[str, int], optional
        Dictionary specifying chunk sizes for dask dimensions, e.g., ``{"latitude": 100, "longitude": 100}``.
        If provided, these chunks override the default chunking scheme when ``use_dask=True``. Dimensions must
        match the underlying dataset, e.g. for ROMS restart files, the dimensions must be "eta_rho", etc.
        Defaults to None (default chunking is used).
    initial_slice_bounds : dict, optional
        Optional horizontal subset to apply when loading with dask. Only Geographic bounds are supported:
         ``{"latitude": (min_lat, max_lat), "longitude": (min_lon, max_lon)}`` in degrees. The
         bounds are applied to the dataset before reading the underlying datasets to reduce memory usage.
         Not used for ROMS restart or other datasets sources.
    allow_flex_time: bool, optional
        Controls how strictly `ini_time` is handled:

        - If False (default): requires an exact match to `ini_time`. Raises a ValueError if no match exists.
        - If True: allows a +24h search window after `ini_time` and selects the closest available
          time entry within that window. Raises a ValueError if none are found.

    bypass_validation: bool, optional
        Indicates whether to skip validation checks in the processed data. When set to True,
        the validation process that ensures no NaN values exist at wet points
        in the processed dataset is bypassed. Defaults to False.
    bgc_interpolation_method : str, optional
        Vertical interpolation method for BGC tracers. One of:

        - ``"depth"`` (default): linear interpolation in depth.
        - ``"density"``: linear interpolation in potential-density (isopycnal) space,
          preserving water-mass properties. Density is computed from temperature and
          salinity via TEOS-10 sigma-0 — the BGC source's own T/S for the source
          coordinate and the physics T/S for the target.
        - ``"density_mld"``: the mixed layer depth (MLD) is found in the source and
          target density fields; the source mixed layer is scaled so its MLD matches the
          target's, and below the MLD the tracer is interpolated 1:1 in depth. This keeps
          the mixed layers aligned while preserving the absolute depth of sub-mixed-layer
          features, and avoids the surface degeneracy of pure density space.

        ``"density"`` and ``"density_mld"`` only apply when ``type="bgc"``, the
        physics source is a lat/lon dataset (not a ROMS restart), and the BGC source
        carries temperature/salinity (e.g. the unified dataset's ``temp_WOA``/
        ``salt_WOA``); otherwise interpolation falls back to depth space and notes in
        the log. Interpolation uses ``xgcm.Grid.transform`` with the linear method
        inside the source range and edge-value extrapolation outside
        (``mask_edges=False``).
    prefill : str or None, optional
        How to fill NaN (land/void) cells in the *source* before regridding. The
        default (``None``) applies **no** source prefill: with xESMF installed,
        masked bilinear interpolation plus destination extrapolation
        (``extrap_method``) produces NaN-free initial-condition fields directly;
        without xESMF, the source is automatically pre-filled with a cheap
        nearest-neighbor fill before scipy interpolation. Set ``prefill`` to fill
        the whole-domain source first (the regrid is then plain bilinear and
        ``extrap_method`` is ignored). Options:

          - ``"2d_lateral_fill"`` -- legacy AMG Poisson fill (smoothest, slow;
            no xESMF required). This reproduces the pre-v4 fill behavior.
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

        Applies only to lat/lon physics/BGC sources; for a ROMS restart source it
        is ignored (the legacy fill path is used) and a note is logged. Defaults to
        ``None``.
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
        for the fill step regardless of ``regrid_method``. Applies only to lat/lon
        sources (ignored for a ROMS restart source). Defaults to ``None``.
    extrap_method : str or None, optional
        xESMF *destination* extrapolation used on the default path
        (``prefill is None``) to fill target points whose source neighbors are
        all land/out of range, guaranteeing NaN-free output. ``"inverse_dist"``
        (the effective default) gives an inverse-distance-weighted average of the
        nearest source points (smoothly varying); ``"nearest_s2d"`` uses the
        single nearest source point. Ignored when ``prefill`` is set. Defaults to
        ``None`` (treated as ``"inverse_dist"``).
    extrap_kwargs : dict, optional
        Method-specific options for ``extrap_method``: ``num_src_pnts`` /
        ``dist_exponent`` for ``"inverse_dist"``. Defaults to ``None``.

    Examples
    --------
    >>> physics = InitialConditionsSource(
    ...     grid=grid,
    ...     ini_time=datetime(2022, 1, 1),
    ...     source={"name": "GLORYS", "path": "physics_data.nc"},
    ... )
    >>> bgc = InitialConditionsSource(
    ...     grid=grid,
    ...     ini_time=datetime(2022, 1, 1),
    ...     type="bgc",
    ...     physics_forcing=physics,
    ...     source={
    ...         "name": "CESM_REGRIDDED",
    ...         "path": "bgc_data.nc",
    ...         "climatology": False,
    ...     },
    ... )

    >>> initial_conditions = InitialConditionsSource(
    ...     grid=grid,
    ...     ini_time=datetime(2022, 1, 1),
    ...     source={"name": "ROMS", "grid": parent_grid, "path": "restart.nc"},
    ... )

    """

    grid: Grid
    """Object representing the grid information."""
    ini_time: datetime
    """The date and time at which the initial conditions are set."""
    type: str = "physics"
    """Whether this object processes the physics source (``"physics"``) or a BGC
    source (``"bgc"``). ``"bgc"`` requires ``physics_forcing``."""
    source: RawDataSource | None = None
    """Dictionary specifying the source: the physical dataset when
    ``type="physics"``, the BGC dataset when ``type="bgc"``. Always required."""
    physics_forcing: "InitialConditionsSource | None" = None
    """Required when ``type="bgc"``: a ``type="physics"`` InitialConditionsSource
    object supplying temperature/salinity, so this object's own physics regridding
    is skipped entirely. See the class docstring for the full contract."""
    model_reference_date: datetime = datetime(2000, 1, 1)
    """The reference date for the model."""
    allow_flex_time: bool = False
    """Whether to handle ini_time flexibly."""
    use_dask: bool = False
    """Whether to use dask for processing."""
    chunks: dict[str, int] | None = None
    """Optional Dask chunk sizes for lat/lon and ROMS-restart initial-condition sources."""
    initial_slice_bounds: dict[str, tuple[int | float, int | float]] | None = None
    """Optional initial bounding slice when loading lat/lon forcing data with Dask."""
    bypass_validation: bool = False
    """Whether to skip validation checks in the processed data."""
    bgc_interpolation_method: str = "depth"
    """Vertical interpolation method for BGC tracers: ``"depth"``, ``"density"``, or
    ``"density_mld"``."""
    use_vars: list[str] | None = None
    """Optional down-selection of the BGC variables written from ``source`` (only
    applies when ``type="bgc"``). When set, only these variables are kept
    (presence-only check — a ``ValueError`` is
    raised if any requested variable is not provided by the source). No MARBL
    derivation is performed here; call :meth:`BGCMarbl.process_bgc_fields` on the
    finished object(s) to complete the tracer set."""
    prefill: str | None = None
    """Source-side fill applied before lateral regridding. ``None`` (default) applies
    **no** whole-domain source fill: with xESMF the masked-bilinear regrid plus
    destination extrapolation (``extrap_method``) produces NaN-free output directly;
    without xESMF a nearest-neighbor pre-fill is applied automatically before scipy
    interpolation. Set to ``"2d_lateral_fill"`` (legacy AMG Poisson fill),
    ``"nearest_neighbor"``, ``"inverse_dist"``, ``"nearest_s2d"``, or ``"creep_fill"``
    to fill the whole-domain source first (the last three require xESMF).
    Applies only to lat/lon sources; ignored for a ROMS restart source."""
    prefill_kwargs: dict | None = None
    """Method-specific options for ``prefill`` (e.g. ``num_src_pnts`` /
    ``dist_exponent``). Applies only to lat/lon sources."""
    regrid_method: str | None = None
    """Horizontal regrid engine, chosen independently of ``prefill``: ``None``/``"auto"``
    uses xESMF when installed (else scipy), ``"xesmf"`` forces xESMF, ``"scipy"`` forces
    scipy. Applies only to lat/lon sources; ignored for a ROMS restart source."""
    extrap_method: str | None = None
    """xESMF destination extrapolation used on the default no-prefill path (``None`` is
    treated as ``"inverse_dist"``) to fill target points whose source neighbors are all
    masked. Ignored when ``prefill`` is set, on the scipy path, or for a ROMS restart
    source."""
    extrap_kwargs: dict | None = None
    """Method-specific options for ``extrap_method``. Applies only to lat/lon sources."""
    ds: xr.Dataset = field(init=False, repr=False)
    """An xarray Dataset containing post-processed variables ready for input into
    ROMS."""
    adjust_depth_for_sea_surface_height: bool = field(init=False)
    """Whether to account for sea surface height when computing depth coordinates."""
    ds_depth_coords: xr.Dataset = field(init=False, repr=False)
    """An xarray Dataset containing the depth coordinates."""

    def __post_init__(self):
        # Initialize depth coordinates
        self.ds_depth_coords = xr.Dataset()

        self._resolve_prefill_options()
        self._input_checks()

        processed_fields = {}
        if self.type == "bgc":
            # BGC-only construction: reuse the required physics companion's T/S and
            # depth coordinates instead of redundantly regridding the full physics
            # variable set (u, v, zeta, w, barotropic velocities, ...) again for this
            # object. Both DataArrays are taken as-is from
            # ``physics_forcing.ds``/``ds_depth_coords`` -- still dask-backed and
            # unevaluated if ``physics_forcing`` is -- so nothing is materialized here.
            self.adjust_depth_for_sea_surface_height = (
                self.physics_forcing.adjust_depth_for_sea_surface_height
            )
            self.ds_depth_coords = self.physics_forcing.ds_depth_coords
            temp = self.physics_forcing.ds["temp"]
            salt = self.physics_forcing.ds["salt"]
            # physics_forcing.ds carries "ocean_time" as its time dim, with "abs_time"
            # as a datetime64 companion coord (set by its own _add_global_metadata).
            # Swap back to the datetime view named "time" so the ESPER estimator and
            # this object's own _add_global_metadata can rebuild a fresh
            # ocean_time/abs_time pair from it -- mirrors BoundaryForcingSource's
            # handling of its physics_forcing's "bry_time"/"abs_time" pair.
            if "abs_time" in temp.coords:
                temp = temp.swap_dims({"ocean_time": "abs_time"}).rename(
                    {"abs_time": "time"}
                )
                salt = salt.swap_dims({"ocean_time": "abs_time"}).rename(
                    {"abs_time": "time"}
                )
            processed_fields["temp"] = temp
            processed_fields["salt"] = salt
            # No physics variables of our own to NaN-check in _validate().
            object.__setattr__(self, "variable_info_physics", {})

            # BGC processing appends the source's own (raw) tracers to the shared
            # ``processed_fields`` dict (the borrowed physics T/S above is used as
            # the target for density/MLD vertical interpolation). MARBL tracer
            # derivation/fill is intentionally NOT done here — call
            # ``BGCMarbl().process_bgc_fields()`` on the finished object for that.
            phys_keys = set(processed_fields)
            processed_fields = self._process_data(processed_fields, type="bgc")
            self._apply_use_vars(processed_fields, phys_keys)
            # This object contributes only its own BGC tracers -- the borrowed
            # physics fields belong to ``physics_forcing`` and are written to
            # output there, not here (the caller merges the two together, e.g.
            # via ``xr.merge``/``merge()``).
            for k in phys_keys:
                processed_fields.pop(k, None)
        else:
            processed_fields = self._process_data(processed_fields, type="physics")

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

    def _resolve_prefill_options(self) -> None:
        """Build the validated :class:`RegridConfig` from the public options.

        Delegates all prefill/extrap/regrid validation to
        :meth:`RegridConfig.from_options`, then writes the resolved ``prefill`` back
        to the public field (as a plain string or ``None``) so the YAML round-trip
        emits a clean ``prefill``. Derived state (``use_xesmf``, ``effective_extrap``,
        ...) is read off ``self._regrid``. These options apply to lat/lon sources
        only; the ROMS-restart path (see :meth:`_process_data`) ignores them.
        """
        self._regrid = RegridConfig.from_options(
            prefill=self.prefill,
            prefill_kwargs=self.prefill_kwargs,
            regrid_method=self.regrid_method,
            extrap_method=self.extrap_method,
            extrap_kwargs=self.extrap_kwargs,
            xesmf_available=_xesmf_available(),
        )
        self.prefill = (
            None if self._regrid.prefill is None else str(self._regrid.prefill)
        )

    def _warn_if_regrid_options_set_for_roms(self) -> None:
        """Log a note when prefill/regrid options are set but the source is ROMS.

        Mirrors the ``bgc_interpolation_method`` house style: the options are
        accepted but have no effect on a ROMS restart source (which uses the
        legacy lateral fill + ``LateralRegridFromROMS``), so we note the fallback
        in the log rather than raising.
        """
        if any(
            opt is not None
            for opt in (
                self.prefill,
                self.prefill_kwargs,
                self.regrid_method,
                self.extrap_method,
                self.extrap_kwargs,
            )
        ):
            logging.info(
                "prefill/regrid_method/extrap_method apply to lat/lon sources only; "
                "ignoring them for the ROMS restart source and using the legacy "
                "lateral fill."
            )

    def _process_data(self, processed_fields, type="physics"):
        # BGC "constants" source: broadcast each user-supplied value onto the finalized
        # physics tracer grid (temp on sigma levels). No dataset load or regridding.
        if type == "bgc" and self.source["name"] == "constants":
            template = processed_fields["temp"]
            constants = self.source["constants"]
            for var, val in constants.items():
                processed_fields[var] = xr.full_like(template, float(val))
            # These raw tracers are validated/described model-agnostically (only ALK is
            # NaN-checked); the metadata is needed by _validate.
            object.__setattr__(
                self, "variable_info_bgc", bgc_variable_info(list(constants.keys()))
            )
            return processed_fields

        # BGC "ESPER" source: derive tracers from the finalized physics T/S already on
        # the ROMS grid via PyESPER (no dataset load, no regridding). Lazy when physics
        # T/S are dask-backed.
        if type == "bgc" and self.source["name"] == "ESPER":
            from roms_tools.setup.esper import ESPER_SUPPORTED_VARS, estimate_bgc_fields

            temp = processed_fields["temp"]
            salt = processed_fields["salt"]
            depth = self.ds_depth_coords["layer_depth_rho"]
            lon = self.grid.ds["lon_rho"]
            lat = self.grid.ds["lat_rho"]
            year = self.ini_time.year + (self.ini_time.timetuple().tm_yday - 1) / 365.25
            fields = estimate_bgc_fields(
                temp,
                salt,
                lon,
                lat,
                depth,
                source=self.source,
                roms_variables=ESPER_SUPPORTED_VARS,
                est_dates=year,
            )
            processed_fields.update(fields)
            object.__setattr__(
                self, "variable_info_bgc", bgc_variable_info(list(fields))
            )
            return processed_fields

        target_coords = get_target_coords(self.grid)

        data = self._get_data(forcing_type=type)
        data.choose_subdomain(
            target_coords,
            unchunk_lateral_dims=True,
        )
        # Enforce double precision to ensure reproducibility
        data.convert_to_float64()
        data.extrapolate_deepest_to_bottom()
        if isinstance(data, ROMSDataset):
            # ROMS restart source: the bespoke ROMS-grid fill feeds only
            # ``LateralRegridFromROMS`` (unchanged). The prefill/regrid/extrap options
            # do not apply here; warn if the user explicitly set any of them.
            self._warn_if_regrid_options_set_for_roms()
            data.apply_lateral_fill()
        else:
            # `self.source` is always the source relevant to `type` now (contextual
            # on `self.type`), so no physics-vs-bgc branch is needed here anymore.
            source_name = self.source["name"]
            if self._regrid.extrap_is_active:
                check_source_coverage(data, target_coords, source_name)
            apply_source_prefill(data, self._regrid, self.prefill_kwargs)
            apply_scipy_fallback_fill(data, self._regrid)
        data.rotate_velocities_to_east_and_north()

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
                "is_3d": variable_info[var]["is_3d"],
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
                    "is_3d": variable_info[var]["is_3d"],
                }
                for var in data.opt_var_names.keys()
                if data.opt_var_names[var] in data.ds.data_vars
            }
        )

        # Lateral regridding
        processed_fields = self._regrid_laterally(
            data, target_coords, processed_fields, var_names
        )
        # Rotation of velocities and interpolation to u/v points
        if "u" in var_names and "v" in var_names:
            processed_fields["u"], processed_fields["v"] = rotate_velocities(
                processed_fields["u"],
                processed_fields["v"],
                target_coords["angle"],
                interpolate_after=True,
            )

        if type == "bgc":
            # Ensure time coordinate matches that of physical variables
            ref_time = processed_fields["temp"]["time"]
            for var_name in var_names:
                field = processed_fields[var_name]
                if "time" in field.dims:
                    processed_fields[var_name] = field.assign_coords({"time": ref_time})
                else:
                    # Static source (an observational climatology such as GLODAP, which
                    # ships no time dimension) -- there is no axis to relabel, so give it
                    # the physics one. ROMS's ``inifile`` is a single scalar path, so every
                    # bgc source is merged into one dataset with one time axis; without
                    # this the merge fails on a field whose dims are (depth, eta_rho,
                    # xi_rho), with a CoordinateValidationError naming neither the source
                    # nor the reason. Broadcasting a constant-in-time field onto the
                    # single initial-conditions record changes no values.
                    processed_fields[var_name] = field.expand_dims(
                        {"time": ref_time}, axis=0
                    )

        # Get depth coordinates
        zeta = (
            processed_fields["zeta"] if self.adjust_depth_for_sea_surface_height else 0
        )
        for location in ["rho", "u", "v"]:
            self._get_depth_coordinates(zeta, location, "layer")

        # Vertical regridding
        processed_fields = self._regrid_vertically(
            data, processed_fields, var_names, type=type
        )

        # Compute barotropic velocities
        if "u" in var_names and "v" in var_names:
            for location in ["u", "v"]:
                self._get_depth_coordinates(zeta, location, "interface")
                processed_fields[f"{location}bar"] = compute_barotropic_velocity(
                    processed_fields[location],
                    self.ds_depth_coords[f"interface_depth_{location}"],
                )

        return processed_fields

    def _apply_use_vars(self, processed_fields, phys_keys):
        """Down-select the BGC variables in ``processed_fields`` to ``self.use_vars``.

        ``phys_keys`` is the set of keys present before BGC processing, so the newly
        added BGC keys are ``processed_fields - phys_keys``. This is a presence-only
        check: it raises ``ValueError`` if any requested variable is not among the
        source's own regridded BGC variables. No MARBL/derivation logic is applied —
        that is handled later by :meth:`BGCMarbl.process_bgc_fields`.
        """
        if self.use_vars is None:
            return
        bgc_keys = [k for k in processed_fields if k not in phys_keys]
        requested = list(self.use_vars)
        missing = [v for v in requested if v not in bgc_keys]
        if missing:
            src_name = self.source.get("name", "?")
            raise ValueError(
                f"use_vars requested variable(s) not present in the '{src_name}' BGC "
                f"source: {sorted(missing)}. Available here: {sorted(bgc_keys)}."
            )
        for k in bgc_keys:
            if k not in requested:
                del processed_fields[k]
                # Keep variable_info_bgc in sync with processed_fields: _validate()
                # iterates variable_info_bgc directly and does ds[var_name] for every
                # entry, so a stale entry for a variable use_vars just excluded (still
                # present because _set_variable_info populated it from the source's
                # full available-variable catalog, before this down-select ran) would
                # raise KeyError there instead of writing a clean use_vars-only dataset.
                self.variable_info_bgc.pop(k, None)

    def _regrid_laterally(
        self,
        data: ROMSDataset | LatLonDataset,
        target_coords: dict[str, xr.DataArray],
        processed_fields: dict[str, xr.DataArray],
        var_names: dict[str, dict[str, str]],
    ):
        """Regrid variables in data.ds laterally to target coordinates.

        Parameters
        ----------
        data : ROMSDataset or LatLonDataset
            The dataset containing variables to regrid.
        target_coords : dict[str, xr.DataArray]
            Dictionary of target coordinates for regridding.
        processed_fields : dict[str, xr.DataArray]
            Dictionary where regridded variables will be stored.
        var_names : dict[str, dict[str, str]]
            Mapping from variable keys to dataset variable names and metadata.

        Returns
        -------
        processed_fields : dict[str, xr.DataArray]
            Updated dictionary with regridded variables.
        """
        if isinstance(data, ROMSDataset):
            # Compute depth coordinates on source data for rho
            data._get_depth_coordinates(depth_type="layer", locations=["rho"])
            # Subset depth coordinate to target subdomain
            data.ds_depth_coords = choose_subdomain(
                data.ds_depth_coords,
                data.grid.ds,
                target_coords,
                unchunk_lateral_dims=True,
                dim_names=data.dim_names,
            )

            # Regrid all rho variables
            ds_rho = data.ds[[var_names[var]["name"] for var in var_names]].rename(
                {"lat_rho": "lat", "lon_rho": "lon"}
            )
            lateral_regrid_from_roms = LateralRegridFromROMS(ds_rho, target_coords)
            ds_rho = lateral_regrid_from_roms.apply(ds_rho)

            for var_name in var_names:
                processed_fields[var_name] = ds_rho[var_name]

            # Regrid depth coordinates
            processed_fields["layer_depth_rho"] = lateral_regrid_from_roms.apply(
                data.ds_depth_coords["layer_depth_rho"]
            )

        else:
            # Velocity fields (location "u"/"v") use the velocity mask; build a
            # separate vector regridder only when a velocity var is present so
            # mask_vel-less sources (e.g. BGC) don't pay for unused xESMF weights.
            def _mask(is_vector):
                return select_source_mask(
                    data.ds,
                    is_vector=is_vector,
                    use_xesmf=self._regrid.use_xesmf,
                    prefill=self._regrid.prefill,
                )

            scalar_rg = build_lateral_regridder(
                target_coords, data, self._regrid, _mask(False)
            )
            has_vel = any(var_names[v]["location"] in ("u", "v") for v in var_names)
            vector_rg = (
                build_lateral_regridder(target_coords, data, self._regrid, _mask(True))
                if has_vel
                else scalar_rg
            )
            for var_name in var_names:
                rg = (
                    vector_rg
                    if var_names[var_name]["location"] in ("u", "v")
                    else scalar_rg
                )
                processed_fields[var_name] = rg.apply(
                    data.ds[var_names[var_name]["name"]]
                )

        return processed_fields

    def _regrid_vertically(
        self,
        data: ROMSDataset | LatLonDataset,
        processed_fields: dict[str, xr.DataArray],
        var_names: dict[str, dict[str, str | bool]],
        type: str = "physics",
    ) -> dict[str, xr.DataArray]:
        """
        Perform vertical regridding of 3D variables to the model's vertical grid.

        For each vertical location ('rho', 'u', 'v'), this method regrids variables
        that are flagged as 3D in `var_names`. The regridding procedure differs
        depending on whether the source dataset is a ROMSDataset or a LatLonDataset.

        Parameters
        ----------
        data : ROMSDataset or LatLonDataset
            Dataset containing the variables to regrid.
        processed_fields : dict[str, xarray.DataArray]
            Dictionary containing fields that have already been regridded laterally.
            This method updates the entries in-place with vertically regridded fields.
        var_names : dict[str, dict[str, str | bool]]
            Mapping of variable keys to dataset variable metadata:
                - 'name': dataset variable name
                - 'location': vertical location ('rho', 'u', 'v')
                - 'is_3d': whether the variable is 3D and requires vertical regridding

        Returns
        -------
        processed_fields : dict[str, xarray.DataArray]
            Dictionary containing the same variables as `processed_fields`, now updated
            with vertically regridded values.
        """
        for location in ["rho", "u", "v"]:
            # Select variables for this vertical location that are 3D
            filtered_vars = [
                var_name
                for var_name, info in var_names.items()
                if info["location"] == location and info["is_3d"]
            ]

            if not filtered_vars:
                continue

            if isinstance(data, ROMSDataset):
                # Interpolate depth coordinates from rho to u/v points if needed
                if location == "u":
                    processed_fields["layer_depth_u"] = interpolate_from_rho_to_u(
                        processed_fields["layer_depth_rho"]
                    )
                elif location == "v":
                    processed_fields["layer_depth_v"] = interpolate_from_rho_to_v(
                        processed_fields["layer_depth_rho"]
                    )

                # Use the first variable to initialize VerticalRegrid
                ds_tmp = xr.Dataset(
                    {filtered_vars[0]: processed_fields[filtered_vars[0]]}
                )
                vertical_regrid = VerticalRegrid(ds_tmp, source_dim="s_rho")

                for var_name in filtered_vars:
                    if var_name in processed_fields:
                        processed_fields[var_name] = vertical_regrid.apply(
                            processed_fields[var_name],
                            source_depth_coords=processed_fields[
                                f"layer_depth_{location}"
                            ],
                            target_depth_coords=self.ds_depth_coords[
                                f"layer_depth_{location}"
                            ],
                            mask_edges=False,
                        )
            else:
                # LatLonDataset: create a regrid object for all variables
                vertical_regrid = VerticalRegrid(
                    data.ds, source_dim=data.dim_names["depth"]
                )

                # The BGC dataset declares its own source temperature/salinity pair
                # (``bgc_source_ts``, e.g. ``temp_bgc``/``salt_bgc``) that defines the
                # source density coordinate. These are not ROMS output variables, so they
                # are handled separately from the tracers and dropped afterwards.
                ts_keys = tuple(getattr(data, "bgc_source_ts", ()))
                aux_ts_vars = [
                    v for v in ts_keys if v in filtered_vars and v in processed_fields
                ]
                tracer_vars = [v for v in filtered_vars if v not in aux_ts_vars]

                has_source_ts = len(aux_ts_vars) == 2
                has_target_ts = (
                    "temp" in processed_fields and "salt" in processed_fields
                )
                # Resolve the requested method against availability of the T/S needed
                # to build the density/MLD coordinates; fall back to depth otherwise.
                method = self.bgc_interpolation_method if type == "bgc" else "depth"
                if method != "depth" and not (has_source_ts and has_target_ts):
                    logging.info(
                        f"{method!r} interpolation requested but the BGC source has "
                        "no temperature/salinity; falling back to depth-space "
                        "interpolation."
                    )
                    method = "depth"

                source_coord = None
                target_coord = None
                if method != "depth":
                    temp_key, salt_key = ts_keys
                    s_dim = next(
                        d for d in processed_fields["temp"].dims if d.startswith("s_")
                    )
                    # Source coordinate uses the BGC dataset's own T/S (already on its
                    # source depth grid); target uses the model's (physics) sigma-level
                    # T/S, present in the shared processed_fields from physics processing.
                    source_coord, target_coord = build_bgc_vertical_coords(
                        method,
                        source_temp=processed_fields[temp_key],
                        source_salt=processed_fields[salt_key],
                        source_depth=data.ds[data.dim_names["depth"]],
                        source_depth_dim=data.dim_names["depth"],
                        target_temp=processed_fields["temp"],
                        target_salt=processed_fields["salt"],
                        target_depth=self.ds_depth_coords[f"layer_depth_{location}"],
                        target_depth_dim=s_dim,
                    )

                for var_name in tracer_vars:
                    if var_name not in processed_fields:
                        continue
                    if method != "depth":
                        processed_fields[var_name] = vertical_regrid.apply(
                            processed_fields[var_name],
                            source_depth_coords=source_coord,
                            target_depth_coords=target_coord,
                        )
                    else:
                        processed_fields[var_name] = vertical_regrid.apply(
                            processed_fields[var_name],
                            source_depth_coords=data.ds[data.dim_names["depth"]],
                            target_depth_coords=self.ds_depth_coords[
                                f"layer_depth_{location}"
                            ],
                        )

                # Drop the auxiliary source T/S; they are not ROMS output variables.
                for v in aux_ts_vars:
                    processed_fields.pop(v, None)

        return processed_fields

    def _input_checks(self):
        if self.type not in ("physics", "bgc"):
            raise ValueError(f"`type` must be 'physics' or 'bgc', got {self.type!r}.")

        # -------------------------------------------------------
        # type / physics_forcing checks
        # -------------------------------------------------------
        if self.type == "bgc":
            if self.physics_forcing is None:
                raise ValueError(
                    "`type='bgc'` requires `physics_forcing` (a `type='physics'` "
                    "InitialConditionsSource supplying T/S on the ROMS grid) -- "
                    "there is no standalone bgc-only mode without it."
                )
        elif self.physics_forcing is not None:
            raise ValueError("`physics_forcing` only applies when `type='bgc'`.")

        if self.source is None:
            raise ValueError("`source` is required.")
        if "name" not in self.source.keys():
            raise ValueError("`source` must include a 'name'.")

        if self.type == "physics":
            if "path" not in self.source.keys():
                if self.source["name"] != "GLORYS":
                    raise ValueError("`source` must include a 'path'.")

                self.source["path"] = GLORYSDefaultDataset.dataset_name

            # set self.source["climatology"] to False if not provided
            self.source = {
                **self.source,
                "climatology": self.source.get("climatology", False),
            }
        else:
            name = self.source["name"]
            if name == "constants":
                if not self.source.get("constants"):
                    raise ValueError(
                        "For source={'name': 'constants', ...} you must provide a "
                        "non-empty 'constants' mapping."
                    )
            elif name == "ESPER":
                from roms_tools.setup.esper import validate_esper_source

                validate_esper_source(self.source)
            elif name not in _BGC_SOURCE_NAMES:
                raise ValueError(
                    f"Unknown BGC source name '{name}'. Valid options: "
                    f"{sorted(_BGC_SOURCE_NAMES)}."
                )
            elif "path" not in self.source and name not in _SELF_DOWNLOADING_BGC:
                raise ValueError("`source` must include a 'path'.")
            # Default the climatology flag. Sources that only ever exist as a
            # 12-month climatology default to True, since False fails later with a
            # confusing message about integer time values.
            self.source = {
                **self.source,
                "climatology": self.source.get(
                    "climatology", name in _CLIMATOLOGY_ONLY_BGC
                ),
            }
        if not isinstance(self.ini_time, datetime):
            raise TypeError(
                f"`ini_time` must be a datetime object, got {type(self.ini_time).__name__} instead."
            )
        if self.bgc_interpolation_method not in BGC_INTERPOLATION_METHODS:
            raise ValueError(
                f"`bgc_interpolation_method` must be one of "
                f"{BGC_INTERPOLATION_METHODS}, got {self.bgc_interpolation_method!r}."
            )

    def _get_data(
        self, forcing_type=Literal["physics", "bgc"]
    ) -> LatLonDataset | ROMSDataset:
        """Determine the correct `Dataset` type and return an instance.

        forcing_type : str
            Specifies the type of forcing data. Options are:

            - "physics": for physical atmospheric forcing.
            - "bgc": for biogeochemical forcing.
        Returns
        -------
        Dataset
            The `LatLonDataset` or `ROMSDataset` instance
        """
        dataset_map = _DATASET_MAP

        # `self.source` is always the source relevant to `type`/`forcing_type` now
        # (contextual on `self.type`).
        source_dict = self.source

        if source_dict is None:
            raise ValueError(f"{forcing_type} source is not set")

        source_name = str(source_dict["name"])
        if source_name not in dataset_map[forcing_type]:
            tpl = 'Valid options for source["name"] for type {} include: {}'
            msg = tpl.format(
                forcing_type, " and ".join(dataset_map[forcing_type].keys())
            )
            raise ValueError(msg)

        has_no_path = "path" not in source_dict
        has_default_path = source_dict.get("path") == GLORYSDefaultDataset.dataset_name
        use_default = has_no_path or has_default_path

        variant = "default" if use_default else "external"

        data_type = dataset_map[forcing_type][source_name][variant]

        if isinstance(source_dict.get("path"), bool):
            raise ValueError('source["path"] cannot be a boolean here')

        if source_dict["name"] == "ROMS":
            var_names = _set_required_vars(forcing_type)
            self.adjust_depth_for_sea_surface_height = True

            data = data_type(
                path=source_dict["path"],  # type: ignore
                grid=source_dict["grid"],  # type: ignore
                var_names=var_names,
                start_time=self.ini_time,
                allow_flex_time=self.allow_flex_time,
                adjust_depth_for_sea_surface_height=True,
                use_dask=self.use_dask,
                chunks=self.chunks,
            )

        else:
            self.adjust_depth_for_sea_surface_height = False

            data = data_type(
                # A self-downloading source (see _SELF_DOWNLOADING_BGC) may carry no
                # "path"; it fetches its own data when handed a falsy filename.
                filename=source_dict.get("path", ""),  # type: ignore
                start_time=self.ini_time,
                climatology=source_dict["climatology"],  # type: ignore
                allow_flex_time=self.allow_flex_time,
                use_dask=self.use_dask,
                chunks=self.chunks,
                initial_slice_bounds=self.initial_slice_bounds,
                **bgc_source_extra_kwargs(source_dict),
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
                    if var_name == "zeta":
                        variable_info[var_name] = {
                            "location": "rho",
                            "is_vector": False,
                            "vector_pair": None,
                            "is_3d": False,
                            "validate": False,
                        }
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

        if self.type == "physics":
            # Initialize vertical velocity to zero. A type="bgc" object skips this:
            # "w" is a physics prognostic variable that belongs to physics_forcing's
            # own dataset, and processed_fields has neither "u" (this method's
            # original time template) nor "temp"/"salt" left in it by this point
            # (borrowed only transiently, then stripped so they aren't
            # double-written -- see __post_init__).
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
        ocean_time = (ds["time"] - model_reference_date).dt.total_seconds()
        ds = ds.assign_coords(ocean_time=("time", ocean_time.data.astype("float64")))
        ds["ocean_time"].attrs["long_name"] = (
            f"relative time: seconds since {self.model_reference_date!s}"
        )
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
        Only variables flagged ``validate=True`` are checked, to keep this cheap:
        ``zeta`` for a physics object, and ``ALK`` for a bgc one (a bgc object sets
        ``variable_info_physics`` to ``{}``, so it checks no physics variables at
        all). For a bgc source whose variables share one expensive computation --
        ESPER -- every variable is materialised first and cached, so the check does
        not force a second compute; see :func:`~roms_tools.setup.utils.materialize_before_check`.
        """
        if self.type == "bgc":
            variable_info = {**self.variable_info_physics, **self.variable_info_bgc}
        else:
            variable_info = self.variable_info_physics

        # Materialize every variable sharing this source's expensive computation (not
        # just the ones actually NaN-checked below -- bgc_variable_info() flags only
        # ALK as validate=True, but all of an ESPER source's use_vars come from one
        # shared per-chunk PyESPER call) so a later .save() on this same `ds` reuses
        # these values instead of recomputing them. Must happen before any check view
        # is built from `ds` -- see materialize_before_check's docstring for why.
        materialize_before_check(
            ds, list(variable_info), materialize=self._is_esper_source
        )

        # Build the NaN checks lazily and evaluate them in a single computation so a
        # lazy subgraph shared across variables (e.g. the density/MLD interpolation
        # coordinate reused across BGC tracers) is computed once, not once per variable.
        checks = []
        for var_name in variable_info:
            if variable_info[var_name]["validate"]:
                if variable_info[var_name]["location"] == "rho":
                    mask = self.grid.ds.mask_rho
                elif variable_info[var_name]["location"] == "u":
                    mask = self.grid.ds.mask_u
                elif variable_info[var_name]["location"] == "v":
                    mask = self.grid.ds.mask_v
                checks.append((ds[var_name].squeeze(), mask, None))

        nan_check_batch(checks, serialize_dask=False)

    def _add_global_metadata(self, ds):
        ds.attrs["title"] = "ROMS initial conditions file created by ROMS-Tools"
        # Include the version of roms-tools -- both the semantic version (which can
        # go stale relative to an editable install's actual source, see
        # get_roms_tools_version_info's docstring) and, when available, the exact
        # git commit that produced this file.
        version_info = get_roms_tools_version_info()
        ds.attrs["roms_tools_version"] = version_info["roms_tools_version"]
        ds.attrs["roms_tools_git_commit"] = str(version_info["roms_tools_git_commit"])
        ds.attrs["ini_time"] = str(self.ini_time)
        ds.attrs["model_reference_date"] = str(self.model_reference_date)
        ds.attrs["adjust_depth_for_sea_surface_height"] = str(
            self.adjust_depth_for_sea_surface_height
        )
        if self.type == "physics":
            ds.attrs["source"] = self.source["name"]
        else:
            # A bgc-only object (type="bgc") reports the physics companion's source
            # name here -- it describes the physical data that produced this
            # object's borrowed T/S, even though this object never regridded it
            # itself -- plus its own bgc source name.
            ds.attrs["source"] = self.physics_forcing.source["name"]
            ds.attrs["bgc_source"] = self.source["name"]

        ds.attrs["prefill"] = str(self.prefill)
        ds.attrs["regrid_method"] = "xesmf" if self._regrid.use_xesmf else "scipy"
        ds.attrs["extrap_method"] = str(self._regrid.effective_extrap)

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
            The name of the initial conditions field to plot. Format:

            "{base_var_name}_{direction}" ,

            where {base_var_name} is a physical, BGC, or other boundary tracer name,
            and {direction} is one of ["south", "east", "north", "west"].

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

        if self.adjust_depth_for_sea_surface_height and "zeta" in self.ds:
            zeta = self.ds.zeta.squeeze().load()
        else:
            # A BGC-only object built with `physics_forcing` carries no "zeta" of its
            # own (it belongs to `physics_forcing`) even when
            # adjust_depth_for_sea_surface_height is inherited as True; plot the
            # companion `physics_forcing` object directly to see SSH-adjusted depths.
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

    @property
    def _is_esper_source(self) -> bool:
        """True when this source derives its tracers via PyESPER (the ESPER source).

        All of an ESPER source's ``use_vars`` come out of one shared, expensive lazy
        computation per chunk, which is why validation materialises them once and
        caches (see :func:`roms_tools.setup.utils.materialize_before_check`).
        """
        return (
            self.type == "bgc"
            and self.source is not None
            and self.source.get("name") == "ESPER"
        )

    def save(
        self,
        filepath: str | Path,
        format: NetCDFFormat = DEFAULT_NETCDF_FORMAT,
        serialize_dask: bool | None = None,
    ) -> None:
        """Save the initial conditions information to one NetCDF file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The base path or filename where the dataset should be saved.
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

    @classmethod
    def merge(
        cls,
        physics: "InitialConditionsSource",
        bgc: "InitialConditionsSource | Sequence[InitialConditionsSource]",
        filepath: str | Path | None = None,
        format: NetCDFFormat = DEFAULT_NETCDF_FORMAT,
        serialize_dask: bool | None = None,
    ) -> xr.Dataset | list[Path]:
        """Merge a physics object with one or more bgc-only objects into a single
        ROMS-ready initial-conditions dataset. Fully dask-lazy.

        Note: ROMS's ``inifile`` namelist parameter is a single scalar path, unlike
        boundary/surface forcing's file list. Initial condition has to be assembled
        into ONE dataset before writing, rather than written out as separate files.

        Parameters
        ----------
        physics : InitialConditionsSource
            The ``type="physics"`` object that was passed as ``physics_forcing=``
            to every object in ``bgc``. Its global attrs/coords are authoritative
            in the merge.
        bgc : InitialConditionsSource or sequence of InitialConditionsSource
            One or more bgc-only objects, each built with
            ``physics_forcing=physics``. Only contribute their own
            (non-overlapping, by convention of ``use_vars``) BGC variables.
        filepath : str or Path, optional
            If given, the merged dataset is saved via
            :func:`~roms_tools.utils.save_datasets` and the saved path(s) are
            returned instead of the dataset itself.
        format : {"NETCDF4", "NETCDF3_CLASSIC", "NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA"}, optional
            NetCDF file format, passed through to ``save_datasets`` when
            ``filepath`` is given. Defaults to ``"NETCDF4"``.
        serialize_dask : bool, optional
            See :func:`roms_tools.utils.save_datasets`; only relevant when
            ``filepath`` is given. Defaults to ``None``, which resolves to
            ``False`` (the ordinary concurrent write). Pass ``True`` to force
            the serialized, one-task-at-a-time write -- a manual low-memory /
            troubleshooting tool.

        Returns
        -------
        xr.Dataset or list[Path]
            The merged dataset (when ``filepath`` is omitted), or the saved file
            path(s) (when ``filepath`` is given).

        Raises
        ------
        ValueError
            If ``bgc`` is empty, or any object in it was not built with
            ``physics_forcing=physics``.

        Examples
        --------
        >>> ic_physics = InitialConditionsSource(
        ...     grid=grid, ini_time=t, source={"name": "GLORYS", "path": "..."}
        ... )
        >>> ic_esper = InitialConditionsSource(
        ...     grid=grid,
        ...     ini_time=t,
        ...     type="bgc",
        ...     physics_forcing=ic_physics,
        ...     source={"name": "ESPER", "path": "..."},
        ... )
        >>> ic_unified = InitialConditionsSource(
        ...     grid=grid,
        ...     ini_time=t,
        ...     type="bgc",
        ...     physics_forcing=ic_physics,
        ...     source={"name": "UNIFIED", "path": "...", "climatology": True},
        ...     use_vars=["Fe", "CHL"],
        ... )
        >>> BGCMarbl().process_bgc_fields([ic_esper, ic_unified])
        >>> merged = InitialConditionsSource.merge(ic_physics, [ic_esper, ic_unified])
        """
        bgc_list = [bgc] if isinstance(bgc, InitialConditionsSource) else list(bgc)
        if not bgc_list:
            raise ValueError(
                "`bgc` must contain at least one InitialConditionsSource object."
            )
        for i, b in enumerate(bgc_list):
            if b.physics_forcing is not physics:
                raise ValueError(
                    f"bgc[{i}].physics_forcing is not `physics` -- "
                    "InitialConditionsSource.merge() requires every object in `bgc` to "
                    "have been built with `physics_forcing=<the same physics "
                    "object passed here>`."
                )

        # The physics object's global attrs/coords are authoritative; the bgc
        # objects only contribute their own (non-overlapping, by convention of
        # use_vars) BGC variables.
        merged_ds = xr.merge(
            [physics.ds] + [b.ds for b in bgc_list],
            compat="override",
            combine_attrs="override",
        )

        if filepath is None:
            return merged_ds

        filepath = Path(filepath)
        if filepath.suffix == ".nc":
            filepath = filepath.with_suffix("")

        if serialize_dask is None:
            serialize_dask = False

        _bgc_objs = [bgc] if isinstance(bgc, InitialConditionsSource) else list(bgc)
        return save_datasets(
            [merged_ds],
            [str(filepath)],
            use_dask=physics.use_dask,
            format=format,
            serialize_dask=serialize_dask,
            # Any ESPER contributor is enough: the merged dataset carries its
            # chunks, so the bar misbehaves the same way. See `show_progress`.
            show_progress=not any(o._is_esper_source for o in _bgc_objs),
        )

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
        # Embed the companion physics InitialConditionsSource (supplying T/S for this
        # object's BGC processing) as an optional sub-item of the BGC block,
        # mirroring how Grids are embedded and BoundaryForcingSource's own
        # ``physics_forcing``. The shared "Grid" is dropped since the physics
        # forcing reuses the same grid on reconstruction.
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
            forcing_dict["InitialConditionsSource"]["physics_forcing"] = physics_dict[
                "InitialConditionsSource"
            ]
        write_to_yaml(forcing_dict, filepath)

    @classmethod
    def from_yaml(
        cls,
        filepath: str | Path,
        use_dask: bool = False,
    ) -> "InitialConditionsSource":
        """Create an instance of the InitialConditionsSource class from a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file from which the parameters will be read.
        use_dask: bool, optional
            Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.

        Returns
        -------
        InitialConditionsSource
            An instance of the InitialConditionsSource class.
        """
        filepath = Path(filepath)

        grid = Grid.from_yaml(filepath)
        initial_conditions_params = from_yaml(cls, filepath)

        # Reconstruct an optional embedded physics InitialConditionsSource, reusing the
        # shared grid. The generic `from_yaml` only deserializes the top-level block,
        # so the nested block's datetimes/paths/source are restored here.
        physics_data = initial_conditions_params.pop("physics_forcing", None)
        physics_forcing = None
        if physics_data is not None:
            physics_data = deserialize_forcing_data(physics_data)
            src_dict = physics_data.get("source")
            if src_dict and src_dict.get("grid") is not None:
                src_dict["grid"] = Grid(**pop_grid_data(src_dict["grid"]))
            physics_forcing = cls(grid=grid, **physics_data, use_dask=use_dask)

        # Deserialize a nested grid inside 'source' (e.g. a "ROMS" restart source).
        src_dict = initial_conditions_params.get("source")
        if src_dict and "grid" in src_dict and src_dict["grid"] is not None:
            grid_data = pop_grid_data(src_dict["grid"])
            src_dict["grid"] = Grid(**grid_data)

        return cls(
            grid=grid,
            **initial_conditions_params,
            physics_forcing=physics_forcing,
            use_dask=use_dask,
        )


@dataclass(kw_only=True)
class InitialConditions:
    """Wrapper class that can initialize and process multiple constituent
    :class:`InitialConditionsSource` objects.  This class is the intended
    interface for generating and writing ROMS initial conditions files, and
    its use is fully supported by the ``to_yaml()``/``from_yaml()``
    conventions.

    Internally builds one ``type="physics"`` :class:`InitialConditionsSource` plus
    one ``type="bgc"`` :class:`InitialConditionsSource` per ``bgc_sources`` item
    (each wired with ``physics_forcing=`` to reuse the physics object's T/S instead
    of redundantly regridding it -- see :class:`InitialConditionsSource`'s own
    docstring for that mechanism), completes the BGC tracer set via
    ``bgc_model().process_bgc_fields()``, and merges everything into one ``.ds`` via
    :meth:`InitialConditionsSource.merge`.

    The constituent `InitialConditionsSource` objects are  public and
    documented.  They can be accessed as ``.physics`` and ``.bgc[i]``,
    each a :class:`InitialConditionsSource` carrying its own ``.ds``
    xarray DataSet and ``.plot()`` capability.

    Parameters
    ----------
    grid, ini_time, model_reference_date, allow_flex_time, use_dask, chunks,
    initial_slice_bounds, bypass_validation, prefill, prefill_kwargs,
    regrid_method, extrap_method, extrap_kwargs
        Forwarded to the internal physics :class:`InitialConditionsSource` (and, for
        the fields that apply to a lat/lon bgc source too, to each bgc companion).
        See :class:`InitialConditionsSource` for the full description of each.
    source : RawDataSource
        The physical initial-condition dataset. Required.
    bgc_source : RawDataSource, optional
        Legacy single-bgc-source convenience, equivalent to
        ``bgc_sources=[{"source": bgc_source}]``. Mutually exclusive with
        ``bgc_sources``.
    use_vars : list[str], optional
        Legacy single-bgc-source convenience paired with ``bgc_source``:
        equivalent to ``bgc_sources=[{"source": bgc_source, "use_vars": use_vars}]``.
        Only valid alongside ``bgc_source``; use the per-item ``use_vars`` in
        ``bgc_sources`` instead when combining multiple sources.
    bgc_sources : list[dict], optional
        Zero or more BGC sources, one dict per source:
        ``{"source": RawDataSource, "use_vars": list[str] | None,
        "bgc_interpolation_method": str | None}``. A per-item
        ``bgc_interpolation_method`` overrides the wrapper-level default below.
    bgc_model : type[BGCModel], optional
        The :class:`~roms_tools.setup.bgc_model.BGCModel` subclass (e.g.
        :class:`~roms_tools.setup.bgc_model.BGCMarbl`) used to complete the tracer
        set -- passed as the class itself, since every ``BGCModel`` is instantiated
        with zero arguments. Required whenever ``bgc_source``/``bgc_sources`` is
        given.
    bgc_interpolation_method : str, optional
        Wrapper-level default vertical interpolation method for BGC tracers
        (``"depth"``, ``"density"``, or ``"density_mld"``); see
        :class:`InitialConditionsSource` for the full description. Overridden
        per-source by that source's own ``bgc_interpolation_method`` entry.

    Examples
    --------
    >>> ic = InitialConditions(
    ...     grid=grid,
    ...     ini_time=datetime(2013, 1, 1),
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
    >>> ic.save("my_ic.nc")
    """

    grid: Grid
    """Object representing the grid information."""
    ini_time: datetime
    """The date and time at which the initial conditions are set."""
    source: RawDataSource
    """Dictionary specifying the source of the physical initial condition data."""
    bgc_source: RawDataSource | None = None
    """Legacy single-source convenience; equivalent to
    ``bgc_sources=[{"source": bgc_source}]``. Mutually exclusive with
    ``bgc_sources``."""
    use_vars: list[str] | None = None
    """Legacy single-source convenience, paired with ``bgc_source``: down-selects
    the variables kept from that one BGC source, equivalent to
    ``bgc_sources=[{"source": bgc_source, "use_vars": use_vars}]``. Only valid
    alongside ``bgc_source``; with ``bgc_sources`` (N sources), set
    ``use_vars`` per-item instead, since a single top-level value would be
    ambiguous."""
    bgc_sources: list[dict] | None = None
    """Zero or more BGC sources; see the class docstring for the per-item shape."""
    bgc_model: type[BGCModel] | None = None
    """The BGCModel subclass (e.g. BGCMarbl) used to complete the tracer set.
    Required whenever ``bgc_source``/``bgc_sources`` is given."""
    model_reference_date: datetime = datetime(2000, 1, 1)
    """The reference date for the model."""
    allow_flex_time: bool = False
    """Whether to handle ini_time flexibly."""
    use_dask: bool = False
    """Whether to use dask for processing."""
    chunks: dict[str, int] | None = None
    """Optional Dask chunk sizes for lat/lon and ROMS-restart initial-condition sources."""
    initial_slice_bounds: dict[str, tuple[int | float, int | float]] | None = None
    """Optional initial bounding slice when loading lat/lon forcing data with Dask."""
    bypass_validation: bool = False
    """Whether to skip validation checks in the processed data."""
    bgc_interpolation_method: str = "depth"
    """Wrapper-level default vertical interpolation method for BGC tracers;
    overridden per-source by that source's own entry in ``bgc_sources``."""
    prefill: str | None = None
    """Source-side fill applied before lateral regridding; see
    :class:`InitialConditionsSource` for the full description."""
    prefill_kwargs: dict | None = None
    """Method-specific options for ``prefill``."""
    regrid_method: str | None = None
    """Horizontal regrid engine, chosen independently of ``prefill``."""
    extrap_method: str | None = None
    """xESMF destination extrapolation used on the default no-prefill path."""
    extrap_kwargs: dict | None = None
    """Method-specific options for ``extrap_method``."""

    physics: InitialConditionsSource = field(init=False, repr=False)
    """The internal physics-only InitialConditionsSource object. Exposed for
    advanced use (e.g. ``ic.physics.plot(...)``)."""
    bgc: list[InitialConditionsSource] = field(init=False, repr=False)
    """The internal bgc-only InitialConditionsSource objects, one per
    ``bgc_sources`` item, in order. Exposed for advanced use."""
    ds: xr.Dataset = field(init=False, repr=False)
    """The complete (physics + all BGC sources) initial-conditions dataset --
    ``self.physics.ds`` when there are no BGC sources, otherwise
    :meth:`InitialConditionsSource.merge`. Set once in ``__post_init__``; a plain,
    reassignable attribute (not a property) so this behaves exactly like the
    pre-split monolithic class -- dask-lazy either way, since ``merge()`` performs
    no computation by itself."""

    def __post_init__(self):
        if self.bgc_source is not None and self.bgc_sources is not None:
            raise ValueError(
                "Provide at most one of `bgc_source` (legacy single-source) or "
                "`bgc_sources` (list) -- not both."
            )
        if self.use_vars is not None and self.bgc_source is None:
            raise ValueError(
                "`use_vars` only applies alongside `bgc_source` (legacy "
                "single-source convenience); set it per-item in `bgc_sources` "
                "instead."
            )
        if self.bgc_interpolation_method not in BGC_INTERPOLATION_METHODS:
            raise ValueError(
                f"`bgc_interpolation_method` must be one of "
                f"{BGC_INTERPOLATION_METHODS}, got {self.bgc_interpolation_method!r}."
            )
        bgc_sources = self.bgc_sources
        if self.bgc_source is not None:
            bgc_sources = [{"source": self.bgc_source, "use_vars": self.use_vars}]
        bgc_sources = bgc_sources or []

        physics_kwargs = dict(
            ini_time=self.ini_time,
            source=self.source,
            model_reference_date=self.model_reference_date,
            allow_flex_time=self.allow_flex_time,
            use_dask=self.use_dask,
            chunks=self.chunks,
            initial_slice_bounds=self.initial_slice_bounds,
            bypass_validation=self.bypass_validation,
            prefill=self.prefill,
            prefill_kwargs=self.prefill_kwargs,
            regrid_method=self.regrid_method,
            extrap_method=self.extrap_method,
            extrap_kwargs=self.extrap_kwargs,
        )
        self.physics = InitialConditionsSource(
            grid=self.grid, type="physics", **physics_kwargs
        )

        if bgc_sources:
            if self.bgc_model is None:
                raise ValueError(
                    "`bgc_model` is required when `bgc_source`/`bgc_sources` is "
                    "provided (e.g. `bgc_model=rt.BGCMarbl`)."
                )
            shared_kwargs = dict(
                ini_time=self.ini_time,
                model_reference_date=self.model_reference_date,
                allow_flex_time=self.allow_flex_time,
                use_dask=self.use_dask,
                chunks=self.chunks,
                initial_slice_bounds=self.initial_slice_bounds,
                bypass_validation=self.bypass_validation,
                bgc_interpolation_method=self.bgc_interpolation_method,
                prefill=self.prefill,
                prefill_kwargs=self.prefill_kwargs,
                regrid_method=self.regrid_method,
                extrap_method=self.extrap_method,
                extrap_kwargs=self.extrap_kwargs,
            )
            self.bgc = build_bgc_companions(
                InitialConditionsSource,
                self.grid,
                self.physics,
                bgc_sources,
                shared_kwargs,
                type_="bgc",
            )
            self.bgc_model().process_bgc_fields(self.bgc)
        else:
            self.bgc = []

        if self.bgc:
            self.ds = InitialConditionsSource.merge(self.physics, self.bgc)
        else:
            self.ds = self.physics.ds

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
        """Plot a field from the complete (physics + BGC) dataset.

        Thin delegate over the merged ``.ds`` -- see
        :meth:`InitialConditionsSource.plot` for the full parameter description.
        Sea-surface-height adjustment always uses the physics object's own
        ``adjust_depth_for_sea_surface_height`` (zeta always belongs to the physics
        source, never a bgc companion).
        """
        if var_name not in self.ds:
            raise ValueError(f"Variable '{var_name}' is not found in the dataset.")

        if self.use_dask:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                self.ds[var_name].load()

        if self.physics.adjust_depth_for_sea_surface_height and "zeta" in self.ds:
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

    def save(
        self,
        filepath: str | Path,
        format: NetCDFFormat = DEFAULT_NETCDF_FORMAT,
        serialize_dask: bool | None = None,
    ) -> list[Path]:
        """Save the complete initial conditions to one NetCDF file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The base path or filename where the dataset should be saved.
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
        list[Path]
            The saved file path(s).
        """
        filepath = Path(filepath)
        if filepath.suffix == ".nc":
            filepath = filepath.with_suffix("")
        if serialize_dask is None:
            serialize_dask = False
        return save_datasets(
            [self.ds],
            [str(filepath)],
            use_dask=self.use_dask,
            format=format,
            serialize_dask=serialize_dask,
            # self.ds is the merge of physics + every bgc companion, so one ESPER
            # contributor is enough to make the bar misbehave. See `show_progress`.
            show_progress=not any(o._is_esper_source for o in self.bgc),
        )

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
        forcing_dict["InitialConditions"]["bgc_model"] = bgc_model_to_name(
            self.bgc_model
        )
        write_to_yaml(forcing_dict, filepath)

    @classmethod
    def from_yaml(
        cls,
        filepath: str | Path,
        use_dask: bool = False,
    ) -> "InitialConditions":
        """Create an instance of the InitialConditions class from a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file from which the parameters will be read.
        use_dask : bool, optional
            Indicates whether to use dask for processing. Defaults to False.

        Returns
        -------
        InitialConditions
            An instance of the InitialConditions class.
        """
        filepath = Path(filepath)
        grid = Grid.from_yaml(filepath)
        params = from_yaml(cls, filepath)
        params["bgc_model"] = bgc_model_from_name(params.get("bgc_model"))

        # Deserialize nested grids: the top-level source/bgc_source, and each
        # bgc_sources[i]["source"] (e.g. a "ROMS" restart bgc source).
        for name in ["source", "bgc_source"]:
            src_dict = params.get(name)
            if src_dict and src_dict.get("grid") is not None:
                src_dict["grid"] = Grid(**pop_grid_data(src_dict["grid"]))
        for item in params.get("bgc_sources") or []:
            src_dict = item.get("source")
            if src_dict and src_dict.get("grid") is not None:
                src_dict["grid"] = Grid(**pop_grid_data(src_dict["grid"]))

        return cls(grid=grid, **params, use_dask=use_dask)


def _set_dask_chunks(location: str, chunk_size: int):
    """Returns the appropriate Dask chunking dictionary based on grid location.

    Parameters
    ----------
    location : str
        The grid location, one of "rho", "u", or "v".
    chunk_size : int
        The chunk size to apply.

    Returns
    -------
    dict
        Dictionary specifying the chunking strategy.
    """
    chunk_mapping = {
        "rho": {"eta_rho": chunk_size, "xi_rho": chunk_size},
        "u": {"eta_rho": chunk_size, "xi_u": chunk_size},
        "v": {"eta_v": chunk_size, "xi_rho": chunk_size},
    }
    return chunk_mapping.get(location, {})


def _set_required_vars(var_type: str = "physics") -> dict[str, str]:
    """
    Return the canonical variable-name mapping for a ROMS dataset.

    Parameters
    ----------
    var_type : str, optional
        Category of variables. Supported values:
        - "physics": physical variables (temperature, salinity, currents, etc.)
        - "bgc": biogeochemical variables (nutrients, pigments, carbon, etc.)
        Default is "physics".

    Returns
    -------
    dict[str, str]
        Mapping from logical variable names to dataset variable names.

    Raises
    ------
    ValueError
        If an unsupported `var_type` is provided.
    """
    var_mappings = {
        "physics": {
            "zeta": "zeta",
            "temp": "temp",
            "salt": "salt",
            "u": "u",
            "v": "v",
        },
        "bgc": {
            "zeta": "zeta",  # to infer vertical coordinate
            "PO4": "PO4",
            "NO3": "NO3",
            "SiO3": "SiO3",
            "NH4": "NH4",
            "Fe": "Fe",
            "Lig": "Lig",
            "O2": "O2",
            "DIC": "DIC",
            "DIC_ALT_CO2": "DIC_ALT_CO2",
            "ALK": "ALK",
            "ALK_ALT_CO2": "ALK_ALT_CO2",
            "DOC": "DOC",
            "DON": "DON",
            "DOP": "DOP",
            "DOPr": "DOPr",
            "DONr": "DONr",
            "DOCr": "DOCr",
            "spChl": "spChl",
            "spC": "spC",
            "spP": "spP",
            "spFe": "spFe",
            "diatChl": "diatChl",
            "diatC": "diatC",
            "diatP": "diatP",
            "diatFe": "diatFe",
            "diatSi": "diatSi",
            "diazChl": "diazChl",
            "diazC": "diazC",
            "diazP": "diazP",
            "diazFe": "diazFe",
            "spCaCO3": "spCaCO3",
            "zooC": "zooC",
        },
    }

    if var_type not in var_mappings:
        raise ValueError(
            f"Unsupported var_type '{var_type}'. Choose from {list(var_mappings.keys())}."
        )

    return var_mappings[var_type]
