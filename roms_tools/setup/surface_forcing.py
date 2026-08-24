import importlib.metadata
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from roms_tools import Grid
from roms_tools.blend import coverage_fraction, layer_fields
from roms_tools.datasets.curvilinear_datasets import (
    DEFAULT_CONUS404_PATH,
    CONUS404Dataset,
    CurvilinearDataset,
)
from roms_tools.datasets.lat_lon_datasets import (
    CESMBGCSurfaceForcingDataset,
    ERA5Correction,
    LatLonDataset,
    MBLco2Dataset,
    SODARestoringSurfaceDataset,
    UnifiedBGCSurfaceDataset,
    UnifiedRestoringSurfaceDataset,
    WOARestoringSurfaceDataset,
    resolve_era5_source,
)
from roms_tools.plot import plot
from roms_tools.processing_methods import (
    RegridConfig,
    _xesmf_available,
)
from roms_tools.regrid import (
    LateralRegridFromROMS,
    LateralRegridToROMS,
    build_lateral_regridder,
    select_source_mask,
)
from roms_tools.setup.utils import (
    RawDataSource,
    add_time_info_to_ds,
    apply_scipy_fallback_fill,
    apply_source_prefill,
    check_source_coverage,
    compute_missing_surface_bgc_variables,
    from_yaml,
    get_target_coords,
    get_variable_metadata,
    group_dataset,
    min_dist_to_land,
    nan_check,
    substitute_nans_by_fillvalue,
    to_dict,
    write_to_yaml,
)
from roms_tools.utils import (
    DEFAULT_NETCDF_FORMAT,
    NetCDFFormat,
    interpolate_from_climatology,
    rotate_velocities,
    save_datasets,
    transpose_dimensions,
)

# Number of forcing time steps per block when interpolating the radiation-correction
# climatology onto the forcing time axis. Keeps the interpolated time axis chunked so
# long records don't build one giant task and slicing a few steps stays cheap. Larger
# values mean fewer, bigger dask tasks; smaller values mean cheaper per-slice reads.
_DEFAULT_CLIMATOLOGY_TIME_CHUNK = 100

DEFAULT_MBL_co2_PATH = (
    "https://gml.noaa.gov/ccgg/mbl/tmp/co2_GHGreference.1785677502_surface.txt"
)

DEFAULT_SODA_PATH = "https://www.ncei.noaa.gov/data/oceans/archive/arc0160/0220059/6.6/data/0-data/OceanSODA_ETHZ-v2025.OCADS.01-1982-2024.nc"


@dataclass(kw_only=True)
class SurfaceForcing:
    """Represents surface forcing input data for ROMS.

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
    source : RawDataSource
        Dictionary specifying the source of the surface forcing data. Keys include:

          - "name" (str): Name of the data source. For ``type="physics"``:
            "ERA5" (global, ~28 km) or "CONUS404" (North America only, 4 km).
            See the CONUS404 notes below.
          - "path" (optional; Union[str, Path, List[Union[str, Path]]]): Path(s) to the raw data file(s). Accepted formats:

            - A single string (supports wildcards),
            - A single Path object,
            - A list of strings or Path objects.
            If omitted or set to the ARCO URL, the data will be streamed from the cloud.
          - "climatology" (bool): Indicates if the data is climatology data. Defaults to False.

    type : str
        Specifies the type of forcing data. Options are:

          - "physics": for physical atmospheric forcing.
          - "bgc": for biogeochemical forcing.
          - "restoring": for restoring forces.

    correct_radiation : bool
        Whether to correct shortwave and longwave radiation. Default is True.

    wind_dropoff : bool, optional
        Whether to apply a coastal wind speed reduction to mimic nearshore wind drop-off.
        This applies an exponential decay to wind magnitude near the coast, based on
        a 12.5 km e-folding scale, with up to 40% reduction at the coastline. Default is False.

    restoring_forces : list[str], optional
        Specifies which variables to apply restoring forces to. Sea surface salinity, DIC and alkalinity are supported:
        ```['sss', 'sDIC', 'sALK']```.

    coarse_grid_mode : str, optional
        Specifies whether to interpolate onto grid coarsened by a factor of two. Options are:

          - "auto" (default): Automatically decide based on the comparison of source and target spatial resolutions.
          - "always": Always interpolate onto the coarse grid.
          - "never": Never use the coarse grid; interpolate onto the fine grid instead.

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


    Notes
    -----
    **CONUS404.** A 4 km WRF reanalysis over North America, hourly from
    1979-10-01 to 2024-10-01, streamed by default from the USGS/OSN zarr store
    (which needs ``use_dask=True`` and the ``s3fs`` package; install the
    ``stream`` extra). Note that its lat/lon bounding box overstates coverage --
    the domain is a rectangle in Lambert Conformal space, so along the US west
    coast its northern limit drops from ~55 N at 125 W to nothing west of ~139 W.
    See :class:`~roms_tools.datasets.curvilinear_datasets.CONUS404Dataset` for the
    exact corners. Points of note:

    - Being a **regional** source on a **curvilinear** grid, it is regridded with
      non-extrapolating xESMF bilinear, so any part of the ROMS grid outside its
      footprint comes back as **NaN by design**. Use a fully contained grid, pass
      ``bypass_validation=True``, or layer it over a global source. It requires
      xESMF; the scipy engine cannot interpolate a 2D source grid.
    - ``correct_radiation`` must be ``False``: the ERA5 correction climatology is
      matched by exact lat/lon coordinate value, which a curvilinear grid cannot
      satisfy.
    - Its radiation is derived by differencing float32 accumulations since the
      1979 model start, which carries an intrinsic ~+/-4.6 W/m^2 quantization
      noise floor. See
      :data:`~roms_tools.datasets.curvilinear_datasets.CONUS404_RADIATION_NOISE_FLOOR_W_M2`.

    **Layering two sources.** Setting ``fallback_source`` takes each target point
    from ``source`` where that source has data and from ``fallback_source``
    everywhere else, emitting one file. It exists for a regional product that
    cannot cover the whole domain on its own, e.g. CONUS404 over ERA5. Behavior
    worth knowing before relying on it:

    - By default the boundary is a **hard edge**: every point comes from exactly one
      source. Set ``blend_width_km`` to average the two across a band of that width,
      measured inward from the edge of the primary's footprint.
    - With ``coarse_grid_mode="auto"`` the **primary source decides** the output
      resolution. The two will usually disagree, since a limited-extent source is
      normally the finer one, and taking the fallback's answer would coarsen away
      the resolution the primary was added for. A note is logged when they differ.
    - ``correct_radiation`` applies **only to the sources that have a correction
      registered** (today, ERA5). Applying an ERA5-vs-observations ratio field to a
      different product would not be a correction, so this leaves a small step in
      ``swrad``/``lwrad`` at the boundary.
    - A comparison of the two sources over their overlap region is logged at INFO.
      It is the cheapest way to catch a unit or convention mismatch between them;
      read it before trusting the output.
    - ``prefill`` is rejected alongside ``fallback_source``: it makes the primary
      NaN-free, erasing the coverage gaps the fallback exists to fill.
    - Only for ``type="physics"``, and not for climatologies.

    Examples
    --------
    >>> surface_forcing = SurfaceForcing(
    ...     grid=grid,
    ...     start_time=datetime(2000, 1, 1),
    ...     end_time=datetime(2000, 1, 2),
    ...     source={"name": "ERA5", "path": "era5_data.nc"},
    ...     type="physics",
    ...     correct_radiation=True,
    ... )
    """

    grid: Grid
    """Object representing the grid information."""
    start_time: datetime | None = None
    """The start time of the desired surface forcing data."""
    end_time: datetime | None = None
    """The end time of the desired surface forcing data."""
    source: RawDataSource
    """Dictionary specifying the source of the surface forcing data."""
    type: str = "physics"
    """Specifies the type of forcing data ("physics", "bgc", "restoring")."""
    correct_radiation: bool = True
    """Whether to correct shortwave and longwave radiation."""
    wind_dropoff: bool = False
    """Whether to apply a coastal wind speed reduction to mimic nearshore wind drop-
    off."""
    restoring_forces: list[str] | None = None
    """The variables to create the restoring forces for."""
    coarse_grid_mode: str = "auto"
    """Specifies whether to interpolate onto grid coarsened by a factor of two."""
    model_reference_date: datetime = datetime(2000, 1, 1)
    """Reference date for the model."""
    use_dask: bool = False
    """Whether to use dask for processing."""
    chunks: dict[str, int] | None = None
    """Dask chunk sizes for lat/lon surface-forcing sources; default None."""
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
    regrid_method: str = "auto"
    """Horizontal regrid engine: ``"auto"`` (xESMF if installed, else scipy),
    ``"xesmf"``, or ``"scipy"``."""
    prefill: str | None = None
    """Source-side fill applied before regridding. ``None`` (default) applies no
    whole-domain fill: with xESMF the regrid is masked bilinear with inverse-distance
    destination extrapolation; without xESMF the source is nearest-neighbor pre-filled
    before scipy interpolation. Set to ``"2d_lateral_fill"`` (legacy AMG Poisson),
    ``"nearest_neighbor"``, ``"inverse_dist"``, ``"nearest_s2d"``, or ``"creep_fill"``
    to fill the whole-domain source first (the last three require xESMF)."""
    prefill_kwargs: dict | None = None
    """Method-specific keyword arguments for ``prefill`` (e.g. ``num_src_pnts`` /
    ``dist_exponent`` for ``"inverse_dist"``)."""
    extrap_method: str | None = None
    """xESMF destination extrapolation on the default no-prefill path; defaults to
    ``"inverse_dist"``. Ignored when a ``prefill`` is set or on the scipy path."""
    extrap_kwargs: dict | None = None
    """Method-specific keyword arguments for ``extrap_method``."""
    fallback_source: RawDataSource | None = None
    """Optional secondary source, used wherever ``source`` has no coverage.

    Same shape as ``source``. Only valid for ``type="physics"``. See the
    "Layering two sources" notes on this class."""
    blend_width_km: float = 0.0
    """Width of the transition between the two sources, in km.

    ``0.0`` (default) is a hard edge. A positive value ramps the primary source's
    weight from 0 at the edge of its footprint to 1 that far inside it, so the two
    sources are averaged across the band. The ramp is one-sided, running inward
    through the primary's own territory -- see :meth:`_compute_blend_weights`."""
    blend_options: dict | None = None
    """Tuning for the two-source merge. Recognized keys:

    ``"time_align"`` (``"exact"`` | ``"nearest"`` | ``"linear"``, default
    ``"exact"``), ``"time_tolerance"`` (for ``"nearest"``),
    ``"on_missing_primary_var"`` (``"error"`` | ``"fallback"``, default
    ``"error"``), ``"blend_profile"`` (``"smoothstep"`` | ``"linear"``, default
    ``"smoothstep"``; only used when ``blend_width_km`` > 0)."""

    ds: xr.Dataset = field(init=False, repr=False)
    """An xarray Dataset containing post-processed variables ready for input into
    ROMS."""
    use_coarse_grid: bool = field(init=False, repr=False)
    """Whether data is interpolated onto grid coarsened by factor 2."""

    def __post_init__(self):
        self._input_checks()
        # Resolve/validate the regrid engine + source-prefill + extrapolation options
        # once (mirrors BoundaryForcing); derived decisions are read off self._regrid.
        self._regrid = RegridConfig.from_options(
            prefill=self.prefill,
            prefill_kwargs=self.prefill_kwargs,
            regrid_method=self.regrid_method,
            extrap_method=self.extrap_method,
            extrap_kwargs=self.extrap_kwargs,
            xesmf_available=_xesmf_available(),
        )
        # Persist the resolved prefill as a plain string (or None) for the YAML round-trip.
        self.prefill = (
            None if self._regrid.prefill is None else str(self._regrid.prefill)
        )

        data = self._get_data_for(self.source)
        fallback_data = (
            self._get_data_for(self.fallback_source)
            if self.fallback_source is not None
            else None
        )

        if self.coarse_grid_mode == "always":
            use_coarse_grid = True
        elif self.coarse_grid_mode == "never":
            use_coarse_grid = False
        elif self.coarse_grid_mode == "auto":
            # Decided from the primary source alone. A limited-extent source only
            # exists because it is finer than the global one, so the primary sets
            # the resolution the output can support; taking the fallback's answer
            # would coarsen away exactly what the primary was added for. There is
            # also only one output file and one `interp_bulk_frc` flag, so there is
            # no way to represent "coarse here, fine there".
            use_coarse_grid = self._determine_coarse_grid_usage(data)
            if fallback_data is not None:
                self._log_coarse_grid_disagreement(
                    fallback_data, use_coarse_grid, self.fallback_source["name"]
                )
        self.use_coarse_grid = use_coarse_grid

        if self.type == "bgc":
            if self.source["name"] == "MBL_co2":
                cppdefs_flags = set()
                cppdefs_flags.add("PCO2AIR_FORCING")
        elif self.type == "restoring":
            opt_file = "cppdefs.opt"
            cppdefs_flags = set()

            for var in self.restoring_forces:
                if var == "sss":
                    cppdefs_flags.add("SFLX_CORR")
                if var == "sDIC":
                    cppdefs_flags.add("CFLX_CORR")
                if var == "sALK":
                    cppdefs_flags.add("CFLX_CORR")

        grid_desc = "grid coarsened by factor 2" if use_coarse_grid else "fine grid"

        if self.type in ["physics", "bgc"]:
            if self.type == "physics":
                nml_key, nml_group = "interp_bulk_frc", "SURF_FRC_SETTINGS"
            else:
                nml_key, nml_group = "interp_bgc_frc", "BGC_SETTINGS"
            nml_value = ".true." if use_coarse_grid else ".false."
            logging.info(
                "Data will be interpolated onto the %s. Remember to set "
                "`%s = %s` in the `&%s` group of your ROMS `namelist.nml` file.",
                grid_desc,
                nml_key,
                nml_value,
                nml_group,
            )
            if self.source["name"] == "MBL_co2":
                logging.info(
                    "Time-varying CO2 values being used."
                    "Remember to define the following flags in your `cppdefs.opt` file: %s`.",
                    cppdefs_flags,
                )
        elif self.type == "restoring":
            logging.info(
                "Data will be interpolated onto the %s. "
                "Restoring data being created for %s. "
                "Remember to define the following flags in your `%s` file: %s`.",
                grid_desc,
                self.restoring_forces,
                opt_file,
                cppdefs_flags,
            )

        target_coords = get_target_coords(self.grid, self.use_coarse_grid)
        self.target_coords = target_coords

        self._source_is_curvilinear = isinstance(data, CurvilinearDataset)
        processed_fields = self._build_processed_fields(
            data,
            target_coords,
            self.source["name"],
            # A limited-extent primary is *expected* not to cover the whole grid;
            # that is the point of having a fallback.
            coverage_required=fallback_data is None,
            suppress_extrapolation=fallback_data is not None,
        )

        if fallback_data is not None:
            self._primary_coverage_fraction = coverage_fraction(
                next(iter(processed_fields.values())), target_coords["mask"]
            )
            logging.info(
                "Primary source %s covers %.1f%% of the wet target points; the "
                "remainder comes from the fallback source %s.",
                self.source["name"],
                100 * self._primary_coverage_fraction,
                self.fallback_source["name"],
            )
            fallback_fields = self._build_processed_fields(
                fallback_data,
                target_coords,
                self.fallback_source["name"],
                # The fallback is the guarantee that the grid is covered at all, so
                # its coverage check stays on.
                coverage_required=True,
            )
            self._align_radiation_time_conventions(
                data,
                fallback_data,
                processed_fields,
                fallback_fields,
                self.fallback_source["name"],
            )
            self._check_blend_compatibility(processed_fields, fallback_fields)
            blend_options = self.blend_options or {}
            processed_fields = layer_fields(
                processed_fields,
                fallback_fields,
                weights=self._blend_weights(processed_fields, target_coords),
                on_missing_primary_var=blend_options.get(
                    "on_missing_primary_var", "error"
                ),
                time_align=blend_options.get("time_align", "exact"),
                time_tolerance=blend_options.get("time_tolerance"),
            )

        # rotation of velocities
        if "uwnd" in processed_fields and "vwnd" in processed_fields:
            processed_fields["uwnd"], processed_fields["vwnd"] = rotate_velocities(
                processed_fields["uwnd"],
                processed_fields["vwnd"],
                target_coords["angle"],
            )

        if self.type == "physics":
            if self.wind_dropoff:
                (
                    processed_fields["uwnd"],
                    processed_fields["vwnd"],
                ) = self._apply_wind_correction(
                    processed_fields["uwnd"], processed_fields["vwnd"]
                )

        if self.type == "bgc" and self.source["name"] != "MBL_co2":
            processed_fields = compute_missing_surface_bgc_variables(processed_fields)

        # Reorder dimensions
        for var_name in processed_fields:
            processed_fields[var_name] = transpose_dimensions(
                processed_fields[var_name]
            )

        d_meta = get_variable_metadata()

        ds = self._write_into_dataset(processed_fields, data, d_meta)

        if not self.bypass_validation:
            self._validate(ds)

        # Shift radiation time for sources whose radiation is a backward-hourly
        # mean. Driven by the dataset class's `rad_time_offset` rather than the
        # source name, so a source with instantaneous radiation gets no shift.
        rad_time_offset = getattr(data, "rad_time_offset", None)
        if self.type == "physics" and rad_time_offset:
            # The primary owns the emitted rad_time; any fallback has already been
            # shifted onto its convention by `_align_radiation_time_conventions`.
            ds = self._apply_rad_time(ds, rad_time_offset)

        # substitute NaNs over land by a fill value to avoid blow-up of ROMS
        for var_name in ds.data_vars:
            ds[var_name] = substitute_nans_by_fillvalue(ds[var_name])

        self.ds = ds

    def _normalize_source_dict(
        self, source: dict[str, Any], label: str = "source"
    ) -> dict[str, Any]:
        """Validate one source dict and fill in its defaults.

        Returns a new dict rather than mutating in place, so the same normalization
        can be applied to more than one source without the order of the two
        mattering. Factored out of :meth:`_input_checks` for exactly that reason:
        any divergence between how two sources are normalized would surface as a
        YAML round-trip inequality rather than as an obvious bug.

        Parameters
        ----------
        source : RawDataSource
            The source dict to normalize.
        label : str
            Field name, used in error messages.

        Returns
        -------
        dict
            The normalized source dict.
        """
        if "name" not in source:
            raise ValueError(f"`{label}` must include a 'name'.")

        source = dict(source)
        if "path" not in source:
            if source["name"] == "ERA5":
                # ERA5's default path (the ARCO cloud archive) is applied
                # later, by `resolve_era5_source` in `_get_data_for`.
                source["path"] = None
            elif source["name"] == "CONUS404":
                # CONUS404's default path (the OSN zarr store) is applied later,
                # by `_get_data_for`, so an explicitly provided path round-trips
                # through `to_yaml` exactly as given.
                source["path"] = None
            elif source["name"] == "MBL_co2":
                logging.info(
                    "No path specified for MBL_co2 source; defaulting to the MBL dataset from GML, NOAA."
                )
                source["path"] = DEFAULT_MBL_co2_PATH
            elif source["name"] == "SODA":
                logging.info(
                    "No path specified for SODA source; defaulting to the OceanSODA-ETHZ v2025 dataset from NCEI, NOAA."
                )
                source["path"] = DEFAULT_SODA_PATH
            else:
                raise ValueError(f"`{label}` must include a 'path'.")

        # Set 'climatology' to False if not provided
        source["climatology"] = source.get("climatology", False)
        return source

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
        if self.type not in ["physics", "bgc", "restoring"]:
            raise ValueError("`type` must be either 'physics', 'bgc', or 'restoring'.")

        self.source = self._normalize_source_dict(self.source, label="source")

        if self.fallback_source is not None:
            if self.type != "physics":
                raise ValueError(
                    "`fallback_source` is only supported for `type='physics'`. "
                    "BGC surface forcing is climatological and restoring forcing "
                    "drops the time axis, so neither has a layering use case."
                )
            self.fallback_source = self._normalize_source_dict(
                self.fallback_source, label="fallback_source"
            )
            if self.source == self.fallback_source:
                # Not an error: it is the natural way to check that layering is
                # transparent. But it does double the work for no result, so say so.
                logging.info(
                    "`source` and `fallback_source` are identical (%s); the merge "
                    "will be a no-op and the source will be read twice.",
                    self.source["name"],
                )
            if self.source["climatology"] or self.fallback_source["climatology"]:
                raise ValueError(
                    "Layering is not supported for climatologies: a monthly "
                    "climatology and a time-varying record cannot share one time "
                    "axis. Set 'climatology' to False on both sources."
                )
            if self.prefill is not None:
                raise ValueError(
                    f"`prefill={self.prefill!r}` fills the whole primary source "
                    "before regridding, which makes it NaN-free and so erases the "
                    "coverage footprint that `fallback_source` exists to fill -- the "
                    "fallback would silently contribute nothing. Use `prefill=None` "
                    "together with `fallback_source`."
                )
            if self.blend_width_km < 0:
                raise ValueError(
                    f"`blend_width_km` must be >= 0, got {self.blend_width_km}."
                )
            valid_blend_keys = {
                "time_align",
                "time_tolerance",
                "on_missing_primary_var",
                "blend_profile",
            }
            unknown = set(self.blend_options or {}) - valid_blend_keys
            if unknown:
                raise ValueError(
                    f"Unknown `blend_options` keys: {sorted(unknown)}. Valid keys "
                    f"are {sorted(valid_blend_keys)}."
                )
        elif self.blend_width_km:
            raise ValueError(
                "`blend_width_km` only applies when `fallback_source` is set."
            )

        # Validate 'coarse_grid_mode'
        valid_modes = ["auto", "always", "never"]
        if self.coarse_grid_mode not in valid_modes:
            raise ValueError(
                f"`coarse_grid_mode` must be one of {valid_modes}, but got '{self.coarse_grid_mode}'."
            )

        # Check if restoring variables are accepted
        if self.type == "restoring":
            if not self.restoring_forces:
                raise ValueError(
                    "When type='restoring', `restoring_forces` must be defined."
                )

            valid_vars = ["sss", "sDIC", "sALK"]
            for var in self.restoring_forces:
                if var not in valid_vars:
                    raise ValueError(
                        f"`restoring_forces` must be any of {valid_vars}, but got '{var}'."
                    )
            has_dic = "sDIC" in self.restoring_forces
            has_alk = "sALK" in self.restoring_forces
            has_sss = "sss" in self.restoring_forces

            if has_dic != has_alk:
                raise ValueError(
                    "'sDIC' and 'sALK' must both be present or both absent"
                )

            if has_dic and has_sss:
                raise ValueError(
                    "'sss' must be called separately from 'sDIC' and 'sALK'."
                )

        # The radiation correction is an ERA5-vs-observations ratio climatology
        # defined on the ERA5 0.25-degree lat/lon grid, and
        # `ERA5Correction.match_subdomain` selects by exact coordinate match --
        # which a curvilinear source cannot satisfy. (`_get_correction_data` would
        # also raise, but its message tells the user to switch to ERA5, which is
        # not the useful advice here.)
        if self.type == "physics" and self.source["name"] == "CONUS404":
            if self.correct_radiation and self.fallback_source is None:
                raise ValueError(
                    "`correct_radiation=True` is not supported for the CONUS404 "
                    "source: the ERA5 correction climatology is defined on the "
                    "ERA5 lat/lon grid and is matched by exact coordinate value, "
                    "which a curvilinear source cannot satisfy. Set "
                    "`correct_radiation=False`."
                )
            if self.source["climatology"]:
                raise ValueError(
                    "CONUS404 provides hourly time-varying data; "
                    "'climatology' must be 'False'."
                )

        # Check that climatology is false for t-varying co2
        if self.type == "bgc" and self.source["name"] == "MBL_co2":
            if self.source["climatology"]:
                raise ValueError(
                    "When 'name' is 'MBL_co2', time-varying xco2 data is expected. 'climatology' must be 'False'"
                )

        # Check that climatology is false for restoring of 'sDIC' and 'sALK'
        if self.type == "restoring" and self.source["name"] == "SODA":
            if self.source["climatology"]:
                raise ValueError(
                    "When 'name' is 'SODA', monthly `dic` and `talk` data is expected. 'climatology' must be 'False'"
                )

    def _determine_coarse_grid_usage(self, data):
        """Determine if coarse grid interpolation should be used based on the resolution
        of the dataset and the target grid.

        Parameters
        ----------
        data : object
            The dataset object containing the data to be analyzed for grid spacing.

        Returns
        -------
        use_coarse_grid : bool
            Whether to use the coarse grid or not.
        """
        # Get the target coordinates and select the subdomain of the data
        target_coords = get_target_coords(self.grid, use_coarse_grid=False)
        data_coords = data.choose_subdomain(
            target_coords, buffer_points=1, return_coords_only=True
        )

        # Compute minimal grid spacing in the data subdomain
        min_grid_spacing_data = data.compute_minimal_grid_spacing(data_coords)

        # Compute the maximum grid spacing in the ROMS grid
        max_grid_spacing = max((1 / self.grid.ds.pm).max(), (1 / self.grid.ds.pn).max())

        # Determine whether to use coarse grid based on grid spacing comparison
        if 2 * max_grid_spacing < min_grid_spacing_data:
            use_coarse_grid = True
        else:
            use_coarse_grid = False

        return use_coarse_grid

    def _build_processed_fields(
        self,
        data,
        target_coords: dict[str, xr.DataArray],
        source_name: str,
        *,
        coverage_required: bool = True,
        suppress_extrapolation: bool = False,
    ) -> dict[str, xr.DataArray]:
        """Take one source dataset from raw to ROMS-grid fields.

        Everything from here is per-source: subdomain selection, the coverage
        guard, the source fills, the lateral regrid, and that source's own
        radiation correction. Wind rotation and the coastal wind drop-off are
        deliberately *not* here -- they are target-grid operations that belong
        after any combination of sources, and rotation is linear so its position
        relative to a merge is immaterial anyway.

        Parameters
        ----------
        data : LatLonDataset or CurvilinearDataset
            The source dataset, mutated in place by the subdomain selection and
            any fill.
        target_coords : dict
            Target grid coordinates from :func:`get_target_coords`.
        source_name : str
            Used in the coverage-guard error message.
        coverage_required : bool, optional
            Whether the source must cover the whole target grid. True (default)
            for a sole source. A limited-extent source that is only expected to
            cover part of the domain should pass False.
        suppress_extrapolation : bool, optional
            Disable destination extrapolation for this source, so target points
            outside its coverage stay NaN. Required for a lat/lon primary that is
            to be layered over a fallback: otherwise ``"inverse_dist"`` fills the
            whole domain from it, the coverage gaps vanish, and the fallback
            silently contributes nothing. A curvilinear source never extrapolates,
            so this is a no-op there.

        Returns
        -------
        dict[str, xr.DataArray]
            Regridded fields on the ROMS grid, keyed by ROMS variable name.
        """
        # Unifies chunking on the *source* grid (data.ds). Output self.ds is built
        # later from the regridded fields, so its chunks follow the regrid ops.
        data.choose_subdomain(
            target_coords,
            # unchunk_lateral_dims=True required for lateral fill, consider trying False if lateral fill ever becomes optional
            unchunk_lateral_dims=True,
        )
        # Enforce double precision to ensure reproducibility
        data.convert_to_float64()

        regrid = self._regrid
        if isinstance(data, CurvilinearDataset):
            # A curvilinear source is regridded by `LateralRegridFromROMS`, which
            # never consults `RegridConfig`. Coverage checking is skipped because
            # `check_source_coverage` compares 1D coordinate axes, which for a
            # projected source are index coordinates in metres; and the fills are
            # skipped because such a source carries no masked cells. The regrid
            # does not extrapolate, so a target point outside the source footprint
            # comes back as NaN.
            self._warn_if_regrid_options_set_for_curvilinear()
        else:
            # On the no-prefill xESMF path, destination extrapolation would silently fill
            # points outside the source coverage. Guard against a grid that outruns the
            # data (coastal gaps *within* coverage are still filled by the masked regrid).
            if coverage_required and regrid.extrap_is_active:
                check_source_coverage(data, target_coords, source_name)

            # Whole-domain source prefill when requested, else a nearest-neighbor
            # pre-fill on the scipy path so interpolation cannot propagate NaNs.
            #
            # Both are safe to run even when extrapolation is suppressed, and both
            # need to. `apply_source_prefill` is already a no-op here, since
            # `_input_checks` rejects an explicit `prefill` alongside
            # `fallback_source` (a whole-domain fill really would erase the
            # coverage footprint). `apply_scipy_fallback_fill` only fills masked
            # cells *within* the source grid, and `DataArray.interp` returns NaN
            # beyond the source's coordinate range regardless, so the footprint
            # survives. Skipping it would let the source's land NaNs propagate into
            # coastal target points, which would then be taken from the fallback
            # instead of the primary -- a silent, and wrong, source swap.
            apply_source_prefill(data, regrid, self.prefill_kwargs)
            apply_scipy_fallback_fill(data, regrid)

        self._set_variable_info(data)
        var_names = {
            var: {"name": name}
            for d in [data.var_names, data.opt_var_names]
            for var, name in d.items()
            if name in data.ds.data_vars
        }

        processed_fields = self._regrid_laterally(
            data,
            target_coords,
            var_names,
            suppress_extrapolation=suppress_extrapolation,
        )

        # The radiation correction is welded to its source: the climatology is
        # regridded through that source's own lat/lon grid and borrows its mask, so
        # it has to be applied here rather than after any merge.
        if self.type == "physics" and self.correct_radiation:
            (
                processed_fields["swrad"],
                processed_fields["lwrad"],
            ) = self._apply_radiation_corrections(
                processed_fields["swrad"], processed_fields["lwrad"], data, source_name
            )

        return processed_fields

    def _log_coarse_grid_disagreement(
        self, fallback_data, use_coarse_grid: bool, fallback_name: str
    ) -> None:
        """Note when the two sources would choose different output resolutions.

        They usually will: the whole reason for a limited-extent primary is that
        it is finer than the fallback. Logged rather than raised, since the
        primary's answer is the intended one.
        """
        fallback_choice = self._determine_coarse_grid_usage(fallback_data)
        if fallback_choice != use_coarse_grid:
            logging.info(
                "The two sources disagree about the output grid: %s implies the %s "
                "and %s implies the %s. Using the primary source's choice (%s). "
                "Set `coarse_grid_mode` explicitly to override.",
                self.source["name"],
                "coarse grid" if use_coarse_grid else "fine grid",
                fallback_name,
                "coarse grid" if fallback_choice else "fine grid",
                "coarse grid" if use_coarse_grid else "fine grid",
            )

    def _align_radiation_time_conventions(
        self, data, fallback_data, primary_fields, fallback_fields, fallback_name: str
    ) -> None:
        """Put the fallback's radiation on the primary's time convention.

        Radiation carries a per-source ``rad_time_offset``: an hourly-mean product
        labels its flux at the end of the averaging interval, an instantaneous one
        does not. The output file has a single ``rad_time``, taken from the primary,
        so if the fallback's convention differs its radiation has to be shifted to
        match before the merge. Applying one shift to a field that is half
        hourly-mean and half instantaneous would be wrong over half the domain.

        Modifies ``fallback_fields`` in place.
        """
        primary_offset = getattr(data, "rad_time_offset", None) or timedelta(0)
        fallback_offset = getattr(fallback_data, "rad_time_offset", None) or timedelta(
            0
        )
        if primary_offset == fallback_offset:
            return

        shift = fallback_offset - primary_offset
        logging.info(
            "The two sources label radiation differently (%s offset %s, %s offset "
            "%s); shifting the fallback's radiation by %s onto the primary's "
            "convention before merging.",
            self.source["name"],
            primary_offset,
            fallback_name,
            fallback_offset,
            shift,
        )
        for var_name in ("swrad", "lwrad"):
            if var_name not in fallback_fields or var_name not in primary_fields:
                continue
            field = fallback_fields[var_name]
            if "time" not in field.dims:
                continue
            shifted = field.assign_coords(time=field["time"] - np.timedelta64(shift))
            fallback_fields[var_name] = shifted.interp(
                time=primary_fields[var_name]["time"]
            )

    def _check_blend_compatibility(
        self,
        primary_fields: dict[str, xr.DataArray],
        fallback_fields: dict[str, xr.DataArray],
    ) -> None:
        """Compare the two sources where they overlap, and log the difference.

        The cheapest high-value check available: one two-dimensional,
        single-timestep reduction per variable over the region where both sources
        are valid. Every unit or convention mismatch between two independently
        written ``post_process`` implementations shows up here immediately -- a
        Kelvin-versus-Celsius slip as a ~273 offset, millimetres-versus-centimetres
        as a factor of 10, net-versus-downward shortwave as ~+6%, and a wind
        rotation error as a large spread with a near-zero mean.

        Logged rather than raised for now, because a legitimate difference between
        two reanalyses is not an error and no threshold would be defensible without
        data behind it.
        """
        for var_name in sorted(set(primary_fields) & set(fallback_fields)):
            p_units = primary_fields[var_name].attrs.get("units")
            f_units = fallback_fields[var_name].attrs.get("units")
            if p_units and f_units and p_units != f_units:
                raise ValueError(
                    f"The two sources disagree on the units of {var_name!r}: the "
                    f"primary reports {p_units!r} and the fallback {f_units!r}. "
                    f"Both must emit the ROMS surface-forcing conventions."
                )

        logging.info("Source comparison over the overlap region (primary - fallback):")
        for var_name in sorted(set(primary_fields) & set(fallback_fields)):
            p = primary_fields[var_name]
            f = fallback_fields[var_name]
            if "time" in p.dims:
                p = p.isel(time=0)
            if "time" in f.dims:
                f = f.isel(time=0)
            try:
                diff = (p - f).where(p.notnull() & f.notnull())
                mean = float(diff.mean().compute())
                std = float(diff.std().compute())
            except Exception as exc:  # pragma: no cover - diagnostic only
                logging.info("  %-6s could not be compared (%s)", var_name, exc)
                continue
            logging.info("  %-6s mean %+.4g   std %.4g", var_name, mean, std)

    def _blend_weights(
        self,
        primary_fields: dict[str, xr.DataArray],
        target_coords: dict[str, xr.DataArray],
    ) -> "xr.DataArray | None":
        """Build the primary-source weight field, or ``None`` for a hard edge."""
        if not self.blend_width_km:
            return None

        # A point counts as covered only where *every* primary field has a value.
        # Taking the intersection is the conservative reading: a variable missing
        # there would fall back anyway, so including the point in the feather band
        # would only smooth toward a value the merge does not use.
        footprint = None
        for da in primary_fields.values():
            valid = (da.isel(time=0) if "time" in da.dims else da).notnull()
            footprint = valid if footprint is None else (footprint & valid)

        return self._compute_blend_weights(
            footprint, target_coords, self.blend_width_km
        )

    def _compute_blend_weights(
        self,
        primary_footprint: xr.DataArray,
        target_coords: dict[str, xr.DataArray],
        width_km: float,
    ) -> xr.DataArray:
        """Ramp the primary's weight from 0 at its footprint edge to 1 inside it.

        The ramp is **one-sided**: it runs inward from the edge through the
        primary's own territory, rather than straddling the seam. That is the right
        way round for a nested product. The outer rim of a regional model's domain
        is its lateral relaxation zone, where it is nudged toward the reanalysis
        that drove it and is least independent of it -- CONUS404 is ERA5-forced,
        with interior spectral nudging besides. So preferring the global source
        there gives up almost nothing, while the reverse (letting the regional
        product bleed outward past its own domain) would extrapolate it into
        territory it never simulated.

        Distance is computed with :func:`scipy.ndimage.distance_transform_edt` on
        the footprint mask, scaled by the grid's mean cell size. That is O(N) --
        unlike :func:`~roms_tools.setup.utils.min_dist_to_land`, whose
        every-point-to-every-point search is fine for a coastline but would take
        minutes on a large domain. It is exact for a uniform grid and approximate
        where the spacing varies appreciably, which is well inside the tolerance of
        a smoothing profile.

        Parameters
        ----------
        primary_footprint : xr.DataArray
            2D boolean, True where the primary source has data.
        target_coords : dict
            Target grid coordinates (unused; kept for interface stability).
        width_km : float
            Distance over which the primary's weight rises from 0 to 1.

        Returns
        -------
        xr.DataArray
            Weights in [0, 1] on ``(eta_rho, xi_rho)``, time-invariant.
        """
        from scipy.ndimage import distance_transform_edt

        footprint = np.asarray(primary_footprint.values, dtype=bool)

        if footprint.all():
            # No seam anywhere, so nothing to feather. This case must be caught
            # explicitly: with no zeros in the input, `distance_transform_edt`
            # measures to the array border instead, which would paint a spurious
            # feather band around the whole domain.
            logging.info(
                "The primary source covers the entire domain, so blend_width_km "
                "has no effect."
            )
            return xr.ones_like(primary_footprint, dtype=float)

        # Mean cell size, in metres. `pm`/`pn` are the inverse spacings along xi
        # and eta on the fine grid; the coarse grid doubles them.
        factor = 2.0 if self.use_coarse_grid else 1.0
        dx = factor * float((1 / self.grid.ds.pm).mean())
        dy = factor * float((1 / self.grid.ds.pn).mean())

        # Distance from each covered point to the nearest uncovered one. Array
        # order is (eta, xi), hence sampling=(dy, dx). Points outside the footprint
        # come back as 0, which is what we want: weight 0, i.e. pure fallback.
        distance_m = distance_transform_edt(footprint, sampling=(dy, dx))

        w = np.clip(distance_m / (width_km * 1000.0), 0.0, 1.0)

        profile = (self.blend_options or {}).get("blend_profile", "smoothstep")
        if profile == "smoothstep":
            # Hermite smoothstep: continuous first derivative at both ends, so the
            # blend introduces no kink in the field's gradient. A linear ramp is
            # continuous in value but not slope, which shows up in derived
            # quantities like wind stress curl.
            w = w * w * (3.0 - 2.0 * w)
        elif profile != "linear":
            raise ValueError(
                f"Unknown blend_profile {profile!r}; expected 'smoothstep' or 'linear'."
            )

        weights = xr.DataArray(w, dims=primary_footprint.dims)
        weights.attrs["long_name"] = "primary source weight"
        return weights

    def _warn_if_regrid_options_set_for_curvilinear(self) -> None:
        """Log a note when prefill/regrid options are set but the source is curvilinear.

        Mirrors ``InitialConditions._warn_if_regrid_options_set_for_roms``: the
        options are accepted but have no effect, because a curvilinear source is
        regridded by :class:`~roms_tools.regrid.LateralRegridFromROMS`, which does
        not consult ``RegridConfig``. Noted in the log rather than raised, so a
        blueprint that sets them globally still runs.
        """
        if any(
            opt is not None
            for opt in (
                self.prefill,
                self.prefill_kwargs,
                self.extrap_method,
                self.extrap_kwargs,
            )
        ):
            logging.info(
                "prefill/extrap_method apply to lat/lon sources only; ignoring them "
                "for the curvilinear %s source, which is regridded with "
                "non-extrapolating xESMF bilinear.",
                self.source["name"],
            )

    def _regrid_laterally(
        self,
        data: LatLonDataset | CurvilinearDataset,
        target_coords: dict[str, xr.DataArray],
        var_names: dict[str, dict[str, str]],
        *,
        suppress_extrapolation: bool = False,
    ) -> dict[str, xr.DataArray]:
        """Regrid every source variable onto the ROMS grid.

        A curvilinear source goes through
        :class:`~roms_tools.regrid.LateralRegridFromROMS` (xESMF, curvilinear
        source, ``unmapped_to_nan=True``, no extrapolation), mirroring the
        ROMS-source branch in ``InitialConditions._regrid_laterally``. A lat/lon
        source takes the configured engine as before.

        Wind components are regridded as scalars here and rotated onto the ROMS
        grid angle by the caller.
        """
        processed_fields: dict[str, xr.DataArray] = {}

        if isinstance(data, CurvilinearDataset):
            # xESMF locates the source grid by the names "lat"/"lon".
            rename = {
                data.coord_names["latitude"]: "lat",
                data.coord_names["longitude"]: "lon",
            }
            rename = {k: v for k, v in rename.items() if k != v}

            # dict.fromkeys keeps insertion order while de-duplicating the several
            # var_names entries that may alias one source variable.
            source_vars = list(
                dict.fromkeys(var_names[var]["name"] for var in var_names)
            )
            ds_in = data.ds[source_vars]
            if rename:
                ds_in = ds_in.rename(rename)

            regridder = LateralRegridFromROMS(ds_in, target_coords)
            try:
                regridded = regridder.apply(ds_in)
            finally:
                regridder.destroy()

            for var_name in var_names:
                processed_fields[var_name] = regridded[var_names[var_name]["name"]]
            return processed_fields

        regrid = self._regrid
        # On the default (no-prefill) xESMF path, use the source "mask" for masked
        # bilinear regridding; a set prefill / the scipy path leaves the source
        # already NaN-free, so no mask is needed (plain bilinear / scipy interp).
        source_mask = select_source_mask(
            data.ds, is_vector=False, use_xesmf=regrid.use_xesmf, prefill=regrid.prefill
        )
        lateral_regrid = build_lateral_regridder(
            target_coords, data, regrid, source_mask
        )
        if suppress_extrapolation and lateral_regrid.use_xesmf:
            # Rebuild without destination extrapolation, so points outside the
            # source's coverage come back as NaN for the fallback to fill.
            lateral_regrid = LateralRegridToROMS(
                target_coords,
                data.dim_names,
                source_ds=data.ds,
                use_xesmf=True,
                source_mask=source_mask,
                extrap_method=None,
            )
        for var_name in var_names:
            processed_fields[var_name] = lateral_regrid.apply(
                data.ds[var_names[var_name]["name"]]
            )
        return processed_fields

    def _get_data_for(
        self, source: dict[str, Any]
    ) -> "LatLonDataset | CurvilinearDataset":
        """Build the source-dataset object for one normalized source dict.

        Parameterized on the given source dict rather than reading
        ``self.source``, so the same dispatch can serve any number of sources.
        Note that the ERA5 and CONUS404 branches write a resolved default path
        back into the dict they are handed, which the caller relies on for the
        YAML round-trip.
        """
        data_dict: dict[str, Any] = {
            "filename": source["path"],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "climatology": source["climatology"],
            "use_dask": self.use_dask,
            "chunks": self.chunks,
            "initial_slice_bounds": self.initial_slice_bounds,
            "start_time_pad": self.start_time_pad,
            "end_time_pad": self.end_time_pad,
        }

        data: LatLonDataset | CurvilinearDataset
        if self.type == "physics":
            if source["name"] == "ERA5":
                # Add 1 hr since radiation time will shift by 1 hr
                if data_dict["end_time"] is not None:
                    data_dict["end_time"] = data_dict["end_time"] + timedelta(hours=1)
                resolved_path, is_arco, dataset_cls = resolve_era5_source(
                    source["path"]
                )
                if not source["path"]:
                    # Only rewrite when defaulting -- an explicitly provided
                    # path (str or Path) is left exactly as given, so
                    # the source dict round-trips (e.g. through `to_yaml`)
                    # without silently changing its type.
                    logging.info(
                        "No path specified for ERA5 source; defaulting to ARCO ERA5 dataset on Google Cloud."
                    )
                    source["path"] = resolved_path
                    data_dict["filename"] = resolved_path
                if is_arco and not self.use_dask:
                    raise ValueError(
                        "Cloud-based ERA5 access requires `use_dask=True`. Please enable Dask by setting `use_dask=True`."
                    )
                data = dataset_cls(**data_dict)
            elif source["name"] == "CONUS404":
                if not self._regrid.use_xesmf:
                    raise ValueError(
                        "CONUS404 is a curvilinear source and requires the xESMF "
                        "regrid engine; the scipy engine interpolates along 1D "
                        "coordinate axes and cannot handle a 2D source grid. "
                        "Install `roms-tools` via conda (which includes xesmf) and "
                        "leave `regrid_method='auto'`."
                    )
                # Add 1 hr since radiation time will shift by 1 hr
                if data_dict["end_time"] is not None:
                    data_dict["end_time"] = data_dict["end_time"] + timedelta(hours=1)
                if not source["path"]:
                    logging.info(
                        "No path specified for CONUS404 source; defaulting to the "
                        "HyTEST zarr store on the USGS/OSN pod (%s).",
                        DEFAULT_CONUS404_PATH,
                    )
                    source["path"] = DEFAULT_CONUS404_PATH
                    data_dict["filename"] = DEFAULT_CONUS404_PATH
                if str(data_dict["filename"]).startswith("s3://") and not self.use_dask:
                    raise ValueError(
                        "Cloud-based CONUS404 access requires `use_dask=True`. "
                        "Please enable Dask by setting `use_dask=True`."
                    )
                # `initial_slice_bounds` names lat/lon dimensions and is a no-op on
                # the zarr path; CurvilinearDataset logs and ignores it, so don't
                # pass it at all.
                data_dict.pop("initial_slice_bounds", None)
                data = CONUS404Dataset(**data_dict)
            else:
                raise ValueError(
                    'Only "ERA5" and "CONUS404" are valid options for '
                    'source["name"] when type is "physics".'
                )

        elif self.type == "bgc":
            if source["name"] == "CESM_REGRIDDED":
                data = CESMBGCSurfaceForcingDataset(**data_dict)
            elif source["name"] == "UNIFIED":
                data = UnifiedBGCSurfaceDataset(**data_dict)
            elif source["name"] == "MBL_co2":
                data = MBLco2Dataset(**data_dict)
            else:
                raise ValueError(
                    'Only "CESM_REGRIDDED", "UNIFIED", and "MBL_co2" are valid options for source["name"] when type is "bgc".'
                )

        elif self.type == "restoring":
            restoring_forces = self.restoring_forces or []
            if "sss" in restoring_forces:
                if source["name"] == "WOA":
                    data = WOARestoringSurfaceDataset(**data_dict)
                elif source["name"] == "UNIFIED":
                    data = UnifiedRestoringSurfaceDataset(**data_dict)
                else:
                    raise ValueError(
                        'Only "WOA" and "UNIFIED" are valid options for source["name"] when type is "restoring", and restoring_forces is ["sss"].'
                    )
            if "sDIC" in restoring_forces:
                if source["name"] == "SODA":
                    data = SODARestoringSurfaceDataset(**data_dict)
                else:
                    raise ValueError(
                        'Only "SODA" is a valid option for source["name"] when type is "restoring", and restoring_forces is ["sDIC", "sALK"].'
                    )

        return data

    def _get_correction_data(self, source_name: str):
        """Return the radiation-correction climatology for one source, or None.

        The correction is an ERA5-vs-observations ratio field, so it belongs to
        ERA5 alone. When layering, a source with no registered correction simply
        goes uncorrected -- applying ERA5's correction to another product would not
        be a correction. As the sole source it is still an error, since the user
        asked for a correction and would silently not get one.
        """
        if source_name == "ERA5":
            return ERA5Correction(use_dask=self.use_dask)
        if self.fallback_source is not None:
            logging.info(
                "No radiation correction is defined for %s, so its shortwave and "
                "longwave are used uncorrected; the correction still applies to the "
                "ERA5-sourced points. Expect a small step in swrad/lwrad at the "
                "boundary between the two sources.",
                source_name,
            )
            return None
        raise ValueError(
            "The 'correct_radiation' feature is currently only supported for 'ERA5' as the source. "
            "Please ensure your 'source' is set to 'ERA5' or implement additional handling for other sources."
        )

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
        None
            This method updates the instance attribute `variable_info` with the metadata dictionary for the variables.
        """
        default_info = {
            "location": "rho",
            "is_vector": False,
            "vector_pair": None,
            "is_3d": False,
        }

        # Define a dictionary for variable names and their associated information
        if self.type == "physics":
            variable_info = {
                "swrad": {**default_info, "validate": True},
                "lwrad": {**default_info, "validate": False},
                "Tair": {**default_info, "validate": False},
                "qair": {**default_info, "validate": True},
                "rain": {**default_info, "validate": False},
                "uwnd": {
                    "location": "rho",
                    "is_vector": True,
                    "vector_pair": "vwnd",
                    "is_3d": False,
                    "validate": True,
                },
                "vwnd": {
                    "location": "rho",
                    "is_vector": True,
                    "vector_pair": "uwnd",
                    "is_3d": False,
                    "validate": True,
                },
            }
        elif self.type == "bgc":
            if self.source["name"] == "MBL_co2":
                variable_info = {}
                for var_name in list(data.var_names.keys()) + list(
                    data.opt_var_names.keys()
                ):
                    variable_info[var_name] = default_info
                    if var_name == "xco2_air":
                        variable_info[var_name] = {**default_info, "validate": True}
                    else:
                        variable_info[var_name] = {**default_info, "validate": False}
            else:
                variable_info = {}
                for var_name in list(data.var_names.keys()) + list(
                    data.opt_var_names.keys()
                ):
                    variable_info[var_name] = {**default_info, "validate": False}
        elif self.type == "restoring":
            variable_info = {}
            for var_name in list(data.var_names.keys()) + list(
                data.opt_var_names.keys()
            ):
                variable_info[var_name] = default_info
                if var_name in ["sss", "sDIC", "sALK"]:
                    variable_info[var_name] = {**default_info, "validate": True}
                else:
                    variable_info[var_name] = {**default_info, "validate": False}

        self.variable_info = variable_info

    def _apply_radiation_corrections(
        self,
        swrad: xr.DataArray,
        lwrad: xr.DataArray,
        data: LatLonDataset,
        source_name: str,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Apply climatological corrections to shortwave and longwave radiation.

        The correction dataset is loaded and preprocessed once. The 12-month
        climatology is spatially regridded to the ROMS grid first (only 12
        spatial interpolations per variable), then lazily interpolated in time
        to match the full forcing time axis. Correction factors are rechunked
        to align with the radiation fields before multiplication so that dask
        can execute the multiply chunk-by-chunk during save().

        Parameters
        ----------
        swrad : xr.DataArray
            Shortwave radiation field to be corrected. Must include a ``time`` coordinate.
        lwrad : xr.DataArray
            Longwave radiation field to be corrected. Must include a ``time`` coordinate.
        data : LatLonDataset
            ERA5 dataset providing the mask and spatial coordinates used to
            align the correction data.
        source_name : str
            Name of the source these fields came from. A source with no registered
            correction is returned unchanged (only possible when layering).

        Returns
        -------
        swrad_corrected : xr.DataArray
            Shortwave radiation scaled by the SSR correction factor.
        lwrad_corrected : xr.DataArray
            Longwave radiation scaled by the STRD correction factor.
        """
        correction_data = self._get_correction_data(source_name)
        if correction_data is None:
            return swrad, lwrad

        coords_correction = {
            "lat": data.ds[data.dim_names["latitude"]],
            "lon": data.ds[data.dim_names["longitude"]],
        }
        # unchunk_lateral_dims=True required for lateral fill, consider trying False if lateral fill ever becomes optional
        correction_data.match_subdomain(coords_correction, unchunk_lateral_dims=True)
        correction_data.ds["mask"] = data.ds["mask"]
        correction_data.ds["time"] = correction_data.ds["time"].dt.days

        # Use the same regrid engine / source-fill choice as the main data so the
        # correction climatology is treated consistently.
        regrid = self._regrid
        apply_source_prefill(correction_data, regrid, self.prefill_kwargs)
        apply_scipy_fallback_fill(correction_data, regrid)

        source_mask = select_source_mask(
            correction_data.ds,
            is_vector=False,
            use_xesmf=regrid.use_xesmf,
            prefill=regrid.prefill,
        )

        # Spatial regrid first: only 12 interpolations per variable regardless of
        # the length of the forcing time series. lateral_regrid.apply() forces eager
        # compute on the 12-step climatology, which is acceptable (~MB of data).
        lateral_regrid = build_lateral_regridder(
            self.target_coords, correction_data, regrid, source_mask
        )
        time_dim = correction_data.dim_names["time"]

        swr_12 = lateral_regrid.apply(
            correction_data.ds[correction_data.var_names["swr_corr"]]
        )
        lwr_12 = lateral_regrid.apply(
            correction_data.ds[correction_data.var_names["lwr_corr"]]
        )

        # Wrap back to dask so that temporal interpolation builds a lazy graph
        # rather than materialising the full (N, ny, nx) output as numpy.
        if self.use_dask:
            swr_12 = swr_12.chunk({time_dim: len(swr_12[time_dim])})
            lwr_12 = lwr_12.chunk({time_dim: len(lwr_12[time_dim])})

        # Interpolate onto the forcing time axis in blocks (when using dask) so the
        # result stays chunked along time: this bounds peak memory for long forcing
        # records and makes slicing a few time steps (e.g. validation) cheap, instead
        # of producing a single (N, ny, nx) chunk.
        interp_chunk_size = _DEFAULT_CLIMATOLOGY_TIME_CHUNK if self.use_dask else None
        swr_corr_factor = interpolate_from_climatology(
            field=swr_12,
            time_dim=time_dim,
            time_coord=time_dim,
            time=swrad.time,
            interp_chunk_size=interp_chunk_size,
        )
        lwr_corr_factor = interpolate_from_climatology(
            field=lwr_12,
            time_dim=time_dim,
            time_coord=time_dim,
            time=lwrad.time,
            interp_chunk_size=interp_chunk_size,
        )

        # Rechunk time to match the radiation fields so that the element-wise
        # multiply is chunk-aligned and dask can execute it slice-by-slice.
        if self.use_dask:
            swr_corr_factor = swr_corr_factor.chunk(swrad.chunksizes)
            lwr_corr_factor = lwr_corr_factor.chunk(lwrad.chunksizes)

        return swrad * swr_corr_factor, lwrad * lwr_corr_factor

    def _apply_wind_correction(
        self, uwnd: xr.DataArray, vwnd: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Apply coastal wind drop-off correction to wind components.

        This correction reduces wind speed near the coastline by up to 40%,
        transitioning smoothly from full magnitude offshore using an
        exponential decay with an e-folding scale of 12.5 km.

        Reanalysis wind products often lack sufficient resolution to capture
        sharp coastal wind gradients caused by orography and land-sea contrasts.
        This method adjusts wind magnitude to better reflect these coastal effects.

        Parameters
        ----------
        uwnd : xr.DataArray
            Zonal (east-west) wind component on the ROMS grid.
        vwnd : xr.DataArray
            Meridional (north-south) wind component on the ROMS grid.

        Returns
        -------
        uwnd_corrected : xr.DataArray
            Corrected zonal wind component with reduced coastal values.
        vwnd_corrected : xr.DataArray
            Corrected meridional wind component with reduced coastal values.
        """
        # calculate the distance from each ocean point to the closest land point
        cdist = min_dist_to_land(
            self.target_coords["lon"].values,
            self.target_coords["lat"].values,
            self.target_coords["mask"].values,
        )

        # Compute a spatially varying scaling factor to reduce wind near the coast.
        # This uses an exponential decay with a 12.5 km e-folding scale,
        # reducing wind magnitude by up to 40% at the coastline.
        mult = 1 - 0.4 * np.exp(-0.08 * cdist / 1000)

        mult = xr.DataArray(data=mult, dims=["eta_rho", "xi_rho"])

        uwnd_corrected = mult * uwnd
        vwnd_corrected = mult * vwnd

        return uwnd_corrected, vwnd_corrected

    def _write_into_dataset(self, processed_fields, data, d_meta):
        # save in new dataset
        ds = xr.Dataset()

        for var_name in list(processed_fields.keys()):
            ds[var_name] = processed_fields[var_name].astype(np.float32)
            del processed_fields[var_name]
            ds[var_name].attrs["long_name"] = d_meta[var_name]["long_name"]
            ds[var_name].attrs["units"] = d_meta[var_name]["units"]

        ds = self._add_global_metadata(ds)

        # Convert the time coordinate to the format expected by ROMS
        ds, sfc_time = add_time_info_to_ds(
            ds, self.model_reference_date, data.climatology
        )

        if self.type == "physics":
            if getattr(data, "rad_time_offset", None):
                time_coords = [
                    "time",
                    "rad_time",
                ]
            else:
                time_coords = [
                    "time",
                ]
        elif self.type == "bgc":
            if self.source["name"] == "MBL_co2":
                time_coords = [
                    "xco2_time",
                ]
            else:
                time_coords = [
                    "iron_time",
                    "dust_time",
                    "nox_time",
                    "nhy_time",
                ]
        elif self.type == "restoring":
            time_coords = []
            for var in self.restoring_forces:
                if var == "sss":
                    time_coords.append("sss_time")
                if var == "sDIC":
                    time_coords.append("sDIC_time")
                if var == "sALK":
                    time_coords.append("sALK_time")
        for time_coord in time_coords:
            ds = ds.assign_coords({time_coord: sfc_time})

        if self.type == "bgc":
            ds = ds.drop_vars(["time"])

        if self.type == "restoring":
            ds = ds.drop_vars(["time"])

        variables_to_drop = ["lat_rho", "lon_rho", "lat_coarse", "lon_coarse"]
        existing_vars = [var_name for var_name in variables_to_drop if var_name in ds]
        ds = ds.drop_vars(existing_vars)

        return ds

    def _validate(self, ds):
        """Validates the dataset by checking for NaN values at wet points, which would
        indicate missing raw data coverage over the target domain.

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
        This check is applied to the first time step (`time=0`) of each variable in the provided dataset.
        """
        for var_name in ds.data_vars:
            if self.variable_info[var_name]["validate"]:
                # all variables are at rho-points
                mask = self.target_coords["mask"]
                nan_check(ds[var_name].isel(time=0), mask)

    def _apply_rad_time(self, ds, offset: timedelta):
        """Add a ``rad_time`` coordinate offset from ``time``.

        Sources whose radiation is an average over the preceding hour (ERA5's
        hourly ``ssr``/``strd``, or a differenced CONUS404 accumulation) label the
        flux at the end of its averaging interval, so the value is representative
        of the interval midpoint. ROMS reads radiation on ``rad_time`` to account
        for that.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to add ``rad_time`` to. Its ``time`` coordinate is already
            in days relative to ``model_reference_date``.
        offset : timedelta
            The source dataset's ``rad_time_offset``; negative for a
            backward-looking average.

        """
        # `time` is in relative days at this point, so express the offset in days.
        offset_days = offset.total_seconds() / 86400.0
        ds = ds.assign_coords(rad_time=("time", ds["time"].values + offset_days))
        ds.rad_time.attrs["long_name"] = ds.time.attrs["long_name"]
        ds.rad_time.attrs["units"] = ds.time.attrs["units"]

        return ds

    def _add_global_metadata(self, ds=None):
        if ds is None:
            ds = xr.Dataset()
        ds.attrs["title"] = "ROMS surface forcing file created by ROMS-Tools"
        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"
        ds.attrs["roms_tools_version"] = roms_tools_version
        ds.attrs["start_time"] = str(self.start_time)
        ds.attrs["end_time"] = str(self.end_time)
        ds.attrs["source"] = self.source["name"]
        ds.attrs["correct_radiation"] = str(self.correct_radiation)
        ds.attrs["wind_dropoff"] = str(self.wind_dropoff)
        ds.attrs["use_coarse_grid"] = str(self.use_coarse_grid)
        ds.attrs["model_reference_date"] = str(self.model_reference_date)
        if getattr(self, "_source_is_curvilinear", False):
            # A curvilinear source bypasses RegridConfig entirely, so recording
            # the resolved config here would misreport what actually ran: the
            # regrid is non-extrapolating xESMF bilinear with no source prefill.
            ds.attrs["prefill"] = "None"
            ds.attrs["regrid_method"] = "xesmf"
            ds.attrs["extrap_method"] = "None"
        else:
            ds.attrs["prefill"] = str(self.prefill)
            ds.attrs["regrid_method"] = "xesmf" if self._regrid.use_xesmf else "scipy"
            ds.attrs["extrap_method"] = str(self._regrid.effective_extrap)

        ds.attrs["type"] = self.type

        if self.fallback_source is not None:
            # `source` stays the primary name, for anything already parsing it.
            ds.attrs["layered"] = "True"
            ds.attrs["fallback_source"] = self.fallback_source["name"]
            ds.attrs["blend_method"] = (
                "feathered" if self.blend_width_km else "hard_edge"
            )
            ds.attrs["blend_width_km"] = str(self.blend_width_km)
            if self.blend_width_km:
                ds.attrs["blend_profile"] = (self.blend_options or {}).get(
                    "blend_profile", "smoothstep"
                )
            ds.attrs["coarse_grid_decided_by"] = self.source["name"]
            if self.correct_radiation:
                ds.attrs["radiation_correction_applied_to"] = ",".join(
                    name
                    for name in (self.source["name"], self.fallback_source["name"])
                    if name == "ERA5"
                )
            coverage = getattr(self, "_primary_coverage_fraction", None)
            if coverage is not None:
                ds.attrs["primary_coverage_fraction"] = f"{coverage:.4f}"
        else:
            ds.attrs["layered"] = "False"

        return ds

    def plot(
        self,
        var_name: str,
        time: int = 0,
        save_path: str | None = None,
    ) -> None:
        """Plot the specified surface forcing field for a given time slice.

        Parameters
        ----------
        var_name : str
            The name of the surface forcing field to plot. Options include:

            - "uwnd": 10 meter wind in x-direction.
            - "vwnd": 10 meter wind in y-direction.
            - "swrad": Downward short-wave (solar) radiation.
            - "lwrad": Downward long-wave (thermal) radiation.
            - "Tair": Air temperature at 2m.
            - "qair": Absolute humidity at 2m.
            - "rain": Total precipitation.
            - "xco2_air": CO2 in Marine boundary layer.
            - "xco2_air_alt": CO2 in Marine boundary layer, alternative CO2.
            - "iron": Iron decomposition.
            - "dust": Dust decomposition.
            - "nox": NOx decomposition.
            - "nhy": NHy decomposition.

        time : int, optional
            The time index to plot. Default is 0, which corresponds to the first
            time slice.

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
            If the specified var_name is not found in dataset.


        Examples
        --------
        >>> atm_forcing.plot("uwnd", time=0)
        """
        if var_name not in self.ds:
            raise ValueError(f"Variable '{var_name}' is not found in dataset.")

        field = self.ds[var_name].isel(time=time)

        if self.use_dask:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                field = field.load()

        if var_name in ["uwnd", "vwnd"]:
            cmap_name = "RdBu_r"
        elif var_name in ["swrad", "lwrad", "Tair", "qair"]:
            cmap_name = "YlOrRd"
        else:
            cmap_name = "YlGnBu"

        plot(
            field=field,
            grid_ds=self.grid.ds,
            use_coarse_grid=self.use_coarse_grid,
            save_path=save_path,
            cmap_name=cmap_name,
        )

    def save(
        self,
        filepath: str | Path,
        group: bool = True,
        format: NetCDFFormat = DEFAULT_NETCDF_FORMAT,
    ) -> None:
        """Save the surface forcing fields to one or more NetCDF files.

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

        Returns
        -------
        List[Path]
            A list of `Path` objects representing the filenames of the saved file(s).
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
            dataset_list,
            output_filenames,
            use_dask=self.use_dask,
            format=format,
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
        forcing_dict = to_dict(self, exclude=["use_dask", "use_coarse_grid"])
        write_to_yaml(forcing_dict, filepath)

    @classmethod
    def from_yaml(
        cls,
        filepath: str | Path,
        use_dask: bool = False,
    ) -> "SurfaceForcing":
        """Create an instance of the SurfaceForcing class from a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file from which the parameters will be read.
        use_dask: bool, optional
            Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.

        Returns
        -------
        SurfaceForcing
            An instance of the SurfaceForcing class.
        """
        filepath = Path(filepath)

        grid = Grid.from_yaml(filepath)
        params = from_yaml(cls, filepath)

        return cls(grid=grid, **params, use_dask=use_dask)
