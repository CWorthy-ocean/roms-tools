"""Source datasets defined on a curvilinear (2D) horizontal grid.

The counterpart to :mod:`roms_tools.datasets.lat_lon_datasets`, which handles
sources whose horizontal coordinates are 1D *dimension* coordinates. Here the
horizontal coordinates are 2D *variables* over a pair of index dimensions, as
produced by regional models on projected grids (WRF on a Lambert Conformal grid,
for example).

That distinction is why these cannot reuse ``LatLonDataset``: almost every one of
its horizontal operations -- ascending-order enforcement, resolution inference,
the global-wraparound check, longitude concatenation, subdomain selection -- is
written against 1D coordinate axes. Nor can they reuse
:class:`~roms_tools.datasets.roms_dataset.ROMSDataset`, whose required ``grid``
field is the *target* ROMS grid and whose initialization asserts that the data
and that grid have the same shape.

Regridding onto a ROMS grid goes through
:class:`~roms_tools.regrid.LateralRegridFromROMS`, which handles a curvilinear
source natively via xESMF. That regridder uses ``unmapped_to_nan=True`` with no
destination extrapolation, so a target point outside the source footprint comes
back as NaN rather than being silently extrapolated -- the property that lets a
limited-extent source be layered over a global one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import xarray as xr

from roms_tools.datasets.utils import (
    bbox_indices_from_2d_latlon,
    check_dataset,
    convert_to_float64,
    select_relevant_fields,
    select_relevant_times,
    specific_humidity_from_dewpoint,
    validate_start_end_time,
)
from roms_tools.fill import LateralFill, nearest_neighbor_fill
from roms_tools.processing_methods import METHOD_META, PrefillMethod
from roms_tools.utils import (
    get_dask_chunks,
    has_s3fs,
    get_pkg_error_msg,
    load_data,
    unchunk_dask,
)

DEFAULT_NR_BUFFER_POINTS = 20
"""Number of source cells to extend the subdomain beyond the target grid.

Matches the lat/lon and ROMS defaults; see
https://github.com/CWorthy-ocean/roms-tools/issues/153.
"""

SECONDS_PER_HOUR = 3600.0

DEFAULT_CONUS404_PATH = "s3://hytest/conus404/conus404_hourly.zarr"
"""Default streaming path to the CONUS404 hourly archive on the USGS/OSN pod."""

CONUS404_STORAGE_OPTIONS: dict[str, Any] = {
    "anon": True,
    "client_kwargs": {"endpoint_url": "https://usgs.osn.mghpcc.org/"},
}
"""Anonymous access options for the OSN S3-compatible endpoint.

The same store is also reachable over plain HTTPS at
``https://usgs.osn.mghpcc.org/hytest/conus404/conus404_hourly.zarr``, which needs
no ``s3fs`` but coalesces range requests less efficiently.
"""

CONUS404_RADIATION_NOISE_FLOOR_W_M2 = 4.55
"""Quantization noise floor of CONUS404 radiation, in W/m^2.

The radiation fields are stored as float32 accumulations since the 1979-10-01
model start. By 2020 they reach ~2.4e11 J/m^2, where one float32 ULP is
16384 J/m^2 -- i.e. 4.55 W/m^2 once differenced over an hour. Measured on the
real store (2020-06-15): night-time differences come out exactly 0.0, daytime
peaks reach ~1030 W/m^2, and occasional -4.55 W/m^2 artifacts appear (which
:meth:`CONUS404Dataset.post_process` clips to zero). This is intrinsic to the
product and cannot be recovered.
"""


@dataclass(kw_only=True)
class CurvilinearDataset:
    """Source data on a 2D (curvilinear) horizontal grid.

    Implements the subset of the ``LatLonDataset`` interface that the setup
    classes consume, but keyed on 2D latitude/longitude coordinate *variables*
    rather than 1D coordinate *dimensions*.

    The split between ``dim_names`` and ``coord_names`` is the point of this
    class. ``dim_names`` maps to real dataset dimensions -- what
    :func:`~roms_tools.datasets.utils.check_dataset`,
    :func:`~roms_tools.utils.get_dask_chunks` and
    :func:`~roms_tools.utils.unchunk_dask` need. ``coord_names`` maps to the 2D
    coordinate variables -- what subdomain selection and the regridder need. For
    a rectilinear source the two coincide, which is why the rest of the codebase
    conflates them.

    Parameters
    ----------
    filename : str or Path
        Path or URI of the source store.
    start_time, end_time : datetime, optional
        Time window to select. Both or neither.
    dim_names : dict[str, str]
        Maps ``"latitude"``/``"longitude"``/``"time"`` to the dataset's index
        dimension names (e.g. ``{"latitude": "y", "longitude": "x", "time": "time"}``).
        Note that the ``"latitude"``/``"longitude"`` keys name *index* dimensions
        here, not geographic axes; the names are kept for interface compatibility
        with ``LatLonDataset``, which is what lets shared helpers such as
        ``get_dask_chunks`` work unchanged.
    coord_names : dict[str, str]
        Maps ``"latitude"``/``"longitude"`` to the 2D coordinate variable names.
    var_names : dict[str, str]
        Maps required ROMS-side names to source variable names.
    opt_var_names : dict[str, str], optional
        As ``var_names``, for variables that may be absent.
    climatology : bool, optional
        Whether the source is a climatology. Defaults to False.
    has_encoded_times : bool, optional
        Whether times are CF-encoded and should be decoded. Defaults to True.
    needs_lateral_fill : bool, optional
        Whether the source contains masked cells needing a fill before regridding.
        Defaults to False: an atmospheric source is valid over land and ocean
        alike.
    use_dask : bool, optional
        Whether to load lazily with dask. Defaults to False.
    chunks : dict[str, int], optional
        Dask chunks. Ignored on the zarr path, which inherits store chunking.
    storage_options : dict, optional
        Backend options forwarded to :func:`xarray.open_zarr` for remote stores.
    read_zarr : bool, optional
        Whether to read with the zarr engine. Defaults to False.
    allow_flex_time : bool, optional
        Widen the single-record search window; see
        :func:`~roms_tools.datasets.utils.select_relevant_times`.
    start_time_pad, end_time_pad : bool, optional
        Include one record before/after the window so ROMS can interpolate at the
        exact simulation boundaries. Both default to True.
    apply_post_processing : bool, optional
        Whether to run :meth:`post_process` during initialization. Defaults to True.
    ds_loader_fn : callable, optional
        Custom loader, bypassing :func:`~roms_tools.utils.load_data`.
    initial_slice_bounds : dict, optional
        Accepted for interface compatibility with ``LatLonDataset`` and ignored;
        see :meth:`load_data`.
    """

    filename: str | Path
    start_time: datetime | None = None
    end_time: datetime | None = None
    dim_names: dict[str, str] = field(
        default_factory=lambda: {
            "latitude": "y",
            "longitude": "x",
            "time": "time",
        }
    )
    coord_names: dict[str, str] = field(
        default_factory=lambda: {"latitude": "lat", "longitude": "lon"}
    )
    var_names: dict[str, str]
    opt_var_names: dict[str, str] = field(default_factory=dict)
    climatology: bool = False
    has_encoded_times: bool = True
    needs_lateral_fill: bool = False
    use_dask: bool = False
    chunks: dict[str, int] | None = None
    storage_options: dict[str, Any] | None = None
    read_zarr: bool = False
    allow_flex_time: bool = False
    start_time_pad: bool = True
    end_time_pad: bool = True
    apply_post_processing: bool = True
    ds_loader_fn: Callable[[], xr.Dataset] | None = None
    initial_slice_bounds: dict[str, tuple[int | float, int | float]] | None = None
    _default_lateral_dask_chunk: ClassVar[int | None] = None

    is_global: bool = field(init=False, repr=False, default=False)
    resolution: float = field(init=False, repr=False)
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load, validate, time-subset and post-process the source dataset.

        Deliberately parallel to ``LatLonDataset.__post_init__``, minus the steps
        that only make sense for 1D coordinate axes (ascending-order enforcement
        and the global-wraparound check).
        """
        validate_start_end_time(self.start_time, self.end_time)
        if self.chunks is None:
            lateral = type(self)._default_lateral_dask_chunk
            if lateral is not None:
                self.chunks = get_dask_chunks(self.dim_names, lateral_chunk=lateral)

        ds = self.load_data()
        ds = self.clean_up(ds)
        self._check_curvilinear_coords(ds)
        check_dataset(ds, self.dim_names, self.var_names, self.opt_var_names)
        ds = self.select_relevant_fields(ds)

        if "time" in self.dim_names and self.start_time is not None:
            ds = self.add_time_info(ds)
            ds = self.select_relevant_times(ds)
            if self.dim_names["time"] != "time":
                ds = ds.rename({self.dim_names["time"]: "time"})

        self.resolution = self.infer_horizontal_resolution(ds)
        # A curvilinear source is regional by construction; there is no seam to
        # stitch and nothing to wrap.
        self.is_global = False
        self.ds = ds

        if self.apply_post_processing:
            self.post_process()

    # -- validation ---------------------------------------------------------

    def _check_curvilinear_coords(self, ds: xr.Dataset) -> None:
        """Verify the 2D horizontal coordinates exist with the expected dims."""
        y_dim, x_dim = self.dim_names["latitude"], self.dim_names["longitude"]
        for key in ("latitude", "longitude"):
            name = self.coord_names[key]
            if name not in ds.variables:
                raise ValueError(
                    f"{type(self).__name__} expects a 2D {key} coordinate named "
                    f"{name!r}; the dataset has none. Available coordinates: "
                    f"{sorted(ds.coords)}."
                )
            dims = ds[name].dims
            if dims != (y_dim, x_dim):
                raise ValueError(
                    f"{type(self).__name__} expects {name!r} to have dimensions "
                    f"{(y_dim, x_dim)}, but it has {dims}."
                )

    # -- loading ------------------------------------------------------------

    def load_data(self) -> xr.Dataset:
        """Load the dataset from ``self.filename``.

        ``initial_slice_bounds`` is deliberately not forwarded. It is only wired
        into the ``open_mfdataset`` preprocess hook, so it is a no-op on the zarr
        path; and :meth:`choose_subdomain` derives a tighter subset from the
        target grid anyway.
        """
        if self.initial_slice_bounds is not None:
            logging.info(
                "`initial_slice_bounds` does not apply to %s and is ignored; the "
                "subdomain is derived from the target grid instead.",
                type(self).__name__,
            )
        return load_data(
            filename=self.filename,
            dim_names=self.dim_names,
            use_dask=self.use_dask,
            decode_times=self.has_encoded_times,
            read_zarr=self.read_zarr,
            ds_loader_fn=self.ds_loader_fn,
            chunks=self.chunks,
            storage_options=self.storage_options,
        )

    def clean_up(self, ds: xr.Dataset) -> xr.Dataset:
        """Hook for subclasses to rename dimensions or fix metadata. No-op here."""
        return ds

    def select_relevant_fields(self, ds: xr.Dataset) -> xr.Dataset:
        """Drop data variables not named in ``var_names``/``opt_var_names``."""
        return select_relevant_fields(
            ds, [*self.var_names.values(), *self.opt_var_names.values()]
        )

    def add_time_info(self, ds: xr.Dataset) -> xr.Dataset:
        """Hook for subclasses to build a usable time coordinate. No-op here."""
        return ds

    def select_relevant_times(self, ds: xr.Dataset) -> xr.Dataset:
        """Restrict the dataset to the requested time window."""
        time_dim = self.dim_names["time"]
        return select_relevant_times(
            ds=ds,
            time_dim=time_dim,
            time_coord=time_dim,
            start_time=self.start_time,
            end_time=self.end_time,
            climatology=self.climatology,
            allow_flex_time=self.allow_flex_time,
            start_time_pad=self.start_time_pad,
            end_time_pad=self.end_time_pad,
        )

    def post_process(self) -> None:
        """Hook for subclasses to convert units and derive fields. No-op here."""
        pass

    # -- geometry -----------------------------------------------------------

    def infer_horizontal_resolution(self, ds: xr.Dataset) -> float:
        """Estimate the mean horizontal resolution in degrees.

        Averages the mean latitude step along the ``y`` dimension with the mean
        longitude step along ``x``, the latter scaled by ``cos(lat)`` so both are
        comparable as great-circle distances.
        """
        y_dim, x_dim = self.dim_names["latitude"], self.dim_names["longitude"]
        lat = ds[self.coord_names["latitude"]]
        lon = ds[self.coord_names["longitude"]]

        d_lat = float(np.abs(lat.diff(dim=y_dim)).mean())
        d_lon = float(np.abs(lon.diff(dim=x_dim)).mean())
        cos_lat = float(np.cos(np.deg2rad(lat.mean())))

        return 0.5 * (d_lat + d_lon * max(cos_lat, 0.1))

    def compute_minimal_grid_spacing(self, ds: xr.Dataset) -> float:
        """Return the smallest horizontal grid spacing, in metres.

        Prefers the projected index coordinates when they carry metre units (a
        projected source such as WRF gives an exact answer that way); otherwise
        falls back to converting the coordinate resolution to metres.
        """
        y_dim, x_dim = self.dim_names["latitude"], self.dim_names["longitude"]

        spacings = []
        for dim in (y_dim, x_dim):
            if dim in ds.coords and ds[dim].attrs.get("units") in ("m", "meter", "metre", "meters", "metres"):
                values = np.asarray(ds[dim].values, dtype=float)
                if values.size > 1:
                    spacings.append(float(np.abs(np.diff(values)).min()))
        if spacings:
            return min(spacings)

        # No projected axes: fall back on the geographic resolution. R_EARTH is
        # imported lazily to keep this module free of a constants import cycle.
        from roms_tools.constants import R_EARTH

        return (2 * np.pi * R_EARTH * self.infer_horizontal_resolution(ds)) / 360

    def choose_subdomain(
        self,
        target_coords: dict[str, Any],
        buffer_points: int = DEFAULT_NR_BUFFER_POINTS,
        return_copy: bool = False,
        return_coords_only: bool = False,
        verbose: bool = False,
        unchunk_lateral_dims: bool = False,
    ):
        """Restrict the dataset to the smallest index box covering the target grid.

        The signature matches ``LatLonDataset.choose_subdomain`` because the setup
        classes call it both ways -- with ``return_coords_only=True`` for the
        coarse-grid heuristic, and with ``unchunk_lateral_dims=True`` for the main
        pass.

        Unlike the lat/lon version there is nothing to concatenate: a regional
        curvilinear source is never global, so longitudes only need normalizing
        into the target's convention.

        Parameters
        ----------
        target_coords : dict
            Target grid coordinates, with ``"lat"``, ``"lon"`` and ``"straddle"``.
        buffer_points : int, optional
            Source cells to extend beyond the target extent.
        return_copy : bool, optional
            Return a new object instead of mutating this one.
        return_coords_only : bool, optional
            Return only the subset horizontal coordinates.
        verbose : bool, optional
            Accepted for interface compatibility; unused.
        unchunk_lateral_dims : bool, optional
            Reset dask chunking so the lateral dimensions are single chunks.

        Returns
        -------
        xr.Dataset or CurvilinearDataset or None
            A coordinates-only Dataset, a new object, or ``None`` when mutating
            in place.
        """
        y_dim, x_dim = self.dim_names["latitude"], self.dim_names["longitude"]
        lat_name = self.coord_names["latitude"]
        lon_name = self.coord_names["longitude"]

        lat = self.ds[lat_name]
        lon = self._wrap_longitudes(self.ds[lon_name], target_coords["straddle"])

        margin_lat = buffer_points * self.resolution
        lat_center = 0.5 * (
            float(target_coords["lat"].min()) + float(target_coords["lat"].max())
        )
        margin_lon = margin_lat / max(np.cos(np.deg2rad(lat_center)), 0.1)

        y_slice, x_slice = bbox_indices_from_2d_latlon(
            lat,
            lon,
            y_dim,
            x_dim,
            float(target_coords["lat"].min()) - margin_lat,
            float(target_coords["lat"].max()) + margin_lat,
            float(target_coords["lon"].min()) - margin_lon,
            float(target_coords["lon"].max()) + margin_lon,
            source_name=type(self).__name__,
        )

        subdomain = self.ds.isel({y_dim: y_slice, x_dim: x_slice})
        # Carry the normalized longitudes through, so the regridder and any
        # downstream consumer see the same convention as the target grid.
        subdomain = subdomain.assign_coords(
            {lon_name: lon.isel({y_dim: y_slice, x_dim: x_slice})}
        )

        if return_coords_only:
            return subdomain[[lat_name, lon_name]]

        if unchunk_lateral_dims:
            subdomain = unchunk_dask(subdomain, self.dim_names)

        if return_copy:
            return type(self).from_ds(self, subdomain)

        self.ds = subdomain
        return None

    @staticmethod
    def _wrap_longitudes(lon: xr.DataArray, straddle: bool) -> xr.DataArray:
        """Normalize longitudes to [-180, 180] when ``straddle``, else [0, 360]."""
        if straddle:
            return xr.where(lon > 180, lon - 360, lon, keep_attrs=True)
        return xr.where(lon < 0, lon + 360, lon, keep_attrs=True)

    # -- numerics -----------------------------------------------------------

    def convert_to_float64(self) -> None:
        """Promote all data variables to float64, in place."""
        self.ds = convert_to_float64(self.ds)

    def rotate_velocities_to_east_and_north(self) -> None:
        """Rotate velocities to earth-relative east/north. No-op by default.

        Subclasses whose winds are grid-relative must override this. Note that
        ``SurfaceForcing`` never calls it (unlike ``InitialConditions``), so such
        a subclass has to invoke its own rotation from :meth:`post_process`.
        """
        return None

    # -- fills --------------------------------------------------------------

    def apply_prefill(
        self,
        method: str,
        prefill_kwargs: dict | None = None,
        prefill_was_user_set: bool = False,
    ) -> None:
        """Fill masked cells by the named method.

        Mirrors ``LatLonDataset.apply_prefill``, except that the xESMF
        source-on-source fill is unavailable: it builds a source-to-source
        regridder from 1D coordinate axes.
        """
        prefill_kwargs = prefill_kwargs or {}

        if not self.needs_lateral_fill:
            if prefill_was_user_set:
                logging.info(
                    "Source data is already NaN-free (needs_lateral_fill=False); "
                    "prefill=%r is a no-op.",
                    method,
                )
            return

        match method:
            case PrefillMethod.lateral_fill_2d:
                self.apply_lateral_fill()
            case PrefillMethod.nearest_neighbor:
                self.apply_nearest_neighbor_fill()
            case _ if (
                spec := METHOD_META.get(method)
            ) is not None and spec.requires_xesmf:
                raise NotImplementedError(
                    f"prefill={method!r} performs an xESMF source-on-source regrid, "
                    f"which requires 1D source coordinate axes and so is not "
                    f"available for the curvilinear source {type(self).__name__}. "
                    f"Use prefill='2d_lateral_fill', 'nearest_neighbor', or None."
                )
            case _:
                raise ValueError(f"Unknown prefill method: {method!r}")

    def _iter_fillable_vars(self):
        """Yield ``(var_name, mask_name)`` for each non-mask data variable."""
        for var_name in self.ds.data_vars:
            if var_name.startswith("mask"):
                continue
            yield var_name, "mask"

    def _fill_masked_vars(self, fill_one) -> None:
        """Apply ``fill_one(da, mask_name)`` to every fillable variable."""
        if not self.needs_lateral_fill:
            return
        for var_name, mask_name in self._iter_fillable_vars():
            self.ds[var_name] = fill_one(self.ds[var_name], mask_name)

    def apply_lateral_fill(self) -> None:
        """Fill masked cells with the iterative AMG Poisson fill."""
        dims = (self.dim_names["latitude"], self.dim_names["longitude"])
        fillers: dict[str, LateralFill] = {}

        def fill_one(da, mask_name):
            if mask_name not in fillers:
                fillers[mask_name] = LateralFill(self.ds[mask_name], dims)
            return fillers[mask_name].apply(da)

        self._fill_masked_vars(fill_one)

    def apply_nearest_neighbor_fill(self) -> None:
        """Fill masked cells from the nearest valid cell."""
        dims = (self.dim_names["latitude"], self.dim_names["longitude"])
        self._fill_masked_vars(
            lambda da, mask_name: nearest_neighbor_fill(da, self.ds[mask_name], dims)
        )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_ds(
        cls, original_dataset: CurvilinearDataset, ds: xr.Dataset
    ) -> CurvilinearDataset:
        """Build a new instance around ``ds``, copying attributes from the original.

        Bypasses ``__init__``/``__post_init__``, so ``ds`` is taken as already
        processed.
        """
        dataset = cls.__new__(cls)
        for attr, value in original_dataset.__dict__.items():
            if attr != "ds":
                setattr(dataset, attr, value)
        dataset.ds = ds
        return dataset


@dataclass(kw_only=True)
class CONUS404Dataset(CurvilinearDataset):
    """CONUS404 hourly WRF reanalysis as a ROMS physics surface-forcing source.

    A 4 km Lambert Conformal Conic reanalysis over the conterminous United States
    and surrounding waters, hourly from 1979-10-01 to 2024-10-01, distributed as a
    zarr store on the USGS Open Storage Network pod (see
    :data:`DEFAULT_CONUS404_PATH`).

    **Footprint.** The lat/lon bounding box (17.65-57.34 N, 138.73-57.07 W)
    substantially overstates coverage, because the domain is a rectangle in
    Lambert Conformal space, not in lat/lon. Its actual corners are::

        SW  17.65 N, 122.57 W        SE  17.65 N,  73.23 W
        NW  51.69 N, 138.73 W        NE  51.69 N,  57.07 W

    so the 57.34 N maximum occurs only near the top *centre* of the domain. Along
    the US west coast the northern limit falls off quickly: ~55.1 N at 125 W,
    ~54.1 N at 130 W, ~52.9 N at 135 W, and nothing at all west of ~139 W. A Gulf
    of Alaska domain is therefore almost entirely outside CONUS404, while a
    California Current domain is fully inside. A ROMS grid extending past the
    footprint yields NaN there, by design -- see the notes below.

    :meth:`post_process` emits exactly the variables and units that
    ``ERA5Dataset`` does (``uwnd``, ``vwnd``, ``swrad``, ``lwrad``, ``Tair``,
    ``qair``, ``rain``), so the two products are interchangeable and can be
    layered over one another without a unit or convention step at the seam.

    Parameters
    ----------
    qair_method : {"psfc", "era5_magnus"}, optional
        How to derive specific humidity, both via
        :func:`~roms_tools.datasets.utils.specific_humidity_from_dewpoint` from
        ``T2``/``TD2``:

        - ``"psfc"`` (default) uses CONUS404's own ``PSFC`` field.
        - ``"era5_magnus"`` uses the fixed 1010 hPa that ERA5 processing assumes,
          making the result bit-identical to ``ERA5Dataset`` for the same inputs.

        Measured difference between the two, on 2020-06-15 CONUS404 data: ~1.1%
        (mean, max 1.3%) over open ocean off central California, where ``PSFC``
        runs 1018-1023 hPa under the North Pacific High; up to ~7% over high
        terrain. So 1010 hPa is not a neutral choice, and ``"psfc"`` is the
        physically correct one -- but it does leave a ~1% step in ``qair`` at a
        seam against an ERA5-derived field, since ERA5 processing uses the fixed
        value. Choose ``"era5_magnus"`` to remove that step instead.

        CONUS404's own ``Q2`` is deliberately *not* used: it is a mixing ratio
        rather than specific humidity, and deriving humidity by a different route
        than the blend partner would put a step at the seam even though each
        route is individually defensible.

    Notes
    -----
    **Radiation.** The store carries no instantaneous fluxes, only
    ``ACSWDNB``/``ACSWUPB``/``ACLWDNB`` accumulated in J/m^2 since the model
    start. :meth:`post_process` differences them, which consumes one leading
    record -- hence the :meth:`select_relevant_times` override. See
    :data:`CONUS404_RADIATION_NOISE_FLOOR_W_M2` for the resulting
    ~+/-4.6 W/m^2 quantization noise, which is intrinsic to the product.

    Differencing a since-model-start accumulation with ``label="upper"`` yields
    the mean flux over ``(t-1h, t]``, labelled at ``t`` -- the same
    backward-hourly-mean convention as ERA5's hourly ``ssr``/``strd``. So the
    same 30-minute ``rad_time`` shift applies; see :attr:`rad_time_offset`.

    ``rain`` is also a backward hourly mean, yet ROMS reads it on ``time`` rather
    than ``rad_time``. That is inherited from the existing ERA5 handling and kept
    deliberately: consistency with the blend partner beats unilaterally changing
    one source.

    **Winds.** ``U10``/``V10`` are relative to the Lambert Conformal *model*
    grid, not to east/north. :meth:`post_process` rotates them with
    ``COSALPHA``/``SINALPHA``; ``SurfaceForcing`` then rotates the result onto the
    ROMS grid angle.

    **Regridding.** Being curvilinear, this source must be regridded with xESMF
    (via :class:`~roms_tools.regrid.LateralRegridFromROMS`); the scipy engine
    interpolates on 1D coordinate axes and cannot handle it. Because that
    regridder does not extrapolate, target points outside the CONUS404 footprint
    come back as NaN. Pass ``bypass_validation=True``, use a fully contained
    grid, or layer this source over a global one.
    """

    # The on-disk zarr chunk is (time=144, y=175, x=175). Only relevant on a
    # non-zarr path; `read_zarr` inherits the store's own chunking.
    _default_lateral_dask_chunk: ClassVar[int] = 175

    rad_time_offset: ClassVar[timedelta] = timedelta(minutes=-30)
    """Offset from the radiation time stamp to the middle of its averaging period.

    Negative because the differenced accumulation is a mean over the *preceding*
    hour, matching ERA5.
    """

    var_names: dict[str, str] = field(
        default_factory=lambda: {
            "uwnd": "U10",  # grid-relative; rotated in post_process
            "vwnd": "V10",
            "swrad": "ACSWDNB",  # J/m^2 since model start; becomes NET shortwave
            "swup": "ACSWUPB",  # subtracted from the above, then dropped
            "lwrad": "ACLWDNB",  # J/m^2 since model start; DOWNWARD longwave
            "Tair": "T2",
            "d2m": "TD2",  # -> qair, then dropped
            "psfc": "PSFC",  # -> qair, then dropped
            "rain": "PREC_ACC_NC",
            "cosalpha": "COSALPHA",  # wind rotation, then dropped
            "sinalpha": "SINALPHA",
            "mask": "LANDMASK",  # -> mask / mask_land, then dropped
        }
    )
    dim_names: dict[str, str] = field(
        default_factory=lambda: {
            "latitude": "y",
            "longitude": "x",
            "time": "time",
        }
    )
    coord_names: dict[str, str] = field(
        default_factory=lambda: {"latitude": "lat", "longitude": "lon"}
    )
    climatology: bool = False
    needs_lateral_fill: bool = False
    qair_method: Literal["psfc", "era5_magnus"] = "psfc"

    def __post_init__(self) -> None:
        """Resolve the store backend, then run the base initialization."""
        filename = str(self.filename)

        if filename.startswith("s3://"):
            self.read_zarr = True
            if not has_s3fs():
                raise RuntimeError(
                    get_pkg_error_msg("cloud-based CONUS404 data", "s3fs", "stream")
                )
            if self.storage_options is None:
                self.storage_options = dict(CONUS404_STORAGE_OPTIONS)
        elif filename.startswith(("http://", "https://")) or filename.endswith(".zarr"):
            self.read_zarr = True

        if self.qair_method == "era5_magnus":
            # PSFC is unused in this mode; don't pay to read it.
            self.var_names = {
                k: v for k, v in self.var_names.items() if k != "psfc"
            }

        if self.start_time is None:
            logging.warning(
                "CONUS404 radiation is a since-model-start accumulation that must be "
                "differenced along time. With no `start_time`, the whole %s-record "
                "archive is differenced, which builds a very large task graph. "
                "Consider setting `start_time` and `end_time`.",
                "45-year hourly",
            )

        super().__post_init__()

    def clean_up(self, ds: xr.Dataset) -> xr.Dataset:
        """Drop the staggered (u/v-point) coordinates.

        CONUS404 carries ``lat_u``/``lon_u``/``lat_v``/``lon_v`` alongside the
        mass-point coordinates. Nothing here uses them, and leaving them attached
        keeps the ``x_stag``/``y_stag`` dimensions in the dataset, where they add
        noise to every subsequent ``sizes``/``chunk`` operation.
        """
        staggered = [
            name
            for name in ("lat_u", "lon_u", "lat_v", "lon_v")
            if name in ds.variables
        ]
        if staggered:
            ds = ds.drop_vars(staggered)
        return ds

    def select_relevant_times(self, ds: xr.Dataset) -> xr.Dataset:
        """Select the requested window plus one extra leading hourly record.

        :meth:`post_process` differences the radiation accumulations, which
        consumes the first record. Widening ``start_time`` by exactly one hour
        here makes the *differenced* series begin where the unwidened selection
        would have, under either padding setting:

        - ``start_time_pad=True``: the lower bound becomes the last record
          strictly before ``start_time - 1h``, i.e. ``start_time - 2h``. Two
          leading records -- one eaten by the diff, one left as the pad.
        - ``start_time_pad=False``: the lower bound is exactly
          ``start_time - 1h``. One leading record, eaten by the diff, so the
          series starts at ``start_time``.

        Independent of the ``end_time + 1h`` bump that ``SurfaceForcing`` applies
        for the radiation time shift.
        """
        time_dim = self.dim_names["time"]
        return select_relevant_times(
            ds=ds,
            time_dim=time_dim,
            time_coord=time_dim,
            start_time=self.start_time - timedelta(hours=1),
            end_time=self.end_time,
            climatology=self.climatology,
            allow_flex_time=self.allow_flex_time,
            start_time_pad=self.start_time_pad,
            end_time_pad=self.end_time_pad,
        )

    def rotate_velocities_to_east_and_north(self) -> None:
        """Rotate ``U10``/``V10`` from the model grid to earth-relative east/north.

        WRF stores 10 m winds relative to the projected model grid. ``COSALPHA``
        and ``SINALPHA`` are the local cosine and sine of the map rotation, so:

        ``u_east  = u_grid * COSALPHA - v_grid * SINALPHA``
        ``v_north = v_grid * COSALPHA + u_grid * SINALPHA``

        Equivalent to ``rotate_velocities(u, v, -arctan2(SINALPHA, COSALPHA))``,
        but applied directly to avoid a needless ``arctan2``/``cos``/``sin`` round
        trip and to match the form WRF documents.

        ``SurfaceForcing`` does not call this (unlike ``InitialConditions``), so
        :meth:`post_process` invokes it.
        """
        vn = self.var_names
        if vn.get("cosalpha") not in self.ds or vn.get("sinalpha") not in self.ds:
            return None

        cos_a = self.ds[vn["cosalpha"]]
        sin_a = self.ds[vn["sinalpha"]]
        u_grid = self.ds[vn["uwnd"]]
        v_grid = self.ds[vn["vwnd"]]

        self.ds = self.ds.assign(
            {
                vn["uwnd"]: u_grid * cos_a - v_grid * sin_a,
                vn["vwnd"]: v_grid * cos_a + u_grid * sin_a,
            }
        )
        return None

    def post_process(self) -> None:
        """Convert raw CONUS404 fields to the ROMS surface-forcing conventions.

        Matches ``ERA5Dataset.post_process`` in units and conventions:

        - ``swrad``: NET shortwave, W/m^2 (as ERA5's ``ssr``)
        - ``lwrad``: DOWNWARD longwave, W/m^2 (as ERA5's ``strd``)
        - ``Tair``: degrees Celsius
        - ``qair``: specific humidity, kg/kg
        - ``rain``: cm/day
        - ``uwnd``/``vwnd``: earth-relative m/s at 10 m

        Everything stays lazy under dask.
        """
        ds = self.ds
        vn = self.var_names
        time_dim = "time"

        self._check_radiation_lead_record(ds, time_dim)

        # --- Radiation: difference the since-model-start accumulations --------
        # `.diff` labels the result at the interval end (label="upper" is the
        # default), giving the mean flux over (t-1h, t] labelled at t.
        sw_down = ds[vn["swrad"]].diff(time_dim) / SECONDS_PER_HOUR
        sw_up = ds[vn["swup"]].diff(time_dim) / SECONDS_PER_HOUR
        lw_down = ds[vn["lwrad"]].diff(time_dim) / SECONDS_PER_HOUR

        # The diff dropped the leading record; realign everything else onto the
        # differenced time axis before assigning back.
        ds = ds.isel({time_dim: slice(1, None)})

        updates = {
            # Net shortwave, to match ERA5's `ssr`. Downward-only would be ~6%
            # high over the ocean -- both a seam step and a systematic heat-flux
            # bias across the whole CONUS404 region.
            vn["swrad"]: (sw_down - sw_up).clip(min=0.0),
            # Downward longwave, to match ERA5's `strd`. ROMS' bulk formulation
            # derives the upward component from its own SST, so ACLWUPB is not
            # needed and is never read.
            vn["lwrad"]: lw_down.clip(min=0.0),
        }

        # --- Scalar unit conversions -----------------------------------------
        updates[vn["Tair"]] = ds[vn["Tair"]] - 273.15
        updates[vn["d2m"]] = ds[vn["d2m"]] - 273.15
        # PREC_ACC_NC is mm accumulated over the prior 60 min, i.e. mm/hr.
        # cm/day = mm/hr * (1 cm / 10 mm) * (24 hr / day) = mm/hr * 2.4.
        # (ERA5's `tp` is m/hr, hence its *2400 to the same target unit.)
        updates[vn["rain"]] = ds[vn["rain"]] * 2.4

        ds = ds.assign(updates)
        ds[vn["swrad"]].attrs["units"] = "W/m^2"
        ds[vn["lwrad"]].attrs["units"] = "W/m^2"
        ds[vn["Tair"]].attrs["units"] = "degrees C"
        ds[vn["d2m"]].attrs["units"] = "degrees C"
        ds[vn["rain"]].attrs["units"] = "cm/day"
        self.ds = ds

        # --- Winds: grid-relative -> earth-relative --------------------------
        self.rotate_velocities_to_east_and_north()
        ds = self.ds

        # --- Humidity --------------------------------------------------------
        if self.qair_method == "psfc":
            patm = ds[vn["psfc"]] / 100.0  # Pa -> hPa
        else:
            patm = 1010.0
        qair = specific_humidity_from_dewpoint(
            ds[vn["Tair"]], ds[vn["d2m"]], patm=patm
        )
        ds = ds.assign({"qair": qair})
        ds["qair"].attrs["long_name"] = "Absolute humidity at 2m"
        ds["qair"].attrs["units"] = "kg/kg"

        # --- Masks -----------------------------------------------------------
        # A roms-tools source "mask" means "the source has valid data here": it
        # becomes xESMF's `mask_in` and gates the lateral fills. CONUS404 is an
        # atmospheric product, valid everywhere inside its footprint, land
        # included, so the mask must be all ones. Using LANDMASK here would
        # renormalize the bilinear weights over water cells only and punch NaN
        # holes through every coastal ROMS point -- exactly backwards.
        land = ds[vn["mask"]]
        ds = ds.assign({"mask": xr.ones_like(land, dtype=np.int32)})
        # Keep the land/sea mask for diagnostics under a "mask"-prefixed name, so
        # the fills and `convert_to_float64` skip it and nothing regrids it.
        ds = ds.assign({"mask_land": land.astype(np.int32)})

        # --- Drop auxiliaries and collapse var_names -------------------------
        aux_keys = ("swup", "d2m", "psfc", "cosalpha", "sinalpha", "mask")
        ds = ds.drop_vars(
            [vn[k] for k in aux_keys if k in vn and vn[k] in ds.data_vars]
        )
        self.var_names = {
            k: vn[k] for k in ("uwnd", "vwnd", "swrad", "lwrad", "Tair", "rain")
        } | {"qair": "qair"}
        self.opt_var_names = {}
        self.ds = ds

    def _check_radiation_lead_record(self, ds: xr.Dataset, time_dim: str) -> None:
        """Verify a leading record exists for the radiation differencing.

        :func:`~roms_tools.datasets.utils.select_relevant_times` only *warns*
        when no record exists before the requested start, falling back to the
        first record in the store. For a differenced accumulation that would
        silently shorten the series by an hour, so it is an error here.
        """
        if self.start_time is None:
            return
        if ds.sizes.get(time_dim, 0) < 2:
            raise ValueError(
                f"CONUS404 radiation must be differenced along time, which needs at "
                f"least two records; the selection returned "
                f"{ds.sizes.get(time_dim, 0)}."
            )

        required = self.start_time - (
            timedelta(hours=1) if self.start_time_pad else timedelta(0)
        )
        first_after_diff = ds[time_dim].values[1]
        if first_after_diff > np.datetime64(required):
            raise ValueError(
                f"CONUS404 radiation is accumulated since the model start and needs "
                f"one hourly record before {required}, but the earliest available "
                f"record is {ds[time_dim].values[0]}. Move `start_time` later by at "
                f"least two hours."
            )
