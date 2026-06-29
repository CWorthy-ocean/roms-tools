import logging
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import xarray as xr

from roms_tools.datasets.download import (
    download_river_data,
    download_river_tracer_defaults,
)
from roms_tools.datasets.utils import check_dataset, select_relevant_times
from roms_tools.setup.utils import MARBL_TRACER_NAMES
from roms_tools.utils import _get_file_matches, load_data

RiverBGCTemporalInterpolation = Literal["none", "calendar_year"]

RIVR2O_FILL_VALUE = -999.0
RIVR2O_TRACER_NAMES = ("DIC", "DOC_l", "DOC_sl", "POC", "NO3", "PO4")
RIVR2O_MIN_YEAR = 1903
RIVR2O_MAX_YEAR = 2024
SECONDS_PER_YEAR = 365.25 * 24 * 3600

VALUE_OPTION_DIM = "value_option"
RECOMMENDED_VALUE_INDEX = 0
RIVER_TRACER_DEFAULTS_FILENAME = "river_tracer_defaults.nc"


class RiverBGCDataset(Protocol):
    """Protocol for river BGC datasets used by ``RiverForcing``."""

    @property
    def requires_calendar_discharge_time(self) -> bool:
        """Whether discharge climatology must be expanded to a calendar axis."""
        ...

    @property
    def temporal_interpolation(self) -> RiverBGCTemporalInterpolation:
        """How interior temporal gaps in dynamic concentrations are filled."""
        ...

    @property
    def fill_value(self) -> float | None:
        """Sentinel marking missing concentrations, or ``None`` if not used.

        ``RiverForcing`` masks this value out of dynamic concentrations before
        merging in fill defaults, so any dataset that uses a sentinel must
        surface it here.
        """
        ...

    def forcing_concentrations(
        self,
        river_volume: xr.DataArray,
        abs_time: xr.DataArray,
        lons: np.ndarray,
        lats: np.ndarray,
        *,
        straddle: bool,
        river_names: list[str],
    ) -> dict[str, xr.DataArray]:
        """Return ROMS tracer concentrations on ``(river_time, nriver)``."""
        ...


def fill_river_bgc_concentrations(
    dynamic: dict[str, xr.DataArray],
    fill: dict[str, float],
    tracer_names: Iterable[str],
    template: xr.DataArray,
    *,
    fill_at_nan: bool = True,
) -> dict[str, xr.DataArray]:
    """Merge dynamic river BGC concentrations with fill values.

    For each tracer in ``tracer_names``:

    - If the tracer is absent from ``dynamic``, broadcast the fill scalar to
      ``template`` shape.
    - If the tracer is present and ``fill_at_nan`` is True, use dynamic values
      where finite and fill elsewhere.
    - If the tracer is present and ``fill_at_nan`` is False, use dynamic values
      as-is.

    Parameters
    ----------
    dynamic
        Tracer concentrations supplied by a dynamic BGC dataset (may be partial).
    fill
        Scalar fill concentrations keyed by tracer name.
    tracer_names
        Tracer names to include in the output (typically ``ds.tracer_name``).
    template
        Array with shape ``(river_time, nriver)`` used to define output dimensions.
    fill_at_nan
        If True, replace non-finite dynamic values with fill scalars.

    Returns
    -------
    dict[str, xr.DataArray]
        Merged concentrations on ``(river_time, nriver)``.
    """
    merged: dict[str, xr.DataArray] = {}
    for tracer_name in tracer_names:
        if tracer_name not in fill:
            logging.warning(
                "Fill source has no value for tracer %s; skipping.", tracer_name
            )
            continue
        fill_arr = xr.full_like(template, fill[tracer_name], dtype=np.float32)
        if tracer_name in dynamic:
            dyn = dynamic[tracer_name]
            if fill_at_nan:
                merged[tracer_name] = dyn.where(np.isfinite(dyn), fill_arr).astype(
                    np.float32
                )
            else:
                merged[tracer_name] = dyn.astype(np.float32)
        else:
            merged[tracer_name] = fill_arr
    return merged


@dataclass(kw_only=True)
class RiverTracerDefaultsDataset(RiverBGCDataset):
    """Default MARBL river tracer concentrations.

    Used as the ``"CONSTANTS"`` ``bgc_source`` and as the default ``fill`` source
    in ``RiverForcing``. Loads recommended boundary concentrations from
    ``river_tracer_defaults.nc`` in the roms-tools-data repository. Each tracer
    variable has dimension ``value_option`` (0 = recommended, 1 = alternate).

    ``forcing_concentrations`` broadcasts scalar defaults to every
    ``(river_time, nriver)`` point.

    Parameters
    ----------
    filename : str or Path, optional
        Path to the NetCDF file. Defaults to the file from roms-tools-data.
    value_option_index : int, optional
        Index along ``value_option`` to read. Defaults to 0 (recommended values).

    Attributes
    ----------
    defaults : dict[str, float]
        Tracer name to concentration for the selected value option.
    ds : xr.Dataset
        The loaded NetCDF dataset (in memory).
    """

    filename: str | Path = field(default_factory=download_river_tracer_defaults)
    value_option_index: int = RECOMMENDED_VALUE_INDEX
    defaults: dict[str, float] = field(init=False, repr=False)
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self) -> None:
        with xr.open_dataset(self.filename) as ds:
            self.defaults = self._read_defaults(ds)
            self.ds = ds.load()

    def _read_defaults(self, ds: xr.Dataset) -> dict[str, float]:
        """Extract tracer concentrations for the selected value option."""
        expected_tracers = set(MARBL_TRACER_NAMES)

        if VALUE_OPTION_DIM not in ds.dims:
            raise ValueError(
                f"{RIVER_TRACER_DEFAULTS_FILENAME} must contain a "
                f"'{VALUE_OPTION_DIM}' dimension."
            )

        defaults: dict[str, float] = {}
        for tracer_name in ds.data_vars:
            if tracer_name == VALUE_OPTION_DIM:
                continue

            da = ds[tracer_name]
            if VALUE_OPTION_DIM not in da.dims:
                raise ValueError(
                    f"Variable '{tracer_name}' must have dimension "
                    f"'{VALUE_OPTION_DIM}'."
                )

            value = float(da.isel({VALUE_OPTION_DIM: self.value_option_index}).values)
            if np.isnan(value):
                raise ValueError(
                    f"Value for tracer '{tracer_name}' at "
                    f"{VALUE_OPTION_DIM}={self.value_option_index} is missing (NaN)."
                )
            defaults[tracer_name] = value

        missing = expected_tracers - set(defaults)
        if missing:
            raise ValueError(
                f"{RIVER_TRACER_DEFAULTS_FILENAME} is missing tracers: "
                f"{sorted(missing)}"
            )

        extra = set(defaults) - expected_tracers
        if extra:
            raise ValueError(
                f"{RIVER_TRACER_DEFAULTS_FILENAME} contains unknown tracers: "
                f"{sorted(extra)}"
            )

        return defaults

    @property
    def requires_calendar_discharge_time(self) -> bool:
        return False

    @property
    def temporal_interpolation(self) -> RiverBGCTemporalInterpolation:
        return "none"

    @property
    def fill_value(self) -> float | None:
        return None

    def forcing_concentrations(
        self,
        river_volume: xr.DataArray,
        abs_time: xr.DataArray,
        lons: np.ndarray,
        lats: np.ndarray,
        *,
        straddle: bool,
        river_names: list[str],
    ) -> dict[str, xr.DataArray]:
        """Broadcast constant default concentrations to ``river_volume`` shape."""
        del abs_time, lons, lats, straddle, river_names
        return {
            tracer_name: xr.full_like(river_volume, value, dtype=np.float32)
            for tracer_name, value in self.defaults.items()
        }


def rivr2o_boundary_time(year: int) -> np.datetime64:
    """Return the mid-year timestamp used for a RIVR2O annual file."""
    return np.datetime64(datetime(year, 7, 1), "ns")


def clamp_rivr2o_time(time: xr.DataArray) -> xr.DataArray:
    """Clamp times to the valid RIVR2O product range, holding boundary values.

    Expects absolute datetimes (e.g. ``ds['abs_time']``), not ROMS relative
    ``river_time`` in days since the model reference date.
    """
    return time.clip(
        min=rivr2o_boundary_time(RIVR2O_MIN_YEAR),
        max=rivr2o_boundary_time(RIVR2O_MAX_YEAR),
    )


def rivr2o_coerce_time(
    time: datetime | np.datetime64 | np.integer | int | float | str,
) -> np.datetime64:
    """Convert a scalar time to ``datetime64[ns]`` for RIVR2O time indexing.

    Handles ``datetime``, ``numpy.datetime64``, and integer nanosecond stamps
    (as returned by ``.item()`` on some xarray ``abs_time`` values under Dai
    climatology).
    """
    import pandas as pd

    return np.datetime64(pd.Timestamp(time).to_datetime64())


@dataclass(kw_only=True)
class RiverDataset:
    """Represents river data.

    Parameters
    ----------
    filename : Union[str, Path, List[Union[str, Path]]]
        The path to the data file(s). Can be a single string (with or without wildcards), a single Path object,
        or a list of strings or Path objects containing multiple files.
    start_time : datetime
        The start time for selecting relevant data.
    end_time : datetime
        The end time for selecting relevant data.
    dim_names: Dict[str, str]
        Dictionary specifying the names of dimensions in the dataset.
        Requires "station" and "time" as keys.
    var_names: Dict[str, str]
        Dictionary of variable names that are required in the dataset.
        Requires the keys "latitude", "longitude", "flux", "ratio", and "name".
    opt_var_names: Dict[str, str], optional
        Dictionary of variable names that are optional in the dataset.
        Defaults to an empty dictionary.
    climatology : bool
        Indicates whether the dataset is climatological. Defaults to False.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the forcing data on its original grid.
    """

    filename: str | Path | list[str | Path]
    start_time: datetime
    end_time: datetime
    dim_names: dict[str, str]
    var_names: dict[str, str]
    opt_var_names: dict[str, str] | None = field(default_factory=dict)
    climatology: bool = False
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        # Validate start_time and end_time
        if not isinstance(self.start_time, datetime):
            raise TypeError(
                f"start_time must be a datetime object, but got {type(self.start_time).__name__}."
            )
        if not isinstance(self.end_time, datetime):
            raise TypeError(
                f"end_time must be a datetime object, but got {type(self.end_time).__name__}."
            )

        ds = self.load_data()
        ds = self.clean_up(ds)
        self.check_dataset(ds)
        ds = _deduplicate_river_names(
            ds, self.var_names["name"], self.dim_names["station"]
        )

        # Select relevant times
        ds = self.add_time_info(ds)
        self.ds = ds

    def load_data(self) -> xr.Dataset:
        """Load dataset from the specified file.

        Returns
        -------
        ds : xr.Dataset
            The loaded xarray Dataset containing the forcing data.
        """
        ds = load_data(self.filename, self.dim_names, use_dask=True, decode_times=True)

        # Subset time lazily before compute to avoid loading full multi-year dataset.
        # Add 1-day buffer on each side to ensure select_relevant_times has data at boundaries.
        time_dim = self.dim_names["time"]
        ds = ds.sel(
            {
                time_dim: slice(
                    np.datetime64(self.start_time) - np.timedelta64(1, "D"),
                    np.datetime64(self.end_time) + np.timedelta64(1, "D"),
                )
            }
        )
        return ds

    def clean_up(self, ds: xr.Dataset) -> xr.Dataset:
        """Decodes the 'name' variable (if byte-encoded) and updates the dataset.

        This method checks if the 'name' variable is of dtype 'object' (i.e., byte-encoded),
        and if so, decodes each byte array to a string and updates the dataset.
        It also ensures that the 'station' dimension is of integer type.


        Parameters
        ----------
        ds : xr.Dataset
            The dataset containing the 'name' variable to decode.

        Returns
        -------
        ds : xr.Dataset
            The dataset with the decoded 'name' variable.
        """
        if ds[self.var_names["name"]].dtype == "object":
            names = []
            for i in range(len(ds[self.dim_names["station"]])):
                byte_array = ds[self.var_names["name"]].isel(
                    **{self.dim_names["station"]: i}
                )
                name = _decode_string(byte_array)
                names.append(name)
            ds[self.var_names["name"]] = xr.DataArray(
                data=names, dims=self.dim_names["station"]
            )

        if ds[self.dim_names["station"]].dtype == "float64":
            ds[self.dim_names["station"]] = ds[self.dim_names["station"]].astype(int)

        # Drop all variables that have chars dim
        vars_to_drop = ["ocn_name", "stn_name", "ct_name", "cn_name", "chars"]
        existing_vars = [var for var in vars_to_drop if var in ds]
        ds = ds.drop_vars(existing_vars)

        return ds

    def check_dataset(self, ds: xr.Dataset) -> None:
        """Validate required variables, dimensions, and uniqueness of river names.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset to check.

        Raises
        ------
        ValueError
            If the dataset does not contain the specified variables or dimensions.
        """
        check_dataset(ds, self.dim_names, self.var_names, self.opt_var_names)

    def add_time_info(self, ds: xr.Dataset) -> xr.Dataset:
        """Dummy method to be overridden by child classes to add time information to the
        dataset.

        This method is intended as a placeholder and should be implemented in subclasses
        to provide specific functionality for adding time-related information to the dataset.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset to which time information will be added.

        Returns
        -------
        xr.Dataset
            The xarray Dataset with time information added (as implemented by child classes).
        """
        return ds

    def select_relevant_times(self, ds) -> xr.Dataset:
        """Select a subset of the dataset based on the specified time range.

        This method filters the dataset to include all records between `start_time` and `end_time`.
        Additionally, it ensures that one record at or before `start_time` and one record at or
        after `end_time` are included, even if they fall outside the strict time range.

        If no `end_time` is specified, the method will select the time range of
        [start_time, start_time + 24 hours] and return the closest time entry to `start_time` within that range.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset to be filtered. Must contain a time dimension.

        Returns
        -------
        xr.Dataset
            A dataset filtered to the specified time range, including the closest entries
            at or before `start_time` and at or after `end_time` if applicable.

        Warns
        -----
        UserWarning
            If no records at or before `start_time` or no records at or after `end_time` are found.

        UserWarning
            If the dataset does not contain any time dimension or the time dimension is incorrectly named.
        """
        time_dim = self.dim_names["time"]

        ds = select_relevant_times(
            ds=ds,
            time_dim=time_dim,
            time_coord=time_dim,
            start_time=self.start_time,
            end_time=self.end_time,
        )

        return ds

    def compute_climatology(self):
        from roms_tools.setup.utils import assign_dates_to_climatology

        logging.info("Compute climatology for river forcing.")

        time_dim = self.dim_names["time"]

        flux = self.ds[self.var_names["flux"]].groupby(f"{time_dim}.month").mean()
        self.ds[self.var_names["flux"]] = flux

        ds = assign_dates_to_climatology(self.ds, "month")
        ds = ds.swap_dims({"month": "time"})
        self.ds = ds

        updated_dim_names = {**self.dim_names}
        updated_dim_names["time"] = "time"
        self.dim_names = updated_dim_names

        self.climatology = True

    def sort_by_river_volume(self, ds: xr.Dataset) -> xr.Dataset:
        """Sorts the dataset by river volume in descending order (largest rivers first),
        if the volume variable is available.

        This method uses the river volume to reorder the dataset such that the rivers with
        the largest volumes come first in the `station` dimension. If the volume variable
        is not present in the dataset, a warning is logged.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset containing the river data to be sorted by volume.

        Returns
        -------
        xr.Dataset
            The dataset with rivers sorted by their volume in descending order.
            If the volume variable is not available, the original dataset is returned.
        """
        if self.opt_var_names is not None and "vol" in self.opt_var_names:
            volume_values = ds[self.opt_var_names["vol"]].values
            if isinstance(volume_values, np.ndarray):
                if len(volume_values) == 0:
                    return ds
                # Check if all volume values are the same
                if np.all(volume_values == volume_values[0]):
                    # If all volumes are the same, no need to reverse order
                    sorted_indices = np.argsort(
                        volume_values
                    )  # Sort in ascending order
                else:
                    # If volumes differ, reverse order for descending sort
                    sorted_indices = np.argsort(volume_values)[
                        ::-1
                    ]  # Reverse for descending order

                ds = ds.isel(**{self.dim_names["station"]: sorted_indices})

            else:
                logging.warning("The volume data is not in a valid array format.")
        else:
            logging.warning(
                "Cannot sort rivers by volume. 'vol' is missing in the variable names."
            )

        return ds

    def extract_relevant_rivers(
        self, target_coords, dx, coast_snap_buffer_km=None, domain_edge_buffer=20
    ):
        """Extract rivers within the ROMS domain and assign each to the nearest coastal grid cell.

        Uses a three-step memory-efficient approach to avoid building the full
        (eta_rho x xi_rho x n_stations) distance matrix, which would require
        hundreds of GiB for large river datasets (like gloFAS):

        1. Bounding box pre-filter — coarse reduction using lat/lon comparison to grid boundary.
        2. cKDTree query_ball_point — finds stations within coast_snap_buffer_km of any coastal cell.
        3. cKDTree nearest-neighbor query — assigns each surviving station to
        its closest grid cell (any cell, not just coastal). Used for
        original_indices to show pre-coast-snap positions in plot_locations.

        Parameters
        ----------
        target_coords : dict
            A dictionary containing the target coordinates for the comparison:
            - "lon" (xarray.DataArray): Longitude coordinates of the ROMS grid.
            - "lat" (xarray.DataArray): Latitude coordinates of the ROMS grid.
            - "straddle" (bool): If True, longitudes > 180 are adjusted by
            subtracting 360; otherwise, negative longitudes are adjusted by adding 360.
            - "mask" (xarray.DataArray, optional): Ocean mask (1=ocean, 0=land).
            If provided, the cKDTree is built on coastal cells only (~5-10% of
            grid), reducing memory and computation by ~10-20x.
        dx : float
            Maximum distance threshold in meters. Only river mouths within dx
            of any ROMS grid cell are included.
        coast_snap_buffer_km : float, optional
            If provided, only river mouths within this distance (in km) of any
            coastal grid cell are included. Useful for datasets like GloFAS where
            river mouths may be slightly offshore. If None, all rivers within the
            bounding box are included.
        domain_edge_buffer : int, optional
            Number of grid cells beyond the domain edge to include in the bounding
            box pre-filter. Catches rivers just outside the domain that may still
            have relevant freshwater forcing. Default is 20.

        Returns
        -------
        indices : dict[str, list[tuple]]
            River names as keys, each mapping to a list containing one tuple of
            (eta_rho, xi_rho) grid indices of the nearest coastal cell.
        """
        from roms_tools.setup.utils import (
            build_kdtree_from_latlon,
            query_kdtree_nearest,
        )

        # Retrieve longitude and latitude of river mouths
        river_lon = self.ds[self.var_names["longitude"]]
        river_lat = self.ds[self.var_names["latitude"]]

        # Adjust longitude for date line straddling
        if target_coords["straddle"]:
            river_lon = xr.where(river_lon > 180, river_lon - 360, river_lon)
        else:
            river_lon = xr.where(river_lon < 0, river_lon + 360, river_lon)

        station_dim = self.dim_names["station"]

        # Build candidate point set — coastal cells only if mask provided, else all grid cells
        target_lon_np = target_coords["lon"].values
        target_lat_np = target_coords["lat"].values
        eta_rho, xi_rho = target_lon_np.shape

        if target_coords.get("mask") is not None:
            mask_rho = target_coords["mask"].values
            faces = np.zeros_like(mask_rho)
            faces[1:, :] += mask_rho[:-1, :]
            faces[:-1, :] += mask_rho[1:, :]
            faces[:, 1:] += mask_rho[:, :-1]
            faces[:, :-1] += mask_rho[:, 1:]
            coast = (1 - mask_rho) * (faces > 0)
            coast_eta, coast_xi = np.where(coast)
            tree_lat = target_lat_np[coast_eta, coast_xi]
            tree_lon = target_lon_np[coast_eta, coast_xi]
        else:
            coast_eta = np.arange(eta_rho * xi_rho) // xi_rho
            coast_xi = np.arange(eta_rho * xi_rho) % xi_rho
            tree_lat = target_lat_np.ravel()
            tree_lon = target_lon_np.ravel()

        # Step 1: bounding box pre-filter with buffer to catch rivers just outside the domain
        buffer_deg = float(dx) / 111000 * (domain_edge_buffer + 1)
        lat_min = target_lat_np.min() - buffer_deg
        lat_max = target_lat_np.max() + buffer_deg
        lon_min = target_lon_np.min() - buffer_deg
        lon_max = target_lon_np.max() + buffer_deg

        rlat = river_lat.values
        rlon = river_lon.values

        in_bbox = (
            (rlat >= lat_min)
            & (rlat <= lat_max)
            & (rlon >= lon_min)
            & (rlon <= lon_max)
        )
        bbox_indices = np.where(in_bbox)[0]

        if len(bbox_indices) == 0:
            self.ds = xr.Dataset()
            return {}

        # Step 2: optional distance filter using query_ball_point
        if coast_snap_buffer_km is not None:
            from roms_tools.setup.utils import latlon_to_xyz

            dx_chord = 2 * np.sin((coast_snap_buffer_km / 6371.0) / 2)
            tree = build_kdtree_from_latlon(tree_lat, tree_lon)
            counts = tree.query_ball_point(
                latlon_to_xyz(rlat[bbox_indices], rlon[bbox_indices]),
                r=dx_chord,
                return_length=True,
            )
            final_indices = bbox_indices[counts > 0]
        else:
            tree = build_kdtree_from_latlon(tree_lat, tree_lon)
            final_indices = bbox_indices

        # Subset dataset and sort first
        ds = self.ds.isel({station_dim: final_indices})
        ds = self.sort_by_river_volume(ds)
        self.ds = ds

        # Step 3: assign each station to nearest any-grid-cell using sorted positions
        sorted_lat = ds[self.var_names["latitude"]].values
        sorted_lon = ds[self.var_names["longitude"]].values
        if target_coords["straddle"]:
            sorted_lon = np.where(sorted_lon > 180, sorted_lon - 360, sorted_lon)
        else:
            sorted_lon = np.where(sorted_lon < 0, sorted_lon + 360, sorted_lon)

        all_eta = np.arange(eta_rho * xi_rho) // xi_rho
        all_xi = np.arange(eta_rho * xi_rho) % xi_rho
        all_tree = build_kdtree_from_latlon(
            target_lat_np.ravel(), target_lon_np.ravel()
        )
        eta_argmin, xi_argmin, _ = query_kdtree_nearest(
            all_tree,
            sorted_lat,
            sorted_lon,
            all_eta,
            all_xi,
        )

        names_final = ds[self.var_names["name"]].values
        river_indices = {
            str(names_final[i]): [(int(eta_argmin[i]), int(xi_argmin[i]))]
            for i in range(len(ds[self.dim_names["station"]]))
        }

        return river_indices

    def extract_named_rivers(self, indices):
        """Extracts a subset of the dataset based on the provided river names in the
        indices dictionary.

        This method filters the dataset to include only the rivers specified in the `indices` dictionary.
        The resulting subset is stored in the `ds` attribute of the class.

        Parameters
        ----------
        indices : dict
            A dictionary where the keys are river names (strings) and the values are dictionaries
            containing river-related data (e.g., river indices, coordinates).

        Returns
        -------
        None
            The method modifies the `self.ds` attribute in place, setting it to the filtered dataset
            containing only the data related to the specified rivers.

        Raises
        ------
        ValueError
            - If `indices` is not a dictionary.
            - If any of the requested river names are not found in the dataset.
        """
        if not isinstance(indices, dict):
            raise ValueError("`indices` must be a dictionary.")

        river_names = list(indices.keys())

        # Ensure the dataset is filtered based on the provided river names.
        # Use isel with a boolean mask rather than where(drop=True): the latter
        # routes through apply_ufunc and unexpectedly collapses unrelated
        # dimensions (e.g. time) in some xarray versions.
        mask = self.ds[self.var_names["name"]].isin(river_names).values
        ds_filtered = self.ds.isel({self.dim_names["station"]: mask})

        # Check that all requested rivers exist in the dataset
        filtered_river_names = set(ds_filtered[self.var_names["name"]].values)
        missing_rivers = set(river_names) - filtered_river_names

        if missing_rivers:
            raise ValueError(
                f"The following rivers were not found in the dataset: {missing_rivers}"
            )

        # Set the filtered dataset as the new `ds`
        self.ds = ds_filtered


@dataclass(kw_only=True)
class GloFASRiverDataset(RiverDataset):
    """River discharge dataset from GloFAS v4.0.
    Expects a NetCDF file preprocessed using the GloFAS Large-scale Drainage
    Direction (LDD) algorithm, which places river mouths on coastal cells.
    Time is encoded as CF-compliant datetime64 and decoded directly.

    Variable name mappings:
        - latitude:  ``lat_mou``
        - longitude: ``lon_mou``
        - flux:      ``FLOW`` (m³/s)
        - ratio:     ``ratio_m2s``
        - name:      ``riv_name``
    """

    dim_names: dict = field(
        default_factory=lambda: {"station": "station", "time": "time"}
    )
    var_names: dict = field(
        default_factory=lambda: {
            "latitude": "lat_mou",
            "longitude": "lon_mou",
            "flux": "FLOW",
            "ratio": "ratio_m2s",
            "name": "riv_name",
        }
    )
    opt_var_names: dict = field(default_factory=lambda: {"vol": "vol_stn"})

    def extract_relevant_rivers(
        self, target_coords, dx, coast_snap_buffer_km=50.0, domain_edge_buffer=20
    ):
        """Extract relevant rivers, defaulting to a 50 km coastal snap buffer.

        GloFAS river mouths are preprocessed using the LDD algorithm to lie on
        or very near the coast, so a tight 50 km buffer is appropriate.
        See :meth:`RiverDataset.extract_relevant_rivers` for full documentation.
        """
        return super().extract_relevant_rivers(
            target_coords,
            dx,
            coast_snap_buffer_km=coast_snap_buffer_km,
            domain_edge_buffer=domain_edge_buffer,
        )

    def add_time_info(self, ds: xr.Dataset) -> xr.Dataset:
        """Time is CF-compliant datetime64 — decode directly."""
        import pandas as pd

        time_dim = self.dim_names["time"]
        ds[time_dim] = pd.DatetimeIndex(ds[time_dim].values)
        return ds


@dataclass(kw_only=True)
class DaiRiverDataset(RiverDataset):
    """River discharge dataset from Dai & Trenberth (2009).

    Provides monthly climatological or time-varying discharge for ~1000 of
    the world's largest rivers. Time is encoded as numeric YYYYMM values and
    decoded manually in ``add_time_info``. River mouths may be placed inland,
    so a generous coastal snap buffer (200 km default) is used.
    """

    filename: str | Path | list[str | Path] = field(
        default_factory=lambda: download_river_data("dai_trenberth_may2019.nc")
    )
    dim_names: dict[str, str] = field(
        default_factory=lambda: {
            "station": "station",
            "time": "time",
        }
    )
    var_names: dict[str, str] = field(
        default_factory=lambda: {
            "latitude": "lat_mou",
            "longitude": "lon_mou",
            "flux": "FLOW",
            "ratio": "ratio_m2s",
            "name": "riv_name",
        }
    )
    opt_var_names: dict[str, str] = field(
        default_factory=lambda: {
            "vol": "vol_stn",
        }
    )
    climatology: bool = False

    def load_data(self) -> xr.Dataset:
        """Load dataset from the specified file. Load Dai dataset from the specified file.
        Overrides the base class to use use_dask=False, decode_times=False because Dai encodes
        time as numeric YYYYMM values which are decoded manually in add_time_info().
        Time subsetting is handled downstream by select_relevant_times() after
        add_time_info() converts the numeric coordinates to datetimes.
        """
        return load_data(
            self.filename, self.dim_names, use_dask=False, decode_times=False
        )

    def add_time_info(self, ds: xr.Dataset) -> xr.Dataset:
        """Adds time information to the dataset based on the climatology flag and
        dimension names.

        This method processes the dataset to include time information according to the climatology
        setting. If the dataset represents climatology data and the time dimension is labeled as
        "month", it assigns dates to the dataset based on a monthly climatology. Additionally, it
        handles dimension name updates if necessary.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset to which time information will be added.

        Returns
        -------
        xr.Dataset
            The dataset with time information added, including adjustments for climatology and
            dimension names.
        """
        time_dim = self.dim_names["time"]

        # Extract the 'time' variable as a numpy array
        time_vals = ds[time_dim].values

        # Handle rounding of the time values
        year = np.round(time_vals * 1e-2).astype(int)
        month = np.round((time_vals * 1e-2 - year) * 1e2).astype(int)

        # Convert to datetime (assuming the day is always 15th for this example)
        dates = [datetime(year=i, month=m, day=15) for i, m in zip(year, month)]

        ds[time_dim] = dates

        return ds

    def extract_relevant_rivers(
        self, target_coords, dx, coast_snap_buffer_km=200.0, domain_edge_buffer=20
    ):
        """Extract relevant rivers, defaulting to a 200 km coastal snap buffer.

        Dai river mouths may be placed far inland relative to the actual river
        mouth, so a generous buffer is used to avoid excluding legitimate rivers.
        """
        return super().extract_relevant_rivers(
            target_coords,
            dx,
            coast_snap_buffer_km=coast_snap_buffer_km,
            domain_edge_buffer=domain_edge_buffer,
        )


def _parse_rivr2o_year(filename: str | Path) -> int:
    """Extract the calendar year from a RIVR2O river inputs filename."""
    stem = Path(filename).stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(
            f"Could not parse year from RIVR2O filename '{filename}'. "
            "Expected pattern 'rivr2o_riverinputs_YYYY.nc'."
        ) from exc


_RIVR2O_MOLAR_MASS_G = {
    "DIC": 12.011,
    "DOC_l": 12.011,
    "DOC_sl": 12.011,
    "POC": 12.011,
    "NO3": 14.007,
    "PO4": 30.974,
}
_DON_FROM_DOC_SL = 103 / 2583
_DON_FROM_POC = 25 / 276
_DOP_FROM_DOC_SL = 1 / 2583
_DOP_FROM_POC = 1 / 276


@dataclass(kw_only=True)
class Rivr2oRiverBGCDataset(RiverBGCDataset):
    """River BGC export data from the RIVR2O river inputs product.

    The product is distributed as one NetCDF file per year. Each file contains
    global river export fields on a regular lat/lon grid (typically 0.5°). Raw
    variables are annual mass exports in ``10^6 g element yr-1``.

    On load, raw fields are converted to export variables used for river forcing:

    - ``DIC``, ``DOC_l``, ``DOC_sl``, and ``POC`` are kept as separate carbon exports
    - ``DIN`` is renamed to ``NO3``
    - ``DIP`` is renamed to ``PO4``

    The resulting dataset exposes ``DIC``, ``DOC_l``, ``DOC_sl``, ``POC``, ``NO3``,
    and ``PO4`` on dimensions ``(time, lat, lon)``. Times are assigned from the year
    in each filename (mid-year, 1 July), because the native ``time`` variable
    is often unset in the source files.

    The product spans 1903-2024. Requests before 1903 or after 2024 use the boundary
    years when selecting data and when mapping onto river forcing times.

    Spatial sampling (``sample_at_points``) uses the nearest grid cell with positive
    ``DIC`` export. DIC has the same spatial coverage each year; that cell is used
    for all tracers and all years. When several rivers share a cell,
    ``discharge_partition_weights`` splits export in proportion to discharge so
    co-located rivers receive similar concentrations. Use ``forcing_concentrations``
    to convert exports to MARBL river tracer concentrations for ROMS forcing.

    Parameters
    ----------
    filename : str, Path, or list[str | Path]
        Path to one file, a wildcard pattern (e.g.
        ``"/data/rivr2o_riverinputs_*.nc"``), or a list of file paths.
    start_time : datetime
        Start of the time range to retain.
    end_time : datetime
        End of the time range to retain.
    use_dask : bool, optional
        If True, open files with dask chunking along time. Defaults to False.

    Attributes
    ----------
    ds : xr.Dataset
        Processed dataset with MARBL tracer variables on the native lat/lon grid.
    """

    filename: str | Path | list[str | Path]
    start_time: datetime
    end_time: datetime
    dim_names: dict[str, str] = field(
        default_factory=lambda: {
            "latitude": "lat",
            "longitude": "lon",
            "time": "time",
        }
    )
    var_names: dict[str, str] = field(
        default_factory=lambda: {
            "DIC": "DIC",
            "DIN": "DIN",
            "DOC_l": "DOC_l",
            "DOC_sl": "DOC_sl",
            "POC": "POC",
            "DIP": "DIP",
        }
    )
    tracer_names: tuple[str, ...] = RIVR2O_TRACER_NAMES
    fill_value: float = RIVR2O_FILL_VALUE
    use_dask: bool = False
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.start_time, datetime):
            raise TypeError(
                f"start_time must be a datetime object, but got {type(self.start_time).__name__}."
            )
        if not isinstance(self.end_time, datetime):
            raise TypeError(
                f"end_time must be a datetime object, but got {type(self.end_time).__name__}."
            )

        ds = self.load_data()
        ds = self.clean_up(ds)
        self.check_dataset(ds)
        ds = self.select_relevant_times(ds)
        self.ds = ds

    def load_data(self) -> xr.Dataset:
        """Load and concatenate yearly RIVR2O files along time.

        Returns
        -------
        xr.Dataset
            Concatenated dataset with one time step per input file.
        """
        match_result = _get_file_matches(self.filename)
        if not match_result.matches:
            raise FileNotFoundError(f"No RIVR2O files matched: {self.filename}")

        ds_list = []
        for file in match_result.matches:
            chunks = {self.dim_names["time"]: 1} if self.use_dask else None
            ds = xr.open_dataset(file, decode_times=False, chunks=chunks)
            year = _parse_rivr2o_year(file)
            ds = ds.assign_coords(
                {self.dim_names["time"]: [np.datetime64(datetime(year, 7, 1), "ns")]}
            )
            ds_list.append(ds)

        return xr.concat(
            ds_list,
            dim=self.dim_names["time"],
            coords="minimal",
            compat="override",
            combine_attrs="override",
        )

    def clean_up(self, ds: xr.Dataset) -> xr.Dataset:
        """Replace fill values and rename raw RIVR2O variables."""
        lat_dim = self.dim_names["latitude"]
        lon_dim = self.dim_names["longitude"]

        for var_name in self.var_names.values():
            if var_name in ds:
                ds[var_name] = ds[var_name].where(ds[var_name] != self.fill_value)

        ds["NO3"] = ds[self.var_names["DIN"]].rename("NO3")
        ds["NO3"].attrs = ds[self.var_names["DIN"]].attrs

        ds["PO4"] = ds[self.var_names["DIP"]].rename("PO4")
        ds["PO4"].attrs = ds[self.var_names["DIP"]].attrs

        ds = ds.drop_vars(
            [self.var_names["DIN"], self.var_names["DIP"]],
            errors="ignore",
        )

        ds = ds.drop_vars(
            [var for var in ds.data_vars if var not in self.tracer_names],
            errors="ignore",
        )

        if lat_dim in ds.coords and ds[lat_dim].ndim == 1:
            ds = ds.sortby(lat_dim)
        if lon_dim in ds.coords and ds[lon_dim].ndim == 1:
            ds = ds.sortby(lon_dim)

        return ds

    def check_dataset(self, ds: xr.Dataset) -> None:
        """Validate dimensions and MARBL tracer variables."""
        for dim_key in ("latitude", "longitude", "time"):
            dim_name = self.dim_names[dim_key]
            if dim_name not in ds.dims:
                raise ValueError(
                    f"Dataset is missing required dimension '{dim_name}' ({dim_key})."
                )

        missing_tracers = [name for name in self.tracer_names if name not in ds]
        if missing_tracers:
            raise ValueError(
                f"Dataset is missing required tracer variables: {missing_tracers}"
            )

    def select_relevant_times(self, ds: xr.Dataset) -> xr.Dataset:
        """Select records within the simulation window, clamped to the product range."""
        time_dim = self.dim_names["time"]

        select_start = self.start_time
        select_end = self.end_time

        if self.start_time.year < RIVR2O_MIN_YEAR:
            logging.info(
                "Simulation start time is before %s; using RIVR2O values from %s.",
                RIVR2O_MIN_YEAR,
                RIVR2O_MIN_YEAR,
            )
            select_start = datetime(RIVR2O_MIN_YEAR, 1, 1)

        if self.end_time.year > RIVR2O_MAX_YEAR:
            logging.info(
                "Simulation end time is after %s; using RIVR2O values from %s.",
                RIVR2O_MAX_YEAR,
                RIVR2O_MAX_YEAR,
            )
            select_end = datetime(RIVR2O_MAX_YEAR, 12, 31, 23, 59, 59)

        if select_start > select_end:
            boundary_year = (
                RIVR2O_MIN_YEAR
                if self.end_time.year < RIVR2O_MIN_YEAR
                else RIVR2O_MAX_YEAR
            )
            target_time = rivr2o_boundary_time(boundary_year)
            idx = int(np.abs(ds[time_dim].values - target_time).argmin())
            return ds.isel({time_dim: [idx]})

        # Annual files use mid-year (1 July) timestamps; match on calendar year.
        start_year = select_start.year
        end_year = select_end.year
        in_range = (ds[time_dim].dt.year >= start_year) & (
            ds[time_dim].dt.year <= end_year
        )
        if not bool(in_range.any()):
            available = sorted({int(y) for y in ds[time_dim].dt.year.values})
            raise ValueError(
                f"No RIVR2O files cover the requested years {start_year}-{end_year}. "
                f"Loaded files span years {available}. Provide RIVR2O files that "
                "overlap the simulation window."
            )
        return ds.sel({time_dim: ds[time_dim].where(in_range, drop=True)})

    def _adjust_lon_to_grid(self, lon: np.ndarray, *, straddle: bool) -> np.ndarray:
        """Put query longitudes in the same convention as the RIVR2O lon coordinate."""
        lon_dim = self.dim_names["longitude"]
        grid_lon = self.ds[lon_dim]
        if float(grid_lon.max()) > 180:
            return np.where(lon < 0, lon + 360, lon)
        if straddle:
            return np.where(lon > 180, lon - 360, lon)
        return lon

    def _nearest_time_index(
        self, time: datetime | np.datetime64 | np.integer | int | float | str
    ) -> int:
        """Index of the RIVR2O annual record closest to ``time``."""
        time_dim = self.dim_names["time"]
        target = rivr2o_coerce_time(time)
        return int(np.argmin(np.abs(self.ds[time_dim].values - target)))

    def _valid_dic_export_cell_indices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Grid cells with positive DIC export (union over loaded years).

        DIC spatial coverage is the same each year; DIN/DIP masks are not used.
        """
        # Cache result — this scans the full dataset and is called twice per forcing_concentrations call
        if not hasattr(self, "_cached_dic_indices"):
            lat_dim = self.dim_names["latitude"]
            lon_dim = self.dim_names["longitude"]
            time_dim = self.dim_names["time"]

            valid = (self.ds["DIC"] > 0).any(dim=time_dim)
            lat_indices, lon_indices = np.where(valid.values)
            if lat_indices.size == 0:
                raise ValueError(
                    "No grid cells with positive RIVR2O DIC export found in the "
                    "loaded dataset."
                )

            grid_lats = self.ds[lat_dim].values[lat_indices]
            grid_lons = self.ds[lon_dim].values[lon_indices]
            self._cached_dic_indices = (lat_indices, lon_indices, grid_lats, grid_lons)

        return self._cached_dic_indices

    def sample_at_points(
        self,
        lon: xr.DataArray | np.ndarray | float,
        lat: xr.DataArray | np.ndarray | float,
        *,
        straddle: bool = False,
        method: str = "nearest",
    ) -> xr.Dataset:
        """Sample MARBL tracer exports at point locations.

        For each query point, the nearest grid cell with positive ``DIC`` export
        is used for **all tracers and all years** in the loaded dataset. Cell
        choice does not depend on simulation year; exports vary in time only
        along the ``time`` dimension of the returned sample.

        Returns the full cell export at each point. ``forcing_concentrations``
        applies ``discharge_partition_weights`` so rivers on the same cell share
        export in proportion to discharge.

        Parameters
        ----------
        lon, lat : array-like or scalar
            Longitude and latitude of sample points in degrees.
        straddle : bool, optional
            If True, longitudes greater than 180° are converted to -180-180 before
            sampling. If False, negative longitudes are converted to 0-360.
        method : str, optional
            Accepted for API compatibility; sampling always uses the nearest cell
            with positive DIC export.

        Returns
        -------
        xr.Dataset
            Tracer variables with dimensions ``(time, points)`` when multiple
            locations are provided, or ``(time,)`` for a scalar point.
        """
        del method  # nearest DIC export cell, not xarray interp at the mouth
        lat_dim = self.dim_names["latitude"]
        lon_dim = self.dim_names["longitude"]

        scalar_point = not np.ndim(lon)
        query_lon = np.atleast_1d(np.asarray(lon, dtype=float))
        query_lat = np.atleast_1d(np.asarray(lat, dtype=float))
        query_lon = self._adjust_lon_to_grid(query_lon, straddle=straddle)

        nearest_lat, nearest_lon = self.nearest_dic_cell_indices_for_points(
            query_lon, query_lat, straddle=straddle
        )

        sampled = self.ds.isel(
            {
                lat_dim: xr.DataArray(nearest_lat, dims="points"),
                lon_dim: xr.DataArray(nearest_lon, dims="points"),
            }
        )

        if scalar_point:
            return sampled.squeeze("points", drop=True)

        return sampled

    def nearest_dic_cell_indices_for_points(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        *,
        straddle: bool = False,
        river_names: list[str] | None = None,
    ) -> tuple[list[int], list[int]]:
        """RIVR2O ``(lat, lon)`` indices of the nearest cell with positive DIC export."""
        from roms_tools.setup.utils import (
            build_kdtree_from_latlon,
            query_kdtree_nearest,
        )

        query_lon = np.atleast_1d(np.asarray(lon, dtype=float))
        query_lat = np.atleast_1d(np.asarray(lat, dtype=float))
        query_lon = self._adjust_lon_to_grid(query_lon, straddle=straddle)

        if river_names is not None and len(river_names) != len(query_lon):
            raise ValueError(
                "river_names must have the same length as lon/lat query points."
            )

        # Cache tree — built from the same DIC export cells every call
        if not hasattr(self, "_cached_dic_tree"):
            lat_indices, lon_indices, grid_lats, grid_lons = (
                self._valid_dic_export_cell_indices()
            )
            self._cached_dic_tree = (
                build_kdtree_from_latlon(grid_lats, grid_lons),
                lat_indices,
                lon_indices,
            )
        tree, lat_indices, lon_indices = self._cached_dic_tree

        nearest_lat, nearest_lon, _ = query_kdtree_nearest(
            tree,
            query_lat,
            query_lon,
            lat_indices,
            lon_indices,
            labels=river_names,
        )

        return nearest_lat, nearest_lon

    def discharge_partition_weights(
        self,
        river_volume: xr.DataArray,
        nearest_lat: list[int],
        nearest_lon: list[int],
    ) -> xr.DataArray:
        """Scale RIVR2O export by discharge for shared cells and through the year.

        For rivers on the same RIVR2O cell at each ``river_time`` step, export is
        allocated in proportion to ``Q_i / sum(Q_j)``. Weights are then normalized
        by the mean total discharge of that cell group over ``river_time`` so that
        ``export * weight / Q_i`` gives the same concentration for co-located rivers
        at every month, while months with higher discharge receive a larger share of
        the annual export (more carbon flux when ``Q`` is high).
        """
        import warnings

        # RuntimeWarning for division by zero in cell_weight is expected and handled by .where()
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
            module="dask",
        )

        volume = river_volume
        if "points" in volume.dims:
            volume = volume.rename(points="nriver")

        n_points = len(nearest_lat)
        if volume.sizes["nriver"] != n_points:
            raise ValueError(
                f"river_volume nriver size {volume.sizes['nriver']} does not match "
                f"{n_points} sample points."
            )

        cell_to_points: dict[tuple[int, int], list[int]] = defaultdict(list)
        for point_idx, (lat_idx, lon_idx) in enumerate(
            zip(nearest_lat, nearest_lon, strict=True)
        ):
            cell_to_points[(lat_idx, lon_idx)].append(point_idx)

        weights = xr.ones_like(volume, dtype=np.float64)
        time_dim = "river_time"
        shared_msgs: list[str] = []

        def _weight_column(values: xr.DataArray, point_idx: int) -> None:
            weights.values[:, point_idx] = np.asarray(values).reshape(-1)

        for (lat_idx, lon_idx), point_indices in cell_to_points.items():
            q_cell = volume.isel(nriver=point_indices)
            if len(point_indices) == 1:
                point_idx = point_indices[0]
                q_series = volume.isel(nriver=point_idx, drop=True)
                q_mean = q_series.mean(dim=time_dim, skipna=True)
                cell_weight = (q_series / q_mean).where(q_mean > 0, other=1.0)
                _weight_column(cell_weight, point_idx)
                continue

            q_sum = q_cell.sum(dim="nriver", min_count=1)
            q_sum_mean = q_sum.mean(dim=time_dim, skipna=True)
            n_shared = len(point_indices)
            for point_idx in point_indices:
                q_series = volume.isel(nriver=point_idx, drop=True)
                cell_weight = q_series / q_sum_mean
                cell_weight = cell_weight.where(q_sum_mean > 0, other=1.0 / n_shared)
                _weight_column(cell_weight, point_idx)
            shared_msgs.append(f"({lat_idx},{lon_idx})x{n_shared}")

        if shared_msgs:
            logging.info(
                "Partitioning RIVR2O export by discharge among rivers sharing "
                "grid cell(s): %s.",
                ", ".join(shared_msgs),
            )

        return weights

    @property
    def requires_calendar_discharge_time(self) -> bool:
        """RIVR2O BGC varies by calendar year and needs a calendar ``river_time`` axis."""
        return True

    @property
    def temporal_interpolation(self) -> RiverBGCTemporalInterpolation:
        return "calendar_year"

    @staticmethod
    def _align_concentration(
        values: xr.DataArray, nriver_coord: xr.DataArray
    ) -> xr.DataArray:
        """Match sample dimensions to ``river_tracer`` (river_time, nriver)."""
        if "points" in values.dims:
            values = values.rename(points="nriver")
        if "time" in values.dims:
            values = values.rename(time="river_time")
        values = values.assign_coords(nriver=nriver_coord)
        dims = [d for d in ("river_time", "nriver") if d in values.dims]
        return values.transpose(*dims)

    def _export_to_concentration(
        self,
        export: xr.DataArray,
        molar_mass_g: float,
        river_volume: xr.DataArray,
        target_time: xr.DataArray,
        *,
        partition_weight: xr.DataArray,
    ) -> xr.DataArray:
        """Convert RIVR2O annual export (10^6 g element yr-1) to mmol m-3."""
        time_dim = target_time.dims[0]
        source_time_dim = "time" if "time" in export.dims else export.dims[0]
        rivr2o_times = xr.DataArray(
            [
                rivr2o_boundary_time(int(year))
                for year in clamp_rivr2o_time(target_time).dt.year.values
            ],
            dims=[time_dim],
            coords={time_dim: target_time.coords[time_dim]},
        )
        requested_times = np.unique(rivr2o_times.values)
        source_times = export[source_time_dim].values
        missing_times = requested_times[~np.isin(requested_times, source_times)]
        if missing_times.size:
            export = export.reindex(
                {source_time_dim: np.concatenate([source_times, missing_times])},
                fill_value=np.nan,
            )
        export = export.sel({source_time_dim: rivr2o_times})
        export = self._align_concentration(export, river_volume.coords["nriver"])
        export = export * partition_weight
        mass_flux_g_s = export * 1e6 / SECONDS_PER_YEAR
        mmol_flux = mass_flux_g_s / molar_mass_g * 1000.0
        return (mmol_flux / river_volume).astype(np.float32)

    def forcing_concentrations(
        self,
        river_volume: xr.DataArray,
        abs_time: xr.DataArray,
        lons: np.ndarray,
        lats: np.ndarray,
        *,
        straddle: bool,
        river_names: list[str],
    ) -> dict[str, xr.DataArray]:
        """Convert RIVR2O exports to MARBL river tracer concentrations (mmol m-3).

        Samples export at each river mouth, applies ``discharge_partition_weights``,
        converts annual mass export to concentration, and maps to ROMS tracer names:

        - ``DIC`` = file ``DIC`` + ``DOC_l``
        - ``DOC`` = ``DOC_sl`` + ``POC``
        - ``DON`` and ``DOP`` from stoichiometric ratios of ``DOC_sl`` and ``POC``
        - ``ALK`` = file ``DIC``; ``DIC_ALT_CO2`` and ``ALK_ALT_CO2`` mirror ``DIC``
          and ``ALK``
        - ``NO3`` and ``PO4`` from renamed file fields

        Returns only the tracers listed above; other ROMS BGC tracers are supplied
        by the fill source in ``RiverForcing``.
        """
        nearest_lat, nearest_lon = self.nearest_dic_cell_indices_for_points(
            lons,
            lats,
            straddle=straddle,
            river_names=river_names,
        )
        partition_weight = self.discharge_partition_weights(
            river_volume.compute(), nearest_lat, nearest_lon
        )
        sampled = self.sample_at_points(lon=lons, lat=lats, straddle=straddle)
        conc_kw = {"partition_weight": partition_weight}

        dic_from_file = self._export_to_concentration(
            sampled["DIC"],
            _RIVR2O_MOLAR_MASS_G["DIC"],
            river_volume,
            abs_time,
            **conc_kw,
        )
        doc_l_conc = self._export_to_concentration(
            sampled["DOC_l"],
            _RIVR2O_MOLAR_MASS_G["DOC_l"],
            river_volume,
            abs_time,
            **conc_kw,
        )
        doc_sl_conc = self._export_to_concentration(
            sampled["DOC_sl"],
            _RIVR2O_MOLAR_MASS_G["DOC_sl"],
            river_volume,
            abs_time,
            **conc_kw,
        )
        poc_conc = self._export_to_concentration(
            sampled["POC"],
            _RIVR2O_MOLAR_MASS_G["POC"],
            river_volume,
            abs_time,
            **conc_kw,
        )

        dic_forcing = dic_from_file + doc_l_conc
        doc_forcing = doc_sl_conc + poc_conc
        alk_forcing = dic_from_file
        don_forcing = doc_sl_conc * _DON_FROM_DOC_SL + poc_conc * _DON_FROM_POC
        dop_forcing = doc_sl_conc * _DOP_FROM_DOC_SL + poc_conc * _DOP_FROM_POC

        return {
            "DIC": dic_forcing,
            "DOC": doc_forcing,
            "DON": don_forcing.astype(np.float32),
            "DOP": dop_forcing.astype(np.float32),
            "ALK": alk_forcing,
            "DIC_ALT_CO2": dic_forcing,
            "ALK_ALT_CO2": alk_forcing,
            "NO3": self._export_to_concentration(
                sampled["NO3"],
                _RIVR2O_MOLAR_MASS_G["NO3"],
                river_volume,
                abs_time,
                **conc_kw,
            ),
            "PO4": self._export_to_concentration(
                sampled["PO4"],
                _RIVR2O_MOLAR_MASS_G["PO4"],
                river_volume,
                abs_time,
                **conc_kw,
            ),
        }


def _decode_string(byte_array):
    # Decode each byte and handle errors with 'ignore'
    decoded_string = "".join(
        [
            x.decode("utf-8", errors="ignore")  # Ignore invalid byte sequences
            for x in byte_array.values
            if isinstance(x, bytes) and x != b" " and x is not np.nan
        ]
    )

    return decoded_string


def get_indices_of_nearest_grid_cell_for_rivers(
    dist: xr.DataArray, data: RiverDataset
) -> dict[str, list[tuple[int, int]]]:
    """Get the indices of the nearest grid cell for each river based on distance.

    Parameters
    ----------
    dist : xr.DataArray
        A 2D or 3D array representing distances from each river to coastal grid cells,
        with dimensions including "eta_rho" and "xi_rho".
    data : RiverDataset
        An instance of RiverDataset containing river names and dimension metadata.

    Returns
    -------
    dict[str, list[tuple[int, int]]]
        Dictionary mapping each river name to a list containing the (eta_rho, xi_rho) index
        of the closest coastal grid cell.
    """
    # Find indices of the nearest coastal grid cell for each river
    indices = dist.argmin(dim=["eta_rho", "xi_rho"])

    eta_rho_values = indices["eta_rho"].values
    xi_rho_values = indices["xi_rho"].values

    # Get the corresponding station indices and river names
    stations = indices["eta_rho"][data.dim_names["station"]].values
    names = (
        data.ds[data.var_names["name"]]
        .sel({data.dim_names["station"]: stations})
        .values
    )

    # Build dictionary of river name to grid index
    river_indices = {
        str(names[i]): [(int(eta_rho_values[i]), int(xi_rho_values[i]))]
        for i in range(len(stations))
    }

    return river_indices


def _deduplicate_river_names(
    ds: xr.Dataset, name_var: str, station_dim: str
) -> xr.Dataset:
    """Ensure river names are unique by appending _1, _2 to duplicates, excluding non-
    duplicates.
    """
    original = ds[name_var]

    # Force cast to plain Python strings
    names = [str(name) for name in original.values]

    # Count all names
    name_counts = Counter(names)
    seen: defaultdict[str, int] = defaultdict(int)

    unique_names = []
    for name in names:
        if name_counts[name] > 1:
            seen[name] += 1
            unique_names.append(f"{name}_{seen[name]}")
        else:
            unique_names.append(name)

    # Replace with updated names while preserving dtype, dims, attrs
    updated_array = xr.DataArray(
        data=np.array(unique_names, dtype=f"<U{max(len(n) for n in unique_names)}"),
        dims=original.dims,
        attrs=original.attrs,
    )
    ds[name_var] = updated_array

    return ds
