import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from roms_tools import Grid
from roms_tools.datasets.utils import (
    check_dataset,
    convert_to_float64,
    extrapolate_deepest_to_bottom,
    select_relevant_fields,
    select_relevant_times,
    validate_start_end_time,
)
from roms_tools.fill import LateralFill
from roms_tools.utils import load_data, wrap_longitudes
from roms_tools.vertical_coordinate import (
    compute_depth_coordinates,
)

DEFAULT_NR_BUFFER_POINTS = (
    20  # Default number of buffer points for subdomain selection.
)
# Balances performance and accuracy:
# - Too many points → more expensive computations
# - Too few points → potential boundary artifacts when lateral refill is performed
# See discussion: https://github.com/CWorthy-ocean/roms-tools/issues/153
# This default will be applied consistently across all datasets requiring lateral fill.


@dataclass(kw_only=True)
class ROMSDataset:
    """Represents ROMS model output.

    Parameters
    ----------
    path: str | Path | list[str | Path]
        Filename, or list of filenames with model output.
    grid : Grid
        Object representing the grid information.
    start_time : Optional[datetime], optional
        Start time for selecting relevant data. If not provided, no time-based filtering is applied.
    end_time : Optional[datetime], optional
        End time for selecting relevant data. If not provided, the dataset selects the time entry
        closest to `start_time` within the range `[start_time, start_time + 24 hours)`.
        If `start_time` is also not provided, no time-based filtering is applied.
    allow_flex_time: bool, optional
        Controls how strictly the dataset selects a time entry when `end_time` is not provided (relevant for initial conditions):

        - If False (default): requires an exact match to `start_time`. Raises a ValueError if no match exists.
        - If True: allows a +24h search window after `start_time` and selects the closest available
          time entry within that window. Raises a ValueError if none are found.

        Only used when `end_time` is None. Has no effect otherwise.
    dim_names: dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
    var_names: dict[str, str], optional
        Dictionary of variable names that are required in the dataset.
    opt_var_names: dict[str, str], optional
        Dictionary of variable names that are optional in the dataset.
        Defaults to an empty dictionary.
    model_reference_date : datetime, optional
        Reference date of ROMS simulation.
        If not specified, this is inferred from metadata of the model output
        If specified and does not coincide with metadata, a warning is raised.
    adjust_depth_for_sea_surface_height : bool, optional
        Whether to account for sea surface height variations when computing depth coordinates.
        Defaults to `False`.
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.
    """

    path: str | Path | list[str | Path]
    """Filename, or list of filenames with model output."""
    grid: Grid
    """Object representing the grid information."""
    start_time: datetime | None = None
    """Start time for selecting relevant data."""
    end_time: datetime | None = None
    """End time for selecting relevant data."""
    allow_flex_time: bool = False
    """Controls how strictly the dataset selects a time entry when `end_time` is not provided."""
    dim_names: dict[str, str] = field(
        default_factory=lambda: {
            "eta_rho": "eta_rho",
            "xi_rho": "xi_rho",
            "time": "time",
        }
    )
    """Dictionary specifying the names of dimensions in the dataset."""
    var_names: dict[str, str] | None = None
    """Dictionary of variable names that are required in the dataset."""
    opt_var_names: dict[str, str] = field(default_factory=dict)
    """Dictionary of variable names that are optional in the dataset."""
    use_dask: bool = False
    """Whether to use dask for processing."""
    model_reference_date: datetime | None = None
    """Reference date of ROMS simulation."""
    adjust_depth_for_sea_surface_height: bool | None = False
    """Whether to account for sea surface height variations when computing depth
    coordinates."""

    ds: xr.Dataset = field(init=False, repr=False)
    """An xarray Dataset containing the ROMS output."""

    def __post_init__(self):
        validate_start_end_time(self.start_time, self.end_time)
        ds = self.load_data()
        self._check_consistency_data_grid(ds)

        self._set_default_var_names(ds)

        check_dataset(ds, self.dim_names, self.var_names)
        self._check_vertical_coordinate(ds)
        self._infer_model_reference_date_from_metadata(ds)
        ds = self._add_absolute_time(ds)

        ds = self.select_relevant_fields(ds)
        if self.start_time is not None:
            ds = self.select_relevant_times(ds)

        ds = self._add_lat_lon_coords_and_masks(ds)
        self.ds = ds

        # Dataset for depth coordinates
        self.ds_depth_coords = xr.Dataset()

    def _check_consistency_data_grid(self, ds: xr.Dataset) -> None:
        """
        Ensure that the input dataset `ds` is consistent with the grid dataset.

        Specifically, checks that the dimensions of the dataset match the grid's
        `eta_rho` and `xi_rho` dimensions.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to check against the grid.

        Raises
        ------
        ValueError
            If the `eta_rho` or `xi_rho` dimensions of `ds` do not match those of `self.grid.ds`.
        """
        eta = self.dim_names["eta_rho"]
        xi = self.dim_names["xi_rho"]
        grid_eta = self.grid.ds.sizes.get(eta)
        grid_xi = self.grid.ds.sizes.get(xi)
        ds_eta = ds.sizes.get(eta)
        ds_xi = ds.sizes.get(xi)

        if grid_eta != ds_eta or grid_xi != ds_xi:
            raise ValueError(
                f"Inconsistent dataset dimensions: "
                f"grid ({eta}={grid_eta}, {xi}={grid_xi}), "
                f"dataset ({eta}={ds_eta}, {xi}={ds_xi})."
            )

    def _set_default_var_names(self, ds: xr.Dataset) -> None:
        """
        Ensure ``self.var_names`` is a valid mapping.

        If ``self.var_names`` is ``None``, it is initialized as an identity
        mapping for all variables in ``ds``.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset whose variable names are used when creating defaults.
        """
        if self.var_names is None:
            self.var_names = {name: name for name in ds.data_vars}

    def _get_depth_coordinates(self, depth_type="layer", locations=["rho"]):
        """Ensure depth coordinates are stored for a given location and depth type.

        Calculates vertical depth coordinates (layer or interface) for specified locations (e.g., rho, u, v points)
        and updates them in the dataset (`self.ds`).

        Parameters
        ----------
        depth_type : str
            The type of depth coordinate to compute. Valid options:
            - "layer": Compute layer depth coordinates.
            - "interface": Compute interface depth coordinates.
        locations : list[str], optional
            Locations for which to compute depth coordinates. Default is ["rho", "u", "v"].
            Valid options include:
            - "rho": Depth coordinates at rho points.
            - "u": Depth coordinates at u points.
            - "v": Depth coordinates at v points.

        Updates
        -------
        self.ds_depth_coords : xarray.Dataset

        Raises
        ------
        ValueError
            If `adjust_depth_for_sea_surface_height` is enabled but `zeta` is missing from `self.ds`.

        Notes
        -----
        - This method relies on the `compute_depth_coordinates` function to perform calculations.
        - If `adjust_depth_for_sea_surface_height` is `True`, the method accounts for variations
          in sea surface height (`zeta`).
        """
        if self.adjust_depth_for_sea_surface_height:
            if "zeta" not in self.ds:
                raise ValueError(
                    "`zeta` is required in provided ROMS output when `adjust_depth_for_sea_surface_height` is enabled."
                )
            zeta = self.ds.zeta
        else:
            zeta = 0

        for location in locations:
            var_name = f"{depth_type}_depth_{location}"
            if var_name not in self.ds_depth_coords:
                depth_da = compute_depth_coordinates(
                    self.grid.ds, zeta, depth_type, location
                )
                depth_da = depth_da.assign_coords(
                    {
                        f"lat_{location}": self.grid.ds[f"lat_{location}"],
                        f"lon_{location}": self.grid.ds[f"lon_{location}"],
                    }
                )
                self.ds_depth_coords[var_name] = depth_da

    def load_data(self) -> xr.Dataset:
        """Load the ROMS data."""
        ds = load_data(
            filename=self.path,
            dim_names={"time": "time"},
            use_dask=self.use_dask,
            decode_times=False,
            decode_timedelta=False,
            time_chunking=True,
            force_combine_nested=True,
        )

        return ds

    def select_relevant_fields(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Return a subset of the dataset containing only the required and optional
        variables defined for this object.

        Variables retained are those listed in ``self.var_names`` and
        ``self.opt_var_names``. Any other data variables are removed, except for
        the special variable ``"mask"``, which is always preserved if present.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset from which relevant variables will be selected.

        Returns
        -------
        xr.Dataset
            A new dataset containing only the required variables specified in
            ``self.var_names`` and the optional variables specified in
            ``self.opt_var_names``, along with ``"mask"`` if present.
        """
        return select_relevant_fields(
            ds,
            [*self.var_names.values(), *self.opt_var_names.values()],  # type: ignore
        )

    def select_relevant_times(self, ds: xr.Dataset) -> xr.Dataset:
        """Select a subset of the dataset based on the specified time range.

        This method filters the dataset to include all records between `start_time` and `end_time`.
        Additionally, it ensures that one record at or before `start_time` and one record at or
        after `end_time` are included, even if they fall outside the strict time range.

        If no `end_time` is specified, the method will select the time range of
        [start_time, start_time + 24 hours) and return the closest time entry to `start_time` within that range.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset to be filtered. Must contain a time dimension.

        Returns
        -------
        xr.Dataset
            A dataset filtered to the specified time range, including the closest entries
            at or before `start_time` and at or after `end_time` if applicable.

        Raises
        ------
        ValueError
            If no matching times are found between `start_time` and `start_time + 24 hours`.

        Warns
        -----
        UserWarning
            If the dataset contains exactly 12 time steps but the climatology flag is not set.
            This may indicate that the dataset represents climatology data.

        UserWarning
            If no records at or before `start_time` or no records at or after `end_time` are found.

        UserWarning
            If the dataset does not contain any time dimension or the time dimension is incorrectly named.

        Notes
        -----
        - If the `climatology` flag is set and `end_time` is not provided, the method will
          interpolate initial conditions from climatology data.
        - If the dataset uses `cftime` datetime objects, these will be converted to standard
          `np.datetime64` objects before filtering.
        """
        time_dim = self.dim_names["time"]

        # Ensure start_time is not None for type safety
        if self.start_time is None:
            raise ValueError("select_relevant_times called but start_time is None.")

        ds = select_relevant_times(
            ds=ds,
            time_dim=time_dim,
            time_coord="time",
            start_time=self.start_time,
            end_time=self.end_time,
            allow_flex_time=self.allow_flex_time,
        )

        return ds

    def _infer_model_reference_date_from_metadata(self, ds: xr.Dataset) -> None:
        """Infer and validate the model reference date from `ocean_time` metadata.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset with an `ocean_time` variable and a `long_name` attribute
            in the format `Time since YYYY/MM/DD`.

        Raises
        ------
        ValueError
            If `self.model_reference_date` is not set and the reference date cannot
            be inferred, or if the inferred date does not match `self.model_reference_date`.

        Warns
        -----
        UserWarning
            If `self.model_reference_date` is set but the reference date cannot be inferred.
        """
        # Check if 'long_name' exists in the attributes of 'ocean_time'
        if "long_name" in ds.ocean_time.attrs:
            input_string = ds.ocean_time.attrs["long_name"]
            match = re.search(r"(\d{4})/(\d{2})/(\d{2})", input_string)

            if match:
                # If a match is found, extract year, month, day and create the inferred date
                year, month, day = map(int, match.groups())
                inferred_date = datetime(year, month, day)

                if hasattr(self, "model_reference_date") and self.model_reference_date:
                    # Check if the inferred date matches the provided model reference date
                    if self.model_reference_date != inferred_date:
                        raise ValueError(
                            f"Mismatch between `self.model_reference_date` ({self.model_reference_date}) "
                            f"and inferred reference date ({inferred_date})."
                        )
                else:
                    # Set the model reference date if not already set
                    self.model_reference_date = inferred_date
            else:
                # Handle case where no match is found
                if hasattr(self, "model_reference_date") and self.model_reference_date:
                    logging.warning(
                        "Could not infer the model reference date from the metadata. "
                        "`self.model_reference_date` will be used.",
                    )
                else:
                    raise ValueError(
                        "Model reference date could not be inferred from the metadata, "
                        "and `self.model_reference_date` is not set."
                    )
        else:
            # Handle case where 'long_name' attribute doesn't exist
            if hasattr(self, "model_reference_date") and self.model_reference_date:
                logging.warning(
                    "`long_name` attribute not found in ocean_time. "
                    "`self.model_reference_date` will be used instead.",
                )
            else:
                raise ValueError(
                    "Model reference date could not be inferred from the metadata, "
                    "and `self.model_reference_date` is not set."
                )

    def _check_vertical_coordinate(self, ds: xr.Dataset) -> None:
        """Check that the vertical coordinate parameters in the dataset are consistent
        with the model grid.

        This method compares the vertical coordinate parameters (`theta_s`, `theta_b`, `hc`, `Cs_r`, `Cs_w`) in
        the provided dataset (`ds`) with those in the model grid (`self.grid`). The first three parameters are
        checked for exact equality, while the last two are checked for numerical closeness.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset containing vertical coordinate parameters in its attributes, such as `theta_s`, `theta_b`,
            `hc`, `Cs_r`, and `Cs_w`.

        Raises
        ------
        ValueError
            If the vertical coordinate parameters do not match the expected values (based on exact or approximate equality).

        Notes
        -----
        - Missing attributes trigger a warning instead of an exception.
        - `theta_s`, `theta_b`, and `hc` are checked for exact equality using `np.array_equal`.
        - `Cs_r` and `Cs_w` are checked for numerical closeness using `np.allclose`.
        """
        required_exact = ["theta_s", "theta_b", "hc"]
        required_close = ["Cs_r", "Cs_w"]

        # Check exact equality
        for param in required_exact:
            value = ds.attrs.get(param, None)
            if value is None:
                logging.warning(
                    f"Dataset is missing attribute '{param}'. Skipping this check."
                )
                continue
            if not np.array_equal(getattr(self.grid, param), value):
                raise ValueError(
                    f"{param} from grid ({getattr(self.grid, param)}) does not match dataset ({value})."
                )

        # Check numerical closeness
        for param in required_close:
            value = ds.attrs.get(param, None)
            if value is None:
                logging.warning(
                    f"Dataset is missing attribute '{param}'. Skipping this check."
                )
                continue
            grid_value = getattr(self.grid.ds, param)
            if not np.allclose(grid_value, value):
                raise ValueError(
                    f"{param} from grid ({grid_value}) is not close to dataset ({value})."
                )

    def _add_absolute_time(self, ds: xr.Dataset) -> xr.Dataset:
        """Add absolute time as a coordinate to the dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing "ocean_time" in seconds since the model reference date.

        Returns
        -------
        xarray.Dataset
            Dataset with absolute time added.
        """
        if self.model_reference_date is None:
            raise ValueError(
                "`model_reference_date` must be set before computing absolute time."
            )

        ocean_time_seconds = ds["ocean_time"].values

        abs_time = np.array(
            [
                self.model_reference_date + timedelta(seconds=seconds)
                for seconds in ocean_time_seconds
            ]
        )

        abs_time = xr.DataArray(
            abs_time, dims=["time"], coords={"time": ds["ocean_time"]}
        )
        abs_time.attrs["long_name"] = "absolute time"
        ds = ds.assign_coords({"abs_time": abs_time})
        ds = ds.drop_vars("time")
        ds = ds.set_index(time="abs_time")

        return ds

    def _add_lat_lon_coords_and_masks(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Attach horizontal coordinate fields (lat/lon) and grid masks to a dataset.

        This method augments the input dataset with the appropriate geographic
        coordinates taken from the grid object. It *always* adds `lat_rho` and
        `lon_rho`. If the dataset contains staggered horizontal dimensions
        (`xi_u` or `eta_v`), the corresponding `u`- or `v`-point coordinates are
        added as well (`lat_u`, `lon_u`, `lat_v`, `lon_v`).

        In addition, the grid masks (`mask_rho`, `mask_u`, `mask_v`) are copied
        into the dataset for later use in operations such as lateral filling.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to be augmented with horizontal coordinates and masks.

        Returns
        -------
        xarray.Dataset
            A new dataset with the appropriate latitude/longitude coordinates
            and grid masks assigned based on the dataset's horizontal staggering.

        Notes
        -----
        This routine does not modify the input dataset in place; a new dataset
        with added coordinates and mask variables is returned.
        """
        coords_to_add = {
            "lat_rho": self.grid.ds["lat_rho"],
            "lon_rho": self.grid.ds["lon_rho"],
        }
        vars_to_add = {"mask_rho": self.grid.ds["mask_rho"]}

        if "xi_u" in ds.dims:
            coords_to_add.update(
                {"lat_u": self.grid.ds["lat_u"], "lon_u": self.grid.ds["lon_u"]}
            )
            vars_to_add.update({"mask_u": self.grid.ds["mask_u"]})
        if "eta_v" in ds.dims:
            coords_to_add.update(
                {"lat_v": self.grid.ds["lat_v"], "lon_v": self.grid.ds["lon_v"]}
            )
            vars_to_add.update({"mask_v": self.grid.ds["mask_v"]})

        ds = ds.assign_coords(coords_to_add)
        for mask_name, mask_data in vars_to_add.items():
            ds[mask_name] = mask_data

        return ds

    def choose_subdomain(
        self,
        target_coords: dict[str, Any],
        buffer_points: int = DEFAULT_NR_BUFFER_POINTS,
    ) -> None:
        """Selects a subdomain from the xarray Dataset based on specified target
        coordinates, extending the selection by a defined buffer. Adjusts longitude
        ranges as necessary to accommodate the dataset's expected range and handles
        potential discontinuities.

        Parameters
        ----------
        target_coords : dict
            A dictionary containing the target latitude and longitude coordinates, typically
            with keys "lat", "lon", and "straddle".
        buffer_points : int
            The number of grid points to extend beyond the specified latitude and longitude
            ranges when selecting the subdomain. Defaults to 20.

        Returns
        -------
        None
            The subdomain of the xarray Dataset is assigned to `self.ds`.

        Raises
        ------
        ValueError
            If the selected latitude or longitude range does not intersect with the dataset.
        """
        subdomain = choose_subdomain(
            self.ds, self.grid.ds, target_coords, buffer_points
        )
        self.ds = subdomain

        subdomain_grid_ds = choose_subdomain(
            self.grid.ds, self.grid.ds, target_coords, buffer_points
        )

        self.grid = self.grid.copy_with_ds(subdomain_grid_ds)

    def convert_to_float64(self) -> None:
        """Convert all data variables in the dataset to float64.

        This method updates the dataset by converting all of its data variables to the
        `float64` data type, ensuring consistency for numerical operations that require
        high precision. The dataset is modified in place.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method modifies the dataset in place and does not return anything.
        """
        ds = convert_to_float64(self.ds)
        self.ds = ds

        return None

    def extrapolate_deepest_to_bottom(self):
        """Extrapolate deepest non-NaN values to fill bottom NaNs along the s (depth)
        dimension.

        For each variable with a depth dimension, fills missing values at the bottom by
        propagating the deepest available data downward.
        """
        self.ds = extrapolate_deepest_to_bottom(self.ds, "s_rho")

    def apply_lateral_fill(self) -> None:
        """Apply lateral fill to variables using available masks and grid dimensions.

        Lateral fill is applied only when:
        - A corresponding mask exists in the dataset, and
        - At least one variable is defined on the associated horizontal grid.

        Raises
        ------
        ValueError
            If variables exist on a horizontal grid (rho, u, or v) but the
            corresponding mask is missing.
        """
        # Mapping of horizontal dims to required mask name
        dim_to_mask: dict[tuple[str, str], str] = {
            ("eta_rho", "xi_rho"): "mask_rho",
            ("eta_rho", "xi_u"): "mask_u",
            ("eta_v", "xi_rho"): "mask_v",
        }
        horiz_dim_order = ("eta_rho", "eta_v", "xi_rho", "xi_u")

        # Identify which horizontal dim sets are actually used
        used_dim_sets: set[tuple[str, str]] = set()

        for var in self.ds.data_vars.values():
            horiz_dims = tuple(d for d in horiz_dim_order if d in var.dims)
            if len(horiz_dims) == 2:
                used_dim_sets.add(horiz_dims)

        # Enforce required masks for all grids (rho, u, v)
        for dims, mask_name in dim_to_mask.items():
            if dims in used_dim_sets and mask_name not in self.ds:
                raise ValueError(
                    f"Variable(s) found on grid {tuple(dims)}, but required mask "
                    f"'{mask_name}' is missing from the dataset."
                )

        # Build lateral fillers
        lateral_fillers: dict[tuple[str, str], LateralFill] = {
            dims: LateralFill(
                xr.where(self.ds[mask_name] == 1, True, False),
                list(dims),  # ORDER PRESERVED
            )
            for dims, mask_name in dim_to_mask.items()
            if dims in used_dim_sets
        }

        # Apply lateral fill
        for var_name, var in self.ds.data_vars.items():
            if var_name.startswith("mask"):
                continue

            # Keep dims in canonical order
            var_horiz_dims = tuple(d for d in horiz_dim_order if d in var.dims)
            if len(var_horiz_dims) == 2:
                filler = lateral_fillers.get(var_horiz_dims)
                if filler is not None:
                    self.ds[var_name] = filler.apply(var)


def choose_subdomain(
    ds: xr.Dataset,
    ds_grid: xr.Dataset,
    target_coords: dict[str, Any],
    buffer_points: int = DEFAULT_NR_BUFFER_POINTS,
):
    """Selects a subdomain from the xarray Dataset based on specified target
    coordinates, extending the selection by a defined buffer. Adjusts longitude
    ranges as necessary to accommodate the dataset's expected range and handles
    potential discontinuities.

    Parameters
    ----------
    ds : xr.Dataset
        The full ROMS xarray Dataset to subset.
    ds_grid: xr.Dataset
        Dataset containing the grid coordinates, in particular `pm` and `pn`.
    target_coords : dict
        A dictionary containing the target latitude and longitude coordinates, typically
        with keys "lat", "lon", and "straddle".
    buffer_points : int
        The number of grid points to extend beyond the specified latitude and longitude
        ranges when selecting the subdomain. Defaults to 20.

    Returns
    -------
    xr.Dataset
        Returns the subset of the original dataset.

    Raises
    ------
    ValueError
        If the selected latitude or longitude range does not intersect with the dataset.
    """
    # Extract lat/lon min/max from target
    lat_min = target_coords["lat"].min().values
    lat_max = target_coords["lat"].max().values
    lon_min = target_coords["lon"].min().values
    lon_max = target_coords["lon"].max().values

    # Compute buffer in degrees
    dx = 0.5 * ((1 / ds_grid.pm).mean() + (1 / ds_grid.pn).mean())
    buffer = dx * buffer_points
    lat_center = np.deg2rad(0.5 * (lat_min + lat_max))
    margin_lat = buffer / 111_320.0
    margin_lon = buffer / (111_320.0 * np.cos(lat_center))

    lon_min_buf = lon_min - margin_lon
    lon_max_buf = lon_max + margin_lon

    # Normalize buffered bounds to target convention
    if target_coords["straddle"]:
        # [-180, 180]
        if lon_min_buf < -180:
            lon_min_buf += 360
        if lon_max_buf > 180:
            lon_max_buf -= 360
    else:
        # [0, 360]
        if lon_min_buf < 0:
            lon_min_buf += 360
        if lon_max_buf >= 360:
            lon_max_buf -= 360

    # Wrap dataset longitudes to target convention
    ds = wrap_longitudes(ds, target_coords["straddle"])

    # Rho points
    location = "rho"
    eta_dim, xi_dim = "eta_rho", "xi_rho"
    lat_coord, lon_coord = f"lat_{location}", f"lon_{location}"
    _check_latlon_coords(ds, eta_dim, xi_dim, location)
    ds_lon = ds[lon_coord]

    if lon_max_buf < lon_min_buf:  # crosses dateline
        subset_mask_lon = (ds_lon >= lon_min_buf) | (ds_lon <= lon_max_buf)
    else:
        subset_mask_lon = (ds_lon >= lon_min_buf) & (ds_lon <= lon_max_buf)

    # Full mask including latitude
    subset_mask = (
        (ds[lat_coord] >= lat_min - margin_lat)
        & (ds[lat_coord] <= lat_max + margin_lat)
        & subset_mask_lon
    )

    eta_mask = subset_mask.any(dim=xi_dim)
    xi_mask = subset_mask.any(dim=eta_dim)
    eta_indices = np.where(eta_mask)[0]
    xi_indices = np.where(xi_mask)[0]
    first_eta, last_eta = eta_indices[0], eta_indices[-1]
    first_xi, last_xi = xi_indices[0], xi_indices[-1]

    # Subset rho points
    ds = ds.isel(
        **{
            "eta_rho": slice(first_eta, last_eta + 1),
            "xi_rho": slice(first_xi, last_xi + 1),
        }
    )

    # Subset u points only if these dimensions exist
    if "xi_u" in ds.dims:
        ds = ds.isel(
            **{
                "xi_u": slice(first_xi, last_xi),
            }
        )

    # Subset v points only if these dimensions exist
    if "eta_v" in ds.dims:
        ds = ds.isel(
            **{
                "eta_v": slice(first_eta, last_eta),
            }
        )

    return ds


def _check_latlon_coords(
    ds: xr.Dataset, eta_dim: str, xi_dim: str, location: str
) -> None:
    """
    Ensure latitude and longitude coordinates exist for a given grid location.

    Raises ValueError if the expected coordinates are missing.
    """
    if eta_dim in ds.dims and xi_dim in ds.dims:
        lat_coord = f"lat_{location}"
        lon_coord = f"lon_{location}"

        if lat_coord not in ds.coords or lon_coord not in ds.coords:
            raise ValueError(
                f"Dataset missing coordinates for location '{location}': "
                f"expected '{lat_coord}' and '{lon_coord}'"
            )
