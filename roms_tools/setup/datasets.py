import time
import xarray as xr
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Optional, Union, List
from pathlib import Path
import logging
from roms_tools.utils import _load_data
from roms_tools.setup.utils import (
    assign_dates_to_climatology,
    interpolate_from_climatology,
    get_time_type,
    convert_cftime_to_datetime,
    one_dim_fill,
    gc_dist,
)
from roms_tools.download import (
    download_correction_data,
    download_topo,
    download_river_data,
)
from roms_tools.setup.fill import LateralFill

# lat-lon datasets


@dataclass(frozen=True, kw_only=True)
class Dataset:
    """Represents forcing data on original grid.

    Parameters
    ----------
    filename : Union[str, Path, List[Union[str, Path]]]
        The path to the data file(s). Can be a single string (with or without wildcards), a single Path object,
        or a list of strings or Path objects containing multiple files.
    start_time : Optional[datetime], optional
        The start time for selecting relevant data. If not provided, the data is not filtered by start time.
    end_time : Optional[datetime], optional
        The end time for selecting relevant data. If not provided, only data at the start_time is selected if start_time is provided,
        or no filtering is applied if start_time is not provided.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
    var_names: Dict[str, str]
        Dictionary of variable names that are required in the dataset.
    climatology : bool
        Indicates whether the dataset is climatological. Defaults to False.
    use_dask: bool
        Indicates whether to use dask for chunking. If True, data is loaded with dask; if False, data is loaded eagerly. Defaults to False.
    apply_post_processing: bool
        Indicates whether to post-process the dataset for futher use. Defaults to True.

    Attributes
    ----------
    is_global : bool
        Indicates whether the dataset covers the entire globe.
    ds : xr.Dataset
        The xarray Dataset containing the forcing data on its original grid.

    Examples
    --------
    >>> dataset = Dataset(
    ...     filename="data.nc",
    ...     start_time=datetime(2022, 1, 1),
    ...     end_time=datetime(2022, 12, 31),
    ... )
    """

    filename: Union[str, Path, List[Union[str, Path]]]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    dim_names: Dict[str, str] = field(
        default_factory=lambda: {
            "longitude": "longitude",
            "latitude": "latitude",
            "time": "time",
        }
    )
    var_names: Dict[str, str]
    climatology: Optional[bool] = False
    use_dask: Optional[bool] = False
    apply_post_processing: Optional[bool] = True

    is_global: bool = field(init=False, repr=False)
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        """
        Post-initialization processing:
        1. Loads the dataset from the specified filename.
        2. Applies time filtering based on start_time and end_time if provided.
        3. Selects relevant fields as specified by var_names.
        4. Ensures latitude values and depth values are in ascending order.
        5. Checks if the dataset covers the entire globe and adjusts if necessary.
        """

        # Validate start_time and end_time
        if self.start_time is not None and not isinstance(self.start_time, datetime):
            raise TypeError(
                f"start_time must be a datetime object, but got {type(self.start_time).__name__}."
            )
        if self.end_time is not None and not isinstance(self.end_time, datetime):
            raise TypeError(
                f"end_time must be a datetime object, but got {type(self.end_time).__name__}."
            )

        ds = self.load_data()
        ds = self.clean_up(ds)
        self.check_dataset(ds)

        # Select relevant times
        if "time" in self.dim_names and self.start_time is not None:
            ds = self.add_time_info(ds)
            ds = self.select_relevant_times(ds)

            if self.dim_names["time"] != "time":
                ds = ds.rename({self.dim_names["time"]: "time"})

        # Select relevant fields
        ds = self.select_relevant_fields(ds)

        # Make sure that latitude is ascending
        ds = self.ensure_dimension_is_ascending(ds, dim="latitude")
        # Make sure there are no 360 degree jumps in longitude
        ds = self.ensure_dimension_is_ascending(ds, dim="longitude")

        if "depth" in self.dim_names:
            # Make sure that depth is ascending
            ds = self.ensure_dimension_is_ascending(ds, dim="depth")

        # Enforce double precision to ensure reproducibility
        ds = convert_to_float64(ds)

        self.infer_horizontal_resolution(ds)

        # Check whether the data covers the entire globe
        object.__setattr__(self, "is_global", self.check_if_global(ds))
        object.__setattr__(self, "ds", ds)

        if self.apply_post_processing:
            self.post_process()

    def load_data(self) -> xr.Dataset:
        """Load dataset from the specified file.

        Returns
        -------
        ds : xr.Dataset
            The loaded xarray Dataset containing the forcing data.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If a list of files is provided but self.dim_names["time"] is not available or use_dask=False.
        """

        ds = _load_data(self.filename, self.dim_names, self.use_dask)

        return ds

    def clean_up(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """Dummy method to be overridden by child classes to clean up the dataset.

        This method is intended as a placeholder and should be implemented in subclasses
        to provide specific functionality.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset to be cleaned up.

        Returns
        -------
        xr.Dataset
            The cleaned-up xarray Dataset (as implemented by child classes).
        """
        return ds  # Default behavior (no-op, subclasses should override)

    def check_dataset(self, ds: xr.Dataset) -> None:
        """Check if the dataset contains the specified variables and dimensions.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset to check.

        Raises
        ------
        ValueError
            If the dataset does not contain the specified variables or dimensions.
        """

        _check_dataset(ds, self.dim_names, self.var_names)

    def select_relevant_fields(self, ds) -> xr.Dataset:
        """Selects and returns a subset of the dataset containing only the variables
        specified in `self.var_names`.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset from which variables will be selected.

        Returns
        -------
        xr.Dataset
            A dataset containing only the variables specified in `self.var_names`.
        """

        for var in ds.data_vars:
            if var not in self.var_names.values() and var != "mask":
                ds = ds.drop_vars(var)

        return ds

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

        ds = _select_relevant_times(
            ds, time_dim, self.start_time, self.end_time, self.climatology
        )

        return ds

    def ensure_dimension_is_ascending(
        self, ds: xr.Dataset, dim="latitude"
    ) -> xr.Dataset:
        """Ensure that the specified dimension in the dataset is in ascending order.

        This function checks the order of values along the specified dimension. If they
        are in descending order, it reverses the dimension to make it ascending. For
        the "longitude" dimension, if it has a discontinuity (e.g., [0, 180][-180, 0]),
        the function adjusts values to eliminate the 360-degree jump, transforming
        the range into a continuous [0, 360) span.

        Parameters
        ----------
        ds : xr.Dataset
            The input `xarray.Dataset` whose dimension is to be checked and, if necessary, reordered.
        dim : str, optional
            The name of the dimension to check for ascending order.
            Defaults to "latitude". The dimension is expected to be one of the keys in `self.dim_names`.

        Returns
        -------
        xr.Dataset
            A new `xarray.Dataset` with the specified dimension in ascending order.
            - If the dimension was already in ascending order, the original dataset is returned unchanged.
            - If the dimension was in descending order, the dataset is returned with the dimension reversed.
            - If the dimension is "longitude" with a discontinuity (e.g., [0, 180][-180, 0]), the values are adjusted to eliminate the 360-degree jump.
        """
        # Check if the dimension is in descending order and reverse if needed
        diff = np.diff(ds[self.dim_names[dim]])
        if np.all(diff < 0):
            ds = ds.isel(**{self.dim_names[dim]: slice(None, None, -1)})

        # Check for a discontinuity in longitude and adjust values if present
        elif np.any(diff < 0) and dim == "longitude":
            ds[self.dim_names[dim]] = xr.where(
                ds[self.dim_names[dim]] < 0,
                ds[self.dim_names[dim]] + 360,
                ds[self.dim_names[dim]],
            )

        return ds

    def infer_horizontal_resolution(self, ds: xr.Dataset):
        """Estimate and set the average horizontal resolution of a dataset based on
        latitude and longitude spacing.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing latitude and longitude dimensions.

        Sets
        ----
        resolution : float
            The average horizontal resolution, derived from the mean spacing
            between points in latitude and longitude.
        """
        lat_dim = self.dim_names["latitude"]
        lon_dim = self.dim_names["longitude"]

        # Calculate mean difference along latitude and longitude
        lat_resolution = ds[lat_dim].diff(dim=lat_dim).mean(dim=lat_dim)
        lon_resolution = ds[lon_dim].diff(dim=lon_dim).mean(dim=lon_dim)

        # Compute the average horizontal resolution
        resolution = np.mean([lat_resolution, lon_resolution])

        # Set the computed resolution as an attribute
        object.__setattr__(self, "resolution", resolution)

    def compute_minimal_grid_spacing(self, ds: xr.Dataset):
        """Compute the minimal grid spacing in a dataset based on latitude and longitude
        spacing, considering Earth's radius.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing latitude and longitude dimensions.

        Returns
        -------
        minimal_spacing : float
            The smallest horizontal grid spacing derived from the latitude
            and longitude differences, in meters.
        """

        r_earth = 6371315.0
        lat_dim = self.dim_names["latitude"]
        lon_dim = self.dim_names["longitude"]

        # Get latitude and longitude values from the dataset
        latitudes = ds[lat_dim].values
        longitudes = ds[lon_dim].values

        # Compute differences along latitude and longitude
        lat_diff = np.abs(np.diff(latitudes)).min()  # Minimal latitude spacing
        lon_diff = np.abs(np.diff(longitudes)).min()  # Minimal longitude spacing

        # Latitude spacing is constant at all longitudes
        min_lat_spacing = (2 * np.pi * r_earth * lat_diff) / 360

        # Longitude spacing varies with latitude
        min_lon_spacing = (
            2 * np.pi * r_earth * lon_diff * np.cos(np.radians(latitudes.min()))
        ) / 360

        # The minimal spacing is the smaller of the two
        minimal_spacing = min(min_lat_spacing, min_lon_spacing)

        return minimal_spacing

    def check_if_global(self, ds) -> bool:
        """Checks if the dataset covers the entire globe in the longitude dimension.

        This function calculates the mean difference between consecutive longitude values.
        It then checks if the difference between the first and last longitude values (plus 360 degrees)
        is close to this mean difference, within a specified tolerance. If it is, the dataset is considered
        to cover the entire globe in the longitude dimension.

        Returns
        -------
        bool
            True if the dataset covers the entire globe in the longitude dimension, False otherwise.
        """
        dlon_mean = (
            ds[self.dim_names["longitude"]].diff(dim=self.dim_names["longitude"]).mean()
        )
        dlon = (
            ds[self.dim_names["longitude"]][0] - ds[self.dim_names["longitude"]][-1]
        ) % 360.0
        is_global = np.isclose(dlon, dlon_mean, rtol=0.0, atol=1e-3).item()

        return is_global

    def concatenate_longitudes(self, ds, end="upper", verbose=False):
        """Concatenates fields in dataset twice along the longitude dimension.

        Parameters
        ----------
        ds: xr.Dataset
            The dataset to be concatenated. The longitude dimension must be present in this dataset.
        end : str, optional
            Specifies which end to shift the longitudes.
            Options are:
                - "lower": shifts longitudes by -360 degrees and concatenates to the lower end.
                - "upper": shifts longitudes by +360 degrees and concatenates to the upper end.
                - "both": shifts longitudes by -360 degrees and 360 degrees and concatenates to both ends.
            Default is "upper".
        verbose : bool, optional
            If True, print message if dataset is concatenated along longitude dimension.
            Defaults to False.

        Returns
        -------
        ds_concatenated : xr.Dataset
            The concatenated dataset.
        """

        if verbose:
            start_time = time.time()

        ds_concatenated = xr.Dataset()

        lon = ds[self.dim_names["longitude"]]
        if end == "lower":
            lon_minus360 = lon - 360
            lon_concatenated = xr.concat(
                [lon_minus360, lon], dim=self.dim_names["longitude"]
            )

        elif end == "upper":
            lon_plus360 = lon + 360
            lon_concatenated = xr.concat(
                [lon, lon_plus360], dim=self.dim_names["longitude"]
            )

        elif end == "both":
            lon_minus360 = lon - 360
            lon_plus360 = lon + 360
            lon_concatenated = xr.concat(
                [lon_minus360, lon, lon_plus360], dim=self.dim_names["longitude"]
            )

        for var in ds.data_vars:
            if self.dim_names["longitude"] in ds[var].dims:
                field = ds[var]

                if end == "both":
                    field_concatenated = xr.concat(
                        [field, field, field], dim=self.dim_names["longitude"]
                    )
                else:
                    field_concatenated = xr.concat(
                        [field, field], dim=self.dim_names["longitude"]
                    )

                if self.use_dask:
                    field_concatenated = field_concatenated.chunk(
                        {self.dim_names["longitude"]: -1}
                    )
                field_concatenated[self.dim_names["longitude"]] = lon_concatenated
                ds_concatenated[var] = field_concatenated
            else:
                ds_concatenated[var] = ds[var]

        ds_concatenated[self.dim_names["longitude"]] = lon_concatenated

        if verbose:
            logging.info(
                f"Concatenating the data along the longitude dimension: {time.time() - start_time:.3f} seconds"
            )

        return ds_concatenated

    def post_process(self):
        """Placeholder method to be overridden by subclasses for dataset post-
        processing.

        Returns
        -------
        None
            This method does not return any value. Subclasses are expected to modify the dataset in-place.
        """
        pass

    def choose_subdomain(
        self,
        target_coords,
        buffer_points=20,
        return_copy=False,
        return_coords_only=False,
        verbose=False,
    ):
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
        return_subdomain : bool, optional
            If True, returns the subset of the original dataset representing the chosen
            subdomain. If False, assigns the subset to `self.ds`. Defaults to False.
        return_coords_only : bool, optional
            If True, returns a new xarray.Dataset containing only the latitude and longitude
            of the subdomain. Defaults to False.
        verbose : bool, optional
            If True, print message if dataset is concatenated along longitude dimension.
            Defaults to False.

        Returns
        -------
        xr.Dataset or None
            Returns the subset of the original dataset as an xarray Dataset if
            `return_subdomain` is True, including an extended area covering additional
            grid points beyond the specified ranges. Returns None if `return_subdomain`
            is False, as the subset is assigned to `self.ds`.

        Raises
        ------
        ValueError
            If the selected latitude or longitude range does not intersect with the dataset.
        """

        lat_min = target_coords["lat"].min().values
        lat_max = target_coords["lat"].max().values
        lon_min = target_coords["lon"].min().values
        lon_max = target_coords["lon"].max().values

        margin = self.resolution * buffer_points

        # Select the subdomain in latitude direction (so that we have to concatenate fewer latitudes below if concatenation is necessary)
        subdomain = self.ds.sel(
            **{
                self.dim_names["latitude"]: slice(lat_min - margin, lat_max + margin),
            }
        )
        lon = subdomain[self.dim_names["longitude"]]

        if self.is_global:
            # Concatenate only if necessary
            if lon_max + margin > lon.max():
                # See if shifting by +360 degrees helps
                if (lon_min - margin > (lon + 360).min()) and (
                    lon_max + margin < (lon + 360).max()
                ):
                    subdomain[self.dim_names["longitude"]] = lon + 360
                    lon = subdomain[self.dim_names["longitude"]]
                else:
                    subdomain = self.concatenate_longitudes(
                        subdomain, end="upper", verbose=verbose
                    )
                    lon = subdomain[self.dim_names["longitude"]]
            if lon_min - margin < lon.min():
                # See if shifting by -360 degrees helps
                if (lon_min - margin > (lon - 360).min()) and (
                    lon_max + margin < (lon - 360).max()
                ):
                    subdomain[self.dim_names["longitude"]] = lon - 360
                    lon = subdomain[self.dim_names["longitude"]]
                else:
                    subdomain = self.concatenate_longitudes(
                        subdomain, end="lower", verbose=verbose
                    )
                    lon = subdomain[self.dim_names["longitude"]]

        else:
            # Adjust longitude range if needed to match the expected range
            if not target_coords["straddle"]:
                if lon.min() < -180:
                    if lon_max + margin > 0:
                        lon_min -= 360
                        lon_max -= 360
                elif lon.min() < 0:
                    if lon_max + margin > 180:
                        lon_min -= 360
                        lon_max -= 360

            if target_coords["straddle"]:
                if lon.max() > 360:
                    if lon_min - margin < 180:
                        lon_min += 360
                        lon_max += 360
                elif lon.max() > 180:
                    if lon_min - margin < 0:
                        lon_min += 360
                        lon_max += 360
        # Select the subdomain in longitude direction
        subdomain = subdomain.sel(
            **{
                self.dim_names["longitude"]: slice(lon_min - margin, lon_max + margin),
            }
        )

        # Check if the selected subdomain has zero dimensions in latitude or longitude
        if subdomain[self.dim_names["latitude"]].size == 0:
            raise ValueError("Selected latitude range does not intersect with dataset.")

        if subdomain[self.dim_names["longitude"]].size == 0:
            raise ValueError(
                "Selected longitude range does not intersect with dataset."
            )

        # Adjust longitudes to expected range if needed
        lon = subdomain[self.dim_names["longitude"]]
        if target_coords["straddle"]:
            subdomain[self.dim_names["longitude"]] = xr.where(lon > 180, lon - 360, lon)
        else:
            subdomain[self.dim_names["longitude"]] = xr.where(lon < 0, lon + 360, lon)

        if return_coords_only:
            # Create and return a dataset with only latitudes and longitudes
            coords_ds = subdomain[
                [self.dim_names["latitude"], self.dim_names["longitude"]]
            ]
            return coords_ds

        if return_copy:
            return Dataset.from_ds(self, subdomain)
        else:
            object.__setattr__(self, "ds", subdomain)

    def apply_lateral_fill(self):
        """Apply lateral fill to variables using the dataset's mask and grid dimensions.

        This method fills masked values in `self.ds` using `LateralFill` based on
        the horizontal grid dimensions. A separate mask (`mask_vel`) is used for
        velocity variables (e.g., `u`, `v`) if available in the dataset.

        Notes
        -----
        Looping over `self.ds.data_vars` instead of `self.var_names` ensures that each
        dataset variable is filled only once, even if multiple entries in `self.var_names`
        point to the same variable in the dataset.
        """
        lateral_fill = LateralFill(
            self.ds["mask"],
            [self.dim_names["latitude"], self.dim_names["longitude"]],
        )

        separate_fill_for_velocities = False
        if "mask_vel" in self.ds.data_vars:
            lateral_fill_vel = LateralFill(
                self.ds["mask_vel"],
                [self.dim_names["latitude"], self.dim_names["longitude"]],
            )
            separate_fill_for_velocities = True

        for var_name in self.ds.data_vars:
            if var_name.startswith("mask"):
                # Skip variables that are mask types
                continue
            elif (
                separate_fill_for_velocities
                and "u" in self.var_names
                and "v" in self.var_names
                and var_name in [self.var_names["u"], self.var_names["v"]]
            ):
                # Apply lateral fill with velocity mask for velocity variables if present
                self.ds[var_name] = lateral_fill_vel.apply(self.ds[var_name])
            else:
                # Apply standard lateral fill for other variables
                self.ds[var_name] = lateral_fill.apply(self.ds[var_name])

    def extrapolate_deepest_to_bottom(self):
        """Extrapolate deepest non-NaN values to fill bottom NaNs along the depth
        dimension.

        For each variable with a depth dimension, fills missing values at the bottom by
        propagating the deepest available data downward.
        """

        if "depth" in self.dim_names:
            for var_name in self.ds.data_vars:
                if self.dim_names["depth"] in self.ds[var_name].dims:
                    self.ds[var_name] = one_dim_fill(
                        self.ds[var_name], self.dim_names["depth"], direction="forward"
                    )

    @classmethod
    def from_ds(cls, original_dataset: "Dataset", ds: xr.Dataset) -> "Dataset":
        """Substitute the internal dataset of a Dataset object with a new xarray
        Dataset.

        This method creates a new Dataset instance, bypassing the usual `__init__`
        and `__post_init__` processes. It allows for the direct assignment of the
        provided xarray Dataset (`ds`) to the new instance's `ds` attribute. All
        other attributes from the original dataset instance are copied to the new one.

        Parameters
        ----------
        original_dataset : Dataset
            The original Dataset instance from which attributes will be copied.
        ds : xarray.Dataset
            The new xarray Dataset to assign to the `ds` attribute of the new instance.

        Returns
        -------
        Dataset
            A new Dataset instance with the `ds` attribute set to the provided dataset
            and other attributes copied from the original instance.
        """
        # Create a new Dataset instance without calling __init__ or __post_init__
        dataset = cls.__new__(cls)

        # Directly set the provided dataset as the 'ds' attribute
        object.__setattr__(dataset, "ds", ds)

        # Copy all other attributes from the original data instance
        for attr in vars(original_dataset):
            if attr != "ds":
                object.__setattr__(dataset, attr, getattr(original_dataset, attr))

        return dataset


@dataclass(frozen=True, kw_only=True)
class TPXODataset(Dataset):
    """Represents tidal data on the original grid from the TPXO dataset.

    Parameters
    ----------
    filename : str
        The path to the TPXO dataset file.
    grid_filename : str
        The path to the TPXO grid file.
    location : str
        "h", "u", "v"
    var_names : Dict[str, str]
        Dictionary of variable names required in the dataset.
    dim_names : Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset. Defaults to:
        {"longitude": "ny", "latitude": "nx", "ntides": "nc"}.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the TPXO tidal model data, loaded from the specified file.
    """

    filename: str
    grid_filename: str
    location: str
    var_names: Dict[str, str]
    dim_names: Dict[str, str] = field(
        default_factory=lambda: {"longitude": "ny", "latitude": "nx", "ntides": "nc"}
    )
    ds: xr.Dataset = field(init=False, repr=False)

    def clean_up(self, ds: xr.Dataset) -> xr.Dataset:
        """Clean up and standardize the dimensions and coordinates of the dataset for
        further processing.

        This method performs several key operations to clean and standardize the dataset:
        - Assigns new coordinate variables for 'longitude', and 'latitude' based on existing dataset variables.
        - Adds the `mask` variable to the dataset, which indicates valid data points based on grid conditions.
        - Renames the dimensions:
            - The 'nx' dimension is renamed to 'longitude'.
            - The 'ny' dimension is renamed to 'latitude'.
            - The tidal dimension is renamed to 'ntides' for standardization.
        - Updates the `dim_names` attribute of the object to reflect the new dimension names: 'longitude', 'latitude', and 'ntides'.
        - Checks if the longitude and latitude values from the dataset match those from the grid. If a mismatch is found, raises a `ValueError` indicating the mismatch details.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset to be cleaned and standardized.

        Returns
        -------
        ds : xr.Dataset
            A cleaned and standardized `xarray.Dataset` with updated coordinates and dimensions.

        Raises
        ------
        ValueError
            If the longitude or latitude values from the dataset do not match those from the grid. The error message will include the mismatched values.
        """

        ds_grid = _load_data(self.grid_filename, self.dim_names, self.use_dask)

        if self.location == "h":
            mask = ds_grid["mz"].isnull()
            lon_name = "lon_z"
            lat_name = "lat_z"
        elif self.location == "u":
            mask = ds_grid["mu"]
            lon_name = "lon_u"
            lat_name = "lat_u"
        elif self.location == "v":
            mask = ds_grid["mv"]
            lon_name = "lon_v"
            lat_name = "lat_v"

        lon_from_grid = ds_grid[lon_name]
        lat_from_grid = ds_grid[lat_name]
        lon = ds[lon_name]
        lat = ds[lat_name]

        # Check if the longitude and latitude match between the grid and the dataset
        if lon.shape != lon_from_grid.shape:
            raise ValueError(
                f"Mismatch in longitude array sizes. Dataset: {lon.shape}, Grid: {lon_from_grid.shape}"
            )

        if not np.allclose(lon.values, lon_from_grid.values):
            raise ValueError(
                f"Longitude values from the dataset do not closely match the grid. Dataset: {lon.values}, Grid: {lon_from_grid.values}"
            )

        # Check if latitude sizes match before comparing values
        if lat.shape != lat_from_grid.shape:
            raise ValueError(
                f"Mismatch in latitude array sizes. Dataset: {lat.shape}, Grid: {lat_from_grid.shape}"
            )

        if not np.allclose(lat.values, lat_from_grid.values):
            raise ValueError(
                f"Latitude values from the dataset do not closely match the grid. Dataset: {lat.values}, Grid: {lat_from_grid.values}"
            )

        ds["mask"] = mask
        ds = ds.rename({"con": "nc"})
        ds = ds.assign_coords(
            {
                "nc": ("nc", [c.decode("utf-8").strip() for c in ds["nc"].values]),
                "nx": lon.isel(
                    ny=0
                ),  # lon is constant along ny, i.e., is only a function of nx
                "ny": lat.isel(
                    nx=0
                ),  # lat is constant along nx, i.e., is only a function of ny
            }
        )
        ds = ds.rename(
            {"nx": "longitude", "ny": "latitude", self.dim_names["ntides"]: "ntides"}
        )

        object.__setattr__(
            self,
            "dim_names",
            {
                "latitude": "latitude",
                "longitude": "longitude",
                "ntides": "ntides",
            },
        )

        return ds

    def select_constituents(self, ntides: int, omega: Dict[str, float]):
        """Selects the first `ntides` tidal constituents based on the provided omega
        values.

        This method filters the dataset to retain only the tidal constituents that match
        the first `ntides` from the provided `omega` dictionary. It ensures that the dataset
        contains the expected constituents before proceeding with the selection. If the dataset
        does not contain the required constituents, an error is raised.

        Parameters
        ----------
        ntides : int
            The number of tidal constituents to retain from the omega values. The method selects
            the first `ntides` constituents based on the provided omega dictionary and maps their
            corresponding values to the dataset.

        omega : dict
            A dictionary where keys are tidal constituent names and values are their associated omega
            values. The first `ntides` keys from this dictionary will be used to filter the tidal
            constituents in the dataset.

        Raises
        ------
        ValueError
            If the dataset does not contain all required tidal constituents from the first `ntides`
            selected from the `omega` dictionary, a `ValueError` is raised, indicating the mismatch
            between the expected and present constituents in the dataset.
        """

        # Expected constituents based on the first 'ntides' from the omega dictionary
        expected_constituents = list(omega.keys())[:ntides]

        # Extract the current tidal constituents from the dataset
        dataset_constituents = self.ds["ntides"].values.tolist()

        # Check if the dataset contains the expected constituents
        if not all(c in dataset_constituents for c in expected_constituents):
            raise ValueError(
                f"The dataset contains tidal constituents {dataset_constituents} that do not match the first {ntides} required constituents "
                f"from the TPXO dataset: {expected_constituents}. "
                "Ensure the dataset includes the required constituents or reduce the 'ntides' parameter."
            )

        # Select only the expected constituents from the dataset
        filtered_constituents = [
            c for c in dataset_constituents if c in expected_constituents
        ]

        # Update the dataset with the filtered constituents
        ds = self.ds.sel(ntides=filtered_constituents)
        object.__setattr__(self, "ds", ds)


@dataclass(frozen=True, kw_only=True)
class GLORYSDataset(Dataset):
    """Represents GLORYS data on original grid.

    Parameters
    ----------
    filename : str
        The path to the data files. Can contain wildcards.
    start_time : Optional[datetime], optional
        The start time for selecting relevant data. If not provided, the data is not filtered by start time.
    end_time : Optional[datetime], optional
        The end time for selecting relevant data. If not provided, only data at the start_time is selected if start_time is provided,
        or no filtering is applied if start_time is not provided.
    var_names: Dict[str, str], optional
        Dictionary of variable names that are required in the dataset.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
    climatology : bool
        Indicates whether the dataset is climatological. Defaults to False.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the GLORYS data on its original grid.
    """

    var_names: Dict[str, str] = field(
        default_factory=lambda: {
            "temp": "thetao",
            "salt": "so",
            "u": "uo",
            "v": "vo",
            "zeta": "zos",
        }
    )

    dim_names: Dict[str, str] = field(
        default_factory=lambda: {
            "longitude": "longitude",
            "latitude": "latitude",
            "depth": "depth",
            "time": "time",
        }
    )

    climatology: Optional[bool] = False

    def post_process(self):
        """Apply a mask to the dataset based on the 'zeta' variable, with 0 where 'zeta'
        is NaN.

        This method creates a mask based on the
        first time step (time=0) of 'zeta'. The mask has 1 for valid data and 0 where 'zeta' is NaN. This mask is applied
        to all data variables, replacing values with NaN where 'zeta' is NaN at time=0.
        The mask itself is stored in the dataset under the variable 'mask'.

        Returns
        -------
        None
            The dataset is modified in-place by applying the mask to each variable.
        """

        mask = xr.where(
            self.ds[self.var_names["zeta"]].isel({self.dim_names["time"]: 0}).isnull(),
            0,
            1,
        )
        mask_vel = xr.where(
            self.ds[self.var_names["u"]]
            .isel({self.dim_names["time"]: 0, self.dim_names["depth"]: 0})
            .isnull(),
            0,
            1,
        )

        self.ds["mask"] = mask
        self.ds["mask_vel"] = mask_vel


@dataclass(frozen=True, kw_only=True)
class CESMDataset(Dataset):
    """Represents CESM data on original grid.

    Parameters
    ----------
    filename : str
        The path to the data files. Can contain wildcards.
    start_time : Optional[datetime], optional
        The start time for selecting relevant data. If not provided, the data is not filtered by start time.
    end_time : Optional[datetime], optional
        The end time for selecting relevant data. If not provided, only data at the start_time is selected if start_time is provided,
        or no filtering is applied if start_time is not provided.
    var_names: Dict[str, str], optional
        Dictionary of variable names that are required in the dataset.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
    climatology : bool
        Indicates whether the dataset is climatological. Defaults to False.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the CESM data on its original grid.
    """

    # overwrite clean_up method from parent class
    def clean_up(self, ds: xr.Dataset) -> xr.Dataset:
        """Ensure the dataset's time dimension is correctly defined and standardized.

        This method verifies that the time dimension exists in the dataset and assigns it appropriately. If the "time" dimension is missing, the method attempts to assign an existing "time" or "month" dimension. If neither exists, it expands the dataset to include a "time" dimension with a size of one.

        Returns
        -------
        ds : xr.Dataset
            The xarray Dataset with the correct time dimension assigned or added.
        """

        if "time" not in self.dim_names:
            if "time" in ds.dims:
                self.dim_names["time"] = "time"
            else:
                if "month" in ds.dims:
                    self.dim_names["time"] = "month"
                else:
                    ds = ds.expand_dims({"time": 1})
                    self.dim_names["time"] = "time"

        return ds

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

        if self.climatology and time_dim == "month":
            ds = assign_dates_to_climatology(ds, time_dim)
            # rename dimension
            ds = ds.swap_dims({time_dim: "time"})
            if time_dim in ds.variables:
                ds = ds.drop_vars(time_dim)
            # Update dimension names
            updated_dim_names = self.dim_names.copy()
            updated_dim_names["time"] = "time"
            object.__setattr__(self, "dim_names", updated_dim_names)

        return ds


@dataclass(frozen=True, kw_only=True)
class CESMBGCDataset(CESMDataset):
    """Represents CESM BGC data on original grid.

    Parameters
    ----------
    filename : str
        The path to the data files. Can contain wildcards.
    start_time : Optional[datetime], optional
        The start time for selecting relevant data. If not provided, the data is not filtered by start time.
    end_time : Optional[datetime], optional
        The end time for selecting relevant data. If not provided, only data at the start_time is selected if start_time is provided,
        or no filtering is applied if start_time is not provided.
    var_names: Dict[str, str], optional
        Dictionary of variable names that are required in the dataset.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
    climatology : bool
        Indicates whether the dataset is climatological. Defaults to False.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the CESM data on its original grid.
    """

    var_names: Dict[str, str] = field(
        default_factory=lambda: {
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
        }
    )

    dim_names: Dict[str, str] = field(
        default_factory=lambda: {
            "longitude": "lon",
            "latitude": "lat",
            "depth": "z_t",
        }
    )

    climatology: Optional[bool] = False

    def post_process(self):
        """
        Processes and converts CESM data values as follows:
        - Convert depth values from cm to m.
        - Apply a mask to the dataset based on the 'P04' variable at the surface.
        """
        if self.dim_names["depth"] == "z_t":
            # Fill variables that only have data in upper 150m with NaNs below
            if (
                "z_t_150m" in self.ds.dims
                and np.equal(
                    self.ds.z_t[: len(self.ds.z_t_150m)].values, self.ds.z_t_150m.values
                ).all()
            ):
                for var in self.var_names:
                    if "z_t_150m" in self.ds[var].dims:
                        self.ds[var] = self.ds[var].rename({"z_t_150m": "z_t"})
                        if self.use_dask:
                            self.ds[var] = self.ds[var].chunk({"z_t": -1})
            # Convert depth from cm to m
            ds = self.ds.assign_coords({"depth": self.ds["z_t"] / 100})
            ds["depth"].attrs["long_name"] = "Depth"
            ds["depth"].attrs["units"] = "m"
            ds = ds.swap_dims({"z_t": "depth"})
            if "z_t" in ds.variables:
                ds = ds.drop_vars("z_t")
            if "z_t_150m" in ds.variables:
                ds = ds.drop_vars("z_t_150m")
            # update dataset
            object.__setattr__(self, "ds", ds)

            # Update dim_names with "depth": "depth" key-value pair
            updated_dim_names = self.dim_names.copy()
            updated_dim_names["depth"] = "depth"
            object.__setattr__(self, "dim_names", updated_dim_names)

        mask = xr.where(
            self.ds[self.var_names["PO4"]]
            .isel({self.dim_names["time"]: 0, self.dim_names["depth"]: 0})
            .isnull(),
            0,
            1,
        )

        self.ds["mask"] = mask


@dataclass(frozen=True, kw_only=True)
class CESMBGCSurfaceForcingDataset(CESMDataset):
    """Represents CESM BGC surface forcing data on original grid.

    Parameters
    ----------
    filename : str
        The path to the data files. Can contain wildcards.
    start_time : Optional[datetime], optional
        The start time for selecting relevant data. If not provided, the data is not filtered by start time.
    end_time : Optional[datetime], optional
        The end time for selecting relevant data. If not provided, only data at the start_time is selected if start_time is provided,
        or no filtering is applied if start_time is not provided.
    var_names: Dict[str, str], optional
        Dictionary of variable names that are required in the dataset.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
    climatology : bool
        Indicates whether the dataset is climatological. Defaults to False.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the CESM data on its original grid.
    """

    var_names: Dict[str, str] = field(
        default_factory=lambda: {
            "pco2_air": "pCO2SURF",
            "pco2_air_alt": "pCO2SURF",
            "iron": "IRON_FLUX",
            "dust": "dust_FLUX_IN",
            "nox": "NOx_FLUX",
            "nhy": "NHy_FLUX",
        }
    )

    dim_names: Dict[str, str] = field(
        default_factory=lambda: {
            "longitude": "lon",
            "latitude": "lat",
        }
    )

    climatology: Optional[bool] = False

    def post_process(self):
        """Perform post-processing on the dataset to remove specific variables.

        This method checks if the variable "z_t" exists in the dataset. If it does,
        the variable is removed from the dataset. The modified dataset is then
        reassigned to the `ds` attribute of the object.
        """

        if "z_t" in self.ds.variables:
            ds = self.ds.drop_vars("z_t")
            object.__setattr__(self, "ds", ds)

        mask = xr.where(
            self.ds[self.var_names["pco2_air"]]
            .isel({self.dim_names["time"]: 0})
            .isnull(),
            0,
            1,
        )

        self.ds["mask"] = mask


@dataclass(frozen=True, kw_only=True)
class ERA5Dataset(Dataset):
    """Represents ERA5 data on original grid.

    Parameters
    ----------
    filename : str
        The path to the data files. Can contain wildcards.
    start_time : Optional[datetime], optional
        The start time for selecting relevant data. If not provided, the data is not filtered by start time.
    end_time : Optional[datetime], optional
        The end time for selecting relevant data. If not provided, only data at the start_time is selected if start_time is provided,
        or no filtering is applied if start_time is not provided.
    var_names: Dict[str, str], optional
        Dictionary of variable names that are required in the dataset.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
    climatology : bool
        Indicates whether the dataset is climatological. Defaults to False.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the ERA5 data on its original grid.
    """

    var_names: Dict[str, str] = field(
        default_factory=lambda: {
            "uwnd": "u10",
            "vwnd": "v10",
            "swrad": "ssr",
            "lwrad": "strd",
            "Tair": "t2m",
            "d2m": "d2m",
            "rain": "tp",
            "mask": "sst",
        }
    )

    dim_names: Dict[str, str] = field(
        default_factory=lambda: {
            "longitude": "longitude",
            "latitude": "latitude",
            "time": "time",
        }
    )

    climatology: Optional[bool] = False

    def post_process(self):
        """
        Processes and converts ERA5 data values as follows:
        - Convert radiation values from J/m^2 to W/m^2.
        - Convert rainfall from meters to cm/day.
        - Convert temperature from Kelvin to Celsius.
        - Compute relative humidity if not present, convert to absolute humidity.
        - Use SST to create mask.
        """
        # Translate radiation to fluxes. ERA5 stores values integrated over 1 hour.
        # Convert radiation from J/m^2 to W/m^2
        self.ds[self.var_names["swrad"]] /= 3600
        self.ds[self.var_names["lwrad"]] /= 3600
        self.ds[self.var_names["swrad"]].attrs["units"] = "W/m^2"
        self.ds[self.var_names["lwrad"]].attrs["units"] = "W/m^2"
        # Convert rainfall from m to cm/day
        self.ds[self.var_names["rain"]] *= 100 * 24

        # Convert temperature from Kelvin to Celsius
        self.ds[self.var_names["Tair"]] -= 273.15
        self.ds[self.var_names["d2m"]] -= 273.15
        self.ds[self.var_names["Tair"]].attrs["units"] = "degrees C"
        self.ds[self.var_names["d2m"]].attrs["units"] = "degrees C"

        # Compute relative humidity if not present
        if "qair" not in self.ds.data_vars:
            qair = np.exp(
                (17.625 * self.ds[self.var_names["d2m"]])
                / (243.04 + self.ds[self.var_names["d2m"]])
            ) / np.exp(
                (17.625 * self.ds[self.var_names["Tair"]])
                / (243.04 + self.ds[self.var_names["Tair"]])
            )
            # Convert relative to absolute humidity
            patm = 1010.0
            cff = (
                (1.0007 + 3.46e-6 * patm)
                * 6.1121
                * np.exp(
                    17.502
                    * self.ds[self.var_names["Tair"]]
                    / (240.97 + self.ds[self.var_names["Tair"]])
                )
            )
            cff = cff * qair

            ds = self.ds
            ds["qair"] = 0.62197 * (cff / (patm - 0.378 * cff))
            ds["qair"].attrs["long_name"] = "Absolute humidity at 2m"
            ds["qair"].attrs["units"] = "kg/kg"
            ds = ds.drop_vars([self.var_names["d2m"]])
            object.__setattr__(self, "ds", ds)

            # Update var_names dictionary
            var_names = {**self.var_names, "qair": "qair"}
            var_names.pop("d2m")
            object.__setattr__(self, "var_names", var_names)

        if "mask" in self.var_names.keys():
            ds = self.ds
            mask = xr.where(self.ds[self.var_names["mask"]].isel(time=0).isnull(), 0, 1)
            ds["mask"] = mask
            ds = ds.drop_vars([self.var_names["mask"]])
            object.__setattr__(self, "ds", ds)

            # Remove mask from var_names dictionary
            var_names = self.var_names
            var_names.pop("mask")
            object.__setattr__(self, "var_names", var_names)


@dataclass(frozen=True, kw_only=True)
class ERA5Correction(Dataset):
    """Global dataset to correct ERA5 radiation. The dataset contains multiplicative
    correction factors for the ERA5 shortwave radiation, obtained by comparing the
    COREv2 climatology to the ERA5 climatology.

    Parameters
    ----------
    filename : str, optional
        The path to the correction files. Defaults to download_correction_data('SSR_correction.nc').
    var_names: Dict[str, str], optional
        Dictionary of variable names that are required in the dataset.
        Defaults to {"swr_corr": "ssr_corr"}.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
        Defaults to {"longitude": "longitude", "latitude": "latitude", "time": "time"}.
    climatology : bool, optional
        Indicates if the correction data is a climatology. Defaults to True.

    Attributes
    ----------
    ds : xr.Dataset
        The loaded xarray Dataset containing the correction data.
    """

    filename: str = field(
        default_factory=lambda: download_correction_data("SSR_correction.nc")
    )
    var_names: Dict[str, str] = field(
        default_factory=lambda: {
            "swr_corr": "ssr_corr",  # multiplicative correction factor for ERA5 shortwave radiation
        }
    )
    dim_names: Dict[str, str] = field(
        default_factory=lambda: {
            "longitude": "longitude",
            "latitude": "latitude",
            "time": "time",
        }
    )
    climatology: Optional[bool] = True

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        if not self.climatology:
            raise NotImplementedError(
                "Correction data must be a climatology. Set climatology to True."
            )

        super().__post_init__()

    def choose_subdomain(self, target_coords, straddle: bool):
        """Converts longitude values in the dataset if necessary and selects a subdomain
        based on the specified coordinates.

        This method converts longitude values between different ranges if required and then extracts a subset of the
        dataset according to the given coordinates. It updates the dataset in place to reflect the selected subdomain.

        Parameters
        ----------
        target_coords : dict
            A dictionary specifying the target coordinates for selecting the subdomain. Keys should correspond to the
            dimension names of the dataset (e.g., latitude and longitude), and values should be the desired ranges or
            specific coordinate values.
        straddle : bool
            If True, assumes that target longitudes are in the range [-180, 180]. If False, assumes longitudes are in the
            range [0, 360]. This parameter determines how longitude values are converted if necessary.

        Raises
        ------
        ValueError
            If the specified subdomain does not fully contain the specified latitude or longitude values. This can occur
            if the dataset does not cover the full range of provided coordinates.

        Notes
        -----
        - The dataset (`self.ds`) is updated in place to reflect the chosen subdomain.
        """

        # Select the subdomain in latitude direction (so that we have to concatenate fewer latitudes below if concatenation is performed)
        subdomain = self.ds.sel({self.dim_names["latitude"]: target_coords["lat"]})

        if self.is_global:
            # Always concatenate because computational overhead should be managable for 1/4 degree ERA5 resolution
            subdomain = self.concatenate_longitudes(
                subdomain, end="both", verbose=False
            )

        # Select the subdomain in longitude direction
        subdomain = subdomain.sel({self.dim_names["longitude"]: target_coords["lon"]})

        # Check if the selected subdomain contains the specified latitude and longitude values
        if not subdomain[self.dim_names["latitude"]].equals(target_coords["lat"]):
            raise ValueError(
                "The correction dataset does not contain all specified latitude values."
            )
        if not subdomain[self.dim_names["longitude"]].equals(target_coords["lon"]):
            raise ValueError(
                "The correction dataset does not contain all specified longitude values."
            )
        object.__setattr__(self, "ds", subdomain)


@dataclass(frozen=True, kw_only=True)
class ETOPO5Dataset(Dataset):
    """Represents topography data on the original grid from the ETOPO5 dataset.

    Parameters
    ----------
    filename : str, optional
        The path to the ETOPO5 dataset file. If not provided, the dataset will be downloaded
        automatically via the `pooch` library.
    var_names : Dict[str, str], optional
        Dictionary of variable names required in the dataset. Defaults to:
        {
            "topo": "topo",
        }
    dim_names : Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset. Defaults to:
        {"longitude": "lon", "latitude": "lat"}.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the ETOPO5 data, loaded from the specified file.
    """

    filename: str = field(default_factory=lambda: download_topo("etopo5.nc"))
    var_names: Dict[str, str] = field(
        default_factory=lambda: {
            "topo": "topo",
        }
    )
    dim_names: Dict[str, str] = field(
        default_factory=lambda: {"longitude": "lon", "latitude": "lat"}
    )
    ds: xr.Dataset = field(init=False, repr=False)

    def clean_up(self, ds: xr.Dataset) -> xr.Dataset:
        """Assign lat and lon as coordinates.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset.

        Returns
        -------
        ds : xr.Dataset
            A cleaned `xarray.Dataset` with updated coordinates.
        """
        ds = ds.assign_coords(
            {
                "lon": ds["topo_lon"],
                "lat": ds["topo_lat"],
            }
        )
        return ds


@dataclass(frozen=True, kw_only=True)
class SRTM15Dataset(Dataset):
    """Represents topography data on the original grid from the SRTM15 dataset.

    Parameters
    ----------
    filename : str
        The path to the SRTM15 dataset file.
    var_names : Dict[str, str], optional
        Dictionary of variable names required in the dataset. Defaults to:
        {
            "topo": "z",
        }
    dim_names : Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset. Defaults to:
        {"longitude": "lon", "latitude": "lat"}.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the SRTM15 data, loaded from the specified file.
    """

    filename: str
    var_names: Dict[str, str] = field(
        default_factory=lambda: {
            "topo": "z",
        }
    )
    dim_names: Dict[str, str] = field(
        default_factory=lambda: {"longitude": "lon", "latitude": "lat"}
    )
    ds: xr.Dataset = field(init=False, repr=False)


# river datasets
@dataclass(frozen=True, kw_only=True)
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

    filename: Union[str, Path, List[Union[str, Path]]]
    start_time: datetime
    end_time: datetime
    dim_names: Dict[str, str]
    var_names: Dict[str, str]
    opt_var_names: Optional[Dict[str, str]] = field(default_factory=dict)
    climatology: Optional[bool] = False
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

        # Select relevant times
        ds = self.add_time_info(ds)
        object.__setattr__(self, "ds", ds)

    def load_data(self) -> xr.Dataset:
        """Load dataset from the specified file.

        Returns
        -------
        ds : xr.Dataset
            The loaded xarray Dataset containing the forcing data.
        """
        ds = _load_data(
            self.filename, self.dim_names, use_dask=False, decode_times=False
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
                name = decode_string(byte_array)
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
        """Check if the dataset contains the specified variables and dimensions.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset to check.

        Raises
        ------
        ValueError
            If the dataset does not contain the specified variables or dimensions.
        """

        _check_dataset(ds, self.dim_names, self.var_names, self.opt_var_names)

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

        ds = _select_relevant_times(ds, time_dim, self.start_time, self.end_time, False)

        return ds

    def compute_climatology(self):
        logging.info("Compute climatology for river forcing.")

        time_dim = self.dim_names["time"]

        flux = self.ds[self.var_names["flux"]].groupby(f"{time_dim}.month").mean()
        self.ds[self.var_names["flux"]] = flux

        ds = assign_dates_to_climatology(self.ds, "month")
        ds = ds.swap_dims({"month": "time"})
        object.__setattr__(self, "ds", ds)

        updated_dim_names = {**self.dim_names}
        updated_dim_names["time"] = "time"
        object.__setattr__(self, "dim_names", updated_dim_names)

        object.__setattr__(self, "climatology", True)

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

        if "vol" in self.opt_var_names:
            volume_values = ds[self.opt_var_names["vol"]].values
            if isinstance(volume_values, np.ndarray):
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

    def extract_relevant_rivers(self, target_coords, dx):
        """Extracts a subset of the dataset based on the proximity of river mouths to
        target coordinates.

        This method calculates the distance between each river mouth and the provided target coordinates
        (latitude and longitude) using the `gc_dist` function. It then filters the dataset to include only those
        river stations whose minimum distance from the target is less than a specified threshold distance (`dx`).

        Parameters
        ----------
        target_coords : dict
            A dictionary containing the target coordinates for the comparison. It should include:
            - "lon" (float): The target longitude in degrees.
            - "lat" (float): The target latitude in degrees.
            - "straddle" (bool): A flag indicating whether to adjust the longitudes for stations that cross the
              International Date Line. If `True`, longitudes greater than 180 degrees are adjusted by subtracting 360,
              otherwise, negative longitudes are adjusted by adding 360.

        dx : float
            The maximum distance threshold (in meters) for including a river station. Only river mouths that are
            within `dx` meters from the target coordinates will be included in the returned dataset.

        Returns
        -------
        indices : dict
            A dictionary containing the indices of the rivers that are within the threshold distance from
            the target coordinates. The dictionary keys are:
            - "station" : numpy.ndarray
                The indices of the rivers that satisfy the distance threshold.
            - "eta_rho" : numpy.ndarray
                The indices of the `eta_rho` dimension corresponding to the selected stations.
            - "xi_rho" : numpy.ndarray
                The indices of the `xi_rho` dimension corresponding to the selected stations.
        """

        # Retrieve longitude and latitude of river mouths
        river_lon = self.ds[self.var_names["longitude"]]
        river_lat = self.ds[self.var_names["latitude"]]

        # Adjust longitude based on whether it crosses the International Date Line (straddle case)
        if target_coords["straddle"]:
            river_lon = xr.where(river_lon > 180, river_lon - 360, river_lon)
        else:
            river_lon = xr.where(river_lon < 0, river_lon + 360, river_lon)

        # Calculate the distance between the target coordinates and each river mouth
        dist = gc_dist(target_coords["lon"], target_coords["lat"], river_lon, river_lat)
        dist_min = dist.min(dim=["eta_rho", "xi_rho"])
        # Filter the dataset to include only stations within the distance threshold
        if (dist_min < dx).any():
            ds = self.ds.where(dist_min < dx, drop=True)
            ds = self.sort_by_river_volume(ds)
            dist = dist.where(dist_min < dx, drop=True).transpose(
                self.dim_names["station"], "eta_rho", "xi_rho"
            )
            dist_min = dist_min.where(dist_min < dx, drop=True)

            # Find the indices of the closest grid cell to the river mouth
            indices = np.where(dist == dist_min)
            names = (
                self.ds[self.var_names["name"]]
                .isel({self.dim_names["station"]: indices[0]})
                .values
            )
            # Return the indices in a dictionary format
            indices = {
                "station": indices[0],
                "eta_rho": indices[1],
                "xi_rho": indices[2],
                "name": names,
            }
        else:
            ds = xr.Dataset()
            indices = {
                "station": [],
                "eta_rho": [],
                "xi_rho": [],
                "name": [],
            }

        object.__setattr__(self, "ds", ds)

        return indices


@dataclass(frozen=True, kw_only=True)
class DaiRiverDataset(RiverDataset):
    """Represents river data from the Dai river dataset.

    Parameters
    ----------
    filename : Union[str, Path, List[Union[str, Path]]], optional
        The path to the Dai River dataset file. If not provided, the dataset will be downloaded
        automatically via the `pooch` library.
    start_time : datetime
        The start time for selecting relevant data.
    end_time : datetime
        The end time for selecting relevant data.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
    var_names: Dict[str, str], optional
        Dictionary of variable names that are required in the dataset.
    opt_var_names: Dict[str, str], optional
        Dictionary of variable names that are optional in the dataset.
    climatology : bool
        Indicates whether the dataset is climatological. Defaults to False.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the forcing data on its original grid.
    """

    filename: Union[str, Path, List[Union[str, Path]]] = field(
        default_factory=lambda: download_river_data("dai_trenberth_may2019.nc")
    )
    start_time: datetime
    end_time: datetime
    dim_names: Dict[str, str] = field(
        default_factory=lambda: {
            "station": "station",
            "time": "time",
        }
    )
    var_names: Dict[str, str] = field(
        default_factory=lambda: {
            "latitude": "lat_mou",
            "longitude": "lon_mou",
            "flux": "FLOW",
            "ratio": "ratio_m2s",
            "name": "riv_name",
        }
    )
    opt_var_names: Dict[str, str] = field(
        default_factory=lambda: {
            "vol": "vol_stn",
        }
    )
    climatology: Optional[bool] = False
    ds: xr.Dataset = field(init=False, repr=False)

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


@dataclass
class TPXOManager:
    """Manages multiple TPXODataset instances and selects and processes tidal
    constituents from the TPXO dataset.

    This class handles data for various tidal constituents ordered according to the TPXO9v2a standard.
    For later products, the order of the tidal constituents may change after the 10th constituent.
    These later products will be reordered to the TPXO9v2a order before selecting the first `ntides` constituents.
    Typically, only the first 10 constituents are used in applications, as the amplitudes of constituents beyond the 10th are minimal.

    Parameters
    ----------
    filenames : dict
        A dictionary containing paths to the TPXO dataset files. Expected keys are:
        - "h": Path to the elevation file.
        - "sal": Path to the self-attraction and loading (SAL) file.
        - "u": Path to the u-velocity component file.
        - "grid": Path to the grid file.

    ntides : int
        The number of tidal constituents to select from the dataset for processing.

    reference_date : datetime, optional
        The reference date for the TPXO data. Defaults to January 1, 1992.
        This date serves as the baseline for tidal time series calculations.

    allan_factor : float, optional
        The Allan factor used in tidal model computation. Default is 2.0.

    use_dask: bool
        Indicates whether to use dask for chunking. If True, data is loaded with dask; if False, data is loaded eagerly. Defaults to False.
    """

    filenames: dict
    ntides: int
    reference_date: datetime = datetime(1992, 1, 1)
    allan_factor: float = 2.0
    use_dask: Optional[bool] = False

    def __post_init__(self):

        # Initialize the data_dict with TPXODataset instances
        data_dict = {
            "h": TPXODataset(
                filename=self.filenames["h"],
                grid_filename=self.filenames["grid"],
                location="h",
                var_names={"ssh_Re": "hRe", "ssh_Im": "hIm"},
                use_dask=self.use_dask,
            ),
            "sal": TPXODataset(
                filename=self.filenames["sal"],
                grid_filename=self.filenames["grid"],
                location="h",
                var_names={"sal_Re": "hRe", "sal_Im": "hIm"},
                use_dask=self.use_dask,
            ),
            "u": TPXODataset(
                filename=self.filenames["u"],
                grid_filename=self.filenames["grid"],
                location="u",
                var_names={"u_Re": "URe", "u_Im": "UIm"},
                use_dask=self.use_dask,
            ),
            "v": TPXODataset(
                filename=self.filenames["u"],
                grid_filename=self.filenames["grid"],
                location="v",
                var_names={"v_Re": "VRe", "v_Im": "VIm"},
                use_dask=self.use_dask,
            ),
        }

        omega = self.get_omega()

        for data in data_dict.values():
            data.select_constituents(self.ntides, omega)

        data_dict["omega"] = xr.DataArray(
            data=list(omega.values())[: self.ntides], dims="ntides"
        )

        object.__setattr__(self, "datasets", data_dict)

    def get_omega(self):
        """Retrieve angular frequencies (omega) for tidal constituents from the TPXO9.v2
        atlas.

        This method returns the angular frequencies (in radians per second) for 15 tidal constituents,
        sourced from the TPXO tidal model and defined in the OTPSnc `constit.h` file, see https://www.tpxo.net/otps.
        These values are essential for tidal modeling and analysis.

        Returns
        -------
        dict
            A dictionary where the keys are tidal constituent labels (str) and the values
            are their respective angular frequencies (float, in radians per second).
        """
        omega = {
            "m2": 1.405189e-04,  # Principal lunar semidiurnal
            "s2": 1.454441e-04,  # Principal solar semidiurnal
            "n2": 1.378797e-04,  # Larger lunar elliptic semidiurnal
            "k2": 1.458423e-04,  # Lunisolar semidiurnal
            "k1": 7.292117e-05,  # Lunar diurnal
            "o1": 6.759774e-05,  # Lunar diurnal
            "p1": 7.252295e-05,  # Solar diurnal
            "q1": 6.495854e-05,  # Larger lunar elliptic diurnal
            "mm": 0.026392e-04,  # Lunar monthly
            "mf": 0.053234e-04,  # Lunar fortnightly
            "m4": 2.810377e-04,  # Shallow water overtide of M2
            "mn4": 2.783984e-04,  # Shallow water quarter diurnal
            "ms4": 2.859630e-04,  # Shallow water quarter diurnal
            "2n2": 1.352405e-04,  # Shallow water semidiurnal
            "s1": 7.2722e-05,  # Solar diurnal
        }
        return omega

    def compute_equilibrium_tide(self, lon, lat):
        """Compute equilibrium tide for given longitudes and latitudes.

        Parameters
        ----------
        lon : xr.DataArray
            Longitudes in degrees.
        lat : xr.DataArray
            Latitudes in degrees.

        Returns
        -------
        tpc : xr.DataArray
            Equilibrium tide complex amplitude.

        Notes
        -----
        This method calculates the equilibrium tide complex amplitude for specified
        longitudes and latitudes, considering 15 tidal constituents and their corresponding
        amplitudes and elasticity factors. The order of the tidal constituents corresponds
        to the order in `self.get_omega()`, which must remain consistent for future use.

        The tidal constituents are categorized as follows:
        - **2**: Semidiurnal
        - **1**: Diurnal
        - **0**: Long-period

        The amplitudes and elasticity factors are sourced from the `constit.h` file in the OTPSnc package.
        """

        # Amplitudes for 15 tidal constituents (from variable amp_d in constit.h of OTPSnc package)
        A = xr.DataArray(
            data=np.array(
                [
                    0.242334,  # M2
                    0.112743,  # S2
                    0.046397,  # N2
                    0.030684,  # K2
                    0.141565,  # K1
                    0.100661,  # O1
                    0.046848,  # P1
                    0.019273,  # Q1
                    0.022191,  # Mm
                    0.042041,  # Mf
                    0.0,  # M4
                    0.0,  # Mn4
                    0.0,  # Ms4
                    0.006141,  # 2n2
                    0.000764,  # S1
                ]
            ),
            dims="ntides",
        )

        # Elasticity factors for 15 tidal constituents (from variable alpha_d in constit.h of OTPSnc package)
        B = xr.DataArray(
            data=np.array(
                [
                    0.693,  # M2
                    0.693,  # S2
                    0.693,  # N2
                    0.693,  # K2
                    0.736,  # K1
                    0.695,  # O1
                    0.706,  # P1
                    0.695,  # Q1
                    0.693,  # Mm
                    0.693,  # Mf
                    0.693,  # M4
                    0.693,  # Mn4
                    0.693,  # Ms4
                    0.693,  # 2n2
                    0.693,  # S1
                ]
            ),
            dims="ntides",
        )

        # Tidal type (from variable ispec_d in constit.h of OTPSnc package)
        # types: 2 = semidiurnal, 1 = diurnal, 0 = long-term
        ityp = xr.DataArray(
            data=np.array(
                [
                    2,  # M2
                    2,  # S2
                    2,  # N2
                    2,  # K2
                    1,  # K1
                    1,  # O1
                    1,  # P1
                    1,  # Q1
                    0,  # Mm
                    0,  # Mf
                    0,  # M4
                    0,  # Mn4
                    0,  # Ms4
                    2,  # 2n2
                    1,  # S1
                ]
            ),
            dims="ntides",
        )

        d2r = np.pi / 180
        coslat2 = np.cos(d2r * lat) ** 2
        sin2lat = np.sin(2 * d2r * lat)

        p_amp = (
            xr.where(ityp == 2, 1, 0) * A * B * coslat2  # semidiurnal
            + xr.where(ityp == 1, 1, 0) * A * B * sin2lat  # diurnal
            + xr.where(ityp == 0, 1, 0) * A * B * (0.5 - 1.5 * coslat2)  # long-term
        )
        p_pha = (
            xr.where(ityp == 2, 1, 0) * (-2 * lon * d2r)  # semidiurnal
            + xr.where(ityp == 1, 1, 0) * (-lon * d2r)  # diurnal
            + xr.where(ityp == 0, 1, 0) * xr.zeros_like(lon)  # long-term
        )

        tpc = p_amp * np.exp(-1j * p_pha)

        return tpc

    def egbert_correction(self, date):
        """Correct phases and amplitudes for real-time runs using parts of the post-
        processing code from Egbert's & Erofeeva's (OSU) TPXO model.

        Parameters
        ----------
        date : datetime.datetime
            The date and time for which corrections are to be applied.

        Returns
        -------
        pf : xr.DataArray
            Amplitude scaling factor for each of the 15 tidal constituents.
        pu : xr.DataArray
            Phase correction [radians] for each of the 15 tidal constituents.
        aa : xr.DataArray
            Astronomical arguments [radians] associated with the corrections.

        Notes
        -----
        The order of the tidal constituents corresponds
        to the order in `self.get_omega()`, which must remain consistent for future use.

        References
        ----------
        - Egbert, G.D., and S.Y. Erofeeva. "Efficient inverse modeling of barotropic ocean
          tides." Journal of Atmospheric and Oceanic Technology 19, no. 2 (2002): 183-204.
        """

        year = date.year
        month = date.month
        day = date.day
        hour = date.hour
        minute = date.minute
        second = date.second

        rad = np.pi / 180.0
        deg = 180.0 / np.pi
        mjd = modified_julian_days(year, month, day)
        tstart = mjd + hour / 24 + minute / (60 * 24) + second / (60 * 60 * 24)

        # Determine nodal corrections pu & pf : these expressions are valid for period 1990-2010 (Cartwright 1990).
        # Reset time origin for astronomical arguments to 4th of May 1860:
        timetemp = tstart - 51544.4993

        # mean longitude of lunar perigee
        P = 83.3535 + 0.11140353 * timetemp
        P = np.mod(P, 360.0)
        if P < 0:
            P = +360
        P *= rad

        # mean longitude of ascending lunar node
        N = 125.0445 - 0.05295377 * timetemp
        N = np.mod(N, 360.0)
        if N < 0:
            N = +360
        N *= rad

        sinn = np.sin(N)
        cosn = np.cos(N)
        sin2n = np.sin(2 * N)
        cos2n = np.cos(2 * N)
        sin3n = np.sin(3 * N)

        pftmp = np.sqrt(
            (1 - 0.03731 * cosn + 0.00052 * cos2n) ** 2
            + (0.03731 * sinn - 0.00052 * sin2n) ** 2
        )

        pf = np.zeros(15)
        pf[0] = pftmp  # M2
        pf[1] = 1.0  # S2
        pf[2] = pftmp  # N2
        pf[3] = np.sqrt(
            (1 + 0.2852 * cosn + 0.0324 * cos2n) ** 2
            + (0.3108 * sinn + 0.0324 * sin2n) ** 2
        )  # K2
        pf[4] = np.sqrt(
            (1 + 0.1158 * cosn - 0.0029 * cos2n) ** 2
            + (0.1554 * sinn - 0.0029 * sin2n) ** 2
        )  # K1
        pf[5] = np.sqrt(
            (1 + 0.189 * cosn - 0.0058 * cos2n) ** 2
            + (0.189 * sinn - 0.0058 * sin2n) ** 2
        )  # O1
        pf[6] = 1.0  # P1
        pf[7] = np.sqrt((1 + 0.188 * cosn) ** 2 + (0.188 * sinn) ** 2)  # Q1
        pf[8] = 1.0 - 0.130 * cosn  # Mm
        pf[9] = 1.043 + 0.414 * cosn  # Mf
        pf[10] = pftmp**2  # M4
        pf[11] = pftmp**2  # Mn4
        pf[12] = pftmp**2  # Ms4
        pf[13] = pftmp  # 2n2
        pf[14] = 1.0  # S1
        pf = xr.DataArray(pf, dims="ntides")

        putmp = (
            np.arctan(
                (-0.03731 * sinn + 0.00052 * sin2n)
                / (1.0 - 0.03731 * cosn + 0.00052 * cos2n)
            )
            * deg
        )

        pu = np.zeros(15)
        pu[0] = putmp  # M2
        pu[1] = 0.0  # S2
        pu[2] = putmp  # N2
        pu[3] = (
            np.arctan(
                -(0.3108 * sinn + 0.0324 * sin2n)
                / (1.0 + 0.2852 * cosn + 0.0324 * cos2n)
            )
            * deg
        )  # K2
        pu[4] = (
            np.arctan(
                (-0.1554 * sinn + 0.0029 * sin2n)
                / (1.0 + 0.1158 * cosn - 0.0029 * cos2n)
            )
            * deg
        )  # K1
        pu[5] = 10.8 * sinn - 1.3 * sin2n + 0.2 * sin3n  # O1
        pu[6] = 0.0  # P1
        pu[7] = np.arctan(0.189 * sinn / (1.0 + 0.189 * cosn)) * deg  # Q1
        pu[8] = 0.0  # Mm
        pu[9] = -23.7 * sinn + 2.7 * sin2n - 0.4 * sin3n  # Mf
        pu[10] = putmp * 2.0  # M4
        pu[11] = putmp * 2.0  # Mn4
        pu[12] = putmp  # Ms4
        pu[13] = putmp  # 2n2
        pu[14] = 0.0  # S1
        pu = xr.DataArray(pu, dims="ntides")
        # convert from degrees to radians
        pu = pu * rad

        aa = xr.DataArray(
            data=np.array(
                [
                    1.731557546,  # M2
                    0.0,  # S2
                    6.050721243,  # N2
                    3.487600001,  # K2
                    0.173003674,  # K1
                    1.558553872,  # O1
                    6.110181633,  # P1
                    5.877717569,  # Q1
                    1.964021610,  # Mm
                    1.756042456,  # Mf
                    3.463115091,  # M4
                    1.499093481,  # Mn4
                    1.731557546,  # Ms4
                    4.086699633,  # 2n2
                    0.0,  # S1
                ]
            ),
            dims="ntides",
        )

        return pf, pu, aa

    def correct_tides(self, model_reference_date):
        """Apply tidal corrections to the dataset. This method corrects the dataset for
        equilibrium tides, self-attraction and loading (SAL) effects, and adjusts phases
        and amplitudes of tidal elevations and transports using Egbert's correction.

        Parameters
        ----------
        model_reference_date : datetime
            The reference date for the ROMS simulation.

        Returns
        -------
        None
            The dataset is modified in-place with corrected real and imaginary components for ssh, u, v, and the
            potential field ('pot_Re', 'pot_Im').
        """

        datasets = self.datasets
        omega = self.datasets["omega"].isel(ntides=slice(None, self.ntides))

        # Get equilibrium tides
        lon = datasets["sal"].ds[datasets["sal"].dim_names["longitude"]]
        lat = datasets["sal"].ds[datasets["sal"].dim_names["latitude"]]
        tpc = self.compute_equilibrium_tide(lon, lat)
        tpc = tpc.isel(ntides=slice(None, self.ntides))

        # Correct for SAL
        tsc = self.allan_factor * (
            datasets["sal"].ds[datasets["sal"].var_names["sal_Re"]]
            + 1j * datasets["sal"].ds[datasets["sal"].var_names["sal_Im"]]
        )
        tpc = tpc - tsc

        # Elevations and transports
        thc = (
            datasets["h"].ds[datasets["h"].var_names["ssh_Re"]]
            + 1j * datasets["h"].ds[datasets["h"].var_names["ssh_Im"]]
        )
        tuc = (
            datasets["u"].ds[datasets["u"].var_names["u_Re"]]
            + 1j * datasets["u"].ds[datasets["u"].var_names["u_Im"]]
        )
        tvc = (
            datasets["v"].ds[datasets["v"].var_names["v_Re"]]
            + 1j * datasets["v"].ds[datasets["v"].var_names["v_Im"]]
        )

        # Apply correction for phases and amplitudes
        pf, pu, aa = self.egbert_correction(model_reference_date)
        pf = pf.isel(ntides=slice(None, self.ntides))
        pu = pu.isel(ntides=slice(None, self.ntides))
        aa = aa.isel(ntides=slice(None, self.ntides))

        dt = (model_reference_date - self.reference_date).days * 3600 * 24

        thc = pf * thc * np.exp(1j * (omega * dt + pu + aa))
        tuc = pf * tuc * np.exp(1j * (omega * dt + pu + aa))
        tvc = pf * tvc * np.exp(1j * (omega * dt + pu + aa))
        tpc = pf * tpc * np.exp(1j * (omega * dt + pu + aa))

        datasets["h"].ds[datasets["h"].var_names["ssh_Re"]] = thc.real
        datasets["h"].ds[datasets["h"].var_names["ssh_Im"]] = thc.imag
        datasets["u"].ds[datasets["u"].var_names["u_Re"]] = tuc.real
        datasets["u"].ds[datasets["u"].var_names["u_Im"]] = tuc.imag
        datasets["v"].ds[datasets["v"].var_names["v_Re"]] = tvc.real
        datasets["v"].ds[datasets["v"].var_names["v_Im"]] = tvc.imag
        datasets["sal"].ds["pot_Re"] = tpc.real
        datasets["sal"].ds["pot_Im"] = tpc.imag

        object.__setattr__(self, "datasets", datasets)

        # Update var_names dictionary
        var_names = {
            **datasets["sal"].var_names,
            "pot_Re": "pot_Re",
            "pot_Im": "pot_Im",
        }
        var_names.pop("sal_Re", None)  # Remove "sal_Re" if it exists
        var_names.pop("sal_Im", None)  # Remove "sal_Im" if it exists
        object.__setattr__(self.datasets["sal"], "var_names", var_names)


# shared functions


def _check_dataset(
    ds: xr.Dataset,
    dim_names: Dict[str, str],
    var_names: Dict[str, str],
    opt_var_names: Optional[Dict[str, str]] = None,
) -> None:
    """Check if the dataset contains the specified variables and dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset to check.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
    var_names: Dict[str, str]
        Dictionary of variable names that are required in the dataset.
    opt_var_names : Optional[Dict[str, str]], optional
        Dictionary of optional variable names.
        These variables are not strictly required, and the function will not raise an error if they are missing.
        Default is None, meaning no optional variables are considered.


    Raises
    ------
    ValueError
        If the dataset does not contain the specified variables or dimensions.
    """
    missing_dims = [dim for dim in dim_names.values() if dim not in ds.dims]
    if missing_dims:
        raise ValueError(
            f"Dataset does not contain all required dimensions. The following dimensions are missing: {missing_dims}"
        )

    missing_vars = [var for var in var_names.values() if var not in ds.data_vars]
    if missing_vars:
        raise ValueError(
            f"Dataset does not contain all required variables. The following variables are missing: {missing_vars}"
        )

    if opt_var_names:
        missing_optional_vars = [
            var for var in opt_var_names.values() if var not in ds.data_vars
        ]
        if missing_optional_vars:
            logging.warning(
                f"Optional variables missing (but not critical): {missing_optional_vars}"
            )


def _select_relevant_times(
    ds, time_dim, start_time=None, end_time=None, climatology=False
) -> xr.Dataset:
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
    time_dim: str
        Name of time dimension.
    start_time : Optional[datetime], optional
        The start time for selecting relevant data. If not provided, the data is not filtered by start time.
    end_time : Optional[datetime], optional
        The end time for selecting relevant data. If not provided, only data at the start_time is selected if start_time is provided,
        or no filtering is applied if start_time is not provided.
    climatology : bool
        Indicates whether the dataset is climatological. Defaults to False.

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

    if time_dim in ds.variables:
        if climatology:
            if len(ds[time_dim]) != 12:
                raise ValueError(
                    f"The dataset contains {len(ds[time_dim])} time steps, but the climatology flag is set to True, which requires exactly 12 time steps."
                )
            if not end_time:
                # Interpolate from climatology for initial conditions
                ds["time"] = ds["time"].dt.days
                ds = interpolate_from_climatology(ds, time_dim, start_time)
        else:
            time_type = get_time_type(ds[time_dim])
            if time_type == "int":
                raise ValueError(
                    "The dataset contains integer time values, which are only supported when the climatology flag is set to True. However, your climatology flag is set to False."
                )
            if time_type == "cftime":
                ds = ds.assign_coords(
                    {time_dim: convert_cftime_to_datetime(ds[time_dim])}
                )
            if end_time:
                end_time = end_time

                # Identify records before or at start_time
                before_start = ds[time_dim] <= np.datetime64(start_time)
                if before_start.any():
                    closest_before_start = (
                        ds[time_dim].where(before_start, drop=True).max()
                    )
                else:
                    logging.warning("No records found at or before the start_time.")
                    closest_before_start = ds[time_dim].min()

                # Identify records after or at end_time
                after_end = ds[time_dim] >= np.datetime64(end_time)
                if after_end.any():
                    closest_after_end = ds[time_dim].where(after_end, drop=True).min()
                else:
                    logging.warning("No records found at or after the end_time.")
                    closest_after_end = ds[time_dim].max()

                # Select records within the time range and add the closest before/after
                within_range = (ds[time_dim] > np.datetime64(start_time)) & (
                    ds[time_dim] < np.datetime64(end_time)
                )
                selected_times = ds[time_dim].where(
                    within_range
                    | (ds[time_dim] == closest_before_start)
                    | (ds[time_dim] == closest_after_end),
                    drop=True,
                )
                ds = ds.sel({time_dim: selected_times})
            else:
                # Look in time range [start_time, start_time + 24h]
                end_time = start_time + timedelta(days=1)
                times = (np.datetime64(start_time) <= ds[time_dim]) & (
                    ds[time_dim] < np.datetime64(end_time)
                )
                if np.all(~times):
                    raise ValueError(
                        f"The dataset does not contain any time entries between the specified start_time: {start_time} "
                        f"and {start_time + timedelta(hours=24)}. "
                        "Please ensure the dataset includes time entries for that range."
                    )

                ds = ds.where(times, drop=True)
                if ds.sizes[time_dim] > 1:
                    # Pick the time closest to start_time
                    ds = ds.isel({time_dim: 0})
                logging.info(
                    f"Selected time entry closest to the specified start_time ({start_time}) within the range [{start_time}, {start_time + timedelta(hours=24)}]: {ds[time_dim].values}"
                )
    else:
        logging.warning(
            "Dataset does not contain any time information. Please check if the time dimension "
            "is correctly named or if the dataset includes time data."
        )

    return ds


def decode_string(byte_array):

    # Decode each byte and handle errors with 'ignore'
    decoded_string = "".join(
        [
            x.decode("utf-8", errors="ignore")  # Ignore invalid byte sequences
            for x in byte_array.values
            if isinstance(x, bytes) and x != b" " and x is not np.nan
        ]
    )

    return decoded_string


def convert_to_float64(ds: xr.Dataset) -> xr.Dataset:
    """Convert all data variables in an xarray.Dataset to float64.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Dataset with all data variables converted to float64.
    """
    return ds.astype({var: "float64" for var in ds.data_vars})


def modified_julian_days(year, month, day, hour=0):
    """Calculate the Modified Julian Day (MJD) for a given date and time.

    The Modified Julian Day (MJD) is a modified Julian day count starting from
    November 17, 1858 AD. It is commonly used in astronomy and geodesy.

    Parameters
    ----------
    year : int
        The year.
    month : int
        The month (1-12).
    day : int
        The day of the month.
    hour : float, optional
        The hour of the day as a fractional number (0 to 23.999...). Default is 0.

    Returns
    -------
    mjd : float
        The Modified Julian Day (MJD) corresponding to the input date and time.

    Notes
    -----
    The algorithm assumes that the input date (year, month, day) is within the
    Gregorian calendar, i.e., after October 15, 1582. Negative MJD values are
    allowed for dates before November 17, 1858.

    References
    ----------
    - Wikipedia article on Julian Day: https://en.wikipedia.org/wiki/Julian_day
    - Wikipedia article on Modified Julian Day: https://en.wikipedia.org/wiki/Modified_Julian_day

    Examples
    --------
    >>> modified_julian_days(2024, 5, 20, 12)
    58814.0
    >>> modified_julian_days(1858, 11, 17)
    0.0
    >>> modified_julian_days(1582, 10, 4)
    -141428.5
    """

    if month < 3:
        year -= 1
        month += 12

    A = year // 100
    B = A // 4
    C = 2 - A + B
    E = int(365.25 * (year + 4716))
    F = int(30.6001 * (month + 1))
    jd = C + day + hour / 24 + E + F - 1524.5
    mjd = jd - 2400000.5

    return mjd
