import time
import re
import xarray as xr
from dataclasses import dataclass, field
import glob
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Optional, Union, List
from pathlib import Path
import warnings
from roms_tools.setup.utils import (
    assign_dates_to_climatology,
    interpolate_from_climatology,
    get_time_type,
    convert_cftime_to_datetime,
    one_dim_fill,
)
from roms_tools.setup.download import download_correction_data, download_topo
from roms_tools.setup.fill import LateralFill


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
    var_names: Dict[str, str]
        Dictionary of variable names that are required in the dataset.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
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
    var_names: Dict[str, str]
    dim_names: Dict[str, str] = field(
        default_factory=lambda: {
            "longitude": "longitude",
            "latitude": "latitude",
            "time": "time",
        }
    )
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

        # Precompile the regex for matching wildcard characters
        wildcard_regex = re.compile(r"[\*\?\[\]]")

        # Convert Path objects to strings
        if isinstance(self.filename, (str, Path)):
            filename_str = str(self.filename)
        elif isinstance(self.filename, list):
            filename_str = [str(f) for f in self.filename]
        else:
            raise ValueError(
                "filename must be a string, Path, or a list of strings/Paths."
            )

        # Handle the case when filename is a string
        contains_wildcard = False
        if isinstance(filename_str, str):
            contains_wildcard = bool(wildcard_regex.search(filename_str))
            if contains_wildcard:
                matching_files = glob.glob(filename_str)
                if not matching_files:
                    raise FileNotFoundError(
                        f"No files found matching the pattern '{filename_str}'."
                    )
            else:
                matching_files = [filename_str]

        # Handle the case when filename is a list
        elif isinstance(filename_str, list):
            contains_wildcard = any(wildcard_regex.search(f) for f in filename_str)
            if contains_wildcard:
                matching_files = []
                for f in filename_str:
                    files = glob.glob(f)
                    if not files:
                        raise FileNotFoundError(
                            f"No files found matching the pattern '{f}'."
                        )
                    matching_files.extend(files)
            else:
                matching_files = filename_str

        # Check if time dimension is available when multiple files are provided
        if isinstance(filename_str, list) and "time" not in self.dim_names:
            raise ValueError(
                "A list of files is provided, but time dimension is not available. "
                "A time dimension must be available to concatenate the files."
            )

        # Determine the kwargs for combining datasets
        if contains_wildcard or len(matching_files) == 1:
            # If there is a wildcard or just one file, use by_coords
            kwargs = {"combine": "by_coords"}
        else:
            # Otherwise, use nested combine based on time
            kwargs = {"combine": "nested", "concat_dim": self.dim_names["time"]}

        # Base kwargs used for dataset combination
        combine_kwargs = {
            "coords": "minimal",
            "compat": "override",
            "combine_attrs": "override",
        }

        if self.use_dask:

            chunks = {
                self.dim_names["latitude"]: -1,
                self.dim_names["longitude"]: -1,
            }
            if "depth" in self.dim_names:
                chunks[self.dim_names["depth"]] = -1
            if "time" in self.dim_names:
                chunks[self.dim_names["time"]] = 1

            ds = xr.open_mfdataset(
                matching_files,
                chunks=chunks,
                **combine_kwargs,
                **kwargs,
            )
        else:
            ds_list = []
            for file in matching_files:
                ds = xr.open_dataset(file, chunks=None)
                ds_list.append(ds)

            if kwargs["combine"] == "by_coords":
                ds = xr.combine_by_coords(ds_list, **combine_kwargs)
            elif kwargs["combine"] == "nested":
                ds = xr.combine_nested(
                    ds_list, concat_dim=kwargs["concat_dim"], **combine_kwargs
                )

        if "time" in self.dim_names and self.dim_names["time"] not in ds.dims:
            ds = ds.expand_dims(self.dim_names["time"])

        return ds

    def clean_up(self, ds: xr.Dataset) -> xr.Dataset:
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
            The xarray Dataset cleaned up (as implemented by child classes).
        """
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
        missing_vars = [
            var for var in self.var_names.values() if var not in ds.data_vars
        ]
        if missing_vars:
            raise ValueError(
                f"Dataset does not contain all required variables. The following variables are missing: {missing_vars}"
            )

        missing_dims = [dim for dim in self.dim_names.values() if dim not in ds.dims]
        if missing_dims:
            raise ValueError(
                f"Dataset does not contain all required dimensions. The following dimensions are missing: {missing_vars}"
            )

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
            if var not in self.var_names.values():
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
        if time_dim in ds.variables:
            if self.climatology:
                if len(ds[time_dim]) != 12:
                    raise ValueError(
                        f"The dataset contains {len(ds[time_dim])} time steps, but the climatology flag is set to True, which requires exactly 12 time steps."
                    )
                if not self.end_time:
                    # Interpolate from climatology for initial conditions
                    ds = interpolate_from_climatology(
                        ds, self.dim_names["time"], self.start_time
                    )
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
                if self.end_time:
                    end_time = self.end_time

                    # Identify records before or at start_time
                    before_start = ds[time_dim] <= np.datetime64(self.start_time)
                    if before_start.any():
                        closest_before_start = (
                            ds[time_dim].where(before_start, drop=True).max()
                        )
                    else:
                        warnings.warn("No records found at or before the start_time.")
                        closest_before_start = ds[time_dim].min()

                    # Identify records after or at end_time
                    after_end = ds[time_dim] >= np.datetime64(end_time)
                    if after_end.any():
                        closest_after_end = (
                            ds[time_dim].where(after_end, drop=True).min()
                        )
                    else:
                        warnings.warn("No records found at or after the end_time.")
                        closest_after_end = ds[time_dim].max()

                    # Select records within the time range and add the closest before/after
                    within_range = (ds[time_dim] > np.datetime64(self.start_time)) & (
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
                    # Look in time range [self.start_time, self.start_time + 24h]
                    end_time = self.start_time + timedelta(days=1)
                    times = (np.datetime64(self.start_time) <= ds[time_dim]) & (
                        ds[time_dim] < np.datetime64(end_time)
                    )
                    if np.all(~times):
                        raise ValueError(
                            f"The dataset does not contain any time entries between the specified start_time: {self.start_time} "
                            f"and {self.start_time + timedelta(hours=24)}. "
                            "Please ensure the dataset includes time entries for that range."
                        )

                    ds = ds.where(times, drop=True)
                    if ds.sizes[time_dim] > 1:
                        # Pick the time closest to self.start_time
                        ds = ds.isel({time_dim: 0})
                    print(
                        f"Selected time entry closest to the specified start_time ({self.start_time}) within the range [{self.start_time}, {self.start_time + timedelta(hours=24)}]: {ds[time_dim].values}"
                    )
        else:
            warnings.warn(
                "Dataset does not contain any time information. Please check if the time dimension "
                "is correctly named or if the dataset includes time data."
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
            print(
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
        self, target_coords, buffer_points=20, return_copy=False, verbose=False
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
    var_names : Dict[str, str], optional
        Dictionary of variable names required in the dataset. Defaults to:
        {
            "h_Re": "h_Re",
            "h_Im": "h_Im",
            "sal_Re": "sal_Re",
            "sal_Im": "sal_Im",
            "u_Re": "u_Re",
            "u_Im": "u_Im",
            "v_Re": "v_Re",
            "v_Im": "v_Im",
            "depth": "depth"
        }
    dim_names : Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset. Defaults to:
        {"longitude": "ny", "latitude": "nx", "ntides": "nc"}.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the TPXO tidal model data, loaded from the specified file.
    reference_date : datetime
        The reference date for the TPXO data. Default is datetime(1992, 1, 1).
    """

    filename: str
    var_names: Dict[str, str] = field(
        default_factory=lambda: {
            "ssh_Re": "h_Re",
            "ssh_Im": "h_Im",
            "sal_Re": "sal_Re",
            "sal_Im": "sal_Im",
            "u_Re": "u_Re",
            "u_Im": "u_Im",
            "v_Re": "v_Re",
            "v_Im": "v_Im",
            "depth": "depth",
        }
    )
    dim_names: Dict[str, str] = field(
        default_factory=lambda: {"longitude": "ny", "latitude": "nx", "ntides": "nc"}
    )
    ds: xr.Dataset = field(init=False, repr=False)
    reference_date: datetime = datetime(1992, 1, 1)

    def clean_up(self, ds: xr.Dataset) -> xr.Dataset:
        """Clean up and standardize the dimensions and coordinates of the dataset for
        further processing.

        This method performs the following operations:
        - Assigns new coordinate variables for 'omega', 'longitude', and 'latitude' based on existing dataset variables.
          - 'omega' is retained as it is.
          - 'longitude' is derived from 'lon_r', assuming it is constant along the 'ny' dimension.
          - 'latitude' is derived from 'lat_r', assuming it is constant along the 'nx' dimension.
        - Renames the dimensions 'nx' and 'ny' to 'longitude' and 'latitude', respectively, for consistency.
        - Renames the tidal dimension to 'ntides' for standardization.
        - Updates the `dim_names` attribute of the object to reflect the new dimension names: 'longitude', 'latitude', and 'ntides'.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset to be cleaned and standardized. It should contain the coordinates 'omega', 'lon_r', 'lat_r', and the tidal dimension.

        Returns
        -------
        ds : xr.Dataset
            A cleaned and standardized `xarray.Dataset` with updated coordinates and dimensions.
        """
        ds = ds.assign_coords(
            {
                "omega": ds["omega"],
                "nx": ds["lon_r"].isel(
                    ny=0
                ),  # lon_r is constant along ny, i.e., is only a function of nx
                "ny": ds["lat_r"].isel(
                    nx=0
                ),  # lat_r is constant along nx, i.e., is only a function of ny
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

    def check_number_constituents(self, ntides: int):
        """Checks if the number of constituents in the dataset is at least `ntides`.

        Parameters
        ----------
        ntides : int
            The required number of tidal constituents.

        Raises
        ------
        ValueError
            If the number of constituents in the dataset is less than `ntides`.
        """
        if len(self.ds[self.dim_names["ntides"]]) < ntides:
            raise ValueError(
                f"The dataset contains fewer than {ntides} tidal constituents."
            )

    def post_process(self):
        """Apply a depth-based mask to the dataset, ensuring only positive depths are
        retained.

        This method checks if the 'depth' variable is present in the dataset. If found, a mask is created where
        depths greater than 0 are considered valid (mask value of 1). This mask is applied to all data variables
        in the dataset, replacing values at invalid depths (depth ≤ 0) with NaN. The mask itself is also stored
        in the dataset under the variable 'mask'.

        Returns
        -------
        None
            The dataset is modified in-place by applying the mask to each variable.
        """

        if "depth" in self.var_names.keys():
            ds = self.ds
            mask = xr.where(self.ds["depth"] > 0, 1, 0)
            ds["mask"] = mask
            ds = ds.drop_vars(["depth"])

            object.__setattr__(self, "ds", ds)

        # Remove "depth" from var_names
        updated_var_names = {**self.var_names}  # Create a copy of the dictionary
        updated_var_names.pop("depth", None)  # Remove "depth" if it exists
        object.__setattr__(self, "var_names", updated_var_names)


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

    def choose_subdomain(self, coords, straddle: bool):
        """Converts longitude values in the dataset if necessary and selects a subdomain
        based on the specified coordinates.

        This method converts longitude values between different ranges if required and then extracts a subset of the
        dataset according to the given coordinates. It updates the dataset in place to reflect the selected subdomain.

        Parameters
        ----------
        coords : dict
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

        lon = self.ds[self.dim_names["longitude"]]

        if not self.is_global:
            if lon.min().values < 0 and not straddle:
                # Convert from [-180, 180] to [0, 360]
                self.ds[self.dim_names["longitude"]] = xr.where(lon < 0, lon + 360, lon)

            if lon.max().values > 180 and straddle:
                # Convert from [0, 360] to [-180, 180]
                self.ds[self.dim_names["longitude"]] = xr.where(
                    lon > 180, lon - 360, lon
                )

        # Select the subdomain based on the specified latitude and longitude ranges
        subdomain = self.ds.sel(**coords)

        # Check if the selected subdomain contains the specified latitude and longitude values
        if not subdomain[self.dim_names["latitude"]].equals(
            coords[self.dim_names["latitude"]]
        ):
            raise ValueError(
                "The correction dataset does not contain all specified latitude values."
            )
        if not subdomain[self.dim_names["longitude"]].equals(
            coords[self.dim_names["longitude"]]
        ):
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
