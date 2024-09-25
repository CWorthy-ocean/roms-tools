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
    is_cftime_datetime,
    convert_cftime_to_datetime,
)
from roms_tools.setup.download import download_correction_data


@dataclass(frozen=True, kw_only=True)
class Dataset:
    """
    Represents forcing data on original grid.

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
    >>> dataset.load_data()
    >>> print(dataset.ds)
    <xarray.Dataset>
    Dimensions:  ...
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
    use_dask: Optional[bool] = True

    is_global: bool = field(init=False, repr=False)
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        """
        Post-initialization processing:
        1. Loads the dataset from the specified filename.
        2. Applies time filtering based on start_time and end_time if provided.
        3. Selects relevant fields as specified by var_names.
        4. Ensures latitude values are in ascending order.
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
        ds = self.ensure_latitude_ascending(ds)

        # Check whether the data covers the entire globe
        object.__setattr__(self, "is_global", self.check_if_global(ds))

        # If dataset is global concatenate three copies of field along longitude dimension
        if self.is_global:
            ds = self.concatenate_longitudes(ds)

        object.__setattr__(self, "ds", ds)

    def load_data(self) -> xr.Dataset:
        """
        Load dataset from the specified file.

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

    def check_dataset(self, ds: xr.Dataset) -> None:
        """
        Check if the dataset contains the specified variables and dimensions.

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
        """
        Selects and returns a subset of the dataset containing only the variables specified in `self.var_names`.

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

    import xarray as xr

    def add_time_info(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Dummy method to be overridden by child classes to add time information to the dataset.

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
        """
        Select a subset of the dataset based on the specified time range.

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
                if not self.end_time:
                    # Interpolate from climatology for initial conditions
                    ds = interpolate_from_climatology(
                        ds, self.dim_names["time"], self.start_time
                    )
            else:
                if len(ds[time_dim]) == 12:
                    warnings.warn(
                        "The dataset contains exactly 12 time steps. This may indicate that it is "
                        "climatological data. Please verify if climatology is appropriate for your "
                        "analysis and set the climatology flag to True."
                    )
                if is_cftime_datetime(ds[time_dim]):
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

    def ensure_latitude_ascending(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Ensure that the latitude dimension is in ascending order.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset to check.

        Returns
        -------
        ds : xr.Dataset
            The xarray Dataset with latitude in ascending order.
        """
        # Make sure that latitude is ascending
        lat_diff = np.diff(ds[self.dim_names["latitude"]])
        if np.all(lat_diff < 0):
            ds = ds.isel(**{self.dim_names["latitude"]: slice(None, None, -1)})

        return ds

    def check_if_global(self, ds) -> bool:
        """
        Checks if the dataset covers the entire globe in the longitude dimension.

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

    def concatenate_longitudes(self, ds):
        """
        Concatenates the field three times: with longitudes shifted by -360, original longitudes, and shifted by +360.

        Parameters
        ----------
        field : xr.DataArray
            The field to be concatenated.

        Returns
        -------
        xr.DataArray
            The concatenated field, with the longitude dimension extended.

        Notes
        -----
        Concatenating three times may be overkill in most situations, but it is safe. Alternatively, we could refactor
        to figure out whether concatenating on the lower end, upper end, or at all is needed.

        """
        ds_concatenated = xr.Dataset()

        lon = ds[self.dim_names["longitude"]]
        lon_minus360 = lon - 360
        lon_plus360 = lon + 360
        lon_concatenated = xr.concat(
            [lon_minus360, lon, lon_plus360], dim=self.dim_names["longitude"]
        )

        ds_concatenated[self.dim_names["longitude"]] = lon_concatenated

        for var in self.var_names.values():
            if self.dim_names["longitude"] in ds[var].dims:
                field = ds[var]
                field_concatenated = xr.concat(
                    [field, field, field], dim=self.dim_names["longitude"]
                )
                if self.use_dask:
                    field_concatenated = field_concatenated.chunk(
                        {self.dim_names["longitude"]: -1}
                    )
                field_concatenated[self.dim_names["longitude"]] = lon_concatenated
                ds_concatenated[var] = field_concatenated
            else:
                ds_concatenated[var] = ds[var]

        return ds_concatenated

    def choose_subdomain(
        self, latitude_range, longitude_range, margin, straddle, return_subdomain=False
    ):
        """
        Selects a subdomain from the xarray Dataset based on specified latitude and longitude ranges,
        extending the selection by a specified margin. Handles longitude conversions to accommodate different
        longitude ranges.

        Parameters
        ----------
        latitude_range : tuple of float
            A tuple (lat_min, lat_max) specifying the minimum and maximum latitude values of the subdomain.
        longitude_range : tuple of float
            A tuple (lon_min, lon_max) specifying the minimum and maximum longitude values of the subdomain.
        margin : float
            Margin in degrees to extend beyond the specified latitude and longitude ranges when selecting the subdomain.
        straddle : bool
            If True, target longitudes are expected in the range [-180, 180].
            If False, target longitudes are expected in the range [0, 360].
        return_subdomain : bool, optional
            If True, returns the subset of the original dataset as an xarray Dataset. If False, assigns the subset to `self.ds`.
            Defaults to False.

        Returns
        -------
        xr.Dataset or None
            If `return_subdomain` is True, returns the subset of the original dataset representing the chosen subdomain,
            including an extended area to cover one extra grid point beyond the specified ranges. If `return_subdomain` is False,
            returns None as the subset is assigned to `self.ds`.

        Notes
        -----
        This method adjusts the longitude range if necessary to ensure it matches the expected range for the dataset.
        It also handles longitude discontinuities that can occur when converting to different longitude ranges.
        This is important for avoiding artifacts in the interpolation process.

        Raises
        ------
        ValueError
            If the selected latitude or longitude range does not intersect with the dataset.
        """

        lat_min, lat_max = latitude_range
        lon_min, lon_max = longitude_range

        if not self.is_global:
            # Adjust longitude range if needed to match the expected range
            lon = self.ds[self.dim_names["longitude"]]
            if not straddle:
                if lon.min() < -180:
                    if lon_max + margin > 0:
                        lon_min -= 360
                        lon_max -= 360
                elif lon.min() < 0:
                    if lon_max + margin > 180:
                        lon_min -= 360
                        lon_max -= 360

            if straddle:
                if lon.max() > 360:
                    if lon_min - margin < 180:
                        lon_min += 360
                        lon_max += 360
                elif lon.max() > 180:
                    if lon_min - margin < 0:
                        lon_min += 360
                        lon_max += 360

        # Select the subdomain
        subdomain = self.ds.sel(
            **{
                self.dim_names["latitude"]: slice(lat_min - margin, lat_max + margin),
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
        if straddle:
            subdomain[self.dim_names["longitude"]] = xr.where(lon > 180, lon - 360, lon)
        else:
            subdomain[self.dim_names["longitude"]] = xr.where(lon < 0, lon + 360, lon)

        if return_subdomain:
            return subdomain
        else:
            object.__setattr__(self, "ds", subdomain)

    def convert_to_negative_depth(self):
        """
        Converts the depth values in the dataset to negative if they are non-negative.

        This method checks the values in the depth dimension of the dataset (`self.ds[self.dim_names["depth"]]`).
        If all values are greater than or equal to zero, it negates them and updates the dataset accordingly.

        """
        depth = self.ds[self.dim_names["depth"]]

        if (depth >= 0).all():
            self.ds[self.dim_names["depth"]] = -depth


@dataclass(frozen=True, kw_only=True)
class TPXODataset(Dataset):
    """
    Represents tidal data on the original grid from the TPXO dataset.

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

    def __post_init__(self):
        # Perform any necessary dataset initialization or modifications here
        ds = super().load_data()

        # Clean up dataset
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
        self.check_dataset(ds)

        # Select relevant fields
        ds = super().select_relevant_fields(ds)

        # Make sure that latitude is ascending
        ds = super().ensure_latitude_ascending(ds)

        # Check whether the data covers the entire globe
        object.__setattr__(self, "is_global", super().check_if_global(ds))

        # If dataset is global concatenate three copies of field along longitude dimension
        if self.is_global:
            ds = super().concatenate_longitudes(ds)

        object.__setattr__(self, "ds", ds)

    def check_number_constituents(self, ntides: int):
        """
        Checks if the number of constituents in the dataset is at least `ntides`.

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
        """
        Apply a depth-based mask to the dataset, ensuring only positive depths are retained.

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
            mask = xr.where(self.ds["depth"] > 0, 1, 0)

            for var in self.ds.data_vars:
                self.ds[var] = xr.where(mask == 1, self.ds[var], np.nan)

            self.ds["mask"] = mask


@dataclass(frozen=True, kw_only=True)
class GLORYSDataset(Dataset):
    """
    Represents GLORYS data on original grid.

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
        """
        Apply a mask to the dataset based on the 'zeta' variable, with 0 where 'zeta' is NaN.

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

        for var in self.ds.data_vars:
            self.ds[var] = xr.where(mask == 1, self.ds[var], np.nan)

        self.ds["mask"] = mask


@dataclass(frozen=True, kw_only=True)
class CESMDataset(Dataset):
    """
    Represents CESM data on original grid.

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
        Indicates whether the dataset is climatological. Defaults to True.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the CESM data on its original grid.
    """

    # overwrite load_data method from parent class
    def load_data(self) -> xr.Dataset:
        """
        Load dataset from the specified file.

        Returns
        -------
        ds : xr.Dataset
            The loaded xarray Dataset containing the forcing data.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """

        ds = super().load_data()

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
        """
        Adds time information to the dataset based on the climatology flag and dimension names.

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
    """
    Represents CESM BGC data on original grid.

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
        Indicates whether the dataset is climatological. Defaults to True.

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

    climatology: Optional[bool] = True

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

        for var in self.ds.data_vars:
            self.ds[var] = xr.where(mask == 1, self.ds[var], np.nan)

        self.ds["mask"] = mask


@dataclass(frozen=True, kw_only=True)
class CESMBGCSurfaceForcingDataset(CESMDataset):
    """
    Represents CESM BGC surface forcing data on original grid.

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
        """
        Perform post-processing on the dataset to remove specific variables.

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

        for var in self.ds.data_vars:
            self.ds[var] = xr.where(mask == 1, self.ds[var], np.nan)

        self.ds["mask"] = mask


@dataclass(frozen=True, kw_only=True)
class ERA5Dataset(Dataset):
    """
    Represents ERA5 data on original grid.

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
            self.ds["qair"] = 0.62197 * (cff / (patm - 0.378 * cff))
            self.ds["qair"].attrs["long_name"] = "Absolute humidity at 2m"
            self.ds["qair"].attrs["units"] = "kg/kg"

            # Update var_names dictionary
            var_names = {**self.var_names, "qair": "qair"}
            object.__setattr__(self, "var_names", var_names)

        if "mask" in self.var_names.keys():
            mask = xr.where(self.ds[self.var_names["mask"]].isel(time=0).isnull(), 0, 1)

            for var in self.ds.data_vars:
                self.ds[var] = xr.where(mask == 1, self.ds[var], np.nan)

            self.ds["mask"] = mask


@dataclass(frozen=True, kw_only=True)
class ERA5Correction(Dataset):
    """
    Global dataset to correct ERA5 radiation. The dataset contains multiplicative correction factors for the ERA5 shortwave radiation, obtained by comparing the COREv2 climatology to the ERA5 climatology.

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
        """
        Converts longitude values in the dataset if necessary and selects a subdomain based on the specified coordinates.

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
