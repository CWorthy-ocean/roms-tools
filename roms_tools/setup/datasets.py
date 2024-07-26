import pooch
import xarray as xr
from dataclasses import dataclass, field
import glob
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Optional, List
import dask

# Create a Pooch object to manage the global topography data
pup_data = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("roms-tools"),
    base_url="https://github.com/CWorthy-ocean/roms-tools-data/raw/main/",
    # The registry specifies the files that can be fetched
    registry={
        "etopo5.nc": "sha256:23600e422d59bbf7c3666090166a0d468c8ee16092f4f14e32c4e928fbcd627b",
    },
)

# Create a Pooch object to manage the test data
pup_test_data = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("roms-tools"),
    base_url="https://github.com/CWorthy-ocean/roms-tools-test-data/raw/main/",
    # The registry specifies the files that can be fetched
    registry={
        "GLORYS_test_data.nc": "648f88ec29c433bcf65f257c1fb9497bd3d5d3880640186336b10ed54f7129d2",
        "ERA5_regional_test_data.nc": "bd12ce3b562fbea2a80a3b79ba74c724294043c28dc98ae092ad816d74eac794",
        "ERA5_global_test_data.nc": "8ed177ab64c02caf509b9fb121cf6713f286cc603b1f302f15f3f4eb0c21dc4f",
        "TPXO_global_test_data.nc": "457bfe87a7b247ec6e04e3c7d3e741ccf223020c41593f8ae33a14f2b5255e60",
        "TPXO_regional_test_data.nc": "11739245e2286d9c9d342dce5221e6435d2072b50028bef2e86a30287b3b4032",
    },
)


def fetch_topo(topography_source: str) -> xr.Dataset:
    """
    Load the global topography data as an xarray Dataset.

    Parameters
    ----------
    topography_source : str
        The source of the topography data to be loaded. Available options:
        - "ETOPO5"

    Returns
    -------
    xr.Dataset
        The global topography data as an xarray Dataset.
    """
    # Mapping from user-specified topography options to corresponding filenames in the registry
    topo_dict = {"ETOPO5": "etopo5.nc"}

    # Fetch the file using Pooch, downloading if necessary
    fname = pup_data.fetch(topo_dict[topography_source])

    # Load the dataset using xarray and return it
    ds = xr.open_dataset(fname)
    return ds


def download_test_data(filename: str) -> str:
    """
    Download the test data file.

    Parameters
    ----------
    filename : str
        The name of the test data file to be downloaded. Available options:
        - "GLORYS_test_data.nc"
        - "ERA5_regional_test_data.nc"
        - "ERA5_global_test_data.nc"

    Returns
    -------
    str
        The path to the downloaded test data file.
    """
    # Fetch the file using Pooch, downloading if necessary
    fname = pup_test_data.fetch(filename)

    return fname


@dataclass(frozen=True, kw_only=True)
class Dataset:
    """
    Represents forcing data on original grid.

    Parameters
    ----------
    filename : str
        The path to the data files. Can contain wildcards.
    start_time : Optional[datetime], optional
        The start time for selecting relevant data. If not provided, the data is not filtered by start time.
    end_time : Optional[datetime], optional
        The end time for selecting relevant data. If not provided, only data at the start_time is selected if start_time is provided,
        or no filtering is applied if start_time is not provided.
    var_names : List[str]
        List of variable names that are required in the dataset.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.

    Attributes
    ----------
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

    filename: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    var_names: List[str]
    dim_names: Dict[str, str] = field(
        default_factory=lambda: {
            "longitude": "longitude",
            "latitude": "latitude",
            "time": "time",
        }
    )

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        ds = self.load_data()

        # Select relevant times
        if "time" in self.dim_names and self.start_time is not None:
            ds = self.select_relevant_times(ds)

        # Select relevant fields
        ds = self.select_relevant_fields(ds)

        # Make sure that latitude is ascending
        diff = np.diff(ds[self.dim_names["latitude"]])
        if np.all(diff < 0):
            ds = ds.isel(**{self.dim_names["latitude"]: slice(None, None, -1)})

        # Check whether the data covers the entire globe
        is_global = self.check_if_global(ds)

        if is_global:
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
        """

        # Check if the file exists
        matching_files = glob.glob(self.filename)
        if not matching_files:
            raise FileNotFoundError(
                f"No files found matching the pattern '{self.filename}'."
            )

        # Load the dataset
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            # Define the chunk sizes
            chunks = {
                self.dim_names["latitude"]: -1,
                self.dim_names["longitude"]: -1,
            }
            if "depth" in self.dim_names.keys():
                chunks[self.dim_names["depth"]] = -1
            if "time" in self.dim_names.keys():
                chunks[self.dim_names["time"]] = 1

                ds = xr.open_mfdataset(
                    self.filename,
                    combine="nested",
                    concat_dim=self.dim_names["time"],
                    coords="minimal",
                    compat="override",
                    chunks=chunks,
                    engine="netcdf4",
                )
            else:
                ds = xr.open_dataset(
                    self.filename,
                    chunks=chunks,
                )

        return ds

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

        Raises
        ------
        ValueError
            If `ds` does not contain all variables listed in `self.var_names`.

        """
        missing_vars = [var for var in self.var_names if var not in ds.data_vars]
        if missing_vars:
            raise ValueError(
                f"Dataset does not contain all required variables. The following variables are missing: {missing_vars}"
            )

        for var in ds.data_vars:
            if var not in self.var_names:
                ds = ds.drop_vars(var)

        return ds

    def select_relevant_times(self, ds) -> xr.Dataset:

        """
        Selects and returns the subset of the dataset corresponding to the specified time range.

        This function filters the dataset to include only the data points within the specified
        time range, defined by `self.start_time` and `self.end_time`. If `self.end_time` is not
        provided, it defaults to one day after `self.start_time`.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset to be filtered.

        Returns
        -------
        xr.Dataset
            A dataset containing only the data points within the specified time range.

        """

        time_dim = self.dim_names["time"]

        if not self.end_time:
            end_time = self.start_time + timedelta(days=1)
        else:
            end_time = self.end_time

        times = (np.datetime64(self.start_time) <= ds[time_dim]) & (
            ds[time_dim] < np.datetime64(end_time)
        )
        ds = ds.where(times, drop=True)

        if not ds.sizes[time_dim]:
            raise ValueError("No matching times found.")

        if not self.end_time:
            if ds.sizes[time_dim] != 1:
                found_times = ds.sizes[time_dim]
                raise ValueError(
                    f"There must be exactly one time matching the start_time. Found {found_times} matching times."
                )

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
        is_global = np.isclose(dlon, dlon_mean, rtol=0.0, atol=1e-3)

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

        for var in self.var_names:
            if self.dim_names["longitude"] in ds[var].dims:
                field = ds[var]
                field_concatenated = xr.concat(
                    [field, field, field], dim=self.dim_names["longitude"]
                ).chunk({self.dim_names["longitude"]: -1})
                field_concatenated[self.dim_names["longitude"]] = lon_concatenated
                ds_concatenated[var] = field_concatenated
            else:
                ds_concatenated[var] = ds[var]

        return ds_concatenated

    def choose_subdomain(
        self, latitude_range, longitude_range, margin, straddle, return_subdomain=False
    ):
        """
        Selects a subdomain from the given xarray Dataset based on latitude and longitude ranges,
        extending the selection by the specified margin. Handles the conversion of longitude values
        in the dataset from one range to another.

        Parameters
        ----------
        latitude_range : tuple
            A tuple (lat_min, lat_max) specifying the minimum and maximum latitude values of the subdomain.
        longitude_range : tuple
            A tuple (lon_min, lon_max) specifying the minimum and maximum longitude values of the subdomain.
        margin : float
            Margin in degrees to extend beyond the specified latitude and longitude ranges when selecting the subdomain.
        straddle : bool
            If True, target longitudes are expected in the range [-180, 180].
            If False, target longitudes are expected in the range [0, 360].
        return_subdomain : bool, optional
            If True, returns the subset of the original dataset. If False, assigns it to self.ds.
            Default is False.

        Returns
        -------
        xr.Dataset
            The subset of the original dataset representing the chosen subdomain, including an extended area
            to cover one extra grid point beyond the specified ranges if return_subdomain is True.
            Otherwise, returns None.

        Raises
        ------
        ValueError
            If the selected latitude or longitude range does not intersect with the dataset.
        """
        lat_min, lat_max = latitude_range
        lon_min, lon_max = longitude_range

        lon = self.ds[self.dim_names["longitude"]]
        # Adjust longitude range if needed to match the expected range
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
