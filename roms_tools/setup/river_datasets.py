import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from roms_tools.download import download_river_data
from roms_tools.setup.utils import (
    assign_dates_to_climatology,
    check_dataset,
    gc_dist,
    select_relevant_times,
)
from roms_tools.utils import load_data


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
        ds = load_data(
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

        ds = select_relevant_times(ds, time_dim, self.start_time, self.end_time, False)

        return ds

    def compute_climatology(self):
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
        indices : dict[str, list[tuple]]
            A dictionary containing the indices of the rivers that are within the threshold distance from
            the target coordinates. The dictionary structure consists of river names as keys, and each value is a list of tuples. Each tuple represents
            a pair of indices corresponding to the `eta_rho` and `xi_rho` grid coordinates of the river.
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

            river_indices = get_indices_of_nearest_grid_cell_for_rivers(dist, self)
        else:
            ds = xr.Dataset()
            river_indices = {}

        self.ds = ds

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

        # Ensure the dataset is filtered based on the provided river names
        ds_filtered = self.ds.where(
            self.ds[self.var_names["name"]].isin(river_names), drop=True
        )

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
class DaiRiverDataset(RiverDataset):
    """Represents river data from the Dai river dataset."""

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
