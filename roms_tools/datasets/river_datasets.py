import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from roms_tools.datasets.download import (
    download_river_data,
    download_river_tracer_defaults,
)
from roms_tools.datasets.utils import check_dataset, select_relevant_times
from roms_tools.utils import _get_file_matches, load_data

RIVR2O_FILL_VALUE = -999.0
RIVR2O_TRACER_NAMES = ("DIC", "DOC_l", "DOC_sl", "POC", "NO3", "PO4")
RIVR2O_MIN_YEAR = 1903
RIVR2O_MAX_YEAR = 2024
SECONDS_PER_YEAR = 365.25 * 24 * 3600

VALUE_OPTION_DIM = "value_option"
RECOMMENDED_VALUE_INDEX = 0
RIVER_TRACER_DEFAULTS_FILENAME = "river_tracer_defaults.nc"

EXPECTED_TRACER_NAMES = (
    "temp",
    "salt",
    "PO4",
    "NO3",
    "SiO3",
    "NH4",
    "Fe",
    "Lig",
    "O2",
    "DIC",
    "DIC_ALT_CO2",
    "ALK",
    "ALK_ALT_CO2",
    "DOC",
    "DON",
    "DOP",
    "DOPr",
    "DONr",
    "DOCr",
    "zooC",
    "spChl",
    "spC",
    "spP",
    "spFe",
    "spCaCO3",
    "diatChl",
    "diatC",
    "diatP",
    "diatFe",
    "diatSi",
    "diazChl",
    "diazC",
    "diazP",
    "diazFe",
)


@dataclass(kw_only=True)
class RiverTracerDefaultsDataset:
    """Default MARBL river tracer concentrations.

    Loads recommended boundary concentrations from ``river_tracer_defaults.nc``
    in the roms-tools-data repository. Each tracer variable has dimension
    ``value_option`` (0 = recommended, 1 = alternate).

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
        expected_tracers = set(EXPECTED_TRACER_NAMES)

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

        from roms_tools.setup.utils import gc_dist

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


@dataclass(kw_only=True)
class Rivr2oRiverBGCDataset:
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
            return ds.isel({time_dim: idx})

        return select_relevant_times(
            ds=ds,
            time_dim=time_dim,
            time_coord=time_dim,
            start_time=select_start,
            end_time=select_end,
        )

    def _adjust_lon_to_grid(self, lon: np.ndarray, *, straddle: bool) -> np.ndarray:
        """Put query longitudes in the same convention as the RIVR2O lon coordinate."""
        lon_dim = self.dim_names["longitude"]
        grid_lon = self.ds[lon_dim]
        if float(grid_lon.max()) > 180:
            return np.where(lon < 0, lon + 360, lon)
        if straddle:
            return np.where(lon > 180, lon - 360, lon)
        return lon

    def _nearest_time_index(self, time: datetime | np.datetime64) -> int:
        """Index of the RIVR2O annual record closest to ``time``."""
        time_dim = self.dim_names["time"]
        target = np.datetime64(time)
        return int(np.argmin(np.abs(self.ds[time_dim].values - target)))

    def _valid_export_cell_indices(
        self,
        *,
        time_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return indices and coordinates of grid cells with non-zero river export.

        When ``time_index`` is set, only cells with positive export in that annual
        record are included. Otherwise any year with export qualifies.
        """
        lat_dim = self.dim_names["latitude"]
        lon_dim = self.dim_names["longitude"]
        time_dim = self.dim_names["time"]

        if time_index is None:
            has_export = self.ds[self.tracer_names[0]] > 0
            for tracer in self.tracer_names[1:]:
                has_export = has_export | (self.ds[tracer] > 0)
            valid = has_export.any(dim=time_dim)
        else:
            has_export = self.ds[self.tracer_names[0]].isel({time_dim: time_index}) > 0
            for tracer in self.tracer_names[1:]:
                has_export = has_export | (
                    self.ds[tracer].isel({time_dim: time_index}) > 0
                )
            valid = has_export

        lat_indices, lon_indices = np.where(valid.values)
        if lat_indices.size == 0:
            if time_index is not None:
                logging.warning(
                    "No non-zero RIVR2O export at time index %s; "
                    "falling back to cells valid in any year.",
                    time_index,
                )
                return self._valid_export_cell_indices(time_index=None)
            raise ValueError(
                "No non-zero RIVR2O export cells found in the loaded dataset."
            )

        grid_lats = self.ds[lat_dim].values[lat_indices]
        grid_lons = self.ds[lon_dim].values[lon_indices]
        return lat_indices, lon_indices, grid_lats, grid_lons

    def sample_at_points(
        self,
        lon: xr.DataArray | np.ndarray | float,
        lat: xr.DataArray | np.ndarray | float,
        *,
        straddle: bool = False,
        method: str = "nearest",
        time: datetime | np.datetime64 | None = None,
    ) -> xr.Dataset:
        """Sample MARBL tracer exports at point locations.

        For each query point, the nearest RIVR2O grid cell with a positive export
        (non-zero, non-fill) is used. This avoids picking land or ocean cells where
        the product is zero when the ROMS river mouth lies between active river
        cells on the RIVR2O grid.

        Parameters
        ----------
        lon, lat : array-like or scalar
            Longitude and latitude of sample points in degrees.
        straddle : bool, optional
            If True, longitudes greater than 180° are converted to -180-180 before
            sampling. If False, negative longitudes are converted to 0-360.
        time : datetime or numpy.datetime64, optional
            Annual RIVR2O record used to decide which grid cells have export. When
            omitted, a cell is eligible if it has export in any loaded year.
        method : str, optional
            Accepted for API compatibility; sampling always uses the nearest
            non-zero export cell.

        Returns
        -------
        xr.Dataset
            Tracer variables with dimensions ``(time, points)`` when multiple
            locations are provided, or ``(time,)`` for a scalar point.
        """
        del method  # nearest non-zero coastal cell, not xarray interp
        lat_dim = self.dim_names["latitude"]
        lon_dim = self.dim_names["longitude"]

        scalar_point = not np.ndim(lon)
        query_lon = np.atleast_1d(np.asarray(lon, dtype=float))
        query_lat = np.atleast_1d(np.asarray(lat, dtype=float))
        query_lon = self._adjust_lon_to_grid(query_lon, straddle=straddle)

        time_index = self._nearest_time_index(time) if time is not None else None
        lat_indices, lon_indices, grid_lats, grid_lons = (
            self._valid_export_cell_indices(time_index=time_index)
        )

        from roms_tools.setup.utils import gc_dist

        nearest_lat = []
        nearest_lon = []
        for q_lon, q_lat in zip(query_lon, query_lat, strict=True):
            dist = gc_dist(q_lon, q_lat, grid_lons, grid_lats)
            nearest = int(np.argmin(dist))
            nearest_lat.append(lat_indices[nearest])
            nearest_lon.append(lon_indices[nearest])

        sampled = self.ds.isel(
            {
                lat_dim: xr.DataArray(nearest_lat, dims="points"),
                lon_dim: xr.DataArray(nearest_lon, dims="points"),
            }
        )

        if scalar_point:
            return sampled.squeeze("points", drop=True)

        return sampled


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
