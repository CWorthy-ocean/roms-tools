import xarray as xr
from dataclasses import dataclass, field
import glob
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Optional
import dask
import warnings
from roms_tools.setup.utils import interpolate_from_climatology


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
    var_names: Dict[str, str]
        Dictionary of variable names that are required in the dataset.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the forcing data on its original grid.
    climatology : bool
        Indicates whether the dataset is climatological. Set to `True` if relevant.

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
    var_names: Dict[str, str]
    dim_names: Dict[str, str] = field(
        default_factory=lambda: {
            "longitude": "longitude",
            "latitude": "latitude",
            "time": "time",
        }
    )

    ds: xr.Dataset = field(init=False, repr=False)
    climatology: bool = field(init=False, default=False)

    def __post_init__(self):
        """
        Post-initialization processing:
        1. Loads the dataset from the specified filename.
        2. Applies time filtering based on start_time and end_time if provided.
        3. Selects relevant fields as specified by var_names.
        4. Ensures latitude values are in ascending order.
        5. Checks if the dataset covers the entire globe and adjusts if necessary.
        6. Sets the climatology attribute based on whether the dataset is climatological.
        """

        ds = self.load_data()

        # Select relevant times
        object.__setattr__(self, "climatology", False)
        if self.start_time is not None:
            if "time" in self.dim_names:
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
        missing_vars = [
            var for var in self.var_names.values() if var not in ds.data_vars
        ]
        if missing_vars:
            raise ValueError(
                f"Dataset does not contain all required variables. The following variables are missing: {missing_vars}"
            )

        for var in ds.data_vars:
            if var not in self.var_names.values():
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

        Notes
        -----
        If the dataset has a 'month' dimension, it assumes the data is climatological and performs
        interpolation or sets the climatology flag accordingly. If the time dimension is 'month',
        the method calculates the time as a cumulative sum of days in each month, converts it to
        `timedelta64[ns]`, and updates the dataset's time coordinates.

        If `self.end_time` is not provided and only a single time slice is required, interpolation
        from climatology is applied. For datasets covering full climatology, the climatology flag is set
        to True, and no additional filtering is performed.

        Raises
        ------
        ValueError
            If no matching times are found or if the number of matching times does not meet expectations.

        Warns
        -----
        Warning
            If no time information is found in the dataset.
        """

        time_dim = self.dim_names["time"]
        if time_dim in ds.coords or time_dim in ds.data_vars:
            if time_dim == "month":
                # Define the days in each month and convert to timedelta
                increments = [15, 30, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30]
                days = np.cumsum(increments)
                timedelta_ns = np.array(days, dtype="timedelta64[D]").astype(
                    "timedelta64[ns]"
                )
                time = xr.DataArray(timedelta_ns, dims=["month"])
                ds = ds.assign_coords({"time": time})
                ds = ds.swap_dims({"month": "time"})

                # Update dimension names
                updated_dim_names = self.dim_names.copy()
                updated_dim_names["time"] = "time"
                object.__setattr__(self, "dim_names", updated_dim_names)

                if not self.end_time:
                    # Interpolate from climatology for initial conditions
                    ds = interpolate_from_climatology(
                        ds, self.dim_names["time"], self.start_time
                    )
                else:
                    # Set climatology flag for full climatology datasets
                    object.__setattr__(self, "climatology", True)
                    pass
            else:
                if not self.end_time:
                    end_time = self.start_time + timedelta(days=1)
                else:
                    end_time = self.end_time

                times = (np.datetime64(self.start_time) <= ds[time_dim]) & (
                    ds[time_dim] < np.datetime64(end_time)
                )
                ds = ds.where(times, drop=True)
        else:
            warnings.warn(
                f"Dataset at {self.filename} does not contain any time information."
            )

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

        for var in self.var_names.values():
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
        ds = ds.rename({"nx": "longitude", "ny": "latitude"})

        object.__setattr__(
            self,
            "dim_names",
            {
                "latitude": "latitude",
                "longitude": "longitude",
                "ntides": self.dim_names["ntides"],
            },
        )
        # Select relevant fields
        ds = super().select_relevant_fields(ds)

        # Check whether the data covers the entire globe
        is_global = self.check_if_global(ds)

        if is_global:
            ds = self.concatenate_longitudes(ds)

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
            "ssh": "zos",
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


@dataclass(frozen=True, kw_only=True)
class CESMBGCDataset(Dataset):
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

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the GLORYS data on its original grid.
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
            "time": "time",
        }
    )
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

            ds = xr.open_mfdataset(
                self.filename,
                combine="nested",
                coords="minimal",
                compat="override",
                chunks=chunks,
                engine="netcdf4",
            )

            if "time" not in ds.dims:
                if "month" in ds.dims:
                    self.dim_names["time"] = "month"
                else:
                    ds = ds.expand_dims({"time": 1})

        return ds

    def post_process(self):
        """
        Processes and converts CESM data values as follows:
        - Convert depth values from cm to m.
        """
        if self.dim_names["depth"] == "z_t":
            # Fill variables that only have data in upper 150m with NaNs below
            if (
                "z_t_150m" in self.ds.dims
                and np.equal(self.ds.z_t[:15].values, self.ds.z_t_150m.values).all()
            ):
                for var in self.var_names:
                    if "z_t_150m" in self.ds[var].dims:
                        self.ds[var] = self.ds[var].rename({"z_t_150m": "z_t"})

            # Convert depth from cm to m
            ds = self.ds.assign_coords({"depth": self.ds["z_t"] / 100})
            ds["depth"].attrs["long_name"] = "Depth"
            ds["depth"].attrs["units"] = "m"
            ds = ds.swap_dims({"z_t": "depth"})
            # update dataset
            object.__setattr__(self, "ds", ds)

            # Update dim_names with "depth": "depth" key-value pair
            updated_dim_names = self.dim_names.copy()
            updated_dim_names["depth"] = "depth"
            object.__setattr__(self, "dim_names", updated_dim_names)


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

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the GLORYS data on its original grid.
    """

    var_names: Dict[str, str] = field(
        default_factory=lambda: {
            "u10": "u10",
            "v10": "v10",
            "swr": "ssr",
            "lwr": "strd",
            "t2m": "t2m",
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

    def post_process(self):
        """
        Processes and converts ERA5 data values as follows:
        - Convert radiation values from J/m^2 to W/m^2.
        - Convert rainfall from meters to cm/day.
        - Convert temperature from Kelvin to Celsius.
        - Compute relative humidity if not present, convert to absolute humidity.
        """
        # Translate radiation to fluxes. ERA5 stores values integrated over 1 hour.
        # Convert radiation from J/m^2 to W/m^2
        self.ds[self.var_names["swr"]] /= 3600
        self.ds[self.var_names["lwr"]] /= 3600
        self.ds[self.var_names["swr"]].attrs["units"] = "W/m^2"
        self.ds[self.var_names["lwr"]].attrs["units"] = "W/m^2"
        # Convert rainfall from m to cm/day
        self.ds[self.var_names["rain"]] *= 100 * 24

        # Convert temperature from Kelvin to Celsius
        self.ds[self.var_names["t2m"]] -= 273.15
        self.ds[self.var_names["d2m"]] -= 273.15
        self.ds[self.var_names["t2m"]].attrs["units"] = "degrees C"
        self.ds[self.var_names["d2m"]].attrs["units"] = "degrees C"

        # Compute relative humidity if not present
        if "qair" not in self.ds.data_vars:
            qair = np.exp(
                (17.625 * self.ds[self.var_names["d2m"]])
                / (243.04 + self.ds[self.var_names["d2m"]])
            ) / np.exp(
                (17.625 * self.ds[self.var_names["t2m"]])
                / (243.04 + self.ds[self.var_names["t2m"]])
            )
            # Convert relative to absolute humidity
            patm = 1010.0
            cff = (
                (1.0007 + 3.46e-6 * patm)
                * 6.1121
                * np.exp(
                    17.502
                    * self.ds[self.var_names["t2m"]]
                    / (240.97 + self.ds[self.var_names["t2m"]])
                )
            )
            cff = cff * qair
            self.ds["qair"] = 0.62197 * (cff / (patm - 0.378 * cff))
            self.ds["qair"].attrs["long_name"] = "Absolute humidity at 2m"
            self.ds["qair"].attrs["units"] = "kg/kg"

            # Update var_names dictionary
            var_names = {**self.var_names, "qair": "qair"}
            object.__setattr__(self, "var_names", var_names)
