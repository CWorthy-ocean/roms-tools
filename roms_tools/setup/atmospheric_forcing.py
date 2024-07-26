import xarray as xr
import dask
import yaml
import importlib.metadata
from dataclasses import dataclass, field, asdict
from roms_tools.setup.grid import Grid
from datetime import datetime
import glob
import numpy as np
from typing import Optional, Dict
from roms_tools.setup.fill import fill_and_interpolate
from roms_tools.setup.datasets import Dataset
from roms_tools.setup.utils import nan_check
from roms_tools.setup.plot import _plot
import calendar
import matplotlib.pyplot as plt


@dataclass(frozen=True, kw_only=True)
class SWRCorrection:
    """
    Configuration for shortwave radiation correction.

    Parameters
    ----------
    filename : str
        Filename of the correction data.
    varname : str
        Variable identifier for the correction.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
        Default is {"longitude": "lon", "latitude": "lat", "time": "time"}.
    temporal_resolution : str, optional
        Temporal resolution of the correction data. Default is "climatology".

    Attributes
    ----------
    ds : xr.Dataset
        The loaded xarray Dataset containing the correction data.

    Examples
    --------
    >>> swr_correction = SWRCorrection(
    ...     filename="correction_data.nc",
    ...     varname="corr",
    ...     dim_names={
    ...         "time": "time",
    ...         "latitude": "latitude",
    ...         "longitude": "longitude",
    ...     },
    ...     temporal_resolution="climatology",
    ... )
    """

    filename: str
    varname: str
    dim_names: Dict[str, str] = field(
        default_factory=lambda: {
            "longitude": "longitude",
            "latitude": "latitutde",
            "time": "time",
        }
    )
    temporal_resolution: str = "climatology"
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        if self.temporal_resolution != "climatology":
            raise NotImplementedError(
                f"temporal_resolution must be 'climatology', got {self.temporal_resolution}"
            )

        ds = self._load_data()
        self._check_dataset(ds)
        ds = self._ensure_latitude_ascending(ds)

        object.__setattr__(self, "ds", ds)

    def _load_data(self):
        """
        Load data from the specified file.

        Returns
        -------
        ds : xr.Dataset
            The loaded xarray Dataset containing the correction data.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.

        """
        # Check if the file exists

        # Check if any file matching the wildcard pattern exists
        matching_files = glob.glob(self.filename)
        if not matching_files:
            raise FileNotFoundError(
                f"No files found matching the pattern '{self.filename}'."
            )

        # Load the dataset
        ds = xr.open_dataset(
            self.filename,
            chunks={
                self.dim_names["time"]: -1,
                self.dim_names["latitude"]: -1,
                self.dim_names["longitude"]: -1,
            },
        )

        return ds

    def _check_dataset(self, ds: xr.Dataset) -> None:
        """
        Check if the dataset contains the specified variable and dimensions.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset to check.

        Raises
        ------
        ValueError
            If the dataset does not contain the specified variable or dimensions.
        """
        if self.varname not in ds:
            raise ValueError(
                f"The dataset does not contain the variable '{self.varname}'."
            )

        for dim in self.dim_names.values():
            if dim not in ds.dims:
                raise ValueError(f"The dataset does not contain the dimension '{dim}'.")

    def _ensure_latitude_ascending(self, ds: xr.Dataset) -> xr.Dataset:
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

    def _handle_longitudes(self, straddle: bool) -> None:
        """
        Handles the conversion of longitude values in the dataset from one range to another.

        Parameters
        ----------
        straddle : bool
            If True, target longitudes are in range [-180, 180].
            If False, target longitudes are in range [0, 360].

        Raises
        ------
        ValueError: If the conversion results in discontinuous longitudes.
        """
        lon = self.ds[self.dim_names["longitude"]]

        if lon.min().values < 0 and not straddle:
            # Convert from [-180, 180] to [0, 360]
            self.ds[self.dim_names["longitude"]] = xr.where(lon < 0, lon + 360, lon)

        if lon.max().values > 180 and straddle:
            # Convert from [0, 360] to [-180, 180]
            self.ds[self.dim_names["longitude"]] = xr.where(lon > 180, lon - 360, lon)

    def _choose_subdomain(self, coords) -> xr.Dataset:
        """
        Selects a subdomain from the dataset based on the specified latitude and longitude ranges.

        Parameters
        ----------
        coords : dict
            A dictionary specifying the target coordinates.

        Returns
        -------
        xr.Dataset
            The subset of the original dataset representing the chosen subdomain.

        Raises
        ------
        ValueError
            If the specified subdomain is not fully contained within the dataset.
        """

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

        return subdomain

    def _interpolate_temporally(self, field, time):
        """
        Interpolates the given field temporally based on the specified time points.

        Parameters
        ----------
        field : xarray.DataArray
            The field data to be interpolated. This can be any variable from the dataset that
            requires temporal interpolation, such as correction factors or any other relevant data.
        time : xarray.DataArray or pandas.DatetimeIndex
            The target time points for interpolation.

        Returns
        -------
        xr.DataArray
            The field values interpolated to the specified time points.

        Raises
        ------
        NotImplementedError
            If the temporal resolution is not set to 'climatology'.

        """
        if self.temporal_resolution != "climatology":
            raise NotImplementedError(
                f"temporal_resolution must be 'climatology', got {self.temporal_resolution}"
            )
        else:
            field[self.dim_names["time"]] = field[self.dim_names["time"]].dt.days
            day_of_year = time.dt.dayofyear

            # Concatenate across the beginning and end of the year
            time_concat = xr.concat(
                [
                    field[self.dim_names["time"]][-1] - 365.25,
                    field[self.dim_names["time"]],
                    365.25 + field[self.dim_names["time"]][0],
                ],
                dim=self.dim_names["time"],
            )
            field_concat = xr.concat(
                [
                    field.isel({self.dim_names["time"]: -1}),
                    field,
                    field.isel({self.dim_names["time"]: 0}),
                ],
                dim=self.dim_names["time"],
            )
            field_concat["time"] = time_concat
            # Interpolate to specified times
            field_interpolated = field_concat.interp(time=day_of_year, method="linear")

        return field_interpolated

    @classmethod
    def from_yaml(cls, filepath: str) -> "SWRCorrection":
        """
        Create an instance of the class from a YAML file.

        Parameters
        ----------
        filepath : str
            The path to the YAML file from which the parameters will be read.

        Returns
        -------
        Grid
            An instance of the Grid class.
        """
        # Read the entire file content
        with open(filepath, "r") as file:
            file_content = file.read()

        # Split the content into YAML documents
        documents = list(yaml.safe_load_all(file_content))

        swr_correction_data = None

        # Iterate over documents to find the header and grid configuration
        for doc in documents:
            if doc is None:
                continue
            if "SWRCorrection" in doc:
                swr_correction_data = doc["SWRCorrection"]
                break

        if swr_correction_data is None:
            raise ValueError("No SWRCorrection configuration found in the YAML file.")

        return cls(**swr_correction_data)


@dataclass(frozen=True, kw_only=True)
class Rivers:
    """
    Configuration for river forcing.

    Parameters
    ----------
    filename : str, optional
        Filename of the river forcing data.
    """

    filename: str = ""

    def __post_init__(self):
        if not self.filename:
            raise ValueError("The 'filename' must be provided.")

    @classmethod
    def from_yaml(cls, filepath: str) -> "Rivers":
        """
        Create an instance of the class from a YAML file.

        Parameters
        ----------
        filepath : str
            The path to the YAML file from which the parameters will be read.

        Returns
        -------
        Grid
            An instance of the Grid class.
        """
        # Read the entire file content
        with open(filepath, "r") as file:
            file_content = file.read()

        # Split the content into YAML documents
        documents = list(yaml.safe_load_all(file_content))

        rivers_data = None

        # Iterate over documents to find the header and grid configuration
        for doc in documents:
            if doc is None:
                continue
            if "Rivers" in doc:
                rivers_data = doc
                break

        if rivers_data is None:
            raise ValueError("No Rivers configuration found in the YAML file.")

        return cls(**rivers_data)


@dataclass(frozen=True, kw_only=True)
class AtmosphericForcing:
    """
    Represents atmospheric forcing data for ocean modeling.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
    use_coarse_grid: bool
        Whether to interpolate to coarsened grid. Default is False.
    start_time : datetime
        Start time of the desired forcing data.
    end_time : datetime
        End time of the desired forcing data.
    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.
    source : str, optional
        Source of the atmospheric forcing data. Default is "ERA5".
    filename: str
        Path to the atmospheric forcing source data file. Can contain wildcards.
    swr_correction : SWRCorrection
        Shortwave radiation correction configuration.
    rivers : Rivers, optional
        River forcing configuration.

    Attributes
    ----------
    ds : xr.Dataset
        Xarray Dataset containing the atmospheric forcing data.


    Examples
    --------
    >>> grid_info = Grid(...)
    >>> start_time = datetime(2000, 1, 1)
    >>> end_time = datetime(2000, 1, 2)
    >>> atm_forcing = AtmosphericForcing(
    ...     grid=grid_info,
    ...     start_time=start_time,
    ...     end_time=end_time,
    ...     source="ERA5",
    ...     filename="atmospheric_data_*.nc",
    ...     swr_correction=swr_correction,
    ... )
    """

    grid: Grid
    use_coarse_grid: bool = False
    start_time: datetime
    end_time: datetime
    model_reference_date: datetime = datetime(2000, 1, 1)
    source: str = "ERA5"
    filename: str
    swr_correction: Optional["SWRCorrection"] = None
    rivers: Optional["Rivers"] = None
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        # Check that the source is "ERA5"
        if self.source != "ERA5":
            raise ValueError('Only "ERA5" is a valid option for source.')
        if self.source == "ERA5":
            dims = {"longitude": "longitude", "latitude": "latitude", "time": "time"}
            varnames = {
                "u10": "u10",
                "v10": "v10",
                "swr": "ssr",
                "lwr": "strd",
                "t2m": "t2m",
                "d2m": "d2m",
                "rain": "tp",
                "mask": "sst",
            }

        data = Dataset(
            filename=self.filename,
            start_time=self.start_time,
            end_time=self.end_time,
            var_names=varnames.values(),
            dim_names=dims,
        )

        if self.use_coarse_grid:
            if "lon_coarse" not in self.grid.ds:
                raise ValueError(
                    "Grid has not been coarsened yet. Execute grid.coarsen() first."
                )

            lon = self.grid.ds.lon_coarse
            lat = self.grid.ds.lat_coarse
            angle = self.grid.ds.angle_coarse
        else:
            lon = self.grid.ds.lon_rho
            lat = self.grid.ds.lat_rho
            angle = self.grid.ds.angle

        # operate on longitudes between -180 and 180 unless ROMS domain lies at least 5 degrees in lontitude away from Greenwich meridian
        lon = xr.where(lon > 180, lon - 360, lon)
        straddle = True
        if not self.grid.straddle and abs(lon).min() > 5:
            lon = xr.where(lon < 0, lon + 360, lon)
            straddle = False

        # The following consists of two steps:
        # Step 1: Choose subdomain of forcing data including safety margin for interpolation, and Step 2: Convert to the proper longitude range.
        # We perform these two steps for two reasons:
        # A) Since the horizontal dimensions consist of a single chunk, selecting a subdomain before interpolation is a lot more performant.
        # B) Step 1 is necessary to avoid discontinuous longitudes that could be introduced by Step 2. Specifically, discontinuous longitudes
        # can lead to artifacts in the interpolation process. Specifically, if there is a data gap if data is not global,
        # discontinuous longitudes could result in values that appear to come from a distant location instead of producing NaNs.
        # These NaNs are important as they can be identified and handled appropriately by the nan_check function.
        data.choose_subdomain(
            latitude_range=[lat.min().values, lat.max().values],
            longitude_range=[lon.min().values, lon.max().values],
            margin=2,
            straddle=straddle,
        )

        # interpolate onto desired grid
        coords = {dims["latitude"]: lat, dims["longitude"]: lon}

        data_vars = {}

        mask = xr.where(data.ds[varnames["mask"]].isel(time=0).isnull(), 0, 1)

        # Fill and interpolate each variable
        for var in varnames.keys():
            if var != "mask":
                data_vars[var] = fill_and_interpolate(
                    data.ds[varnames[var]],
                    mask,
                    list(coords.keys()),
                    coords,
                    method="linear",
                )

        # Access the interpolated variables using data_vars dictionary
        u10 = data_vars["u10"]
        v10 = data_vars["v10"]
        swr = data_vars["swr"]
        lwr = data_vars["lwr"]
        t2m = data_vars["t2m"]
        d2m = data_vars["d2m"]
        rain = data_vars["rain"]

        if self.source == "ERA5":
            # translate radiation to fluxes. ERA5 stores values integrated over 1 hour.
            swr = swr / 3600  # from J/m^2 to W/m^2
            lwr = lwr / 3600  # from J/m^2 to W/m^2
            rain = rain * 100 * 24  # from m to cm/day
            # convert from K to C
            t2m = t2m - 273.15
            d2m = d2m - 273.15
            # relative humidity fraction
            qair = np.exp((17.625 * d2m) / (243.04 + d2m)) / np.exp(
                (17.625 * t2m) / (243.04 + t2m)
            )
            # convert relative to absolute humidity assuming constant pressure
            patm = 1010.0
            cff = (
                (1.0007 + 3.46e-6 * patm)
                * 6.1121
                * np.exp(17.502 * t2m / (240.97 + t2m))
            )
            cff = cff * qair
            qair = 0.62197 * (cff / (patm - 0.378 * cff))

        # correct shortwave radiation
        if self.swr_correction:

            # choose same subdomain as forcing data so that we can use same mask
            self.swr_correction._handle_longitudes(straddle=straddle)
            coords_correction = {
                self.swr_correction.dim_names["latitude"]: data.ds[
                    data.dim_names["latitude"]
                ],
                self.swr_correction.dim_names["longitude"]: data.ds[
                    data.dim_names["longitude"]
                ],
            }
            subdomain = self.swr_correction._choose_subdomain(coords_correction)

            # spatial interpolation
            corr_factor = subdomain[self.swr_correction.varname]
            coords_correction = {
                self.swr_correction.dim_names["latitude"]: lat,
                self.swr_correction.dim_names["longitude"]: lon,
            }
            corr_factor = fill_and_interpolate(
                corr_factor,
                mask,
                list(coords_correction.keys()),
                coords_correction,
                method="linear",
            )

            # temporal interpolation
            corr_factor = self.swr_correction._interpolate_temporally(
                corr_factor, time=swr.time
            )

            swr = corr_factor * swr

        if self.rivers:
            NotImplementedError("River forcing is not implemented yet.")
            # rain = rain + rivers

        # save in new dataset
        ds = xr.Dataset()

        ds["uwnd"] = (u10 * np.cos(angle) + v10 * np.sin(angle)).astype(
            np.float32
        )  # rotate to grid orientation
        ds["uwnd"].attrs["long_name"] = "10 meter wind in x-direction"
        ds["uwnd"].attrs["units"] = "m/s"

        ds["vwnd"] = (v10 * np.cos(angle) - u10 * np.sin(angle)).astype(
            np.float32
        )  # rotate to grid orientation
        ds["vwnd"].attrs["long_name"] = "10 meter wind in y-direction"
        ds["vwnd"].attrs["units"] = "m/s"

        ds["swrad"] = swr.astype(np.float32)
        ds["swrad"].attrs["long_name"] = "Downward short-wave (solar) radiation"
        ds["swrad"].attrs["units"] = "W/m^2"

        ds["lwrad"] = lwr.astype(np.float32)
        ds["lwrad"].attrs["long_name"] = "Downward long-wave (thermal) radiation"
        ds["lwrad"].attrs["units"] = "W/m^2"

        ds["Tair"] = t2m.astype(np.float32)
        ds["Tair"].attrs["long_name"] = "Air temperature at 2m"
        ds["Tair"].attrs["units"] = "degrees C"

        ds["qair"] = qair.astype(np.float32)
        ds["qair"].attrs["long_name"] = "Absolute humidity at 2m"
        ds["qair"].attrs["units"] = "kg/kg"

        ds["rain"] = rain.astype(np.float32)
        ds["rain"].attrs["long_name"] = "Total precipitation"
        ds["rain"].attrs["units"] = "cm/day"

        ds.attrs["title"] = "ROMS atmospheric forcing file created by ROMS-Tools"
        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"
        ds.attrs["roms_tools_version"] = roms_tools_version
        ds.attrs["start_time"] = str(self.start_time)
        ds.attrs["end_time"] = str(self.end_time)
        ds.attrs["model_reference_date"] = str(self.model_reference_date)
        ds.attrs["source"] = self.source
        ds.attrs["use_coarse_grid"] = str(self.use_coarse_grid)
        ds.attrs["swr_correction"] = str(self.swr_correction is not None)
        ds.attrs["rivers"] = str(self.rivers is not None)

        ds = ds.assign_coords({"lon": lon, "lat": lat})
        if self.use_coarse_grid:
            ds = ds.rename({"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"})
            mask_roms = self.grid.ds["mask_coarse"].rename(
                {"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"}
            )
        else:
            mask_roms = self.grid.ds["mask_rho"]

        if dims["time"] != "time":
            ds = ds.rename({dims["time"]: "time"})

        # Preserve the original time coordinate for readability
        ds = ds.assign_coords({"absolute_time": ds["time"]})

        # Translate the time coordinate to days since the model reference date
        model_reference_date = np.datetime64(self.model_reference_date)

        # Convert the time coordinate to the format expected by ROMS (days since model reference date)
        ds["time"] = (
            (ds["time"] - model_reference_date).astype("float64") / 3600 / 24 * 1e-9
        )
        ds["time"].attrs[
            "long_name"
        ] = f"time since {np.datetime_as_string(model_reference_date, unit='D')}"
        ds["time"].attrs["units"] = "days"

        for var in ds.data_vars:
            nan_check(ds[var].isel(time=0), mask_roms)

        object.__setattr__(self, "ds", ds)

    def plot(self, varname, time=0) -> None:
        """
        Plot the specified atmospheric forcing field for a given time slice.

        Parameters
        ----------
        varname : str
            The name of the atmospheric forcing field to plot. Options include:
            - "uwnd": 10 meter wind in x-direction.
            - "vwnd": 10 meter wind in y-direction.
            - "swrad": Downward short-wave (solar) radiation.
            - "lwrad": Downward long-wave (thermal) radiation.
            - "Tair": Air temperature at 2m.
            - "qair": Absolute humidity at 2m.
            - "rain": Total precipitation.
        time : int, optional
            The time index to plot. Default is 0, which corresponds to the first
            time slice.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        Raises
        ------
        ValueError
            If the specified varname is not one of the valid options.


        Examples
        --------
        >>> atm_forcing = AtmosphericForcing(
        ...     grid=grid_info,
        ...     start_time=start_time,
        ...     end_time=end_time,
        ...     source="ERA5",
        ...     filename="atmospheric_data_*.nc",
        ...     swr_correction=swr_correction,
        ... )
        >>> atm_forcing.plot("uwnd", time=0)
        """

        title = "%s at time %s" % (
            self.ds[varname].long_name,
            np.datetime_as_string(self.ds["absolute_time"].isel(time=time), unit="s"),
        )

        field = self.ds[varname].isel(time=time).compute()

        # choose colorbar
        if varname in ["uwnd", "vwnd"]:
            vmax = max(field.max().values, -field.min().values)
            vmin = -vmax
            cmap = plt.colormaps.get_cmap("RdBu_r")
        else:
            vmax = field.max().values
            vmin = field.min().values
            if varname in ["swrad", "lwrad", "Tair", "qair"]:
                cmap = plt.colormaps.get_cmap("YlOrRd")
            else:
                cmap = plt.colormaps.get_cmap("YlGnBu")
        cmap.set_bad(color="gray")

        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

        _plot(
            self.grid.ds,
            field=field,
            straddle=self.grid.straddle,
            coarse_grid=self.use_coarse_grid,
            title=title,
            kwargs=kwargs,
            c="g",
        )

    def save(self, filepath: str, time_chunk_size: int = 1) -> None:
        """
        Save the interpolated atmospheric forcing fields to netCDF4 files.

        This method groups the dataset by year and month, chunks the data by the specified
        time chunk size, and saves each chunked subset to a separate netCDF4 file named
        according to the year, month, and day range if not a complete month of data is included.

        Parameters
        ----------
        filepath : str
            The base path and filename for the output files. The files will be named with
            the format "filepath.YYYYMM.nc" if a full month of data is included, or
            "filepath.YYYYMMDD-DD.nc" otherwise.
        time_chunk_size : int, optional
            Number of time slices to include in each chunk along the time dimension. Default is 1,
            meaning each chunk contains one time slice.

        Returns
        -------
        None
        """

        datasets = []
        filenames = []
        writes = []

        # Group dataset by year
        gb = self.ds.groupby("absolute_time.year")

        for year, group_ds in gb:
            # Further group each yearly group by month
            sub_gb = group_ds.groupby("absolute_time.month")

            for month, ds in sub_gb:
                # Chunk the dataset by the specified time chunk size
                ds = ds.chunk({"time": time_chunk_size})
                datasets.append(ds)

                # Determine the number of days in the month
                num_days_in_month = calendar.monthrange(year, month)[1]
                first_day = ds.time.absolute_time.dt.day.values[0]
                last_day = ds.time.absolute_time.dt.day.values[-1]

                # Create filename based on whether the dataset contains a full month
                if first_day == 1 and last_day == num_days_in_month:
                    # Full month format: "filepath.YYYYMM.nc"
                    year_month_str = f"{year}{month:02}"
                    filename = f"{filepath}.{year_month_str}.nc"
                else:
                    # Partial month format: "filepath.YYYYMMDD-DD.nc"
                    year_month_day_str = f"{year}{month:02}{first_day:02}-{last_day:02}"
                    filename = f"{filepath}.{year_month_day_str}.nc"
                filenames.append(filename)

        print("Saving the following files:")
        for filename in filenames:
            print(filename)

        for ds, filename in zip(datasets, filenames):

            # Prepare the dataset for writing to a netCDF file without immediately computing
            write = ds.to_netcdf(filename, compute=False)
            writes.append(write)

        # Perform the actual write operations in parallel
        dask.persist(*writes)

    def to_yaml(self, filepath: str) -> None:
        """
        Export the parameters of the class to a YAML file, including the version of roms-tools.

        Parameters
        ----------
        filepath : str
            The path to the YAML file where the parameters will be saved.
        """
        # Serialize Grid data
        grid_data = asdict(self.grid)
        grid_data.pop("ds", None)  # Exclude non-serializable fields
        grid_data.pop("straddle", None)

        if self.swr_correction:
            swr_correction_data = asdict(self.swr_correction)
            swr_correction_data.pop("ds", None)
        else:
            swr_correction_data = None

        rivers_data = asdict(self.rivers) if self.rivers else None

        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"

        # Create header
        header = f"---\nroms_tools_version: {roms_tools_version}\n---\n"

        # Create YAML data for Grid and optional attributes
        grid_yaml_data = {"Grid": grid_data}
        swr_correction_yaml_data = (
            {"SWRCorrection": swr_correction_data} if swr_correction_data else {}
        )
        rivers_yaml_data = {"Rivers": rivers_data} if rivers_data else {}

        # Combine all sections
        atmospheric_forcing_data = {
            "AtmosphericForcing": {
                "filename": self.filename,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "model_reference_date": self.model_reference_date.isoformat(),
                "source": self.source,
                "use_coarse_grid": self.use_coarse_grid,
            }
        }

        # Merge YAML data while excluding empty sections
        yaml_data = {
            **grid_yaml_data,
            **swr_correction_yaml_data,
            **rivers_yaml_data,
            **atmospheric_forcing_data,
        }

        with open(filepath, "w") as file:
            # Write header
            file.write(header)
            # Write YAML data
            yaml.dump(yaml_data, file, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filepath: str) -> "AtmosphericForcing":
        """
        Create an instance of the AtmosphericForcing class from a YAML file.

        Parameters
        ----------
        filepath : str
            The path to the YAML file from which the parameters will be read.

        Returns
        -------
        AtmosphericForcing
            An instance of the AtmosphericForcing class.
        """
        # Read the entire file content
        with open(filepath, "r") as file:
            file_content = file.read()

        # Split the content into YAML documents
        documents = list(yaml.safe_load_all(file_content))

        swr_correction_data = None
        rivers_data = None
        atmospheric_forcing_data = None

        # Process the YAML documents
        for doc in documents:
            if doc is None:
                continue
            if "AtmosphericForcing" in doc:
                atmospheric_forcing_data = doc["AtmosphericForcing"]
            if "SWRCorrection" in doc:
                swr_correction_data = doc["SWRCorrection"]
            if "Rivers" in doc:
                rivers_data = doc["Rivers"]

        if atmospheric_forcing_data is None:
            raise ValueError(
                "No AtmosphericForcing configuration found in the YAML file."
            )

        # Convert from string to datetime
        for date_string in ["model_reference_date", "start_time", "end_time"]:
            atmospheric_forcing_data[date_string] = datetime.fromisoformat(
                atmospheric_forcing_data[date_string]
            )

        # Create Grid instance from the YAML file
        grid = Grid.from_yaml(filepath)

        if swr_correction_data is not None:
            swr_correction = SWRCorrection.from_yaml(filepath)
        else:
            swr_correction = None

        if rivers_data is not None:
            rivers = Rivers.from_yaml(filepath)
        else:
            rivers = None

        # Create and return an instance of AtmosphericForcing
        return cls(
            grid=grid,
            swr_correction=swr_correction,
            rivers=rivers,
            **atmospheric_forcing_data,
        )
