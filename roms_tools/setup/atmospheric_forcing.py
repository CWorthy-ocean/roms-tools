import xarray as xr
import dask
from dataclasses import dataclass, field
from roms_tools.setup.grid import Grid
from datetime import datetime
import glob
import numpy as np
from typing import Optional, Dict
from roms_tools.setup.fill import lateral_fill
import calendar


@dataclass(frozen=True, kw_only=True)
class ForcingDataset:
    """
    Represents forcing data on original grid.

    Parameters
    ----------
    filename : str
        The path to the data files. Can contain wildcards.
    start_time: datetime
        The start time for selecting relevant data.
    end_time: datetime
        The end time for selecting relevant data.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the forcing data on its original grid.

    Examples
    --------
    >>> dataset = ForcingDataset(
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
    start_time: datetime
    end_time: datetime
    dim_names: Dict[str, str] = field(
        default_factory=lambda: {"longitude": "lon", "latitude": "lat", "time": "time"}
    )

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        ds = self.load_data()

        # Select relevant times
        times = (np.datetime64(self.start_time) < ds[self.dim_names["time"]]) & (
            ds[self.dim_names["time"]] < np.datetime64(self.end_time)
        )
        ds = ds.where(times, drop=True)

        # Make sure that latitude is ascending
        diff = np.diff(ds[self.dim_names["latitude"]])
        if np.all(diff < 0):
            ds = ds.isel(**{self.dim_names["latitude"]: slice(None, None, -1)})

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
            # initially, we wawnt time chunk size of 1 to enable quick .nan_check() and .plot() methods for AtmosphericForcing
            ds = xr.open_mfdataset(
                self.filename,
                combine="nested",
                concat_dim=self.dim_names["time"],
                coords="minimal",
                compat="override",
                chunks={self.dim_names["time"]: 1},
            )

        return ds

    def choose_subdomain(self, latitude_range, longitude_range, margin, straddle):
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

        Returns
        -------
        xr.Dataset
            The subset of the original dataset representing the chosen subdomain, including an extended area
            to cover one extra grid point beyond the specified ranges.

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

        # Set the modified subdomain to the object attribute
        object.__setattr__(self, "ds", subdomain)


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
        Source of the atmospheric forcing data. Default is "era5".
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

    Notes
    -----
    This class represents atmospheric forcing data used in ocean modeling. It provides a convenient
    interface to work with forcing data including shortwave radiation correction and river forcing.

    Examples
    --------
    >>> grid_info = Grid(...)
    >>> start_time = datetime(2000, 1, 1)
    >>> end_time = datetime(2000, 1, 2)
    >>> atm_forcing = AtmosphericForcing(
    ...     grid=grid_info,
    ...     start_time=start_time,
    ...     end_time=end_time,
    ...     source="era5",
    ...     filename="atmospheric_data_*.nc",
    ...     swr_correction=swr_correction,
    ... )
    """

    grid: Grid
    use_coarse_grid: bool = False
    start_time: datetime
    end_time: datetime
    model_reference_date: datetime = datetime(2000, 1, 1)
    source: str = "era5"
    filename: str
    swr_correction: Optional["SWRCorrection"] = None
    rivers: Optional["Rivers"] = None
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

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

        if self.source == "era5":
            dims = {"longitude": "longitude", "latitude": "latitude", "time": "time"}

        data = ForcingDataset(
            filename=self.filename,
            start_time=self.start_time,
            end_time=self.end_time,
            dim_names=dims,
        )

        # operate on longitudes between -180 and 180 unless ROMS domain lies at least 5 degrees in lontitude away from Greenwich meridian
        lon = xr.where(lon > 180, lon - 360, lon)
        straddle = True
        if not self.grid.straddle and abs(lon).min() > 5:
            lon = xr.where(lon < 0, lon + 360, lon)
            straddle = False

        # Step 1: Choose subdomain of forcing data including safety margin for interpolation, and Step 2: Convert to the proper longitude range.
        # Step 1 is necessary to avoid discontinuous longitudes that could be introduced by Step 2.
        # Discontinuous longitudes can lead to artifacts in the interpolation process. Specifically, if there is a data gap,
        # discontinuous longitudes could result in values that appear to come from a distant location instead of producing NaNs.
        # These NaNs are important as they can be identified and handled appropriately by the nan_check function.
        data.choose_subdomain(
            latitude_range=[lat.min().values, lat.max().values],
            longitude_range=[lon.min().values, lon.max().values],
            margin=2,
            straddle=straddle,
        )

        # interpolate onto desired grid
        if self.source == "era5":
            mask = xr.where(data.ds["sst"].isel(time=0).isnull(), 0, 1)
            varnames = {
                "u10": "u10",
                "v10": "v10",
                "swr": "ssr",
                "lwr": "strd",
                "t2m": "t2m",
                "d2m": "d2m",
                "rain": "tp",
            }

        coords = {dims["latitude"]: lat, dims["longitude"]: lon}
        u10 = self._interpolate(
            data.ds[varnames["u10"]], mask, coords=coords, method="linear"
        )
        v10 = self._interpolate(
            data.ds[varnames["v10"]], mask, coords=coords, method="linear"
        )
        swr = self._interpolate(
            data.ds[varnames["swr"]], mask, coords=coords, method="linear"
        )
        lwr = self._interpolate(
            data.ds[varnames["lwr"]], mask, coords=coords, method="linear"
        )
        t2m = self._interpolate(
            data.ds[varnames["t2m"]], mask, coords=coords, method="linear"
        )
        d2m = self._interpolate(
            data.ds[varnames["d2m"]], mask, coords=coords, method="linear"
        )
        rain = self._interpolate(
            data.ds[varnames["rain"]], mask, coords=coords, method="linear"
        )

        if self.source == "era5":
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
            corr_factor = self._interpolate(
                corr_factor, mask, coords=coords_correction, method="linear"
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

        ds.attrs["Title"] = "ROMS bulk surface forcing file produced by roms-tools"

        ds = ds.assign_coords({"lon": lon, "lat": lat})
        if dims["time"] != "time":
            ds = ds.rename({dims["time"]: "time"})
        if self.use_coarse_grid:
            ds = ds.rename({"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"})
            mask_roms = self.grid.ds["mask_coarse"].rename(
                {"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"}
            )
        else:
            mask_roms = self.grid.ds["mask_rho"]

        object.__setattr__(self, "ds", ds)

        self.nan_check(mask_roms, time=0)

    @staticmethod
    def _interpolate(field, mask, coords, method="linear"):
        """
        Interpolate a field using specified coordinates and a given method.

        Parameters
        ----------
        field : xr.DataArray
            The data array to be interpolated.

        mask : xr.DataArray
            A data array with same spatial dimensions as `field`, where `1` indicates wet (ocean)
            points and `0` indicates land points.

        coords : dict
            A dictionary specifying the target coordinates for interpolation. The keys
            should match the dimensions of `field` (e.g., {"longitude": lon_values, "latitude": lat_values}).

        method : str, optional, default='linear'
            The interpolation method to use. Valid options are those supported by
            `xarray.DataArray.interp`.

        Returns
        -------
        xr.DataArray
            The interpolated data array.

        Notes
        -----
        This method first sets land values to NaN based on the provided mask. It then uses the
        `lateral_fill` function to propagate ocean values. These two steps serve the purpose to
        avoid interpolation across the land-ocean boundary. Finally, it performs interpolation
        over the specified coordinates.

        """

        dims = list(coords.keys())

        # set land values to nan
        field = field.where(mask)
        # propagate ocean values into land interior before interpolation
        field = lateral_fill(field, 1 - mask, dims)
        # interpolate
        field_interpolated = field.interp(**coords, method=method).drop_vars(dims)

        return field_interpolated

    def nan_check(self, mask, time=0) -> None:
        """
        Checks for NaN values at wet points in all variables of the dataset at a specified time step.

        Parameters
        ----------
        mask : array-like
            A boolean mask indicating the wet points in the dataset.
        time : int
            The time step at which to check for NaN values. Default is 0.

        Raises
        ------
        ValueError
            If any variable contains NaN values at the specified time step.

        """

        for var in self.ds.data_vars:
            da = xr.where(mask == 1, self.ds[var].isel(time=time), 0)
            if da.isnull().any().values:
                raise ValueError(
                    f"NaN values found in the variable '{var}' at time step {time} over the ocean. This likely "
                    "occurs because the ROMS grid, including a small safety margin for interpolation, is not "
                    "fully contained within the dataset's longitude/latitude range. Please ensure that the "
                    "dataset covers the entire area required by the ROMS grid."
                )

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
        ...     source="era5",
        ...     filename="atmospheric_data_*.nc",
        ...     swr_correction=swr_correction,
        ... )
        >>> atm_forcing.plot("uwnd", time=0)
        """

        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        lon_deg = self.ds.lon
        lat_deg = self.ds.lat

        # check if North or South pole are in domain
        if lat_deg.max().values > 89 or lat_deg.min().values < -89:
            raise NotImplementedError(
                "Plotting the atmospheric forcing is not implemented for the case that the domain contains the North or South pole."
            )

        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)

        # Define projections
        proj = ccrs.PlateCarree()

        trans = ccrs.NearsidePerspective(
            central_longitude=lon_deg.mean().values,
            central_latitude=lat_deg.mean().values,
        )

        lon_deg = lon_deg.values
        lat_deg = lat_deg.values

        # find corners
        (lo1, la1) = (lon_deg[0, 0], lat_deg[0, 0])
        (lo2, la2) = (lon_deg[0, -1], lat_deg[0, -1])
        (lo3, la3) = (lon_deg[-1, -1], lat_deg[-1, -1])
        (lo4, la4) = (lon_deg[-1, 0], lat_deg[-1, 0])

        # transform coordinates to projected space
        lo1t, la1t = trans.transform_point(lo1, la1, proj)
        lo2t, la2t = trans.transform_point(lo2, la2, proj)
        lo3t, la3t = trans.transform_point(lo3, la3, proj)
        lo4t, la4t = trans.transform_point(lo4, la4, proj)

        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=trans)

        ax.plot(
            [lo1t, lo2t, lo3t, lo4t, lo1t],
            [la1t, la2t, la3t, la4t, la1t],
            "go-",
        )

        ax.coastlines(
            resolution="50m", linewidth=0.5, color="black"
        )  # add map of coastlines
        ax.gridlines()

        field = self.ds[varname].isel(time=time).compute()
        if varname in ["uwnd", "vwnd"]:
            vmax = max(field.max().values, -field.min().values)
            vmin = -vmax
            cmap = "RdBu_r"
        else:
            vmax = field.max().values
            vmin = field.min().values
            if varname in ["swrad", "lwrad", "Tair", "qair"]:
                cmap = "YlOrRd"
            else:
                cmap = "YlGnBu"

        p = ax.pcolormesh(
            lon_deg, lat_deg, field, transform=proj, vmax=vmax, vmin=vmin, cmap=cmap
        )
        plt.colorbar(p, label=field.units)
        ax.set_title(
            "%s at time %s"
            % (field.long_name, np.datetime_as_string(field.time, unit="s"))
        )
        plt.show()

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
        gb = self.ds.groupby("time.year")

        for year, group_ds in gb:
            # Further group each yearly group by month
            sub_gb = group_ds.groupby("time.month")

            for month, ds in sub_gb:
                # Chunk the dataset by the specified time chunk size
                ds = ds.chunk({"time": time_chunk_size})
                datasets.append(ds)

                # Determine the number of days in the month
                num_days_in_month = calendar.monthrange(year, month)[1]
                first_day = ds.time.dt.day.values[0]
                last_day = ds.time.dt.day.values[-1]

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

            # Translate the time coordinate to days since the model reference date
            model_reference_date = np.datetime64(self.model_reference_date)

            # Preserve the original time coordinate for readability
            ds["Time"] = ds["time"]

            # Convert the time coordinate to the format expected by ROMS (days since model reference date)
            ds["time"] = (
                (ds["time"] - model_reference_date).astype("float64") / 3600 / 24 * 1e-9
            )
            ds["time"].attrs[
                "long_name"
            ] = f"time since {np.datetime_as_string(model_reference_date, unit='D')}"

            # Prepare the dataset for writing to a netCDF file without immediately computing
            write = ds.to_netcdf(filename, compute=False)
            writes.append(write)

        # Perform the actual write operations in parallel
        dask.compute(*writes)
