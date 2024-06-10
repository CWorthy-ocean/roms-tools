import xarray as xr
import dask
from dataclasses import dataclass, field
from roms_tools.setup.grid import Grid
from datetime import datetime, timedelta
import glob
import numpy as np
from typing import Optional, Dict, Union
from scipy.sparse import spdiags, coo_matrix
from scipy.sparse.linalg import spsolve
from roms_tools.setup.fill import lateral_fill

def choose_subdomain(ds, dim_names, latitude_range, longitude_range) -> xr.Dataset:
    """
    Choose a subdomain from the given xarray Dataset based on latitude and longitude ranges,
    including one extra grid point beyond the specified ranges.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the data.
    dim_names : dict
        A dictionary containing the dimension names for latitude and longitude.
    latitude_range : tuple
        A tuple specifying the minimum and maximum latitude values of the subdomain.
    longitude_range : tuple
        A tuple specifying the minimum and maximum longitude values of the subdomain.

    Returns
    -------
    xr.Dataset
        The subset of the original dataset representing the chosen subdomain.

    Raises
    ------
    ValueError
        If the specified subdomain is not fully contained within the dataset.
    """

    lat_min, lat_max = latitude_range
    lon_min, lon_max = longitude_range

    # Get latitude and longitude values
    lats = ds[dim_names["latitude"]]
    lons = ds[dim_names["longitude"]]

    # Find indices for the specified ranges
    lat_indices = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_indices = np.where((lons >= lon_min) & (lons <= lon_max))[0]

    if lat_indices.size == 0 or lon_indices.size == 0:
        raise ValueError("The specified ranges do not overlap with the dataset.")

    # Extend the range by two grid point on each side to interpolate safely
    lat_min_index = lat_indices[0] - 2
    lat_max_index = min(len(lats) - 1, lat_indices[-1] + 2)
    lon_min_index = max(0, lon_indices[0] - 2)
    lon_max_index = min(len(lons) - 1, lon_indices[-1] + 2)
    
    if lat_indices[0] - 2 < 0:
        ValueError

    # Select the subdomain based on the extended indices
    subdomain = ds.isel(
        **{
            dim_names["latitude"]: slice(lat_min_index, lat_max_index + 1),
            dim_names["longitude"]: slice(lon_min_index, lon_max_index + 1)
        }
    )

    # Check if the selected subdomain fully contains the specified ranges
    dataset_lat_min = subdomain[dim_names["latitude"]].min().values
    dataset_lat_max = subdomain[dim_names["latitude"]].max().values
    dataset_lon_min = subdomain[dim_names["longitude"]].min().values
    dataset_lon_max = subdomain[dim_names["longitude"]].max().values

    if (lat_min < dataset_lat_min or lat_max > dataset_lat_max or
            lon_min < dataset_lon_min or lon_max > dataset_lon_max):
        raise ValueError(
            f"The specified subdomain is not fully contained within the dataset.\n"
            f"Dataset latitude range: ({dataset_lat_min}, {dataset_lat_max})\n"
            f"Dataset longitude range: ({dataset_lon_min}, {dataset_lon_max})\n"
            f"Specified latitude range: ({lat_min}, {lat_max})\n"
            f"Specified longitude range: ({lon_min}, {lon_max})"
        )
    
    return subdomain

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
    chunks : dict, optional
        Dictionary specifying chunk sizes for dask.
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the forcing data on its original grid.

    Examples
    --------
    >>> dataset = ForcingDataset(filename="data.nc", start_time=datetime(2022, 1, 1), end_time=datetime(2022, 12, 31))
    >>> dataset.load_data()
    >>> print(dataset.ds)
    <xarray.Dataset>
    Dimensions:  ...
    """

    filename: str
    start_time: datetime
    end_time: datetime
    chunks: Dict[str, Union[int, Dict[str, int]]] = field(default_factory=lambda: {"time": 1})
    #chunks: dict = field(default_factory=lambda: {"time": 1})
    dim_names: Dict[str, str] = field(default_factory=lambda: {"longitude": "lon", "latitude": "lat", "time": "time"})

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        ds = self.load_data()
        
        # Select relevant times
        times = (np.datetime64(self.start_time) < ds[self.dim_names["time"]]) & (ds[self.dim_names["time"]] < np.datetime64(self.end_time))
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
            raise FileNotFoundError(f"No files found matching the pattern '{self.filename}'.")

        # Load the dataset
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            ds = xr.open_mfdataset(self.filename, combine='nested', concat_dim=self.dim_names["time"], chunks=self.chunks)

        return ds

    def select_subdomain(self, latitude_range, longitude_range):
        ds = choose_subdomain(self.ds, self.dim_names, latitude_range, longitude_range)
        object.__setattr__(self, "ds", ds)


#@dataclass(frozen=True, kw_only=True)
#class SWRCorrection:
#    """
#    Configuration for shortwave radiation correction.
#
#    Parameters
#    ----------
#    filename : str
#        Filename of the correction data.
#    varname : str
#        Variable identifier for the correction.
#    temporal_resolution : str, optional
#        Temporal resolution of the correction data. Default is "climatology".
#    spatial_coverage : str, optional
#        Spatial coverage of the correction data. Default is "global".
#
#    Attributes
#    ----------
#    filename : str
#        Filename of the correction data.
#    varname : str
#        Variable identifier for the correction.
#    temporal_resolution : str
#        Temporal resolution of the correction data.
#    spatial_coverage : str
#        Spatial coverage of the correction data.
#    da : xr.DataArray
#        The loaded xarray DataArray containing the correction data.
#
#    Examples
#    --------
#    config = SWRCorrection(filename="correction_data.nc", varname="swr", temporal_resolution="climatology", spatial_coverage="global")
#    print(config.var)
#    <xarray.DataArray 'swr' (time: 12, lat: 180, lon: 360)>
#    array([[[...]]])
#
#    """
#
#    filename: str
#    temporal_resolution: str = "climatology"
#    spatial_coverage: str = "global"
#    da: xr.DataArray = field(init=False, repr=False)
#
#    def __post_init__(self):
#        if self.temporal_resolution != "climatology":
#            raise NotImplementedError(f"temporal_resolution must be 'climatology', got {self.temporal_resolution}")
#        if self.spatial_coverage != "global":
#            raise NotImplementedError(f"spatial_coverage must be 'global', got {self.spatial_coverage}")
#
#        da = self.load_data(self.filename)
#        object.__setattr__(self, "da", da)
#
#    @staticmethod
#    def load_data(filename, varname):
#        """
#        Load data from the specified file.
#
#        Parameters
#        ----------
#        filename : str
#            The path to the correction dataset.
#        varname : str
#            The variable identifier for the correction.
#
#        Returns
#        -------
#        da : xr.DataArray
#            The loaded xarray DataArray containing the correction data.
#
#        Raises
#        ------
#        FileNotFoundError
#            If the specified file does not exist.
#
#        """
#        # Check if the file exists
#
#        # Check if any file matching the wildcard pattern exists
#        matching_files = glob.glob(filename)
#        if not matching_files:
#            raise FileNotFoundError(f"No files found matching the pattern '{filename}'.")
#
#        # Load the dataset
#        ds = xr.open_dataset(filename)
#        da = ds[varname]
#        lon = ds[lonname]
#        lat = ds[latname]
#
#        return da, lon, lat
#    
#    def interpolate_spatially(self, coords):
#        """
#        Interpolate correction in space.
#    
#        Parameters
#        ----------
#        coords : dict
#            A dictionary specifying the spatial target coordinates for interpolation, e.g., 
#            {"latitude": lat_values, "longitude": lon_values}.
#
#
#        Returns
#        -------
#        swr_corrected : xr.DataArray
#            Corrected shortwave radiation values.
#    
#        Raises
#        ------
#        ValueError
#            - If the correction dataset dimensions do not match expectations (time, longitude, latitude).
#            - If the temporal dimension of the correction dataset does not have length 12, assuming monthly climatology.
#
#        Notes
#        -----
#        This function performs both spatial and temporal interpolation to align the correction
#        data with the input radiation data grid and time points, respectively. The corrected radiation values are
#        then obtained by multiplying the input radiation data with the interpolated correction factors.
#    
#        Examples
#        --------
#        >>> corrected_swr = correct_shortwave_radiation("correction_data.nc", swr_data, lon, lat)
#        """
#
#        dims = list(coords.keys())
#
#        # Spatial interpolation
#        lon_min = self.lon.min().values
#        lon_max = self.lon.max().values
#        if lon_min > 0.0 and lon_max < 365.0:
#            da = concatenate_across_dateline(self.da, end='both')
#        elif lon_min > 0.0:
#            da = concatenate_across_dateline(self.da, end='lower')
#        else:
#            da = concatenate_across_dateline(self.da, end='upper')
#        # set land values to nan
#        # field = field.where(mask)
#        # propagate ocean values into land interior before interpolation
#        #field = lateral_fill(field, 1-mask, dims)
#        # interpolate
#
#        da_interpolated = da.interp(**coords, method='nearest').drop_vars(dims)
#        
#    
#        return da_interpolated
#
#
#    def interpolate_spatially(self, coords):
#        """
#        Interpolate correction in space. This method assumes that the correction dataset contains a
#        climatology with 12 time entries and global spatial coverage.
#    
#        Parameters
#        ----------
#        coords : dict
#            A dictionary specifying the spatial target coordinates for interpolation, e.g., 
#            {"time": time_values, "longitude": lon_values, "latitude": lat_values}.
#
#
#        Returns
#        -------
#        swr_corrected : xr.DataArray
#            Corrected shortwave radiation values.
#    
#        Raises
#        ------
#        ValueError
#            - If the correction dataset dimensions do not match expectations (time, longitude, latitude).
#            - If the temporal dimension of the correction dataset does not have length 12, assuming monthly climatology.
#
#        Notes
#        -----
#        This function performs both spatial and temporal interpolation to align the correction
#        data with the input radiation data grid and time points, respectively. The corrected radiation values are
#        then obtained by multiplying the input radiation data with the interpolated correction factors.
#    
#        Examples
#        --------
#        >>> corrected_swr = correct_shortwave_radiation("correction_data.nc", swr_data, lon, lat)
#        """
#
#        dims = list(coords.keys())
#
#        # Spatial interpolation
#        lon_min = ds_correction.longitude.min().values
#        lon_max = ds_correction.longitude.max().values
#        if lon_min > 0.0 and lon_max < 365.0:
#            corr_factor = concatenate_across_dateline(ds_correction["ssr_corr"], end='both')
#        elif lon_min > 0.0:
#            corr_factor = concatenate_across_dateline(ds_correction["ssr_corr"], end='lower')
#        else:
#            corr_factor = concatenate_across_dateline(ds_correction["ssr_corr"], end='upper')
#
#        corr_factor = corr_factor.interp(longitude=lon, latitude=lat, method='nearest').drop_vars(["longitude", "latitude"])
#        
#        # Temporal interpolation
#        corr_factor["time"] = corr_factor.time.dt.days
#        swr["day_of_year"] = swr.time.dt.dayofyear
#        # Concatenate across the beginning and end of the year
#        time = xr.concat([corr_factor.time[-1] - 365.25, corr_factor.time, 365.25 + corr_factor.time[0]], dim="time")
#        corr_factor = xr.concat([corr_factor.isel(time=-1), corr_factor, corr_factor.isel(time=0)], dim="time")
#        corr_factor["time"] = time
#        # Interpolate correction data to ERA5 times
#        corr_factor = corr_factor.interp(time=swr.day_of_year, method='linear')    
#        
#        # Apply correction
#        swr_corrected = swr * corr_factor
#    
#        return swr_corrected
#    def concatenate_across_dateline(field, end):
#        """
#        Concatenates a field across the dateline based on the specified end.
#    
#        Parameters
#        ----------
#        field : xr.DataArray
#            Input field to be concatenated.
#        end : {'upper', 'lower', 'both'}
#            Specifies which end of the dateline to concatenate the field.
#            - 'upper': Concatenate on the upper end.
#            - 'lower': Concatenate on the lower end.
#            - 'both': Concatenate on both ends.
#    
#        Returns
#        -------
#        field_concatenated : xr.DataArray
#            Concatenated field along the longitude axis.
#        """
#        lon = field['longitude']
#    
#        if end == 'upper':
#            lon_plus360 = lon + 360
#            lon_concatenated = xr.concat([lon, lon_plus360], dim="longitude")
#            field_concatenated = xr.concat([field, field], dim="longitude")
#        elif end == 'lower':
#            lon_minus360 = lon - 360
#            lon_concatenated = xr.concat([lon_minus360, lon], dim="longitude")
#            field_concatenated = xr.concat([field, field], dim="longitude")
#        elif end == 'both':
#            lon_minus360 = lon - 360
#            lon_plus360 = lon + 360
#            lon_concatenated = xr.concat([lon_minus360, lon, lon_plus360], dim="longitude")
#            field_concatenated = xr.concat([field, field, field], dim="longitude")
#    
#        field_concatenated["longitude"] = lon_concatenated
#    
#        return field_concatenated

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
    >>> swr_correction = AtmosphericForcing.SWRCorrection(filename="correction_data.nc")
    >>> atm_forcing = AtmosphericForcing(grid=grid_info, start_time=start_time, end_time=end_time, source='era5', filename='atmospheric_data_*.nc', swr_correction=swr_correction)
    """

    grid: Grid
    use_coarse_grid: bool = False
    start_time: datetime
    end_time: datetime
    model_reference_date: datetime = datetime(2000, 1, 1)
    source: str = "era5"
    filename: str
    swr_correction: Optional['SWRCorrection'] = None
    rivers: Optional['Rivers'] = None
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        
        if self.use_coarse_grid:
            if 'lon_coarse' not in self.grid.ds:
                raise ValueError('Grid has not been coarsened yet. Execute grid.coarsen() first.')

            lon = self.grid.ds.lon_coarse
            lat = self.grid.ds.lat_coarse
            angle = self.grid.ds.angle_coarse
        else:
            lon = self.grid.ds.lon_rho
            lat = self.grid.ds.lat_rho
            angle = self.grid.ds.angle
        
        if self.source == "era5":
            dims = {"longitude": "longitude", "latitude": "latitude", "time": "time"}

        data = ForcingDataset(filename=self.filename, start_time=self.start_time, end_time=self.end_time, dim_names=dims)

        
        if self.grid.straddle:
            lon = xr.where(lon > 180, lon - 360, lon)
            data.ds[dims['longitude']] = xr.where(data.ds[dims['longitude']] > 180, data.ds[dims['longitude']] - 360, data.ds[dims['longitude']])
        else:
            lon = xr.where(lon < 0, lon + 360, lon)
            data.ds[dims['longitude']] = xr.where(data.ds[dims['longitude']] < 0, data.ds[dims['longitude']] + 360, data.ds[dims['longitude']])

        data.select_subdomain(latitude_range=[lat.min().values, lat.max().values], longitude_range=[lon.min().values, lon.max().values])

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
                "rain": "tp"
            }

        coords={dims["latitude"]: lat, dims["longitude"]: lon}

        u10 = self.interpolate(data.ds[varnames["u10"]], mask, coords=coords, method='nearest')
        v10 = self.interpolate(data.ds[varnames["v10"]], mask, coords=coords, method='nearest')
        swr = self.interpolate(data.ds[varnames["swr"]], mask, coords=coords, method='linear')
        lwr = self.interpolate(data.ds[varnames["lwr"]], mask, coords=coords, method='linear')
        t2m = self.interpolate(data.ds[varnames["t2m"]], mask, coords=coords, method='linear')
        d2m = self.interpolate(data.ds[varnames["d2m"]], mask, coords=coords, method='linear')
        rain = self.interpolate(data.ds[varnames["rain"]], mask, coords=coords, method='linear')
            
        if self.source == "era5":
            # translate radiation to fluxes. ERA5 stores values integrated over 1 hour.
            swr = swr / 3600  # from J/m^2 to W/m^2
            lwr = lwr / 3600  # from J/m^2 to W/m^2
            rain = rain * 100 * 24  # from m to cm/day
            # convert from K to C
            t2m = t2m - 273.15
            d2m = d2m - 273.15
            # relative humidity fraction
            qair = np.exp((17.625*d2m)/(243.04+d2m)) / np.exp((17.625*t2m)/(243.04+t2m))
            # convert relative to absolute humidity assuming constant pressure
            patm = 1010.0
            cff=(1.0007+3.46e-6*patm)*6.1121 *np.exp(17.502*t2m/(240.97+t2m))
            cff = cff * qair
            qair = 0.62197 *(cff /(patm-0.378*cff))
            
        # correct shortwave radiation
        if self.swr_correction:
            swr = era5.correct_shortwave_radiation(self.swr_correction.filename, swr, lon, lat)

        if self.rivers:
            NotImplementedError("River forcing is not implemented yet.")
            # rain = rain + rivers

        # save in new dataset
        ds = xr.Dataset()

        ds["uwnd"] = (u10 * np.cos(angle) + v10 * np.sin(angle)).astype(np.float32)  # rotate to grid orientation
        ds["uwnd"].attrs["long_name"] = "10 meter wind in x-direction"
        ds["uwnd"].attrs["units"] = "m/s"

        ds["vwnd"] = (v10 * np.cos(angle) - u10 * np.sin(angle)).astype(np.float32)  # rotate to grid orientation
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
        if self.use_coarse_grid:
            ds = ds.rename({"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"})

        object.__setattr__(self, "ds", ds)
    

    @staticmethod
    def interpolate(field, mask, coords, method='linear'):
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
        field = lateral_fill(field, 1-mask, dims)
        # interpolate
        field_interpolated = field.interp(**coords, method=method).drop_vars(dims)

        return field_interpolated


    def plot(self, field, time=0) -> None:
        """
        Plot the specified atmospheric forcing field for a given time slice.
    
        Parameters
        ----------
        field : str
            The atmospheric forcing field to plot. Options include:
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
            If the specified field is not one of the valid options.
    
        Notes
        -----
        The `cartopy` and `matplotlib` libraries are required to use this method. Ensure 
        these libraries are installed in your environment.
    
        Examples
        --------
        >>> atm_forcing = AtmosphericForcing(grid=grid_info, start_time=start_time, end_time=end_time, source='era5', filename='atmospheric_data_*.nc', swr_correction=swr_correction)
        >>> atm_forcing.plot("uwnd", time=0)
        """

        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt


        lon_deg = self.ds.lon
        lat_deg = self.ds.lat

        # check if North or South pole are in domain
        if lat_deg.max().values > 89 or lat_deg.min().values < -89:
            raise NotImplementedError("Plotting the atmospheric forcing is not implemented for the case that the domain contains the North or South pole.")

        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)

        # Define projections
        proj = ccrs.PlateCarree()

        trans = ccrs.NearsidePerspective(
                central_longitude=lon_deg.mean().values, central_latitude=lat_deg.mean().values
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

        if field in ["uwnd", "vwnd"]:
            vmax = max(self.ds[field].isel(time=time).max().values, -self.ds[field].isel(time=time).min().values)
            vmin = -vmax
            cmap = "RdBu_r"
        else:
            vmax = self.ds[field].isel(time=time).max().values
            vmin = self.ds[field].isel(time=time).min().values
            if field in ["swrad", "lwrad", "Tair", "qair"]:
                cmap = "YlOrRd"
            else:
                cmap = "YlGnBu"

        p = ax.pcolormesh(
                    lon_deg, lat_deg,
                    self.ds[field].isel(time=time),
                    transform=proj,
                    vmax=vmax, vmin=vmin,
                    cmap=cmap
            )
        plt.colorbar(p, label="%s [%s]" %(self.ds[field].long_name, self.ds[field].units))
        plt.show()


    def save(self, filepath: str) -> None:
        """
        Save the atmospheric forcing information to a netCDF4 file.

        Parameters
        ----------
        filepath
        """

        datasets = []
        filenames = []
        writes = []

        gb = self.ds.groupby("time.year")

        for year, group_ds in gb:
            sub_gb = group_ds.groupby("time.month")

            for month, ds in sub_gb:
                
                datasets.append(ds)    
                
                year_month_str = f"{year}{month:02}"
                filename = "%s.%s.nc" %(filepath, year_month_str)
                filenames.append(filename)

        
        for ds, filename in zip(datasets, filenames):

            # translate to days since model reference date
            model_reference_date = np.datetime64(self.model_reference_date)
            ds["time"] = (ds["time"] - model_reference_date).astype('float64') / 3600 / 24 * 1e-9
            ds["time"].attrs["long_name"] = f"time since {np.datetime_as_string(model_reference_date, unit='D')}"

            write = ds.to_netcdf(filename, compute=False)
            writes.append(write)

        dask.compute(*writes)


