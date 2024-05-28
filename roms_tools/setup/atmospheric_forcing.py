import xarray as xr
import dask
from dataclasses import dataclass, field
from roms_tools.setup.grid import Grid
from datetime import datetime
import glob
import numpy as np
from typing import Optional


def concatenate_across_dateline(field, end):
    """
    Concatenates a field across the dateline based on the specified end.

    Parameters
    ----------
    field : xr.DataArray
        Input field to be concatenated.
    end : {'upper', 'lower', 'both'}
        Specifies which end of the dateline to concatenate the field.
        - 'upper': Concatenate on the upper end.
        - 'lower': Concatenate on the lower end.
        - 'both': Concatenate on both ends.

    Returns
    -------
    field_concatenated : xr.DataArray
        Concatenated field along the longitude axis.
    """
    lon = field['longitude']

    if end == 'upper':
        lon_plus360 = lon + 360
        lon_concatenated = xr.concat([lon, lon_plus360], dim="longitude")
        field_concatenated = xr.concat([field, field], dim="longitude")
    elif end == 'lower':
        lon_minus360 = lon - 360
        lon_concatenated = xr.concat([lon_minus360, lon], dim="longitude")
        field_concatenated = xr.concat([field, field], dim="longitude")
    elif end == 'both':
        lon_minus360 = lon - 360
        lon_plus360 = lon + 360
        lon_concatenated = xr.concat([lon_minus360, lon, lon_plus360], dim="longitude")
        field_concatenated = xr.concat([field, field, field], dim="longitude")

    field_concatenated["longitude"] = lon_concatenated

    return field_concatenated

@dataclass(frozen=True, kw_only=True)
class ERA5:
    """
    Represents ERA5 data.

    Parameters
    ----------
    filename : str
        The path to the ERA5 files.
    start_time: datetime
    end_time: datetime

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing ERA5 data.


    Examples
    --------
    >>> era5 = ERA5()
    >>> era5.load_data("era5_data.nc")
    >>> print(era5.ds)
    <xarray.Dataset>
    Dimensions:  ...
    """

    filename: str
    start_time: datetime
    end_time: datetime
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        ds = self.load_data(self.filename)

        # select relevant times
        times = (np.datetime64(self.start_time) < ds.time) & (ds.time < np.datetime64(self.end_time))
        ds = ds.where(times, drop=True)

        ds['longitude'] = xr.where(ds.longitude <= 0, ds.longitude + 360, ds.longitude)
        
        object.__setattr__(self, "ds", ds)


    @staticmethod
    def load_data(filename):
        """
        Load tidal forcing data from the specified file.

        Parameters
        ----------
        filename : str
            The path to the tidal dataset file.

        Returns
        -------
        ds : xr.Dataset
            The loaded xarray Dataset containing the tidal forcing data.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.

        """
        # Check if the file exists

        # Check if any file matching the wildcard pattern exists
        matching_files = glob.glob(filename)
        if not matching_files:
            raise FileNotFoundError(f"No files found matching the pattern '{filename}'.")

        # Load the dataset
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            ds = xr.open_mfdataset(filename, combine='nested', concat_dim='time', chunks={'time': 10})

        return ds

    @staticmethod
    def correct_shortwave_radiation(filename, swr, grid):
        """
        Apply shortwave radiation correction.
    
        Parameters
        ----------
        filename : str
            Path to the correction dataset.
        swr : xr.DataArray
            Shortwave radiation data to be corrected.
        grid : Grid
            Object containing grid information with latitude and longitude data.
    
        Returns
        -------
        swr_corrected : xr.DataArray
            Corrected shortwave radiation values.
    
        Raises
        ------
        ValueError
            - If the correction dataset dimensions do not match expectations (time, longitude, latitude).
            - If the temporal dimension of the correction dataset does not have length 12, assuming monthly climatology.

        Notes
        -----
        This function corrects shortwave radiation values using correction data provided in the dataset located
        at the specified file path. It performs both spatial and temporal interpolation to align the correction
        data with the input radiation data grid and time points, respectively. The corrected radiation values are
        then obtained by multiplying the input radiation data with the interpolated correction factors.
    
        Examples
        --------
        >>> corrected_swr = correct_shortwave_radiation("correction_data.nc", swr_data, grid_info)
        """
    
        # Open and load the correction dataset
        ds_correction = xr.open_dataset(filename)
        
        # Check if required dimensions are present
        required_dims = {"time", "longitude", "latitude"}
        if not required_dims.issubset(ds_correction.dims):
            raise ValueError(f"The dataset must contain the dimensions: {required_dims}")

        # Check if the time dimension has length 12
        if "time" in ds_correction.dims and len(ds_correction["time"]) != 12:
            raise ValueError("The length of the 'time' dimension must be 12 for monthly climatology.")

        # Spatial interpolation
        lon_min = ds_correction.longitude.min().values
        lon_max = ds_correction.longitude.max().values
        if lon_min > 0.0 and lon_max < 365.0:
            corr_factor = concatenate_across_dateline(ds_correction["ssr_corr"], end='both')
        elif lon_min > 0.0:
            corr_factor = concatenate_across_dateline(ds_correction["ssr_corr"], end='lower')
        else:
            corr_factor = concatenate_across_dateline(ds_correction["ssr_corr"], end='upper')

        corr_factor = corr_factor.interp(longitude=grid.ds.lon_rho, latitude=grid.ds.lat_rho, method='nearest').drop_vars(["longitude", "latitude"])
        
        # Temporal interpolation
        corr_factor["time"] = corr_factor.time.dt.days
        swr["day_of_year"] = swr.time.dt.dayofyear
        # Concatenate across the beginning and end of the year
        time = xr.concat([corr_factor.time[-1] - 365.25, corr_factor.time, 365.25 + corr_factor.time[0]], dim="time")
        corr_factor = xr.concat([corr_factor.isel(time=-1), corr_factor, corr_factor.isel(time=0)], dim="time")
        corr_factor["time"] = time
        # Interpolate correction data to ERA5 times
        corr_factor = corr_factor.interp(time=swr.day_of_year, method='linear')    
        
        # Apply correction
        swr_corrected = swr * corr_factor
    
        return swr_corrected


@dataclass(frozen=True, kw_only=True)
class AtmosphericForcing:
    """
    Represents atmospheric forcing data for ocean modeling.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
    start_time : datetime
        Start time of the forcing data.
    end_time : datetime
        End time of the forcing data.
    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.
    source : str, optional
        Source of the atmospheric forcing data. Default is "era5".
    filename: str
        Path to the atmospheric forcing data file on native grid. Can contain wildcards.
    swr_correction : SWRCorrection
        Shortwave radiation correction configuration.
    rivers : Rivers, optional
        River forcing configuration.

    Attributes
    ----------
    grid : Grid
        Object representing the grid information.
    start_time : datetime
        Start time of the forcing data.
    end_time : datetime
        End time of the forcing data.
    model_reference_date : datetime
        Reference date for the model.
    source : str
        Source of the atmospheric forcing data.
    filename : str
        Path to the atmospheric forcing data file. Can contain wildcards.
    swr_correction : SWRCorrection
        Shortwave radiation correction configuration.
    rivers : Rivers
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
    start_time: datetime
    end_time: datetime
    model_reference_date: datetime = datetime(2000, 1, 1)
    source: str = "era5"
    filename: str
    swr_correction: Optional['SWRCorrection'] = None
    rivers: Optional['Rivers'] = None
    ds: xr.Dataset = field(init=False, repr=False)

    @dataclass(frozen=True, kw_only=True)
    class SWRCorrection:
        """
        Configuration for shortwave radiation correction.

        Parameters
        ----------
        apply : bool, optional
            Flag to apply shortwave radiation correction. Default is False.
        filename : str, optional
            Filename of the correction data.
        """

        apply: bool = False
        filename: str = ""

        def __post_init__(self):
            if not self.filename:
                raise ValueError("The 'filename' must be provided.")


    @dataclass(frozen=True, kw_only=True)
    class Rivers:
        """
        Configuration for river forcing.

        Parameters
        ----------
        apply : bool, optional
            Flag to apply river forcing. Default is False.
        filename : str, optional
            Filename of the river forcing data.
        """

        apply: bool = False
        filename: str = ""
 
        def __post_init__(self):
            if not self.filename:
                raise ValueError("The 'filename' must be provided.")

    def __post_init__(self):
        if self.source == "era5":
            era5 = ERA5(filename=self.filename, start_time=self.start_time, end_time=self.end_time)
            
            # interpolate onto desired grid
            u10 = era5.ds["u10"].interp(longitude=self.grid.ds.lon_rho, latitude=self.grid.ds.lat_rho, method='nearest').drop_vars(["longitude", "latitude"])
            v10 = era5.ds["v10"].interp(longitude=self.grid.ds.lon_rho, latitude=self.grid.ds.lat_rho, method='nearest').drop_vars(["longitude", "latitude"])
            swr = era5.ds["ssr"].interp(longitude=self.grid.ds.lon_rho, latitude=self.grid.ds.lat_rho, method='linear').drop_vars(["longitude", "latitude"])
            lwr = era5.ds["strd"].interp(longitude=self.grid.ds.lon_rho, latitude=self.grid.ds.lat_rho, method='linear').drop_vars(["longitude", "latitude"])
            t2m = era5.ds["t2m"].interp(longitude=self.grid.ds.lon_rho, latitude=self.grid.ds.lat_rho, method='linear').drop_vars(["longitude", "latitude"])
            d2m = era5.ds["d2m"].interp(longitude=self.grid.ds.lon_rho, latitude=self.grid.ds.lat_rho, method='linear').drop_vars(["longitude", "latitude"])
            rain = era5.ds["tp"].interp(longitude=self.grid.ds.lon_rho, latitude=self.grid.ds.lat_rho, method='linear').drop_vars(["longitude", "latitude"])
            
            # translate radiation to fluxes. ERA5 stores values integrated over 1 hour.
            swr = swr / 3600  # from J/m^2 to W/m^2
            lwr = lwr / 3600  # from J/m^2 to W/m^2
            rain = rain * 100 * 24  # from m to cm/day
            # correct shortwave radiation
            if self.swr_correction:
                swr = era5.correct_shortwave_radiation(self.swr_correction.filename, swr, self.grid)

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
    
        if self.rivers and self.rivers.apply:
            NotImplementedError("River forcing is not implemented yet.")
            # rain = rain + rivers

        # save in new dataset
        ds = xr.Dataset()

        ds["uwnd"] = (u10 * np.cos(self.grid.ds.angle) + v10 * np.sin(self.grid.ds.angle)).astype(np.float32)  # rotate to grid orientation
        ds["uwnd"].attrs["long_name"] = "10 meter wind in x-direction"
        ds["uwnd"].attrs["units"] = "m/s"

        ds["vwnd"] = (v10 * np.cos(self.grid.ds.angle) - u10 * np.sin(self.grid.ds.angle)).astype(np.float32)  # rotate to grid orientation
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

        # translate to days since model reference date
        model_reference_date = np.datetime64(self.model_reference_date)
        ds["time"] = (ds["time"] - model_reference_date).astype('float64') / 3600 / 24 * 1e-9
        ds["time"].attrs["long_name"] = f"time since {np.datetime_as_string(model_reference_date, unit='D')}"

        object.__setattr__(self, "ds", ds)

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
        >>> tidal_forcing = TidalForcing(grid)
        >>> tidal_forcing.plot("ssh_Re", nc=0)
        """

        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt


        lon_deg = self.grid.ds["lon_rho"]
        lat_deg = self.grid.ds["lat_rho"]

        # check if North or South pole are in domain
        if lat_deg.max().values > 89 or lat_deg.min().values < -89:
            raise NotImplementedError("Plotting the bathymetry is not implemented for the case that the domain contains the North or South pole. Please set bathymetry to False.")

        # check if Greenwhich meridian goes through domain
        if np.abs(lon_deg.diff('xi_rho')).max() > 300 or np.abs(lon_deg.diff('eta_rho')).max() > 300:
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
        self.ds.to_netcdf(filepath)


