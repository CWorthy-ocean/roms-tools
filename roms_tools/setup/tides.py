from datetime import datetime
import xarray as xr
import numpy as np
from dataclasses import dataclass, field
from roms_tools.setup.grid import Grid
from roms_tools.setup.plot import _plot
from roms_tools.setup.fill import interpolate_from_rho_to_u, interpolate_from_rho_to_v, fill_and_interpolate
from roms_tools.setup.datasets import Dataset
import os
import hashlib
from typing import Dict, Optional, List
import matplotlib.pyplot as plt

@dataclass(frozen=True, kw_only=True)
class TPXO(Dataset):
    """
    Represents tidal data on original grid.

    Parameters
    ----------
    filename : str
        The path to the TPXO dataset.
    var_names : List[str], optional
        List of variable names that are required in the dataset. Defaults to
        ["h_Re", "h_Im", "sal_Re", "sal_Im", "u_Re", "u_Im", "v_Re", "v_Im"].
    dim_names: Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset. Defaults to
        {"longitude": "ny", "latitude": "nx"}.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing TPXO tidal model data.
    """

    filename: str
    var_names: List[str] = field(
        default_factory=lambda: ["h_Re", "h_Im", "sal_Re", "sal_Im", "u_Re", "u_Im", "v_Re", "v_Im"]
    )
    dim_names: Dict[str, str] = field(
        default_factory=lambda: {
            "longitude": "ny",
            "latitude": "nx",
            "ntides": "nc"
        }
    )
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        # Perform any necessary dataset initialization or modifications here
        ds = super().load_data()
        
        # Clean up dataset
        ds = ds.assign_coords({
            "omega": ds["omega"],
            "nx": ds["lon_r"].isel(ny=0),  # lon_r is constant along ny, i.e., is only a function of nx
            "ny": ds["lat_r"].isel(nx=0)  # lat_r is constant along nx, i.e., is only a function of ny
        })
        ds = ds.rename({"nx": "longitude", "ny": "latitude"})

        object.__setattr__(self, "dim_names", {"latitude": "latitude", "longitude": "longitude", "ntides": self.dim_names["ntides"]})
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
            raise ValueError(f"The dataset contains fewer than {ntides} tidal constituents.")

    def get_corrected_tides(self, model_reference_date, allan_factor):
        # Get equilibrium tides
        tpc = compute_equilibrium_tide(self.ds["longitude"], self.ds["latitude"]).isel(nc=self.ds.nc)
        # Correct for SAL
        tsc = allan_factor * (self.ds["sal_Re"] + 1j * self.ds["sal_Im"])
        tpc = tpc - tsc

        # Elevations and transports
        thc = self.ds["h_Re"] + 1j * self.ds["h_Im"]
        tuc = self.ds["u_Re"] + 1j * self.ds["u_Im"]
        tvc = self.ds["v_Re"] + 1j * self.ds["v_Im"]

        # Apply correction for phases and amplitudes
        pf, pu, aa = egbert_correction(model_reference_date)
        pf = pf.isel(nc=self.ds.nc)
        pu = pu.isel(nc=self.ds.nc)
        aa = aa.isel(nc=self.ds.nc)

        tpxo_reference_date = datetime(1992, 1, 1)
        dt = (model_reference_date - tpxo_reference_date).days * 3600 * 24

        thc = pf * thc * np.exp(1j * (self.ds["omega"] * dt + pu + aa))
        tuc = pf * tuc * np.exp(1j * (self.ds["omega"] * dt + pu + aa))
        tvc = pf * tvc * np.exp(1j * (self.ds["omega"] * dt + pu + aa))
        tpc = pf * tpc * np.exp(1j * (self.ds["omega"] * dt + pu + aa))

        tides = {"ssh": thc, "u": tuc, "v": tvc, "pot": tpc, "omega": self.ds["omega"]}
            
        for k in tides.keys():
            tides[k] = tides[k].rename({"nc": "ntides"})

        return tides



@dataclass(frozen=True, kw_only=True)
class TidalForcing:
    """
    Represents tidal forcing data used in ocean modeling.

    Parameters
    ----------
    grid : Grid
        The grid object representing the ROMS grid associated with the tidal forcing data.
    filename: str
        The path to the native tidal dataset.
    ntides : int, optional
        Number of constituents to consider. Maximum number is 14. Default is 10.
    model_reference_date : datetime, optional
        The reference date for the ROMS simulation. Default is datetime(2000, 1, 1).
    source : str, optional
        The source of the tidal data. Default is "tpxo".
    allan_factor : float, optional
        The Allan factor used in tidal model computation. Default is 2.0.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the tidal forcing data.

    Notes
    -----
    This class represents tidal forcing data used in ocean modeling. It provides
    functionality to load and process tidal data for use in numerical simulations.
    The tidal forcing data is loaded from a TPXO dataset and processed to generate
    tidal elevation, tidal potential, and tidal velocity fields.

    Examples
    --------
    >>> grid = Grid(...)
    >>> tidal_forcing = TidalForcing(grid)
    >>> print(tidal_forcing.ds)
    """

    grid: Grid
    filename: str
    ntides: int = 10
    model_reference_date: datetime = datetime(2000, 1, 1)
    source: str = "tpxo"
    allan_factor: float = 2.0
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        if self.source == "tpxo":
            data = TPXO(filename=self.filename)
 
        data.check_number_constituents(self.ntides)
        # operate on longitudes between -180 and 180 unless ROMS domain lies at least 5 degrees in lontitude away from Greenwich meridian
        lon = self.grid.ds.lon_rho
        lat = self.grid.ds.lat_rho
        angle = self.grid.ds.angle

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

        tides = data.get_corrected_tides(
            self.model_reference_date, self.allan_factor
        )
        
        # select desired number of constituents
        for k in tides.keys():
            tides[k] = tides[k].isel(ntides=slice(None, self.ntides))
        

        # interpolate onto desired grid
        coords = {"latitude": lat, "longitude": lon}
        #mask = xr.where(data.ds.depth>0, 1, 0)

        varnames = ["ssh", "pot", "u", "v"]
        data_vars = {}

        for var in varnames:
            data_vars[var] = (
                tides[var].interp(**coords).drop_vars(list(coords.keys()))
            )

        # Rotate to grid orientation
        u_tide = data_vars["u"] * np.cos(angle) + data_vars["v"] * np.sin(angle)
        v_tide = data_vars["v"] * np.cos(angle) - data_vars["u"] * np.sin(angle)

        # Convert to barotropic velocity
        u_tide = u_tide / self.grid.ds.h
        v_tide = v_tide / self.grid.ds.h

        # Interpolate from rho- to velocity points
        u_tide = interpolate_from_rho_to_u(u_tide)
        v_tide = interpolate_from_rho_to_v(u_tide)

        # save in new dataset
        ds = xr.Dataset()

        #ds["omega"] = tides["omega"]

        ds["ssh_Re"] = data_vars["ssh"].real
        ds["ssh_Im"] = data_vars["ssh"].imag
        ds["ssh_Re"].attrs["long_name"] = "Tidal elevation, real part"
        ds["ssh_Im"].attrs["long_name"] = "Tidal elevation, complex part"
        ds["ssh_Re"].attrs["units"] = "m"
        ds["ssh_Im"].attrs["units"] = "m"

        ds["pot_Re"] = data_vars["pot"].real
        ds["pot_Im"] = data_vars["pot"].imag
        ds["pot_Re"].attrs["long_name"] = "Tidal potential, real part"
        ds["pot_Im"].attrs["long_name"] = "Tidal potential, complex part"
        ds["pot_Re"].attrs["units"] = "m"
        ds["pot_Im"].attrs["units"] = "m"

        ds["u_Re"] = u_tide.real
        ds["u_Im"] = u_tide.imag
        ds["u_Re"].attrs["long_name"] = "Tidal velocity in x-direction, real part"
        ds["u_Im"].attrs["long_name"] = "Tidal velocity in x-direction, complex part"
        ds["u_Re"].attrs["units"] = "m/s"
        ds["u_Im"].attrs["units"] = "m/s"

        ds["v_Re"] = v_tide.real
        ds["v_Im"] = v_tide.imag
        ds["v_Re"].attrs["long_name"] = "Tidal velocity in y-direction, real part"
        ds["v_Im"].attrs["long_name"] = "Tidal velocity in y-direction, complex part"
        ds["v_Re"].attrs["units"] = "m/s"
        ds["v_Im"].attrs["units"] = "m/s"

        ds.attrs["source"] = self.source
        ds.attrs["model_reference_date"] = str(self.model_reference_date)
        ds.attrs["allan_factor"] = self.allan_factor

        object.__setattr__(self, "ds", ds)

    def plot(self, varname, ntides=0) -> None:
        """
        Plot the specified tidal forcing variable for a given tidal constituent.

        Parameters
        ----------
        varname : str
            The tidal forcing variable to plot. Options include:
            - "ssh_Re": Real part of tidal elevation.
            - "ssh_Im": Imaginary part of tidal elevation.
            - "pot_Re": Real part of tidal potential.
            - "pot_Im": Imaginary part of tidal potential.
            - "u_Re": Real part of tidal velocity in the x-direction.
            - "u_Im": Imaginary part of tidal velocity in the x-direction.
            - "v_Re": Real part of tidal velocity in the y-direction.
            - "v_Im": Imaginary part of tidal velocity in the y-direction.
        ntides : int, optional
            The index of the tidal constituent to plot. Default is 0, which corresponds
            to the first constituent.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        Raises
        ------
        ValueError
            If the specified field is not one of the valid options.


        Examples
        --------
        >>> tidal_forcing = TidalForcing(grid)
        >>> tidal_forcing.plot("ssh_Re", nc=0)
        """
        
        field = self.ds[varname].isel(ntides=ntides).compute()
        
        title = "%s, ntides = %i" % (
            field.long_name,
            self.ds[varname].ntides[ntides]
        )

        vmax = max(
            field.max(), -field.min()
        )
        vmin = - vmax
        cmap = plt.colormaps.get_cmap("RdBu_r")
        cmap.set_bad(color="gray")

        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

        _plot(
            self.grid.ds,
            field=field,
            straddle=self.grid.straddle,
            c="g",
            kwargs=kwargs,
        )

    def save(self, filepath: str) -> None:
        """
        Save the tidal forcing information to a netCDF4 file.

        Parameters
        ----------
        filepath
        """
        self.ds.to_netcdf(filepath)


def modified_julian_days(year, month, day, hour=0):
    """
    Calculate the Modified Julian Day (MJD) for a given date and time.

    The Modified Julian Day (MJD) is a modified Julian day count starting from
    November 17, 1858 AD. It is commonly used in astronomy and geodesy.

    Parameters
    ----------
    year : int
        The year.
    month : int
        The month (1-12).
    day : int
        The day of the month.
    hour : float, optional
        The hour of the day as a fractional number (0 to 23.999...). Default is 0.

    Returns
    -------
    mjd : float
        The Modified Julian Day (MJD) corresponding to the input date and time.

    Notes
    -----
    The algorithm assumes that the input date (year, month, day) is within the
    Gregorian calendar, i.e., after October 15, 1582. Negative MJD values are
    allowed for dates before November 17, 1858.

    References
    ----------
    - Wikipedia article on Julian Day: https://en.wikipedia.org/wiki/Julian_day
    - Wikipedia article on Modified Julian Day: https://en.wikipedia.org/wiki/Modified_Julian_day

    Examples
    --------
    >>> modified_julian_days(2024, 5, 20, 12)
    58814.0
    >>> modified_julian_days(1858, 11, 17)
    0.0
    >>> modified_julian_days(1582, 10, 4)
    -141428.5
    """

    if month < 3:
        year -= 1
        month += 12

    A = year // 100
    B = A // 4
    C = 2 - A + B
    E = int(365.25 * (year + 4716))
    F = int(30.6001 * (month + 1))
    jd = C + day + hour / 24 + E + F - 1524.5
    mjd = jd - 2400000.5

    return mjd


def egbert_correction(date):
    """
    Correct phases and amplitudes for real-time runs using parts of the
    post-processing code from Egbert's & Erofeeva's (OSU) TPXO model.

    Parameters
    ----------
    date : datetime.datetime
        The date and time for which corrections are to be applied.

    Returns
    -------
    pf : xr.DataArray
        Amplitude scaling factor for each of the 15 tidal constituents.
    pu : xr.DataArray
        Phase correction [radians] for each of the 15 tidal constituents.
    aa : xr.DataArray
        Astronomical arguments [radians] associated with the corrections.

    References
    ----------
    - Egbert, G.D., and S.Y. Erofeeva. "Efficient inverse modeling of barotropic ocean
      tides." Journal of Atmospheric and Oceanic Technology 19, no. 2 (2002): 183-204.

    """

    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    second = date.second

    rad = np.pi / 180.0
    deg = 180.0 / np.pi
    mjd = modified_julian_days(year, month, day)
    tstart = mjd + hour / 24 + minute / (60 * 24) + second / (60 * 60 * 24)

    # Determine nodal corrections pu & pf : these expressions are valid for period 1990-2010 (Cartwright 1990).
    # Reset time origin for astronomical arguments to 4th of May 1860:
    timetemp = tstart - 51544.4993

    # mean longitude of lunar perigee
    P = 83.3535 + 0.11140353 * timetemp
    P = np.mod(P, 360.0)
    if P < 0:
        P = +360
    P *= rad

    # mean longitude of ascending lunar node
    N = 125.0445 - 0.05295377 * timetemp
    N = np.mod(N, 360.0)
    if N < 0:
        N = +360
    N *= rad

    sinn = np.sin(N)
    cosn = np.cos(N)
    sin2n = np.sin(2 * N)
    cos2n = np.cos(2 * N)
    sin3n = np.sin(3 * N)

    pftmp = np.sqrt(
        (1 - 0.03731 * cosn + 0.00052 * cos2n) ** 2
        + (0.03731 * sinn - 0.00052 * sin2n) ** 2
    )  # 2N2

    pf = np.zeros(15)
    pf[0] = pftmp  # M2
    pf[1] = 1.0  # S2
    pf[2] = pftmp  # N2
    pf[3] = np.sqrt(
        (1 + 0.2852 * cosn + 0.0324 * cos2n) ** 2
        + (0.3108 * sinn + 0.0324 * sin2n) ** 2
    )  # K2
    pf[4] = np.sqrt(
        (1 + 0.1158 * cosn - 0.0029 * cos2n) ** 2
        + (0.1554 * sinn - 0.0029 * sin2n) ** 2
    )  # K1
    pf[5] = np.sqrt(
        (1 + 0.189 * cosn - 0.0058 * cos2n) ** 2 + (0.189 * sinn - 0.0058 * sin2n) ** 2
    )  # O1
    pf[6] = 1.0  # P1
    pf[7] = np.sqrt((1 + 0.188 * cosn) ** 2 + (0.188 * sinn) ** 2)  # Q1
    pf[8] = 1.043 + 0.414 * cosn  # Mf
    pf[9] = 1.0 - 0.130 * cosn  # Mm
    pf[10] = pftmp**2  # M4
    pf[11] = pftmp**2  # Mn4
    pf[12] = pftmp**2  # Ms4
    pf[13] = pftmp  # 2n2
    pf[14] = 1.0  # S1
    pf = xr.DataArray(pf, dims="nc")

    putmp = (
        np.arctan(
            (-0.03731 * sinn + 0.00052 * sin2n)
            / (1.0 - 0.03731 * cosn + 0.00052 * cos2n)
        )
        * deg
    )  # 2N2

    pu = np.zeros(15)
    pu[0] = putmp  # M2
    pu[1] = 0.0  # S2
    pu[2] = putmp  # N2
    pu[3] = (
        np.arctan(
            -(0.3108 * sinn + 0.0324 * sin2n) / (1.0 + 0.2852 * cosn + 0.0324 * cos2n)
        )
        * deg
    )  # K2
    pu[4] = (
        np.arctan(
            (-0.1554 * sinn + 0.0029 * sin2n) / (1.0 + 0.1158 * cosn - 0.0029 * cos2n)
        )
        * deg
    )  # K1
    pu[5] = 10.8 * sinn - 1.3 * sin2n + 0.2 * sin3n  # O1
    pu[6] = 0.0  # P1
    pu[7] = np.arctan(0.189 * sinn / (1.0 + 0.189 * cosn)) * deg  # Q1
    pu[8] = -23.7 * sinn + 2.7 * sin2n - 0.4 * sin3n  # Mf
    pu[9] = 0.0  # Mm
    pu[10] = putmp * 2.0  # M4
    pu[11] = putmp * 2.0  # Mn4
    pu[12] = putmp  # Ms4
    pu[13] = putmp  # 2n2
    pu[14] = 0.0  # S1
    pu = xr.DataArray(pu, dims="nc")
    # convert from degrees to radians
    pu = pu * rad

    aa = xr.DataArray(
        data=np.array(
            [
                1.731557546,  # M2
                0.0,  # S2
                6.050721243,  # N2
                3.487600001,  # K2
                0.173003674,  # K1
                1.558553872,  # O1
                6.110181633,  # P1
                5.877717569,  # Q1
                1.964021610,  # Mm
                1.756042456,  # Mf
                3.463115091,  # M4
                1.499093481,  # Mn4
                1.731557546,  # Ms4
                4.086699633,  # 2n2
                0.0,  # S1
            ]
        ),
        dims="nc",
    )

    return pf, pu, aa
    
def compute_equilibrium_tide(lon, lat):
    """
    Compute equilibrium tide for given longitudes and latitudes.

    Parameters
    ----------
    lon : xr.DataArray
        Longitudes in degrees.
    lat : xr.DataArray
        Latitudes in degrees.

    Returns
    -------
    tpc : xr.DataArray
        Equilibrium tide complex amplitude.

    Notes
    -----
    This method computes the equilibrium tide complex amplitude for given longitudes
    and latitudes. It considers 15 tidal constituents and their corresponding
    amplitudes and elasticity factors. The types of tides are classified as follows:
        - 2: semidiurnal
        - 1: diurnal
        - 0: long-term

    """

    # Amplitudes and elasticity factors for 15 tidal constituents
    A = xr.DataArray(
        data=np.array(
            [
                0.242334,  # M2
                0.112743,  # S2
                0.046397,  # N2
                0.030684,  # K2
                0.141565,  # K1
                0.100661,  # O1
                0.046848,  # P1
                0.019273,  # Q1
                0.042041,  # Mf
                0.022191,  # Mm
                0.0,  # M4
                0.0,  # Mn4
                0.0,  # Ms4
                0.006141,  # 2n2
                0.000764,  # S1
            ]
        ),
        dims="nc",
    )
    B = xr.DataArray(
        data=np.array(
            [
                0.693,  # M2
                0.693,  # S2
                0.693,  # N2
                0.693,  # K2
                0.736,  # K1
                0.695,  # O1
                0.706,  # P1
                0.695,  # Q1
                0.693,  # Mf
                0.693,  # Mm
                0.693,  # M4
                0.693,  # Mn4
                0.693,  # Ms4
                0.693,  # 2n2
                0.693,  # S1
            ]
        ),
        dims="nc",
    )

    # types: 2 = semidiurnal, 1 = diurnal, 0 = long-term
    ityp = xr.DataArray(
        data=np.array([2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 1]), dims="nc"
    )

    d2r = np.pi / 180
    coslat2 = np.cos(d2r * lat) ** 2
    sin2lat = np.sin(2 * d2r * lat)

    p_amp = (
        xr.where(ityp == 2, 1, 0) * A * B * coslat2  # semidiurnal
        + xr.where(ityp == 1, 1, 0) * A * B * sin2lat  # diurnal
        + xr.where(ityp == 0, 1, 0) * A * B * (0.5 - 1.5 * coslat2)  # long-term
    )
    p_pha = (
        xr.where(ityp == 2, 1, 0) * (-2 * lon * d2r)  # semidiurnal
        + xr.where(ityp == 1, 1, 0) * (-lon * d2r)  # diurnal
        + xr.where(ityp == 0, 1, 0) * xr.zeros_like(lon)  # long-term
    )

    tpc = p_amp * np.exp(-1j * p_pha)

    return tpc


