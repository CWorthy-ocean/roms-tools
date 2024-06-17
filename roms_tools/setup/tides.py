from datetime import datetime
import xarray as xr
import numpy as np
from dataclasses import dataclass, field
from roms_tools.setup.grid import Grid
from roms_tools.setup.plot import _plot
import os
import hashlib

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
            P =+ 360
        P *= rad

        # mean longitude of ascending lunar node
        N = 125.0445 - 0.05295377 * timetemp
        N = np.mod(N, 360.0)
        if N < 0:
            N =+ 360
        N *= rad

        sinn = np.sin(N)
        cosn = np.cos(N)
        sin2n = np.sin(2 * N)
        cos2n = np.cos(2 * N)
        sin3n = np.sin(3 * N)

        tmp1 = 1.36 * np.cos(P) + 0.267 * np.cos((P - N))
        tmp2 = 0.64 * np.sin(P) + 0.135 * np.sin((P - N))
        temp1 = 1.0 - 0.25 * np.cos(2 * P) - 0.11 * np.cos((2 * P - N)) - 0.04 * cosn
        temp2 = 0.25 * np.sin(2 * P) + 0.11 * np.sin((2 * P - N)) + 0.04 * sinn

        pftmp = np.sqrt((1 - 0.03731 * cosn + 0.00052 * cos2n) ** 2 +
                        (0.03731 * sinn - 0.00052 * sin2n) ** 2)  # 2N2

        pf = np.zeros(15)
        pf[0] = pftmp  # M2
        pf[1] = 1.0  # S2
        pf[2] = pftmp  # N2
        pf[3] = np.sqrt((1 + 0.2852 * cosn + 0.0324 * cos2n) ** 2 +
                        (0.3108 * sinn + 0.0324 * sin2n) ** 2)  # K2
        pf[4] = np.sqrt((1 + 0.1158 * cosn - 0.0029 * cos2n) ** 2 +
                        (0.1554 * sinn - 0.0029 * sin2n) ** 2)  # K1
        pf[5] = np.sqrt((1 + 0.189 * cosn - 0.0058 * cos2n) ** 2 +
                        (0.189 * sinn - 0.0058 * sin2n) ** 2)  # O1
        pf[6] = 1.0  # P1
        pf[7] = np.sqrt((1 + 0.188 * cosn) ** 2 + (0.188 * sinn) ** 2)  # Q1
        pf[8] = 1.043 + 0.414 * cosn  # Mf
        pf[9] = 1.0 - 0.130 * cosn  # Mm
        pf[10] = pftmp ** 2  # M4
        pf[11] = pftmp ** 2  # Mn4
        pf[12] = pftmp ** 2  # Ms4
        pf[13] = pftmp  # 2n2
        pf[14] = 1.0  # S1
        pf = xr.DataArray(pf, dims='nc')

        putmp = np.arctan((-0.03731 * sinn + 0.00052 * sin2n) /
                          (1.0 - 0.03731 * cosn + 0.00052 * cos2n)) * deg  # 2N2

        pu = np.zeros(15)
        pu[0] = putmp  # M2
        pu[1] = 0.0  # S2
        pu[2] = putmp  # N2
        pu[3] = np.arctan(- (0.3108 * sinn + 0.0324 * sin2n) /
                          (1.0 + 0.2852 * cosn + 0.0324 * cos2n)) * deg  # K2
        pu[4] = np.arctan((-0.1554 * sinn + 0.0029 * sin2n) /
                          (1.0 + 0.1158 * cosn - 0.0029 * cos2n)) * deg  # K1
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
        pu = xr.DataArray(pu, dims='nc')
        # convert from degrees to radians
        pu = pu * rad

        t0 = modified_julian_days(1992, 1, 1) * 24.0

        aa = xr.DataArray(
            data = np.array([
                1.731557546,   # M2
                0.0,           # S2
                6.050721243,   # N2
                3.487600001,   # K2
                0.173003674,   # K1
                1.558553872,   # O1
                6.110181633,   # P1
                5.877717569,   # Q1
                1.964021610,   # Mm
                1.756042456,   # Mf
                3.463115091,   # M4
                1.499093481,   # Mn4
                1.731557546,   # Ms4
                4.086699633,   # 2n2
                0.0            # S1
            ]),
            dims = 'nc'
        )


        return pf, pu, aa


@dataclass(frozen=True, kw_only=True)
class TPXO:
    """
    Represents TPXO tidal atlas.

    Parameters
    ----------
    filename : str
        The path to the TPXO dataset.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing TPXO tidal model data.

    Notes
    -----
    This class provides a convenient interface to work with TPXO tidal atlas.

    Examples
    --------
    >>> tpxo = TPXO()
    >>> tpxo.load_data("tpxo_data.nc")
    >>> print(tpxo.ds)
    <xarray.Dataset>
    Dimensions:  ...
    """

    filename: str
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        ds = self.load_data(self.filename)
        # Lon_r is constant along ny, i.e., is only a function of nx
        ds["nx"] = ds["lon_r"].isel(ny=0)
        # Lat_r is constant along nx, i.e., is only a function of ny
        ds["ny"] = ds["lat_r"].isel(nx=0)
                
        object.__setattr__(self, "ds", ds)

    def get_corrected_tides(self, model_reference_date, allan_factor):
        # Get equilibrium tides
        tpc = self.compute_equilibrium_tide(self.ds["lon_r"], self.ds["lat_r"])
        # Correct for SAL
        tsc = allan_factor * (self.ds['sal_Re'] + 1j * self.ds['sal_Im'])
        tpc = tpc - tsc

        # Elevations and transports
        thc = self.ds["h_Re"] + 1j * self.ds["h_Im"]
        tuc = self.ds["u_Re"] + 1j * self.ds["u_Im"]
        tvc = self.ds["v_Re"] + 1j * self.ds["v_Im"]

        # Apply correction for phases and amplitudes
        pf, pu, aa = egbert_correction(model_reference_date)
        tpxo_reference_date = datetime(1992, 1, 1)
        dt = (model_reference_date - tpxo_reference_date).days * 3600 * 24

        thc = pf * thc * np.exp(1j * (self.ds["omega"] * dt + pu + aa))
        tuc = pf * tuc * np.exp(1j * (self.ds["omega"] * dt + pu + aa))
        tvc = pf * tvc * np.exp(1j * (self.ds["omega"] * dt + pu + aa))
        tpc = pf * tpc * np.exp(1j * (self.ds["omega"] * dt + pu + aa))

        tides = {"ssh": thc, "u": tuc, "v": tvc, "pot": tpc, "omega": self.ds["omega"]}

        return tides
    
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
        ValueError
            If the file checksum does not match the expected value.

        Notes
        -----
        This method performs basic file existence and checksum checks to ensure the integrity of the loaded dataset.

        """
        # Check if the file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' not found.")

        # Calculate the checksum of the file
        expected_checksum = "306956d8769737ba39040118d8d08f467187fe453e02a5651305621d095bce6e"
        with open(filename, "rb") as file:
            actual_checksum = hashlib.sha256(file.read()).hexdigest()

        # Compare the checksums
        if actual_checksum != expected_checksum:
            raise ValueError("Checksum mismatch. The file may be corrupted or tampered with.")

        # Load the dataset
        ds = xr.open_dataset(filename)

        return ds


    @staticmethod
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
            data = np.array([
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
                0.0,       # M4
                0.0,       # Mn4
                0.0,       # Ms4
                0.006141,  # 2n2
                0.000764   # S1
            ]),
            dims = 'nc'
        )
        B = xr.DataArray(
            data = np.array([
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
                0.693   # S1
            ]),
            dims = 'nc'
        )
    
        # types: 2 = semidiurnal, 1 = diurnal, 0 = long-term
        ityp = xr.DataArray(data=np.array([2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 1]), dims='nc')
    
        d2r = np.pi / 180
        coslat2 = np.cos(d2r * lat) ** 2
        sin2lat = np.sin(2 * d2r * lat)
    
        p_amp = (
            xr.where(ityp == 2, 1, 0) *  A * B * coslat2                  # semidiurnal
            + xr.where(ityp == 1, 1, 0) *  A * B * sin2lat                # diurnal
            + xr.where(ityp == 0, 1, 0) *  A * B * (0.5 - 1.5 * coslat2)  # long-term
        )
        p_pha = (
            xr.where(ityp == 2, 1, 0) *  (-2 * lon * d2r)                 # semidiurnal
            + xr.where(ityp == 1, 1, 0) *  (-lon * d2r)                   # diurnal
            + xr.where(ityp == 0, 1, 0) *  xr.zeros_like(lon)             # long-term
        )
    
        tpc = p_amp * np.exp(-1j * p_pha)
    
        return tpc

    @staticmethod
    def concatenate_across_dateline(field):
        """
        Concatenate a field across the dateline for TPXO atlas.

        Parameters
        ----------
        field : xr.DataArray
            The field to be concatenated across the dateline.

        Returns
        -------
        field_concatenated : xr.DataArray
            The field concatenated across the dateline.

        Notes
        -----
        The TPXO atlas has a minimum longitude of 0.167 and a maximum longitude of 360.0.
        This method concatenates the field along the dateline on the lower end, considering 
        the discontinuity in longitudes.

        """
        lon = field['nx']
        lon_minus360 = lon - 360
        lon_concatenated = xr.concat([lon_minus360, lon], dim="nx")
        field_concatenated = xr.concat([field, field], dim="nx")
        field_concatenated["nx"] = lon_concatenated

        return field_concatenated


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
    nc : int, optional
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
    nc: int = 10
    model_reference_date: datetime = datetime(2000, 1, 1)
    source: str = "tpxo"
    allan_factor: float = 2.0
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        if self.source == "tpxo":
            tpxo = TPXO(filename=self.filename)

            tides = tpxo.get_corrected_tides(self.model_reference_date, self.allan_factor)

            # rename dimension and select desired number of constituents
            for k in tides.keys():
                tides[k] = tides[k].rename({"nc": "ntides"})
                tides[k] = tides[k].isel(ntides=slice(None, self.nc))
            
            # make sure interpolation works across dateline
            for key in ["ssh", "pot", "u", "v"]:
                tides[key] = tpxo.concatenate_across_dateline(tides[key])

            # interpolate onto desired grid
            ssh_tide = tides["ssh"].interp(nx=self.grid.ds.lon_rho, ny=self.grid.ds.lat_rho).drop_vars(["nx", "ny"])
            pot_tide = tides["pot"].interp(nx=self.grid.ds.lon_rho, ny=self.grid.ds.lat_rho).drop_vars(["nx", "ny"])
            u = tides["u"].interp(nx=self.grid.ds.lon_rho, ny=self.grid.ds.lat_rho).drop_vars(["nx", "ny"])
            v = tides["v"].interp(nx=self.grid.ds.lon_rho, ny=self.grid.ds.lat_rho).drop_vars(["nx", "ny"])

        # Rotate to grid orientation
        u_tide = u * np.cos(self.grid.ds.angle) + v * np.sin(self.grid.ds.angle)
        v_tide = v * np.cos(self.grid.ds.angle) - u * np.sin(self.grid.ds.angle)

        # Convert to barotropic velocity
        u_tide = u_tide / self.grid.ds.h
        v_tide = v_tide / self.grid.ds.h

        # Interpolate from rho- to velocity points
        u_tide = (u_tide + u_tide.shift(xi_rho=1)).isel(xi_rho=slice(1, None)).drop_vars(["lat_rho", "lon_rho"])
        u_tide = u_tide.swap_dims({"xi_rho": "xi_u"})
        v_tide = (v_tide + v_tide.shift(eta_rho=1)).isel(eta_rho=slice(1, None)).drop_vars(["lat_rho", "lon_rho"])
        v_tide = v_tide.swap_dims({"eta_rho": "eta_v"})

        # save in new dataset
        ds = xr.Dataset()

        ds["omega"] = tides["omega"]

        ds["ssh_Re"] = ssh_tide.real
        ds["ssh_Im"] = ssh_tide.imag
        ds["ssh_Re"].attrs["long_name"] = "Tidal elevation, real part"
        ds["ssh_Im"].attrs["long_name"] = "Tidal elevation, complex part"
        ds["ssh_Re"].attrs["units"] = "m"
        ds["ssh_Im"].attrs["units"] = "m"

        ds["pot_Re"] = pot_tide.real
        ds["pot_Im"] = pot_tide.imag
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
        ds.attrs["model_reference_date"] = self.model_reference_date 
        ds.attrs["allan_factor"] = self.allan_factor

        object.__setattr__(self, "ds", ds)

    def plot(self, var, nc=0) -> None:
        """
        Plot the specified tidal forcing variable for a given tidal constituent.
    
        Parameters
        ----------
        var : str
            The tidal forcing variable to plot. Options include:
            - "ssh_Re": Real part of tidal elevation.
            - "ssh_Im": Imaginary part of tidal elevation.
            - "pot_Re": Real part of tidal potential.
            - "pot_Im": Imaginary part of tidal potential.
            - "u_Re": Real part of tidal velocity in the x-direction.
            - "u_Im": Imaginary part of tidal velocity in the x-direction.
            - "v_Re": Real part of tidal velocity in the y-direction.
            - "v_Im": Imaginary part of tidal velocity in the y-direction.
        nc : int, optional
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

        vmax = max(self.ds[var].isel(ntides=nc).max(), -self.ds[var].isel(ntides=nc).min())
        kwargs = {"cmap": "RdBu_r", "vmax": vmax, "vmin": -vmax}

        fig = _plot(self.ds, field=self.ds[var].isel(ntides=nc), straddle=self.grid.straddle, c='g', kwargs=kwargs)

    def save(self, filepath: str) -> None:
        """
        Save the tidal forcing information to a netCDF4 file.

        Parameters
        ----------
        filepath
        """
        self.ds.to_netcdf(filepath)

