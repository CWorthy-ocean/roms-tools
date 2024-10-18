from datetime import datetime
import xarray as xr
import numpy as np
import yaml
import importlib.metadata
from typing import Dict, Union, List

from dataclasses import dataclass, field, asdict
from roms_tools.setup.grid import Grid
from roms_tools.setup.plot import _plot
from roms_tools.setup.datasets import TPXODataset
from roms_tools.setup.utils import (
    nan_check,
    substitute_nans_by_fillvalue,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    get_variable_metadata,
    save_datasets,
)
from roms_tools.setup.mixins import ROMSToolsMixins
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class TidalForcing(ROMSToolsMixins):
    """Represents tidal forcing data used in ocean modeling.

    Parameters
    ----------
    grid : Grid
        The grid object representing the ROMS grid associated with the tidal forcing data.
    source : Dict[str, Union[str, Path, List[Union[str, Path]]]]
        Dictionary specifying the source of the tidal data. Keys include:

          - name (str): Name of the data source (e.g., "TPXO").
          - path (Union[str, Path, List[Union[str, Path]]]): The path to the raw data file(s). This can be:

            - A single string (with or without wildcards).
            - A single Path object.
            - A list of strings or Path objects containing multiple files.

    ntides : int, optional
        Number of constituents to consider. Maximum number is 14. Default is 10.
    allan_factor : float, optional
        The Allan factor used in tidal model computation. Default is 2.0.
    model_reference_date : datetime, optional
        The reference date for the ROMS simulation. Default is datetime(2000, 1, 1).
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.

    Examples
    --------
    >>> tidal_forcing = TidalForcing(
    ...     grid=grid, source={"name": "TPXO", "path": "tpxo_data.nc"}
    ... )
    """

    grid: Grid
    source: Dict[str, Union[str, Path, List[Union[str, Path]]]]
    ntides: int = 10
    allan_factor: float = 2.0
    model_reference_date: datetime = datetime(2000, 1, 1)
    use_dask: bool = False

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        self._input_checks()
        lon, lat, angle, straddle = super().get_target_lon_lat()

        data = self._get_data()

        data.check_number_constituents(self.ntides)
        data.choose_subdomain(
            latitude_range=[lat.min().values, lat.max().values],
            longitude_range=[lon.min().values, lon.max().values],
            margin=2,
            straddle=straddle,
        )

        # select desired number of constituents
        object.__setattr__(data, "ds", data.ds.isel(ntides=slice(None, self.ntides)))

        self._correct_tides(data)

        vars_2d = [
            "ssh_Re",
            "ssh_Im",
            "pot_Re",
            "pot_Im",
            "u_Re",
            "u_Im",
            "v_Re",
            "v_Im",
        ]
        vars_3d = []

        data_vars = super().regrid_data(data, vars_2d, vars_3d, lon, lat)

        data_vars = super().process_velocities(
            data_vars, angle, "u_Re", "v_Re", interpolate=False
        )
        data_vars = super().process_velocities(
            data_vars, angle, "u_Im", "v_Im", interpolate=False
        )

        # Convert to barotropic velocity
        for varname in ["u_Re", "v_Re", "u_Im", "v_Im"]:
            data_vars[varname] = data_vars[varname] / self.grid.ds.h

        # Interpolate from rho- to velocity points
        for uname in ["u_Re", "u_Im"]:
            data_vars[uname] = interpolate_from_rho_to_u(data_vars[uname])
        for vname in ["v_Re", "v_Im"]:
            data_vars[vname] = interpolate_from_rho_to_v(data_vars[vname])

        d_meta = get_variable_metadata()
        ds = self._write_into_dataset(data_vars, d_meta)
        ds["omega"] = data.ds["omega"]

        ds = self._add_global_metadata(ds)

        # NaN values at wet points indicate that the raw data did not cover the domain, and the following will raise a ValueError
        for var in ["ssh_Re", "u_Re", "v_Im"]:
            nan_check(ds[var].isel(ntides=0), self.grid.ds.mask_rho)

        # substitute NaNs over land by a fill value to avoid blow-up of ROMS
        for var in ds.data_vars:
            ds[var] = substitute_nans_by_fillvalue(ds[var])

        object.__setattr__(self, "ds", ds)

    def _input_checks(self):

        if "name" not in self.source.keys():
            raise ValueError("`source` must include a 'name'.")
        if "path" not in self.source.keys():
            raise ValueError("`source` must include a 'path'.")

    def _get_data(self):

        if self.source["name"] == "TPXO":
            data = TPXODataset(filename=self.source["path"], use_dask=self.use_dask)
        else:
            raise ValueError('Only "TPXO" is a valid option for source["name"].')
        return data

    def _write_into_dataset(self, data_vars, d_meta):

        # save in new dataset
        ds = xr.Dataset()

        for var in data_vars.keys():
            ds[var] = data_vars[var].astype(np.float32)
            ds[var].attrs["long_name"] = d_meta[var]["long_name"]
            ds[var].attrs["units"] = d_meta[var]["units"]

        ds = ds.drop_vars(["lat_rho", "lon_rho"])

        return ds

    def _add_global_metadata(self, ds):

        ds.attrs["title"] = "ROMS tidal forcing created by ROMS-Tools"
        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"

        ds.attrs["roms_tools_version"] = roms_tools_version

        ds.attrs["source"] = self.source["name"]
        ds.attrs["model_reference_date"] = str(self.model_reference_date)
        ds.attrs["allan_factor"] = self.allan_factor

        return ds

    def plot(self, varname, ntides=0) -> None:
        """Plot the specified tidal forcing variable for a given tidal
        constituent.

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
        if all(dim in field.dims for dim in ["eta_rho", "xi_rho"]):
            field = field.where(self.grid.ds.mask_rho)
            field = field.assign_coords(
                {"lon": self.grid.ds.lon_rho, "lat": self.grid.ds.lat_rho}
            )

        elif all(dim in field.dims for dim in ["eta_rho", "xi_u"]):
            field = field.where(self.grid.ds.mask_u)
            field = field.assign_coords(
                {"lon": self.grid.ds.lon_u, "lat": self.grid.ds.lat_u}
            )

        elif all(dim in field.dims for dim in ["eta_v", "xi_rho"]):
            field = field.where(self.grid.ds.mask_v)
            field = field.assign_coords(
                {"lon": self.grid.ds.lon_v, "lat": self.grid.ds.lat_v}
            )
        else:
            ValueError("provided field does not have two horizontal dimension")

        title = "%s, ntides = %i" % (field.long_name, self.ds[varname].ntides[ntides])

        vmax = max(field.max(), -field.min())
        vmin = -vmax
        cmap = plt.colormaps.get_cmap("RdBu_r")
        cmap.set_bad(color="gray")

        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

        _plot(
            self.grid.ds,
            field=field,
            straddle=self.grid.straddle,
            c="g",
            kwargs=kwargs,
            title=title,
        )

    def save(
        self, filepath: Union[str, Path], np_eta: int = None, np_xi: int = None
    ) -> None:
        """Save the tidal forcing information to a netCDF4 file.

        This method supports saving the dataset in two modes:

        1. **Single File Mode (default)**:
           - If both `np_eta` and `np_xi` are `None`, the entire dataset is saved as a single file at the specified `filepath.nc`.

        2. **Partitioned Mode**:
           - If either `np_eta` or `np_xi` is specified, the dataset is divided into spatial tiles along the eta-axis and xi-axis.
           - The files are saved as `filepath.0.nc`, `filepath.1.nc`, ..., where the numbering corresponds to the partition index.

        Parameters
        ----------
        filepath : Union[str, Path]
            The base path or filename where the dataset should be saved.
        np_eta : int, optional
            The number of partitions along the `eta` direction. If `None`, no spatial partitioning is performed.
        np_xi : int, optional
            The number of partitions along the `xi` direction. If `None`, no spatial partitioning is performed.

        Returns
        -------
        List[Path]
            A list of Path objects for the filenames that were saved.
        """

        # Ensure filepath is a Path object
        filepath = Path(filepath)

        # Remove ".nc" suffix if present
        if filepath.suffix == ".nc":
            filepath = filepath.with_suffix("")

        dataset_list = [self.ds.load()]
        output_filenames = [str(filepath)]

        saved_filenames = save_datasets(
            dataset_list, output_filenames, np_eta=np_eta, np_xi=np_xi
        )

        return saved_filenames

    def to_yaml(self, filepath: Union[str, Path]) -> None:
        """Export the parameters of the class to a YAML file, including the
        version of roms-tools.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file where the parameters will be saved.
        """
        filepath = Path(filepath)

        grid_data = asdict(self.grid)
        grid_data.pop("ds", None)  # Exclude non-serializable fields
        grid_data.pop("straddle", None)

        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"

        # Create header
        header = f"---\nroms_tools_version: {roms_tools_version}\n---\n"

        # Extract grid data
        grid_yaml_data = {"Grid": grid_data}

        # Extract tidal forcing data
        tidal_forcing_data = {
            "TidalForcing": {
                "source": self.source,
                "ntides": self.ntides,
                "model_reference_date": self.model_reference_date.isoformat(),
                "allan_factor": self.allan_factor,
            }
        }

        # Combine both sections
        yaml_data = {**grid_yaml_data, **tidal_forcing_data}

        with filepath.open("w") as file:
            # Write header
            file.write(header)
            # Write YAML data
            yaml.dump(yaml_data, file, default_flow_style=False)

    @classmethod
    def from_yaml(
        cls, filepath: Union[str, Path], use_dask: bool = False
    ) -> "TidalForcing":
        """Create an instance of the TidalForcing class from a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file from which the parameters will be read.
        use_dask: bool, optional
            Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.

        Returns
        -------
        TidalForcing
            An instance of the TidalForcing class.
        """
        filepath = Path(filepath)
        # Read the entire file content
        with filepath.open("r") as file:
            file_content = file.read()

        # Split the content into YAML documents
        documents = list(yaml.safe_load_all(file_content))

        tidal_forcing_data = None

        # Process the YAML documents
        for doc in documents:
            if doc is None:
                continue
            if "TidalForcing" in doc:
                tidal_forcing_data = doc["TidalForcing"]
                break

        if tidal_forcing_data is None:
            raise ValueError("No TidalForcing configuration found in the YAML file.")

        # Convert the model_reference_date from string to datetime
        tidal_forcing_params = tidal_forcing_data
        tidal_forcing_params["model_reference_date"] = datetime.fromisoformat(
            tidal_forcing_params["model_reference_date"]
        )

        # Create Grid instance from the YAML file
        grid = Grid.from_yaml(filepath)

        # Create and return an instance of TidalForcing
        return cls(grid=grid, **tidal_forcing_params, use_dask=use_dask)

    def _correct_tides(self, data):
        """Apply tidal corrections to the dataset. This method corrects the
        dataset for equilibrium tides, self-attraction and loading (SAL)
        effects, and adjusts phases and amplitudes of tidal elevations and
        transports using Egbert's correction.

        Parameters
        ----------
        data : Dataset
            The dataset containing tidal data, including variables for sea surface height (ssh), zonal and meridional
            currents (u, v), and self-attraction and loading corrections (sal).
        Returns
        -------
        None
            The dataset is modified in-place with corrected real and imaginary components for ssh, u, v, and the
            potential field ('pot_Re', 'pot_Im').
        """

        # Get equilibrium tides
        tpc = compute_equilibrium_tide(
            data.ds[data.dim_names["longitude"]], data.ds[data.dim_names["latitude"]]
        )
        tpc = tpc.isel(ntides=data.ds["ntides"])
        # Correct for SAL
        tsc = self.allan_factor * (
            data.ds[data.var_names["sal_Re"]] + 1j * data.ds[data.var_names["sal_Im"]]
        )
        tpc = tpc - tsc

        # Elevations and transports
        thc = data.ds[data.var_names["ssh_Re"]] + 1j * data.ds[data.var_names["ssh_Im"]]
        tuc = data.ds[data.var_names["u_Re"]] + 1j * data.ds[data.var_names["u_Im"]]
        tvc = data.ds[data.var_names["v_Re"]] + 1j * data.ds[data.var_names["v_Im"]]

        # Apply correction for phases and amplitudes
        pf, pu, aa = egbert_correction(self.model_reference_date)
        pf = pf.isel(ntides=data.ds["ntides"])
        pu = pu.isel(ntides=data.ds["ntides"])
        aa = aa.isel(ntides=data.ds["ntides"])

        dt = (self.model_reference_date - data.reference_date).days * 3600 * 24

        thc = pf * thc * np.exp(1j * (data.ds["omega"] * dt + pu + aa))
        tuc = pf * tuc * np.exp(1j * (data.ds["omega"] * dt + pu + aa))
        tvc = pf * tvc * np.exp(1j * (data.ds["omega"] * dt + pu + aa))
        tpc = pf * tpc * np.exp(1j * (data.ds["omega"] * dt + pu + aa))

        data.ds[data.var_names["ssh_Re"]] = thc.real
        data.ds[data.var_names["ssh_Im"]] = thc.imag
        data.ds[data.var_names["u_Re"]] = tuc.real
        data.ds[data.var_names["u_Im"]] = tuc.imag
        data.ds[data.var_names["v_Re"]] = tvc.real
        data.ds[data.var_names["v_Im"]] = tvc.imag
        data.ds["pot_Re"] = tpc.real
        data.ds["pot_Im"] = tpc.imag

        # Update var_names dictionary
        var_names = {**data.var_names, "pot_Re": "pot_Re", "pot_Im": "pot_Im"}
        object.__setattr__(data, "var_names", var_names)


def modified_julian_days(year, month, day, hour=0):
    """Calculate the Modified Julian Day (MJD) for a given date and time.

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
    """Correct phases and amplitudes for real-time runs using parts of the
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
    pf = xr.DataArray(pf, dims="ntides")

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
    pu = xr.DataArray(pu, dims="ntides")
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
        dims="ntides",
    )

    return pf, pu, aa


def compute_equilibrium_tide(lon, lat):
    """Compute equilibrium tide for given longitudes and latitudes.

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
        dims="ntides",
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
        dims="ntides",
    )

    # types: 2 = semidiurnal, 1 = diurnal, 0 = long-term
    ityp = xr.DataArray(
        data=np.array([2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 1]), dims="ntides"
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
