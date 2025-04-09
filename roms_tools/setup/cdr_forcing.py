from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Union
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from roms_tools import Grid
from roms_tools.plot import _plot, _get_projection
from roms_tools.regrid import LateralRegridFromROMS
from roms_tools.utils import _generate_coordinate_range, _remove_edge_nans
from roms_tools.setup.utils import gc_dist


@dataclass(kw_only=True)
class CDRPipeForcing:
    """Represents CDR pipe forcing data for ROMS, supporting both constant and time-
    varying tracer concentrations and volumes.

    Parameters
    ----------
    grid : Grid, optional
        Object representing the grid for spatial context.
    start_time : datetime
        Start time of the model simulation.
    end_time : datetime
        End time of the model simulation.
    model_reference_date : datetime, optional
        Reference date for converting absolute times to model-relative time. Defaults to Jan 1, 2000.
    releases : dict, optional
        A dictionary of existing CDR releases. Defaults to empty dictionary.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray dataset containing CDR release metadata and forcing variables.
    """

    grid: Optional["Grid"] = None
    start_time: datetime
    end_time: datetime
    model_reference_date: datetime = datetime(2000, 1, 1)
    releases: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        if self.start_time >= self.end_time:
            raise ValueError("`start_time` must be earlier than `end_time`.")

        if not self.releases:

            self.ds = xr.Dataset(
                {
                    "cdr_time": (["time"], np.empty(0)),
                    "cdr_volume": (["time", "ncdr"], np.empty((0, 0))),
                    # "cdr_tracer": (["time", "ntracers", "ncdr"], np.empty((0, 0))),
                },
                coords={
                    "time": (["time"], np.empty(0)),
                    "release_name": (["ncdr"], np.empty(0, dtype=str)),
                },
            )
            self._add_global_metadata()
        else:
            for name, params in self.releases.items():
                self._add_grid_indices_to_dict(
                    name=name, lat=params["lat"], lon=params["lon"]
                )
                self._add_release_to_ds(name=name, **params)

    def _add_global_metadata(self):
        tracer_names = [
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
        ]
        self.ds = self.ds.assign_coords({"tracer_name": (["ntracers"], tracer_names)})

    def add_release(
        self,
        *,
        name: str,
        lat: float,
        lon: float,
        depth: float,
        release_start_time: Optional[datetime] = None,
        release_end_time: Optional[datetime] = None,
        times: Optional[List[datetime]] = None,
        # tracer_conentrations: Optional[Dict[str, Union[float, List[float]]]] = None,
        volume: Union[float, List[float]] = 0.0,
        fill_values: Optional[str] = "auto_fill",
    ):
        """Adds a CDR pipe release to the forcing dataset and dictionary.

        Parameters
        ----------
        name : str
            Unique identifier for the release.
        lat : float
            Latitude of the release location. Must be between -90 and 90.
        lon : float
            Longitude of the release location. No restrictions on bounds; longitude can be any value.
        depth : float
            Depth of the release.
        release_start_time : datetime, optional
            Start time of the release. Required if `times` is `None`.
        release_end_time : datetime, optional
            End time of the release. Required if `times` is `None`.
        times : List[datetime], optional
            Explicit time points for time-varying tracer concentrations and volumes.
        tracer_concentrations : dict, optional
            A dictionary of tracer names and their corresponding concentration values, which can be constant or time-varying.
            Example formats:
            - Constant tracer concentrations: {"temp": 20.0, "salt": 1.0, "ALK": 2000.0}
            - Time-varying tracer concentrations: {"temp": [19.5, 20, 20, 20], "salt": [1.1, 2, 1, 1], "ALK": [2000.0, 2014.3, 2001.0, 2004.2]} (with `times` set to four corresponding datetime entries)
        volume : float or list of float, optional
            Volume of release over time.
        fill_values : str, optional
            Strategy for filling missing tracer concentration values. Options: "auto_fill", "zero_fill".
        """

        self._add_release_to_dict(
            name=name,
            lat=lat,
            lon=lon,
            depth=depth,
            release_start_time=release_start_time,
            release_end_time=release_end_time,
            times=times,
            volume=volume,
        )
        self._add_release_to_ds(
            name=name,
            lat=lat,
            lon=lon,
            depth=depth,
            release_start_time=release_start_time,
            release_end_time=release_end_time,
            times=times,
            volume=volume,
        )

    def _add_release_to_ds(
        self,
        *,
        name: str,
        lat: float,
        lon: float,
        depth: float,
        release_start_time: Optional[datetime] = None,
        release_end_time: Optional[datetime] = None,
        times: Optional[List[datetime]] = None,
        tracer_concentrations: Optional[Dict[str, Union[float, List[float]]]] = None,
        volume: Union[float, List[float]] = 0.0,
    ):
        """Adds a CDR pipe release to the forcing dataset."""

        self._input_checks(
            name,
            lat,
            lon,
            depth,
            release_start_time,
            release_end_time,
            times,
            tracer_concentrations,
            volume,
        )

        # Check that the name is unique
        if name in self.ds["release_name"].values:
            raise ValueError(f"A release with the name '{name}' already exists.")

        if release_start_time and release_end_time and not times:
            times = [release_start_time, release_end_time]
        elif times and not (release_start_time and release_end_time):
            release_start_time, release_end_time = times[0], times[-1]

        times = np.array(times)

        # Convert times to model-relative days
        rel_times = (times - self.model_reference_date).astype(
            "timedelta64[ns]"
        ) / np.timedelta64(1, "D")

        # Merge with existing time dimension
        existing_times = (
            self.ds["time"].values
            if len(self.ds["time"]) > 0
            else np.array([], dtype="datetime64[ns]")
        )
        existing_rel_times = (
            self.ds["cdr_time"].values if len(self.ds["cdr_time"]) > 0 else []
        )
        times = np.array(times, dtype="datetime64[ns]")
        union_time = np.union1d(existing_times, times)
        union_rel_time = np.union1d(existing_rel_times, rel_times)

        # Initialize updated dataset
        ds = xr.Dataset()
        ds["cdr_time"] = ("time", union_rel_time)
        ds["time"] = ("time", union_time)

        release_names = np.concatenate([self.ds.release_name.values, [name]])
        ds = ds.assign_coords({"release_name": (["ncdr"], release_names)})
        ds["cdr_volume"] = xr.zeros_like(ds.cdr_time * ds.ncdr)
        # ds["cdr_tracer"] = xr.zeros_like(ds.cdr_time * ds.ncdr)

        # Interpolate and retain previous experiment volumes and tracer concentrations
        if len(self.ds["ncdr"]) > 0:
            for i in range(len(self.ds.ncdr)):
                for key in ["volume"]:  # , "tracer"]:
                    interpolated = np.interp(
                        union_time.astype(np.float64),
                        self.ds["cdr_time"].values.astype(np.float64),
                        self.ds[f"cdr_{key}"].isel(ncdr=i).values,
                    )
                    ds[f"cdr_{key}"].loc[{"ncdr": i}] = interpolated

        # Handle new experiment volume and tracer concentrations
        if isinstance(volume, list):
            new_volume = np.interp(
                union_time.astype(np.float64), times.astype(np.float64), volume
            )
        else:
            new_volume = np.full(len(union_time), volume)

        ds["cdr_volume"].loc[{"ncdr": ds.sizes["ncdr"] - 1}] = new_volume

        self.ds = ds
        self._add_global_metadata()

    def _add_release_to_dict(self, name: str, **params):
        """Add the release data for a specific 'name' to the releases dictionary.

        Parameters
        ----------
        name : str
            The unique name for the release to be added to the dictionary.
        **params : keyword arguments
            Parameters to be added for the specific release (e.g., location, volume, etc.).
        """
        # Check if the name already exists in the dictionary
        if name in self.releases:
            raise ValueError(f"Release name '{name}' already exists in the dictionary.")

        # Extract lat and lon from params
        lat = params.get("lat")
        lon = params.get("lon")
        self._add_grid_indices_to_dict(name=name, lat=lat, lon=lon)

        # Add the parameters to the dictionary under the given name
        if name not in self.releases:
            self.releases[name] = {}
        self.releases[name].update(params)

    def plot(self, var_name):

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

        for ncdr in range(len(self.ds.ncdr)):
            self.ds[var_name].isel(ncdr=ncdr).plot(
                ax=ax,
                linewidth=2,
                label=self.ds["release_name"].isel(ncdr=ncdr).item(),
            )

        if len(self.ds.ncdr) > 0:
            ax.legend()

        title = ""
        ylabel = ""

        if var_name == "cdr_volume":
            title = "Volume flux of CDR release"
            ylabel = r"m$^3$/s"

        ax.set(title=title, ylabel=ylabel)

    def plot_location_top(self, releases="all"):
        """Plot the top-down view of Carbon Dioxide Removal (CDR) release locations.

        Parameters
        ----------
        releases : list of str or str, optional
            A single release name (string) or a list of release names (strings) to plot.
            Each release should correspond to a key in `self.releases`. Default is 'all', which will plot all releases.

        Raises
        ------
        ValueError
            If any of the specified releases do not exist in `self.releases`.
            If `releases` is not a string or list of strings.
            If `self.grid` is not set.
        """
        # Ensure that the grid is provided
        if self.grid is None:
            raise ValueError(
                "A grid must be provided for plotting. Please pass a valid `Grid` object."
            )

        # Handle "all" releases case
        if releases == "all":
            releases = list(self.releases.keys())

        # Validate the releases input
        if isinstance(releases, str):
            releases = [releases]  # Convert to list if a single string is provided
        elif not isinstance(releases, list):
            raise ValueError(
                "`releases` should be a string (single release name) or a list of strings (release names)."
            )

        # Validate that the specified releases exist in self.releases
        invalid_releases = [
            release for release in releases if release not in self.releases
        ]
        if invalid_releases:
            raise ValueError(f"Invalid releases: {', '.join(invalid_releases)}")

        # Proceed with plotting
        field = self.grid.ds.mask_rho
        lon_deg = self.grid.ds.lon_rho
        lat_deg = self.grid.ds.lat_rho
        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)
        field = field.assign_coords({"lon": lon_deg, "lat": lat_deg})

        vmax = 3
        vmin = 0
        cmap = plt.colormaps.get_cmap("Blues")
        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

        trans = _get_projection(lon_deg, lat_deg)

        fig, ax = plt.subplots(1, 1, figsize=(13, 7), subplot_kw={"projection": trans})

        _plot(field, kwargs=kwargs, ax=ax, c=None, add_colorbar=False)

        proj = ccrs.PlateCarree()

        if len(releases) <= 10:
            color_map = cm.get_cmap("tab10")
        elif len(releases) <= 20:
            color_map = cm.get_cmap("tab20")
        else:
            color_map = cm.get_cmap("tab20b")
        # Create a dictionary of colors
        colors = {name: color_map(i) for i, name in enumerate(self.releases.keys())}

        for name in releases:
            eta_index = self.releases[name]["eta_rho"]
            xi_index = self.releases[name]["xi_rho"]

            # transform coordinates to projected space
            transformed_lon, transformed_lat = trans.transform_point(
                self.grid.ds.lon_rho[eta_index, xi_index],
                self.grid.ds.lat_rho[eta_index, xi_index],
                proj,
            )

            ax.plot(
                transformed_lon,
                transformed_lat,
                marker="x",
                markersize=8,
                markeredgewidth=2,
                label=name,
                color=colors[name],
            )

        ax.set_title("CDR release locations")
        ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))

    def plot_location_side(self, release: str):
        """Plots the Carbon Dioxide Removal (CDR) release locations from a side view,
        displaying bathymetry sections along both fixed longitude and latitude.

        This method creates two plots:

        - The first plot shows a bathymetry section along a fixed longitude (latitudinal view),
        with the CDR release location marked by a red "x".
        - The second plot shows a bathymetry section along a fixed latitude (longitudinal view),
        with the CDR release location marked by a red "x".

        Parameters
        ----------
        release : str
            The name of the CDR release to plot. This should correspond to a key in `self.releases`.

        Raises
        ------
        ValueError
            If the specified `release` does not exist in `self.releases`.
        """

        # Validate if the release exists in the dataset
        if release not in self.releases:
            raise ValueError(
                f"Invalid release: {release}. It does not exist in self.releases."
            )

        def _plot_bathymetry_section(
            ax, h, dim, fixed_val, coord_deg, resolution, title
        ):
            """Plots a bathymetry section along a fixed latitude or longitude.

            Parameters
            ----------
            ax : matplotlib.axes.Axes
                The axis on which the plot will be drawn.

            h : xarray.DataArray
                The bathymetry data to plot.

            dim : str
                The dimension along which to plot the section, either "lat" or "lon".

            fixed_val : float
                The fixed value of latitude or longitude for the section.

            coord_deg : xarray.DataArray
                The array of latitude or longitude coordinates.

            resolution : float
                The resolution at which to generate the coordinate range.

            title : str
                The title for the plot.

            Returns
            -------
            None
                The function does not return anything. It directly plots the bathymetry section on the provided axis.
            """
            # Determine coordinate names and build target range
            var_range = _generate_coordinate_range(
                coord_deg.min().values, coord_deg.max().values, resolution
            )
            var_name = "lat" if dim == "lon" else "lon"
            range_da = xr.DataArray(
                var_range,
                dims=[var_name],
                attrs={"units": "°N" if var_name == "lat" else "°E"},
            )

            # Construct target coordinates for regridding
            target_coords = {dim: [fixed_val], var_name: range_da}
            regridder = LateralRegridFromROMS(h, target_coords)
            section = regridder.apply(h)
            section, _ = _remove_edge_nans(section, var_name)

            # Plot the bathymetry section
            section.plot(ax=ax, color="k")
            ax.fill_between(
                section[var_name], section.squeeze(), y2=0, color="lightblue"
            )
            ax.invert_yaxis()
            ax.set_xlabel("Latitude [°N]" if var_name == "lat" else "Longitude [°E]")
            ax.set_ylabel("Depth [m]")
            ax.set_title(title)

        # Prepare grid coordinates
        lon_deg = self.grid.ds.lon_rho
        lat_deg = self.grid.ds.lat_rho
        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)

        resolution = self.grid._infer_nominal_horizontal_resolution()
        h = self.grid.ds.h.assign_coords({"lon": lon_deg, "lat": lat_deg})

        # Set up plot
        fig, axs = plt.subplots(2, 1, figsize=(7, 8))

        # Plot along fixed longitude
        _plot_bathymetry_section(
            ax=axs[0],
            h=h,
            dim="lon",
            fixed_val=self.releases[release]["lon"],
            coord_deg=lat_deg,
            resolution=resolution,
            title=f"Longitude: {self.releases[release]['lon']}°E",
        )
        axs[0].plot(
            self.releases[release]["lat"],
            self.releases[release]["depth"],
            color="r",
            marker="x",
            markersize=8,
            markeredgewidth=2,
        )

        # Plot along fixed latitude
        _plot_bathymetry_section(
            ax=axs[1],
            h=h,
            dim="lat",
            fixed_val=self.releases[release]["lat"],
            coord_deg=lon_deg,
            resolution=resolution,
            title=f"Latitude: {self.releases[release]['lat']}°N",
        )
        axs[1].plot(
            self.releases[release]["lon"],
            self.releases[release]["depth"],
            color="r",
            marker="x",
            markersize=8,
            markeredgewidth=2,
        )

        # Adjust layout and title
        fig.subplots_adjust(hspace=0.4)
        fig.suptitle(f"CDR release location for: {release}")

    def _input_checks(
        self,
        name,
        lat,
        lon,
        depth,
        release_start_time,
        release_end_time,
        times,
        tracer_concentrations,
        volume,
    ):
        # Check that lat is valid
        if not (-90 <= lat <= 90):
            raise ValueError("Latitude must be between -90 and 90.")

        # Check that depth is non-negative
        if depth < 0:
            raise ValueError("Depth must be a non-negative number.")

        if release_start_time >= release_end_time:
            raise ValueError("`release_start_time` must be before `release_end_time`.")

        # Check that release_start_time is not before start_time
        if release_start_time:
            if release_start_time < self.start_time:
                raise ValueError(
                    "`release_start_time` cannot be before `self.start_time`."
                )

        # Check that release_end_time is not after end_time
        if release_end_time:
            if release_end_time > self.end_time:
                raise ValueError("`release_end_time` cannot be after `self.end_time`.")

        # Check that release_start_time is before release_end_time
        if release_start_time and release_end_time:
            if release_start_time >= release_end_time:
                raise ValueError("release_start_time must be before release_end_time.")

        # Ensure that times is either None (for constant tracer concentrations) or a list of datetimes
        if times is not None and not all(isinstance(t, datetime) for t in times):
            raise ValueError(
                "If 'times' is provided, all entries must be datetime objects."
            )

        if times is not None and times[0] < release_start_time:
            raise ValueError(
                "First entry in `times` cannot be before `release_start_time`."
            )

        if times is not None and times[-1] > release_end_time:
            raise ValueError(
                "Last entry in `times` cannot be after `release_end_time`."
            )

        # Ensure that tracer concentrations dictionary is not empty for time-varying forcing
        # if times is not None and not tracer_concentrations:
        #    raise ValueError(
        #        "The 'tracer_concenrations' dictionary cannot be empty when 'times' is provided."
        #    )
        #    raise ValueError("The 'tracer_concentrations' dictionary cannot be empty.")

        # Check that volume is valid
        if isinstance(volume, float) and volume < 0:
            raise ValueError("Volume must be a non-negative number.")
        elif isinstance(volume, list) and not all(v >= 0 for v in volume):
            raise ValueError(
                "All entries in 'volume' list must be non-negative numbers."
            )

        # Ensure that time series for 'times', 'volume', and tracer_concentrations are all the same length
        if times is not None:
            num_times = len(times)

        # Check that volume is either a constant or has the same length as 'times'
        if isinstance(volume, list) and len(volume) != num_times:
            raise ValueError(
                f"The length of 'volume' ({len(volume)}) does not match the length of 'times' ({num_times})."
            )

        # Check that each time-varying tracer_concentrations has the same length as 'times'
        # for key, tracer_values in tracer_concentrations.items():
        #    if isinstance(tracer_values, list) and len(tracer_values) != num_times:
        #        raise ValueError(
        #            f"The length of tracer '{key}' ({len(tracer_values)}) does not match the length of 'times' ({num_times})."
        #        )

    def _add_grid_indices_to_dict(self, name, lat, lon):
        """Adds the closest grid indices (eta_rho, xi_rho) for a given release location
        (lat, lon) to the releases dictionary.

        Adjusts longitude for the International Date Line, checks that the release
        location is within the grid domain and not on land, and adds the indices.

        Parameters
        ----------
        name : str
            Unique name for the release.
        lat : float
            Latitude of the release location.
        lon : float
            Longitude of the release location.

        Raises
        ------
        ValueError
            If the location is outside the grid or on land.
        Warning
            If no grid is provided for validation.
        """
        if self.grid:
            # Adjust longitude based on whether it crosses the International Date Line (straddle case)
            if self.grid.straddle:
                lon = xr.where(lon > 180, lon - 360, lon)
            else:
                lon = xr.where(lon < 0, lon + 360, lon)

            # maximum dx in grid
            dx = (
                np.sqrt((1 / self.grid.ds.pm) ** 2 + (1 / self.grid.ds.pn) ** 2) / 2
            ).max()

            # Calculate the distance between the grid coordinates and the CDR release location
            dist = gc_dist(self.grid.ds.lon_rho, self.grid.ds.lat_rho, lon, lat)
            dist_min = dist.min(dim=["eta_rho", "xi_rho"])

            if (dist_min > dx).all():
                raise ValueError(
                    "The specified release location lies outside the model grid domain. "
                    "Please ensure the provided latitude and longitude fall within the grid boundaries."
                )

        else:
            logging.warning(
                "Grid not provided: cannot verify whether the specified lat/lon/depth location is within the domain or on land. "
                "Please check manually or provide a grid when instantiating the class."
            )

        # Find the indices of the closest grid cell
        indices = np.where(dist == dist_min)
        eta_rho = indices[0][0]
        xi_rho = indices[1][0]

        if self.grid.ds.mask_rho[eta_rho, xi_rho].values == 0:
            raise ValueError(
                f"The specified release location at grid indices (eta_rho={eta_rho}, xi_rho={xi_rho}) lies on land. "
                "Please ensure that the provided latitude and longitude fall over ocean in the model grid domain. "
                "You may need to adjust the release coordinates or provide a grid of finer resolution."
            )

        if name not in self.releases:
            self.releases[name] = {}

        self.releases[name]["eta_rho"] = int(eta_rho)
        self.releases[name]["xi_rho"] = int(xi_rho)
