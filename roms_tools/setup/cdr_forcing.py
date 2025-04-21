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
from roms_tools.setup.utils import (
    gc_dist,
    get_river_tracer_defaults,
    add_tracer_metadata,
)


@dataclass(kw_only=True)
class VolumeSourceWithTracers:
    """Represents one or several volume sources of water with tracers, simulating the
    introduction of water and tracer concentrations at specific location(s). This class
    is particularly useful for modeling point sources of Carbon Dioxide Removal (CDR)
    forcing data, such as the injection of water and biogeochemical tracers (e.g.,
    alkalinity (ALK) or dissolved inorganic carbon (DIC)) through a pipe.

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
        A dictionary of existing releases. Defaults to empty dictionary.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray dataset containing release metadata and forcing variables.
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

            ds = xr.Dataset(
                {
                    "cdr_time": (["time"], np.empty(0)),
                    "cdr_lon": (["ncdr"], np.empty(0)),
                    "cdr_lat": (["ncdr"], np.empty(0)),
                    "cdr_dep": (["ncdr"], np.empty(0)),
                    "cdr_hsc": (["ncdr"], np.empty(0)),
                    "cdr_vsc": (["ncdr"], np.empty(0)),
                    "cdr_volume": (["time", "ncdr"], np.empty((0, 0))),
                    "cdr_tracer": (["time", "ntracers", "ncdr"], np.empty((0, 34, 0))),
                },
                coords={
                    "time": (["time"], np.empty(0)),
                    "release_name": (["ncdr"], np.empty(0, dtype=str)),
                },
            )
            ds, tracer_metadata = add_tracer_metadata(ds, return_dict=True)
            self.ds = ds
            self.releases["_tracer_metadata"] = tracer_metadata

        else:
            if "_metadata" not in self.releases:
                tracer_metadata = add_tracer_metadata(ds=None, return_dict=True)
                self.releases["_tracer_metadata"] = tracer_metadata

            for name, params in self.releases.items():
                self._validate_release_location(
                    name=name,
                    lat=params["lat"],
                    lon=params["lon"],
                    depth=params["depth"],
                )
                self._add_release_to_ds(name=name, **params)

    def add_release(
        self,
        *,
        name: str,
        lat: float,
        lon: float,
        depth: float,
        times: Optional[List[datetime]] = None,
        volume_fluxes: Union[float, List[float]] = 0.0,
        tracer_concentrations: Optional[Dict[str, Union[float, List[float]]]] = None,
        fill_values: Optional[str] = "auto_fill",
    ):
        """Adds a release (point source) of water with tracers to the forcing dataset
        and dictionary.

        This method registers a point source at a specific location (latitude, longitude, and depth).
        The release includes both a volume flux of water and tracer
        concentrations, which can be constant or time-varying.

        Parameters
        ----------
        name : str
            Unique identifier for the release.
        lat : float
            Latitude of the release location. Must be between -90 and 90.
        lon : float
            Longitude of the release location. No restrictions on bounds.
        depth : float
            Depth of the release. Must be non-negative.
        times : list of datetime.datetime, optional
            Explicit time points for volume fluxes and tracer concentrations. Defaults to [self.start_time, self.end_time] if None.
            Must contain at least two datetime objects if provided.

            Example: `times=[datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)]`

        volume_fluxes : float or list of float, optional
            Volume flux(es) of the release in m³/s over time.
            - Constant: applies uniformly from `times[0]` to `times[-1]`.
            - Time-varying: must match the length of `times`.

            Example:
            - Constant: `volume_fluxes=1000.0` (uniform across the time period).
            - Time-varying: `volume_fluxes=[1000.0, 1500.0, 2000.0]` (corresponds to each `times` entry).

        tracer_concentrations : dict, optional
            Dictionary of tracer names and their concentration values. Can be either constant or time-varying.
            - Constant: applies uniformly from `times[0]` to `times[-1]`.
            - Time-varying: must match the length of `times`.

            Default is an empty dictionary (`{}`) if not provided.
            Example:
            - Constant: `{"ALK": 2000.0, "DIC": 1900.0}`
            - Time-varying: `{"ALK": [2000.0, 2050.0, 2013.3], "DIC": [1900.0, 1920.0, 1910.2]}`
            - Mixed: `{"ALK": 2000.0, "DIC": [1900.0, 1920.0, 1910.2]}`

        fill_values : str, optional
            Strategy for filling missing tracer values. Options:
            - "auto_fill" (default): automatically set values
            - "zero_fill": fill missing values with 0.0
        """
        # Check that the name is unique
        if name in self.releases:
            raise ValueError(f"A release with the name '{name}' already exists.")

        # Set default for times if None
        if times is None:
            times = []

        # Set default for tracer_concentrations if None
        if tracer_concentrations is None:
            tracer_concentrations = {}

        # Fill in missing tracer concentrations
        defaults = get_river_tracer_defaults()
        for tracer_name in self.ds.tracer_name.values:
            if tracer_name not in tracer_concentrations:
                tracer_name = str(tracer_name)
                if tracer_name in ["temp", "salt"]:
                    tracer_concentrations[tracer_name] = defaults[tracer_name]
                else:
                    if fill_values == "auto_fill":
                        tracer_concentrations[tracer_name] = defaults[tracer_name]
                    elif fill_values == "zero_fill":
                        tracer_concentrations[tracer_name] = 0.0

        # Check input parameters
        self._input_checks(
            name=name,
            lat=lat,
            lon=lon,
            depth=depth,
            times=times,
            volume_fluxes=volume_fluxes,
            tracer_concentrations=tracer_concentrations,
        )

        # Extend volume fluxes and tracer_concentrations across simulation period if necessary
        times, volume_fluxes, tracer_concentrations = self._handle_simulation_endpoints(
            times, volume_fluxes, tracer_concentrations
        )

        # Validate release location
        self._validate_release_location(name=name, lat=lat, lon=lon, depth=depth)

        self._add_release_to_dict(
            name=name,
            lat=lat,
            lon=lon,
            depth=depth,
            times=times,
            volume_fluxes=volume_fluxes,
            tracer_concentrations=tracer_concentrations,
        )

        self._add_release_to_ds(
            name=name,
            lat=lat,
            lon=lon,
            depth=depth,
            times=times,
            volume_fluxes=volume_fluxes,
            tracer_concentrations=tracer_concentrations,
        )

    def _add_release_to_ds(
        self,
        *,
        name: str,
        lat: float,
        lon: float,
        depth: float,
        times: Optional[List[datetime]] = None,
        tracer_concentrations: Optional[Dict[str, Union[float, List[float]]]] = None,
        volume_fluxes: Union[float, List[float]] = 0.0,
    ):
        """Add the release data for a specific release to the forcing dataset."""

        # Convert times to datetime64[ns]
        times = np.array(times, dtype="datetime64[ns]")

        # Ensure reference date is also datetime64[ns]
        ref = np.datetime64(self.model_reference_date, "ns")

        # Compute model-relative times in days
        rel_times = (times - ref) / np.timedelta64(1, "D")

        # Merge with existing time dimension
        existing_times = (
            self.ds["time"].values
            if len(self.ds["time"]) > 0
            else np.array([], dtype="datetime64[ns]")
        )
        existing_rel_times = (
            self.ds["cdr_time"].values if len(self.ds["cdr_time"]) > 0 else []
        )
        union_times = np.union1d(existing_times, times)
        union_rel_times = np.union1d(existing_rel_times, rel_times)

        # Initialize a fresh dataset to accommodate the new release.
        # xarray does not handle dynamic resizing of dimensions well (e.g., increasing 'ncdr' by 1),
        # so we recreate the dataset with the updated size.
        ds = xr.Dataset()
        ds["time"] = ("time", union_times)
        ds["cdr_time"] = ("time", union_rel_times)
        ds = add_tracer_metadata(ds)

        release_names = np.concatenate([self.ds.release_name.values, [name]])
        ds = ds.assign_coords({"release_name": (["ncdr"], release_names)})
        ds["cdr_lon"] = xr.zeros_like(ds.ncdr, dtype=np.float64)
        ds["cdr_lat"] = xr.zeros_like(ds.ncdr, dtype=np.float64)
        ds["cdr_dep"] = xr.zeros_like(ds.ncdr, dtype=np.float64)
        ds["cdr_hsc"] = xr.zeros_like(ds.ncdr, dtype=np.float64)
        ds["cdr_vsc"] = xr.zeros_like(ds.ncdr, dtype=np.float64)

        ds["cdr_volume"] = xr.zeros_like(ds.cdr_time * ds.ncdr, dtype=np.float64)
        ds["cdr_tracer"] = xr.zeros_like(
            ds.cdr_time * ds.ntracers * ds.ncdr, dtype=np.float64
        )

        # Retain previous experiment locations
        if len(self.ds["ncdr"]) > 0:
            for i in range(len(self.ds.ncdr)):
                for var_name in ["cdr_lon", "cdr_lat", "cdr_dep", "cdr_hsc", "cdr_vsc"]:
                    ds[var_name].loc[{"ncdr": i}] = self.ds[var_name].isel(ncdr=i)

        # Add the new experiment location
        for var_name, value in zip(
            ["cdr_lon", "cdr_lat", "cdr_dep", "cdr_hsc", "cdr_vsc"],
            [lon, lat, depth, 0.0, 0.0],
        ):
            ds[var_name].loc[{"ncdr": ds.sizes["ncdr"] - 1}] = np.float64(value)

        # Interpolate and retain previous experiment volume fluxes and tracer concentrations
        if len(self.ds["ncdr"]) > 0:
            for i in range(len(self.ds.ncdr)):
                interpolated = np.interp(
                    union_rel_times,
                    self.ds["cdr_time"].values,
                    self.ds["cdr_volume"].isel(ncdr=i).values,
                )
                ds["cdr_volume"].loc[{"ncdr": i}] = interpolated

                for n in range(len(self.ds.ntracers)):
                    interpolated = np.interp(
                        union_rel_times,
                        self.ds["cdr_time"].values,
                        self.ds["cdr_tracer"].isel(ntracers=n, ncdr=i).values,
                    )
                    ds["cdr_tracer"].loc[{"ntracers": n, "ncdr": i}] = interpolated

        # Handle new experiment volume fluxes and tracer concentrations
        if isinstance(volume_fluxes, list):
            interpolated = np.interp(union_rel_times, rel_times, volume_fluxes)
        else:
            interpolated = np.full(len(union_rel_times), volume_fluxes)

        ds["cdr_volume"].loc[{"ncdr": ds.sizes["ncdr"] - 1}] = interpolated

        for n in range(len(self.ds.ntracers)):
            tracer_name = ds.tracer_name[n].item()
            if isinstance(tracer_concentrations[tracer_name], list):
                interpolated = np.interp(
                    union_rel_times, rel_times, tracer_concentrations[tracer_name]
                )
            else:
                interpolated = np.full(
                    len(union_rel_times), tracer_concentrations[tracer_name]
                )

            ds["cdr_tracer"].loc[
                {"ntracers": n, "ncdr": ds.sizes["ncdr"] - 1}
            ] = interpolated

        self.ds = ds

    def _add_release_to_dict(self, name: str, **params):
        """Add the release data for a specific 'name' to the releases dictionary.

        Parameters
        ----------
        name : str
            The unique name for the release to be added to the dictionary.
        **params : keyword arguments
            Parameters to be added for the specific release (e.g., location, volume fluxes, etc.).
        """
        # Add the parameters to the dictionary under the given name
        if name not in self.releases:
            self.releases[name] = {}
        self.releases[name].update(params)

    def plot_volume_flux(self, start=None, end=None):
        """Plot the volume flux for each release.

        Parameters
        ----------
        start : datetime or None
            Start datetime. Defaults to self.start_time.
        end : datetime or None
            End datetime. Defaults to self.end_time.
        """
        start = start or self.start_time
        end = end or self.end_time

        data = self.ds["cdr_volume"]

        self._plot_line(
            data,
            start,
            end,
            title="Volume flux of release",
            ylabel=r"m$^3$/s",
        )

    def plot_tracer_concentration(self, name: str, start=None, end=None):
        """Plot the concentration of a given tracer for each release.

        Parameters
        ----------
        name : str
            Name of the tracer to plot, e.g., "ALK", "DIC", etc.
        start : datetime or None
            Start datetime. Defaults to self.start_time.
        end : datetime or None
            End datetime. Defaults to self.end_time.
        """
        start = start or self.start_time
        end = end or self.end_time
        tracer_names = list(self.ds["tracer_name"].values)
        if name not in tracer_names:
            raise ValueError(
                f"Tracer '{name}' not found. Available: {', '.join(tracer_names)}"
            )

        tracer_index = tracer_names.index(name)
        data = self.ds["cdr_tracer"].isel(ntracers=tracer_index)

        if name == "temp":
            title = "Temperature of release water"
        elif name == "salt":
            title = "Salinity of release water"
        else:
            title = f"{name} concentration of release"

        self._plot_line(
            data,
            start,
            end,
            title=title,
            ylabel=f"{self.ds['tracer_unit'].isel(ntracers=tracer_index).values.item()}",
        )

    def _plot_line(self, data, start, end, title="", ylabel=""):
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

        for ncdr in range(len(self.ds.ncdr)):
            data.isel(ncdr=ncdr).plot(
                ax=ax,
                linewidth=2,
                label=self.ds["release_name"].isel(ncdr=ncdr).item(),
                marker="x",
            )

        if len(self.ds.ncdr) > 0:
            ax.legend()

        ax.set(title=title, ylabel=ylabel)
        ax.set_xlim([start, end])

    def plot_location_top_view(self, releases="all"):
        """Plot the top-down view of release locations.

        Parameters
        ----------
        releases : list of str or str, optional
            A single release name (string) or a list of release names (strings) to plot.
            Default is 'all', which will plot all releases.

        Raises
        ------
        ValueError
            If `self.grid` is not set.
            If `releases` is not a string or list of strings.
            If any of the specified releases do not exist in `self.releases`.
        """
        # Ensure that the grid is provided
        if self.grid is None:
            raise ValueError(
                "A grid must be provided for plotting. Please pass a valid `Grid` object."
            )

        # Handle "all" releases case
        if releases == "all":
            releases = [k for k in self.releases if k != "_tracer_metadata"]

        # Validate the releases input
        if isinstance(releases, str):
            releases = [releases]  # Convert to list if a single string is provided
        elif isinstance(releases, list):
            if not all(isinstance(r, str) for r in releases):
                raise ValueError("All elements in `releases` list must be strings.")
        else:
            raise ValueError(
                "`releases` should be a string (single release name) or a list of strings (release names)."
            )

        # Validate that the specified releases exist in self.releases
        valid_release_names = [k for k in self.releases if k != "_tracer_metadata"]
        invalid_releases = [
            release for release in releases if release not in valid_release_names
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

        vmax = 6
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
        colors = {name: color_map(i) for i, name in enumerate(releases)}

        for name in releases:
            # transform coordinates to projected space
            transformed_lon, transformed_lat = trans.transform_point(
                self.releases[name]["lon"],
                self.releases[name]["lat"],
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

        ax.set_title("Release locations")
        ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))

    def plot_location_side_view(self, release: str = None):
        """Plot the release location from a side view, showing bathymetry sections along
        both fixed longitude and latitude.

        This method creates two plots:

        - A bathymetry section along a fixed longitude (latitudinal view),
          with the release location marked by a red "x".
        - A bathymetry section along a fixed latitude (longitudinal view),
          with the release location also marked by a red "x".

        Parameters
        ----------
        release : str, optional
            Name of the release to plot. If only one release is available,
            it is used by default. If multiple releases are available, this must be specified.

        Raises
        ------
        ValueError
            If `self.grid` is not set.
            If the specified `release` does not exist in `self.releases`.
            If no `release` is provided when multiple releases are available.
        """
        if self.grid is None:
            raise ValueError(
                "A grid must be provided for plotting. Please pass a valid `Grid` object."
            )

        valid_releases = [r for r in self.releases if r != "_tracer_metadata"]

        if release is None:
            if len(valid_releases) == 1:
                release = valid_releases[0]
            else:
                raise ValueError(
                    f"Multiple releases found: {valid_releases}. Please specify a release to plot."
                )

        if release not in valid_releases:
            raise ValueError(
                f"Invalid release: {release}. Valid options are: {valid_releases}"
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
            ax.fill_between(section[var_name], section.squeeze(), y2=0, color="#deebf7")
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

        # Use same colors as in top view
        if len(valid_releases) <= 10:
            color_map = cm.get_cmap("tab10")
        elif len(valid_releases) <= 20:
            color_map = cm.get_cmap("tab20")
        else:
            color_map = cm.get_cmap("tab20b")
        color_index = valid_releases.index(release)
        color = color_map(color_index)

        axs[0].plot(
            self.releases[release]["lat"],
            self.releases[release]["depth"],
            color=color,
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
            color=color,
            marker="x",
            markersize=8,
            markeredgewidth=2,
        )

        # Adjust layout and title
        fig.subplots_adjust(hspace=0.4)
        fig.suptitle(f"Release location for: {release}")

    def _input_checks(
        self,
        name,
        lat,
        lon,
        depth,
        times,
        volume_fluxes,
        tracer_concentrations,
    ):
        """Perform various input checks on release parameters.

        - Checks that latitude is between -90 and 90.
        - Checks that depth is non-negative.
        - Ensures 'times' is a list of datetime objects and is monotonically increasing.
        - Verifies that times are within the defined start and end time.
        - Ensures volume fluxes is either a list of floats/ints or a single float/int.
        - Ensures each tracer concentration is either a float/int or a list of floats/ints.
        - Ensures the lengths of 'volume_fluxes' and 'tracer_concentrations' match the length of 'times' if they are lists.
        - Ensures all entries in 'volume_fluxes' and 'tracer_concentrations' are non-negative.
        """

        # Check that lat is valid
        if not (-90 <= lat <= 90):
            raise ValueError(
                f"Invalid latitude {lat}. Latitude must be between -90 and 90."
            )

        # Check that depth is non-negative
        if depth < 0:
            raise ValueError(
                f"Invalid depth {depth}. Depth must be a non-negative number."
            )

        # Ensure that times is a list of datetimes
        if not all(isinstance(t, datetime) for t in times):
            raise ValueError(
                f"If 'times' is provided, all entries must be datetime objects. Got: {[type(t) for t in times]}"
            )

        if len(times) > 0:
            if len(times) > 1:
                # Check that times is strictly monotonically increasing sequence
                if not all(t1 < t2 for t1, t2 in zip(times, times[1:])):
                    raise ValueError(
                        f"The 'times' list must be strictly monotonically increasing. Got: {[t for t in times]}"
                    )

            # Check that first time is not before start_time
            if times[0] < self.start_time:
                raise ValueError(
                    f"First entry in `times` ({times[0]}) cannot be before `self.start_time` ({self.start_time})."
                )

            # Check that last time is not after end_time
            if times[-1] > self.end_time:
                raise ValueError(
                    f"Last entry in `times` ({times[-1]}) cannot be after `self.end_time` ({self.end_time})."
                )

        # Ensure volume fluxes is either a list of floats/ints or a single float/int
        if not isinstance(volume_fluxes, (float, int)) and not (
            isinstance(volume_fluxes, list)
            and all(isinstance(v, (float, int)) for v in volume_fluxes)
        ):
            raise ValueError(
                "Invalid 'volume_fluxes' input: must be a float/int or a list of floats/ints."
            )

        # Ensure each tracer concentration is either a float/int or a list of floats/ints
        for key, val in tracer_concentrations.items():
            if not isinstance(val, (float, int)) and not (
                isinstance(val, list) and all(isinstance(v, (float, int)) for v in val)
            ):
                raise ValueError(
                    f"Invalid tracer concentration for '{key}': must be a float/int or a list of floats/ints."
                )

        # Ensure that time series for 'times', 'volume_fluxes', and 'tracer_concentrations' are all the same length
        num_times = len(times)

        # Check that volume fluxes is either a constant or has the same length as 'times'
        if isinstance(volume_fluxes, list) and len(volume_fluxes) != num_times:
            raise ValueError(
                f"The length of `volume_fluxes` ({len(volume_fluxes)}) does not match the length of `times` ({num_times})."
            )

        # Check that tracer_concentrations are either constants or have the same length as 'times'
        for key, tracer_values in tracer_concentrations.items():
            if isinstance(tracer_values, list) and len(tracer_values) != num_times:
                raise ValueError(
                    f"The length of tracer '{key}' ({len(tracer_values)}) does not match the length of `times` ({num_times})."
                )

        # Check that volume fluxes and tracer concentrations are valid
        if isinstance(volume_fluxes, (float, int)) and volume_fluxes < 0:
            raise ValueError(f"Volume flux must be non-negative. Got: {volume_fluxes}")
        elif isinstance(volume_fluxes, list) and not all(v >= 0 for v in volume_fluxes):
            raise ValueError(
                f"All entries in `volume_fluxes` must be non-negative. Got: {volume_fluxes}"
            )
        for key, tracer_values in tracer_concentrations.items():
            if key != "temp":
                if isinstance(tracer_values, (float, int)) and tracer_values < 0:
                    raise ValueError(
                        f"The concentration of tracer '{key}' must be non-negative. Got: {tracer_values}"
                    )
                elif isinstance(tracer_values, list) and not all(
                    c >= 0 for c in tracer_values
                ):
                    raise ValueError(
                        f"All entries in `tracer_concentrations['{key}']` must be non-negative. Got: {tracer_values}"
                    )

    def _handle_simulation_endpoints(self, times, volume_fluxes, tracer_concentrations):
        """Ensure that the release time series starts at self.start_time and ends at
        self.end_time.

        If `volume_fluxes` is a list and does not cover the endpoints, zero volume fluxes are added.
        Tracer concentrations are extended accordingly by duplicating endpoint values.
        """

        if len(times) > 0:
            # Handle start_time
            if times[0] != self.start_time:
                if isinstance(volume_fluxes, list):
                    volume_fluxes.insert(0, 0.0)

                for key, vals in tracer_concentrations.items():
                    if isinstance(vals, list):
                        vals.insert(0, vals[0])

                times.insert(0, self.start_time)

            # Handle end_time
            if times[-1] != self.end_time:
                if isinstance(volume_fluxes, list):
                    volume_fluxes.append(0.0)

                for key, vals in tracer_concentrations.items():
                    if isinstance(vals, list):
                        vals.append(vals[-1])

                times.append(self.end_time)

        else:
            times = [self.start_time, self.end_time]

        return times, volume_fluxes, tracer_concentrations

    def _validate_release_location(self, name, lat, lon, depth):
        """Validates the closest grid location for a release site.

        This function ensures that the given release site (lat, lon, depth) lies
        within the ocean portion of the model grid domain. It:

        - Checks if the point is within the grid domain (with buffer for boundary artifacts).
        - Verifies that the location is not on land.
        - Verifies that the location is not below the seafloor.

        Parameters
        ----------
        name : str
            A unique identifier for the release location.
        lat : float
            Latitude of the release location.
        lon : float
            Longitude of the release location.
        depth : float
            Depth (positive, in meters) of the release location.

        Raises
        ------
        ValueError
            If the location is:
                - Outside the model grid.
                - On the boundary of the grid domain (eta_rho, xi_rho = 0 or max).
                - On land (based on `mask_rho`).
                - Below the ocean bottom (`h < depth`).
        Warning
            If no grid is available to validate the location.
        """
        if self.grid:
            # Adjust longitude based on whether it crosses the International Date Line (straddle case)
            if self.grid.straddle:
                lon = xr.where(lon > 180, lon - 360, lon)
            else:
                lon = xr.where(lon < 0, lon + 360, lon)

            dx = 1 / self.grid.ds.pm
            dy = 1 / self.grid.ds.pn
            max_grid_spacing = np.sqrt(dx**2 + dy**2) / 2

            # Compute great-circle distance to all grid points
            dist = gc_dist(self.grid.ds.lon_rho, self.grid.ds.lat_rho, lon, lat)
            dist_min = dist.min(dim=["eta_rho", "xi_rho"])

            if (dist_min > max_grid_spacing).all():
                raise ValueError(
                    f"Release site '{name}' is outside of the grid domain. "
                    "Ensure the provided (lat, lon) falls within the model grid extent."
                )

            # Find the indices of the closest grid cell
            indices = np.where(dist == dist_min)
            eta_rho = indices[0][0]
            xi_rho = indices[1][0]

            eta_max = self.grid.ds.sizes["eta_rho"] - 1
            xi_max = self.grid.ds.sizes["xi_rho"] - 1

            if eta_rho in [0, eta_max] or xi_rho in [0, xi_max]:
                raise ValueError(
                    f"Release site '{name}' is located too close to the grid boundary. "
                    "Place release location (lat, lon) away from grid boundaries."
                )

            if self.grid.ds.mask_rho[eta_rho, xi_rho].values == 0:
                raise ValueError(
                    f"Release site '{name}' is on land. "
                    "Please provide coordinates (lat, lon) over ocean."
                )

            if self.grid.ds.h[eta_rho, xi_rho].values < depth:
                raise ValueError(
                    f"Release site '{name}' lies below the seafloor. "
                    f"Seafloor depth is {self.grid.ds.h[eta_rho, xi_rho].values:.2f} m, "
                    f"but requested depth is {depth:.2f} m. Adjust depth or location (lat, lon)."
                )

        else:
            logging.warning(
                "Grid not provided: cannot verify whether the specified lat/lon/depth location is within the domain or on land. "
                "Please check manually or provide a grid when instantiating the class."
            )
