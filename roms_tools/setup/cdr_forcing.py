from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, model_validator
from typing import Optional, List, Dict, Union
from typing_extensions import Annotated
from annotated_types import Ge, Le
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from roms_tools import Grid
from roms_tools.constants import NUM_TRACERS
from roms_tools.plot import _plot, _get_projection
from roms_tools.regrid import LateralRegridFromROMS
from roms_tools.utils import (
    _generate_coordinate_range,
    _remove_edge_nans,
    save_datasets,
)
from roms_tools.setup.utils import (
    gc_dist,
    get_tracer_defaults,
    get_tracer_metadata_dict,
    add_tracer_metadata_to_ds,
    _to_yaml,
    _from_yaml,
)

NonNegativeFloat = Annotated[float, Ge(0)]


class Release(BaseModel):
    """Represents a single point-source release of water and tracers.

    Attributes
    ----------
    name : str
        Unique identifier for the release.
    lat : float or int
        Latitude of the release location in degrees North. Must be between -90 and 90.
    lon : float or int
        Longitude of the release location in degrees East.
    depth : float or int
        Depth of the release in meters. Must be non-negative.
    times : list of datetime
        Time points for the release events. Must be within `start_time` and `end_time`, and strictly increasing.
    volume_fluxes : float, int or list of floats/ints
        Volume flux of the release in m³/s. Can be a constant (applied uniformly) or a list matching `times`.
    tracer_concentrations : dict
        Tracer names mapped to their concentrations (float or list matching `times`).
    start_time : datetime
        Start of the simulation.
    end_time : datetime
        End of the simulation.

    Notes
    -----
    - Validates that `lat`, `depth`, `times`, `volume_fluxes`, and `tracer_concentrations` follow expected formats.
    - Converts single values for `volume_fluxes` and `tracer_concentrations` into time-series matching `times`.
    """

    name: str
    lat: Annotated[float, Ge(-90), Le(90)]
    lon: float
    depth: NonNegativeFloat
    times: List[datetime]
    volume_fluxes: Union[NonNegativeFloat, List[NonNegativeFloat]]
    tracer_concentrations: Dict[str, Union[NonNegativeFloat, List[NonNegativeFloat]]]

    start_time: datetime
    end_time: datetime

    @model_validator(mode="after")
    def check_times(self) -> "Release":
        """Validate that 'times' are sorted and fall within start_time and end_time."""

        if len(self.times) > 0:

            # Ensure times are strictly increasing
            if not all(t1 < t2 for t1, t2 in zip(self.times, self.times[1:])):
                raise ValueError(
                    f"'times' must be strictly monotonically increasing. Got: {self.times}"
                )

            # First time must not be before start_time
            if self.times[0] < self.start_time:
                raise ValueError(
                    f"First time in 'times' cannot be before start_time ({self.start_time})."
                )

            # Last time must not be after end_time
            if self.times[-1] > self.end_time:
                raise ValueError(
                    f"Last time in 'times' cannot be after end_time ({self.end_time})."
                )
        return self

    @model_validator(mode="after")
    def check_volume_fluxes(self) -> "Release":
        """Ensure 'volume_fluxes' matches length of 'times'."""

        num_times = len(self.times)

        if isinstance(self.volume_fluxes, list):
            if len(self.volume_fluxes) != num_times:
                raise ValueError(
                    f"The length of 'volume_fluxes' ({len(self.volume_fluxes)}) does not match the number of times ({num_times})."
                )

        return self

    @model_validator(mode="after")
    def check_tracer_concentrations(self) -> "Release":
        """Ensure tracer concentractions match length of 'times'."""

        num_times = len(self.times)

        for tracer, val in self.tracer_concentrations.items():

            if isinstance(val, list):
                if len(val) != num_times:
                    raise ValueError(
                        f"The length of tracer '{tracer}' ({len(val)}) does not match the number of times ({num_times})."
                    )

        return self

    @model_validator(mode="after")
    def extend_to_endpoints(self) -> "Release":
        """Ensure that the release time series starts at `start_time` and ends at
        `end_time`.

        If `volume_fluxes` is a list and does not cover the endpoints, zero volume fluxes are added.
        Tracer concentrations are extended accordingly by duplicating endpoint values.

        If `times` is empty, [start_time, end_time] is used with zero fluxes and duplicated tracer values.
        """
        if not self.times:
            self.times = [self.start_time, self.end_time]
            self.volume_fluxes = [self.volume_fluxes, self.volume_fluxes]
            self.tracer_concentrations = {
                tracer: [vals, vals]
                if not isinstance(vals, list)
                else [vals[0], vals[0]]
                for tracer, vals in self.tracer_concentrations.items()
            }
        else:
            self.times = list(self.times)  # Make mutable
            self.volume_fluxes = (
                list(self.volume_fluxes)
                if isinstance(self.volume_fluxes, list)
                else [self.volume_fluxes] * len(self.times)
            )
            self.tracer_concentrations = {
                tracer: list(vals)
                if isinstance(vals, list)
                else [vals] * len(self.times)
                for tracer, vals in self.tracer_concentrations.items()
            }

            if self.times[0] > self.start_time:
                self.times.insert(0, self.start_time)
                self.volume_fluxes.insert(0, 0.0)
                for k in self.tracer_concentrations:
                    self.tracer_concentrations[k].insert(
                        0, self.tracer_concentrations[k][0]
                    )
            if self.times[-1] < self.end_time:
                self.times.append(self.end_time)
                self.volume_fluxes.append(0.0)
                for k in self.tracer_concentrations:
                    self.tracer_concentrations[k].append(
                        self.tracer_concentrations[k][-1]
                    )

        return self


@dataclass(kw_only=True)
class CDRVolumePointSource:
    """Represents one or several volume sources of water with tracers at specific
    location(s). This class is particularly useful for modeling point sources of Carbon
    Dioxide Removal (CDR) forcing data, such as the injection of water and
    biogeochemical tracers, e.g., alkalinity (ALK) or dissolved inorganic carbon (DIC),
    through a pipe.

    Parameters
    ----------
    grid : Grid, optional
        Object representing the grid for spatial context.
    start_time : datetime
        Start time of the ROMS model simulation.
    end_time : datetime
        End time of the ROMS model simulation.
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

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        if self.start_time >= self.end_time:
            raise ValueError("`start_time` must be earlier than `end_time`.")

        # Start with an empty dataset representing zero releases
        ds = xr.Dataset(
            {
                "cdr_time": (["time"], np.empty(0)),
                "cdr_lon": (["ncdr"], np.empty(0)),
                "cdr_lat": (["ncdr"], np.empty(0)),
                "cdr_dep": (["ncdr"], np.empty(0)),
                "cdr_hsc": (["ncdr"], np.empty(0)),
                "cdr_vsc": (["ncdr"], np.empty(0)),
                "cdr_volume": (["time", "ncdr"], np.empty((0, 0))),
                "cdr_tracer": (
                    ["time", "ntracers", "ncdr"],
                    np.empty((0, NUM_TRACERS, 0)),
                ),
            },
            coords={
                "time": (["time"], np.empty(0)),
                "release_name": (["ncdr"], np.empty(0, dtype=str)),
            },
        )
        ds = add_tracer_metadata_to_ds(ds)
        self.ds = ds

        tracer_metadata = get_tracer_metadata_dict()
        self.releases["_tracer_metadata"] = tracer_metadata

        if self.releases:
            if "_metadata" not in self.releases:
                tracer_metadata = get_tracer_metadata_dict()
                self.releases["_tracer_metadata"] = tracer_metadata

            for name, params in self.releases.items():
                if name == "_tracer_metadata":
                    continue  # skip metadata entry

                release = Release(
                    **{
                        "name": name,
                        **params,
                        "start_time": self.start_time,
                        "end_time": self.end_time,
                    }
                )
                self._validate_release_location(release)
                self._add_release_to_ds(release)

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
        fill_values: str = "auto",
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
        lat : float or int
            Latitude of the release location in degrees North. Must be between -90 and 90.
        lon : float or int
            Longitude of the release location in degrees East. No restrictions on bounds.
        depth : float or int
            Depth of the release in meters. Must be non-negative.
        times : list of datetime.datetime, optional
            Explicit time points for volume fluxes and tracer concentrations. Defaults to [self.start_time, self.end_time] if None.

            Example: `times=[datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)]`

        volume_fluxes : float, int, or list of float/int, optional

            Volume flux(es) of the release in m³/s over time.

            - Constant: applies uniformly across the entire simulation period.
            - Time-varying: must match the length of `times`.

            Example:

            - Constant: `volume_fluxes=1000.0` (uniform across the entire simulation period).
            - Time-varying: `volume_fluxes=[1000.0, 1500.0, 2000.0]` (corresponds to each `times` entry).

        tracer_concentrations : dict, optional

            Dictionary of tracer names and their concentration values. The concentration values can be either
            a float/int (constant in time) or a list of float/int (time-varying).

            - Constant: applies uniformly across the entire simulation period.
            - Time-varying: must match the length of `times`.

            Default is an empty dictionary (`{}`) if not provided.
            Example:

            - Constant: `{"ALK": 2000.0, "DIC": 1900.0}`
            - Time-varying: `{"ALK": [2000.0, 2050.0, 2013.3], "DIC": [1900.0, 1920.0, 1910.2]}`
            - Mixed: `{"ALK": 2000.0, "DIC": [1900.0, 1920.0, 1910.2]}`

        fill_values : str, optional

            Strategy for filling missing tracer concentration values. Options:

            - "auto" (default): automatically set values to non-zero defaults
            - "zero": fill missing values with 0.0
        """
        # Check that the name is unique
        if name in self.releases:
            raise ValueError(f"A release with the name '{name}' already exists.")

        # Check that fill_values has proper string
        if fill_values not in ("auto", "zero"):
            raise ValueError(
                f"Invalid fill_values option: '{fill_values}'. "
                "Must be 'auto' or 'zero'."
            )

        # Set default for times if None
        if times is None:
            times = []

        # Set default for tracer_concentrations if None
        if tracer_concentrations is None:
            tracer_concentrations = {}

        # Fill in missing tracer concentrations
        defaults = get_tracer_defaults()
        for tracer_name in self.ds.tracer_name.values:
            if tracer_name not in tracer_concentrations:
                tracer_name = str(tracer_name)
                if tracer_name in ["temp", "salt"]:
                    tracer_concentrations[tracer_name] = defaults[tracer_name]
                else:
                    if fill_values == "auto":
                        tracer_concentrations[tracer_name] = defaults[tracer_name]
                    elif fill_values == "zero":
                        tracer_concentrations[tracer_name] = 0.0

        release = Release(
            name=name,
            lat=lat,
            lon=lon,
            depth=depth,
            times=times,
            volume_fluxes=volume_fluxes,
            tracer_concentrations=tracer_concentrations,
            start_time=self.start_time,
            end_time=self.end_time,
        )

        self._validate_release_location(release)
        self._add_release_to_dict(release)
        self._add_release_to_ds(release)

    def _add_release_to_ds(self, release: Release):
        """Add the release data for a specific release to the forcing dataset."""

        # Convert times to datetime64[ns]
        times = np.array(release.times, dtype="datetime64[ns]")

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
        ds = add_tracer_metadata_to_ds(ds)

        release_names = np.concatenate([self.ds.release_name.values, [release.name]])
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
            [release.lon, release.lat, release.depth, 0.0, 0.0],
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
        if isinstance(release.volume_fluxes, list):
            interpolated = np.interp(union_rel_times, rel_times, release.volume_fluxes)
        else:
            interpolated = np.full(len(union_rel_times), release.volume_fluxes)

        ds["cdr_volume"].loc[{"ncdr": ds.sizes["ncdr"] - 1}] = interpolated

        for n in range(len(self.ds.ntracers)):
            tracer_name = ds.tracer_name[n].item()
            if isinstance(release.tracer_concentrations[tracer_name], list):
                interpolated = np.interp(
                    union_rel_times,
                    rel_times,
                    release.tracer_concentrations[tracer_name],
                )
            else:
                interpolated = np.full(
                    len(union_rel_times), release.tracer_concentrations[tracer_name]
                )

            ds["cdr_tracer"].loc[
                {"ntracers": n, "ncdr": ds.sizes["ncdr"] - 1}
            ] = interpolated

        self.ds = ds

    def _add_release_to_dict(self, release: Release):
        """Add the release data to the releases dictionary."""
        self.releases[release.name] = {
            "lat": release.lat,
            "lon": release.lon,
            "depth": release.depth,
            "times": release.times,
            "tracer_concentrations": release.tracer_concentrations,
            "volume_fluxes": release.volume_fluxes,
        }

    def plot_volume_flux(self, start=None, end=None, releases="all"):
        """Plot the volume flux for each specified release within the given time range.

        Parameters
        ----------
        start : datetime or None
            Start datetime for the plot. If None, defaults to `self.start_time`.
        end : datetime or None
            End datetime for the plot. If None, defaults to `self.end_time`.
        releases : str, list of str, or "all", optional
            A string or list of release names to plot.
            If "all", the method will plot all releases.
            The default is "all".
        """

        start = start or self.start_time
        end = end or self.end_time

        # Handle "all" releases case
        if releases == "all":
            releases = [k for k in self.releases if k != "_tracer_metadata"]
        # Validate input for release names
        self._validate_release_input(releases)

        data = self.ds["cdr_volume"]

        self._plot_line(
            data,
            releases,
            start,
            end,
            title="Volume flux of release(s)",
            ylabel=r"m$^3$/s",
        )

    def plot_tracer_concentration(
        self, name: str, start=None, end=None, releases="all"
    ):
        """Plot the concentration of a given tracer for each specified release within
        the given time range.

        Parameters
        ----------
        name : str
            Name of the tracer to plot, e.g., "ALK", "DIC", etc.
        start : datetime or None
            Start datetime for the plot. If None, defaults to `self.start_time`.
        end : datetime or None
            End datetime for the plot. If None, defaults to `self.end_time`.
        releases : str, list of str, or "all", optional
            A string or list of release names to plot.
            If "all", the method will plot all releases.
            The default is "all".
        """
        start = start or self.start_time
        end = end or self.end_time

        # Handle "all" releases case
        if releases == "all":
            releases = [k for k in self.releases if k != "_tracer_metadata"]
        # Validate input for release names
        self._validate_release_input(releases)

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
            title = f"{name} concentration of release(s)"

        self._plot_line(
            data,
            releases,
            start,
            end,
            title=title,
            ylabel=f"{self.ds['tracer_unit'].isel(ntracers=tracer_index).values.item()}",
        )

    def _plot_line(self, data, releases, start, end, title="", ylabel=""):
        """Plots a line graph for the specified releases and time range."""
        colors = self._get_release_colors()

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        for release in releases:
            ncdr = np.where(self.ds["release_name"].values == release)[0][0]
            data.isel(ncdr=ncdr).plot(
                ax=ax,
                linewidth=2,
                label=release,
                color=colors[release],
                marker="x",
            )

        if len(releases) > 0:
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

        # Validate input for release names
        self._validate_release_input(releases)

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

        colors = self._get_release_colors()

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
          with the release location marked by an "x".
        - A bathymetry section along a fixed latitude (longitudinal view),
          with the release location also marked by an "x".

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
                    f"Multiple releases found: {valid_releases}. Please specify a single release to plot."
                )

        self._validate_release_input(release, list_allowed=False)

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

        colors = self._get_release_colors()

        axs[0].plot(
            self.releases[release]["lat"],
            self.releases[release]["depth"],
            color=colors[release],
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
            color=colors[release],
            marker="x",
            markersize=8,
            markeredgewidth=2,
        )

        # Adjust layout and title
        fig.subplots_adjust(hspace=0.4)
        fig.suptitle(f"Release location for: {release}")

    def save(
        self,
        filepath: Union[str, Path],
    ) -> None:
        """Save the volume source with tracers to netCDF4 file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The base path and filename for the output files.

        Returns
        -------
        List[Path]
            A list of `Path` objects for the saved files. Each element in the list corresponds to a file that was saved.
        """

        # Ensure filepath is a Path object
        filepath = Path(filepath)

        # Remove ".nc" suffix if present
        if filepath.suffix == ".nc":
            filepath = filepath.with_suffix("")

        dataset_list = [self.ds]
        output_filenames = [str(filepath)]

        saved_filenames = save_datasets(dataset_list, output_filenames)

        return saved_filenames

    def to_yaml(self, filepath: Union[str, Path]) -> None:
        """Export the parameters of the class to a YAML file, including the version of
        roms-tools.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file where the parameters will be saved.
        """

        _to_yaml(self, filepath)

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "CDRVolumePointSource":
        """Create an instance of the CDRVolumePointSource class from a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file from which the parameters will be read.

        Returns
        -------
        CDRVolumePointSource
            An instance of the CDRVolumePointSource class.
        """
        filepath = Path(filepath)

        grid = Grid.from_yaml(filepath)
        params = _from_yaml(cls, filepath)

        return cls(grid=grid, **params)

    def _validate_release_location(self, release: Release):
        """Validates the closest grid location for a release site.

        This function ensures that the given release site (lat, lon, depth) lies
        within the ocean portion of the model grid domain. It:

        - Checks if the point is within the grid domain (with buffer for boundary artifacts).
        - Verifies that the location is not on land.
        - Verifies that the location is not below the seafloor.
        """
        if self.grid:
            # Adjust longitude based on whether it crosses the International Date Line (straddle case)
            if self.grid.straddle:
                lon = xr.where(release.lon > 180, release.lon - 360, release.lon)
            else:
                lon = xr.where(release.lon < 0, release.lon + 360, release.lon)

            dx = 1 / self.grid.ds.pm
            dy = 1 / self.grid.ds.pn
            max_grid_spacing = np.sqrt(dx**2 + dy**2) / 2

            # Compute great-circle distance to all grid points
            dist = gc_dist(self.grid.ds.lon_rho, self.grid.ds.lat_rho, lon, release.lat)
            dist_min = dist.min(dim=["eta_rho", "xi_rho"])

            if (dist_min > max_grid_spacing).all():
                raise ValueError(
                    f"Release site '{release.name}' is outside of the grid domain. "
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
                    f"Release site '{release.name}' is located too close to the grid boundary. "
                    "Place release location (lat, lon) away from grid boundaries."
                )

            if self.grid.ds.mask_rho[eta_rho, xi_rho].values == 0:
                raise ValueError(
                    f"Release site '{release.name}' is on land. "
                    "Please provide coordinates (lat, lon) over ocean."
                )

            if self.grid.ds.h[eta_rho, xi_rho].values < release.depth:
                raise ValueError(
                    f"Release site '{release.name}' lies below the seafloor. "
                    f"Seafloor depth is {self.grid.ds.h[eta_rho, xi_rho].values:.2f} m, "
                    f"but requested depth is {release.depth:.2f} m. Adjust depth or location (lat, lon)."
                )

        else:
            logging.warning(
                "Grid not provided: cannot verify whether the specified lat/lon/depth location is within the domain or on land. "
                "Please check manually or provide a grid when instantiating the class."
            )

    def _validate_release_input(self, releases, list_allowed=True):
        """Validates the input for release names in plotting methods to ensure they are
        in an acceptable format and exist within the set of valid releases.

        This method ensures that the `releases` parameter is either a single release name (string) or a list
        of release names (strings), and checks that each release exists in the set of valid releases.

        Parameters
        ----------
        releases : str or list of str
            A single release name as a string, or a list of release names (strings) to validate.

        list_allowed : bool, optional
            If `True`, a list of release names is allowed. If `False`, only a single release name (string)
            is allowed. Default is `True`.

        Raises
        ------
        ValueError
            If `releases` is not a string or list of strings, or if any release name is invalid (not in `self.releases`).

        Notes
        -----
        This method checks that the `releases` input is in a valid format (either a string or a list of strings),
        and ensures each release is present in the set of valid releases defined in `self.releases`. Invalid releases
        are reported in the error message.

        If `list_allowed` is set to `False`, only a single release name (string) will be accepted. Otherwise, a
        list of release names is also acceptable.
        """

        # Ensure that a list of releases is only allowed if `list_allowed` is True
        if not list_allowed and not isinstance(releases, str):
            raise ValueError(
                f"Only a single release name (string) is allowed. Got: {releases}"
            )

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
        valid_releases = [k for k in self.releases if k != "_tracer_metadata"]
        invalid_releases = [
            release for release in releases if release not in valid_releases
        ]
        if invalid_releases:
            raise ValueError(f"Invalid releases: {', '.join(invalid_releases)}")

    def _get_release_colors(self):
        """Returns a dictionary of colors for the valid releases, based on a consistent
        colormap.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            A dictionary where the keys are release names and the values are their corresponding colors,
            assigned based on the order of releases in the valid releases list.

        Raises
        ------
        ValueError
            If the number of valid releases exceeds the available colormap capacity.

        Notes
        -----
        The colormap is chosen dynamically based on the number of valid releases:

        - If there are 10 or fewer releases, the "tab10" colormap is used.
        - If there are more than 10 but fewer than or equal to 20 releases, the "tab20" colormap is used.
        - For more than 20 releases, the "tab20b" colormap is used.
        """

        valid_releases = [k for k in self.releases if k != "_tracer_metadata"]

        # Determine the colormap based on the number of releases
        if len(valid_releases) <= 10:
            color_map = cm.get_cmap("tab10")
        elif len(valid_releases) <= 20:
            color_map = cm.get_cmap("tab20")
        else:
            color_map = cm.get_cmap("tab20b")

        # Ensure the number of releases doesn't exceed the available colormap capacity
        if len(valid_releases) > color_map.N:
            raise ValueError(
                f"Too many releases. The selected colormap supports up to {color_map.N} releases."
            )

        # Create a dictionary of colors based on the release indices
        colors = {name: color_map(i) for i, name in enumerate(valid_releases)}

        return colors
