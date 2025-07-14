import itertools
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Annotated, Iterator

import cartopy.crs as ccrs
import gcm_filters
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from pydantic import (
    BaseModel,
    Field,
    RootModel,
    conlist,
    model_serializer,
    model_validator,
)

from roms_tools import Grid
from roms_tools.constants import R_EARTH
from roms_tools.plot import _get_projection, _plot
from roms_tools.regrid import LateralRegridFromROMS
from roms_tools.setup.cdr_release import (
    Release,
    ReleaseType,
    TracerPerturbation,
    VolumeRelease,
)
from roms_tools.setup.utils import (
    _from_yaml,
    _to_dict,
    _write_to_yaml,
    add_tracer_metadata_to_ds,
    convert_to_relative_days,
    gc_dist,
    get_target_coords,
)
from roms_tools.utils import (
    _generate_focused_coordinate_range,
    _remove_edge_nans,
    normalize_longitude,
    save_datasets,
)

INCLUDE_ALL_RELEASE_NAMES = "all"


class ReleaseSimulationManager(BaseModel):
    """Validates and adjusts a single release against a ROMS simulation time window and
    grid."""

    release: Release
    grid: Grid | None = None
    start_time: datetime
    end_time: datetime

    @model_validator(mode="after")
    def check_release_times_within_simulation_window(
        self,
    ) -> "ReleaseSimulationManager":
        """Ensure the release times are within the [start_time, end_time] simulation
        window."""

        times = self.release.times
        if len(times) == 0:
            return self

        if times[0] < self.start_time:
            raise ValueError(
                f"First time in release '{self.release.name}' is before start_time ({self.start_time})."
            )

        if times[-1] > self.end_time:
            raise ValueError(
                f"Last time in release '{self.release.name}' is after end_time ({self.end_time})."
            )

        return self

    @model_validator(mode="after")
    def validate_release_location(self) -> "ReleaseSimulationManager":
        """Ensure the release is consistent with the simulation grid."""
        _validate_release_location(self.grid, self.release)
        return self

    @model_validator(mode="after")
    def extend_to_endpoints(self) -> "ReleaseSimulationManager":
        """Extend the release time series to include the simulation time endpoints."""
        self.release._extend_to_endpoints(self.start_time, self.end_time)
        return self


class ReleaseCollector(RootModel):
    """Collects and validates multiple releases against each other."""

    root: conlist(
        Annotated[
            VolumeRelease | TracerPerturbation, Field(discriminator="release_type")
        ],
        min_length=1,
    ) = Field(alias="releases")

    _release_type: ReleaseType = None

    def __iter__(self) -> Iterator[Release]:
        return iter(self.root)

    def __getitem__(self, item: int | str) -> Release:
        if isinstance(item, int):
            return self.root[item]
        elif isinstance(item, str):
            for release in self.root:
                if release.name == item:
                    return release
            raise KeyError(f"Release named '{item}' not found.")
        else:
            raise TypeError(f"Invalid key type: {type(item)}. Must be int or str.")

    @model_validator(mode="before")
    @classmethod
    def unpack_dict(cls, data):
        """This helps directly translate a dict of {"releases": [...]} into just the
        list of releases."""
        if isinstance(data, dict):
            try:
                return data["releases"]
            except KeyError:
                raise ValueError(
                    "Expected a dictionary with a 'releases' key, or else a list of releases"
                )
        return data

    @model_validator(mode="after")
    def check_unique_name(self) -> "ReleaseCollector":
        """Check that all releases have unique names."""
        names = [release.name for release in self.root]
        duplicates = [name for name, count in Counter(names).items() if count > 1]

        if duplicates:
            raise ValueError(
                f"Multiple releases share the following name(s): {', '.join(repr(d) for d in duplicates)}. "
                "Each release must have a unique name."
            )

        return self

    @model_validator(mode="after")
    def check_all_releases_same_type(self):
        """Ensure all releases are of the same type, and set the release_type."""
        release_types = set(r.release_type for r in self.root)
        if len(release_types) > 1:
            type_list = ", ".join(map(str, release_types))
            raise ValueError(
                f"Not all releases have the same type. Received: {type_list}"
            )
        return self

    @property
    def release_type(self):
        release_types = set(r.release_type for r in self.root)
        return release_types.pop()


class CDRForcingDatasetBuilder:
    """Constructs the xarray `Dataset` to be saved as NetCDF."""

    def __init__(self, releases, model_reference_date, release_type: ReleaseType):
        self.releases = releases
        self.model_reference_date = model_reference_date
        self.release_type = release_type

    def build(self) -> xr.Dataset:
        """Build the CDR forcing dataset."""

        all_times = itertools.chain.from_iterable(r.times for r in self.releases)
        unique_times = np.unique(np.array(list(all_times), dtype="datetime64[ns]"))
        unique_rel_times = convert_to_relative_days(
            unique_times, self.model_reference_date
        )

        ds = self._initialize_dataset(unique_times, unique_rel_times)

        for ncdr, release in enumerate(self.releases):
            times = np.array(release.times, dtype="datetime64[ns]")
            rel_times = convert_to_relative_days(times, self.model_reference_date)

            if self.release_type == ReleaseType.volume:
                ds["cdr_volume"].loc[{"ncdr": ncdr}] = np.interp(
                    unique_rel_times, rel_times, release.volume_fluxes.values
                )
                tracer_key = "cdr_tracer"
                tracer_data = release.tracer_concentrations
            elif self.release_type == ReleaseType.tracer_perturbation:
                tracer_key = "cdr_trcflx"
                tracer_data = release.tracer_fluxes

            for ntracer in range(ds.ntracers.size):
                tracer_name = ds.tracer_name[ntracer].item()
                ds[tracer_key].loc[{"ntracers": ntracer, "ncdr": ncdr}] = np.interp(
                    unique_rel_times,
                    rel_times,
                    tracer_data[tracer_name].values,
                )

        return ds

    def _initialize_dataset(self, unique_times, unique_rel_times) -> xr.Dataset:
        """Create and initialize a CDR xarray.Dataset with metadata and empty variables.

        Parameters
        ----------
        unique_times : array-like
            Array of unique absolute times for the release.
        unique_rel_times : array-like
            Array of unique relative times (days since model reference date).

        Returns
        -------
        xr.Dataset
            Initialized dataset with time, location, and release-type-dependent variables.
        """

        ds = xr.Dataset()
        ds["time"] = ("time", unique_times)
        ds["cdr_time"] = ("time", unique_rel_times)
        ds["cdr_lon"] = ("ncdr", [r.lon for r in self.releases])
        ds["cdr_lat"] = ("ncdr", [r.lat for r in self.releases])
        ds["cdr_dep"] = ("ncdr", [r.depth for r in self.releases])
        ds["cdr_hsc"] = ("ncdr", [r.hsc for r in self.releases])
        ds["cdr_vsc"] = ("ncdr", [r.vsc for r in self.releases])
        ds = ds.assign_coords(
            {"release_name": (["ncdr"], [r.name for r in self.releases])}
        )

        if self.release_type == ReleaseType.volume:
            ds = add_tracer_metadata_to_ds(
                ds, with_flux_units=False
            )  # adds the coordinate "tracer_name"
            ds["cdr_volume"] = xr.zeros_like(ds.cdr_time * ds.ncdr, dtype=np.float64)
            ds["cdr_tracer"] = xr.zeros_like(
                ds.cdr_time * ds.ntracers * ds.ncdr, dtype=np.float64
            )

        elif self.release_type == ReleaseType.tracer_perturbation:
            ds = add_tracer_metadata_to_ds(
                ds, with_flux_units=True
            )  # adds the coordinate "tracer_name"
            ds["cdr_trcflx"] = xr.zeros_like(
                ds.cdr_time * ds.ntracers * ds.ncdr, dtype=np.float64
            )

        # Assign attributes
        attr_map = self._get_attr_map()
        for var, attrs in attr_map.items():
            if var in ds.data_vars or var in ds.coords:
                ds[var].attrs.update(attrs)

        return ds

    def _get_attr_map(self) -> dict[str, dict[str, str]]:
        """Returns metadata (long name and units) for variables in the CDRForcing xarray
        dataset.

        Returns
        -------
        dict
            Keys are variable names, values are dicts with 'long_name' and 'units'.
        """
        return {
            "time": {"long_name": "absolute time"},
            "cdr_time": {
                "long_name": f"relative time: days since {self.model_reference_date}",
                "units": "days",
            },
            "release_name": {"long_name": "Name of release"},
            "cdr_lon": {
                "long_name": "Longitude of CDR release",
                "units": "degrees east",
            },
            "cdr_lat": {
                "long_name": "Latitude of CDR release",
                "units": "degrees north",
            },
            "cdr_dep": {"long_name": "Depth of CDR release", "units": "meters"},
            "cdr_hsc": {
                "long_name": "Horizontal scale of CDR release",
                "units": "meters",
            },
            "cdr_vsc": {
                "long_name": "Vertical scale of CDR release",
                "units": "meters",
            },
            "cdr_trcflx": {
                "long_name": "CDR tracer flux",
                "description": "Tracer fluxes for CDR releases",
            },
            "cdr_volume": {
                "long_name": "CDR volume flux",
                "units": "m3/s",
                "description": "Volume flux associated with CDR releases",
            },
            "cdr_tracer": {
                "long_name": "CDR tracer concentration",
                "description": "Tracer concentrations for CDR releases",
            },
        }


class CDRForcing(BaseModel):
    """Represents Carbon Dioxide Removal (CDR) forcing.

    Parameters
    ----------
    grid : Grid, optional
        Object representing the grid.
    start_time : datetime
        Start time of the ROMS model simulation.
    end_time : datetime
        End time of the ROMS model simulation.
    model_reference_date : datetime, optional
        Reference date for converting absolute times to model-relative time. Defaults to Jan 1, 2000.
    releases : list of Release
        A list of one or more CDR release objects.
    """

    grid: Grid | None = None
    """Object representing the grid."""
    start_time: datetime
    """Start time of the ROMS model simulation."""
    end_time: datetime
    """End time of the ROMS model simulation."""
    model_reference_date: datetime = datetime(2000, 1, 1)
    """The reference date for the ROMS simulation."""
    releases: ReleaseCollector
    """A list of one or more CDR release objects."""

    # this is defined during init and shouldn't be serialized
    _ds: xr.Dataset = None

    @model_validator(mode="after")
    def _validate(self):
        if self.start_time >= self.end_time:
            raise ValueError(
                f"`start_time` ({self.start_time}) must be earlier than `end_time` ({self.end_time})."
            )

        for release in self.releases:
            ReleaseSimulationManager(
                release=release,
                grid=self.grid,
                start_time=self.start_time,
                end_time=self.end_time,
            )

        builder = CDRForcingDatasetBuilder(
            self.releases, self.model_reference_date, self.release_type
        )
        self._ds = builder.build()
        return self

    @property
    def release_type(self) -> ReleaseType:
        return self.releases.release_type

    @property
    def ds(self) -> xr.Dataset:
        """The xarray dataset containing release metadata and forcing variables."""
        return self._ds

    def plot_volume_flux(
        self, start=None, end=None, release_names=INCLUDE_ALL_RELEASE_NAMES
    ):
        """Plot the volume flux for each specified release within the given time range.

        Parameters
        ----------
        start : datetime or None
            Start datetime for the plot. If None, defaults to `self.start_time`.
        end : datetime or None
            End datetime for the plot. If None, defaults to `self.end_time`.
        release_names : list[str], or str, optional
            A list of release names to plot.
            If a string equal to "all", all releases will be plotted.
            Defaults to "all".

        Raises
        ------
        ValueError
            If self.releases are not of type VolumeRelease.
            If `release_names` is not a list of strings or "all".
            If any of the specified release names do not exist in `self.releases`.
        """

        if self.release_type != ReleaseType.volume:
            raise ValueError(
                "plot_volume_flux is only supported when all releases are of type VolumeRelease."
            )

        start = start or self.start_time
        end = end or self.end_time

        valid_release_names = [r.name for r in self.releases]

        if release_names == INCLUDE_ALL_RELEASE_NAMES:
            release_names = valid_release_names

        _validate_release_input(release_names, valid_release_names)

        data = self.ds["cdr_volume"]

        self._plot_line(
            data,
            release_names,
            start,
            end,
            title="Volume flux of release(s)",
            ylabel=r"m$^3$/s",
        )

    def plot_tracer_concentration(
        self,
        tracer_name: str,
        start=None,
        end=None,
        release_names=INCLUDE_ALL_RELEASE_NAMES,
    ):
        """Plot the concentration of a given tracer for each specified release within
        the given time range.

        Parameters
        ----------
        tracer_name : str
            Name of the tracer to plot, e.g., "ALK", "DIC", etc.
        start : datetime or None
            Start datetime for the plot. If None, defaults to `self.start_time`.
        end : datetime or None
            End datetime for the plot. If None, defaults to `self.end_time`.
        release_names : list[str], or str, optional
            A list of release names to plot.
            If a string equal to "all", all releases will be plotted.
            Defaults to "all".

        Raises
        ------
        ValueError
            If self.releases are not of type VolumeRelease.
            If `release_names` is not a list of strings or "all".
            If any of the specified release names do not exist in `self.releases`.
            If `tracer_name` does not exist in self.ds["tracer_name"])
        """
        if self.release_type != ReleaseType.volume:
            raise ValueError(
                "plot_tracer_concentration is only supported when all releases are of type VolumeRelease."
            )

        start = start or self.start_time
        end = end or self.end_time

        valid_release_names = [r.name for r in self.releases]

        if release_names == INCLUDE_ALL_RELEASE_NAMES:
            release_names = valid_release_names

        _validate_release_input(release_names, valid_release_names)

        tracer_names = list(self.ds["tracer_name"].values)
        if tracer_name not in tracer_names:
            raise ValueError(
                f"Tracer '{tracer_name}' not found. Available: {', '.join(tracer_names)}"
            )

        tracer_index = tracer_names.index(tracer_name)
        data = self.ds["cdr_tracer"].isel(ntracers=tracer_index)

        if tracer_name == "temp":
            title = "Temperature of release water"
        elif tracer_name == "salt":
            title = "Salinity of release water"
        else:
            title = f"{tracer_name} concentration of release(s)"

        self._plot_line(
            data,
            release_names,
            start,
            end,
            title=title,
            ylabel=f"{self.ds['tracer_unit'].isel(ntracers=tracer_index).values.item()}",
        )

    def plot_tracer_flux(
        self,
        tracer_name: str,
        start=None,
        end=None,
        release_names=INCLUDE_ALL_RELEASE_NAMES,
    ):
        """Plot the flux of a given tracer for each specified release within the given
        time range.

        Parameters
        ----------
        tracer_name : str
            Name of the tracer to plot, e.g., "ALK", "DIC", etc.
        start : datetime or None
            Start datetime for the plot. If None, defaults to `self.start_time`.
        end : datetime or None
            End datetime for the plot. If None, defaults to `self.end_time`.
        release_names : list[str], or str, optional
            A list of release names to plot.
            If a string equal to "all", all releases will be plotted.
            Defaults to "all".

        Raises
        ------
        ValueError
            If self.releases are not of type TracerPerturbation.
            If `release_names` is not a list of strings or "all".
            If any of the specified release names do not exist in `self.releases`.
            If `tracer_name` does not exist in self.ds["tracer_name"])
        """
        if self.release_type != ReleaseType.tracer_perturbation:
            raise ValueError(
                "plot_tracer_flux is only supported when all releases are of type TracerPerturbation."
            )

        start = start or self.start_time
        end = end or self.end_time

        valid_release_names = [r.name for r in self.releases]

        if release_names == INCLUDE_ALL_RELEASE_NAMES:
            release_names = valid_release_names

        _validate_release_input(release_names, valid_release_names)

        tracer_names = list(self.ds["tracer_name"].values)
        if tracer_name not in tracer_names:
            raise ValueError(
                f"Tracer '{tracer_name}' not found. Available: {', '.join(tracer_names)}"
            )

        tracer_index = tracer_names.index(tracer_name)
        data = self.ds["cdr_trcflx"].isel(ntracers=tracer_index)

        title = f"{tracer_name} flux of release(s)"

        self._plot_line(
            data,
            release_names,
            start,
            end,
            title=title,
            ylabel=f"{self.ds['tracer_unit'].isel(ntracers=tracer_index).values.item()}",
        )

    def _plot_line(self, data, release_names, start, end, title="", ylabel=""):
        """Plots a line graph for the specified releases and time range."""
        valid_release_names = [r.name for r in self.releases]
        colors = _get_release_colors(valid_release_names)

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        for name in release_names:
            ncdr = np.where(self.ds["release_name"].values == name)[0].item()
            data.isel(ncdr=ncdr).plot(
                ax=ax,
                linewidth=2,
                label=name,
                color=colors[name],
                marker="x",
            )

        if len(release_names) > 0:
            ax.legend()

        ax.set(title=title, ylabel=ylabel, xlabel="time")
        ax.set_xlim([start, end])

    def plot_locations(self, release_names="all"):
        """Plot centers of release locations in top-down view.

        Parameters
        ----------
        release_names : list of str or "all", optional
            A list of release names to plot.
            If "all", the method will plot all releases.
            The default is "all".

        Raises
        ------
        ValueError
            If `release_names` is not a list of strings or "all".
            If any of the specified release names do not exist in `self.releases`.
            If `self.grid` is not set.
        """

        # Ensure that the grid is provided
        if self.grid is None:
            raise ValueError(
                "A grid must be provided for plotting. Please pass a valid `Grid` object."
            )

        valid_release_names = [r.name for r in self.releases]

        if release_names == "all":
            release_names = valid_release_names

        _validate_release_input(release_names, valid_release_names)

        lon_deg = self.grid.ds.lon_rho
        lat_deg = self.grid.ds.lat_rho
        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)
        trans = _get_projection(lon_deg, lat_deg)
        fig, ax = plt.subplots(1, 1, figsize=(13, 7), subplot_kw={"projection": trans})

        # Plot blue background on map
        field = self.grid.ds.mask_rho
        field = field.assign_coords({"lon": lon_deg, "lat": lat_deg})
        vmax = 6
        vmin = 0
        cmap = plt.colormaps.get_cmap("Blues")
        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}
        _plot(field, kwargs=kwargs, ax=ax, c=None, add_colorbar=False)

        # Plot release locations
        colors = _get_release_colors(valid_release_names)
        _plot_location(
            grid=self.grid,
            releases=[self.releases[name] for name in release_names],
            ax=ax,
            colors=colors,
        )

    def plot_distribution(self, release_name: str, mark_release_center: bool = True):
        """Plot the release location from a top and side view.

        This method creates three plots:

        - A top view of the release distribution.
        - A side view of the release distribution along a fixed longitude.
        - A side view of the release distribution along a fixed latitude.

        Parameters
        ----------
        release_name : str
            Name of the release to plot.
        mark_release_center : bool, default True
            Whether to mark the center of the release distribution with an "x".

        Raises
        ------
        ValueError
            If `self.grid` is not set.
            If the specified `release_name` does not exist in `self.releases`.
        """
        if self.grid is None:
            raise ValueError(
                "A grid must be provided for plotting. Please pass a valid `Grid` object."
            )

        valid_release_names = [r.name for r in self.releases]
        _validate_release_input(release_name, valid_release_names, list_allowed=False)
        release = self.releases[release_name]

        # Prepare grid coordinates
        lon_deg = self.grid.ds.lon_rho
        lat_deg = self.grid.ds.lat_rho
        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)

        # Setup figure
        fig = plt.figure(figsize=(12, 5.5))
        gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
        trans = _get_projection(lon_deg, lat_deg)
        ax0 = fig.add_subplot(gs[:, 0], projection=trans)
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        cmap = plt.colormaps.get_cmap("RdPu")
        cmap.set_bad(color="gray")
        kwargs = {"cmap": cmap}

        # Top down view plot
        horizontal_field = _map_horizontal_gaussian(self.grid, release)
        horizontal_field = horizontal_field.assign_coords(
            {"lon": lon_deg, "lat": lat_deg}
        )
        _plot(
            horizontal_field.where(self.grid.ds.mask_rho),
            kwargs=kwargs,
            ax=ax0,
            c=None,
            add_colorbar=False,
        )
        if mark_release_center:
            _plot_location(
                grid=self.grid, releases=[release], ax=ax0, include_legend=False
            )

        # Side view plots
        kwargs = {
            "cmap": cmap,
            "y": "depth",
            "yincrease": False,
            "add_colorbar": False,
        }
        # Plot along latitude
        vertical_field = _map_vertical_gaussian(
            self.grid, release, horizontal_field, orientation="latitude"
        )
        more_kwargs = {"x": "lat"}
        vertical_field.plot(**kwargs, **more_kwargs, ax=ax1)
        if mark_release_center:
            ax1.plot(
                release.lat,
                release.depth,
                color="k",
                marker="x",
                markersize=8,
                markeredgewidth=2,
            )

        ax1.set(title=f"Longitude: {release.lon}°E", xlabel="Latitude [°N]")
        # Plot along longitude
        vertical_field = _map_vertical_gaussian(
            self.grid, release, horizontal_field, orientation="longitude"
        )
        more_kwargs = {"x": "lon"}
        vertical_field.plot(**kwargs, **more_kwargs, ax=ax2)
        if mark_release_center:
            release_lon = normalize_longitude(release.lon, self.grid.straddle)
            ax2.plot(
                release_lon,
                release.depth,
                color="k",
                marker="x",
                markersize=8,
                markeredgewidth=2,
            )
        ax2.set(title=f"Latitude: {release.lat}°N", xlabel="Longitude [°E]")

        # Adjust layout and title
        fig.subplots_adjust(hspace=0.45)
        fig.suptitle(f"Release distribution for: {release_name}")

    def save(
        self,
        filepath: str | Path,
    ) -> list[Path]:
        """Save the volume source with tracers to netCDF4 file.

        Parameters
        ----------
        filepath : str | Path
            The base path and filename for the output files.

        Returns
        -------
        list[Path]
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

    @model_serializer
    def _serialize(self) -> dict:
        return _to_dict(self)

    def to_yaml(self, filepath: str | Path) -> None:
        """Export the parameters of the class to a YAML file, including the version of
        roms-tools.

        Parameters
        ----------
        filepath : str | Path
            The path to the YAML file where the parameters will be saved.
        """

        forcing_dict = self.model_dump()
        metadata = self.releases[0].get_tracer_metadata()
        forcing_dict["CDRForcing"]["_tracer_metadata"] = metadata

        _write_to_yaml(forcing_dict, filepath)

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> "CDRForcing":
        """Create an instance of the CDRForcing class from a YAML file.

        Parameters
        ----------
        filepath : str | Path
            The path to the YAML file from which the parameters will be read.

        Returns
        -------
        CDRForcing
            An instance of the CDRForcing class.
        """
        filepath = Path(filepath)

        grid = Grid.from_yaml(filepath)
        params = _from_yaml(cls, filepath)
        params.pop("_tracer_metadata", None)

        return cls(grid=grid, **params)


def _validate_release_input(releases, valid_releases, list_allowed=True):
    """Validates the input for release names in plotting methods to ensure they are in
    an acceptable format and exist within the set of valid releases.

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
    invalid_releases = [
        release for release in releases if release not in valid_releases
    ]
    if invalid_releases:
        raise ValueError(f"Invalid releases: {', '.join(invalid_releases)}")


def _get_release_colors(valid_releases: list[str]) -> dict[str, tuple]:
    """Returns a dictionary of colors for the valid releases, based on a consistent
    colormap.

    Parameters
    ----------
    valid_releases : List[str]
        List of release names to assign colors to.

    Returns
    -------
    Dict[str, tuple]
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

    # Determine the colormap based on the number of releases
    if len(valid_releases) <= 10:
        color_map = plt.get_cmap("tab10")
    elif len(valid_releases) <= 20:
        color_map = plt.get_cmap("tab20")
    else:
        color_map = plt.get_cmap("tab20b")

    # Ensure the number of releases doesn't exceed the available colormap capacity
    if len(valid_releases) > color_map.N:
        raise ValueError(
            f"Too many releases. The selected colormap supports up to {color_map.N} releases."
        )

    # Create a dictionary of colors based on the release indices
    colors = {name: color_map(i) for i, name in enumerate(valid_releases)}

    return colors


def _validate_release_location(grid, release: Release):
    """Validates the closest grid location for a release site.

    This function ensures that the given release site (lat, lon, depth) lies
    within the ocean portion of the model grid domain. It:

    - Checks if the point is within the grid domain (with buffer for boundary artifacts).
    - Verifies that the location is not on land.
    - Verifies that the location is not below the seafloor.
    """
    if grid:
        # Adjust longitude based on whether it crosses the International Date Line (straddle case)
        if grid.straddle:
            lon = xr.where(release.lon > 180, release.lon - 360, release.lon)
        else:
            lon = xr.where(release.lon < 0, release.lon + 360, release.lon)

        dx = 1 / grid.ds.pm
        dy = 1 / grid.ds.pn
        max_grid_spacing = np.sqrt(dx**2 + dy**2) / 2

        # Compute great-circle distance to all grid points
        dist = gc_dist(grid.ds.lon_rho, grid.ds.lat_rho, lon, release.lat)
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

        eta_max = grid.ds.sizes["eta_rho"] - 1
        xi_max = grid.ds.sizes["xi_rho"] - 1

        if eta_rho in [0, eta_max] or xi_rho in [0, xi_max]:
            raise ValueError(
                f"Release site '{release.name}' is located too close to the grid boundary. "
                "Place release location (lat, lon) away from grid boundaries."
            )

        if grid.ds.mask_rho[eta_rho, xi_rho].values == 0:
            raise ValueError(
                f"Release site '{release.name}' is on land. "
                "Please provide coordinates (lat, lon) over ocean."
            )

        if grid.ds.h[eta_rho, xi_rho].values < release.depth:
            raise ValueError(
                f"Release site '{release.name}' lies below the seafloor. "
                f"Seafloor depth is {grid.ds.h[eta_rho, xi_rho].values:.2f} m, "
                f"but requested depth is {release.depth:.2f} m. Adjust depth or location (lat, lon)."
            )

    else:
        logging.warning(
            "Grid not provided: cannot verify whether the specified lat/lon/depth location is within the domain or on land. "
            "Please check manually or provide a grid when instantiating the class."
        )


def _map_horizontal_gaussian(grid: Grid, release: Release):
    """Map a tracer release to the ROMS grid as a normalized 2D Gaussian distribution.

    The tracer is centered at the nearest grid cell to the release location, then smoothed
    using a Gaussian filter (via GCM-Filters) with horizontal scale `release.hsc`. Land points
    are masked out, and the distribution is renormalized to integrate to 1 over the ocean.

    Parameters
    ----------
    grid : Grid
        ROMS grid object with grid metrics and land mask.
    release : Release
        Release location and horizontal scale (hsc) in meters.

    Returns
    -------
    delta_smooth : xarray.DataArray
        Normalized 2D tracer distribution on the ROMS grid (zero over land).
    """
    # Find closest grid cell center
    target_coords = get_target_coords(grid)
    lon = release.lon
    if target_coords["straddle"]:
        lon = xr.where(lon > 180, lon - 360, lon)
    else:
        lon = xr.where(lon < 0, lon + 360, lon)
    dist = gc_dist(target_coords["lon"], target_coords["lat"], lon, release.lat)
    dist_min = dist.min(dim=["eta_rho", "xi_rho"])

    # Find the indices of the closest grid cell
    indices = np.where(dist == dist_min)
    eta_rho = indices[0][0]
    xi_rho = indices[1][0]

    # Create a delta function at the center of the Gaussian release
    delta = xr.zeros_like(grid.ds.mask_rho)
    delta[eta_rho, xi_rho] = 1

    # Calculate the grid cell area from the inverse grid metrics (pm, pn)
    area = 1 / (grid.ds.pm * grid.ds.pn)

    # Compute the mean grid spacing (dx) as the average of mean dx and dy
    dx = (((1 / grid.ds.pm).mean() + (1 / grid.ds.pn).mean()) / 2).item()

    # Extend the domain in both dimensions to mitigate boundary effects during filtering
    # - Create a mask of ones ignoring land (will be renormalized later)
    mask = xr.ones_like(grid.ds.mask_rho)

    # Extend the mask along eta_rho by concatenating three copies plus a zeroed edge cell
    margin_mask = xr.concat(
        [mask, mask, mask, 0 * mask.isel(eta_rho=-1)], dim="eta_rho"
    )
    # Similarly extend along xi_rho
    margin_mask = xr.concat(
        [margin_mask, margin_mask, margin_mask, 0 * margin_mask.isel(xi_rho=-1)],
        dim="xi_rho",
    )

    # Extend the delta function in the same manner to match the enlarged domain
    delta_extended = xr.concat(
        [0 * delta, delta, 0 * delta, 0 * delta.isel(eta_rho=-1)], dim="eta_rho"
    )
    delta_extended = xr.concat(
        [
            0 * delta_extended,
            delta_extended,
            0 * delta_extended,
            0 * delta_extended.isel(xi_rho=-1),
        ],
        dim="xi_rho",
    )

    # Extend the cell area array to match the enlarged domain
    area_extended = xr.concat([area, area, area, area.isel(eta_rho=-1)], dim="eta_rho")
    area_extended = xr.concat(
        [area_extended, area_extended, area_extended, area_extended.isel(xi_rho=-1)],
        dim="xi_rho",
    )

    # Define Gaussian filter parameters:
    # - filter_scale is computed so that the Gaussian's std dev matches
    #   the equivalent boxcar filter of width equal to release.hsc / dx
    #   multiplied by sqrt(12) to convert from boxcar to Gaussian std dev.
    filter_scale = release.hsc / dx * np.sqrt(12)

    filter = gcm_filters.Filter(
        filter_scale=filter_scale,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
        grid_vars={"wet_mask": margin_mask, "area": area_extended},
    )

    # Apply the Gaussian filter to the extended delta function over spatial dimensions
    delta_smooth = filter.apply(delta_extended, dims=["eta_rho", "xi_rho"])

    # Remove the extended margins to return to the original domain size
    delta_smooth = delta_smooth.isel(
        eta_rho=slice(grid.ds.sizes["eta_rho"], -grid.ds.sizes["eta_rho"] - 1),
        xi_rho=slice(grid.ds.sizes["xi_rho"], -grid.ds.sizes["eta_rho"] - 1),
    )

    # Mask out land cells and set values to zero there
    delta_smooth = delta_smooth.where(grid.ds.mask_rho, 0.0)

    # Renormalize so the integral over ocean points sums to 1
    integral = delta_smooth.sum(dim=["eta_rho", "xi_rho"])
    delta_smooth = delta_smooth / integral

    return delta_smooth


def _map_vertical_gaussian(grid, release, field, orientation="latitude"):
    """Extract a vertical section from a ROMS grid and apply a Gaussian distribution in
    depth.

    This function interpolates a horizontally Gaussian field (e.g., from a tracer release)
    along either a constant latitude or longitude section and distributes it vertically using
    a Gaussian profile centered around the release depth.

    Parameters
    ----------
    grid : Grid
        ROMS grid object with methods and attributes used for horizontal resolution,
        depth computation, and straddling logic.
    release : Release
        Release object containing coordinates (`lat`, `lon`, `depth`) and vertical
        spread (`vsc`) for the Gaussian distribution.
    field : xr.DataArray
        2D horizontal tracer field defined on the ROMS grid.
    orientation : {"latitude", "longitude"}, default "latitude"
        Orientation of the extracted vertical section. If "latitude", extracts
        a section along constant longitude. If "longitude", extracts a section
        along constant latitude.

    Returns
    -------
    vertical_field : xr.DataArray
        2D field (depth vs. latitude or longitude) with the vertically-distributed
        Gaussian mapped along the specified section.
    """
    meters_per_degree = 2 * np.pi * R_EARTH / 360
    release_lon = normalize_longitude(release.lon, grid.straddle)

    if orientation == "longitude":
        lon_deg = grid.ds.lon_rho
        if grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)
        hsc_in_degrees = release.hsc / (meters_per_degree * np.cos(release.lat))
        lons, _ = _generate_focused_coordinate_range(
            center=release_lon,
            sc=hsc_in_degrees,
            min_val=lon_deg.min().values,
            max_val=lon_deg.max().values,
            N=2000,
        )
        lons = xr.DataArray(lons, dims=["lon"], attrs={"units": "°E"})
        lats = [release.lat]

    elif orientation == "latitude":
        lons = [release_lon]
        hsc_in_degrees = release.hsc / (
            meters_per_degree * np.cos(grid.ds.lat_rho.mean().values)
        )
        lats, _ = _generate_focused_coordinate_range(
            center=release.lat,
            sc=hsc_in_degrees,
            min_val=grid.ds.lat_rho.min().values,
            max_val=grid.ds.lat_rho.max().values,
            N=2000,
        )
        lats = xr.DataArray(lats, dims=["lat"], attrs={"units": "°N"})
    else:
        raise ValueError(
            "`section_orientation` must be either 'latitude' or 'longitude'."
        )

    # Regrid 2D horizontal Gaussian onto desired 1D horizontal section
    target_coords = {"lat": lats, "lon": lons}
    lateral_regrid = LateralRegridFromROMS(field, target_coords)
    field = lateral_regrid.apply(field).squeeze()
    h = lateral_regrid.apply(grid.ds.h).squeeze()

    # Define depth levels
    depth_levels, _ = _generate_focused_coordinate_range(
        center=release.depth,
        sc=release.vsc,
        min_val=0.0,
        max_val=h.max().values,
        N=2000,
    )
    depth_levels = xr.DataArray(
        np.asarray(depth_levels),
        dims=["depth"],
        attrs={"long_name": "Depth", "units": "m"},
    )
    depth_levels = depth_levels.astype(np.float32)

    if release.vsc == 0:
        # Find index of depth_level closest to release.depth
        closest_idx = np.abs(depth_levels - release.depth).argmin()
        weights = xr.zeros_like(depth_levels)
        weights[closest_idx] = 1.0
    else:
        # Compute Gaussian weights at each layer center
        weights = np.exp(-0.5 * ((depth_levels - release.depth) / release.vsc) ** 2)
        weights /= weights.sum()

    # Redistribute Gaussian mass from under topography to open ocean
    weights = weights.where(depth_levels < h)
    weights = weights / weights.sum()

    # Map 1D to 2D Gaussian
    vertical_field = field * weights

    # Remove NaNs at the edges
    if orientation == "longitude":
        vertical_field, _ = _remove_edge_nans(vertical_field, "lon", layer_depth=h)
    if orientation == "latitude":
        vertical_field, _ = _remove_edge_nans(vertical_field, "lat", layer_depth=h)

    vertical_field = vertical_field.assign_coords({"depth": depth_levels})

    return vertical_field


def _plot_location(
    grid: Grid,
    releases: ReleaseCollector,
    ax: Axes,
    colors: dict[str, tuple] | None = None,
    include_legend: bool = True,
) -> None:
    """Plot the center location of each release on a top-down map view.

    Each release is represented as a point on the map, with its color
    determined by the `colors` dictionary.

    Parameters
    ----------
    grid : Grid
        The grid object defining the spatial extent and coordinate system for the plot.

    releases : ReleaseCollector
        Collection of `Release` objects to plot. Each `Release` must have `.lat`, `.lon`,
        and `.name` attributes.

    ax : matplotlib.axes.Axes
        The Matplotlib axis object to plot on.

    colors : dict of str to tuple, optional
        Optional dictionary mapping release names to RGBA color tuples. If not provided,
        all releases are plotted in a default color (`"#dd1c77"`).

    include_legend : bool, default True
        Whether to include a legend showing release names.

    Returns
    -------
    None
    """

    lon_deg = grid.ds.lon_rho
    lat_deg = grid.ds.lat_rho
    if grid.straddle:
        lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)
    trans = _get_projection(lon_deg, lat_deg)

    proj = ccrs.PlateCarree()

    for release in releases:
        # transform coordinates to projected space
        transformed_lon, transformed_lat = trans.transform_point(
            release.lon,
            release.lat,
            proj,
        )

        if colors is not None:
            color = colors[release.name]
        else:
            color = "k"

        ax.plot(
            transformed_lon,
            transformed_lat,
            marker="x",
            markersize=8,
            markeredgewidth=2,
            label=release.name,
            color=color,
        )

    if include_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))
