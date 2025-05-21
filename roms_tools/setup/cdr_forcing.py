from datetime import datetime
from pathlib import Path
from typing import Annotated, Iterator
from pydantic import (
    BaseModel,
    model_validator,
    Field,
    model_serializer,
    RootModel,
    conlist,
)
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import logging
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from roms_tools import Grid
from roms_tools.plot import _plot, _get_projection
from roms_tools.regrid import LateralRegridFromROMS
from roms_tools.utils import (
    _generate_coordinate_range,
    _remove_edge_nans,
    save_datasets,
)
from roms_tools.setup.utils import (
    convert_to_relative_days,
    gc_dist,
    add_tracer_metadata_to_ds,
    _to_dict,
    _write_to_yaml,
    _from_yaml,
)
from roms_tools.setup.cdr_release import (
    Release,
    VolumeRelease,
    TracerPerturbation,
    ReleaseType,
)


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

    def __getitem__(self, item) -> Release:
        return self.root[item]

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
        Object representing the grid for spatial context.
    start_time : datetime
        Start time of the ROMS model simulation.
    end_time : datetime
        End time of the ROMS model simulation.
    model_reference_date : datetime, optional
        Reference date for converting absolute times to model-relative time. Defaults to Jan 1, 2000.
    releases : list of Release
        A list of one or more CDR release objects.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray dataset containing release metadata and forcing variables.
    """

    grid: Grid | None = None
    start_time: datetime
    end_time: datetime
    model_reference_date: datetime = datetime(2000, 1, 1)
    releases: ReleaseCollector

    # this is defined during init and shouldn't be serialized
    _ds: xr.Dataset = None

    @model_validator(mode="after")
    def validate(self):
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
        return self._ds

    def plot_volume_flux(self, start=None, end=None, release_names="all"):
        """Plot the volume flux for each specified release within the given time range.

        Parameters
        ----------
        start : datetime or None
            Start datetime for the plot. If None, defaults to `self.start_time`.
        end : datetime or None
            End datetime for the plot. If None, defaults to `self.end_time`.
        release_names : list of str, or "all", optional
            A list of release names to plot.
            If "all", the method will plot all releases.
            The default is "all".

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

        if release_names == "all":
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
        self, tracer_name: str, start=None, end=None, release_names="all"
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
        release_names : list of str, or "all", optional
            A list of release names to plot.
            If "all", the method will plot all releases.
            The default is "all".

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

        if release_names == "all":
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
        self, tracer_name: str, start=None, end=None, release_names="all"
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
        release_names : list of str, or "all", optional
            A list of release names to plot.
            If "all", the method will plot all releases.
            The default is "all".

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

        if release_names == "all":
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

    def plot_location_top_view(self, release_names="all"):
        """Plot the top-down view of release locations.

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

        colors = _get_release_colors(valid_release_names)

        for name in release_names:
            ncdr = np.where(self.ds["release_name"].values == name)[0].item()

            # transform coordinates to projected space
            transformed_lon, transformed_lat = trans.transform_point(
                self.ds.cdr_lon.isel(ncdr=ncdr).item(),
                self.ds.cdr_lat.isel(ncdr=ncdr).item(),
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

    def plot_location_side_view(self, release_name: str = None):
        """Plot the release location from a side view, showing bathymetry sections along
        both fixed longitude and latitude.

        This method creates two plots:

        - A bathymetry section along a fixed longitude (latitudinal view),
          with the release location marked by an "x".
        - A bathymetry section along a fixed latitude (longitudinal view),
          with the release location also marked by an "x".

        Parameters
        ----------
        release_name : str, optional
            Name of the release to plot. If only one release is available,
            it is used by default. If multiple releases are available, this must be specified.

        Raises
        ------
        ValueError

            If `self.grid` is not set.
            If the specified `release_name` does not exist in `self.releases`.
            If no `release_name` is provided when multiple releases are available.
        """
        if self.grid is None:
            raise ValueError(
                "A grid must be provided for plotting. Please pass a valid `Grid` object."
            )

        valid_release_names = [r.name for r in self.releases]

        if release_name is None:
            if len(valid_release_names) == 1:
                release_name = valid_release_names[0]
            else:
                raise ValueError(
                    f"Multiple releases found: {valid_release_names}. Please specify a single release via `release_name` to plot."
                )

        _validate_release_input(release_name, valid_release_names, list_allowed=False)

        # Prepare grid coordinates
        lon_deg = self.grid.ds.lon_rho
        lat_deg = self.grid.ds.lat_rho
        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)

        resolution = self.grid._infer_nominal_horizontal_resolution()
        h = self.grid.ds.h.assign_coords({"lon": lon_deg, "lat": lat_deg})

        ncdr = np.where(self.ds["release_name"].values == release_name)[0].item()

        # Set up plot
        fig, axs = plt.subplots(2, 1, figsize=(7, 8))

        # Plot along fixed longitude
        _plot_bathymetry_section(
            ax=axs[0],
            h=h,
            dim="lon",
            fixed_val=self.ds.cdr_lon.isel(ncdr=ncdr),
            coord_deg=lat_deg,
            resolution=resolution,
            title=f"Longitude: {self.ds.cdr_lon.isel(ncdr=ncdr).item()}°E",
        )

        colors = _get_release_colors(valid_release_names)

        axs[0].plot(
            self.ds.cdr_lat.isel(ncdr=ncdr),
            self.ds.cdr_dep.isel(ncdr=ncdr),
            color=colors[release_name],
            marker="x",
            markersize=8,
            markeredgewidth=2,
        )

        # Plot along fixed latitude
        _plot_bathymetry_section(
            ax=axs[1],
            h=h,
            dim="lat",
            fixed_val=self.ds.cdr_lat.isel(ncdr=ncdr),
            coord_deg=lon_deg,
            resolution=resolution,
            title=f"Latitude: {self.ds.cdr_lat.isel(ncdr=ncdr).item()}°N",
        )
        axs[1].plot(
            self.ds.cdr_lon.isel(ncdr=ncdr),
            self.ds.cdr_dep.isel(ncdr=ncdr),
            color=colors[release_name],
            marker="x",
            markersize=8,
            markeredgewidth=2,
        )

        # Adjust layout and title
        fig.subplots_adjust(hspace=0.4)
        fig.suptitle(f"Release location for: {release_name}")

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
    def serialize(self) -> dict:
        return _to_dict(self)

    def to_yaml(self, filepath: str | Path) -> None:
        """Export the parameters of the class to a YAML file, including the version of
        roms-tools.

        Parameters
        ----------
        filepath : str | Path
            The path to the YAML file where the parameters will be saved.
        """

        # Serialize object into dictionary
        forcing_dict = self.model_dump()

        # Write to YAML
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


def _get_release_colors(valid_releases):
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


def _plot_bathymetry_section(
    ax: Axes,
    h: xr.DataArray,
    dim: str,
    fixed_val: float,
    coord_deg: xr.DataArray,
    resolution: float,
    title: str,
) -> None:
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
