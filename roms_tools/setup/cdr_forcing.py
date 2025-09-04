import itertools
import logging
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import (
    BaseModel,
    Field,
    RootModel,
    model_serializer,
    model_validator,
)

from roms_tools import Grid
from roms_tools.constants import MAX_DISTINCT_COLORS
from roms_tools.plot import (
    assign_category_colors,
    get_projection,
    plot,
    plot_2d_horizontal_field,
    plot_location,
)
from roms_tools.setup.cdr_release import (
    Release,
    ReleaseType,
    TracerPerturbation,
    VolumeRelease,
)
from roms_tools.setup.utils import (
    add_tracer_metadata_to_ds,
    convert_to_relative_days,
    from_yaml,
    gc_dist,
    get_target_coords,
    get_tracer_metadata_dict,
    to_dict,
    validate_names,
    write_to_yaml,
)
from roms_tools.utils import (
    normalize_longitude,
    save_datasets,
)
from roms_tools.vertical_coordinate import compute_depth_coordinates

INCLUDE_ALL_RELEASE_NAMES = "all"
MAX_RELEASES_TO_PLOT = 20  # must be <= MAX_DISTINCT_COLORS


class ReleaseSimulationManager(BaseModel):
    """Validates and adjusts a single release against a ROMS simulation time window and
    grid.
    """

    release: Release
    grid: Grid | None = None
    start_time: datetime
    end_time: datetime

    @model_validator(mode="after")
    def check_release_times_within_simulation_window(
        self,
    ) -> "ReleaseSimulationManager":
        """Ensure the release times are within the [start_time, end_time] simulation
        window.
        """
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

    root: Annotated[
        list[
            Annotated[
                VolumeRelease | TracerPerturbation, Field(discriminator="release_type")
            ]
        ],
        Field(alias="releases", min_length=1),
    ]

    _release_type: ReleaseType | None = None

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

    def __len__(self):
        return len(self.root)

    @model_validator(mode="before")
    @classmethod
    def unpack_dict(cls, data):
        """This helps directly translate a dict of {"releases": [...]} into just the
        list of releases.
        """
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
        """Type of all releases."""
        release_types = set(r.release_type for r in self.root)
        return release_types.pop()


class CDRForcingDatasetBuilder:
    """Constructs the xarray `Dataset` to be saved as NetCDF."""

    def __init__(
        self,
        releases: ReleaseCollector,
        model_reference_date: datetime,
        release_type: ReleaseType,
    ):
        """
        Initialize the dataset builder.

        Parameters
        ----------
        releases : list
            List of release objects.
        model_reference_date : datetime
            Reference date for relative time conversion.
        release_type : ReleaseType
            Type of release.
        """
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
        """Type of the release."""
        return self.releases.release_type

    @property
    def ds(self) -> xr.Dataset:
        """The xarray dataset containing release metadata and forcing variables."""
        return self._ds

    def plot_volume_flux(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        release_names: list[str] | str = INCLUDE_ALL_RELEASE_NAMES,
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

        release_names = _validate_release_names(release_names, self.releases)

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
        start: datetime | None = None,
        end: datetime | None = None,
        release_names: list[str] | str = INCLUDE_ALL_RELEASE_NAMES,
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

        release_names = _validate_release_names(release_names, self.releases)

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
        start: datetime | None = None,
        end: datetime | None = None,
        release_names: list[str] | str = INCLUDE_ALL_RELEASE_NAMES,
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

        release_names = _validate_release_names(release_names, self.releases)

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
        if len(valid_release_names) > MAX_DISTINCT_COLORS:
            colors = assign_category_colors(release_names)
        else:
            colors = assign_category_colors(valid_release_names)

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

    def plot_locations(
        self, release_names: list[str] | str = INCLUDE_ALL_RELEASE_NAMES
    ):
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

        release_names = _validate_release_names(release_names, self.releases)

        lon_deg = self.grid.ds.lon_rho
        lat_deg = self.grid.ds.lat_rho
        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)
        trans = get_projection(lon_deg, lat_deg)
        fig, ax = plt.subplots(1, 1, figsize=(13, 7), subplot_kw={"projection": trans})

        # Plot blue background on map
        field = self.grid.ds.mask_rho
        field = field.where(field)
        field = field.assign_coords({"lon": lon_deg, "lat": lat_deg})
        vmax = 6
        vmin = 0
        cmap = plt.colormaps.get_cmap("Blues")
        cmap.set_bad(color="gray")
        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}
        plot_2d_horizontal_field(field, kwargs=kwargs, ax=ax, add_colorbar=False)

        # Plot release locations
        valid_release_names = [r.name for r in self.releases]
        if len(valid_release_names) > MAX_DISTINCT_COLORS:
            colors = assign_category_colors(release_names)
        else:
            colors = assign_category_colors(valid_release_names)
        plot_location(
            grid_ds=self.grid.ds,
            points={
                name: {
                    "lat": self.releases[name].lat,
                    "lon": self.releases[name].lon,
                    "color": colors.get(name, "k"),
                }
                for name in release_names
            },
            ax=ax,
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

        if not isinstance(release_name, str):
            raise ValueError(
                f"Only a single release name (string) is allowed. Got: {release_name!r}"
            )

        release_name = _validate_release_names([release_name], self.releases)[0]

        release = self.releases[release_name]

        # Prepare grid coordinates
        lon_deg = self.grid.ds.lon_rho
        lat_deg = self.grid.ds.lat_rho
        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)

        # Setup figure
        fig = plt.figure(figsize=(12, 5.5))
        gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
        trans = get_projection(lon_deg, lat_deg)
        ax0 = fig.add_subplot(gs[:, 0], projection=trans)
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])

        # Top down view plot of depth-integrated Gaussian
        depth_integrated_field = _map_horizontal_gaussian(self.grid, release)
        depth_integrated_field = depth_integrated_field.assign_coords(
            {"lon": lon_deg, "lat": lat_deg}
        )
        cmap = plt.colormaps.get_cmap("RdPu")
        cmap.set_bad(color="gray")
        plot_2d_horizontal_field(
            field=depth_integrated_field.where(self.grid.ds.mask_rho),
            ax=ax0,
            kwargs={"cmap": cmap},
            add_colorbar=False,
            title="Depth-integrated distribution",
        )
        if mark_release_center:
            plot_location(
                grid_ds=self.grid.ds,
                points={
                    release.name: {
                        "lat": release.lat,
                        "lon": release.lon,
                    }
                },
                ax=ax0,
                include_legend=False,
            )

        # Spread horizontal Gaussian field into the vertical
        distribution_3d = _map_3d_gaussian(self.grid, release, depth_integrated_field)

        release_lon = normalize_longitude(release.lon, self.grid.straddle)

        plot(
            field=distribution_3d,
            grid_ds=self.grid.ds,
            lon=release_lon,
            ax=ax1,
            cmap_name="RdPu",
            add_colorbar=False,
        )
        ax1.set(title=f"Longitude: {release.lon}°E", xlabel="Latitude [°N]")

        plot(
            field=distribution_3d,
            grid_ds=self.grid.ds,
            lat=release.lat,
            ax=ax2,
            cmap_name="RdPu",
            add_colorbar=False,
        )
        ax2.set(title=f"Latitude: {release.lat}°N", xlabel="Longitude [°E]")

        if mark_release_center:
            kwargs = {
                "color": "k",
                "marker": "x",
                "markersize": 8,
                "markeredgewidth": 2,
            }
            ax1.plot(release.lat, release.depth, **kwargs)
            ax2.plot(release_lon, release.depth, **kwargs)

        # Adjust layout and title
        fig.subplots_adjust(hspace=0.45)
        fig.suptitle(f"Release distribution for: {release_name}")

    def compute_total_releases(self, dt: float) -> pd.DataFrame:
        """
        Compute integrated tracer quantities for all releases and return a DataFrame.

        Parameters
        ----------
        dt : float
            Time step in seconds for reconstructing ROMS time stamps.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per release and one row of units at the top.
            Columns 'temp' and 'salt' are excluded from integrated totals.
        """
        # Reconstruct ROMS time stamps
        _, rel_days = _reconstruct_roms_time_stamps(
            self.start_time, self.end_time, dt, self.model_reference_date
        )

        # Collect accounting results for all releases
        records = []
        release_names = []
        for release in self.releases:
            result = release._do_accounting(rel_days, self.model_reference_date)
            records.append(result)
            release_names.append(getattr(release, "name", f"release_{len(records)}"))

        # Build DataFrame: rows = releases, columns = tracer names
        df = pd.DataFrame(records, index=release_names)

        # Exclude temp and salt from units row and integrated totals
        integrated_tracers = [col for col in df.columns if col not in ("temp", "salt")]

        # Add a row of units only for integrated tracers
        tracer_meta = get_tracer_metadata_dict(include_bgc=True, unit_type="integrated")
        units_row = {
            col: tracer_meta.get(col, {}).get("units", "") for col in integrated_tracers
        }

        df_units = pd.DataFrame([units_row], index=["units"])

        # Keep only integrated_tracers columns in df, drop temp and salt
        df_integrated = df[integrated_tracers]

        # Concatenate units row on top
        df_final = pd.concat([df_units, df_integrated])

        # Store dt as metadata
        df_final.attrs["time_step"] = dt
        df_final.attrs["start_time"] = self.start_time
        df_final.attrs["end_time"] = self.end_time
        df_final.attrs["title"] = "Integrated tracer releases"

        return df_final

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
        return to_dict(self)

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

        write_to_yaml(forcing_dict, filepath)

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
        params = from_yaml(cls, filepath)
        params.pop("_tracer_metadata", None)

        return cls(grid=grid, **params)


def _validate_release_names(
    release_names: list[str] | str, releases: ReleaseCollector
) -> list[str]:
    """
    Validate and filter a list of release names.

    Ensures that each release name exists in `releases` and limits the list
    to `MAX_RELEASES_TO_PLOT` entries with a warning if truncated.

    Parameters
    ----------
    release_names : list of str or INCLUDE_ALL_RELEASE_NAMES
        Names of releases to plot, or sentinel to include all.
    releases : ReleaseCollector
        Object containing valid release names.

    Returns
    -------
    list of str
        Validated and truncated list of release names.

    Raises
    ------
    ValueError
        If any names are invalid.
    """
    return validate_names(
        release_names,
        [r.name for r in releases],
        INCLUDE_ALL_RELEASE_NAMES,
        MAX_RELEASES_TO_PLOT,
        label="release",
    )


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

        # Compute the maximum half-diagonal grid spacing across the domain
        max_grid_spacing = (np.sqrt(dx**2 + dy**2) / 2).max()

        # Apply a 10% safety margin
        max_grid_spacing *= 1.1

        # Compute great-circle distance to all grid points
        dist = gc_dist(grid.ds.lon_rho, grid.ds.lat_rho, lon, release.lat)
        dist_min = dist.min(dim=["eta_rho", "xi_rho"])

        if dist_min > max_grid_spacing:
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

    if release.hsc == 0:
        # Find the indices of the closest grid cell
        indices = dist.argmin(dim=["eta_rho", "xi_rho"])
        eta_rho = indices["eta_rho"].values
        xi_rho = indices["xi_rho"].values

        # Create a delta function at the center of the Gaussian release
        distribution_2d = xr.zeros_like(grid.ds.mask_rho)
        distribution_2d[eta_rho, xi_rho] = 1

    else:
        frac = np.exp(-((dist / release.hsc) ** 2))
        distribution_2d = frac.where(frac > 1e-3, 0.0)

        # Mask out land
        distribution_2d = distribution_2d.where(grid.ds.mask_rho, 0.0)

        # Renormalize so the integral over ocean points sums to 1
        integral = distribution_2d.sum(dim=["eta_rho", "xi_rho"])
        distribution_2d = distribution_2d / integral

    return distribution_2d


def _map_3d_gaussian(
    grid: Grid, release: Release, distribution_2d: xr.DataArray
) -> xr.DataArray:
    """Extends 2D Gaussian to 3D Gaussian.

    Parameters
    ----------
    grid : Grid
        ROMS grid object with methods and attributes used for horizontal resolution,
        depth computation, and straddling logic.
    release : Release
        Release object containing coordinates (`lat`, `lon`, `depth`) and vertical
        spread (`vsc`) for the Gaussian distribution.
    distribution_2d : xr.DataArray
        2D horizontal tracer field defined on the ROMS grid.

    Returns
    -------
    xr.DataArray
        3D tracer distribution (z vs lat/lon) with Gaussian vertical structure.
    """
    # Compute depth at rho-points (3D: s_rho, eta_rho, xi_rho)
    depth = compute_depth_coordinates(
        grid.ds, zeta=0, depth_type="layer", location="rho"
    )

    # Initialize 3D distribution with zeros
    distribution_3d = xr.zeros_like(depth)

    if release.vsc == 0:
        # Find vertical index closest to release depth
        abs_diff = abs(depth - release.depth)
        vertical_idx = abs_diff.argmin(dim="s_rho")
        # Stack 2D distribution at that vertical level
        distribution_3d[{"s_rho": vertical_idx}] = distribution_2d
    else:
        # Compute vertical Gaussian shape
        exponent = -(((depth - release.depth) / release.vsc) ** 2)
        vertical_profile = np.exp(exponent)

        # Apply vertical Gaussian scaling
        distribution_3d = distribution_2d * vertical_profile

        # Normalize
        distribution_3d /= release.vsc * np.sqrt(np.pi)
        distribution_3d /= distribution_3d.sum()

    return distribution_3d


def _reconstruct_roms_time_stamps(
    start_time: datetime,
    end_time: datetime,
    dt: float,
    model_reference_date: datetime,
) -> tuple[list[datetime], np.ndarray]:
    """
    Reconstruct ROMS time stamps between `start_time` and `end_time` with step `dt`.

    Uses `convert_to_relative_days` to express times relative to the
    model reference date in ROMS convention (days since reference date).

    Parameters
    ----------
    start_time : datetime
        Beginning of the time series.
    end_time : datetime
        End of the time series (inclusive if it falls exactly on a step).
    dt : float
        Time step in seconds (can be fractional if needed).
    model_reference_date : datetime
        The reference date for ROMS time (elapsed time will be relative to this).

    Returns
    -------
    times : list of datetime
        Sequence of datetimes from `start_time` to `end_time`.
    rel_days : np.ndarray
        Array of elapsed times in **days** relative to `model_reference_date`.

    Raises
    ------
    ValueError
        If `end_time` is not after `start_time` or if `dt` is not positive.
    """
    if end_time <= start_time:
        raise ValueError("end_time must be after start_time")
    if dt <= 0:
        raise ValueError("dt must be positive")

    # Generate absolute times
    delta = timedelta(seconds=dt)
    times: list[datetime] = []
    t = start_time
    while t <= end_time:
        times.append(t)
        t += delta

    # Convert to relative ROMS time (days since model_reference_date)
    rel_days = convert_to_relative_days(times, model_reference_date)

    return times, rel_days
