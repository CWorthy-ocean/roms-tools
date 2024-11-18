import xarray as xr
import numpy as np
from dataclasses import dataclass, field
from roms_tools.setup.grid import Grid
from datetime import datetime
from typing import Dict, Union, List
from roms_tools.setup.datasets import DaiRiverDataset
from pathlib import Path
import matplotlib.pyplot as plt
from roms_tools.setup.utils import get_target_coords, gc_dist
from roms_tools.setup.plot import _get_projection, _add_plot_to_ax
import cartopy.crs as ccrs


@dataclass(frozen=True, kw_only=True)
class RiverForcing:
    """Represents river forcing input data for ROMS.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
    start_time : datetime
        Start time of the desired river forcing data.
    end_time : datetime
        End time of the desired river forcing data.
    source : Dict[str, Union[str, Path, List[Union[str, Path]]], bool]
        Dictionary specifying the source of the river forcing data. Keys include:

          - "name" (str): Name of the data source (e.g., "DAI").
          - "path" (Union[str, Path, List[Union[str, Path]]]): The path to the raw data file(s). This can be:

            - A single string (with or without wildcards).
            - A single Path object.
            - A list of strings or Path objects containing multiple files.
          - "climatology" (bool): Indicates if the data is climatology data. Defaults to False.

    convert_to_climatology : str, optional
        Determines when to compute climatology for river forcing. Options are:
          - "if_any_missing" (default): Compute climatology for all rivers if any river has missing values.
          - "never": Do not compute climatology.
          - "always": Compute climatology for all rivers, regardless of missing data.

    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.
    """

    grid: Grid
    start_time: datetime
    end_time: datetime
    source: Dict[str, Union[str, Path, List[Union[str, Path]]]]
    convert_to_climatology: str = "if_any_missing"
    model_reference_date: datetime = datetime(2000, 1, 1)

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        self._input_checks()
        target_coords = get_target_coords(self.grid)
        # maximum dx in grid
        dx = (
            np.sqrt((1 / self.grid.ds.pm) ** 2 + (1 / self.grid.ds.pn) ** 2) / 2
        ).max()

        data = self._get_data()

        original_indices = data.extract_relevant_rivers(target_coords, dx)
        object.__setattr__(self, "original_indices", original_indices)

        self._create_river_forcing(data)

        self.move_rivers_to_closest_coast(target_coords, data)

    def _input_checks(self):
        # Ensure 'source' dictionary contains required keys
        if "name" not in self.source:
            raise ValueError("`source` must include a 'name'.")
        if "path" not in self.source:
            raise ValueError("`source` must include a 'path'.")

        # Set 'climatology' to False if not provided in 'source'
        object.__setattr__(
            self,
            "source",
            {**self.source, "climatology": self.source.get("climatology", False)},
        )

    def _get_data(self):

        data_dict = {
            "filename": self.source["path"],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "climatology": self.source["climatology"],
        }

        if self.source["name"] == "DAI":
            data = DaiRiverDataset(**data_dict)
        else:
            raise ValueError('Only "DAI" is a valid option for source["name"].')

        return data

    def _create_river_forcing(self, data):
        """Create river forcing data for volume flux and tracers.

        This method computes river volume flux and associated tracers (temperature
        and salinity) based on the provided input data. It creates a new dataset that
        contains:
        - `river_volume`: The river volume flux, calculated as the product of
          river flux and a specified ratio, with units of m³/s.
        - `river_tracer`: A tracer array for temperature and salinity at each river
          station over time.

        Parameters:
        -----------
        data : object
            An object containing the dataset and necessary variables. It must have the following attributes:
            - `ds`: The dataset containing the river flux and ratio data.
            - `var_names`: A dictionary of variable names in the dataset (e.g., "flux", "ratio", "name").
            - `dim_names`: A dictionary of dimension names (e.g., "time", "station").

        Returns:
        --------
        None
            This method creates a new dataset `ds` containing the river volume and
            tracer data, and stores it as an attribute of the object.
        """
        if not self.source["climatology"]:
            if self.convert_to_climatology in ["never", "if_any_missing"]:
                data_ds = data.select_relevant_times(data.ds)
                if self.convert_to_climatology == "if_any_missing":
                    if data_ds[data.var_names["flux"]].isnull().any():
                        data.compute_climatology()
                        self.source["climatology"] = True
                    else:
                        object.__setattr__(data, "ds", data_ds)
                else:
                    object.__setattr__(data, "ds", data_ds)
            elif self.convert_to_climatology == "always":
                data.compute_climatology()
                self.source["climatology"] = True

        ds = xr.Dataset()

        river_volume = (
            data.ds[data.var_names["flux"]] * data.ds[data.var_names["ratio"]]
        ).astype(np.float32)
        river_volume.attrs["long_name"] = "River volume flux"
        river_volume.attrs["units"] = "m^3/s"
        river_volume = river_volume.rename(
            {data.dim_names["time"]: "river_time", data.dim_names["station"]: "nriver"}
        )
        name = data.ds[data.var_names["name"]].rename(
            {data.dim_names["station"]: "nriver"}
        )
        river_volume.coords["nriver"] = name
        ds["river_volume"] = river_volume

        tracer_data = np.zeros((len(ds.river_time), len(ds.nriver), 2), dtype=float)
        tracer_data[:, :, 0] = 17.0
        tracer_data[:, :, 1] = 1.0

        river_tracer = xr.DataArray(
            tracer_data, dims=("river_time", "nriver", "ntracers")
        )
        river_tracer.coords["ntracers"] = ["temperature", "salinity"]

        ds["river_tracer"] = river_tracer

        object.__setattr__(self, "ds", ds)

    def move_rivers_to_closest_coast(self, target_coords, data):
        """Move river mouths to the closest coastal grid cell.

        This method computes the closest coastal grid point to each river mouth
        based on geographical distance.

        Parameters:
        -----------
        target_coords : dict
            A dictionary containing the following keys:
            - "lon" (xarray.DataArray): Longitude coordinates of the target grid points.
            - "lat" (xarray.DataArray): Latitude coordinates of the target grid points.
            - "straddle" (bool): A flag indicating whether the river mouth crosses the International Date Line.

        data : object
            An object that contains the dataset and related variables. It must have the following attributes:
            - `ds`: The dataset containing river information.
            - `var_names`: A dictionary of variable names in the dataset (e.g., longitude, latitude, station names).
            - `dim_names`: A dictionary containing dimension names for the dataset (e.g., "station", "eta_rho", "xi_rho").

        Returns:
        --------
        None
            This method modifies the `self.updated_indices` attribute and writes the updated indices
            of the river mouths to the grid file using `write_indices_into_grid_file`.
        """

        # Retrieve longitude and latitude of river mouths
        river_lon = data.ds[data.var_names["longitude"]]
        river_lat = data.ds[data.var_names["latitude"]]

        # Adjust longitude based on whether it crosses the International Date Line (straddle case)
        if target_coords["straddle"]:
            river_lon = xr.where(river_lon > 180, river_lon - 360, river_lon)
        else:
            river_lon = xr.where(river_lon < 0, river_lon + 360, river_lon)

        mask = self.grid.ds.mask_rho
        faces = (
            mask.shift(eta_rho=1)
            + mask.shift(eta_rho=-1)
            + mask.shift(xi_rho=1)
            + mask.shift(xi_rho=-1)
        )

        # We want all grid points on land that are adjacent to the ocean
        coast = (1 - mask) * (faces > 0)
        dist_coast = gc_dist(
            target_coords["lon"].where(coast),
            target_coords["lat"].where(coast),
            river_lon,
            river_lat,
        ).transpose(data.dim_names["station"], "eta_rho", "xi_rho")
        dist_coast_min = dist_coast.min(dim=["eta_rho", "xi_rho"])

        # Find the indices of the closest coastal grid cell to the river mouth
        indices = np.where(dist_coast == dist_coast_min)
        names = (
            data.ds[data.var_names["name"]]
            .isel({data.dim_names["station"]: indices[0]})
            .values
        )

        # Return the indices in a dictionary format
        indices = {
            "station": indices[0],
            "eta_rho": indices[1],
            "xi_rho": indices[2],
            "name": names,
        }
        self.write_indices_into_grid_file(indices)
        object.__setattr__(self, "updated_indices", indices)

    def write_indices_into_grid_file(self, indices):
        """Writes river location indices into the grid dataset as the "river_flux"
        variable.

        This method checks if the variable "river_flux" already exists in the grid dataset.
        If it does, the method removes it. Then, it creates a new variable called "river_flux"
        based on the provided `indices` and assigns it to the dataset. The `indices` dictionary
        provides information about the river station locations and their corresponding grid
        cell indices (eta_rho and xi_rho).

        Parameters:
        -----------
        indices : dict
            A dictionary containing information about river station locations.
            It must contain the following keys:
            - "name" (list or array): Names of the river stations.
            - "station" (list or array): River station identifiers.
            - "eta_rho" (list or array): The eta (row) index for each river location.
            - "xi_rho" (list or array): The xi (column) index for each river location.

        Returns:
        --------
        None
            This method modifies the grid's dataset in-place by adding the "river_flux" variable.
        """

        if "river_flux" in self.grid.ds:
            ds = self.grid.ds.drop_vars("river_flux")
            object.__setattr__(self.grid, "ds", ds)

        river_locations = xr.zeros_like(self.grid.ds.mask_rho)
        for i in range(len(indices["name"])):
            station = indices["station"][i]
            eta_index = indices["eta_rho"][i]
            xi_index = indices["xi_rho"][i]
            river_locations[eta_index, xi_index] = station + 1

            self.grid.ds["river_flux"] = river_locations

    def plot_locations(self):
        """Plots the original and updated river locations on a map projection."""

        field = self.grid.ds.mask_rho
        field = field.assign_coords(
            {"lon": self.grid.ds.lon_rho, "lat": self.grid.ds.lat_rho}
        )
        vmax = 3
        vmin = 0
        cmap = plt.colormaps.get_cmap("Blues")
        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

        lon_deg = field.lon
        lat_deg = field.lat

        # check if North or South pole are in domain
        if lat_deg.max().values > 89 or lat_deg.min().values < -89:
            raise NotImplementedError(
                "Plotting is not implemented for the case that the domain contains the North or South pole."
            )

        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)

        trans = _get_projection(lon_deg, lat_deg)

        lon_deg = lon_deg.values
        lat_deg = lat_deg.values

        fig, axs = plt.subplots(
            1, 2, figsize=(13, 13), subplot_kw={"projection": trans}
        )

        for ax in axs:
            _add_plot_to_ax(
                ax,
                lon_deg,
                lat_deg,
                trans,
                field,
                c=None,
                add_colorbar=False,
                kwargs=kwargs,
            )

        for ax, indices in zip(axs, [self.original_indices, self.updated_indices]):
            for i in range(len(indices["name"])):
                name = indices["name"][i]
                xi_index = indices["xi_rho"][i]
                eta_index = indices["eta_rho"][i]
                # transform coordinates to projected space
                proj = ccrs.PlateCarree()
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
                )

        axs[0].set_title("Original river locations")
        axs[1].set_title("Updated river locations")

        axs[1].legend(loc="center left", bbox_to_anchor=(1.1, 0.5))

    def plot(self):
        """Plots the river volume flux for all rivers."""
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))

        if self.source["climatology"]:
            xticks = np.arange(1, 13)
            xlabel = "months"
        else:
            xticks = self.ds.river_time.values
            xlabel = "time"
        for i in range(len(self.ds.nriver)):

            ax.plot(
                xticks,
                self.ds.isel(nriver=i).river_volume.values,
                marker="x",
                markersize=8,
                markeredgewidth=2,
                lw=2,
                label=self.ds.isel(nriver=i).nriver.values,
            )

        ax.set_xticks(xticks)
        ax.set_xlabel(xlabel)
        if not self.source["climatology"]:
            n = len(self.ds.river_time)
            ticks = self.ds.river_time.values[:: n // 6 + 1]
            ax.set_xticks(ticks)
        ax.set_ylabel(f"${self.ds.river_volume.units}$")
        ax.set_title(self.ds.river_volume.long_name)
        ax.grid()
        ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))
