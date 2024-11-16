import xarray as xr
import numpy as np
from dataclasses import dataclass, field, asdict
from roms_tools.setup.grid import Grid
from datetime import datetime
from typing import Dict, Union, List
from roms_tools.setup.datasets import DaiRiverDataset
from pathlib import Path
import matplotlib.pyplot as plt
from roms_tools.setup.utils import (
    get_target_coords,
    gc_dist
)
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
        Start time of the desired surface forcing data.
    end_time : datetime
        End time of the desired surface forcing data.
    source : Dict[str, Union[str, Path, List[Union[str, Path]]], bool]
        Dictionary specifying the source of the surface forcing data. Keys include:

          - "name" (str): Name of the data source (e.g., "ERA5").
          - "path" (Union[str, Path, List[Union[str, Path]]]): The path to the raw data file(s). This can be:

            - A single string (with or without wildcards).
            - A single Path object.
            - A list of strings or Path objects containing multiple files.
          - "climatology" (bool): Indicates if the data is climatology data. Defaults to False.

    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.

    """

    grid: Grid
    start_time: datetime
    end_time: datetime
    source: Dict[str, Union[str, Path, List[Union[str, Path]]]]
    model_reference_date: datetime = datetime(2000, 1, 1)

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        self._input_checks()
        target_coords = get_target_coords(self.grid)
        # maximum dx in grid
        dx = (np.sqrt((1 / self.grid.ds.pm)**2 + (1 / self.grid.ds.pn)**2) / 2).max()

        data = self._get_data()

        original_indices = data.extract_relevant_rivers(target_coords, dx)
        object.__setattr__(self, "original_indices", original_indices)
        
        self._create_river_forcing(data)

        updated_indices = self.move_rivers_to_coast(target_coords, data)
        object.__setattr__(self, "updated_indices", updated_indices)
        self.write_indices_into_grid_file()

    
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
            raise ValueError(
                'Only "DAI" is a valid option for source["name"].'
            )


        return data
    
    def _create_river_forcing(self, data):
        
        ds = xr.Dataset()
        
        river_volume = (data.ds[data.var_names["flux"]] * data.ds[data.var_names["ratio"]]).astype(np.float32)
        river_volume.attrs["long_name"] = "River volume flux"
        river_volume.attrs["units"] = "m^3/s"
        river_volume = river_volume.rename({data.dim_names["time"]: "river_time", data.dim_names["station"]: "nriver"})
        river_volume.coords["nriver"] = data.ds[data.var_names["name"]].rename({data.dim_names["station"]: "nriver"})
        ds["river_volume"] = river_volume

        tracer_data = np.zeros((len(ds.river_time), len(ds.nriver), 2), dtype=float)
        tracer_data[:, :, 0] = 17.0
        tracer_data[:, :, 1] = 1.0
    
        river_tracer = xr.DataArray(tracer_data, dims=("river_time", "nriver", "ntracers"))
        river_tracer.coords['ntracers'] = ['temperature', 'salinity']
        
        ds["river_tracer"] = river_tracer
        
        object.__setattr__(self, "ds", ds)

    def move_rivers_to_coast(self, target_coords, data):
        
        # Retrieve longitude and latitude of river mouths 
        river_lon = data.ds[data.var_names["longitude"]]
        river_lat = data.ds[data.var_names["latitude"]]
    
        # Adjust longitude based on whether it crosses the International Date Line (straddle case)
        if target_coords["straddle"]:
            river_lon = xr.where(river_lon > 180, river_lon - 360, river_lon)
        else:
            river_lon = xr.where(river_lon < 0, river_lon + 360, river_lon)

        mask = self.grid.ds.mask_rho
        faces = mask.shift(eta_rho=1) + mask.shift(eta_rho=-1) + mask.shift(xi_rho=1) + mask.shift(xi_rho=-1)
        
        # We want all grid points on land that are adjacent to the ocean
        coast = (1-mask) * (faces>0)
        dist_coast = gc_dist(target_coords["lon"].where(coast), target_coords["lat"].where(coast), river_lon, river_lat).transpose(data.dim_names["station"], "eta_rho", "xi_rho")
        dist_coast_min = dist_coast.min(dim=["eta_rho", "xi_rho"])
        
        # Find the indices of the closest coastal grid cell to the river mouth
        indices = np.where(dist_coast == dist_coast_min)
        names = data.ds[data.var_names["name"]].isel({data.dim_names["station"]: indices[0]}).values
        
        # Return the indices in a dictionary format
        indices = {
            "station": indices[0],
            "eta_rho": indices[1],
            "xi_rho": indices[2],
            "name": names
        }

        return indices

    def write_indices_into_grid_file(self):
        river_locations = xr.zeros_like(self.grid.mask_rho)
        for i in range(len(self.updated_indices["name"])):
            name =  self.updated_indices["name"][i]
            eta_index =  self.updated_indices["eta_rho"][i]
            xi_index =  self.updated_indices["xi_rho"][i]
            river_locations[eta_index, xi_index] = 1

            self.grid.ds["river_flux"] = river_locations

    def plot(self):

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

        fig, axs = plt.subplots(1, 2, figsize=(13, 13), subplot_kw={"projection": trans})

        for ax in axs:
            _add_plot_to_ax(ax, lon_deg, lat_deg, trans, field, c=None, add_colorbar=False, kwargs=kwargs)

        for ax, indices in zip(axs, [self.original_indices, self.updated_indices]):
            for i in range(len(indices["name"])):
                name = indices["name"][i]
                xi_index = indices["xi_rho"][i]
                eta_index = indices["eta_rho"][i]
                # transform coordinates to projected space
                proj = ccrs.PlateCarree()
                transformed_lon, transformed_lat = trans.transform_point(self.grid.ds.lon_rho[eta_index, xi_index], self.grid.ds.lat_rho[eta_index, xi_index], proj)
                ax.plot(transformed_lon, transformed_lat, marker='x', markersize=8, markeredgewidth=2, label=name)

        axs[0].set_title("Original river locations")
        axs[1].set_title("Updated river locations")
        
        axs[1].legend(loc='center left', bbox_to_anchor=(1.1, 0.5))


