import xarray as xr
import numpy as np
import logging
from dataclasses import dataclass, field
import cartopy.crs as ccrs
from datetime import datetime
from typing import Dict, Union, List
from pathlib import Path
import matplotlib.pyplot as plt
from roms_tools.grid import Grid
from roms_tools.plot import _get_projection, _add_field_to_ax
from roms_tools.setup.datasets import DaiRiverDataset
from roms_tools.setup.utils import (
    get_target_coords,
    gc_dist,
    substitute_nans_by_fillvalue,
    convert_to_roms_time,
    save_datasets,
    _to_yaml,
    _from_yaml,
    get_variable_metadata,
)


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
    source : Dict[str, Union[str, Path, List[Union[str, Path]]], bool], optional
        Dictionary specifying the source of the river forcing data. Keys include:

          - "name" (str): Name of the data source (e.g., "DAI").
          - "path" (Union[str, Path, List[Union[str, Path]]]): The path to the raw data file(s). This can be:

            - A single string (with or without wildcards).
            - A single Path object.
            - A list of strings or Path objects containing multiple files.
          - "climatology" (bool): Indicates if the data is climatology data. Defaults to False.

        The default is the Dai and Trenberth global river dataset (updated in May 2019), which does not require a path.

    convert_to_climatology : str, optional
        Determines when to compute climatology for river forcing. Options are:
          - "if_any_missing" (default): Compute climatology for all rivers if any river has missing values.
          - "never": Do not compute climatology.
          - "always": Compute climatology for all rivers, regardless of missing data.

    include_bgc : bool, optional
        Whether to include BGC tracers. Defaults to `False`.
    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the river forcing data.
    climatology : bool
        Indicates whether the final river forcing is climatological.
    """

    grid: Grid
    start_time: datetime
    end_time: datetime
    source: Dict[str, Union[str, Path, List[Union[str, Path]]]] = None
    convert_to_climatology: str = "if_any_missing"
    include_bgc: bool = False
    model_reference_date: datetime = datetime(2000, 1, 1)

    ds: xr.Dataset = field(init=False, repr=False)
    climatology: xr.Dataset = field(init=False, repr=False)

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

        if len(original_indices["station"]) > 0:
            self._move_rivers_to_closest_coast(target_coords, data)
            ds = self._create_river_forcing(data)
            self._validate(ds)

            for var_name in ds.data_vars:
                ds[var_name] = substitute_nans_by_fillvalue(
                    ds[var_name], fill_value=0.0
                )

            object.__setattr__(self, "ds", ds)

        else:
            raise ValueError(
                "No relevant rivers found. Consider increasing domain size or using a different river dataset."
            )

    def _input_checks(self):
        if self.source is None:
            object.__setattr__(self, "source", {"name": "DAI"})

        if "name" not in self.source:
            raise ValueError("`source` must include a 'name'.")
        if "path" not in self.source:
            if self.source["name"] != "DAI":
                raise ValueError("`source` must include a 'path'.")

        # Set 'climatology' to False if not provided in 'source'
        object.__setattr__(
            self,
            "source",
            {**self.source, "climatology": self.source.get("climatology", False)},
        )

    def _get_data(self):

        data_dict = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "climatology": self.source["climatology"],
        }

        if self.source["name"] == "DAI":
            if "path" in self.source.keys():
                data_dict["filename"] = self.source["path"]
            data = DaiRiverDataset(**data_dict)
        else:
            raise ValueError('Only "DAI" is a valid option for source["name"].')

        return data

    def _create_river_forcing(self, data):
        """Create river forcing data for volume flux and tracers (temperature, salinity,
        BGC tracers).

        This method computes the river volume flux and associated tracers (temperature, salinity, BGC tracers)
        based on the provided input data. It generates a new `xarray.Dataset` that contains:
        - `river_volume`: The river volume flux, calculated as the product of river flux and a specified ratio, with units of m³/s.
        - `river_tracer`: A tracer array containing temperature, salinity, and BGC tracer values for each river over time.

        The method also handles climatological adjustments for missing or incomplete data, depending on the `convert_to_climatology` setting.

        Parameters
        ----------
        data : object
            An object containing the necessary dataset and variables for river forcing creation. The object must have the following attributes:
            - `ds`: The dataset containing the river flux, ratio, and other related variables.
            - `var_names`: A dictionary mapping variable names (e.g., `"flux"`, `"ratio"`, `"name"`) to the corresponding variable names in the dataset.
            - `dim_names`: A dictionary mapping dimension names (e.g., `"time"`, `"station"`) to the corresponding dimension names in the dataset.

        Returns
        -------
        xr.Dataset
            A new `xarray.Dataset` containing the computed river forcing data. The dataset includes:
            - `river_volume`: A `DataArray` representing the river volume flux (m³/s).
            - `river_tracer`: A `DataArray` representing tracer data for temperature, salinity and BGC tracers (if specified) for each river over time.
        """
        if self.source["climatology"]:
            object.__setattr__(self, "climatology", True)
        else:
            if self.convert_to_climatology in ["never", "if_any_missing"]:
                data_ds = data.select_relevant_times(data.ds)
                if self.convert_to_climatology == "if_any_missing":
                    if data_ds[data.var_names["flux"]].isnull().any():
                        data.compute_climatology()
                        object.__setattr__(self, "climatology", True)
                    else:
                        object.__setattr__(data, "ds", data_ds)
                        object.__setattr__(self, "climatology", False)
                else:
                    object.__setattr__(data, "ds", data_ds)
                    object.__setattr__(self, "climatology", False)
            elif self.convert_to_climatology == "always":
                data.compute_climatology()
                object.__setattr__(self, "climatology", True)

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
        name.attrs["long_name"] = "River name"
        river_volume.coords["river_name"] = name
        ds["river_volume"] = river_volume

        if self.include_bgc:
            ntracers = 2 + 32
        else:
            ntracers = 2
        tracer_data = np.zeros(
            (len(ds.river_time), ntracers, len(ds.nriver)), dtype=np.float32
        )
        tracer_data[:, 0, :] = 17.0
        tracer_data[:, 1, :] = 1.0
        tracer_data[:, 2:, :] = 0.0

        river_tracer = xr.DataArray(
            tracer_data, dims=("river_time", "ntracers", "nriver")
        )
        river_tracer.attrs["long_name"] = "River tracer data"

        if self.include_bgc:
            tracer_names = xr.DataArray(
                [
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
                ],
                dims="ntracers",
            )
        else:
            tracer_names = xr.DataArray(["temp", "salt"], dims="ntracers")
        tracer_names.attrs["long_name"] = "Tracer name"
        river_tracer.coords["tracer_name"] = tracer_names
        ds["river_tracer"] = river_tracer

        ds, time = convert_to_roms_time(
            ds, self.model_reference_date, self.climatology, time_name="river_time"
        )

        ds = ds.assign_coords({"river_time": time})

        ds = ds.drop_vars("nriver")

        return ds

    def _move_rivers_to_closest_coast(self, target_coords, data):
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
        self._write_indices_into_grid_file(indices)
        object.__setattr__(self, "updated_indices", indices)

    def _write_indices_into_grid_file(self, indices):
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
            river_locations[eta_index, xi_index] = station + 2

        river_locations.attrs["long_name"] = "River volume flux partition"
        river_locations.attrs["units"] = "none"
        self.grid.ds["river_flux"] = river_locations

    def _validate(self, ds):
        """Validates the dataset by checking for NaN values in river forcing data.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to validate.

        Raises
        ------
        Warning
            If NaN values are found in any of the dataset variables, a warning message is logged.
        """

        for var_name in ds.data_vars:
            da = ds[var_name]
            if da.isnull().any().values:
                logging.warning(
                    f"NaN values detected in the '{var_name}' field. These values are being set to zero. "
                    "This may indicate missing river data, which could affect model accuracy. Consider setting "
                    "`convert_to_climatology = 'if_any_missing'` to automatically fill missing values with climatological data."
                )

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
            _add_field_to_ax(
                ax,
                lon_deg,
                lat_deg,
                field,
                add_colorbar=False,
                kwargs=kwargs,
            )
            # Add gridlines with labels for latitude and longitude
            gridlines = ax.gridlines(
                draw_labels=True, linewidth=0.5, color="gray", alpha=0.7, linestyle="--"
            )
            gridlines.top_labels = False  # Hide top labels
            gridlines.right_labels = False  # Hide right labels
            gridlines.xlabel_style = {
                "size": 10,
                "color": "black",
            }  # Customize longitude label style
            gridlines.ylabel_style = {
                "size": 10,
                "color": "black",
            }  # Customize latitude label style

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

    def plot(self, var_name="river_volume"):
        """Plots the river flux (e.g., volume, temperature, or salinity) over time for
        all rivers.

        This method generates a time-series plot for a specified river-related variable (such as river volume,
        river temperature, or river salinity) for each river in the dataset. It can handle climatology data
        as well as time series data, and customizes the x-axis and labels accordingly.

        Parameters
        ----------
        var_name : str, optional
            The variable to plot. It can be one of the following:

            - 'river_volume' : river volume flux.
            - 'river_temp' : river temperature (from river_tracer).
            - 'river_salt' : river salinity (from river_tracer).
            - 'river_PO4' : river PO4 (from river_tracer).
            - 'river_NO3' : river NO3 (from river_tracer).
            - 'river_SiO3' : river SiO3 (from river_tracer).
            - 'river_NH4' : river NH4 (from river_tracer).
            - 'river_Fe' : river Fe (from river_tracer).
            - 'river_Lig' : river Lig (from river_tracer).
            - 'river_O2' : river O2 (from river_tracer).
            - 'river_DIC' : river DIC (from river_tracer).
            - 'river_DIC_ALT_CO2' : river DIC_ALT_CO2 (from river_tracer).
            - 'river_ALK' : river ALK (from river_tracer).
            - 'river_ALK_ALT_CO2' : river ALK_ALT_CO2 (from river_tracer).
            - 'river_DOC' : river DOC (from river_tracer).
            - 'river_DON' : river DON (from river_tracer).
            - 'river_DOP' : river DOP (from river_tracer).
            - 'river_DOPr' : river DOPr (from river_tracer).
            - 'river_DONr' : river DONr (from river_tracer).
            - 'river_DOCr' : river DOCr (from river_tracer).
            - 'river_zooC' : river zooC (from river_tracer).
            - 'river_spChl' : river sphChl (from river_tracer).
            - 'river_spC' : river spC (from river_tracer).
            - 'river_spP' : river spP (from river_tracer).
            - 'river_spFe' : river spFe (from river_tracer).
            - 'river_spCaCO3' : river spCaCO3 (from river_tracer).
            - 'river_diatChl' : river diatChl (from river_tracer).
            - 'river_diatC' : river diatC (from river_tracer).
            - 'river_diatP' : river diatP (from river_tracer).
            - 'river_diatFe' : river diatFe (from river_tracer).
            - 'river_diatSi' : river diatSi (from river_tracer).
            - 'river_diazChl' : river diazChl (from river_tracer).
            - 'river_diazC' : river diazC (from river_tracer).
            - 'river_diazP' : river diazP (from river_tracer).
            - 'river_diazFe' : river diazFe (from river_tracer).

            The default is 'river_volume'.
        """
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))

        if self.climatology:
            xticks = self.ds.month.values
            xlabel = "month"
        else:
            xticks = self.ds.abs_time.values
            xlabel = "time"

        if var_name == "river_volume":
            field = self.ds[var_name]
            units = f"${self.ds.river_volume.units}$"
            long_name = self.ds[var_name].long_name
        else:
            d = get_variable_metadata()
            var_name_wo_river = var_name.split("_")[1]
            field = self.ds["river_tracer"].isel(
                ntracers=self.ds.tracer_name == var_name_wo_river
            )
            units = d[var_name_wo_river]["units"]
            long_name = f"River {d[var_name_wo_river]['long_name']}"

        for i in range(len(self.ds.nriver)):

            ax.plot(
                xticks,
                field.isel(nriver=i),
                marker="x",
                markersize=8,
                markeredgewidth=2,
                lw=2,
                label=self.ds.isel(nriver=i).river_name.values,
            )

        ax.set_xticks(xticks)
        ax.set_xlabel(xlabel)
        if not self.climatology:
            n = len(self.ds.river_time)
            ticks = self.ds.abs_time.values[:: n // 6 + 1]
            ax.set_xticks(ticks)
        ax.set_ylabel(units)
        ax.set_title(long_name)
        ax.grid()
        ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))

    def save(
        self,
        filepath: Union[str, Path],
        filepath_grid: Union[str, Path],
        np_eta: int = None,
        np_xi: int = None,
    ) -> None:
        """Save the river forcing and grid file to netCDF4 files. The grid file is
        required because a new field `river_flux` has been added.

        This method allows saving the river forcing and grid data either each as a single file or each partitioned into multiple files, based on the provided options. The dataset can be saved in two modes:

        1. **Single File Mode (default)**:
            - If both `np_eta` and `np_xi` are `None`, the entire dataset is saved as a single netCDF4 file.
            - The file is named based on the provided `filepath`, with `.nc` automatically appended to the filename.

        2. **Partitioned Mode**:
            - If either `np_eta` or `np_xi` is specified, the dataset is partitioned spatially along the `eta` and `xi` axes into tiles.
            - Each tile is saved as a separate netCDF4 file. Filenames will be modified with an index to represent each partition, e.g., `"filepath_YYYYMM.0.nc"`, `"filepath_YYYYMM.1.nc"`, etc.

        Parameters
        ----------
        filepath : Union[str, Path]
            The base path and filename for the output files. The filenames will include the specified path and the `.nc` extension.
            If partitioning is used, additional indices will be appended to the filenames, e.g., `"filepath.0.nc"`, `"filepath.1.nc"`, etc.

        filepath_grid : Union[str, Path]
            The base path and filename for saving the grid file.

        np_eta : int, optional
            The number of partitions along the `eta` direction. If `None`, no spatial partitioning is performed along the `eta` axis.

        np_xi : int, optional
            The number of partitions along the `xi` direction. If `None`, no spatial partitioning is performed along the `xi` axis.

        Returns
        -------
        List[Path]
            A list of `Path` objects for the saved files. Each element in the list corresponds to a file that was saved.
        """

        # Ensure filepath is a Path object
        filepath = Path(filepath)
        filepath_grid = Path(filepath_grid)

        # Remove ".nc" suffix if present
        if filepath.suffix == ".nc":
            filepath = filepath.with_suffix("")
        if filepath_grid.suffix == ".nc":
            filepath_grid = filepath_grid.with_suffix("")

        dataset_list = [self.ds, self.grid.ds]
        output_filenames = [str(filepath), str(filepath_grid)]

        saved_filenames = save_datasets(
            dataset_list, output_filenames, np_eta=np_eta, np_xi=np_xi
        )

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
    def from_yaml(cls, filepath: Union[str, Path]) -> "RiverForcing":
        """Create an instance of the RiverForcing class from a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file from which the parameters will be read.

        Returns
        -------
        RiverForcing
            An instance of the RiverForcing class.
        """
        filepath = Path(filepath)

        grid = Grid.from_yaml(filepath)
        params = _from_yaml(cls, filepath)

        return cls(grid=grid, **params)
