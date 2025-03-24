import xarray as xr
import numpy as np
import logging
from dataclasses import dataclass, field
import cartopy.crs as ccrs
from datetime import datetime
from typing import Dict, Union, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from roms_tools import Grid
from roms_tools.plot import _get_projection, _add_field_to_ax
from roms_tools.utils import save_datasets
from roms_tools.setup.datasets import DaiRiverDataset
from roms_tools.setup.utils import (
    get_target_coords,
    gc_dist,
    substitute_nans_by_fillvalue,
    convert_to_roms_time,
    _to_yaml,
    _from_yaml,
    get_variable_metadata,
)


@dataclass(kw_only=True)
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
    indices : dict[str, list[tuple]], optional
        A dictionary specifying the river indices for each river to be included in the river forcing. This parameter is optional. If not provided,
        the river indices will be automatically determined based on the grid and the source dataset. If provided, it allows for explicit specification
        of river locations. The dictionary structure consists of river names as keys, and each value is a list of tuples. Each tuple represents
        a pair of indices corresponding to the `eta_rho` and `xi_rho` grid coordinates of the river.

        Example:
            indices = {
                'Hvita(Olfusa)': [(8, 6), (7, 6)],
                'Thjorsa': [(8, 6)],
                'JkulsFjll': [(11, 12)],
                'Lagarfljot': [(9, 13), (8, 13), (10, 13)],
                'Bruara': [(8, 6)],
                'Svarta': [(12, 9)]
            }

        In the example, the dictionary provides the river names as keys, and the values are lists of tuples, where each tuple represents the
        `(eta_rho, xi_rho)` indices for a river location.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the river forcing data.
    climatology : bool
        Indicates whether the final river forcing is climatological.
    Dict[str, Union[int, List[int]]]
        A dictionary of river indices. If not provided during initialization, it will be automatically determined
        based on the grid and the source dataset. The dictionary structure is the same as described in the `indices` parameter docstring.
    """

    grid: Grid
    start_time: datetime
    end_time: datetime
    source: Dict[str, Union[str, Path, List[Union[str, Path]]]] = None
    convert_to_climatology: str = "if_any_missing"
    include_bgc: bool = False
    model_reference_date: datetime = datetime(2000, 1, 1)
    indices: Optional[Dict[str, Dict[str, Union[int, List[int]]]]] = None

    ds: xr.Dataset = field(init=False, repr=False)
    climatology: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        self._input_checks()
        data = self._get_data()

        if self.indices is None:
            logging.info(
                "No river indices provided. Identify all rivers within the ROMS domain and assign each of them to the nearest coastal point."
            )
            target_coords = get_target_coords(self.grid)
            # maximum dx in grid
            dx = (
                np.sqrt((1 / self.grid.ds.pm) ** 2 + (1 / self.grid.ds.pn) ** 2) / 2
            ).max()
            original_indices = data.extract_relevant_rivers(target_coords, dx)
            if len(original_indices) == 0:
                raise ValueError(
                    "No relevant rivers found. Consider increasing domain size or using a different river dataset."
                )
            self.original_indices = original_indices
            updated_indices = self._move_rivers_to_closest_coast(target_coords, data)
            self.indices = updated_indices

        else:
            logging.info("Use provided river indices.")
            self.original_indices = self.indices
            check_river_locations_are_along_coast(self.grid.ds.mask_rho, self.indices)
            data.extract_named_rivers(self.indices)

        ds = self._create_river_forcing(data)
        ds = self._write_indices_into_dataset(ds)
        self._validate(ds)

        for var_name in ds.data_vars:
            ds[var_name] = substitute_nans_by_fillvalue(ds[var_name], fill_value=0.0)

        self.ds = ds

    def _input_checks(self):
        if self.source is None:
            self.source = {"name": "DAI"}

        if "name" not in self.source:
            raise ValueError("`source` must include a 'name'.")
        if "path" not in self.source:
            if self.source["name"] != "DAI":
                raise ValueError("`source` must include a 'path'.")

        # Set 'climatology' to False if not provided in 'source'
        self.source = {
            **self.source,
            "climatology": self.source.get("climatology", False),
        }

        # Check if 'indices' is provided and has the correct format
        if self.indices is not None:
            if not isinstance(self.indices, dict):
                raise ValueError("`indices` must be a dictionary.")

            # Ensure the dictionary contains at least one river
            if len(self.indices) == 0:
                raise ValueError(
                    "The provided 'indices' dictionary must contain at least one river."
                )

            for river_name, river_data in self.indices.items():
                if not isinstance(river_name, str):
                    raise ValueError(f"River name `{river_name}` must be a string.")

                if not isinstance(river_data, list):
                    raise ValueError(
                        f"Data for river `{river_name}` must be a list of tuples."
                    )

                # Ensure each element in the list is a tuple of length 2
                seen_tuples = set()
                for idx_pair in river_data:
                    if not isinstance(idx_pair, tuple) or len(idx_pair) != 2:
                        raise ValueError(
                            f"Each item for river `{river_name}` must be a tuple of length 2 representing (eta_rho, xi_rho)."
                        )

                    eta_rho, xi_rho = idx_pair

                    # Ensure both eta_rho and xi_rho are integers
                    if not isinstance(eta_rho, int):
                        raise ValueError(
                            f"First element of tuple for river `{river_name}` must be an integer (eta_rho), but got {type(eta_rho)}."
                        )
                    if not isinstance(xi_rho, int):
                        raise ValueError(
                            f"Second element of tuple for river `{river_name}` must be an integer (xi_rho), but got {type(xi_rho)}."
                        )

                    # Check that eta_rho and xi_rho are within the valid range
                    if not (0 <= eta_rho < len(self.grid.ds.eta_rho)):
                        raise ValueError(
                            f"Value of eta_rho for river `{river_name}` ({eta_rho}) is out of valid range [0, {len(self.grid.ds.eta_rho)-1}]."
                        )
                    if not (0 <= xi_rho < len(self.grid.ds.xi_rho)):
                        raise ValueError(
                            f"Value of xi_rho for river `{river_name}` ({xi_rho}) is out of valid range [0, {len(self.grid.ds.xi_rho)-1}]."
                        )

                    # Check for duplicate tuples
                    if idx_pair in seen_tuples:
                        raise ValueError(
                            f"Duplicate location {idx_pair} found for river `{river_name}`."
                        )
                    seen_tuples.add(idx_pair)

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
            self.climatology = True
        else:
            if self.convert_to_climatology in ["never", "if_any_missing"]:
                data_ds = data.select_relevant_times(data.ds)
                if self.convert_to_climatology == "if_any_missing":
                    if data_ds[data.var_names["flux"]].isnull().any():
                        data.compute_climatology()
                        self.climatology = True
                    else:
                        data.ds = data_ds
                        self.climatology = False
                else:
                    data.ds = data_ds
                    self.climatology = False
            elif self.convert_to_climatology == "always":
                data.compute_climatology()
                self.climatology = True

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

        nriver = xr.DataArray(np.arange(1, len(ds.nriver) + 1), dims="nriver")
        nriver.attrs["long_name"] = "River ID (1-based Fortran indexing)"
        ds = ds.assign_coords({"nriver": nriver})

        if self.include_bgc:
            ntracers = 2 + 32
        else:
            ntracers = 2
        tracer_data = np.zeros(
            (len(ds.river_time), ntracers, len(ds.nriver)), dtype=np.float32
        )
        tracer_data[:, 0, :] = 17.0
        tracer_data[:, 1, :] = 1.0
        tracer_data[:, 2:, :] = 10.0

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

        return ds

    def _move_rivers_to_closest_coast(self, target_coords, data):
        """Move river mouths to the closest coastal grid cell.

        This method computes the closest coastal grid point to each river mouth
        based on geographical distance. It identifies the nearest grid point on the coast and returns the updated river mouth indices.

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

        Returns
        -------
        indices : dict[str, list[tuple]]
            A dictionary consisting of river names as keys, and each value is a list of tuples. Each tuple represents
            a pair of indices corresponding to the `eta_rho` and `xi_rho` grid coordinates of the river.
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
        stations = indices[0]
        eta_rho_values = indices[1]
        xi_rho_values = indices[2]
        names = (
            data.ds[data.var_names["name"]]
            .isel({data.dim_names["station"]: stations})
            .values
        )
        # Return the indices in a dictionary format
        river_indices = {}
        for i in range(len(stations)):
            river_name = names[i]
            river_indices[river_name] = [
                (int(eta_rho_values[i]), int(xi_rho_values[i]))
            ]  # list of tuples

        return river_indices

    def _write_indices_into_dataset(self, ds):
        """Adds river location indices to the dataset as the "river_location" variable.

        This method creates a new "river_location" variable
        using river station indices from `self.indices` and assigns it to the dataset.
        The indices specify the river station locations in terms of eta_rho and xi_rho grid cell indices.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to which the "river_location" variable will be added.

        Returns
        -------
        xarray.Dataset
            The modified dataset with the "river_location" variable added.
        """

        river_locations = xr.zeros_like(self.grid.ds.h)

        for nriver in ds.nriver:
            river_name = str(ds.river_name.sel(nriver=nriver).values)
            indices = self.indices[river_name]
            fraction = 1.0 / len(indices)

            for eta_index, xi_index in indices:

                river_locations[eta_index, xi_index] = (
                    nriver  # assign unique nriver ID (Fortran-based indexing)
                    + fraction  # Fractional contribution for multiple grid points
                )

        river_locations.attrs["long_name"] = "River ID plus local volume fraction"
        river_locations.attrs["units"] = "none"
        ds["river_location"] = river_locations

        ds = ds.drop_vars(["lat_rho", "lon_rho"])

        return ds

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
        vmax = 3
        vmin = 0
        cmap = plt.colormaps.get_cmap("Blues")
        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

        lon_deg = self.grid.ds.lon_rho
        lat_deg = self.grid.ds.lat_rho

        # check if North or South pole are in domain
        if lat_deg.max().values > 89 or lat_deg.min().values < -89:
            raise NotImplementedError(
                "Plotting is not implemented for the case that the domain contains the North or South pole."
            )

        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)
        field = field.assign_coords({"lon": lon_deg, "lat": lat_deg})

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

        proj = ccrs.PlateCarree()

        if len(self.indices) <= 10:
            color_map = cm.get_cmap("tab10")
        elif len(self.indices) <= 20:
            color_map = cm.get_cmap("tab20")
        else:
            color_map = cm.get_cmap("tab20b")
        # Create a dictionary of colors
        colors = {name: color_map(i) for i, name in enumerate(self.indices.keys())}

        for ax, indices in zip(axs, [self.original_indices, self.indices]):
            added_labels = set()
            for name in indices.keys():
                for tuple in indices[name]:
                    eta_index = tuple[0]
                    xi_index = tuple[1]

                    # transform coordinates to projected space
                    transformed_lon, transformed_lat = trans.transform_point(
                        self.grid.ds.lon_rho[eta_index, xi_index],
                        self.grid.ds.lat_rho[eta_index, xi_index],
                        proj,
                    )

                    if name not in added_labels:
                        added_labels.add(name)
                        label = name
                    else:
                        label = "_None"

                    ax.plot(
                        transformed_lon,
                        transformed_lat,
                        marker="x",
                        markersize=8,
                        markeredgewidth=2,
                        label=label,
                        color=colors[name],
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
    ) -> None:
        """Save the river forcing to netCDF4 file.

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

        def convert_indices_format(indices):
            # Remove the '_convention' key from the dictionary if present
            indices = {
                key: value for key, value in indices.items() if key != "_convention"
            }

            # Convert the string of indices into tuples
            for river, index_list in indices.items():
                # Split the string by ',' and convert to tuples of integers
                indices[river] = [tuple(map(int, idx.split(","))) for idx in index_list]

            return indices

        params["indices"] = convert_indices_format(params["indices"])

        return cls(grid=grid, **params)


def check_river_locations_are_along_coast(mask, indices):
    """Check if the river locations are along the coast.

    This function checks if the river locations specified in the `indices` dictionary are located on coastal grid cells.
    A coastal grid cell is defined as a land grid cell adjacent to an ocean grid cell.

    Parameters
    ----------
    mask : xarray.DataArray
        A mask representing the land and ocean cells in the grid, where 1 represents ocean and 0 represents land.

    indices : dict
        A dictionary where the keys are river names, and the values are dictionaries containing the river's grid locations (`eta_rho` and `xi_rho`).
        Each entry should have keys `"eta_rho"` and `"xi_rho"`, which are lists of grid cell indices representing river mouth locations.

    Raises
    ------
    ValueError
        If any river is not located on the coast.
    """

    faces = (
        mask.shift(eta_rho=1)
        + mask.shift(eta_rho=-1)
        + mask.shift(xi_rho=1)
        + mask.shift(xi_rho=-1)
    )
    coast = (1 - mask) * (faces > 0)

    for key, river_data in indices.items():
        for idx_pair in river_data:
            eta_rho, xi_rho = idx_pair

            # Check if the river location is along the coast
            if not coast[eta_rho, xi_rho]:
                raise ValueError(
                    f"River `{key}` is not located on the coast at grid cell ({eta_rho}, {xi_rho})."
                )
