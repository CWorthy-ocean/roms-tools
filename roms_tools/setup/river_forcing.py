import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from roms_tools import Grid
from roms_tools.constants import MAX_DISTINCT_COLORS
from roms_tools.plot import (
    assign_category_colors,
    get_projection,
    plot_2d_horizontal_field,
    plot_location,
)
from roms_tools.setup.datasets import (
    DaiRiverDataset,
    get_indices_of_nearest_grid_cell_for_rivers,
)
from roms_tools.setup.utils import (
    add_time_info_to_ds,
    add_tracer_metadata_to_ds,
    from_yaml,
    gc_dist,
    get_target_coords,
    get_tracer_defaults,
    get_variable_metadata,
    substitute_nans_by_fillvalue,
    to_dict,
    validate_names,
    write_to_yaml,
)
from roms_tools.utils import save_datasets

INCLUDE_ALL_RIVER_NAMES = "all"
MAX_RIVERS_TO_PLOT = 20  # must be <= MAX_DISTINCT_COLORS


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
        Reference date for the ROMS simulation. Default is January 1, 2000.
    indices : dict[str, list[tuple[int, int]]], optional
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
    """

    grid: Grid
    """Object representing the grid information."""
    start_time: datetime
    """Start time of the desired river forcing data."""
    end_time: datetime
    """End time of the desired river forcing data."""
    source: dict[str, str | Path | list[str | Path]] | None = None
    """Dictionary specifying the source of the river forcing data."""
    convert_to_climatology: str = "if_any_missing"
    """Determines when to compute climatology for river forcing."""
    include_bgc: bool = False
    """Whether to include BGC tracers."""
    model_reference_date: datetime = datetime(2000, 1, 1)
    """Reference date for the ROMS simulation."""

    indices: dict[str, list[tuple[int, int]]] | None = None
    """A dictionary of river indices.

    If not provided during initialization, it will be automatically determined based on
    the grid and the source dataset.
    """
    ds: xr.Dataset = field(init=False, repr=False)
    """An xarray Dataset containing post-processed variables ready for input into
    ROMS."""
    climatology: xr.Dataset = field(init=False, repr=False)
    """Indicates whether the final river forcing is climatological."""

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
        ds = self._handle_overlapping_rivers(ds)
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

                seen_tuples = set()
                # Ensure each element in the list is a tuple of length 2
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
                            f"Value of eta_rho for river `{river_name}` ({eta_rho}) is out of valid range [0, {len(self.grid.ds.eta_rho) - 1}]."
                        )
                    if not (0 <= xi_rho < len(self.grid.ds.xi_rho)):
                        raise ValueError(
                            f"Value of xi_rho for river `{river_name}` ({xi_rho}) is out of valid range [0, {len(self.grid.ds.xi_rho) - 1}]."
                        )

                    # Check for duplicate tuples for a single river
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

        # Find the indices of the closest coastal grid cell to the river mouth
        river_indices = get_indices_of_nearest_grid_cell_for_rivers(dist_coast, data)

        return river_indices

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

        # Tracer metadata
        ds = add_tracer_metadata_to_ds(ds, self.include_bgc)

        # River volume
        river_volume = (
            data.ds[data.var_names["flux"]] * data.ds[data.var_names["ratio"]]
        ).astype(np.float32)
        river_volume = river_volume.rename(
            {data.dim_names["time"]: "river_time", data.dim_names["station"]: "nriver"}
        )
        ds["river_volume"] = river_volume
        ds["river_volume"].attrs["long_name"] = "River volume flux"
        ds["river_volume"].attrs["units"] = "m^3/s"

        # River tracers
        ds["river_tracer"] = xr.zeros_like(
            ds.river_time.astype(np.float32) * ds.ntracers * ds.nriver, dtype=np.float32
        )
        ds["river_tracer"].attrs["long_name"] = "River tracer data"

        defaults = get_tracer_defaults()
        for ntracer in range(ds.ntracers.size):
            tracer_name = ds.tracer_name[ntracer].item()
            ds["river_tracer"].loc[{"ntracers": ntracer}] = defaults[tracer_name]

        # River names
        river_names = data.ds[data.var_names["name"]].rename(
            {data.dim_names["station"]: "nriver"}
        )
        river_names.attrs["long_name"] = "River name"
        ds = ds.assign_coords({"river_name": river_names})

        # River IDs
        nriver = xr.DataArray(np.arange(1, len(ds.nriver) + 1), dims="nriver")
        nriver.attrs["long_name"] = "River ID (1-based Fortran indexing)"
        ds = ds.assign_coords({"nriver": nriver})

        # River time
        ds["river_time"] = data.ds[data.dim_names["time"]].rename(
            {"time": "river_time"}
        )
        ds, time = add_time_info_to_ds(
            ds, self.model_reference_date, self.climatology, time_name="river_time"
        )
        ds = ds.assign_coords({"river_time": time})

        return ds

    def _handle_overlapping_rivers(self, ds: xr.Dataset) -> xr.Dataset:
        """Detect and resolve overlapping river grid cell assignments.

        If multiple rivers are assigned to the same grid cell (i.e., overlapping index pairs),
        this method creates new uniquely named rivers ('overlap_1', 'overlap_2', ...)
        that consolidate the contributions from all original rivers sharing that location.

        For each overlapping grid cell:
        - A new river is created using a volume-weighted sum of the original rivers' volume and tracer.
        - The volume fraction of the original river is reduced proportionally to reflect the removal of the shared grid cell.
        - The new river is appended to the dataset and indexed in `self.indices`.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the existing river forcing fields, including:
            - `river_volume` : volume transport per river [river_time, nriver]
            - `river_tracer` : tracer concentration per river [river_time, ntracers, nriver]
            - `river_name`   : river name labels [nriver]

        Returns
        -------
        xr.Dataset
            A new dataset with overlapping rivers resolved and new entries added.
        """
        overlapping_rivers = self._get_overlapping_rivers()

        if len(overlapping_rivers) > 0:
            logging.info(
                f"Creating {len(overlapping_rivers)} synthetic river(s) to handle overlapping entries."
            )

            # Add new unique river for each overlapping index
            combined_river_volumes = []
            combined_river_tracers = []

            for i, (idx_pair, river_list) in enumerate(overlapping_rivers.items()):
                (
                    combined_river_volume,
                    combined_river_tracer,
                ) = self._create_combined_river(ds, i + 1, idx_pair, river_list)
                combined_river_volumes.append(combined_river_volume)
                combined_river_tracers.append(combined_river_tracer)

            ds_updated = xr.Dataset()
            ds_updated["river_volume"] = xr.concat(
                [ds["river_volume"], *combined_river_volumes], dim="nriver"
            )
            ds_updated["river_tracer"] = xr.concat(
                [ds["river_tracer"], *combined_river_tracers], dim="nriver"
            )
            ds_updated.attrs = ds.attrs
        else:
            ds_updated = ds

        # Reduce volume fraction of original rivers by appropriate amount
        ds_updated = self._reduce_river_volumes(ds_updated, overlapping_rivers)

        return ds_updated

    def _get_overlapping_rivers(self) -> dict[tuple[int, int], list[str]]:
        """Identify grid cells shared by multiple rivers.

        Scans through the river indices and finds all grid cell indices
        (as tuples) that are associated with more than one river.

        Returns
        -------
        overlapping_rivers : dict[tuple[int, int], list[str]]
            A dictionary mapping grid cell indices (eta_rho, xi_rho) to a list
            of river names that overlap at that grid cell.
        """
        if self.indices is None:
            return {}

        index_to_rivers: dict[tuple[int, int], list[str]] = defaultdict(list)

        # Collect all index pairs used by multiple rivers
        for river_name, index_list in self.indices.items():
            for idx_pair in index_list:
                index_to_rivers[idx_pair].append(river_name)

        overlapping_rivers = {
            idx: names for idx, names in index_to_rivers.items() if len(names) > 1
        }

        return overlapping_rivers

    def _create_combined_river(
        self,
        ds: xr.Dataset,
        i: int,
        idx_pair: tuple[int, int],
        river_list: list[str],
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Create a combined river entry from multiple overlapping rivers.

        This method generates a synthetic river by merging contributions from several rivers
        that map to the same grid cell. It performs a weighted sum of river volumes and
        computes volume-weighted averages of tracer concentrations. The new synthetic river is also
        registered in `self.indices`.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing river data, including `river_volume`, `river_tracer`, and `river_name`.
        i : int
            Index used for naming the new synthetic river (e.g., "overlap_1", "overlap_2", ...).
        idx_pair : tuple[int, int]
            Grid cell index (eta_rho, xi_rho) that the combined river will occupy.
        river_list : list[str]
            List of river names contributing to the overlapping grid cell.

        Returns
        -------
        combined_river_volume : xr.DataArray
            The total river volume at the overlapping grid cell, as a new 1-entry DataArray with updated coordinates.
        combined_river_tracer : xr.DataArray
            The volume-weighted tracer concentration at the overlapping grid cell,
            as a new 1-entry DataArray with updated coordinates.
        """
        if self.indices is None:
            self.indices = {}

        new_name = f"overlap_{i}"
        self.indices[new_name] = [idx_pair]

        # Get index of all rivers contributing to this overlapping cell
        contributing_indices = [
            np.where(ds["river_name"].values == name)[0].item() for name in river_list
        ]

        # Get the number of grid points each river originally contributed to
        num_cells_per_river = [len(self.indices[name]) for name in river_list]

        # Weighted sum of river volume contributions at the overlapping location
        weighted_volumes = xr.concat(
            [
                (ds["river_volume"].isel(nriver=i) / n_cells).astype("float64")
                for i, n_cells in zip(contributing_indices, num_cells_per_river)
            ],
            dim="tmp",
        )
        combined_river_volume = weighted_volumes.sum(dim="tmp", skipna=True)

        # Volume-weighted sum of river tracer contributions at the overlapping location
        weighted_tracers = xr.concat(
            [
                (ds["river_tracer"].isel(nriver=i) * weight.fillna(0.0)).astype(
                    "float64"
                )
                for i, weight in zip(contributing_indices, weighted_volumes)
            ],
            dim="tmp",
        )
        combined_river_tracer = (
            weighted_tracers.sum(dim="tmp", skipna=True) / combined_river_volume
        )
        # If combined_river_volume is 0.0, the result will be NaN. Replace with default for clarity.
        # This is mainly for plotting and to avoid confusing users—ROMS will ignore tracers with zero volume anyway.
        defaults = get_tracer_defaults()
        for ntracer in range(combined_river_tracer.sizes["ntracers"]):
            tracer_name = combined_river_tracer.tracer_name[ntracer].item()
            default = defaults[tracer_name]
            combined_river_tracer.loc[{"ntracers": ntracer}] = (
                combined_river_tracer.loc[{"ntracers": ntracer}].fillna(default)
            )

        # Expand, assign coordinates, and name for both volume and tracer
        new_nriver = ds.sizes["nriver"] + i
        combined_river_volume = combined_river_volume.expand_dims(nriver=1)
        combined_river_volume = combined_river_volume.assign_coords(
            nriver=[new_nriver], river_name=new_name
        )
        combined_river_tracer = combined_river_tracer.expand_dims(nriver=1)
        combined_river_tracer = combined_river_tracer.assign_coords(
            nriver=[new_nriver], river_name=new_name
        )

        return combined_river_volume, combined_river_tracer

    def _reduce_river_volumes(
        self, ds: xr.Dataset, overlapping_rivers: dict[tuple[int, int], list[str]]
    ) -> xr.Dataset:
        """Reduce river volumes for rivers contributing to overlapping grid cells.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing `river_name` and `river_volume` variables.
        overlapping_rivers : dict
            Mapping from (eta_rho, xi_rho) grid cell indices to lists of river names
            contributing to that cell.

        Returns
        -------
        ds : xr.Dataset
            Updated dataset with reduced river volumes.
        """
        if self.indices is None:
            raise ValueError(
                "`self.indices` must be set before calling _reduce_river_volumes"
            )

        # Count number of overlaps for each river
        river_overlap_count: dict[str, int] = defaultdict(int)

        for rivers in overlapping_rivers.values():
            for name in rivers:
                river_overlap_count[name] += 1

        for name in ds["river_name"].values:
            n_cells = len(self.indices[name])
            n_overlaps = river_overlap_count.get(name, 0)

            if n_cells == 0:
                continue  # Avoid division by zero

            # Scale river volume based on non-overlapping cells
            scale_factor = (n_cells - n_overlaps) / n_cells
            river_idx = np.where(ds["river_name"].values == name)[0].item()
            nriver_val = ds["nriver"].values[river_idx]
            ds["river_volume"].loc[{"nriver": nriver_val}] *= scale_factor

        return ds

    def _write_indices_into_dataset(self, ds):
        """Adds river location indices to the dataset as the "river_index" and
        "river_fraction" variables.

        This method creates new "river_index" and "river_fraction" variables using river station indices
        from `self.indices` and assigns them to the dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to which the "river_index" and "river_fraction" variables will be added.

        Returns
        -------
        xarray.Dataset
            The modified dataset with the "river_index" and "river_fraction" variables added.
        """
        river_index = xr.zeros_like(self.grid.ds.h, dtype=np.float32)
        river_fraction = xr.zeros_like(self.grid.ds.h, dtype=np.float32)

        for nriver in ds.nriver:
            river_name = str(ds.river_name.sel(nriver=nriver).values)
            indices = self.indices[river_name]
            fraction = 1.0 / len(indices)
            for eta_index, xi_index in indices:
                # Assign unique nriver ID (Fortran-based indexing)
                river_index[eta_index, xi_index] = nriver
                # Fractional contribution for multiple grid points
                river_fraction[eta_index, xi_index] = fraction

        river_index.attrs["long_name"] = "River ID"
        river_index.attrs["units"] = "none"
        ds["river_index"] = river_index

        river_fraction.attrs["long_name"] = "River volume fraction"
        river_fraction.attrs["units"] = "none"
        ds["river_fraction"] = river_fraction

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

    def plot_locations(self, river_names: list[str] | str = INCLUDE_ALL_RIVER_NAMES):
        """Plots the original and updated river locations on a map projection.

        Parameters
        ----------
        river_names : list[str], or str, optional
            A list of release names to plot.
            If a string equal to "all", all rivers will be plotted.
            Defaults to "all".

        """
        if self.indices is None:
            valid_river_names = []
        else:
            valid_river_names = list(self.indices.keys())

        river_names = _validate_river_names(river_names, valid_river_names)
        if len(valid_river_names) > MAX_DISTINCT_COLORS:
            colors = assign_category_colors(river_names)
        else:
            colors = assign_category_colors(valid_river_names)

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

        trans = get_projection(lon_deg, lat_deg)

        fig, axs = plt.subplots(
            1, 2, figsize=(13, 13), subplot_kw={"projection": trans}
        )

        for ax in axs:
            plot_2d_horizontal_field(field, kwargs=kwargs, ax=ax, add_colorbar=False)

        points = {}
        for j, (ax, indices) in enumerate(
            [(ax, ind) for ax, ind in zip(axs, [self.original_indices, self.indices])]
        ):
            for name in river_names:
                if name in indices:
                    for i, (eta_index, xi_index) in enumerate(indices[name]):
                        lon = self.grid.ds.lon_rho[eta_index, xi_index].item()
                        lat = self.grid.ds.lat_rho[eta_index, xi_index].item()
                        key = name if i == 0 else f"_{name}_{i}"
                        points[key] = {
                            "lon": lon,
                            "lat": lat,
                            "color": colors[name],
                        }

            plot_location(
                grid_ds=self.grid.ds,
                points=points,
                ax=ax,
                include_legend=(j == 1),
            )

        axs[0].set_title("Original river locations")
        axs[1].set_title("Updated river locations")

    def plot(
        self,
        var_name: str = "river_volume",
        river_names: list[str] | str = INCLUDE_ALL_RIVER_NAMES,
    ):
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

        river_names : list[str], or str, optional
            A list of release names to plot.
            If a string equal to "all", all rivers will be plotted.
            Defaults to "all".

        """
        if self.indices is None:
            valid_river_names = []
        else:
            valid_river_names = list(self.indices.keys())

        river_names = _validate_river_names(river_names, valid_river_names)
        if len(valid_river_names) > MAX_DISTINCT_COLORS:
            colors = assign_category_colors(river_names)
        else:
            colors = assign_category_colors(valid_river_names)

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

        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        for name in river_names:
            nriver = np.where(self.ds["river_name"].values == name)[0].item()

            ax.plot(
                xticks,
                field.isel(nriver=nriver),
                marker="x",
                markersize=8,
                markeredgewidth=2,
                lw=2,
                label=name,
                color=colors[name],
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
        filepath: str | Path,
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

    def to_yaml(self, filepath: str | Path) -> None:
        """Export the parameters of the class to a YAML file, including the version of
        roms-tools.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file where the parameters will be saved.
        """
        forcing_dict = to_dict(self, exclude=["climatology"])

        indices_data = forcing_dict.get("RiverForcing", {}).get("indices")
        if not indices_data:
            # If no indices, just write the dict as is
            write_to_yaml(forcing_dict, filepath)
            return

        # Convert tuple indices to string format for YAML
        serialized_indices: dict[str, str | list[str]] = {}
        for key, value in indices_data.items():
            serialized_indices[key] = [f"{tup[0]}, {tup[1]}" for tup in value]
        serialized_indices["_convention"] = "eta_rho, xi_rho"

        # Remove keys starting with "overlap_"
        filtered_indices = {
            key: value
            for key, value in serialized_indices.items()
            if not key.startswith("overlap_")
        }

        forcing_dict["RiverForcing"]["indices"] = filtered_indices

        write_to_yaml(forcing_dict, filepath)

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> "RiverForcing":
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
        params = from_yaml(cls, filepath)

        def convert_indices_format(indices):
            indices = {
                key: value for key, value in indices.items() if key != "_convention"
            }

            for river, index_list in indices.items():
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


def _validate_river_names(
    river_names: list[str] | str, valid_river_names: list[str]
) -> list[str]:
    """
    Validate and filter a list of river names.

    Ensures that each river name exists in `valid_river_names` and limits the list
    to `MAX_RIVERS_TO_PLOT` entries with a warning if truncated.

    Parameters
    ----------
    river_names : list of str or INCLUDE_ALL_RIVER_NAMES
        Names of rivers to plot, or sentinel to include all.
    valid_river_names : list of str
        List of valid river names.

    Returns
    -------
    list of str
        Validated and truncated list of river names.

    Raises
    ------
    ValueError
        If any names are invalid.
    """
    return validate_names(
        river_names,
        valid_river_names,
        INCLUDE_ALL_RIVER_NAMES,
        MAX_RIVERS_TO_PLOT,
        label="river",
    )
