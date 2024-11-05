import xarray as xr
import pandas as pd
import yaml
import importlib.metadata
from dataclasses import dataclass, field, asdict
from roms_tools.setup.grid import Grid
from datetime import datetime
import numpy as np
from typing import Dict, Union, List
from roms_tools.setup.regrid import LateralRegrid
from roms_tools.setup.datasets import (
    ERA5Dataset,
    ERA5Correction,
    CESMBGCSurfaceForcingDataset,
)
from roms_tools.setup.utils import (
    nan_check,
    substitute_nans_by_fillvalue,
    interpolate_from_climatology,
    get_variable_metadata,
    group_dataset,
    save_datasets,
    get_target_coords,
    rotate_velocities,
)
from roms_tools.setup.plot import _plot
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class SurfaceForcing:
    """Represents surface forcing input data for ROMS.

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

    type : str
        Specifies the type of forcing data. Options are:

          - "physics": for physical atmospheric forcing.
          - "bgc": for biogeochemical forcing.

    correct_radiation : bool
        Whether to correct shortwave radiation. Default is False.
    use_coarse_grid: bool
        Whether to interpolate to coarsened grid. Default is False.
    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.

    Examples
    --------
    >>> surface_forcing = SurfaceForcing(
    ...     grid=grid,
    ...     start_time=datetime(2000, 1, 1),
    ...     end_time=datetime(2000, 1, 2),
    ...     source={"name": "ERA5", "path": "era5_data.nc"},
    ...     type="physics",
    ...     correct_radiation=True,
    ... )
    """

    grid: Grid
    start_time: datetime
    end_time: datetime
    source: Dict[str, Union[str, Path, List[Union[str, Path]]]]
    type: str = "physics"
    correct_radiation: bool = False
    use_coarse_grid: bool = False
    model_reference_date: datetime = datetime(2000, 1, 1)
    use_dask: bool = False

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        self._input_checks()
        target_coords = get_target_coords(self.grid, self.use_coarse_grid)
        object.__setattr__(self, "target_coords", target_coords)

        data = self._get_data()
        data.choose_subdomain(
            target_coords,
            buffer_points=20,  # lateral fill needs some buffer from data margin
        )

        data.apply_lateral_fill()

        variable_info = self._set_variable_info(data)
        var_names = variable_info.keys()

        processed_fields = {}
        # lateral regridding
        lateral_regrid = LateralRegrid(target_coords, data.dim_names)
        for var_name in var_names:
            if var_name in data.var_names.keys():
                processed_fields[var_name] = lateral_regrid.apply(
                    data.ds[data.var_names[var_name]]
                )

        # rotation of velocities and interpolation to u/v points
        if "uwnd" in variable_info and "vwnd" in variable_info:
            processed_fields["uwnd"], processed_fields["vwnd"] = rotate_velocities(
                processed_fields["uwnd"],
                processed_fields["vwnd"],
                target_coords["angle"],
                interpolate=False,
            )

        # correct radiation
        if self.type == "physics" and self.correct_radiation:
            processed_fields = self._apply_correction(processed_fields, data)

        d_meta = get_variable_metadata()

        ds = self._write_into_dataset(processed_fields, data, d_meta)

        if self.use_coarse_grid:
            mask = self.grid.ds["mask_coarse"].rename(
                {"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"}
            )
        else:
            mask = self.grid.ds["mask_rho"]

        self._validate(ds, mask)

        # substitute NaNs over land by a fill value to avoid blow-up of ROMS
        for var_name in ds.data_vars:
            ds[var_name] = substitute_nans_by_fillvalue(ds[var_name])

        object.__setattr__(self, "ds", ds)

    def _input_checks(self):
        # Validate the 'type' parameter
        if self.type not in ["physics", "bgc"]:
            raise ValueError("`type` must be either 'physics' or 'bgc'.")

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
            "use_dask": self.use_dask,
        }

        if self.type == "physics":
            if self.source["name"] == "ERA5":
                data = ERA5Dataset(**data_dict)
            else:
                raise ValueError(
                    'Only "ERA5" is a valid option for source["name"] when type is "physics".'
                )

        elif self.type == "bgc":
            if self.source["name"] == "CESM_REGRIDDED":

                data = CESMBGCSurfaceForcingDataset(**data_dict)
            else:
                raise ValueError(
                    'Only "CESM_REGRIDDED" is a valid option for source["name"] when type is "bgc".'
                )

        return data

    def _get_correction_data(self):

        if self.source["name"] == "ERA5":
            correction_data = ERA5Correction(use_dask=self.use_dask)
        else:
            raise ValueError(
                "The 'correct_radiation' feature is currently only supported for 'ERA5' as the source. "
                "Please ensure your 'source' is set to 'ERA5' or implement additional handling for other sources."
            )

        return correction_data

    def _set_variable_info(self, data):
        """Sets up a dictionary with metadata for variables based on the type of data
        (physics or BGC).

        The dictionary contains the following information:
        - `location`: Where the variable resides in the grid (e.g., rho, u, or v points).
        - `is_vector`: Whether the variable is part of a vector (True for velocity components like 'u' and 'v').
        - `vector_pair`: For vector variables, this indicates the associated variable that forms the vector (e.g., 'u' and 'v').
        - `is_3d`: Indicates whether the variable is 3D (True for variables like 'temp' and 'salt') or 2D (False for 'zeta').

        Returns
        -------
        dict
            A dictionary where the keys are variable names and the values are dictionaries of metadata
            about each variable, including 'location', 'is_vector', 'vector_pair', and 'is_3d'.
        """
        default_info = {
            "location": "rho",
            "is_vector": False,
            "vector_pair": None,
            "is_3d": False,
        }

        # Define a dictionary for variable names and their associated information
        if self.type == "physics":
            variable_info = {
                "swrad": default_info,
                "lwrad": default_info,
                "Tair": default_info,
                "qair": default_info,
                "rain": default_info,
                "uwnd": {
                    "location": "u",
                    "is_vector": True,
                    "vector_pair": "vwnd",
                    "is_3d": False,
                },
                "vwnd": {
                    "location": "v",
                    "is_vector": True,
                    "vector_pair": "uwnd",
                    "is_3d": False,
                },
            }
        elif self.type == "bgc":
            variable_info = {}
            for var in data.var_names.keys():
                variable_info[var] = default_info

        return variable_info

    def _apply_correction(self, processed_fields, data):

        correction_data = self._get_correction_data()
        # choose same subdomain as forcing data so that we can use same mask
        coords_correction = {
            correction_data.dim_names["latitude"]: data.ds[data.dim_names["latitude"]],
            correction_data.dim_names["longitude"]: data.ds[
                data.dim_names["longitude"]
            ],
        }
        correction_data.choose_subdomain(
            coords_correction, straddle=self.target_coords["straddle"]
        )
        correction_data.ds["mask"] = data.ds["mask"]  # use mask from ERA5 data
        correction_data.apply_lateral_fill()
        # regrid
        lateral_regrid = LateralRegrid(self.target_coords, correction_data.dim_names)
        corr_factor = lateral_regrid.apply(
            correction_data.ds[correction_data.var_names["swr_corr"]]
        )

        # temporal interpolation
        corr_factor = interpolate_from_climatology(
            corr_factor,
            correction_data.dim_names["time"],
            time=processed_fields["swrad"].time,
        )

        processed_fields["swrad"] = processed_fields["swrad"] * corr_factor

        return processed_fields

    def _write_into_dataset(self, processed_fields, data, d_meta):

        # save in new dataset
        ds = xr.Dataset()

        for var_name in processed_fields.keys():
            ds[var_name] = processed_fields[var_name].astype(np.float32)
            ds[var_name].attrs["long_name"] = d_meta[var_name]["long_name"]
            ds[var_name].attrs["units"] = d_meta[var_name]["units"]

        if self.use_coarse_grid:
            ds = ds.rename({"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"})

        ds = self._add_global_metadata(ds)

        # Convert the time coordinate to the format expected by ROMS
        if data.climatology:
            ds.attrs["climatology"] = str(True)
            # Preserve absolute time coordinate for readability
            ds = ds.assign_coords(
                {"abs_time": np.datetime64(self.model_reference_date) + ds["time"]}
            )
            # Convert to pandas TimedeltaIndex
            timedelta_index = pd.to_timedelta(ds["time"].values)

            # Determine the start of the year for the base_datetime
            start_of_year = datetime(self.model_reference_date.year, 1, 1)

            # Calculate the offset from midnight of the new year
            offset = self.model_reference_date - start_of_year

            # Convert the timedelta to nanoseconds first, then to days
            sfc_time = xr.DataArray(
                (timedelta_index - offset).view("int64") / 3600 / 24 * 1e-9,
                dims="time",
            )
        else:
            # Preserve absolute time coordinate for readability
            ds = ds.assign_coords({"abs_time": ds["time"]})

            sfc_time = (
                (ds["time"] - np.datetime64(self.model_reference_date)).astype(
                    "float64"
                )
                / 3600
                / 24
                * 1e-9
            )

        if self.type == "physics":
            time_coords = ["time"]
        elif self.type == "bgc":
            time_coords = [
                "pco2_time",
                "iron_time",
                "dust_time",
                "nox_time",
                "nhy_time",
            ]
        for time_coord in time_coords:
            ds = ds.assign_coords({time_coord: sfc_time})
            ds[time_coord].attrs[
                "long_name"
            ] = f"days since {str(self.model_reference_date)}"
            ds[time_coord].encoding["units"] = "days"
            ds[time_coord].attrs["units"] = "days"
            if data.climatology:
                ds[time_coord].attrs["cycle_length"] = 365.25
        ds.encoding["unlimited_dims"] = "time"

        if self.type == "bgc":
            ds = ds.drop_vars(["time"])

        variables_to_drop = ["lat_rho", "lon_rho", "lat_coarse", "lon_coarse"]
        existing_vars = [var_name for var_name in variables_to_drop if var_name in ds]
        ds = ds.drop_vars(existing_vars)

        return ds

    def _validate(self, ds, mask):
        """Validates the dataset by checking for NaN values at wet points, which would
        indicate missing raw data coverage over the target domain.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to validate.
        mask : xarray.DataArray
            Land mask (1=ocean, 0=land) to determine wet points in the domain.

        Raises
        ------
        ValueError
            If NaN values are found in any of the specified variables at wet points,
            indicating incomplete data coverage.

        Notes
        -----
        This check is applied to the first time step (`time=0`) of each variable in the provided dataset.
        """

        for var_name in ds.data_vars:
            nan_check(ds[var_name].isel(time=0), mask)

    def _add_global_metadata(self, ds=None):

        if ds is None:
            ds = xr.Dataset()
        ds.attrs["title"] = "ROMS surface forcing file created by ROMS-Tools"
        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"
        ds.attrs["roms_tools_version"] = roms_tools_version
        ds.attrs["start_time"] = str(self.start_time)
        ds.attrs["end_time"] = str(self.end_time)
        ds.attrs["source"] = self.source["name"]
        ds.attrs["correct_radiation"] = str(self.correct_radiation)
        ds.attrs["use_coarse_grid"] = str(self.use_coarse_grid)
        ds.attrs["model_reference_date"] = str(self.model_reference_date)

        ds.attrs["type"] = self.type
        ds.attrs["source"] = self.source["name"]

        return ds

    def plot(self, var_name, time=0) -> None:
        """Plot the specified surface forcing field for a given time slice.

        Parameters
        ----------
        var_name : str
            The name of the surface forcing field to plot. Options include:

            - "uwnd": 10 meter wind in x-direction.
            - "vwnd": 10 meter wind in y-direction.
            - "swrad": Downward short-wave (solar) radiation.
            - "lwrad": Downward long-wave (thermal) radiation.
            - "Tair": Air temperature at 2m.
            - "qair": Absolute humidity at 2m.
            - "rain": Total precipitation.
            - "pco2_air": Atmospheric pCO2.
            - "pco2_air_alt": Atmospheric pCO2, alternative CO2.
            - "iron": Iron decomposition.
            - "dust": Dust decomposition.
            - "nox": NOx decomposition.
            - "nhy": NHy decomposition.

        time : int, optional
            The time index to plot. Default is 0, which corresponds to the first
            time slice.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        Raises
        ------
        ValueError
            If the specified var_name is not found in dataset.


        Examples
        --------
        >>> atm_forcing.plot("uwnd", time=0)
        """

        if var_name not in self.ds:
            raise ValueError(f"Variable '{var_name}' is not found in dataset.")

        field = self.ds[var_name].isel(time=time).load()
        title = field.long_name

        # assign lat / lon
        if self.use_coarse_grid:
            field = field.rename({"eta_rho": "eta_coarse", "xi_rho": "xi_coarse"})
            field = field.where(self.grid.ds.mask_coarse)
        else:
            field = field.where(self.grid.ds.mask_rho)

        field = field.assign_coords(
            {"lon": self.target_coords["lon"], "lat": self.target_coords["lat"]}
        )

        # choose colorbar
        if var_name in ["uwnd", "vwnd"]:
            vmax = max(field.max().values, -field.min().values)
            vmin = -vmax
            cmap = plt.colormaps.get_cmap("RdBu_r")
        else:
            vmax = field.max().values
            vmin = field.min().values
            if var_name in ["swrad", "lwrad", "Tair", "qair"]:
                cmap = plt.colormaps.get_cmap("YlOrRd")
            else:
                cmap = plt.colormaps.get_cmap("YlGnBu")
        cmap.set_bad(color="gray")

        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

        _plot(
            self.grid.ds,
            field=field,
            straddle=self.grid.straddle,
            title=title,
            kwargs=kwargs,
            c="g",
        )

    def save(
        self,
        filepath: Union[str, Path],
        np_eta: int = None,
        np_xi: int = None,
        group: bool = False,
    ) -> None:
        """Save the surface forcing fields to one or more netCDF4 files.

        This method saves the dataset either as a single file or as multiple files depending on the partitioning and grouping options.
        The dataset can be saved in two modes:

        1. **Single File Mode (default)**:
            - If both `np_eta` and `np_xi` are `None`, the entire dataset is saved as a single netCDF4 file.
            - The file is named based on the `filepath`, with `.nc` automatically appended.

        2. **Partitioned Mode**:
            - If either `np_eta` or `np_xi` is specified, the dataset is partitioned into spatial tiles along the `eta` and `xi` axes.
            - Each tile is saved as a separate netCDF4 file, and filenames are modified with an index (e.g., `"filepath_YYYYMM.0.nc"`, `"filepath_YYYYMM.1.nc"`).

        Additionally, if `group` is set to `True`, the dataset is first grouped into temporal subsets, resulting in multiple grouped files before partitioning and saving.

        Parameters
        ----------
        filepath : Union[str, Path]
            The base path and filename for the output files. The format of the filenames depends on whether partitioning is used
            and the temporal range of the data. For partitioned datasets, files will be named with an additional index, e.g.,
            `"filepath_YYYYMM.0.nc"`, `"filepath_YYYYMM.1.nc"`, etc.
        np_eta : int, optional
            The number of partitions along the `eta` direction. If `None`, no spatial partitioning is performed.
        np_xi : int, optional
            The number of partitions along the `xi` direction. If `None`, no spatial partitioning is performed.
        group: bool, optional
            If `True`, groups the dataset into multiple files based on temporal data frequency. Defaults to `False`.

        Returns
        -------
        List[Path]
            A list of Path objects for the filenames that were saved.
        """

        # Ensure filepath is a Path object
        filepath = Path(filepath)

        # Remove ".nc" suffix if present
        if filepath.suffix == ".nc":
            filepath = filepath.with_suffix("")

        if group:
            dataset_list, output_filenames = group_dataset(
                self.ds.load(), str(filepath)
            )
        else:
            dataset_list = [self.ds.load()]
            output_filenames = [str(filepath)]

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
        filepath = Path(filepath)

        # Serialize Grid data
        grid_data = asdict(self.grid)
        grid_data.pop("ds", None)  # Exclude non-serializable fields
        grid_data.pop("straddle", None)

        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"

        # Create header
        header = f"---\nroms_tools_version: {roms_tools_version}\n---\n"

        # Create YAML data for Grid and optional attributes
        grid_yaml_data = {"Grid": grid_data}

        # Combine all sections
        surface_forcing_data = {
            "SurfaceForcing": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "source": self.source,
                "type": self.type,
                "correct_radiation": self.correct_radiation,
                "use_coarse_grid": self.use_coarse_grid,
                "model_reference_date": self.model_reference_date.isoformat(),
            }
        }

        # Merge YAML data while excluding empty sections
        yaml_data = {
            **grid_yaml_data,
            **surface_forcing_data,
        }

        with filepath.open("w") as file:
            # Write header
            file.write(header)
            # Write YAML data
            yaml.dump(yaml_data, file, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(
        cls, filepath: Union[str, Path], use_dask: bool = False
    ) -> "SurfaceForcing":
        """Create an instance of the SurfaceForcing class from a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file from which the parameters will be read.
        use_dask: bool, optional
            Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.

        Returns
        -------
        SurfaceForcing
            An instance of the SurfaceForcing class.
        """
        filepath = Path(filepath)
        # Read the entire file content
        with filepath.open("r") as file:
            file_content = file.read()

        # Split the content into YAML documents
        documents = list(yaml.safe_load_all(file_content))

        surface_forcing_data = None

        # Process the YAML documents
        for doc in documents:
            if doc is None:
                continue
            if "SurfaceForcing" in doc:
                surface_forcing_data = doc["SurfaceForcing"]

        if surface_forcing_data is None:
            raise ValueError("No SurfaceForcing configuration found in the YAML file.")

        # Convert from string to datetime
        for date_string in ["model_reference_date", "start_time", "end_time"]:
            surface_forcing_data[date_string] = datetime.fromisoformat(
                surface_forcing_data[date_string]
            )

        # Create Grid instance from the YAML file
        grid = Grid.from_yaml(filepath)

        # Create and return an instance of SurfaceForcing
        return cls(grid=grid, **surface_forcing_data, use_dask=use_dask)
