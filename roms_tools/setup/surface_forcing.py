import xarray as xr
import importlib.metadata
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, Union, List, Optional
from roms_tools import Grid
from roms_tools.utils import save_datasets, transpose_dimensions
from roms_tools.regrid import LateralRegrid
from roms_tools.plot import _plot
from roms_tools.setup.datasets import (
    ERA5Dataset,
    ERA5Correction,
    CESMBGCSurfaceForcingDataset,
    UnifiedBGCSurfaceDataset,
)
from roms_tools.setup.utils import (
    get_target_coords,
    nan_check,
    substitute_nans_by_fillvalue,
    interpolate_from_climatology,
    get_variable_metadata,
    group_dataset,
    rotate_velocities,
    compute_missing_surface_bgc_variables,
    convert_to_roms_time,
    _to_yaml,
    _from_yaml,
)


@dataclass(kw_only=True)
class SurfaceForcing:
    """Represents surface forcing input data for ROMS.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
    start_time : datetime, optional
        The start time of the desired surface forcing data. This time is used to filter the dataset
        to include only records on or after this time, with a single record at or before this time.
        If no time filtering is desired, set it to None. Default is None.
    end_time : datetime, optional
        The end time of the desired surface forcing data. This time is used to filter the dataset
        to include only records on or before this time, with a single record at or after this time.
        If no time filtering is desired, set it to None. Default is None.
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
        Whether to correct shortwave radiation. Default is True.
    coarse_grid_mode : str, optional
        Specifies whether to interpolate onto grid coarsened by a factor of two. Options are:

          - "auto" (default): Automatically decide based on the comparison of source and target spatial resolutions.
          - "always": Always interpolate onto the coarse grid.
          - "never": Never use the coarse grid; interpolate onto the fine grid instead.

    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.
    bypass_validation: bool, optional
        Indicates whether to skip validation checks in the processed data. When set to True,
        the validation process that ensures no NaN values exist at wet points
        in the processed dataset is bypassed. Defaults to False.

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
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    source: Dict[str, Union[str, Path, List[Union[str, Path]]]]
    type: str = "physics"
    correct_radiation: bool = True
    coarse_grid_mode: str = "auto"
    model_reference_date: datetime = datetime(2000, 1, 1)
    use_dask: bool = False
    bypass_validation: bool = False

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        self._input_checks()
        data = self._get_data()

        if self.coarse_grid_mode == "always":
            use_coarse_grid = True
        elif self.coarse_grid_mode == "never":
            use_coarse_grid = False
        elif self.coarse_grid_mode == "auto":
            use_coarse_grid = self._determine_coarse_grid_usage(data)
        if use_coarse_grid:
            logging.info("Data will be interpolated onto grid coarsened by factor 2.")
        else:
            logging.info("Data will be interpolated onto fine grid.")
        self.use_coarse_grid = use_coarse_grid

        target_coords = get_target_coords(self.grid, self.use_coarse_grid)
        self.target_coords = target_coords

        data.choose_subdomain(
            target_coords,
            buffer_points=20,  # lateral fill needs some buffer from data margin
        )
        # Enforce double precision to ensure reproducibility
        data.convert_to_float64()

        data.apply_lateral_fill()

        self._set_variable_info(data)
        var_names = self.variable_info.keys()

        processed_fields = {}
        # lateral regridding
        lateral_regrid = LateralRegrid(target_coords, data.dim_names)
        for var_name in var_names:
            if var_name in data.var_names.keys():
                processed_fields[var_name] = lateral_regrid.apply(
                    data.ds[data.var_names[var_name]]
                )

        # rotation of velocities
        if "uwnd" in self.variable_info and "vwnd" in self.variable_info:
            processed_fields["uwnd"], processed_fields["vwnd"] = rotate_velocities(
                processed_fields["uwnd"],
                processed_fields["vwnd"],
                target_coords["angle"],
                interpolate=False,
            )

        # correct radiation
        if self.type == "physics" and self.correct_radiation:
            processed_fields = self._apply_correction(processed_fields, data)

        if self.type == "bgc":
            processed_fields = compute_missing_surface_bgc_variables(processed_fields)

        # Reorder dimensions
        for var_name in processed_fields:
            processed_fields[var_name] = transpose_dimensions(
                processed_fields[var_name]
            )

        d_meta = get_variable_metadata()

        ds = self._write_into_dataset(processed_fields, data, d_meta)

        if not self.bypass_validation:
            self._validate(ds)

        # substitute NaNs over land by a fill value to avoid blow-up of ROMS
        for var_name in ds.data_vars:
            ds[var_name] = substitute_nans_by_fillvalue(ds[var_name])

        self.ds = ds

    def _input_checks(self):
        # Check that start_time and end_time are both None or none of them is
        if (self.start_time is None) != (self.end_time is None):
            raise ValueError(
                "Both `start_time` and `end_time` must be provided together as datetime objects or both should be None."
            )

        # Trigger a warning if both are None
        if self.start_time is None and self.end_time is None:
            logging.warning(
                "Both `start_time` and `end_time` are None. No time filtering will be applied to the source data."
            )

        # Validate the 'type' parameter
        if self.type not in ["physics", "bgc"]:
            raise ValueError("`type` must be either 'physics' or 'bgc'.")

        # Ensure 'source' dictionary contains required keys
        if "name" not in self.source:
            raise ValueError("`source` must include a 'name'.")
        if "path" not in self.source:
            raise ValueError("`source` must include a 'path'.")

        # Set 'climatology' to False if not provided in 'source'
        self.source = {
            **self.source,
            "climatology": self.source.get("climatology", False),
        }

        # Validate 'coarse_grid_mode'
        valid_modes = ["auto", "always", "never"]
        if self.coarse_grid_mode not in valid_modes:
            raise ValueError(
                f"`coarse_grid_mode` must be one of {valid_modes}, but got '{self.coarse_grid_mode}'."
            )

    def _determine_coarse_grid_usage(self, data):
        """Determine if coarse grid interpolation should be used based on the resolution
        of the dataset and the target grid.

        Parameters
        ----------
        data : object
            The dataset object containing the data to be analyzed for grid spacing.

        Returns
        -------
        use_coarse_grid : bool
            Whether to use the coarse grid or not.
        """
        # Get the target coordinates and select the subdomain of the data
        target_coords = get_target_coords(self.grid, use_coarse_grid=False)
        data_coords = data.choose_subdomain(
            target_coords, buffer_points=1, return_coords_only=True
        )

        # Compute minimal grid spacing in the data subdomain
        min_grid_spacing_data = data.compute_minimal_grid_spacing(data_coords)

        # Compute the maximum grid spacing in the ROMS grid
        max_grid_spacing = max((1 / self.grid.ds.pm).max(), (1 / self.grid.ds.pn).max())

        # Determine whether to use coarse grid based on grid spacing comparison
        if 2 * max_grid_spacing < min_grid_spacing_data:
            use_coarse_grid = True
        else:
            use_coarse_grid = False

        return use_coarse_grid

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
            elif self.source["name"] == "UNIFIED":

                data = UnifiedBGCSurfaceDataset(**data_dict)
            else:
                raise ValueError(
                    'Only "CESM_REGRIDDED" and "UNIFIED" are valid options for source["name"] when type is "bgc".'
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
        None
            This method updates the instance attribute `variable_info` with the metadata dictionary for the variables.
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
                "swrad": {**default_info, "validate": True},
                "lwrad": {**default_info, "validate": False},
                "Tair": {**default_info, "validate": False},
                "qair": {**default_info, "validate": True},
                "rain": {**default_info, "validate": False},
                "uwnd": {
                    "location": "rho",
                    "is_vector": True,
                    "vector_pair": "vwnd",
                    "is_3d": False,
                    "validate": True,
                },
                "vwnd": {
                    "location": "rho",
                    "is_vector": True,
                    "vector_pair": "uwnd",
                    "is_3d": False,
                    "validate": True,
                },
            }
        elif self.type == "bgc":
            variable_info = {}
            for var_name in list(data.var_names.keys()) + list(
                data.opt_var_names.keys()
            ):
                variable_info[var_name] = default_info
                if var_name == "pco2_air":
                    variable_info[var_name] = {**default_info, "validate": True}
                else:
                    variable_info[var_name] = {**default_info, "validate": False}

        self.variable_info = variable_info

    def _apply_correction(self, processed_fields, data):

        correction_data = self._get_correction_data()
        # Match subdomain to forcing data to reuse the mask
        coords_correction = {
            "lat": data.ds[data.dim_names["latitude"]],
            "lon": data.ds[data.dim_names["longitude"]],
        }
        correction_data.choose_subdomain(
            coords_correction, straddle=self.target_coords["straddle"]
        )
        correction_data.ds["mask"] = data.ds["mask"]  # use mask from ERA5 data
        correction_data.ds["time"] = correction_data.ds["time"].dt.days

        correction_data.apply_lateral_fill()

        # Temporal interpolation: Perform before spatial regridding for better performance
        if self.use_dask:
            # Perform temporal interpolation for each time slice to enforce chunking in time.
            # This reduces memory usage by processing one time step at a time.
            # The interpolated slices are then concatenated along the "time" dimension.
            corr_factor = xr.concat(
                [
                    interpolate_from_climatology(
                        correction_data.ds[correction_data.var_names["swr_corr"]],
                        correction_data.dim_names["time"],
                        time=time,
                    )
                    for time in processed_fields["swrad"].time
                ],
                dim="time",
            )
        else:
            # Interpolate across all time steps at once
            corr_factor = interpolate_from_climatology(
                correction_data.ds[correction_data.var_names["swr_corr"]],
                correction_data.dim_names["time"],
                time=processed_fields["swrad"].time,
            )

        # Spatial regridding
        lateral_regrid = LateralRegrid(self.target_coords, correction_data.dim_names)
        corr_factor = lateral_regrid.apply(corr_factor)

        processed_fields["swrad"] = processed_fields["swrad"] * corr_factor

        return processed_fields

    def _write_into_dataset(self, processed_fields, data, d_meta):

        # save in new dataset
        ds = xr.Dataset()

        for var_name in list(processed_fields.keys()):
            ds[var_name] = processed_fields[var_name].astype(np.float32)
            del processed_fields[var_name]
            ds[var_name].attrs["long_name"] = d_meta[var_name]["long_name"]
            ds[var_name].attrs["units"] = d_meta[var_name]["units"]

        ds = self._add_global_metadata(ds)

        # Convert the time coordinate to the format expected by ROMS
        ds, sfc_time = convert_to_roms_time(
            ds, self.model_reference_date, data.climatology
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

        if self.type == "bgc":
            ds = ds.drop_vars(["time"])

        variables_to_drop = ["lat_rho", "lon_rho", "lat_coarse", "lon_coarse"]
        existing_vars = [var_name for var_name in variables_to_drop if var_name in ds]
        ds = ds.drop_vars(existing_vars)

        return ds

    def _validate(self, ds):
        """Validates the dataset by checking for NaN values at wet points, which would
        indicate missing raw data coverage over the target domain.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to validate.

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
            if self.variable_info[var_name]["validate"]:
                # all variables are at rho-points
                mask = self.target_coords["mask"]
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

        field = self.ds[var_name].isel(time=time)

        if self.use_dask:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                field = field.load()

        field = field.where(self.target_coords["mask"])

        lon_deg = self.target_coords["lon"]
        lat_deg = self.target_coords["lat"]
        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)
        field = field.assign_coords({"lon": lon_deg, "lat": lat_deg})

        title = field.long_name

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
            field=field,
            title=title,
            c="g",
            kwargs=kwargs,
        )

    def save(
        self,
        filepath: Union[str, Path],
        group: bool = True,
    ) -> None:
        """Save the surface forcing fields to one or more netCDF4 files.

        This method saves the dataset to disk as either a single netCDF4 file or multiple files, depending on the `group` parameter.
        If `group` is `True`, the dataset is divided into subsets (e.g., monthly or yearly) based on the temporal frequency
        of the data, and each subset is saved to a separate file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The base path and filename for the output file(s). If `group` is `True`, the filenames will include additional
            time-based information (e.g., year or month) to distinguish the subsets.
        group : bool, optional
            Whether to divide the dataset into multiple files based on temporal frequency. Defaults to `True`.

        Returns
        -------
        List[Path]
            A list of `Path` objects representing the filenames of the saved file(s).
        """

        # Ensure filepath is a Path object
        filepath = Path(filepath)

        # Remove ".nc" suffix if present
        if filepath.suffix == ".nc":
            filepath = filepath.with_suffix("")

        if group:
            dataset_list, output_filenames = group_dataset(self.ds, str(filepath))
        else:
            dataset_list = [self.ds]
            output_filenames = [str(filepath)]

        saved_filenames = save_datasets(
            dataset_list, output_filenames, use_dask=self.use_dask
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
    def from_yaml(
        cls,
        filepath: Union[str, Path],
        use_dask: bool = False,
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

        grid = Grid.from_yaml(filepath)
        params = _from_yaml(cls, filepath)

        return cls(grid=grid, **params, use_dask=use_dask)
