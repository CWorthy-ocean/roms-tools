import xarray as xr
import pandas as pd
import yaml
import importlib.metadata
from dataclasses import dataclass, field, asdict
from roms_tools.setup.grid import Grid
from datetime import datetime
import numpy as np
from typing import Dict, Union, List
from roms_tools.setup.fill import LateralFill
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
            latitude_range=[
                target_coords["lat"].min().values,
                target_coords["lat"].max().values,
            ],
            longitude_range=[
                target_coords["lon"].min().values,
                target_coords["lon"].max().values,
            ],
            margin=2,
            straddle=target_coords["straddle"],
        )

        data_vars = {}

        # regrid
        lateral_fill = LateralFill(
            data.ds["mask"],
            [data.dim_names["latitude"], data.dim_names["longitude"]],
        )
        lateral_regrid = LateralRegrid(data, target_coords["lon"], target_coords["lat"])

        if self.type == "physics":
            varnames = ["uwnd", "vwnd", "swrad", "lwrad", "Tair", "qair", "rain"]
        elif self.type == "bgc":
            varnames = data.var_names.keys()

        for var in varnames:
            # Propagate ocean values into land via lateral fill
            filled = lateral_fill.apply(data.ds[data.var_names[var]])
            # Lateral regridding
            data_vars[var] = lateral_regrid.apply(filled)

        if self.type == "physics":
            # rotate velocities
            data_vars["uwnd"], data_vars["vwnd"] = rotate_velocities(
                data_vars["uwnd"],
                data_vars["vwnd"],
                target_coords["angle"],
                interpolate=False,
            )

            # correct radiation if necessary
            if self.correct_radiation:
                data_vars = self._apply_correction(data_vars, data)

        object.__setattr__(data, "data_vars", data_vars)

        d_meta = get_variable_metadata()

        ds = self._write_into_dataset(data, d_meta)

        if self.use_coarse_grid:
            mask = self.grid.ds["mask_coarse"].rename(
                {"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"}
            )
        else:
            mask = self.grid.ds["mask_rho"]

        # NaN values at wet points indicate that the raw data did not cover the domain, and the following will raise a ValueError
        for var in ds.data_vars:
            nan_check(ds[var].isel(time=0), mask)

        # substitute NaNs over land by a fill value to avoid blow-up of ROMS
        for var in ds.data_vars:
            ds[var] = substitute_nans_by_fillvalue(ds[var])

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

    def _apply_correction(self, data_vars, data):

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
        # apply mask from ERA5 data
        if "mask" in data.var_names.keys():
            mask = data.ds["mask"]
            for var in correction_data.ds.data_vars:
                correction_data.ds[var] = xr.where(
                    mask == 1, correction_data.ds[var], np.nan
                )
            correction_data.ds["mask"] = mask

        # regrid
        lateral_fill = LateralFill(
            correction_data.ds["mask"],
            [
                correction_data.dim_names["latitude"],
                correction_data.dim_names["longitude"],
            ],
        )
        lateral_regrid = LateralRegrid(
            correction_data, self.target_coords["lon"], self.target_coords["lat"]
        )

        filled = lateral_fill.apply(
            correction_data.ds[correction_data.var_names["swr_corr"]]
        )
        corr_factor = lateral_regrid.apply(filled)

        # temporal interpolation
        corr_factor = interpolate_from_climatology(
            corr_factor,
            correction_data.dim_names["time"],
            time=data_vars["swrad"].time,
        )

        data_vars["swrad"] = data_vars["swrad"] * corr_factor

        return data_vars

    def _write_into_dataset(self, data, d_meta):

        # save in new dataset
        ds = xr.Dataset()

        for var in data.data_vars.keys():
            ds[var] = data.data_vars[var].astype(np.float32)
            ds[var].attrs["long_name"] = d_meta[var]["long_name"]
            ds[var].attrs["units"] = d_meta[var]["units"]

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
        existing_vars = [var for var in variables_to_drop if var in ds]
        ds = ds.drop_vars(existing_vars)

        return ds

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

    def plot(self, varname, time=0) -> None:
        """Plot the specified surface forcing field for a given time slice.

        Parameters
        ----------
        varname : str
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
            If the specified varname is not found in dataset.


        Examples
        --------
        >>> atm_forcing.plot("uwnd", time=0)
        """

        if varname not in self.ds:
            raise ValueError(f"Variable '{varname}' is not found in dataset.")

        field = self.ds[varname].isel(time=time).load()
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
        if varname in ["uwnd", "vwnd"]:
            vmax = max(field.max().values, -field.min().values)
            vmin = -vmax
            cmap = plt.colormaps.get_cmap("RdBu_r")
        else:
            vmax = field.max().values
            vmin = field.min().values
            if varname in ["swrad", "lwrad", "Tair", "qair"]:
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
        self, filepath: Union[str, Path], np_eta: int = None, np_xi: int = None
    ) -> None:
        """Save the surface forcing fields to netCDF4 files.

        This method saves the dataset by grouping it into subsets based on the data frequency. The subsets are then written
        to one or more netCDF4 files. The filenames of the output files reflect the temporal coverage of the data.

        There are two modes of saving the dataset:

          1. **Single File Mode (default)**:

            If both `np_eta` and `np_xi` are `None`, the entire dataset, divided by temporal subsets, is saved as a single netCDF4 file
            with the base filename specified by `filepath.nc`.

          2. **Partitioned Mode**:

            - If either `np_eta` or `np_xi` is specified, the dataset is divided into spatial tiles along the eta-axis and xi-axis.
            - Each spatial tile is saved as a separate netCDF4 file.

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

        dataset_list, output_filenames = group_dataset(self.ds.load(), str(filepath))
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
            yaml.dump(yaml_data, file, default_flow_style=False)

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
