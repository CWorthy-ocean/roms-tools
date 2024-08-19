import xarray as xr
import pandas as pd
import dask
import yaml
from datatree import DataTree
import importlib.metadata
from dataclasses import dataclass, field, asdict
from roms_tools.setup.grid import Grid
from datetime import datetime
import numpy as np
from typing import Dict, Union
from roms_tools.setup.mixins import ROMSToolsMixins
from roms_tools.setup.datasets import (
    ERA5Dataset,
    ERA5Correction,
    CESMBGCSurfaceForcingDataset,
)
from roms_tools.setup.utils import nan_check, interpolate_from_climatology
from roms_tools.setup.plot import _plot
import calendar
import matplotlib.pyplot as plt


@dataclass(frozen=True, kw_only=True)
class AtmosphericForcing(ROMSToolsMixins):
    """
    Represents atmospheric forcing data for ocean modeling.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
    start_time : datetime
        Start time of the desired forcing data.
    end_time : datetime
        End time of the desired forcing data.
    physics_source : Dict[str, Union[str, None]]
        Dictionary specifying the source of the physical surface forcing data:
        - "name" (str): Name of the data source (e.g., "ERA5").
        - "path" (str): Path to the physical data file. Can contain wildcards.
        - "climatology" (bool): Indicates if the physical data is climatology data. Defaults to False.
    bgc_source : Optional[Dict[str, Union[str, None]]]
        Dictionary specifying the source of the biogeochemical (BGC) initial condition data:
        - "name" (str): Name of the BGC data source (e.g., "CESM_REGRIDDED").
        - "path" (str): Path to the BGC data file. Can contain wildcards.
        - "climatology" (bool): Indicates if the BGC data is climatology data. Defaults to False.
    correct_radiation : bool
        Whether to correct shortwave radiation. Default is False.
    use_coarse_grid: bool
        Whether to interpolate to coarsened grid. Default is False.
    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.

    Attributes
    ----------
    ds : xr.Dataset
        Xarray Dataset containing the atmospheric forcing data.


    Examples
    --------
    >>> atm_forcing = AtmosphericForcing(
    ...     grid=grid,
    ...     start_time=datetime(2000, 1, 1),
    ...     end_time=datetime(2000, 1, 2),
    ...     physics_source={"name": "ERA5", "path": "physics_data.nc"},
    ...     correct_radiation=True,
    ... )
    """

    grid: Grid
    start_time: datetime
    end_time: datetime
    physics_source: Dict[str, Union[str, None]]
    bgc_source: Dict[str, Union[str, None]] = None
    correct_radiation: bool = False
    use_coarse_grid: bool = False
    model_reference_date: datetime = datetime(2000, 1, 1)
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        self._input_checks()
        lon, lat, angle, straddle = super().get_target_lon_lat(self.use_coarse_grid)
        object.__setattr__(self, "target_lon", lon)
        object.__setattr__(self, "target_lat", lat)

        data = self._get_data()
        data.choose_subdomain(
            latitude_range=[lat.min().values, lat.max().values],
            longitude_range=[lon.min().values, lon.max().values],
            margin=2,
            straddle=straddle,
        )
        vars_2d = ["uwnd", "vwnd", "swrad", "lwrad", "Tair", "qair", "rain"]
        vars_3d = []
        data_vars = super().regrid_data(data, vars_2d, vars_3d, lon, lat)
        data_vars = super().process_velocities(data_vars, angle, interpolate=False)

        if self.correct_radiation:
            correction_data = self._get_correction_data()
            # choose same subdomain as forcing data so that we can use same mask
            coords_correction = {
                correction_data.dim_names["latitude"]: data.ds[
                    data.dim_names["latitude"]
                ],
                correction_data.dim_names["longitude"]: data.ds[
                    data.dim_names["longitude"]
                ],
            }
            correction_data.choose_subdomain(coords_correction, straddle=straddle)
            # apply mask from ERA5 data
            if "mask" in data.var_names.keys():
                mask = xr.where(
                    data.ds[data.var_names["mask"]].isel(time=0).isnull(), 0, 1
                )
                for var in correction_data.ds.data_vars:
                    correction_data.ds[var] = xr.where(
                        mask == 1, correction_data.ds[var], np.nan
                    )
            vars_2d = ["swr_corr"]
            vars_3d = []
            # spatial interpolation
            data_vars_corr = super().regrid_data(
                correction_data, vars_2d, vars_3d, lon, lat
            )
            # temporal interpolation
            corr_factor = interpolate_from_climatology(
                data_vars_corr["swr_corr"],
                correction_data.dim_names["time"],
                time=data_vars["swrad"].time,
            )

            data_vars["swrad"] = data_vars["swrad"] * corr_factor

        object.__setattr__(data, "data_vars", data_vars)

        if self.bgc_source is not None:
            bgc_data = self._get_bgc_data()
            bgc_data.choose_subdomain(
                latitude_range=[lat.min().values, lat.max().values],
                longitude_range=[lon.min().values, lon.max().values],
                margin=2,
                straddle=straddle,
            )

            vars_2d = bgc_data.var_names.keys()
            vars_3d = []
            data_vars = super().regrid_data(bgc_data, vars_2d, vars_3d, lon, lat)
            object.__setattr__(bgc_data, "data_vars", data_vars)
        else:
            bgc_data = None

        d_meta = super().get_variable_metadata()

        ds = self._write_into_datatree(data, bgc_data, d_meta)

        if self.use_coarse_grid:
            mask = self.grid.ds["mask_coarse"].rename(
                {"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"}
            )
        else:
            mask = self.grid.ds["mask_rho"]

        for var in ds["physics"].data_vars:
            nan_check(ds["physics"][var].isel(time=0), mask)

        object.__setattr__(self, "ds", ds)

    def _input_checks(self):

        if "name" not in self.physics_source.keys():
            raise ValueError("`physics_source` must include a 'name'.")
        if "path" not in self.physics_source.keys():
            raise ValueError("`physics_source` must include a 'path'.")
        # set self.physics_source["climatology"] to False if not provided
        object.__setattr__(
            self,
            "physics_source",
            {
                **self.physics_source,
                "climatology": self.physics_source.get("climatology", False),
            },
        )

        if self.bgc_source is not None:
            if "name" not in self.bgc_source.keys():
                raise ValueError(
                    "`bgc_source` must include a 'name' if it is provided."
                )
            if "path" not in self.bgc_source.keys():
                raise ValueError(
                    "`bgc_source` must include a 'path' if it is provided."
                )
            # set self.bgc_source["climatology"] to False if not provided
            object.__setattr__(
                self,
                "bgc_source",
                {
                    **self.bgc_source,
                    "climatology": self.bgc_source.get("climatology", False),
                },
            )

    def _get_data(self):

        if self.physics_source["name"] == "ERA5":
            data = ERA5Dataset(
                filename=self.physics_source["path"],
                start_time=self.start_time,
                end_time=self.end_time,
                climatology=self.physics_source["climatology"],
            )
            data.post_process()
        else:
            raise ValueError(
                'Only "ERA5" is a valid option for physics_source["name"].'
            )

        return data

    def _get_correction_data(self):

        if self.physics_source["name"] == "ERA5":
            correction_data = ERA5Correction()
        else:
            raise ValueError(
                "The 'correct_radiation' feature is currently only supported for 'ERA5' as the physics source. "
                "Please ensure your 'physics_source' is set to 'ERA5' or implement additional handling for other sources."
            )

        return correction_data

    def _get_bgc_data(self):

        if self.bgc_source["name"] == "CESM_REGRIDDED":

            bgc_data = CESMBGCSurfaceForcingDataset(
                filename=self.bgc_source["path"],
                start_time=self.start_time,
                end_time=self.end_time,
                climatology=self.bgc_source["climatology"],
            )
        else:
            raise ValueError(
                'Only "CESM_REGRIDDED" is a valid option for bgc_source["name"].'
            )

        return bgc_data

    def _write_into_dataset(self, data, d_meta):

        # save in new dataset
        ds = xr.Dataset()

        for var in data.data_vars.keys():
            ds[var] = data.data_vars[var].astype(np.float32)
            ds[var].attrs["long_name"] = d_meta[var]["long_name"]
            ds[var].attrs["units"] = d_meta[var]["units"]

        if self.use_coarse_grid:
            ds = ds.assign_coords({"lon": self.target_lon, "lat": self.target_lat})
            ds = ds.rename({"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"})

        # Preserve absolute time coordinate for readability
        ds = ds.assign_coords({"abs_time": ds["time"]})

        # Convert the time coordinate to the format expected by ROMS
        if data.climatology:
            # Convert to pandas TimedeltaIndex
            timedelta_index = pd.to_timedelta(ds["time"].values)
            # Determine the start of the year for the base_datetime
            start_of_year = datetime(self.model_reference_date.year, 1, 1)
            # Calculate the offset from midnight of the new year
            offset = self.model_reference_date - start_of_year
            sfc_time = xr.DataArray(
                timedelta_index - offset,
                dims="time",
            )
        else:
            sfc_time = (
                (ds["time"] - np.datetime64(self.model_reference_date)).astype(
                    "float64"
                )
                / 3600
                / 24
                * 1e-9
            )

        ds = ds.assign_coords({"time": sfc_time})
        ds["time"].attrs[
            "long_name"
        ] = f"days since {np.datetime_as_string(np.datetime64(self.model_reference_date), unit='D')}"
        ds["time"].encoding["units"] = "days"
        if data.climatology:
            ds["time"].attrs["cycle_length"] = 365.25

        return ds

    def _write_into_datatree(self, data, bgc_data, d_meta):

        ds = self._add_global_metadata()

        ds = DataTree(name="root", data=ds)

        ds_physics = self._write_into_dataset(data, d_meta)
        ds_physics = self._add_global_metadata(ds_physics)
        ds_physics.attrs["physics_source"] = self.physics_source["name"]
        ds_physics = DataTree(name="physics", parent=ds, data=ds_physics)

        if bgc_data:
            ds_bgc = self._write_into_dataset(bgc_data, d_meta)
            ds_bgc = self._add_global_metadata(ds_bgc)
            ds_bgc.attrs["bgc_source"] = self.bgc_source["name"]
            ds_bgc = DataTree(name="bgc", parent=ds, data=ds_bgc)

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
        ds.attrs["physics_source"] = self.physics_source["name"]
        if self.bgc_source is not None:
            ds.attrs["bgc_source"] = self.bgc_source["name"]
        ds.attrs["correct_radiation"] = str(self.correct_radiation)
        ds.attrs["use_coarse_grid"] = str(self.use_coarse_grid)
        ds.attrs["model_reference_date"] = str(self.model_reference_date)

        return ds

    def plot(self, varname, time=0) -> None:
        """
        Plot the specified atmospheric forcing field for a given time slice.

        Parameters
        ----------
        varname : str
            The name of the atmospheric forcing field to plot. Options include:
            - "uwnd": 10 meter wind in x-direction.
            - "vwnd": 10 meter wind in y-direction.
            - "swrad": Downward short-wave (solar) radiation.
            - "lwrad": Downward long-wave (thermal) radiation.
            - "Tair": Air temperature at 2m.
            - "qair": Absolute humidity at 2m.
            - "rain": Total precipitation.
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
            If the specified varname is not one of the valid options.


        Examples
        --------
        >>> atm_forcing.plot("uwnd", time=0)
        """

        if varname in self.ds["physics"]:
            ds = self.ds["physics"]
        else:
            if "bgc" in self.ds and varname in self.ds["bgc"]:
                ds = self.ds["bgc"]
            else:
                raise ValueError(
                    f"Variable '{varname}' is not found in 'physics' or 'bgc' datasets."
                )

        field = ds[varname].isel(time=time).load()
        title = field.long_name

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
            coarse_grid=self.use_coarse_grid,
            title=title,
            kwargs=kwargs,
            c="g",
        )

    def save(self, filepath: str, time_chunk_size: int = 1) -> None:
        """
        Save the interpolated atmospheric forcing fields to netCDF4 files.

        This method groups the dataset by year and month, chunks the data by the specified
        time chunk size, and saves each chunked subset to a separate netCDF4 file named
        according to the year, month, and day range if not a complete month of data is included.

        Parameters
        ----------
        filepath : str
            The base path and filename for the output files. The files will be named with
            the format "filepath.YYYYMM.nc" if a full month of data is included, or
            "filepath.YYYYMMDD-DD.nc" otherwise.
        time_chunk_size : int, optional
            Number of time slices to include in each chunk along the time dimension. Default is 1,
            meaning each chunk contains one time slice.

        Returns
        -------
        None
        """

        datasets = []
        filenames = []
        writes = []

        for node in ["physics", "bgc"]:
            if node in self.ds:
                ds = self.ds[node].to_dataset()
                if hasattr(ds["time"], "cycle_length"):
                    filename = f"{filepath}_{node}_clim.nc"
                    print("Saving the following file:")
                    print(filename)
                    ds.to_netcdf(filename)
                else:
                    # Group dataset by year
                    gb = ds.groupby("abs_time.year")

                    for year, group_ds in gb:
                        # Further group each yearly group by month
                        sub_gb = group_ds.groupby("abs_time.month")

                        for month, ds in sub_gb:
                            # Chunk the dataset by the specified time chunk size
                            ds = ds.chunk({"time": time_chunk_size})
                            datasets.append(ds)

                            # Determine the number of days in the month
                            num_days_in_month = calendar.monthrange(year, month)[1]
                            first_day = ds.abs_time.dt.day.values[0]
                            last_day = ds.abs_time.dt.day.values[-1]

                            # Create filename based on whether the dataset contains a full month
                            if first_day == 1 and last_day == num_days_in_month:
                                # Full month format: "filepath_physics_YYYYMM.nc"
                                year_month_str = f"{year}{month:02}"
                                filename = f"{filepath}_{node}_{year_month_str}.nc"
                            else:
                                # Partial month format: "filepath_physics_YYYYMMDD-DD.nc"
                                year_month_day_str = (
                                    f"{year}{month:02}{first_day:02}-{last_day:02}"
                                )
                                filename = f"{filepath}_{node}_{year_month_day_str}.nc"
                            filenames.append(filename)

                    print("Saving the following files:")
                    for filename in filenames:
                        print(filename)

                    for ds, filename in zip(datasets, filenames):

                        # Prepare the dataset for writing to a netCDF file without immediately computing
                        write = ds.to_netcdf(filename, compute=False)
                        writes.append(write)

                    # Perform the actual write operations in parallel
                    dask.compute(*writes)

    def to_yaml(self, filepath: str) -> None:
        """
        Export the parameters of the class to a YAML file, including the version of roms-tools.

        Parameters
        ----------
        filepath : str
            The path to the YAML file where the parameters will be saved.
        """
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
        atmospheric_forcing_data = {
            "AtmosphericForcing": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "physics_source": self.physics_source,
                "correct_radiation": self.correct_radiation,
                "use_coarse_grid": self.use_coarse_grid,
                "model_reference_date": self.model_reference_date.isoformat(),
            }
        }
        # Include bgc_source if it's not None
        if self.bgc_source is not None:
            atmospheric_forcing_data["AtmosphericForcing"][
                "bgc_source"
            ] = self.bgc_source

        # Merge YAML data while excluding empty sections
        yaml_data = {
            **grid_yaml_data,
            **atmospheric_forcing_data,
        }

        with open(filepath, "w") as file:
            # Write header
            file.write(header)
            # Write YAML data
            yaml.dump(yaml_data, file, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filepath: str) -> "AtmosphericForcing":
        """
        Create an instance of the AtmosphericForcing class from a YAML file.

        Parameters
        ----------
        filepath : str
            The path to the YAML file from which the parameters will be read.

        Returns
        -------
        AtmosphericForcing
            An instance of the AtmosphericForcing class.
        """
        # Read the entire file content
        with open(filepath, "r") as file:
            file_content = file.read()

        # Split the content into YAML documents
        documents = list(yaml.safe_load_all(file_content))

        atmospheric_forcing_data = None

        # Process the YAML documents
        for doc in documents:
            if doc is None:
                continue
            if "AtmosphericForcing" in doc:
                atmospheric_forcing_data = doc["AtmosphericForcing"]

        if atmospheric_forcing_data is None:
            raise ValueError(
                "No AtmosphericForcing configuration found in the YAML file."
            )

        # Convert from string to datetime
        for date_string in ["model_reference_date", "start_time", "end_time"]:
            atmospheric_forcing_data[date_string] = datetime.fromisoformat(
                atmospheric_forcing_data[date_string]
            )

        # Create Grid instance from the YAML file
        grid = Grid.from_yaml(filepath)

        # Create and return an instance of AtmosphericForcing
        return cls(
            grid=grid,
            **atmospheric_forcing_data,
        )
