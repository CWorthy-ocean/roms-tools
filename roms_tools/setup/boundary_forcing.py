import xarray as xr
import numpy as np
import pandas as pd
import yaml
from datatree import DataTree
import importlib.metadata
from typing import Dict, Union, Optional
from dataclasses import dataclass, field, asdict
from roms_tools.setup.grid import Grid
from roms_tools.setup.mixins import ROMSToolsMixins
from datetime import datetime
from roms_tools.setup.datasets import GLORYSDataset, CESMBGCDataset
from roms_tools.setup.utils import (
    nan_check,
)
from roms_tools.setup.plot import _section_plot, _line_plot
import calendar
import dask
import matplotlib.pyplot as plt


@dataclass(frozen=True, kw_only=True)
class BoundaryForcing(ROMSToolsMixins):
    """
    Represents boundary forcing for ROMS.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
    start_time : datetime
        Start time of the desired boundary forcing data.
    end_time : datetime
        End time of the desired boundary forcing data.
    boundaries : Dict[str, bool], optional
        Dictionary specifying which boundaries are forced (south, east, north, west). Default is all True.
    physics_source : Dict[str, Union[str, None]]
        Dictionary specifying the source of the physical boundary forcing data:
        - "name" (str): Name of the data source (e.g., "GLORYS").
        - "path" (str): Path to the physical data file. Can contain wildcards.
        - "climatology" (bool): Indicates if the physical data is climatology data. Defaults to False.
    bgc_source : Optional[Dict[str, Union[str, None]]]
        Dictionary specifying the source of the biogeochemical (BGC) initial condition data:
        - "name" (str): Name of the BGC data source (e.g., "CESM_REGRIDDED").
        - "path" (str): Path to the BGC data file. Can contain wildcards.
        - "climatology" (bool): Indicates if the BGC data is climatology data. Defaults to True.
    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.

    Attributes
    ----------
    ds : xr.Dataset
        Xarray Dataset containing the atmospheric forcing data.

    Examples
    --------
    >>> boundary_forcing = BoundaryForcing(
    ...     grid=grid,
    ...     boundaries={"south": True, "east": True, "north": False, "west": True},
    ...     start_time=datetime(2022, 1, 1),
    ...     end_time=datetime(2022, 1, 2),
    ...     physics_source={"name": "GLORYS", "path": "physics_data.nc"},
    ...     bgc_source={
    ...         "name": "CESM_REGRIDDED",
    ...         "path": "bgc_data.nc",
    ...         "climatology": True,
    ...     },
    ... )
    """

    grid: Grid
    start_time: datetime
    end_time: datetime
    boundaries: Dict[str, bool] = field(
        default_factory=lambda: {
            "south": True,
            "east": True,
            "north": True,
            "west": True,
        }
    )
    physics_source: Dict[str, Union[str, None]]
    bgc_source: Optional[Dict[str, Union[str, None]]] = None
    model_reference_date: datetime = datetime(2000, 1, 1)

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        self._input_checks()
        lon, lat, angle, straddle = super().get_target_lon_lat()

        data = self._get_data()
        data.choose_subdomain(
            latitude_range=[lat.min().values, lat.max().values],
            longitude_range=[lon.min().values, lon.max().values],
            margin=2,
            straddle=straddle,
        )

        vars_2d = ["zeta"]
        vars_3d = ["temp", "salt", "u", "v"]
        data_vars = super().regrid_data(data, vars_2d, vars_3d, lon, lat)
        data_vars = super().process_velocities(data_vars, angle, "u", "v")
        object.__setattr__(data, "data_vars", data_vars)

        if self.bgc_source is not None:
            bgc_data = self._get_bgc_data()
            bgc_data.choose_subdomain(
                latitude_range=[lat.min().values, lat.max().values],
                longitude_range=[lon.min().values, lon.max().values],
                margin=2,
                straddle=straddle,
            )

            vars_2d = []
            vars_3d = bgc_data.var_names.keys()
            data_vars = super().regrid_data(bgc_data, vars_2d, vars_3d, lon, lat)
            object.__setattr__(bgc_data, "data_vars", data_vars)
        else:
            bgc_data = None

        d_meta = super().get_variable_metadata()
        bdry_coords, rename = super().get_boundary_info()

        ds = self._write_into_datatree(data, bgc_data, d_meta, bdry_coords, rename)

        for direction in ["south", "east", "north", "west"]:
            if self.boundaries[direction]:
                nan_check(
                    ds["physics"][f"zeta_{direction}"].isel(bry_time=0),
                    self.grid.ds.mask_rho.isel(**bdry_coords["rho"][direction]),
                )

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
            # set self.bgc_source["climatology"] to True if not provided
            object.__setattr__(
                self,
                "bgc_source",
                {
                    **self.bgc_source,
                    "climatology": self.bgc_source.get("climatology", True),
                },
            )

    def _get_data(self):

        if self.physics_source["name"] == "GLORYS":
            data = GLORYSDataset(
                filename=self.physics_source["path"],
                start_time=self.start_time,
                end_time=self.end_time,
                climatology=self.physics_source["climatology"],
            )
        else:
            raise ValueError(
                'Only "GLORYS" is a valid option for physics_source["name"].'
            )

        return data

    def _get_bgc_data(self):

        if self.bgc_source["name"] == "CESM_REGRIDDED":

            data = CESMBGCDataset(
                filename=self.bgc_source["path"],
                start_time=self.start_time,
                end_time=self.end_time,
                climatology=self.bgc_source["climatology"],
            )
            data.post_process()
        else:
            raise ValueError(
                'Only "CESM_REGRIDDED" is a valid option for bgc_source["name"].'
            )

        return data

    def _write_into_dataset(self, data, d_meta, bdry_coords, rename):

        # save in new dataset
        ds = xr.Dataset()

        for direction in ["south", "east", "north", "west"]:
            if self.boundaries[direction]:

                for var in data.data_vars.keys():
                    if var in ["u", "ubar"]:
                        ds[f"{var}_{direction}"] = (
                            data.data_vars[var]
                            .isel(**bdry_coords["u"][direction])
                            .rename(**rename["u"][direction])
                            .astype(np.float32)
                        )
                    elif var in ["v", "vbar"]:
                        ds[f"{var}_{direction}"] = (
                            data.data_vars[var]
                            .isel(**bdry_coords["v"][direction])
                            .rename(**rename["v"][direction])
                            .astype(np.float32)
                        )
                    else:
                        ds[f"{var}_{direction}"] = (
                            data.data_vars[var]
                            .isel(**bdry_coords["rho"][direction])
                            .rename(**rename["rho"][direction])
                            .astype(np.float32)
                        )
                    ds[f"{var}_{direction}"].attrs[
                        "long_name"
                    ] = f"{direction}ern boundary {d_meta[var]['long_name']}"
                    ds[f"{var}_{direction}"].attrs["units"] = d_meta[var]["units"]

        # Gracefully handle dropping variables that might not be present
        variables_to_drop = [
            "s_rho",
            "layer_depth_rho",
            "layer_depth_u",
            "layer_depth_v",
            "interface_depth_rho",
            "interface_depth_u",
            "interface_depth_v",
            "lat_rho",
            "lon_rho",
            "lat_u",
            "lon_u",
            "lat_v",
            "lon_v",
        ]
        existing_vars = [var for var in variables_to_drop if var in ds]
        ds = ds.drop_vars(existing_vars)

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
            bry_time = xr.DataArray(
                timedelta_index - offset,
                dims="time",
            )
        else:
            # TODO: Check if we need to convert from 12:00:00 to 00:00:00 as in matlab scripts
            bry_time = ds["time"] - np.datetime64(self.model_reference_date)

        ds = ds.assign_coords({"bry_time": bry_time})
        ds["bry_time"].attrs[
            "long_name"
        ] = f"nanoseconds since {np.datetime_as_string(np.datetime64(self.model_reference_date), unit='ns')}"
        ds["bry_time"].encoding["units"] = "nanoseconds"
        ds = ds.swap_dims({"time": "bry_time"})
        ds = ds.drop_vars("time")

        if data.climatology:
            ds["bry_time"].attrs["cycle_length"] = 365.25

        return ds

    def _write_into_datatree(self, data, bgc_data, d_meta, bdry_coords, rename):

        ds = self._add_global_metadata()
        ds["sc_r"] = self.grid.ds["sc_r"]
        ds["Cs_r"] = self.grid.ds["Cs_r"]

        ds = DataTree(name="root", data=ds)

        ds_physics = self._write_into_dataset(data, d_meta, bdry_coords, rename)
        ds_physics = self._add_coordinates(bdry_coords, rename, ds_physics)
        ds_physics = self._add_global_metadata(ds_physics)
        ds_physics.attrs["physics_source"] = self.physics_source["name"]

        ds_physics = DataTree(name="physics", parent=ds, data=ds_physics)

        if bgc_data:
            ds_bgc = self._write_into_dataset(bgc_data, d_meta, bdry_coords, rename)
            ds_bgc = self._add_coordinates(bdry_coords, rename, ds_bgc)
            ds_bgc = self._add_global_metadata(ds_bgc)
            ds_bgc.attrs["bgc_source"] = self.bgc_source["name"]
            ds_bgc = DataTree(name="bgc", parent=ds, data=ds_bgc)

        return ds

    def _add_coordinates(self, bdry_coords, rename, ds=None):

        if ds is None:
            ds = xr.Dataset()

        for direction in ["south", "east", "north", "west"]:

            if self.boundaries[direction]:

                lat_rho = self.grid.ds.lat_rho.isel(
                    **bdry_coords["rho"][direction]
                ).rename(**rename["rho"][direction])
                lon_rho = self.grid.ds.lon_rho.isel(
                    **bdry_coords["rho"][direction]
                ).rename(**rename["rho"][direction])
                layer_depth_rho = (
                    self.grid.ds["layer_depth_rho"]
                    .isel(**bdry_coords["rho"][direction])
                    .rename(**rename["rho"][direction])
                )
                interface_depth_rho = (
                    self.grid.ds["interface_depth_rho"]
                    .isel(**bdry_coords["rho"][direction])
                    .rename(**rename["rho"][direction])
                )

                lat_u = self.grid.ds.lat_u.isel(**bdry_coords["u"][direction]).rename(
                    **rename["u"][direction]
                )
                lon_u = self.grid.ds.lon_u.isel(**bdry_coords["u"][direction]).rename(
                    **rename["u"][direction]
                )
                layer_depth_u = (
                    self.grid.ds["layer_depth_u"]
                    .isel(**bdry_coords["u"][direction])
                    .rename(**rename["u"][direction])
                )
                interface_depth_u = (
                    self.grid.ds["interface_depth_u"]
                    .isel(**bdry_coords["u"][direction])
                    .rename(**rename["u"][direction])
                )

                lat_v = self.grid.ds.lat_v.isel(**bdry_coords["v"][direction]).rename(
                    **rename["v"][direction]
                )
                lon_v = self.grid.ds.lon_v.isel(**bdry_coords["v"][direction]).rename(
                    **rename["v"][direction]
                )
                layer_depth_v = (
                    self.grid.ds["layer_depth_v"]
                    .isel(**bdry_coords["v"][direction])
                    .rename(**rename["v"][direction])
                )
                interface_depth_v = (
                    self.grid.ds["interface_depth_v"]
                    .isel(**bdry_coords["v"][direction])
                    .rename(**rename["v"][direction])
                )

                ds = ds.assign_coords(
                    {
                        f"layer_depth_rho_{direction}": layer_depth_rho,
                        f"layer_depth_u_{direction}": layer_depth_u,
                        f"layer_depth_v_{direction}": layer_depth_v,
                        f"interface_depth_rho_{direction}": interface_depth_rho,
                        f"interface_depth_u_{direction}": interface_depth_u,
                        f"interface_depth_v_{direction}": interface_depth_v,
                        f"lat_rho_{direction}": lat_rho,
                        f"lat_u_{direction}": lat_u,
                        f"lat_v_{direction}": lat_v,
                        f"lon_rho_{direction}": lon_rho,
                        f"lon_u_{direction}": lon_u,
                        f"lon_v_{direction}": lon_v,
                    }
                )

        # Gracefully handle dropping variables that might not be present
        variables_to_drop = [
            "s_rho",
            "layer_depth_rho",
            "layer_depth_u",
            "layer_depth_v",
            "interface_depth_rho",
            "interface_depth_u",
            "interface_depth_v",
            "lat_rho",
            "lon_rho",
            "lat_u",
            "lon_u",
            "lat_v",
            "lon_v",
        ]
        existing_vars = [var for var in variables_to_drop if var in ds]
        ds = ds.drop_vars(existing_vars)

        return ds

    def _add_global_metadata(self, ds=None):

        if ds is None:
            ds = xr.Dataset()
        ds.attrs["title"] = "ROMS boundary forcing file created by ROMS-Tools"
        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"
        ds.attrs["roms_tools_version"] = roms_tools_version
        ds.attrs["start_time"] = str(self.start_time)
        ds.attrs["end_time"] = str(self.end_time)
        ds.attrs["model_reference_date"] = str(self.model_reference_date)

        ds.attrs["theta_s"] = self.grid.ds.attrs["theta_s"]
        ds.attrs["theta_b"] = self.grid.ds.attrs["theta_b"]
        ds.attrs["hc"] = self.grid.ds.attrs["hc"]

        return ds

    def plot(
        self,
        varname,
        time=0,
        layer_contours=False,
    ) -> None:
        """
        Plot the boundary forcing field for a given time-slice.

        Parameters
        ----------
        varname : str
            The name of the initial conditions field to plot. Options include:
            - "temp_{direction}": Potential temperature, where {direction} can be one of ["south", "east", "north", "west"].
            - "salt_{direction}": Salinity, where {direction} can be one of ["south", "east", "north", "west"].
            - "zeta_{direction}": Sea surface height, where {direction} can be one of ["south", "east", "north", "west"].
            - "u_{direction}": u-flux component, where {direction} can be one of ["south", "east", "north", "west"].
            - "v_{direction}": v-flux component, where {direction} can be one of ["south", "east", "north", "west"].
            - "ubar_{direction}": Vertically integrated u-flux component, where {direction} can be one of ["south", "east", "north", "west"].
            - "vbar_{direction}": Vertically integrated v-flux component, where {direction} can be one of ["south", "east", "north", "west"].
            - "PO4_{direction}": Dissolved Inorganic Phosphate (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "NO3_{direction}": Dissolved Inorganic Nitrate (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "SiO3_{direction}": Dissolved Inorganic Silicate (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "NH4_{direction}": Dissolved Ammonia (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "Fe_{direction}": Dissolved Inorganic Iron (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "Lig_{direction}": Iron Binding Ligand (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "O2_{direction}": Dissolved Oxygen (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "DIC_{direction}": Dissolved Inorganic Carbon (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "DIC_ALT_CO2_{direction}": Dissolved Inorganic Carbon, Alternative CO2 (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "ALK_{direction}": Alkalinity (meq/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "ALK_ALT_CO2_{direction}": Alkalinity, Alternative CO2 (meq/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "DOC_{direction}": Dissolved Organic Carbon (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "DON_{direction}": Dissolved Organic Nitrogen (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "DOP_{direction}": Dissolved Organic Phosphorus (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "DOPr_{direction}": Refractory Dissolved Organic Phosphorus (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "DONr_{direction}": Refractory Dissolved Organic Nitrogen (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "DOCr_{direction}": Refractory Dissolved Organic Carbon (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "zooC_{direction}": Zooplankton Carbon (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "spChl_{direction}": Small Phytoplankton Chlorophyll (mg/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "spC_{direction}": Small Phytoplankton Carbon (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "spP_{direction}": Small Phytoplankton Phosphorous (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "spFe_{direction}": Small Phytoplankton Iron (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "spCaCO3_{direction}": Small Phytoplankton CaCO3 (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "diatChl_{direction}": Diatom Chlorophyll (mg/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "diatC_{direction}": Diatom Carbon (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "diatP_{direction}": Diatom Phosphorus (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "diatFe_{direction}": Diatom Iron (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "diatSi_{direction}": Diatom Silicate (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "diazChl_{direction}": Diazotroph Chlorophyll (mg/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "diazC_{direction}": Diazotroph Carbon (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "diazP_{direction}": Diazotroph Phosphorus (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
            - "diazFe_{direction}": Diazotroph Iron (mmol/m³), where {direction} can be one of ["south", "east", "north", "west"].
        time : int, optional
            The time index to plot. Default is 0.
        layer_contours : bool, optional
            Whether to include layer contours in the plot. This can help visualize the depth levels
            of the field. Default is False.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        Raises
        ------
        ValueError
            If the specified varname is not one of the valid options.
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

        field = ds[varname].isel(bry_time=time).load()
        title = field.long_name

        # chose colorbar
        if varname.startswith(("u", "v", "ubar", "vbar", "zeta")):
            vmax = max(field.max().values, -field.min().values)
            vmin = -vmax
            cmap = plt.colormaps.get_cmap("RdBu_r")
        else:
            vmax = field.max().values
            vmin = field.min().values
            if varname.startswith(("temp", "salt")):
                cmap = plt.colormaps.get_cmap("YlOrRd")
            else:
                cmap = plt.colormaps.get_cmap("YlGn")
        cmap.set_bad(color="gray")
        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

        if len(field.dims) == 2:
            if layer_contours:
                depths_to_check = [
                    "interface_depth_rho",
                    "interface_depth_u",
                    "interface_depth_v",
                ]
                try:
                    interface_depth = next(
                        ds[depth_label]
                        for depth_label in ds.coords
                        if any(
                            depth_label.startswith(prefix) for prefix in depths_to_check
                        )
                        and (
                            set(ds[depth_label].dims) - {"s_w"}
                            == set(field.dims) - {"s_rho"}
                        )
                    )
                except StopIteration:
                    raise ValueError(
                        f"None of the expected depths ({', '.join(depths_to_check)}) have dimensions matching field.dims"
                    )
                # restrict number of layer_contours to 10 for the sake of plot clearity
                nr_layers = len(interface_depth["s_w"])
                selected_layers = np.linspace(
                    0, nr_layers - 1, min(nr_layers, 10), dtype=int
                )
                interface_depth = interface_depth.isel(s_w=selected_layers)

            else:
                interface_depth = None

            _section_plot(
                field, interface_depth=interface_depth, title=title, kwargs=kwargs
            )
        else:
            _line_plot(field, title=title)

    def save(self, filepath: str, time_chunk_size: int = 1) -> None:
        """
        Save the interpolated boundary forcing fields to netCDF4 files.

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
                # copy vertical coordinate variables from parent to children because I believe this is info that ROMS needs
                for var in self.ds.data_vars:
                    ds[var] = self.ds[var]
                if hasattr(ds["bry_time"], "cycle_length"):
                    filename = f"{filepath}_{node}_clim.nc"
                    filenames.append(filename)
                    datasets.append(ds)
                else:
                    # Group dataset by year
                    gb = ds.groupby("abs_time.year")

                    for year, group_ds in gb:
                        # Further group each yearly group by month
                        sub_gb = group_ds.groupby("abs_time.month")

                        for month, ds in sub_gb:
                            # Chunk the dataset by the specified time chunk size
                            ds = ds.chunk({"bry_time": time_chunk_size})
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
        for ds, filename in zip(datasets, filenames):
            print(filename)
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

        grid_yaml_data = {"Grid": grid_data}

        boundary_forcing_data = {
            "BoundaryForcing": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "boundaries": self.boundaries,
                "physics_source": self.physics_source,
                "bgc_source": self.bgc_source,
                "model_reference_date": self.model_reference_date.isoformat(),
            }
        }

        yaml_data = {
            **grid_yaml_data,
            **boundary_forcing_data,
        }

        with open(filepath, "w") as file:
            # Write header
            file.write(header)
            # Write YAML data
            yaml.dump(yaml_data, file, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filepath: str) -> "BoundaryForcing":
        """
        Create an instance of the BoundaryForcing class from a YAML file.

        Parameters
        ----------
        filepath : str
            The path to the YAML file from which the parameters will be read.

        Returns
        -------
        BoundaryForcing
            An instance of the BoundaryForcing class.
        """
        # Read the entire file content
        with open(filepath, "r") as file:
            file_content = file.read()

        # Split the content into YAML documents
        documents = list(yaml.safe_load_all(file_content))

        boundary_forcing_data = None

        # Process the YAML documents
        for doc in documents:
            if doc is None:
                continue
            if "BoundaryForcing" in doc:
                boundary_forcing_data = doc["BoundaryForcing"]
                break

        if boundary_forcing_data is None:
            raise ValueError("No BoundaryForcing configuration found in the YAML file.")

        # Convert from string to datetime
        for date_string in ["model_reference_date", "start_time", "end_time"]:
            boundary_forcing_data[date_string] = datetime.fromisoformat(
                boundary_forcing_data[date_string]
            )

        grid = Grid.from_yaml(filepath)

        # Create and return an instance of InitialConditions
        return cls(
            grid=grid,
            **boundary_forcing_data,
        )
