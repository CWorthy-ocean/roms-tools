import xarray as xr
import numpy as np
import yaml
import importlib.metadata
from typing import Dict, Union, Optional
from dataclasses import dataclass, field, asdict
from roms_tools.setup.grid import Grid
from roms_tools.setup.vertical_coordinate import VerticalCoordinate
from roms_tools.setup.roms_mixin import ROMSToolsMixin
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
class BoundaryForcing(ROMSToolsMixin):
    """
    Represents boundary forcing for ROMS.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
    vertical_coordinate: VerticalCoordinate
        Object representing the vertical coordinate information.
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
        - "climatology" (bool): Indicates if the BGC data is climatology data. Defaults to False.
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
    ...     vertical_coordinate=vertical_coordinate,
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
    vertical_coordinate: VerticalCoordinate
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

        data = self._get_data()

        d_meta = super().get_variable_metadata()
        bdry_coords, rename = super().get_boundary_info()

        vars_2d = ["zeta"]
        vars_3d = ["temp", "salt", "u", "v"]
        data_vars = super().regrid_data(data, vars_2d, vars_3d)
        data_vars = super().process_velocities(data_vars)
        ds = self._write_into_dataset(data_vars, d_meta, bdry_coords, rename)
        ds = self._add_global_metadata(ds, bdry_coords, rename)

        for direction in ["south", "east", "north", "west"]:
            nan_check(
                ds[f"zeta_{direction}"].isel(time=0),
                self.grid.ds.mask_rho.isel(**bdry_coords["rho"][direction]),
            )

        object.__setattr__(self, "ds", ds)

        if self.bgc_source is not None:
            bgc_data = self._get_bgc_data()

            vars_2d = []
            vars_3d = bgc_data.var_names.values()
            bgc_data_vars = super().regrid_data(bgc_data, vars_2d, vars_3d)

            # Ensure time coordinate matches if climatology is applied in one case but not the other
            if (
                not self.physics_source["climatology"]
                and self.bgc_source["climatology"]
            ):
                for var in bgc_data_vars.keys():
                    bgc_data_vars[var] = bgc_data_vars[var].assign_coords(
                        {"time": data_vars["temp"]["time"]}
                    )

            ds_bgc = self._write_into_dataset(bgc_data_vars, d_meta)
            ds_bgc = self._add_global_metadata(ds_bgc)

            object.__setattr__(self, "ds_bgc", ds_bgc)

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
            # set self.physics_source["climatology"] to False if not provided
            object.__setattr__(
                self,
                "bgc_source",
                {
                    **self.bgc_source,
                    "climatology": self.bgc_source.get("climatology", False),
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

    def _write_into_dataset(self, data_vars, d_meta, bdry_coords, rename, ds=None):

        if ds is None:
            # save in new dataset
            ds = xr.Dataset()

        for direction in ["south", "east", "north", "west"]:
            if self.boundaries[direction]:

                for var in data_vars.keys():
                    if var in ["u", "ubar"]:
                        ds[f"{var}_{direction}"] = (
                            data_vars[var]
                            .isel(**bdry_coords["u"][direction])
                            .rename(**rename["u"][direction])
                            .astype(np.float32)
                        )
                    elif var in ["v", "vbar"]:
                        ds[f"{var}_{direction}"] = (
                            data_vars[var]
                            .isel(**bdry_coords["v"][direction])
                            .rename(**rename["v"][direction])
                            .astype(np.float32)
                        )
                    else:
                        ds[f"{var}_{direction}"] = (
                            data_vars[var]
                            .isel(**bdry_coords["rho"][direction])
                            .rename(**rename["rho"][direction])
                            .astype(np.float32)
                        )
                    ds[f"{var}_{direction}"].attrs[
                        "long_name"
                    ] = f"{direction}ern boundary {d_meta[var]['long_name']}"
                    ds[f"{var}_{direction}"].attrs["units"] = d_meta[var]["units"]

        return ds

    def _add_global_metadata(self, ds, bdry_coords, rename):

        for direction in ["south", "east", "north", "west"]:

            if self.boundaries[direction]:

                lat_rho = self.grid.ds.lat_rho.isel(
                    **bdry_coords["rho"][direction]
                ).rename(**rename["rho"][direction])
                lon_rho = self.grid.ds.lon_rho.isel(
                    **bdry_coords["rho"][direction]
                ).rename(**rename["rho"][direction])
                layer_depth_rho = (
                    self.vertical_coordinate.ds["layer_depth_rho"]
                    .isel(**bdry_coords["rho"][direction])
                    .rename(**rename["rho"][direction])
                )
                interface_depth_rho = (
                    self.vertical_coordinate.ds["interface_depth_rho"]
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
                    self.vertical_coordinate.ds["layer_depth_u"]
                    .isel(**bdry_coords["u"][direction])
                    .rename(**rename["u"][direction])
                )
                interface_depth_u = (
                    self.vertical_coordinate.ds["interface_depth_u"]
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
                    self.vertical_coordinate.ds["layer_depth_v"]
                    .isel(**bdry_coords["v"][direction])
                    .rename(**rename["v"][direction])
                )
                interface_depth_v = (
                    self.vertical_coordinate.ds["interface_depth_v"]
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

        ds = ds.drop_vars(
            [
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
        )

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
        ds.attrs["phsyics_source"] = self.physics_source["name"]
        if self.bgc_source is not None:
            ds.attrs["bgc_source"] = self.bgc_source["name"]

        # Translate the time coordinate to days since the model reference date
        # TODO: Check if we need to convert from 12:00:00 to 00:00:00 as in matlab scripts
        model_reference_date = np.datetime64(self.model_reference_date)

        # Convert the time coordinate to the format expected by ROMS (days since model reference date)
        bry_time = ds["time"] - model_reference_date
        ds = ds.assign_coords(bry_time=("time", bry_time.data))
        ds["bry_time"].attrs[
            "long_name"
        ] = f"time since {np.datetime_as_string(model_reference_date, unit='D')}"

        ds.attrs["theta_s"] = self.vertical_coordinate.ds["theta_s"].item()
        ds.attrs["theta_b"] = self.vertical_coordinate.ds["theta_b"].item()
        ds.attrs["Tcline"] = self.vertical_coordinate.ds["Tcline"].item()
        ds.attrs["hc"] = self.vertical_coordinate.ds["hc"].item()
        ds["sc_r"] = self.vertical_coordinate.ds["sc_r"]
        ds["Cs_r"] = self.vertical_coordinate.ds["Cs_r"]

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
            - "temp_{direction}": Potential temperature.
            - "salt_{direction}": Salinity.
            - "zeta_{direction}": Sea surface height.
            - "u_{direction}": u-flux component.
            - "v_{direction}": v-flux component.
            - "ubar_{direction}": Vertically integrated u-flux component.
            - "vbar_{direction}": Vertically integrated v-flux component.
            where {direction} can be one of ["south", "east", "north", "west"].
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

        field = self.ds[varname].isel(time=time).load()

        title = field.long_name

        # chose colorbar
        if varname.startswith(("u", "v", "ubar", "vbar", "zeta")):
            vmax = max(field.max().values, -field.min().values)
            vmin = -vmax
            cmap = plt.colormaps.get_cmap("RdBu_r")
        else:
            vmax = field.max().values
            vmin = field.min().values
            cmap = plt.colormaps.get_cmap("YlOrRd")
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
                        self.ds[depth_label]
                        for depth_label in self.ds.coords
                        if any(
                            depth_label.startswith(prefix) for prefix in depths_to_check
                        )
                        and (
                            set(self.ds[depth_label].dims) - {"s_w"}
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

        # Group dataset by year
        gb = self.ds.groupby("time.year")

        for year, group_ds in gb:
            # Further group each yearly group by month
            sub_gb = group_ds.groupby("time.month")

            for month, ds in sub_gb:
                # Chunk the dataset by the specified time chunk size
                ds = ds.chunk({"time": time_chunk_size})
                datasets.append(ds)

                # Determine the number of days in the month
                num_days_in_month = calendar.monthrange(year, month)[1]
                first_day = ds.time.dt.day.values[0]
                last_day = ds.time.dt.day.values[-1]

                # Create filename based on whether the dataset contains a full month
                if first_day == 1 and last_day == num_days_in_month:
                    # Full month format: "filepath.YYYYMM.nc"
                    year_month_str = f"{year}{month:02}"
                    filename = f"{filepath}.{year_month_str}.nc"
                else:
                    # Partial month format: "filepath.YYYYMMDD-DD.nc"
                    year_month_day_str = f"{year}{month:02}{first_day:02}-{last_day:02}"
                    filename = f"{filepath}.{year_month_day_str}.nc"
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

        # Serialize VerticalCoordinate data
        vertical_coordinate_data = asdict(self.vertical_coordinate)
        vertical_coordinate_data.pop("ds", None)  # Exclude non-serializable fields
        vertical_coordinate_data.pop("grid", None)  # Exclude non-serializable fields

        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"

        # Create header
        header = f"---\nroms_tools_version: {roms_tools_version}\n---\n"

        grid_yaml_data = {"Grid": grid_data}
        vertical_coordinate_yaml_data = {"VerticalCoordinate": vertical_coordinate_data}

        boundary_forcing_data = {
            "BoundaryForcing": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "boundaries": self.boundaries,
                "physics_source": self.physics_source,
                "model_reference_date": self.model_reference_date.isoformat(),
            }
        }

        yaml_data = {
            **grid_yaml_data,
            **vertical_coordinate_yaml_data,
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

        # Create VerticalCoordinate instance from the YAML file
        vertical_coordinate = VerticalCoordinate.from_yaml(filepath)
        grid = vertical_coordinate.grid

        # Create and return an instance of InitialConditions
        return cls(
            grid=grid,
            vertical_coordinate=vertical_coordinate,
            **boundary_forcing_data,
        )
