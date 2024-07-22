import xarray as xr
import numpy as np
from typing import Dict
from dataclasses import dataclass, field
from roms_tools.setup.grid import Grid
from roms_tools.setup.vertical_coordinate import VerticalCoordinate
from datetime import datetime
from roms_tools.setup.datasets import Dataset
from roms_tools.setup.fill import fill_and_interpolate
from roms_tools.setup.utils import (
    nan_check,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    extrapolate_deepest_to_bottom,
)
from roms_tools.setup.plot import _section_plot, _line_plot
import calendar
import dask
import matplotlib.pyplot as plt


@dataclass(frozen=True, kw_only=True)
class BoundaryForcing:
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
    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.
    source : str, optional
        Source of the boundary forcing data. Default is "glorys".
    filename: str
        Path to the source data file. Can contain wildcards.

    Attributes
    ----------
    ds : xr.Dataset
        Xarray Dataset containing the atmospheric forcing data.

    Notes
    -----
    This class represents atmospheric forcing data used in ocean modeling. It provides a convenient
    interface to work with forcing data including shortwave radiation correction and river forcing.
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
    model_reference_date: datetime = datetime(2000, 1, 1)
    source: str = "glorys"
    filename: str
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        lon = self.grid.ds.lon_rho
        lat = self.grid.ds.lat_rho
        angle = self.grid.ds.angle

        if self.source == "glorys":
            dims = {
                "longitude": "longitude",
                "latitude": "latitude",
                "depth": "depth",
                "time": "time",
            }

            varnames = {
                "temp": "thetao",
                "salt": "so",
                "u": "uo",
                "v": "vo",
                "ssh": "zos",
            }
        data = Dataset(
            filename=self.filename,
            start_time=self.start_time,
            end_time=self.end_time,
            var_names=varnames.values(),
            dim_names=dims,
        )

        # operate on longitudes between -180 and 180 unless ROMS domain lies at least 5 degrees in lontitude away from Greenwich meridian
        lon = xr.where(lon > 180, lon - 360, lon)
        straddle = True
        if not self.grid.straddle and abs(lon).min() > 5:
            lon = xr.where(lon < 0, lon + 360, lon)
            straddle = False

        data.choose_subdomain(
            latitude_range=[lat.min().values, lat.max().values],
            longitude_range=[lon.min().values, lon.max().values],
            margin=2,
            straddle=straddle,
        )

        # extrapolate deepest value all the way to bottom ("flooding") to prepare for 3d interpolation
        for var in ["temp", "salt", "u", "v"]:
            data.ds[varnames[var]] = extrapolate_deepest_to_bottom(
                data.ds[varnames[var]], dims["depth"]
            )

        # interpolate onto desired grid
        fill_dims = [dims["latitude"], dims["longitude"]]

        # 2d interpolation
        coords = {dims["latitude"]: lat, dims["longitude"]: lon}
        mask = xr.where(data.ds[varnames["ssh"]].isel(time=0).isnull(), 0, 1)

        ssh = fill_and_interpolate(
            data.ds[varnames["ssh"]].astype(np.float64),
            mask,
            fill_dims=fill_dims,
            coords=coords,
            method="linear",
        )

        # 3d interpolation
        coords = {
            dims["latitude"]: lat,
            dims["longitude"]: lon,
            dims["depth"]: self.vertical_coordinate.ds["layer_depth_rho"],
        }
        mask = xr.where(data.ds[varnames["temp"]].isel(time=0).isnull(), 0, 1)

        data_vars = {}
        # setting fillvalue_interp to None means that we allow extrapolation in the
        # interpolation step to avoid NaNs at the surface if the lowest depth in original
        # data is greater than zero

        for var in ["temp", "salt", "u", "v"]:

            data_vars[var] = fill_and_interpolate(
                data.ds[varnames[var]].astype(np.float64),
                mask,
                fill_dims=fill_dims,
                coords=coords,
                method="linear",
                fillvalue_interp=None,
            )

        # rotate velocities to grid orientation
        u_rot = data_vars["u"] * np.cos(angle) + data_vars["v"] * np.sin(angle)
        v_rot = data_vars["v"] * np.cos(angle) - data_vars["u"] * np.sin(angle)

        # interpolate to u- and v-points
        u = interpolate_from_rho_to_u(u_rot)
        v = interpolate_from_rho_to_v(v_rot)

        # 3d masks for ROMS domain
        umask = self.grid.ds.mask_u.expand_dims({"s_rho": u.s_rho})
        vmask = self.grid.ds.mask_v.expand_dims({"s_rho": v.s_rho})

        u = u * umask
        v = v * vmask

        # Compute barotropic velocity

        # thicknesses
        dz = -self.vertical_coordinate.ds["interface_depth_rho"].diff(dim="s_w")
        dz = dz.rename({"s_w": "s_rho"})
        # thicknesses at u- and v-points
        dzu = interpolate_from_rho_to_u(dz)
        dzv = interpolate_from_rho_to_v(dz)

        ubar = (dzu * u).sum(dim="s_rho") / dzu.sum(dim="s_rho")
        vbar = (dzv * v).sum(dim="s_rho") / dzv.sum(dim="s_rho")

        # Boundary coordinates for rho-points
        bdry_coords_rho = {
            "south": {"eta_rho": 0},
            "east": {"xi_rho": -1},
            "north": {"eta_rho": -1},
            "west": {"xi_rho": 0},
        }
        # How to rename the dimensions at rho-points
        rename_rho = {
            "south": {"xi_rho": "xi_rho_south"},
            "east": {"eta_rho": "eta_rho_east"},
            "north": {"xi_rho": "xi_rho_north"},
            "west": {"eta_rho": "eta_rho_west"},
        }

        # Boundary coordinates for u-points
        bdry_coords_u = {
            "south": {"eta_rho": 0},
            "east": {"xi_u": -1},
            "north": {"eta_rho": -1},
            "west": {"xi_u": 0},
        }
        # How to rename the dimensions at u-points
        rename_u = {
            "south": {"xi_u": "xi_u_south"},
            "east": {"eta_rho": "eta_u_east"},
            "north": {"xi_u": "xi_u_north"},
            "west": {"eta_rho": "eta_u_west"},
        }

        # Boundary coordinates for v-points
        bdry_coords_v = {
            "south": {"eta_v": 0},
            "east": {"xi_rho": -1},
            "north": {"eta_v": -1},
            "west": {"xi_rho": 0},
        }
        # How to rename the dimensions at v-points
        rename_v = {
            "south": {"xi_rho": "xi_v_south"},
            "east": {"eta_v": "eta_v_east"},
            "north": {"xi_rho": "xi_v_north"},
            "west": {"eta_v": "eta_v_west"},
        }

        ds = xr.Dataset()

        for direction in ["south", "east", "north", "west"]:

            if self.boundaries[direction]:

                ds[f"zeta_{direction}"] = (
                    ssh.isel(**bdry_coords_rho[direction])
                    .rename(**rename_rho[direction])
                    .astype(np.float32)
                )
                ds[f"zeta_{direction}"].attrs[
                    "long_name"
                ] = f"{direction}ern boundary sea surface height"
                ds[f"zeta_{direction}"].attrs["units"] = "m"

                ds[f"temp_{direction}"] = (
                    data_vars["temp"]
                    .isel(**bdry_coords_rho[direction])
                    .rename(**rename_rho[direction])
                    .astype(np.float32)
                )
                ds[f"temp_{direction}"].attrs[
                    "long_name"
                ] = f"{direction}ern boundary potential temperature"
                ds[f"temp_{direction}"].attrs["units"] = "Celsius"

                ds[f"salt_{direction}"] = (
                    data_vars["salt"]
                    .isel(**bdry_coords_rho[direction])
                    .rename(**rename_rho[direction])
                    .astype(np.float32)
                )
                ds[f"salt_{direction}"].attrs[
                    "long_name"
                ] = f"{direction}ern boundary salinity"
                ds[f"salt_{direction}"].attrs["units"] = "PSU"

                ds[f"u_{direction}"] = (
                    u.isel(**bdry_coords_u[direction])
                    .rename(**rename_u[direction])
                    .astype(np.float32)
                )
                ds[f"u_{direction}"].attrs[
                    "long_name"
                ] = f"{direction}ern boundary u-flux component"
                ds[f"u_{direction}"].attrs["units"] = "m/s"

                ds[f"v_{direction}"] = (
                    v.isel(**bdry_coords_v[direction])
                    .rename(**rename_v[direction])
                    .astype(np.float32)
                )
                ds[f"v_{direction}"].attrs[
                    "long_name"
                ] = f"{direction}ern boundary v-flux component"
                ds[f"v_{direction}"].attrs["units"] = "m/s"

                ds[f"ubar_{direction}"] = (
                    ubar.isel(**bdry_coords_u[direction])
                    .rename(**rename_u[direction])
                    .astype(np.float32)
                )
                ds[f"ubar_{direction}"].attrs[
                    "long_name"
                ] = f"{direction}ern boundary vertically integrated u-flux component"
                ds[f"ubar_{direction}"].attrs["units"] = "m/s"

                ds[f"vbar_{direction}"] = (
                    vbar.isel(**bdry_coords_v[direction])
                    .rename(**rename_v[direction])
                    .astype(np.float32)
                )
                ds[f"vbar_{direction}"].attrs[
                    "long_name"
                ] = f"{direction}ern boundary vertically integrated v-flux component"
                ds[f"vbar_{direction}"].attrs["units"] = "m/s"

                # assign the correct depth coordinates

                lat_rho = self.grid.ds.lat_rho.isel(
                    **bdry_coords_rho[direction]
                ).rename(**rename_rho[direction])
                lon_rho = self.grid.ds.lon_rho.isel(
                    **bdry_coords_rho[direction]
                ).rename(**rename_rho[direction])
                layer_depth_rho = (
                    self.vertical_coordinate.ds["layer_depth_rho"]
                    .isel(**bdry_coords_rho[direction])
                    .rename(**rename_rho[direction])
                )
                interface_depth_rho = (
                    self.vertical_coordinate.ds["interface_depth_rho"]
                    .isel(**bdry_coords_rho[direction])
                    .rename(**rename_rho[direction])
                )

                lat_u = self.grid.ds.lat_u.isel(**bdry_coords_u[direction]).rename(
                    **rename_u[direction]
                )
                lon_u = self.grid.ds.lon_u.isel(**bdry_coords_u[direction]).rename(
                    **rename_u[direction]
                )
                layer_depth_u = (
                    self.vertical_coordinate.ds["layer_depth_u"]
                    .isel(**bdry_coords_u[direction])
                    .rename(**rename_u[direction])
                )
                interface_depth_u = (
                    self.vertical_coordinate.ds["interface_depth_u"]
                    .isel(**bdry_coords_u[direction])
                    .rename(**rename_u[direction])
                )

                lat_v = self.grid.ds.lat_v.isel(**bdry_coords_v[direction]).rename(
                    **rename_v[direction]
                )
                lon_v = self.grid.ds.lon_v.isel(**bdry_coords_v[direction]).rename(
                    **rename_v[direction]
                )
                layer_depth_v = (
                    self.vertical_coordinate.ds["layer_depth_v"]
                    .isel(**bdry_coords_v[direction])
                    .rename(**rename_v[direction])
                )
                interface_depth_v = (
                    self.vertical_coordinate.ds["interface_depth_v"]
                    .isel(**bdry_coords_v[direction])
                    .rename(**rename_v[direction])
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
                        "s_rho",
                    ]
                )

        ds.attrs["Title"] = "ROMS boundary forcing file produced by roms-tools"

        if dims["time"] != "time":
            ds = ds.rename({dims["time"]: "time"})

        # Translate the time coordinate to days since the model reference date
        # TODO: Check if we need to convert from 12:00:00 to 00:00:00 as in matlab scripts
        model_reference_date = np.datetime64(self.model_reference_date)

        # Convert the time coordinate to the format expected by ROMS (days since model reference date)
        bry_time = ds["time"] - model_reference_date
        ds = ds.assign_coords(bry_time=("time", bry_time.data))
        ds["bry_time"].attrs[
            "long_name"
        ] = f"time since {np.datetime_as_string(model_reference_date, unit='D')}"

        ds["theta_s"] = self.vertical_coordinate.ds["theta_s"]
        ds["theta_b"] = self.vertical_coordinate.ds["theta_b"]
        ds["Tcline"] = self.vertical_coordinate.ds["Tcline"]
        ds["hc"] = self.vertical_coordinate.ds["hc"]
        ds["sc_r"] = self.vertical_coordinate.ds["sc_r"]
        ds["Cs_r"] = self.vertical_coordinate.ds["Cs_r"]

        object.__setattr__(self, "ds", ds)

        for direction in ["south", "east", "north", "west"]:
            nan_check(
                ds[f"zeta_{direction}"].isel(time=0),
                self.grid.ds.mask_rho.isel(**bdry_coords_rho[direction]),
            )

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

        self.ds.to_netcdf(filepath)
