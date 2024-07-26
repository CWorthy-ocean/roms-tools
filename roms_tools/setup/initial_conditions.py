import xarray as xr
import numpy as np
import yaml
import importlib.metadata
from dataclasses import dataclass, field, asdict
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
from roms_tools.setup.plot import _plot, _section_plot, _profile_plot, _line_plot
import matplotlib.pyplot as plt


@dataclass(frozen=True, kw_only=True)
class InitialConditions:
    """
    Represents initial conditions for ROMS.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
    vertical_coordinate: VerticalCoordinate
        Object representing the vertical coordinate information
    ini_time : datetime
        Desired initialization time.
    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.
    source : str, optional
        Source of the initial condition data. Default is "GLORYS".
    filename: str
        Path to the source data file. Can contain wildcards.

    Attributes
    ----------
    ds : xr.Dataset
        Xarray Dataset containing the initial condition data.

    """

    grid: Grid
    vertical_coordinate: VerticalCoordinate
    ini_time: datetime
    model_reference_date: datetime = datetime(2000, 1, 1)
    source: str = "GLORYS"
    filename: str
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        # Check that the source is "GLORYS"
        if self.source != "GLORYS":
            raise ValueError('Only "GLORYS" is a valid option for source.')
        if self.source == "GLORYS":
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
            start_time=self.ini_time,
            var_names=varnames.values(),
            dim_names=dims,
        )

        lon = self.grid.ds.lon_rho
        lat = self.grid.ds.lat_rho
        angle = self.grid.ds.angle

        # operate on longitudes between -180 and 180 unless ROMS domain lies at least 5 degrees in lontitude away from Greenwich meridian
        lon = xr.where(lon > 180, lon - 360, lon)
        straddle = True
        if not self.grid.straddle and abs(lon).min() > 5:
            lon = xr.where(lon < 0, lon + 360, lon)
            straddle = False

        # The following consists of two steps:
        # Step 1: Choose subdomain of forcing data including safety margin for interpolation, and Step 2: Convert to the proper longitude range.
        # We perform these two steps for two reasons:
        # A) Since the horizontal dimensions consist of a single chunk, selecting a subdomain before interpolation is a lot more performant.
        # B) Step 1 is necessary to avoid discontinuous longitudes that could be introduced by Step 2. Specifically, discontinuous longitudes
        # can lead to artifacts in the interpolation process. Specifically, if there is a data gap if data is not global,
        # discontinuous longitudes could result in values that appear to come from a distant location instead of producing NaNs.
        # These NaNs are important as they can be identified and handled appropriately by the nan_check function.
        data.choose_subdomain(
            latitude_range=[lat.min().values, lat.max().values],
            longitude_range=[lon.min().values, lon.max().values],
            margin=2,
            straddle=straddle,
        )

        # interpolate onto desired grid
        fill_dims = [dims["latitude"], dims["longitude"]]

        # 2d interpolation
        mask = xr.where(data.ds[varnames["ssh"]].isel(time=0).isnull(), 0, 1)
        coords = {dims["latitude"]: lat, dims["longitude"]: lon}

        ssh = fill_and_interpolate(
            data.ds[varnames["ssh"]].astype(np.float64),
            mask,
            fill_dims=fill_dims,
            coords=coords,
            method="linear",
        )

        # 3d interpolation

        # extrapolate deepest value all the way to bottom ("flooding")
        for var in ["temp", "salt", "u", "v"]:
            data.ds[varnames[var]] = extrapolate_deepest_to_bottom(
                data.ds[varnames[var]], dims["depth"]
            )

        mask = xr.where(data.ds[varnames["temp"]].isel(time=0).isnull(), 0, 1)
        coords = {
            dims["latitude"]: lat,
            dims["longitude"]: lon,
            dims["depth"]: self.vertical_coordinate.ds["layer_depth_rho"],
        }

        # setting fillvalue_interp to None means that we allow extrapolation in the
        # interpolation step to avoid NaNs at the surface if the lowest depth in original
        # data is greater than zero
        data_vars = {}
        for var in ["temp", "salt", "u", "v"]:
            data_vars[var] = fill_and_interpolate(
                data.ds[varnames[var]].astype(np.float64),
                mask,
                fill_dims=fill_dims,
                coords=coords,
                method="linear",
                fillvalue_interp=None,
            )

        # rotate to grid orientation
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

        # save in new dataset
        ds = xr.Dataset()

        ds["temp"] = data_vars["temp"].astype(np.float32)
        ds["temp"].attrs["long_name"] = "Potential temperature"
        ds["temp"].attrs["units"] = "Celsius"

        ds["salt"] = data_vars["salt"].astype(np.float32)
        ds["salt"].attrs["long_name"] = "Salinity"
        ds["salt"].attrs["units"] = "PSU"

        ds["zeta"] = ssh.astype(np.float32)
        ds["zeta"].attrs["long_name"] = "Free surface"
        ds["zeta"].attrs["units"] = "m"

        ds["u"] = u.astype(np.float32)
        ds["u"].attrs["long_name"] = "u-flux component"
        ds["u"].attrs["units"] = "m/s"

        ds["v"] = v.astype(np.float32)
        ds["v"].attrs["long_name"] = "v-flux component"
        ds["v"].attrs["units"] = "m/s"

        # initialize vertical velocity to zero
        ds["w"] = xr.zeros_like(
            self.vertical_coordinate.ds["interface_depth_rho"].expand_dims(
                time=ds[dims["time"]]
            )
        ).astype(np.float32)
        ds["w"].attrs["long_name"] = "w-flux component"
        ds["w"].attrs["units"] = "m/s"

        ds["ubar"] = ubar.transpose(dims["time"], "eta_rho", "xi_u").astype(np.float32)
        ds["ubar"].attrs["long_name"] = "vertically integrated u-flux component"
        ds["ubar"].attrs["units"] = "m/s"

        ds["vbar"] = vbar.transpose(dims["time"], "eta_v", "xi_rho").astype(np.float32)
        ds["vbar"].attrs["long_name"] = "vertically integrated v-flux component"
        ds["vbar"].attrs["units"] = "m/s"

        ds = ds.assign_coords(
            {
                "layer_depth_u": self.vertical_coordinate.ds["layer_depth_u"],
                "layer_depth_v": self.vertical_coordinate.ds["layer_depth_v"],
                "interface_depth_u": self.vertical_coordinate.ds["interface_depth_u"],
                "interface_depth_v": self.vertical_coordinate.ds["interface_depth_v"],
            }
        )

        ds.attrs["title"] = "ROMS initial conditions file created by ROMS-Tools"
        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"
        ds.attrs["roms_tools_version"] = roms_tools_version
        ds.attrs["ini_time"] = str(self.ini_time)
        ds.attrs["model_reference_date"] = str(self.model_reference_date)
        ds.attrs["source"] = self.source

        if dims["time"] != "time":
            ds = ds.rename({dims["time"]: "time"})

        # Translate the time coordinate to days since the model reference date
        model_reference_date = np.datetime64(self.model_reference_date)

        # Convert the time coordinate to the format expected by ROMS (days since model reference date)
        ocean_time = (ds["time"] - model_reference_date).astype("float64") * 1e-9
        ds = ds.assign_coords(ocean_time=("time", np.float32(ocean_time)))
        ds["ocean_time"].attrs[
            "long_name"
        ] = f"time since {np.datetime_as_string(model_reference_date, unit='D')}"
        ds["ocean_time"].attrs["units"] = "seconds"

        ds["theta_s"] = self.vertical_coordinate.ds["theta_s"]
        ds["theta_b"] = self.vertical_coordinate.ds["theta_b"]
        ds["Tcline"] = self.vertical_coordinate.ds["Tcline"]
        ds["hc"] = self.vertical_coordinate.ds["hc"]
        ds["sc_r"] = self.vertical_coordinate.ds["sc_r"]
        ds["Cs_r"] = self.vertical_coordinate.ds["Cs_r"]

        ds = ds.drop_vars(["s_rho"])

        object.__setattr__(self, "ds", ds)

        ds["zeta"].load()
        nan_check(ds["zeta"].squeeze(), self.grid.ds.mask_rho)

    def plot(
        self,
        varname,
        s=None,
        eta=None,
        xi=None,
        depth_contours=False,
        layer_contours=False,
    ) -> None:
        """
        Plot the initial conditions field for a given eta-, xi-, or s_rho-slice.

        Parameters
        ----------
        varname : str
            The name of the initial conditions field to plot. Options include:
            - "temp": Potential temperature.
            - "salt": Salinity.
            - "zeta": Free surface.
            - "u": u-flux component.
            - "v": v-flux component.
            - "w": w-flux component.
            - "ubar": Vertically integrated u-flux component.
            - "vbar": Vertically integrated v-flux component.
        s : int, optional
            The index of the vertical layer to plot. Default is None.
        eta : int, optional
            The eta-index to plot. Default is None.
        xi : int, optional
            The xi-index to plot. Default is None.
        depth_contours : bool, optional
            Whether to include depth contours in the plot. Default is False.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        Raises
        ------
        ValueError
            If the specified varname is not one of the valid options.
            If field is 3D and none of s_rho, eta, xi are specified.
            If field is 2D and both eta and xi are specified.
        """

        if len(self.ds[varname].squeeze().dims) == 3 and not any(
            [s is not None, eta is not None, xi is not None]
        ):
            raise ValueError(
                "For 3D fields, at least one of s, eta, or xi must be specified."
            )

        if len(self.ds[varname].squeeze().dims) == 2 and all(
            [eta is not None, xi is not None]
        ):
            raise ValueError("For 2D fields, specify either eta or xi, not both.")

        self.ds[varname].load()
        field = self.ds[varname].squeeze()

        if all(dim in field.dims for dim in ["eta_rho", "xi_rho"]):
            interface_depth = self.ds.interface_depth_rho
        elif all(dim in field.dims for dim in ["eta_rho", "xi_u"]):
            interface_depth = self.ds.interface_depth_u
        elif all(dim in field.dims for dim in ["eta_v", "xi_rho"]):
            interface_depth = self.ds.interface_depth_v

        # slice the field as desired
        title = field.long_name
        if s is not None:
            title = title + f", s_rho = {field.s_rho[s].item()}"
            field = field.isel(s_rho=s)
        else:
            depth_contours = False

        if eta is not None:
            if "eta_rho" in field.dims:
                title = title + f", eta_rho = {field.eta_rho[eta].item()}"
                field = field.isel(eta_rho=eta)
                interface_depth = interface_depth.isel(eta_rho=eta)
            elif "eta_v" in field.dims:
                title = title + f", eta_v = {field.eta_v[eta].item()}"
                field = field.isel(eta_v=eta)
                interface_depth = interface_depth.isel(eta_v=eta)
            else:
                raise ValueError(
                    f"None of the expected dimensions (eta_rho, eta_v) found in ds[{varname}]."
                )
        if xi is not None:
            if "xi_rho" in field.dims:
                title = title + f", xi_rho = {field.xi_rho[xi].item()}"
                field = field.isel(xi_rho=xi)
                interface_depth = interface_depth.isel(xi_rho=xi)
            elif "xi_u" in field.dims:
                title = title + f", xi_u = {field.xi_u[xi].item()}"
                field = field.isel(xi_u=xi)
                interface_depth = interface_depth.isel(xi_u=xi)
            else:
                raise ValueError(
                    f"None of the expected dimensions (xi_rho, xi_u) found in ds[{varname}]."
                )

        # chose colorbar
        if varname in ["u", "v", "w", "ubar", "vbar", "zeta"]:
            vmax = max(field.max().values, -field.min().values)
            vmin = -vmax
            cmap = plt.colormaps.get_cmap("RdBu_r")
        else:
            vmax = field.max().values
            vmin = field.min().values
            cmap = plt.colormaps.get_cmap("YlOrRd")
        cmap.set_bad(color="gray")
        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

        if eta is None and xi is None:
            _plot(
                self.grid.ds,
                field=field,
                straddle=self.grid.straddle,
                depth_contours=depth_contours,
                title=title,
                kwargs=kwargs,
                c="g",
            )
        else:
            if not layer_contours:
                interface_depth = None
            else:
                # restrict number of layer_contours to 10 for the sake of plot clearity
                nr_layers = len(interface_depth["s_w"])
                selected_layers = np.linspace(
                    0, nr_layers - 1, min(nr_layers, 10), dtype=int
                )
                interface_depth = interface_depth.isel(s_w=selected_layers)

            if len(field.dims) == 2:
                _section_plot(
                    field, interface_depth=interface_depth, title=title, kwargs=kwargs
                )
            else:
                if "s_rho" in field.dims:
                    _profile_plot(field, title=title)
                else:
                    _line_plot(field, title=title)

    def save(self, filepath: str) -> None:
        """
        Save the initial conditions information to a netCDF4 file.

        Parameters
        ----------
        filepath
        """
        self.ds.to_netcdf(filepath)

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

        initial_conditions_data = {
            "InitialConditions": {
                "filename": self.filename,
                "ini_time": self.ini_time.isoformat(),
                "model_reference_date": self.model_reference_date.isoformat(),
                "source": self.source,
            }
        }

        yaml_data = {
            **grid_yaml_data,
            **vertical_coordinate_yaml_data,
            **initial_conditions_data,
        }

        with open(filepath, "w") as file:
            # Write header
            file.write(header)
            # Write YAML data
            yaml.dump(yaml_data, file, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filepath: str) -> "InitialConditions":
        """
        Create an instance of the InitialConditions class from a YAML file.

        Parameters
        ----------
        filepath : str
            The path to the YAML file from which the parameters will be read.

        Returns
        -------
        InitialConditions
            An instance of the InitialConditions class.
        """
        # Read the entire file content
        with open(filepath, "r") as file:
            file_content = file.read()

        # Split the content into YAML documents
        documents = list(yaml.safe_load_all(file_content))

        initial_conditions_data = None

        # Process the YAML documents
        for doc in documents:
            if doc is None:
                continue
            if "InitialConditions" in doc:
                initial_conditions_data = doc["InitialConditions"]
                break

        if initial_conditions_data is None:
            raise ValueError(
                "No InitialConditions configuration found in the YAML file."
            )

        # Convert from string to datetime
        for date_string in ["model_reference_date", "ini_time"]:
            initial_conditions_data[date_string] = datetime.fromisoformat(
                initial_conditions_data[date_string]
            )

        # Create VerticalCoordinate instance from the YAML file
        vertical_coordinate = VerticalCoordinate.from_yaml(filepath)
        grid = vertical_coordinate.grid

        # Create and return an instance of InitialConditions
        return cls(
            grid=grid,
            vertical_coordinate=vertical_coordinate,
            **initial_conditions_data,
        )
