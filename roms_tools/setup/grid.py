import copy
from dataclasses import dataclass, field, asdict

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import yaml
import importlib.metadata

from roms_tools.setup.topography import _add_topography_and_mask, _add_velocity_masks
from roms_tools.setup.plot import _plot, _section_plot, _profile_plot, _line_plot
from roms_tools.setup.utils import interpolate_from_rho_to_u, interpolate_from_rho_to_v
from roms_tools.setup.vertical_coordinate import sigma_stretch, compute_depth

import warnings

RADIUS_OF_EARTH = 6371315.0  # in m


@dataclass(frozen=True, kw_only=True)
class Grid:
    """
     A single ROMS grid.

     Used for creating, plotting, and then saving a new ROMS domain grid.

     Parameters
     ----------
     nx : int
         Number of grid points in the x-direction.
     ny : int
         Number of grid points in the y-direction.
     size_x : float
         Domain size in the x-direction (in kilometers).
     size_y : float
         Domain size in the y-direction (in kilometers).
     center_lon : float
         Longitude of grid center.
     center_lat : float
         Latitude of grid center.
     rot : float, optional
         Rotation of grid x-direction from lines of constant latitude, measured in degrees.
         Positive values represent a counterclockwise rotation.
         The default is 0, which means that the x-direction of the grid is aligned with lines of constant latitude.
    N : int, optional
        The number of vertical levels. The default is 100.
    theta_s : float, optional
        The surface control parameter. Must satisfy 0 < theta_s <= 10. The default is 5.0.
    theta_b : float, optional
        The bottom control parameter. Must satisfy 0 < theta_b <= 4. The default is 2.0.
    hc : float, optional
        The critical depth (in meters). The default is 300.0.
    topography_source : str, optional
        Specifies the data source to use for the topography. Options are
        "ETOPO5". The default is "ETOPO5".
    hmin : float, optional
        The minimum ocean depth (in meters). The default is 5.0.

     Attributes
     ----------
     nx : int
         Number of grid points in the x-direction.
     ny : int
         Number of grid points in the y-direction.
     size_x : float
         Domain size in the x-direction (in kilometers).
     size_y : float
         Domain size in the y-direction (in kilometers).
     center_lon : float
         Longitude of grid center.
     center_lat : float
         Latitude of grid center.
     rot : float
         Rotation of grid x-direction from lines of constant latitude.
    N : int
        The number of vertical levels.
    theta_s : float
        The surface control parameter.
    theta_b : float
        The bottom control parameter.
    hc : float
        The critical depth (in meters).
     topography_source : str
         Data source used for the topography.
     hmin : float
         Minimum ocean depth (in meters).
     ds : xr.Dataset
         The xarray Dataset containing the grid data.
     straddle : bool
         Indicates if the Greenwich meridian (0° longitude) intersects the domain.
         `True` if it does, `False` otherwise.

     Raises
     ------
     ValueError
         If you try to create a grid with domain size larger than 20000 km.
    """

    nx: int
    ny: int
    size_x: float
    size_y: float
    center_lon: float
    center_lat: float
    rot: float = 0
    N: int = 100
    theta_s: float = 5.0
    theta_b: float = 2.0
    hc: float = 300.0
    topography_source: str = "ETOPO5"
    hmin: float = 5.0
    ds: xr.Dataset = field(init=False, repr=False)
    straddle: bool = field(init=False, repr=False)

    def __post_init__(self):
        ds = _make_grid_ds(
            nx=self.nx,
            ny=self.ny,
            size_x=self.size_x,
            size_y=self.size_y,
            center_lon=self.center_lon,
            center_lat=self.center_lat,
            rot=self.rot,
        )
        # Calling object.__setattr__ is ugly but apparently this really is the best (current) way to combine __post_init__ with a frozen dataclass
        # see https://stackoverflow.com/questions/53756788/how-to-set-the-value-of-dataclass-field-in-post-init-when-frozen-true
        object.__setattr__(self, "ds", ds)

        # Update self.ds with topography and mask information
        self.add_topography_and_mask(
            topography_source=self.topography_source,
            hmin=self.hmin,
        )

        # Check if the Greenwich meridian goes through the domain.
        self._straddle()

        ds = _add_lat_lon_at_velocity_points(self.ds, self.straddle)
        object.__setattr__(self, "ds", ds)

        # Update the grid by adding grid variables that are coarsened versions of the original grid variables
        self._coarsen()

        self.create_vertical_coordinate()

    def add_topography_and_mask(self, topography_source="ETOPO5", hmin=5.0) -> None:
        """
        Add topography and mask to the grid dataset.

        This method processes the topography data and generates a land/sea mask.
        It applies several steps, including interpolating topography, smoothing
        the topography over the entire domain and locally, and filling in enclosed basins. The
        processed topography and mask are added to the grid's dataset as new variables.

        Parameters
        ----------
        topography_source : str, optional
            Specifies the data source to use for the topography. Options are
            "ETOPO5". The default is "ETOPO5".
        hmin : float, optional
            The minimum ocean depth (in meters). The default is 5.

        Returns
        -------
        None
            This method modifies the dataset in place and does not return a value.
        """

        ds = _add_topography_and_mask(self.ds, topography_source, hmin)
        # Assign the updated dataset back to the frozen dataclass
        object.__setattr__(self, "ds", ds)

    def _straddle(self) -> None:
        """
        Check if the Greenwich meridian goes through the domain.

        This method sets the `straddle` attribute to `True` if the Greenwich meridian
        (0° longitude) intersects the domain defined by `lon_rho`. Otherwise, it sets
        the `straddle` attribute to `False`.

        The check is based on whether the longitudinal differences between adjacent
        points exceed 300 degrees, indicating a potential wraparound of longitude.
        """

        if (
            np.abs(self.ds.lon_rho.diff("xi_rho")).max() > 300
            or np.abs(self.ds.lon_rho.diff("eta_rho")).max() > 300
        ):
            object.__setattr__(self, "straddle", True)
        else:
            object.__setattr__(self, "straddle", False)

    def _coarsen(self):
        """
        Update the grid by adding grid variables that are coarsened versions of the original
        fine-resoluion grid variables. The coarsening is by a factor of two.

        The specific variables being coarsened are:
        - `lon_rho` -> `lon_coarse`: Longitude at rho points.
        - `lat_rho` -> `lat_coarse`: Latitude at rho points.
        - `h` -> `h_coarse`: Bathymetry (depth).
        - `angle` -> `angle_coarse`: Angle between the xi axis and true east.
        - `mask_rho` -> `mask_coarse`: Land/sea mask at rho points.

        Returns
        -------
        None

        Modifies
        --------
        self.ds : xr.Dataset
            The dataset attribute of the Grid instance is updated with the new coarser variables.
        """
        d = {
            "angle": "angle_coarse",
            "mask_rho": "mask_coarse",
            "lat_rho": "lat_coarse",
            "lon_rho": "lon_coarse",
        }

        for fine_var, coarse_var in d.items():
            fine_field = self.ds[fine_var]
            if self.straddle and fine_var == "lon_rho":
                fine_field = xr.where(fine_field > 180, fine_field - 360, fine_field)

            coarse_field = _f2c(fine_field)
            if fine_var == "lon_rho":
                coarse_field = xr.where(
                    coarse_field < 0, coarse_field + 360, coarse_field
                )
            if coarse_var in ["lon_coarse", "lat_coarse"]:
                ds = self.ds.assign_coords(
                    {coarse_var: coarse_field}
                )
                object.__setattr__(self, "ds", ds)
            else:
                self.ds[coarse_var] = coarse_field
        self.ds["mask_coarse"] = xr.where(self.ds["mask_coarse"] > 0.5, 1, 0).astype(
            np.int32
        )

    def create_vertical_coordinate(self):

        h = self.ds.h

        cs_r, sigma_r = sigma_stretch(self.theta_s, self.theta_b, self.N, "r")
        zr = compute_depth(h * 0, h, self.hc, cs_r, sigma_r)
        cs_w, sigma_w = sigma_stretch(self.theta_s, self.theta_b, self.N, "w")
        zw = compute_depth(h * 0, h, self.hc, cs_w, sigma_w)

        self.ds["sc_r"] = sigma_r.astype(np.float32)
        self.ds["sc_r"].attrs["long_name"] = "S-coordinate at rho-points"
        self.ds["sc_r"].attrs["units"] = "nondimensional"

        self.ds["Cs_r"] = cs_r.astype(np.float32)
        self.ds["Cs_r"].attrs[
            "long_name"
        ] = "S-coordinate stretching curves at rho-points"
        self.ds["Cs_r"].attrs["units"] = "nondimensional"

        self.ds.attrs["theta_s"] = np.float32(self.theta_s)
        self.ds.attrs["theta_b"] = np.float32(self.theta_b)
        self.ds.attrs["hc"] = np.float32(self.hc)

        depth = -zr
        depth.attrs["long_name"] = "Layer depth at rho-points"
        depth.attrs["units"] = "m"

        depth_u = interpolate_from_rho_to_u(depth)
        depth_u.attrs["long_name"] = "Layer depth at u-points"
        depth_u.attrs["units"] = "m"

        depth_v = interpolate_from_rho_to_v(depth)
        depth_v.attrs["long_name"] = "Layer depth at v-points"
        depth_v.attrs["units"] = "m"

        interface_depth = -zw
        interface_depth.attrs["long_name"] = "Interface depth at rho-points"
        interface_depth.attrs["units"] = "m"

        interface_depth_u = interpolate_from_rho_to_u(interface_depth)
        interface_depth_u.attrs["long_name"] = "Interface depth at u-points"
        interface_depth_u.attrs["units"] = "m"

        interface_depth_v = interpolate_from_rho_to_v(interface_depth)
        interface_depth_v.attrs["long_name"] = "Interface depth at v-points"
        interface_depth_v.attrs["units"] = "m"

        ds = self.ds.assign_coords(
            {
                "layer_depth_rho": depth.astype(np.float32),
                "layer_depth_u": depth_u.astype(np.float32),
                "layer_depth_v": depth_v.astype(np.float32),
                "interface_depth_rho": interface_depth.astype(np.float32),
                "interface_depth_u": interface_depth_u.astype(np.float32),
                "interface_depth_v": interface_depth_v.astype(np.float32),
            }
        )
        ds = ds.drop_vars(["eta_rho", "xi_rho"])

        object.__setattr__(self, "ds", ds)

    def plot(self, bathymetry: bool = False) -> None:
        """
        Plot the grid.

        Parameters
        ----------
        bathymetry : bool
            Whether or not to plot the bathymetry. Default is False.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        """

        if bathymetry:
            kwargs = {"cmap": "YlGnBu"}

            _plot(
                self.ds,
                field=self.ds.h.where(self.ds.mask_rho),
                straddle=self.straddle,
                kwargs=kwargs,
            )
        else:
            _plot(self.ds, straddle=self.straddle)

    def plot_vertical_coordinate(
        self,
        varname="layer_depth_rho",
        s=None,
        eta=None,
        xi=None,
    ) -> None:
        """
        Plot the vertical coordinate system for a given eta-, xi-, or s-slice.

        Parameters
        ----------
        varname : str, optional
            The field to plot. Options are "depth_rho", "depth_u", "depth_v".
        s: int, optional
            The s-index to plot. Default is None.
        eta : int, optional
            The eta-index to plot. Default is None.
        xi : int, optional
            The xi-index to plot. Default is None.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        Raises
        ------
        ValueError
            If the specified varname is not one of the valid options.
            If none of s, eta, xi are specified.
        """

        if not any([s is not None, eta is not None, xi is not None]):
            raise ValueError("At least one of s, eta, or xi must be specified.")

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
            if "s_rho" in field.dims:
                title = title + f", s_rho = {field.s_rho[s].item()}"
                field = field.isel(s_rho=s)
            elif "s_w" in field.dims:
                title = title + f", s_w = {field.s_w[s].item()}"
                field = field.isel(s_w=s)
            else:
                raise ValueError(
                    f"None of the expected dimensions (s_rho, s_w) found in ds[{varname}]."
                )

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

        if eta is None and xi is None:
            vmax = field.max().values
            vmin = field.min().values
            cmap = plt.colormaps.get_cmap("YlGnBu")
            cmap.set_bad(color="gray")
            kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

            _plot(
                self.ds,
                field=field,
                straddle=self.straddle,
                depth_contours=True,
                title=title,
                kwargs=kwargs,
                c="g",
            )
        else:
            if len(field.dims) == 2:
                cmap = plt.colormaps.get_cmap("YlGnBu")
                cmap.set_bad(color="gray")
                kwargs = {"vmax": 0.0, "vmin": 0.0, "cmap": cmap, "add_colorbar": False}

                _section_plot(
                    xr.zeros_like(field),
                    interface_depth=interface_depth,
                    title=title,
                    kwargs=kwargs,
                )
            else:
                if "s_rho" in field.dims or "s_w" in field.dims:
                    _profile_plot(field, title=title)
                else:
                    _line_plot(field, title=title)

    def save(self, filepath: str) -> None:
        """
        Save the grid information to a netCDF4 file.

        Parameters
        ----------
        filepath
        """
        self.ds.to_netcdf(filepath)

    @classmethod
    def from_file(cls, filepath: str) -> "Grid":
        """
        Create a Grid instance from an existing file.

        Parameters
        ----------
        filepath : str
            Path to the file containing the grid information.

        Returns
        -------
        Grid
            A new instance of Grid populated with data from the file.
        """
        # Load the dataset from the file
        ds = xr.open_dataset(filepath)

        if not all(mask in ds for mask in ["mask_u", "mask_v"]):
            ds = _add_velocity_masks(ds)

        # Create a new Grid instance without calling __init__ and __post_init__
        grid = cls.__new__(cls)

        # Set the dataset for the grid instance
        object.__setattr__(grid, "ds", ds)

        # Check if the Greenwich meridian goes through the domain.
        grid._straddle()

        if not all(coord in grid.ds for coord in ["lat_u", "lon_u", "lat_v", "lon_v"]):
            ds = _add_lat_lon_at_velocity_points(grid.ds, grid.straddle)
            object.__setattr__(grid, "ds", ds)

        # Coarsen the grid if necessary
        if not all(
            var in grid.ds
            for var in [
                "lon_coarse",
                "lat_coarse",
                "h_coarse",
                "angle_coarse",
                "mask_coarse",
            ]
        ):
            grid._coarsen()
        if not all(var in grid.ds for var in ["depth_rho", "interface_depth_rho"]):
            grid.create_vertical_coordinate()

        # Manually set the remaining attributes by extracting parameters from dataset
        object.__setattr__(grid, "nx", ds.sizes["xi_rho"] - 2)
        object.__setattr__(grid, "ny", ds.sizes["eta_rho"] - 2)
        object.__setattr__(grid, "center_lon", ds.attrs["center_lon"])
        object.__setattr__(grid, "center_lat", ds.attrs["center_lat"])
        object.__setattr__(grid, "rot", ds.attrs["rot"])

        for attr in [
            "size_x",
            "size_y",
            "topography_source",
            "hmin",
        ]:
            if attr in ds.attrs:
                object.__setattr__(grid, attr, ds.attrs[attr])

        return grid

    def to_yaml(self, filepath: str) -> None:
        """
        Export the parameters of the class to a YAML file, including the version of roms-tools.

        Parameters
        ----------
        filepath : str
            The path to the YAML file where the parameters will be saved.
        """
        data = asdict(self)
        data.pop("ds", None)
        data.pop("straddle", None)

        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"

        # Create header
        header = f"---\nroms_tools_version: {roms_tools_version}\n---\n"

        # Use the class name as the top-level key
        yaml_data = {self.__class__.__name__: data}

        with open(filepath, "w") as file:
            # Write header
            file.write(header)
            # Write YAML data
            yaml.dump(yaml_data, file, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filepath: str) -> "Grid":
        """
        Create an instance of the class from a YAML file.

        Parameters
        ----------
        filepath : str
            The path to the YAML file from which the parameters will be read.

        Returns
        -------
        Grid
            An instance of the Grid class.
        """
        # Read the entire file content
        with open(filepath, "r") as file:
            file_content = file.read()

        # Split the content into YAML documents
        documents = list(yaml.safe_load_all(file_content))

        header_data = None
        grid_data = None

        # Iterate over documents to find the header and grid configuration
        for doc in documents:
            if doc is None:
                continue
            if "roms_tools_version" in doc:
                header_data = doc
            elif "Grid" in doc:
                grid_data = doc["Grid"]

        if header_data is None:
            raise ValueError("Version of ROMS-Tools not found in the YAML file.")
        else:
            # Check the roms_tools_version
            roms_tools_version_header = header_data.get("roms_tools_version")
            # Get current version of roms-tools
            try:
                roms_tools_version_current = importlib.metadata.version("roms-tools")
            except importlib.metadata.PackageNotFoundError:
                roms_tools_version_current = "unknown"

            if roms_tools_version_header != roms_tools_version_current:
                warnings.warn(
                    f"Current roms-tools version ({roms_tools_version_current}) does not match the version in the YAML header ({roms_tools_version_header}).",
                    UserWarning,
                )

        if grid_data is None:
            raise ValueError("No Grid configuration found in the YAML file.")

        return cls(**grid_data)

    # override __repr__ method to only print attributes that are actually set
    def __repr__(self) -> str:
        cls = self.__class__
        cls_name = cls.__name__
        # Create a dictionary of attribute names and values, filtering out those that are not set and 'ds'
        attr_dict = {
            k: v for k, v in self.__dict__.items() if k != "ds" and v is not None
        }
        attr_str = ", ".join(f"{k}={v!r}" for k, v in attr_dict.items())
        return f"{cls_name}({attr_str})"


def _make_grid_ds(
    nx: int,
    ny: int,
    size_x: float,
    size_y: float,
    center_lon: float,
    center_lat: float,
    rot: float,
) -> xr.Dataset:
    _raise_if_domain_size_too_large(size_x, size_y)

    initial_lon_lat_vars = _make_initial_lon_lat_ds(size_x, size_y, nx, ny)

    # rotate coordinate system
    rotated_lon_lat_vars = _rotate(*initial_lon_lat_vars, rot)
    lon, *_ = rotated_lon_lat_vars

    # translate coordinate system
    translated_lon_lat_vars = _translate(*rotated_lon_lat_vars, center_lat, center_lon)
    lon, lat, lonu, latu, lonv, latv, lonq, latq = translated_lon_lat_vars

    # compute 1/dx and 1/dy
    pm, pn = _compute_coordinate_metrics(lon, lonu, latu, lonv, latv)

    # compute angle of local grid positive x-axis relative to east
    ang = _compute_angle(lon, lonu, latu, lonq)

    ds = _create_grid_ds(lon, lat, pm, pn, ang, rot, center_lon, center_lat)

    ds = _add_global_metadata(ds, size_x, size_y, center_lon, center_lat, rot)

    return ds


def _raise_if_domain_size_too_large(size_x, size_y):
    threshold = 20000
    if size_x > threshold or size_y > threshold:
        raise ValueError("Domain size has to be smaller than %g km" % threshold)


def _make_initial_lon_lat_ds(size_x, size_y, nx, ny):
    # Mercator projection around the equator

    # initially define the domain to be longer in x-direction (dimension "length")
    # than in y-direction (dimension "width") to keep grid distortion minimal
    if size_y > size_x:
        domain_length, domain_width = size_y * 1e3, size_x * 1e3  # in m
        nl, nw = ny, nx
    else:
        domain_length, domain_width = size_x * 1e3, size_y * 1e3  # in m
        nl, nw = nx, ny

    domain_length_in_degrees = domain_length / RADIUS_OF_EARTH
    domain_width_in_degrees = domain_width / RADIUS_OF_EARTH

    # 1d array describing the longitudes at cell centers
    x = np.arange(-0.5, nl + 1.5, 1)
    lon_array_1d_in_degrees = (
        domain_length_in_degrees * x / nl - domain_length_in_degrees / 2
    )
    # 1d array describing the longitudes at cell corners (or vorticity points "q")
    xq = np.arange(-1, nl + 2, 1)
    lonq_array_1d_in_degrees_q = (
        domain_length_in_degrees * xq / nl - domain_length_in_degrees / 2
    )

    # convert degrees latitude to y-coordinate using Mercator projection
    y1 = np.log(np.tan(np.pi / 4 - domain_width_in_degrees / 4))
    y2 = np.log(np.tan(np.pi / 4 + domain_width_in_degrees / 4))

    # linearly space points in y-space
    y = (y2 - y1) * np.arange(-0.5, nw + 1.5, 1) / nw + y1
    yq = (y2 - y1) * np.arange(-1, nw + 2) / nw + y1

    # inverse Mercator projections
    lat_array_1d_in_degrees = np.arctan(np.sinh(y))
    latq_array_1d_in_degrees = np.arctan(np.sinh(yq))

    # 2d grid at cell centers
    lon, lat = np.meshgrid(lon_array_1d_in_degrees, lat_array_1d_in_degrees)
    # 2d grid at cell corners
    lonq, latq = np.meshgrid(lonq_array_1d_in_degrees_q, latq_array_1d_in_degrees)

    if size_y > size_x:
        # Rotate grid by 90 degrees because until here the grid has been defined
        # to be longer in x-direction than in y-direction

        lon, lat = _rot_sphere(lon, lat, 90)
        lonq, latq = _rot_sphere(lonq, latq, 90)

        lon = np.transpose(np.flip(lon, 0))
        lat = np.transpose(np.flip(lat, 0))
        lonq = np.transpose(np.flip(lonq, 0))
        latq = np.transpose(np.flip(latq, 0))

    # infer longitudes and latitudes at u- and v-points
    lonu = 0.5 * (lon[:, :-1] + lon[:, 1:])
    latu = 0.5 * (lat[:, :-1] + lat[:, 1:])
    lonv = 0.5 * (lon[:-1, :] + lon[1:, :])
    latv = 0.5 * (lat[:-1, :] + lat[1:, :])

    # TODO wrap up into temporary container Dataset object?
    return lon, lat, lonu, latu, lonv, latv, lonq, latq


def _rotate(lon, lat, lonu, latu, lonv, latv, lonq, latq, rot):
    """Rotate grid counterclockwise relative to surface of Earth by rot degrees"""

    (lon, lat) = _rot_sphere(lon, lat, rot)
    (lonu, latu) = _rot_sphere(lonu, latu, rot)
    (lonv, latv) = _rot_sphere(lonv, latv, rot)
    (lonq, latq) = _rot_sphere(lonq, latq, rot)

    return lon, lat, lonu, latu, lonv, latv, lonq, latq


def _translate(lon, lat, lonu, latu, lonv, latv, lonq, latq, tra_lat, tra_lon):
    """Translate grid so that the centre lies at the position (tra_lat, tra_lon)"""

    (lon, lat) = _tra_sphere(lon, lat, tra_lat)
    (lonu, latu) = _tra_sphere(lonu, latu, tra_lat)
    (lonv, latv) = _tra_sphere(lonv, latv, tra_lat)
    (lonq, latq) = _tra_sphere(lonq, latq, tra_lat)

    lon = lon + tra_lon * np.pi / 180
    lonu = lonu + tra_lon * np.pi / 180
    lonv = lonv + tra_lon * np.pi / 180
    lonq = lonq + tra_lon * np.pi / 180

    lon[lon < -np.pi] = lon[lon < -np.pi] + 2 * np.pi
    lonu[lonu < -np.pi] = lonu[lonu < -np.pi] + 2 * np.pi
    lonv[lonv < -np.pi] = lonv[lonv < -np.pi] + 2 * np.pi
    lonq[lonq < -np.pi] = lonq[lonq < -np.pi] + 2 * np.pi

    return lon, lat, lonu, latu, lonv, latv, lonq, latq


def _rot_sphere(lon, lat, rot):
    (n, m) = np.shape(lon)
    # convert rotation angle from degrees to radians
    rot = rot * np.pi / 180

    # translate into Cartesian coordinates x,y,z
    # conventions:  (lon,lat) = (0,0)  corresponds to (x,y,z) = ( 0,-r, 0)
    #               (lon,lat) = (0,90) corresponds to (x,y,z) = ( 0, 0, r)
    x1 = np.sin(lon) * np.cos(lat)
    y1 = np.cos(lon) * np.cos(lat)
    z1 = np.sin(lat)

    # We will rotate these points around the small circle defined by
    # the intersection of the sphere and the plane that
    # is orthogonal to the line through (lon,lat) (0,0) and (180,0)

    # The rotation is in that plane around its intersection with
    # aforementioned line.

    # Since the plane is orthogonal to the y-axis (in my definition at least),
    # Rotations in the plane of the small circle maintain constant y and are around
    # (x,y,z) = (0,y1,0)

    rp1 = np.sqrt(x1**2 + z1**2)

    ap1 = np.pi / 2 * np.ones((n, m))
    ap1[np.abs(x1) > 1e-7] = np.arctan(
        np.abs(z1[np.abs(x1) > 1e-7] / x1[np.abs(x1) > 1e-7])
    )
    ap1[x1 < 0] = np.pi - ap1[x1 < 0]
    ap1[z1 < 0] = -ap1[z1 < 0]

    ap2 = ap1 + rot
    x2 = rp1 * np.cos(ap2)
    y2 = y1
    z2 = rp1 * np.sin(ap2)

    lon = np.pi / 2 * np.ones((n, m))
    lon[abs(y2) > 1e-7] = np.arctan(
        np.abs(x2[np.abs(y2) > 1e-7] / y2[np.abs(y2) > 1e-7])
    )
    lon[y2 < 0] = np.pi - lon[y2 < 0]
    lon[x2 < 0] = -lon[x2 < 0]

    pr2 = np.sqrt(x2**2 + y2**2)
    lat = np.pi / 2 * np.ones((n, m))
    lat[np.abs(pr2) > 1e-7] = np.arctan(
        np.abs(z2[np.abs(pr2) > 1e-7] / pr2[np.abs(pr2) > 1e-7])
    )
    lat[z2 < 0] = -lat[z2 < 0]

    return (lon, lat)


def _tra_sphere(lon, lat, tra):
    (n, m) = np.shape(lon)
    tra = tra * np.pi / 180  # translation in latitude direction

    # translate into x,y,z
    # conventions:  (lon,lat) = (0,0)  corresponds to (x,y,z) = ( 0,-r, 0)
    #               (lon,lat) = (0,90) corresponds to (x,y,z) = ( 0, 0, r)
    x1 = np.sin(lon) * np.cos(lat)
    y1 = np.cos(lon) * np.cos(lat)
    z1 = np.sin(lat)

    # We will rotate these points around the small circle defined by
    # the intersection of the sphere and the plane that
    # is orthogonal to the line through (lon,lat) (90,0) and (-90,0)

    # The rotation is in that plane around its intersection with
    # aforementioned line.

    # Since the plane is orthogonal to the x-axis (in my definition at least),
    # Rotations in the plane of the small circle maintain constant x and are around
    # (x,y,z) = (x1,0,0)

    rp1 = np.sqrt(y1**2 + z1**2)

    ap1 = np.pi / 2 * np.ones((n, m))
    ap1[np.abs(y1) > 1e-7] = np.arctan(
        np.abs(z1[np.abs(y1) > 1e-7] / y1[np.abs(y1) > 1e-7])
    )
    ap1[y1 < 0] = np.pi - ap1[y1 < 0]
    ap1[z1 < 0] = -ap1[z1 < 0]

    ap2 = ap1 + tra
    x2 = x1
    y2 = rp1 * np.cos(ap2)
    z2 = rp1 * np.sin(ap2)

    ## transformation from (x,y,z) to (lat,lon)
    lon = np.pi / 2 * np.ones((n, m))
    lon[np.abs(y2) > 1e-7] = np.arctan(
        np.abs(x2[np.abs(y2) > 1e-7] / y2[np.abs(y2) > 1e-7])
    )
    lon[y2 < 0] = np.pi - lon[y2 < 0]
    lon[x2 < 0] = -lon[x2 < 0]

    pr2 = np.sqrt(x2**2 + y2**2)
    lat = np.pi / (2 * np.ones((n, m)))
    lat[np.abs(pr2) > 1e-7] = np.arctan(
        np.abs(z2[np.abs(pr2) > 1e-7] / pr2[np.abs(pr2) > 1e-7])
    )
    lat[z2 < 0] = -lat[z2 < 0]

    return (lon, lat)


def _compute_coordinate_metrics(lon, lonu, latu, lonv, latv):
    """Compute the curvilinear coordinate metrics pn and pm, defined as 1/grid spacing"""

    # pm = 1/dx
    pmu = gc_dist(lonu[:, :-1], latu[:, :-1], lonu[:, 1:], latu[:, 1:])
    pm = 0 * lon
    pm[:, 1:-1] = pmu
    pm[:, 0] = pm[:, 1]
    pm[:, -1] = pm[:, -2]
    pm = 1 / pm

    # pn = 1/dy
    pnv = gc_dist(lonv[:-1, :], latv[:-1, :], lonv[1:, :], latv[1:, :])
    pn = 0 * lon
    pn[1:-1, :] = pnv
    pn[0, :] = pn[1, :]
    pn[-1, :] = pn[-2, :]
    pn = 1 / pn

    return pn, pm


def gc_dist(lon1, lat1, lon2, lat2):
    # Distance between 2 points along a great circle
    # lat and lon in radians!!
    # 2008, Jeroen Molemaker, UCLA

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    dang = 2 * np.arcsin(
        np.sqrt(
            np.sin(dlat / 2) ** 2 + np.cos(lat2) * np.cos(lat1) * np.sin(dlon / 2) ** 2
        )
    )  # haversine function

    dis = RADIUS_OF_EARTH * dang

    return dis


def _compute_angle(lon, lonu, latu, lonq):
    """Compute angles of local grid positive x-axis relative to east"""

    dellat = latu[:, 1:] - latu[:, :-1]
    dellon = lonu[:, 1:] - lonu[:, :-1]
    dellon[dellon > np.pi] = dellon[dellon > np.pi] - 2 * np.pi
    dellon[dellon < -np.pi] = dellon[dellon < -np.pi] + 2 * np.pi
    dellon = dellon * np.cos(0.5 * (latu[:, 1:] + latu[:, :-1]))

    ang = copy.copy(lon)
    ang_s = np.arctan(dellat / (dellon + 1e-16))
    ang_s[(dellon < 0) & (dellat < 0)] = ang_s[(dellon < 0) & (dellat < 0)] - np.pi
    ang_s[(dellon < 0) & (dellat >= 0)] = ang_s[(dellon < 0) & (dellat >= 0)] + np.pi
    ang_s[ang_s > np.pi] = ang_s[ang_s > np.pi] - np.pi
    ang_s[ang_s < -np.pi] = ang_s[ang_s < -np.pi] + np.pi

    ang[:, 1:-1] = ang_s
    ang[:, 0] = ang[:, 1]
    ang[:, -1] = ang[:, -2]

    lon[lon < 0] = lon[lon < 0] + 2 * np.pi
    lonq[lonq < 0] = lonq[lonq < 0] + 2 * np.pi

    return ang


def _create_grid_ds(
    lon,
    lat,
    pm,
    pn,
    angle,
    rot,
    center_lon,
    center_lat,
):
    ds = xr.Dataset()

    lon_rho = xr.Variable(
        data=lon * 180 / np.pi,
        dims=["eta_rho", "xi_rho"],
        attrs={"long_name": "longitude of rho-points", "units": "degrees East"},
    )
    lat_rho = xr.Variable(
        data=lat * 180 / np.pi,
        dims=["eta_rho", "xi_rho"],
        attrs={"long_name": "latitude of rho-points", "units": "degrees North"},
    )

    ds = ds.assign_coords(
        {"lat_rho": lat_rho, "lon_rho": lon_rho}
    )

    ds["angle"] = xr.Variable(
        data=angle,
        dims=["eta_rho", "xi_rho"],
        attrs={"long_name": "Angle between xi axis and east", "units": "radians"},
    )

    # Coriolis frequency
    f0 = 4 * np.pi * np.sin(lat) / (24 * 3600)

    ds["f"] = xr.Variable(
        data=f0,
        dims=["eta_rho", "xi_rho"],
        attrs={"long_name": "Coriolis parameter at rho-points", "units": "second-1"},
    )

    ds["pm"] = xr.Variable(
        data=pm,
        dims=["eta_rho", "xi_rho"],
        attrs={
            "long_name": "Curvilinear coordinate metric in xi-direction",
            "units": "meter-1",
        },
    )
    ds["pn"] = xr.Variable(
        data=pn,
        dims=["eta_rho", "xi_rho"],
        attrs={
            "long_name": "Curvilinear coordinate metric in eta-direction",
            "units": "meter-1",
        },
    )

    return ds


def _add_global_metadata(ds, size_x, size_y, center_lon, center_lat, rot):

    ds["spherical"] = xr.DataArray(np.array("T", dtype="S1"))
    ds["spherical"].attrs["Long_name"] = "Grid type logical switch"
    ds["spherical"].attrs["option_T"] = "spherical"

    ds.attrs["title"] = "ROMS grid created by ROMS-Tools"

    # Include the version of roms-tools
    try:
        roms_tools_version = importlib.metadata.version("roms-tools")
    except importlib.metadata.PackageNotFoundError:
        roms_tools_version = "unknown"

    ds.attrs["roms_tools_version"] = roms_tools_version
    ds.attrs["size_x"] = size_x
    ds.attrs["size_y"] = size_y
    ds.attrs["center_lon"] = center_lon
    ds.attrs["center_lat"] = center_lat
    ds.attrs["rot"] = rot

    return ds


def _f2c(f):
    """
    Coarsen input xarray DataArray f in both x- and y-direction.

    Parameters
    ----------
    f : xarray.DataArray
        Input DataArray with dimensions (nxp, nyp).

    Returns
    -------
    fc : xarray.DataArray
        Output DataArray with modified dimensions and values.
    """

    fc = _f2c_xdir(f)
    fc = fc.transpose()
    fc = _f2c_xdir(fc)
    fc = fc.transpose()
    fc = fc.rename({"eta_rho": "eta_coarse", "xi_rho": "xi_coarse"})

    return fc


def _f2c_xdir(f):
    """
    Coarsen input xarray DataArray f in x-direction.

    Parameters
    ----------
    f : xarray.DataArray
        Input DataArray with dimensions (nxp, nyp).

    Returns
    -------
    fc : xarray.DataArray
        Output DataArray with modified dimensions and values.
    """
    nxp, nyp = f.shape
    nxcp = (nxp - 2) // 2 + 2

    fc = xr.DataArray(np.zeros((nxcp, nyp)), dims=f.dims)

    # Calculate the interior values
    fc[1:-1, :] = 0.5 * (f[1:-2:2, :] + f[2:-1:2, :])

    # Calculate the first row
    fc[0, :] = f[0, :] + 0.5 * (f[0, :] - f[1, :])

    # Calculate the last row
    fc[-1, :] = f[-1, :] + 0.5 * (f[-1, :] - f[-2, :])

    return fc


def _add_lat_lon_at_velocity_points(ds, straddle):

    if straddle:
        # avoid jump from 360 to 0 in interpolation
        lon_rho = xr.where(ds["lon_rho"] > 180, ds["lon_rho"] - 360, ds["lon_rho"])
    else:
        lon_rho = ds["lon_rho"]
    lat_rho = ds["lat_rho"]

    lat_u = interpolate_from_rho_to_u(lat_rho)
    lon_u = interpolate_from_rho_to_u(lon_rho)
    lat_v = interpolate_from_rho_to_v(lat_rho)
    lon_v = interpolate_from_rho_to_v(lon_rho)

    if straddle:
        # convert back to range [0, 360]
        lon_u = xr.where(lon_u < 0, lon_u + 360, lon_u)
        lon_v = xr.where(lon_v < 0, lon_v + 360, lon_v)

    lat_u.attrs = {"long_name": "latitude of u-points", "units": "degrees North"}
    lon_u.attrs = {"long_name": "longitude of u-points", "units": "degrees East"}
    lat_v.attrs = {"long_name": "latitude of v-points", "units": "degrees North"}
    lon_v.attrs = {"long_name": "longitude of v-points", "units": "degrees East"}

    ds = ds.assign_coords(
        {
            "lat_u": lat_u,
            "lon_u": lon_u,
            "lat_v": lat_v,
            "lon_v": lon_v,
        }
    )

    return ds
