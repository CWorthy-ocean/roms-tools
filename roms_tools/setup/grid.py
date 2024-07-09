import copy
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from typing import Any

from roms_tools.setup.topography import _add_topography_and_mask
from roms_tools.setup.plot import _plot

RADIUS_OF_EARTH = 6371315.0  # in m


# TODO should we store an xgcm.Grid object instead of an xarray Dataset? Or even subclass xgcm.Grid?


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
    topography_source : str, optional
        Specifies the data source to use for the topography. Options are
        "etopo5". The default is "etopo5".
    smooth_factor : float, optional
        The smoothing factor used in the domain-wide Gaussian smoothing of the
        topography. Smaller values result in less smoothing, while larger
        values produce more smoothing. The default is 8.
    hmin : float, optional
        The minimum ocean depth (in meters). The default is 5.
    rmax : float, optional
        The maximum slope parameter (in meters). This parameter controls
        the local smoothing of the topography. Smaller values result in
        smoother topography, while larger values preserve more detail.
        The default is 0.2.

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
     topography_source : str
         Data source used for the topography.
     smooth_factor : int
         Smoothing factor used in the domain-wide Gaussian smoothing of the topography.
     hmin : float
         Minimum ocean depth (in meters).
     rmax : float
         Maximum slope parameter (in meters).
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
    topography_source: str = "etopo5"
    smooth_factor: int = 8
    hmin: float = 5.0
    rmax: float = 0.2
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
            smooth_factor=self.smooth_factor,
            hmin=self.hmin,
            rmax=self.rmax,
        )

        # Check if the Greenwich meridian goes through the domain.
        self._straddle()

    def add_topography_and_mask(
        self, topography_source="etopo5", smooth_factor=8, hmin=5.0, rmax=0.2
    ) -> None:
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
            "etopo5". The default is "etopo5".
        smooth_factor : float, optional
            The smoothing factor used in the domain-wide Gaussian smoothing of the
            topography. Smaller values result in less smoothing, while larger
            values produce more smoothing. The default is 8.
        hmin : float, optional
            The minimum ocean depth (in meters). The default is 5.
        rmax : float, optional
            The maximum slope parameter (in meters). This parameter controls
            the local smoothing of the topography. Smaller values result in
            smoother topography, while larger values preserve more detail.
            The default is 0.2.

        Returns
        -------
        None
            This method modifies the dataset in place and does not return a value.
        """

        ds = _add_topography_and_mask(
            self.ds, topography_source, smooth_factor, hmin, rmax
        )
        # Assign the updated dataset back to the frozen dataclass
        object.__setattr__(self, "ds", ds)

    def compute_bathymetry_laplacian(self):
        """
        Compute the Laplacian of the 'h' field in the provided grid dataset.

        Adds:
        xarray.DataArray: The Laplacian of the 'h' field as a new variable in the dataset self.ds.
        """

        # Extract the 'h' field and grid spacing variables
        h = self.ds.h
        pm = self.ds.pm  # Reciprocal of grid spacing in x-direction
        pn = self.ds.pn  # Reciprocal of grid spacing in y-direction

        # Compute second derivatives using finite differences
        d2h_dx2 = (h.shift(xi_rho=-1) - 2 * h + h.shift(xi_rho=1)) * pm**2
        d2h_dy2 = (h.shift(eta_rho=-1) - 2 * h + h.shift(eta_rho=1)) * pn**2

        # Compute the Laplacian by summing second derivatives
        laplacian_h = d2h_dx2 + d2h_dy2

        # Add the Laplacian as a new variable in the dataset
        self.ds["h_laplacian"] = laplacian_h
        self.ds["h_laplacian"].attrs["long_name"] = "Laplacian of final bathymetry"
        self.ds["h_laplacian"].attrs["units"] = "1/m"

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

        # Create a new Grid instance without calling __init__ and __post_init__
        grid = cls.__new__(cls)

        # Set the dataset for the grid instance
        object.__setattr__(grid, "ds", ds)

        # Check if the Greenwich meridian goes through the domain.
        grid._straddle()

        # Manually set the remaining attributes by extracting parameters from dataset
        object.__setattr__(grid, "nx", ds.sizes["xi_rho"] - 2)
        object.__setattr__(grid, "ny", ds.sizes["eta_rho"] - 2)
        object.__setattr__(grid, "center_lon", ds["tra_lon"].values.item())
        object.__setattr__(grid, "center_lat", ds["tra_lat"].values.item())
        object.__setattr__(grid, "rot", ds["rotate"].values.item())

        for attr in [
            "size_x",
            "size_y",
            "topography_source",
            "smooth_factor",
            "hmin",
            "rmax",
        ]:
            if attr in ds.attrs:
                object.__setattr__(grid, attr, ds.attrs[attr])

        return grid

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

    def to_xgcm() -> Any:
        # TODO we could convert the dataset to an xgcm.Grid object and return here?
        raise NotImplementedError()

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

    def coarsen(self):
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
            "lon_rho": "lon_coarse",
            "lat_rho": "lat_coarse",
            "h": "h_coarse",
            "angle": "angle_coarse",
            "mask_rho": "mask_coarse",
        }

        for fine_var, coarse_var in d.items():
            fine_field = self.ds[fine_var]
            if self.straddle and fine_var == "lon_rho":
                fine_field = xr.where(fine_field > 180, fine_field - 360, fine_field)
            
            coarse_field = _f2c(fine_field)
            if fine_var == "lon_rho":
                coarse_field = xr.where(coarse_field < 0, coarse_field + 360, coarse_field)

            self.ds[coarse_var] = coarse_field

        self.ds["mask_coarse"] = xr.where(self.ds["mask_coarse"] > 0.5, 1, 0)


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

    ds = _add_global_metadata(ds, size_x, size_y)

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
    # Create xarray.Dataset object with lat_rho and lon_rho as coordinates
    ds = xr.Dataset(
        coords={
            "lat_rho": (("eta_rho", "xi_rho"), lat * 180 / np.pi),
            "lon_rho": (("eta_rho", "xi_rho"), lon * 180 / np.pi),
        }
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

    ds["lon_rho"] = xr.Variable(
        data=lon * 180 / np.pi,
        dims=["eta_rho", "xi_rho"],
        attrs={"long_name": "longitude of rho-points", "units": "degrees East"},
    )

    ds["lat_rho"] = xr.Variable(
        data=lat * 180 / np.pi,
        dims=["eta_rho", "xi_rho"],
        attrs={"long_name": "latitude of rho-points", "units": "degrees North"},
    )

    ds["tra_lon"] = center_lon
    ds["tra_lon"].attrs["long_name"] = "Longitudinal translation of base grid"
    ds["tra_lon"].attrs["units"] = "degrees East"

    ds["tra_lat"] = center_lat
    ds["tra_lat"].attrs["long_name"] = "Latitudinal translation of base grid"
    ds["tra_lat"].attrs["units"] = "degrees North"

    ds["rotate"] = rot
    ds["rotate"].attrs["long_name"] = "Rotation of base grid"
    ds["rotate"].attrs["units"] = "degrees"

    return ds


def _add_global_metadata(ds, size_x, size_y):
    ds.attrs["Type"] = "ROMS grid produced by roms-tools"
    ds.attrs["size_x"] = size_x
    ds.attrs["size_y"] = size_y

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
