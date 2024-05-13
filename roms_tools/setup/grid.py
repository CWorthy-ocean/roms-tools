import copy
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import xarray as xr

from typing import Any

from roms_tools.setup.topography import _add_topography_and_mask


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
        Specifies the data source to use for the topography. Options are "etopo5.nc". The default is "etopo5.nc".
    smooth_factor: int
        The smoothing factor used in the global Gaussian smoothing of the topography. The default is 8.
    hmin: float
        The minimum ocean depth (in meters). The default is 5.
    rmax: float
        The maximum slope parameter (in meters). The default is 0.2.


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
    topography_source: str = 'etopo5.nc'
    smooth_factor: int = 8
    hmin: float = 5.0
    rmax: float = 0.2
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        ds = _make_grid_ds(
            nx=self.nx,
            ny=self.ny,
            size_x=self.size_x,
            size_y=self.size_y,
            center_lon=self.center_lon,
            center_lat=self.center_lat,
            rot=self.rot,
            topography_source=self.topography_source,
            smooth_factor=self.smooth_factor,
            hmin=self.hmin,
            rmax=self.rmax
        )
        # Calling object.__setattr__ is ugly but apparently this really is the best (current) way to combine __post_init__ with a frozen dataclass
        # see https://stackoverflow.com/questions/53756788/how-to-set-the-value-of-dataclass-field-in-post-init-when-frozen-true
        object.__setattr__(self, "ds", ds)

    def save(self, filepath: str) -> None:
        """
        Save the grid information to a netCDF4 file.

        Parameters
        ----------
        filepath
        """
        self.ds.to_netcdf(filepath)

    def from_file(self, filepath: str) -> "Grid":
        """
        Open an existing grid from a file.

        Parameters
        ----------
        filepath
        """
        # TODO set other parameters that were saved into the file, because every parameter we need gets saved.
        # TODO actually we will need to deduce size_x and size_y from the file, that's annoying.
        self.ds = xr.open_dataset(filepath)
        raise NotImplementedError()

    def to_xgcm() -> Any:
        # TODO we could convert the dataset to an xgcm.Grid object and return here?
        raise NotImplementedError()

    def plot(self, bathymetry: bool = False) -> None:
        """
        Plot the grid.

        Requires cartopy and matplotlib.

        Parameters
        ----------
        bathymetry: bool
            Whether or not to plot the bathymetry. Default is False.
        """


        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        lon_deg = (self.ds["lon_rho"] - 360).values
        lat_deg = self.ds["lat_rho"].values

        # Define projections
        proj = ccrs.PlateCarree()
        trans = ccrs.NearsidePerspective(
            central_longitude=np.mean(lon_deg), central_latitude=np.mean(lat_deg)
        )

        # find corners
        (lo1, la1) = (lon_deg[0, 0], lat_deg[0, 0])
        (lo2, la2) = (lon_deg[0, -1], lat_deg[0, -1])
        (lo3, la3) = (lon_deg[-1, -1], lat_deg[-1, -1])
        (lo4, la4) = (lon_deg[-1, 0], lat_deg[-1, 0])

        # transform coordinates to projected space
        lo1t, la1t = trans.transform_point(lo1, la1, proj)
        lo2t, la2t = trans.transform_point(lo2, la2, proj)
        lo3t, la3t = trans.transform_point(lo3, la3, proj)
        lo4t, la4t = trans.transform_point(lo4, la4, proj)

        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=trans)

        ax.plot(
            [lo1t, lo2t, lo3t, lo4t, lo1t],
            [la1t, la2t, la3t, la4t, la1t],
            "ro-",
        )

        ax.coastlines(
            resolution="50m", linewidth=0.5, color="black"
        )  # add map of coastlines
        ax.gridlines()
        if bathymetry:
            p = ax.contourf(
                    lon_deg, lat_deg,
                    self.ds.hraw.where(self.ds.mask_rho_filled),
                    transform=proj,
                    levels=15,
                    cmap="YlGnBu"
            )
            plt.colorbar(p, label="Bathymetry [m]")
        plt.show()


def _make_grid_ds(
    nx: int,
    ny: int,
    size_x: float,
    size_y: float,
    center_lon: float,
    center_lat: float,
    rot: float,
    topography_source: str,
    smooth_factor: int,
    hmin: float,
    rmax: float
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

    ds = _create_grid_ds(
        lon,
        lat,
        pm,
        pn,
        ang,
        rot,
        center_lon,
        center_lat
    )

    ds = _add_topography_and_mask(ds, topography_source, smooth_factor, hmin, rmax)

    ds = _add_global_metadata(ds, nx, ny, size_x, size_y, center_lon, center_lat, rot, topography_source, smooth_factor, hmin, rmax)

    return ds

def _raise_if_domain_size_too_large(size_x, size_y):

    threshold = 20000
    if size_x > threshold or size_y > threshold:
        raise ValueError("Domain size has to be smaller than %g km" %threshold)


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

        lon, lat = rot_sphere(lon, lat, 90)
        lonq, latq = rot_sphere(lonq, latq, 90)

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

    (lon, lat) = rot_sphere(lon, lat, rot)
    (lonu, latu) = rot_sphere(lonu, latu, rot)
    (lonv, latv) = rot_sphere(lonv, latv, rot)
    (lonq, latq) = rot_sphere(lonq, latq, rot)

    return lon, lat, lonu, latu, lonv, latv, lonq, latq


def _translate(lon, lat, lonu, latu, lonv, latv, lonq, latq, tra_lat, tra_lon):
    """Translate grid so that the centre lies at the position (tra_lat, tra_lon)"""

    (lon, lat) = tra_sphere(lon, lat, tra_lat)
    (lonu, latu) = tra_sphere(lonu, latu, tra_lat)
    (lonv, latv) = tra_sphere(lonv, latv, tra_lat)
    (lonq, latq) = tra_sphere(lonq, latq, tra_lat)

    lon = lon + tra_lon * np.pi / 180
    lonu = lonu + tra_lon * np.pi / 180
    lonv = lonv + tra_lon * np.pi / 180
    lonq = lonq + tra_lon * np.pi / 180

    lon[lon < -np.pi] = lon[lon < -np.pi] + 2 * np.pi
    lonu[lonu < -np.pi] = lonu[lonu < -np.pi] + 2 * np.pi
    lonv[lonv < -np.pi] = lonv[lonv < -np.pi] + 2 * np.pi
    lonq[lonq < -np.pi] = lonq[lonq < -np.pi] + 2 * np.pi

    return lon, lat, lonu, latu, lonv, latv, lonq, latq


def rot_sphere(lon, lat, rot):

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


def tra_sphere(lon, lat, tra):

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
            "lon_rho": (("eta_rho", "xi_rho"), lon * 180 / np.pi)
        }
    )

    ds["angle"] = xr.Variable(
        data=angle,
        dims=["eta_rho", "xi_rho"],
        attrs={"long_name": "Angle between xi axis and east", "units": "radians"},
    )

    # Coriolis frequency
    f0 = 4 * np.pi * np.sin(lat) / (24 * 3600)

    ds["f0"] = xr.Variable(
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

    # TODO this is never written to
    # ds['xy_flip']

    # TODO same here?
    # ds["spherical"] = xr.Variable(
    #    data=["T"],
    #    attrs={
    #        "long_name": "Grid type logical switch",
    #        "option_T": "spherical",
    #    },
    #)

    return ds


def _add_global_metadata(ds, nx, ny, size_x, size_y, center_lon, center_lat, rot, topography_source, smooth_factor, hmin, rmax):

    ds.attrs["Title"] = (
        "ROMS grid. Settings:"
        f" nx: {nx} ny: {ny}"
        f" xsize: {size_x / 1e3} ysize: {size_y / 1e3}"
        f" rotate: {rot} Lon: {center_lon} Lat: {center_lat}"
    )
    ds.attrs["Date"] = date.today()
    ds.attrs["Type"] = "ROMS grid produced by roms-tools"
    ds.attrs["Topography source"] = "https://github.com/CWorthy-ocean/roms-tools-data/raw/main/" + topography_source
    ds.attrs["Topography modifications"] = "Global smoothing with factor %i; Minimal depth: %gm; Local smoothing to satisfy r < rmax = %gm" %(smooth_factor, hmin, rmax)

    return ds
