import copy
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import xarray as xr

from typing import Any


RADIUS_OF_EARTH = 6371315.0  # in m


# TODO should we store an xgcm.Grid object instead of an xarray Dataset? Or even subclass xgcm.Grid?


@dataclass(frozen=True, kw_only=True)
class Grid:
    """
    A single ROMS grid.

    Used for creating, plotting, and then saving a new ROMS domain grid.

    Parameters
    ----------
    nx
        Number of grid points in the x-direction
    ny
        Number of grid points in the y-direction
    size_x
        Domain size in the x-direction (in km?)
    size_y
        Domain size in the y-direction (in km?)
    center_lon
        Longitude of grid center
    center_lat
        Latitude of grid center
    rot
        Rotation of grid x-direction from lines of constant latitude.
        Measured in degrees, with positive values meaning a counterclockwise rotation.
        The default is 0, which means that the x-direction of the grid x-direction is aligned with lines of constant latitude.

    Raises
    ------
    ValueError
        If you try to create a grid which crosses the Greenwich Meridian
    """

    nx: int
    ny: int
    size_x: float
    size_y: float
    center_lon: float
    center_lat: float
    rot: float = 0
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

    def equals(self, other: "Grid") -> bool:
        """
        Assert that the parameters of this grid are the same as the parameters of another grid.

        Parameters
        ----------
        other
            Another Grid object to compare to

        Returns
        -------
        equals
            boolean indicating whether or not the two grids have the same parameters
        """
        # TODO can the dataclass handle this for us?
        ...

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

        # TODO optionally plot topography on top?
        if bathymetry:
            raise NotImplementedError()

        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        lon_deg = (self.ds["lon_rho"] - 360).values
        lat_deg = self.ds["lat_rho"].values

        # Define projections
        geodetic = ccrs.Geodetic()
        trans = ccrs.NearsidePerspective(
            central_longitude=np.mean(lon_deg), central_latitude=np.mean(lat_deg)
        )

        # find corners
        (lo1, la1) = (lon_deg[0, 0], lat_deg[0, 0])
        (lo2, la2) = (lon_deg[0, -1], lat_deg[0, -1])
        (lo3, la3) = (lon_deg[-1, -1], lat_deg[-1, -1])
        (lo4, la4) = (lon_deg[-1, 0], lat_deg[-1, 0])

        # transform coordinates to projected space
        lo1t, la1t = trans.transform_point(lo1, la1, geodetic)
        lo2t, la2t = trans.transform_point(lo2, la2, geodetic)
        lo3t, la3t = trans.transform_point(lo3, la3, geodetic)
        lo4t, la4t = trans.transform_point(lo4, la4, geodetic)

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

        plt.show()


def _make_grid_ds(
    nx: int,
    ny: int,
    size_x: float,
    size_y: float,
    center_lon: float,
    center_lat: float,
    rot: float,
) -> xr.Dataset:

    if size_y > size_x:
        domain_length, domain_width = size_x * 1e3, size_y * 1e3  # in m
        nl, nw = nx, ny
    else:
        domain_length, domain_width = size_y * 1e3, size_x * 1e3  # in m
        nl, nw = ny, nx

    initial_lon_lat_vars = _make_initial_lon_lat_ds(domain_length, domain_width, nl, nw)

    rotated_lon_lat_vars = _rotate(*initial_lon_lat_vars, rot)

    lon2, *_ = rotated_lon_lat_vars

    _raise_if_crosses_greenwich_meridian(lon2, center_lon)

    translated_lon_lat_vars = _translate(*rotated_lon_lat_vars, center_lat, center_lon)

    lon4, lat4, lonu, latu, lonv, latv, lone, late = translated_lon_lat_vars

    pm, pn = _compute_coordinate_metrics(lon4, lonu, latu, lonv, latv)

    ang = _compute_angle(lon4, lonu, latu, lone)

    ds = _create_grid_ds(
        nx,
        ny,
        lon4,
        lat4,
        pm,
        pn,
        ang,
        size_x,
        size_y,
        rot,
        center_lon,
        center_lat,
        lone,
        late,
    )

    # TODO topography
    # ds = _make_topography(ds)

    ds = _add_global_metadata(ds, nx, ny, size_x, size_y, center_lon, center_lat, rot)

    return ds


def _raise_if_crosses_greenwich_meridian(lon, center_lon):
    # We have to do this before the grid is translated because we don't trust the grid creation routines in that case.

    # TODO it would be nice to handle this case, but we first need to know what ROMS expects / can handle.

    # TODO what about grids which cross the international dateline?

    if np.min(lon + center_lon) < 0 < np.max(lon + center_lon):
        raise ValueError("Grid cannot cross Greenwich Meridian")


def _make_initial_lon_lat_ds(domain_length, domain_width, nl, nw):

    domain_length_in_degrees_longitude = domain_length / RADIUS_OF_EARTH
    domain_width_in_degrees_latitude = domain_width / RADIUS_OF_EARTH

    longitude_array_1d_in_degrees = (
        domain_length_in_degrees_longitude * np.arange(-0.5, nl + 1.5, 1) / nl
        - domain_length_in_degrees_longitude / 2
    )

    # TODO I don't fully understand what this piece of code achieves
    mul = 1.0
    for it in range(1, 101):

        # convert degrees latitude to y-coordinate using Mercator projection
        y1 = np.log(np.tan(np.pi / 4 - domain_width_in_degrees_latitude / 4))
        y2 = np.log(np.tan(np.pi / 4 + domain_width_in_degrees_latitude / 4))

        # linearly space points in y-space
        y = (y2 - y1) * np.arange(-0.5, nw + 1.5, 1) / nw + y1

        # convert back to longitude using inverse Mercator projection
        # lat1d = 2*np.arctan(np.exp(y)) - np.pi/2
        latitude_array_1d_in_degrees = np.arctan(np.sinh(y))

        # find width and height of new grid at central grid point in degrees
        latitude_array_1d_in_degrees_cen = 0.5 * (
            latitude_array_1d_in_degrees[int(np.round(nw / 2) + 1)]
            - latitude_array_1d_in_degrees[int(np.round(nw / 2) - 1)]
        )
        longitude_array_1d_in_degrees_cen = domain_length_in_degrees_longitude / nl

        # scale the domain width in degreees latitude somehow?
        mul = (
            latitude_array_1d_in_degrees_cen
            / longitude_array_1d_in_degrees_cen
            * domain_length_in_degrees_longitude
            / domain_width_in_degrees_latitude
            * nw
            / nl
        )
        latitude_array_1d_in_degrees = latitude_array_1d_in_degrees / mul

    # TODO what does the 'e' suffix mean?
    lon1de = (
        domain_length_in_degrees_longitude * np.arange(-1, nl + 2, 1) / nl
        - domain_length_in_degrees_longitude / 2
    )
    ye = (y2 - y1) * np.arange(-1, nw + 2) / nw + y1
    # lat1de = 2 * np.arctan(np.exp(ye)) - np.pi/2
    lat1de = np.arctan(np.sinh(ye))
    lat1de = lat1de / mul

    lon1, lat1 = np.meshgrid(
        longitude_array_1d_in_degrees, latitude_array_1d_in_degrees
    )
    lone, late = np.meshgrid(lon1de, lat1de)
    lonu = 0.5 * (lon1[:, :-1] + lon1[:, 1:])
    latu = 0.5 * (lat1[:, :-1] + lat1[:, 1:])
    lonv = 0.5 * (lon1[:-1, :] + lon1[1:, :])
    latv = 0.5 * (lat1[:-1, :] + lat1[1:, :])

    if domain_length > domain_width:
        # Rotate grid 90 degrees so that the width is now longer than the length

        lon1, lat1 = rot_sphere(lon1, lat1, 90)
        lonu, latu = rot_sphere(lonu, latu, 90)
        lonv, latv = rot_sphere(lonv, latv, 90)
        lone, late = rot_sphere(lone, late, 90)

        lon1 = np.transpose(np.flip(lon1, 0))
        lat1 = np.transpose(np.flip(lat1, 0))
        lone = np.transpose(np.flip(lone, 0))
        late = np.transpose(np.flip(late, 1))

        lonu_tmp = np.transpose(np.flip(lonv, 0))
        latu_tmp = np.transpose(np.flip(latv, 0))
        lonv = np.transpose(np.flip(lonu, 0))
        latv = np.transpose(np.flip(latu, 0))
        lonu = lonu_tmp
        latu = latu_tmp

    # TODO wrap up into temporary container Dataset object?
    return lon1, lat1, lonu, latu, lonv, latv, lone, late


def _rotate(lon1, lat1, lonu, latu, lonv, latv, lone, late, rot):
    """Rotate grid counterclockwise relative to surface of Earth by rot degrees"""

    (lon2, lat2) = rot_sphere(lon1, lat1, rot)
    (lonu, latu) = rot_sphere(lonu, latu, rot)
    (lonv, latv) = rot_sphere(lonv, latv, rot)
    (lone, late) = rot_sphere(lone, late, rot)

    return lon2, lat2, lonu, latu, lonv, latv, lone, late


def _translate(lon2, lat2, lonu, latu, lonv, latv, lone, late, tra_lat, tra_lon):
    """Translate grid so that the centre lies at the position (tra_lat, tra_lon)"""

    (lon3, lat3) = tra_sphere(lon2, lat2, tra_lat)
    (lonu, latu) = tra_sphere(lonu, latu, tra_lat)
    (lonv, latv) = tra_sphere(lonv, latv, tra_lat)
    (lone, late) = tra_sphere(lone, late, tra_lat)

    lon4 = lon3 + tra_lon * np.pi / 180
    lonu = lonu + tra_lon * np.pi / 180
    lonv = lonv + tra_lon * np.pi / 180
    lone = lone + tra_lon * np.pi / 180
    lon4[lon4 < -np.pi] = lon4[lon4 < -np.pi] + 2 * np.pi
    lonu[lonu < -np.pi] = lonu[lonu < -np.pi] + 2 * np.pi
    lonv[lonv < -np.pi] = lonv[lonv < -np.pi] + 2 * np.pi
    lone[lone < -np.pi] = lone[lone < -np.pi] + 2 * np.pi
    lat4 = lat3

    return lon4, lat4, lonu, latu, lonv, latv, lone, late


def rot_sphere(lon1, lat1, rot):

    (n, m) = np.shape(lon1)
    rot = rot * np.pi / 180

    # translate into x,y,z
    # conventions:  (lon,lat) = (0,0)  corresponds to (x,y,z) = ( 0,-r, 0)
    #               (lon,lat) = (0,90) corresponds to (x,y,z) = ( 0, 0, r)
    x1 = np.sin(lon1) * np.cos(lat1)
    y1 = np.cos(lon1) * np.cos(lat1)
    z1 = np.sin(lat1)

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

    lon2 = np.pi / 2 * np.ones((n, m))
    lon2[abs(y2) > 1e-7] = np.arctan(
        np.abs(x2[np.abs(y2) > 1e-7] / y2[np.abs(y2) > 1e-7])
    )
    lon2[y2 < 0] = np.pi - lon2[y2 < 0]
    lon2[x2 < 0] = -lon2[x2 < 0]

    pr2 = np.sqrt(x2**2 + y2**2)
    lat2 = np.pi / 2 * np.ones((n, m))
    lat2[np.abs(pr2) > 1e-7] = np.arctan(
        np.abs(z2[np.abs(pr2) > 1e-7] / pr2[np.abs(pr2) > 1e-7])
    )
    lat2[z2 < 0] = -lat2[z2 < 0]

    return (lon2, lat2)


def tra_sphere(lon1, lat1, tra):

    # Rotate sphere around its y-axis
    # Part of easy grid
    # (c) 2008, Jeroen Molemaker, UCLA

    (n, m) = np.shape(lon1)
    tra = tra * np.pi / 180  # translation in latitude direction

    # translate into x,y,z
    # conventions:  (lon,lat) = (0,0)  corresponds to (x,y,z) = ( 0,-r, 0)
    #               (lon,lat) = (0,90) corresponds to (x,y,z) = ( 0, 0, r)
    x1 = np.sin(lon1) * np.cos(lat1)
    y1 = np.cos(lon1) * np.cos(lat1)
    z1 = np.sin(lat1)

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
    lon2 = np.pi / 2 * np.ones((n, m))
    lon2[np.abs(y2) > 1e-7] = np.arctan(
        np.abs(x2[np.abs(y2) > 1e-7] / y2[np.abs(y2) > 1e-7])
    )
    lon2[y2 < 0] = np.pi - lon2[y2 < 0]
    lon2[x2 < 0] = -lon2[x2 < 0]

    pr2 = np.sqrt(x2**2 + y2**2)
    lat2 = np.pi / (2 * np.ones((n, m)))
    lat2[np.abs(pr2) > 1e-7] = np.arctan(
        np.abs(z2[np.abs(pr2) > 1e-7] / pr2[np.abs(pr2) > 1e-7])
    )
    lat2[z2 < 0] = -lat2[z2 < 0]

    return (lon2, lat2)


def _compute_coordinate_metrics(lon4, lonu, latu, lonv, latv):
    """Compute the curvilinear coordinate metrics pn and pm, defined as 1/grid spacing"""

    # pm = 1/dx
    pmu = gc_dist(lonu[:, :-1], latu[:, :-1], lonu[:, 1:], latu[:, 1:])
    pm = 0 * lon4
    pm[:, 1:-1] = pmu
    pm[:, 0] = pm[:, 1]
    pm[:, -1] = pm[:, -2]
    pm = 1 / pm

    # pn = 1/dy
    pnv = gc_dist(lonv[:-1, :], latv[:-1, :], lonv[1:, :], latv[1:, :])
    pn = 0 * lon4
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


def _compute_angle(lon4, lonu, latu, lone):
    """Compute angles of local grid positive x-axis relative to east"""

    dellat = latu[:, 1:] - latu[:, :-1]
    dellon = lonu[:, 1:] - lonu[:, :-1]
    dellon[dellon > np.pi] = dellon[dellon > np.pi] - 2 * np.pi
    dellon[dellon < -np.pi] = dellon[dellon < -np.pi] + 2 * np.pi
    dellon = dellon * np.cos(0.5 * (latu[:, 1:] + latu[:, :-1]))

    ang = copy.copy(lon4)
    ang_s = np.arctan(dellat / (dellon + 1e-16))
    ang_s[(dellon < 0) & (dellat < 0)] = ang_s[(dellon < 0) & (dellat < 0)] - np.pi
    ang_s[(dellon < 0) & (dellat >= 0)] = ang_s[(dellon < 0) & (dellat >= 0)] + np.pi
    ang_s[ang_s > np.pi] = ang_s[ang_s > np.pi] - np.pi
    ang_s[ang_s < -np.pi] = ang_s[ang_s < -np.pi] + np.pi

    ang[:, 1:-1] = ang_s
    ang[:, 0] = ang[:, 1]
    ang[:, -1] = ang[:, -2]

    lon4[lon4 < 0] = lon4[lon4 < 0] + 2 * np.pi
    lone[lone < 0] = lone[lone < 0] + 2 * np.pi

    return ang


def _create_grid_ds(
    nx,
    ny,
    lon,
    lat,
    pm,
    pn,
    angle,
    size_x,
    size_y,
    rot,
    center_lon,
    center_lat,
    lone,
    late,
):

    # Coriolis frequency
    f0 = 4 * np.pi * np.sin(lat) / (24 * 3600)

    # Create empty xarray.Dataset object to store variables in
    ds = xr.Dataset()

    # TODO some of these variables are defined but never written to in Easy Grid

    ds["angle"] = xr.Variable(
        data=angle,
        dims=["eta_rho", "xi_rho"],
        attrs={"long_name": "Angle between xi axis and east", "units": "radians"},
    )

    # ds['h'] = ...
    # TODO hraw comes from topography
    # ds['hraw'] = xr.Variable(data=hraw, dims=['eta_rho', 'xi_rho'])

    ds["f0"] = xr.Variable(
        data=f0,
        dims=["eta_rho", "xi_rho"],
        attrs={"long_name": "Coriolis parameter at rho-points", "units": "second-1"},
    )
    ds["pn"] = xr.Variable(
        data=pn,
        dims=["eta_rho", "xi_rho"],
        attrs={
            "long_name": "Curvilinear coordinate metric in xi-direction",
            "units": "meter-1",
        },
    )
    ds["pn"] = xr.Variable(
        data=pm,
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

    ds["spherical"] = xr.Variable(
        data=["T"],
        dims=["one"],
        attrs={
            "long_name": "Grid type logical switch",
            "option_T": "spherical",
        },
    )

    # TODO this mask is obtained from hraw
    # ds['mask_rho'] = xr.Variable(data=lat * 180 / np.pi, dims=['eta_rho', 'xi_rho'], attrs={'long_name': "latitude of rho-points", 'units': "degrees North"})

    # TODO this 'one' dimension is completely unneccessary as netCDF can store scalars
    ds["tra_lon"] = xr.Variable(
        data=[center_lon],
        dims=["one"],
        attrs={
            "long_name": "Longitudinal translation of base grid",
            "units": "degrees East",
        },
    )
    ds["tra_lat"] = xr.Variable(
        data=[center_lat],
        dims=["one"],
        attrs={
            "long_name": "Latitudinal translation of base grid",
            "units": "degrees North",
        },
    )
    ds["rotate"] = xr.Variable(
        data=[rot],
        dims=["one"],
        attrs={"long_name": "Rotation of base grid", "units": "degrees"},
    )

    # TODO this is never written to
    # ds['xy_flip']

    return ds


def _add_global_metadata(ds, nx, ny, size_x, size_y, center_lon, center_lat, rot):

    ds.attrs["Title"] = (
        "ROMS grid. Settings:"
        f" nx: {nx} ny: {ny} "
        f" xsize: {size_x / 1e3} ysize: {size_y / 1e3}"
        f" rotate: {rot} Lon: {center_lon} Lat: {center_lat}"
    )
    ds.attrs["Date"] = date.today()
    ds.attrs["Type"] = "ROMS grid produced by roms-tools"

    return ds


def _make_topography(ds):
    ...
