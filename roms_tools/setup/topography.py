import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from roms_tools.setup.datasets import fetch_topo

def _add_topography_and_mask(ds, topography_source) -> xr.Dataset:

    # TODO should we just get lon and lat from the dataset?
    # Or refactor to just create the hraw variable to be added?

    # TODO do we need to set h? When does that get set?
    # ds['h'] = ...

    lon = ds.lon_rho.values
    lat = ds.lat_rho.values

    hraw = _make_topography(lon, lat, topography_source)
    ds["hraw"] = xr.Variable(
        data=hraw,
        dims=["eta_rho", "xi_rho"],
        attrs={"long_name": "Working bathymetry at rho-points", "units": "meter"},
    )

    # Mask is obtained by finding locations where height is above sea level (i.e. 0)
    mask = xr.ones_like(ds["hraw"])
    mask[hraw > 0] = 0
    mask.attrs = {"long_name": "Mask at rho-points", "units": "land/water (0/1)"}
    ds["mask_rho"] = mask

    return ds


def _make_topography(lon, lat, topography_source) -> np.ndarray:
    """
    Given a grid of (lon, lat) points, fetch the topography file and interpolate height values onto the desired grid.
    """

    ds = fetch_topo(topography_source)

    # TODO rewrite this to not drop into numpy land
    topo_lon = topo_ds["topo_lon"].data
    topo_lat = topo_ds["topo_lat"].data
    d = np.transpose(topo_ds["topo"].data.astype("float64"))
    topo_lon[topo_lon < 0] = topo_lon[topo_lon < 0] + 360
    topo_lonm = topo_lon - 360

    topo_loncat = np.concatenate((topo_lonm, topo_lon), axis=0)
    d_loncat = np.concatenate((d, d), axis=0)

    interp = RegularGridInterpolator((topo_loncat, topo_lat), d_loncat)

    # Interpolate onto desired domain grid points
    hraw = interp((lon * 180 / np.pi, lat * 180 / np.pi))

    return hraw
