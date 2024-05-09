import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from roms_tools.setup.datasets import fetch_topo

def _add_topography_and_mask(ds, topography_source) -> xr.Dataset:

    lon = ds.lon_rho.values
    lat = ds.lat_rho.values

    hraw = _make_raw_topography(lon, lat, topography_source)
    ds["hraw"] = xr.Variable(
        data=hraw,
        dims=["eta_rho", "xi_rho"],
        attrs={"long_name": "Working bathymetry at rho-points", "units": "meter"},
    )

    # Mask is obtained by finding locations where height is above sea level (i.e. 0)
    mask = xr.where(ds["hraw"] > 0, 0, 1)
    mask.attrs = {"long_name": "Mask at rho-points", "units": "land/water (0/1)"}
    ds["mask_rho"] = mask

    return ds

def _make_raw_topography(lon, lat, topography_source) -> np.ndarray:
    """
    Given a grid of (lon, lat) points, fetch the topography file and interpolate height values onto the desired grid.
    """

    topo_ds = fetch_topo(topography_source)

    # the following will depend on the topography source
    if topography_source == "etopo5.nc":

        topo_lon = topo_ds["topo_lon"].copy()
        # Modify longitude values where necessary
        topo_lon = xr.where(topo_lon < 0, topo_lon + 360, topo_lon)
        # Create a new longitude coordinate with modified values
        topo_lon_minus360 = topo_lon - 360
        # Concatenate along the longitude axis
        topo_lon_concatenated = xr.concat([topo_lon_minus360, topo_lon], dim="lon")
        topo_concatenated = xr.concat([topo_ds["topo"], topo_ds["topo"]], dim="lon")

        interp = RegularGridInterpolator((topo_ds["topo_lat"].values, topo_lon_concatenated.values), topo_concatenated.values)

    # Interpolate onto desired domain grid points
    hraw = interp((lat, lon))

    return hraw
