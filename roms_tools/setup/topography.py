import xarray as xr
import numpy as np
import gcm_filters
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label
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

    # smooth topography globally with Gaussian kernel to avoid grid scale instabilities
    ds["hsmooth"] = _smooth_topography(ds["hraw"], ds["mask_rho"])

    # fill enclosed basins with land
    mask = _fill_enclosed_basins(ds["mask_rho"].copy().values)
    ds["mask_rho_filled"] = xr.DataArray(mask, dims=("eta_rho", "xi_rho"))

    # smooth topography locally where still necessary
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
        topo_lon_minus360 = topo_lon - 360
        topo_lon_plus360 = topo_lon + 360
        # Concatenate along the longitude axis
        topo_lon_concatenated = xr.concat([topo_lon_minus360, topo_lon, topo_lon_plus360], dim="lon")
        topo_concatenated = xr.concat([topo_ds["topo"], topo_ds["topo"], topo_ds["topo"]], dim="lon")

        interp = RegularGridInterpolator((topo_ds["topo_lat"].values, topo_lon_concatenated.values), topo_concatenated.values)

    # Interpolate onto desired domain grid points
    hraw = interp((lat, lon))

    return hraw

def _smooth_topography(hraw, wet_mask) -> xr.DataArray:

    # we choose a Gaussian filter kernel with filter scale 8 (corresponding to a Gaussian with standard deviation 8/sqrt(12),
    # see https://gcm-filters.readthedocs.io/en/latest/theory.html#filter-scale-and-shape)
    filter = gcm_filters.Filter(
        filter_scale=8,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
        grid_vars={'wet_mask': wet_mask}
    
    )
    hsmooth = filter.apply(hraw, dims=["eta_rho", "xi_rho"])

    return hsmooth

def _fill_enclosed_basins(mask) -> np.ndarray:

    # Label connected regions in the mask
    reg, nreg = label(mask)
    # Find the largest region
    lint = 0
    lreg = 0
    for ireg in range(nreg):
        int_ = np.sum(reg == ireg)
        if int_ > lint and mask[reg == ireg].sum() > 0:
            lreg = ireg
            lint = int_

    # Remove regions other than the largest one
    for ireg in range(nreg):
        if ireg != lreg:
            mask[reg == ireg] = 0

    return mask



