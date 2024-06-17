import xarray as xr
import numpy as np
import gcm_filters
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label
from roms_tools.setup.datasets import fetch_topo
import warnings
from itertools import count

def _add_topography_and_mask(ds, topography_source, smooth_factor, hmin, rmax) -> xr.Dataset:

    lon = ds.lon_rho.values
    lat = ds.lat_rho.values

    # interpolate topography onto desired grid
    hraw = _make_raw_topography(lon, lat, topography_source)
    hraw = xr.DataArray(data=hraw, dims=["eta_rho", "xi_rho"])

    # Mask is obtained by finding locations where ocean depth is positive 
    mask = xr.where(hraw > 0, 1, 0)

    # smooth topography globally with Gaussian kernel to avoid grid scale instabilities
    ds["hraw"] = _smooth_topography_globally(hraw, mask, smooth_factor)
    ds["hraw"].attrs = {
        "long_name": "Working bathymetry at rho-points", 
        "source": f"Raw bathymetry from {topography_source} (smoothing diameter {smooth_factor})",
        "units": "meter",
    }

    # fill enclosed basins with land
    mask = _fill_enclosed_basins(mask.values)
    ds["mask_rho"] = xr.DataArray(mask, dims=("eta_rho", "xi_rho"))
    ds["mask_rho"].attrs = {
            "long_name": "Mask at rho-points", 
            "units": "land/water (0/1)"
    }

    # smooth topography locally to satisfy r < rmax
    ds["h"] = _smooth_topography_locally(ds["hraw"] * ds["mask_rho"], hmin, rmax)
    ds["h"].attrs = {
        "long_name": "Final bathymetry at rho-points", 
        "units": "meter",
    }

    ds = _add_topography_metadata(ds, topography_source, smooth_factor, hmin, rmax)

    return ds

def _make_raw_topography(lon, lat, topography_source) -> np.ndarray:
    """
    Given a grid of (lon, lat) points, fetch the topography file and interpolate height values onto the desired grid.
    """

    topo_ds = fetch_topo(topography_source)

    # the following will depend on the topography source
    if topography_source == "etopo5":

        topo_lon = topo_ds["topo_lon"].copy()
        # Modify longitude values where necessary
        topo_lon = xr.where(topo_lon < 0, topo_lon + 360, topo_lon)
        topo_lon_minus360 = topo_lon - 360
        topo_lon_plus360 = topo_lon + 360
        # Concatenate along the longitude axis
        topo_lon_concatenated = xr.concat([topo_lon_minus360, topo_lon, topo_lon_plus360], dim="lon")
        topo_concatenated = xr.concat([-topo_ds["topo"], -topo_ds["topo"], -topo_ds["topo"]], dim="lon")

        interp = RegularGridInterpolator((topo_ds["topo_lat"].values, topo_lon_concatenated.values), topo_concatenated.values, method='linear')

    # Interpolate onto desired domain grid points
    hraw = interp((lat, lon))

    return hraw

def _smooth_topography_globally(hraw, wet_mask, factor) -> xr.DataArray:

    # since GCM-Filters assumes periodic domain, we extend the domain by one grid cell in each dimension
    # and set that margin to land
    margin_mask = xr.concat([wet_mask, 0 * wet_mask.isel(eta_rho=-1)], dim="eta_rho")
    margin_mask = xr.concat([margin_mask, 0 * margin_mask.isel(xi_rho=-1)], dim="xi_rho")
    
    # we choose a Gaussian filter kernel corresponding to a Gaussian with standard deviation factor/sqrt(12);
    # this standard deviation matches the standard deviation of a boxcar kernel with total width equal to factor.
    filter = gcm_filters.Filter(
        filter_scale=factor,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
        grid_vars={'wet_mask': margin_mask}
    )
    hraw_extended = xr.concat([hraw, hraw.isel(eta_rho=-1)], dim="eta_rho")
    hraw_extended = xr.concat([hraw_extended, hraw_extended.isel(xi_rho=-1)], dim="xi_rho")
    
    hsmooth = filter.apply(hraw_extended, dims=["eta_rho", "xi_rho"])
    hsmooth = hsmooth.isel(eta_rho=slice(None, -1), xi_rho=slice(None, -1))

    return hsmooth

def _fill_enclosed_basins(mask) -> np.ndarray:
    """
    Fills in enclosed basins with land
    """

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

def _smooth_topography_locally(h, hmin=5, rmax=0.2):
    """
    Smoothes topography locally to satisfy r < rmax
    """
    # Compute rmax_log
    if rmax > 0.0:
        rmax_log = np.log((1.0 + rmax * 0.9) / (1.0 - rmax * 0.9))
    else:
        rmax_log = 0.0
    
    # Apply hmin threshold
    h = xr.where(h < hmin, hmin, h)    

    # We will smooth logarithmically
    h_log = np.log(h / hmin)

    cf1 = 1.0 / 6
    cf2 = 0.25

    for iter in count():

        # Compute gradients in domain interior
        
        # in eta-direction
        cff = h_log.diff("eta_rho").isel(xi_rho=slice(1, -1))
        cr = np.abs(cff)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore division by zero warning
            Op1 = xr.where(cr < rmax_log, 0, 1.0 * cff * (1 - rmax_log / cr))

        # in xi-direction
        cff = h_log.diff("xi_rho").isel(eta_rho=slice(1, -1))
        cr = np.abs(cff)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore division by zero warning
            Op2 = xr.where(cr < rmax_log, 0, 1.0 * cff * (1 - rmax_log / cr))

        # in diagonal direction
        cff = (h_log - h_log.shift(eta_rho=1, xi_rho=1)).isel(eta_rho=slice(1, None), xi_rho=slice(1, None))
        cr = np.abs(cff)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore division by zero warning
            Op3 = xr.where(cr < rmax_log, 0, 1.0 * cff * (1 - rmax_log / cr))
    
        # in the other diagonal direction
        cff = (h_log.shift(eta_rho=1) - h_log.shift(xi_rho=1)).isel(eta_rho=slice(1, None), xi_rho=slice(1, None))
        cr = np.abs(cff)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore division by zero warning
            Op4 = xr.where(cr < rmax_log, 0, 1.0 * cff * (1 - rmax_log / cr))

        # Update h_log in domain interior
        h_log[1:-1, 1:-1] += cf1 * (Op1[1:, :] - Op1[:-1, :] + Op2[:, 1:] - Op2[:, :-1]
                                  + cf2 * (Op3[1:, 1:] - Op3[:-1, :-1] + Op4[:-1, 1:] - Op4[1:, :-1]))

        # No gradient at the domain boundaries
        h_log[0, :] = h_log[1, :]
        h_log[-1, :] = h_log[-2, :]
        h_log[:, 0] = h_log[:, 1]
        h_log[:, -1] = h_log[:, -2]

        # Update h
        h = hmin * np.exp(h_log)
        # Apply hmin threshold again
        h = xr.where(h < hmin, hmin, h)    

        # compute maximum slope parameter r
        r_eta, r_xi = _compute_rfactor(h)
        rmax0 = np.max([r_eta.max(), r_xi.max()])
        if rmax0 < rmax:
            break


    return h

def _compute_rfactor(h):
    """
    Computes slope parameter (or r-factor) r = |Delta h| / 2h in both horizontal grid directions.
    """
    # compute r_{i-1/2} = |h_i - h_{i-1}| / (h_i + h_{i+1})
    r_eta = np.abs(h.diff("eta_rho")) / (h + h.shift(eta_rho=1)).isel(eta_rho=slice(1, None))
    r_xi = np.abs(h.diff("xi_rho")) / (h + h.shift(xi_rho=1)).isel(xi_rho=slice(1, None))
    
    return r_eta, r_xi


def _add_topography_metadata(ds, topography_source, smooth_factor, hmin, rmax):

    ds.attrs["topography_source"] = topography_source
    ds.attrs["smooth_factor"] = smooth_factor
    ds.attrs["hmin"] = hmin
    ds.attrs["rmax"] = rmax

    return ds

