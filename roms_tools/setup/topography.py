import xarray as xr
import numpy as np
import gcm_filters
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label
from roms_tools.setup.download import fetch_topo
from roms_tools.setup.utils import interpolate_from_rho_to_u, interpolate_from_rho_to_v
import warnings
from itertools import count


def _add_topography_and_mask(
    ds, topography_source, hmin, smooth_factor=8.0, rmax=0.2
) -> xr.Dataset:
    """
    Adds topography and a land/water mask to the dataset based on the provided topography source.

    This function performs the following operations:
    1. Interpolates topography data onto the desired grid.
    2. Applies a mask based on ocean depth.
    3. Smooths the topography globally to reduce grid-scale instabilities.
    4. Fills enclosed basins with land.
    5. Smooths the topography locally to ensure the steepness ratio satisfies the rmax criterion.
    6. Adds topography metadata.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to which topography and the land/water mask will be added.
    topography_source : str
        The source of the topography data.
    hmin : float
        The minimum allowable depth for the topography.
    smooth_factor : float, optional
        The smoothing factor used in the domain-wide Gaussian smoothing of the
        topography. Smaller values result in less smoothing, while larger
        values produce more smoothing. The default is 8.0.
    rmax : float, optional
        The maximum allowable steepness ratio for the topography smoothing.
        This parameter controls the local smoothing of the topography. Smaller values result in
        smoother topography, while larger values preserve more detail. The default is 0.2.

    Returns
    -------
    xr.Dataset
        The dataset with added topography, mask, and metadata.
    """

    lon = ds.lon_rho.values
    lat = ds.lat_rho.values

    # interpolate topography onto desired grid
    hraw = _make_raw_topography(lon, lat, topography_source)
    hraw = xr.DataArray(data=hraw, dims=["eta_rho", "xi_rho"])

    # Mask is obtained by finding locations where ocean depth is positive
    mask = xr.where(hraw > 0, 1.0, 0.0)

    # smooth topography domain-wide with Gaussian kernel to avoid grid scale instabilities
    hraw = _smooth_topography_globally(hraw, smooth_factor)

    # fill enclosed basins with land
    mask = _fill_enclosed_basins(mask.values)

    # adjust mask boundaries by copying values from adjacent cells
    mask = _handle_boundaries(mask)

    ds["mask_rho"] = xr.DataArray(mask.astype(np.int32), dims=("eta_rho", "xi_rho"))
    ds["mask_rho"].attrs = {
        "long_name": "Mask at rho-points",
        "units": "land/water (0/1)",
    }

    ds = _add_velocity_masks(ds)

    # smooth topography locally to satisfy r < rmax
    ds["h"] = _smooth_topography_locally(hraw * ds["mask_rho"], hmin, rmax)
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
    if topography_source == "ETOPO5":
        topo_lon = topo_ds["topo_lon"].copy()
        # Modify longitude values where necessary
        topo_lon = xr.where(topo_lon < 0, topo_lon + 360, topo_lon)
        topo_lon_minus360 = topo_lon - 360
        topo_lon_plus360 = topo_lon + 360
        # Concatenate along the longitude axis
        topo_lon_concatenated = xr.concat(
            [topo_lon_minus360, topo_lon, topo_lon_plus360], dim="lon"
        )
        topo_concatenated = xr.concat(
            [-topo_ds["topo"], -topo_ds["topo"], -topo_ds["topo"]], dim="lon"
        )

        interp = RegularGridInterpolator(
            (topo_ds["topo_lat"].values, topo_lon_concatenated.values),
            topo_concatenated.values,
            method="linear",
        )

    # Interpolate onto desired domain grid points
    hraw = interp((lat, lon))

    return hraw


def _smooth_topography_globally(hraw, factor) -> xr.DataArray:
    # since GCM-Filters assumes periodic domain, we extend the domain by one grid cell in each dimension
    # and set that margin to land

    mask = xr.ones_like(hraw)
    margin_mask = xr.concat([mask, 0 * mask.isel(eta_rho=-1)], dim="eta_rho")
    margin_mask = xr.concat(
        [margin_mask, 0 * margin_mask.isel(xi_rho=-1)], dim="xi_rho"
    )

    # we choose a Gaussian filter kernel corresponding to a Gaussian with standard deviation factor/sqrt(12);
    # this standard deviation matches the standard deviation of a boxcar kernel with total width equal to factor.
    filter = gcm_filters.Filter(
        filter_scale=factor,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
        grid_vars={"wet_mask": margin_mask},
    )
    hraw_extended = xr.concat([hraw, hraw.isel(eta_rho=-1)], dim="eta_rho")
    hraw_extended = xr.concat(
        [hraw_extended, hraw_extended.isel(xi_rho=-1)], dim="xi_rho"
    )

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
        cff = (h_log - h_log.shift(eta_rho=1, xi_rho=1)).isel(
            eta_rho=slice(1, None), xi_rho=slice(1, None)
        )
        cr = np.abs(cff)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore division by zero warning
            Op3 = xr.where(cr < rmax_log, 0, 1.0 * cff * (1 - rmax_log / cr))

        # in the other diagonal direction
        cff = (h_log.shift(eta_rho=1) - h_log.shift(xi_rho=1)).isel(
            eta_rho=slice(1, None), xi_rho=slice(1, None)
        )
        cr = np.abs(cff)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore division by zero warning
            Op4 = xr.where(cr < rmax_log, 0, 1.0 * cff * (1 - rmax_log / cr))

        # Update h_log in domain interior
        h_log[1:-1, 1:-1] += cf1 * (
            Op1[1:, :]
            - Op1[:-1, :]
            + Op2[:, 1:]
            - Op2[:, :-1]
            + cf2 * (Op3[1:, 1:] - Op3[:-1, :-1] + Op4[:-1, 1:] - Op4[1:, :-1])
        )

        # No gradient at the domain boundaries
        h_log = _handle_boundaries(h_log)

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


def _handle_boundaries(field):
    """
    Adjust the boundaries of a 2D field by copying values from adjacent cells.

    Parameters
    ----------
    field : numpy.ndarray or xarray.DataArray
        A 2D array representing a field (e.g., topography or mask) whose boundary values
        need to be adjusted.

    Returns
    -------
    field : numpy.ndarray or xarray.DataArray
        The input field with adjusted boundary values.

    """

    field[0, :] = field[1, :]
    field[-1, :] = field[-2, :]
    field[:, 0] = field[:, 1]
    field[:, -1] = field[:, -2]

    return field


def _compute_rfactor(h):
    """
    Computes slope parameter (or r-factor) r = |Delta h| / 2h in both horizontal grid directions.
    """
    # compute r_{i-1/2} = |h_i - h_{i-1}| / (h_i + h_{i+1})
    r_eta = np.abs(h.diff("eta_rho")) / (h + h.shift(eta_rho=1)).isel(
        eta_rho=slice(1, None)
    )
    r_xi = np.abs(h.diff("xi_rho")) / (h + h.shift(xi_rho=1)).isel(
        xi_rho=slice(1, None)
    )

    return r_eta, r_xi


def _add_topography_metadata(ds, topography_source, smooth_factor, hmin, rmax):
    ds.attrs["topography_source"] = topography_source
    ds.attrs["hmin"] = hmin

    return ds


def _add_velocity_masks(ds):

    # add u- and v-masks
    ds["mask_u"] = interpolate_from_rho_to_u(
        ds["mask_rho"], method="multiplicative"
    ).astype(np.int32)
    ds["mask_v"] = interpolate_from_rho_to_v(
        ds["mask_rho"], method="multiplicative"
    ).astype(np.int32)

    ds["mask_u"].attrs = {"long_name": "Mask at u-points", "units": "land/water (0/1)"}
    ds["mask_v"].attrs = {"long_name": "Mask at v-points", "units": "land/water (0/1)"}

    return ds
