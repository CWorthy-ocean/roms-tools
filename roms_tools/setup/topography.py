import time
import logging
import xarray as xr
import numpy as np
import gcm_filters
from roms_tools.setup.utils import handle_boundaries
import warnings
from itertools import count
from roms_tools.setup.datasets import ETOPO5Dataset, SRTM15Dataset
from roms_tools.setup.regrid import LateralRegrid


def _add_topography(
    ds,
    target_coords,
    topography_source,
    hmin,
    smooth_factor=8.0,
    rmax=0.2,
    verbose=False,
) -> xr.Dataset:
    """Adds topography to the dataset based on the provided topography source.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to which topography will be added.
    topography_source : Dict[str, Union[str, Path]], optional
        Dictionary specifying the source of the topography data:

        - "name" (str): The name of the topography data source (e.g., "SRTM15").
        - "path" (Union[str, Path, List[Union[str, Path]]]): The path to the raw data file. Can be a string or a Path object.

        The default is "ETOPO5", which does not require a path.
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
    verbose: bool, optional
        Indicates whether to print topography generation steps with timing. Defaults to False.

    Returns
    -------
    xr.Dataset
        Updated dataset with added topography and metadata.
    """

    if verbose:
        start_time = time.time()
    data = _get_topography_data(topography_source)
    if verbose:
        logging.info(
            f"Reading the topography data: {time.time() - start_time:.3f} seconds"
        )

    # interpolate topography onto desired grid
    hraw = _make_raw_topography(data, target_coords, verbose=verbose)
    nan_check(hraw)

    # smooth topography domain-wide with Gaussian kernel to avoid grid scale instabilities
    if verbose:
        start_time = time.time()
    hraw = _smooth_topography_globally(hraw, smooth_factor)
    if verbose:
        logging.info(
            f"Smoothing the topography globally: {time.time() - start_time:.3f} seconds"
        )

    # smooth topography locally to satisfy r < rmax
    if verbose:
        start_time = time.time()
    # inserting hraw * mask_rho into this function eliminates any inconsistencies between
    # the land according to the topography and the land according to the mask; land points
    # will always be set to hmin
    ds["h"] = _smooth_topography_locally(hraw * ds["mask_rho"], hmin, rmax)
    ds["h"].attrs = {
        "long_name": "Bathymetry at rho-points",
        "units": "meter",
    }
    if verbose:
        logging.info(
            f"Smoothing the topography locally: {time.time() - start_time:.3f} seconds"
        )

    ds = _add_topography_metadata(ds, topography_source, smooth_factor, hmin, rmax)

    return ds


def _get_topography_data(source):

    kwargs = {"use_dask": False}

    if source["name"] == "ETOPO5":
        if "path" in source.keys():
            kwargs["filename"] = source["path"]
        data = ETOPO5Dataset(**kwargs)
    elif source["name"] == "SRTM15":
        kwargs["filename"] = source["path"]
        data = SRTM15Dataset(**kwargs)
    else:
        raise ValueError(
            'Only "ETOPO5" and "SRTM15" are valid options for topography_source["name"].'
        )

    return data


def _make_raw_topography(
    data, target_coords, method="linear", verbose=False
) -> xr.DataArray:

    data.choose_subdomain(target_coords, buffer_points=3, verbose=verbose)

    if verbose:
        start_time = time.time()
    lateral_regrid = LateralRegrid(target_coords, data.dim_names)
    hraw = lateral_regrid.apply(data.ds[data.var_names["topo"]], method=method)
    if verbose:
        logging.info(
            f"Regridding the topography: {time.time() - start_time:.3f} seconds"
        )

    # flip sign so that bathmetry is positive
    hraw = -hraw

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


def _smooth_topography_locally(h, hmin=5, rmax=0.2):
    """Smoothes topography locally to satisfy r < rmax."""
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
        h_log = handle_boundaries(h_log)

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
    """Computes slope parameter (or r-factor) r = |Delta h| / 2h in both horizontal grid
    directions."""
    # compute r_{i-1/2} = |h_i - h_{i-1}| / (h_i + h_{i+1})
    r_eta = np.abs(h.diff("eta_rho")) / (h + h.shift(eta_rho=1)).isel(
        eta_rho=slice(1, None)
    )
    r_xi = np.abs(h.diff("xi_rho")) / (h + h.shift(xi_rho=1)).isel(
        xi_rho=slice(1, None)
    )

    return r_eta, r_xi


def _add_topography_metadata(ds, topography_source, smooth_factor, hmin, rmax):
    ds.attrs["topography_source"] = topography_source["name"]
    ds.attrs["hmin"] = hmin

    return ds


def nan_check(hraw):
    error_message = (
        "NaN values found in regridded topography. This likely occurs because the ROMS grid, including "
        "a small safety margin for interpolation, is not fully contained within the topography dataset's longitude/latitude range. Please ensure that the "
        "dataset covers the entire area required by the ROMS grid."
    )
    if hraw.isnull().any().values:
        raise ValueError(error_message)
