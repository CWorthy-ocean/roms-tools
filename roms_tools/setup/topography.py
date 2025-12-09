import logging
import time
import warnings
from itertools import count

import gcm_filters
import numpy as np
import xarray as xr

from roms_tools.regrid import LateralRegridToROMS
from roms_tools.setup.datasets import ETOPO5Dataset, SRTM15Dataset
from roms_tools.setup.utils import handle_boundaries


def add_topography(
    ds,
    target_coords,
    topography_source,
    hmin,
    smooth_factor=5.0,
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
        values produce more smoothing. The default is 6.0.
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
    area = 1 / ds["pm"] / ds["pn"]
    hraw = _smooth_topography_globally(hraw, smooth_factor, area)
    if verbose:
        logging.info(
            f"Domain-wide topography smoothing: {time.time() - start_time:.3f} seconds"
        )

    # smooth topography locally to satisfy r < rmax
    if verbose:
        start_time = time.time()

    ds["h"] = _smooth_topography_locally(hraw, hmin, rmax)
    ds["h"].attrs = {
        "long_name": "Bathymetry at rho-points",
        "units": "meter",
    }
    if verbose:
        logging.info(
            f"Local topography smoothing: {time.time() - start_time:.3f} seconds"
        )

    ds = _add_topography_metadata(ds, topography_source, smooth_factor, hmin, rmax)

    return ds


def _get_topography_data(source):
    """Load topography data based on the specified source.

    Parameters
    ----------
    source : dict
        A dictionary containing the source details (e.g., "name" and "path").

    Returns
    -------
    data : object
        The loaded topography dataset (ETOPO5 or SRTM15).
    """
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
    """Regrid topography data to match target coordinates.

    Parameters
    ----------
    data : object
        The dataset object containing the topography data.
    target_coords : object
        The target coordinates to which the data will be regridded.
    method : str, optional
        The regridding method to use, by default "linear".
    verbose : bool, optional
        If True, logs the time taken for regridding, by default False.

    Returns
    -------
    xr.DataArray
        The regridded topography data with the sign flipped (bathymetry positive).
    """
    data.choose_subdomain(target_coords, buffer_points=3, verbose=verbose)
    # Enforce double precision to ensure reproducibility
    data.convert_to_float64()

    if verbose:
        start_time = time.time()
    lateral_regrid = LateralRegridToROMS(target_coords, data.dim_names)
    hraw = lateral_regrid.apply(data.ds[data.var_names["topo"]], method=method)
    if verbose:
        logging.info(
            f"Regridding the topography: {time.time() - start_time:.3f} seconds"
        )

    # flip sign so that bathmetry is positive
    hraw = -hraw

    return hraw


def _smooth_topography_globally(hraw, factor, area) -> xr.DataArray:
    """Apply global smoothing to the topography using a Gaussian filter.

    Parameters
    ----------
    hraw : xr.DataArray
        The raw topography data to be smoothed.
    factor : float
        The smoothing factor (controls the width of the Gaussian filter).

    Returns
    -------
    xr.DataArray
        The smoothed topography data.
    """
    # since GCM-Filters assumes periodic domain, we extend the domain by one grid cell in each dimension
    # and set that margin to land
    mask = xr.ones_like(hraw)
    margin_mask = xr.concat([mask, 0 * mask.isel(eta_rho=-1)], dim="eta_rho")
    margin_mask = xr.concat(
        [margin_mask, 0 * margin_mask.isel(xi_rho=-1)], dim="xi_rho"
    )
    area_extended = xr.concat([area, area.isel(eta_rho=-1)], dim="eta_rho")
    area_extended = xr.concat(
        [area_extended, area_extended.isel(xi_rho=-1)], dim="xi_rho"
    )

    # we choose a Gaussian filter kernel corresponding to a Gaussian with standard deviation factor/sqrt(12);
    # this standard deviation matches the standard deviation of a boxcar kernel with total width equal to factor.
    filter = gcm_filters.Filter(
        filter_scale=factor,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
        grid_vars={"wet_mask": margin_mask, "area": area_extended},
    )
    hraw_extended = xr.concat([hraw, hraw.isel(eta_rho=-1)], dim="eta_rho")
    hraw_extended = xr.concat(
        [hraw_extended, hraw_extended.isel(xi_rho=-1)], dim="xi_rho"
    )

    hsmooth = filter.apply(hraw_extended, dims=["eta_rho", "xi_rho"])
    hsmooth = hsmooth.isel(eta_rho=slice(None, -1), xi_rho=slice(None, -1))

    return hsmooth


def _smooth_topography_locally(h, hmin=5, rmax=0.2):
    """Smooths topography locally to ensure the slope (r-factor) is below the specified
    threshold.

    This function applies a logarithmic transformation to the topography and iteratively smooths
    it in four directions (eta, xi, and two diagonals) until the maximum slope parameter (r) is
    below `rmax`. A threshold `hmin` is applied to prevent values from going below a minimum height.

    Parameters
    ----------
    h : xarray.DataArray
        The topography data to be smoothed.
    hmin : float, optional
        The minimum height threshold. Default is 5.
    rmax : float, optional
        The maximum allowable slope parameter (r-factor). Default is 0.2.

    Returns
    -------
    xarray.DataArray
        The smoothed topography data.
    """
    # Compute rmax_log
    if rmax > 0.0:
        rmax_log = np.log((1.0 + rmax * 0.9) / (1.0 - rmax * 0.9))
    else:
        rmax_log = 0.0

    # Apply hmin threshold
    h = clip_depth(h, hmin)

    # Perform logarithmic transformation of the height field
    h_log = np.log(h / hmin)

    # Constants for smoothing
    smoothing_factor_1 = 1.0 / 6
    smoothing_factor_2 = 0.25

    # Iterate until convergence
    for iter in count():
        # Compute gradients and smoothing for eta, xi, and diagonal directions

        # Gradient in eta-direction
        delta_eta = h_log.diff("eta_rho").isel(xi_rho=slice(1, -1))
        abs_eta_gradient = np.abs(delta_eta)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore division by zero warning
            eta_correction = xr.where(
                abs_eta_gradient < rmax_log,
                0,
                delta_eta * (1 - rmax_log / abs_eta_gradient),
            )

        # Gradient in xi-direction
        delta_xi = h_log.diff("xi_rho").isel(eta_rho=slice(1, -1))
        abs_xi_gradient = np.abs(delta_xi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore division by zero warning
            xi_correction = xr.where(
                abs_xi_gradient < rmax_log,
                0,
                delta_xi * (1 - rmax_log / abs_xi_gradient),
            )

        # Gradient in first diagonal direction
        delta_diag_1 = (h_log - h_log.shift(eta_rho=1, xi_rho=1)).isel(
            eta_rho=slice(1, None), xi_rho=slice(1, None)
        )
        abs_diag_1_gradient = np.abs(delta_diag_1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore division by zero warning
            diag_1_correction = xr.where(
                abs_diag_1_gradient < rmax_log,
                0,
                delta_diag_1 * (1 - rmax_log / abs_diag_1_gradient),
            )

        # Gradient in second diagonal direction
        delta_diag_2 = (h_log.shift(eta_rho=1) - h_log.shift(xi_rho=1)).isel(
            eta_rho=slice(1, None), xi_rho=slice(1, None)
        )
        abs_diag_2_gradient = np.abs(delta_diag_2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore division by zero warning
            diag_2_correction = xr.where(
                abs_diag_2_gradient < rmax_log,
                0,
                delta_diag_2 * (1 - rmax_log / abs_diag_2_gradient),
            )

        # Update h_log in domain interior
        h_log[1:-1, 1:-1] += smoothing_factor_1 * (
            eta_correction[1:, :]
            - eta_correction[:-1, :]
            + xi_correction[:, 1:]
            - xi_correction[:, :-1]
            + smoothing_factor_2
            * (
                diag_1_correction[1:, 1:]
                - diag_1_correction[:-1, :-1]
                + diag_2_correction[:-1, 1:]
                - diag_2_correction[1:, :-1]
            )
        )

        # No gradient at the domain boundaries
        h_log = handle_boundaries(h_log)

        # Recompute the topography after smoothing
        h = hmin * np.exp(h_log)

        # Apply hmin threshold again
        h = clip_depth(h, hmin)

        # Compute maximum slope parameter r
        r_eta, r_xi = _compute_rfactor(h)
        rmax0 = np.max([r_eta.max(), r_xi.max()])
        if rmax0 < rmax:
            break

    return h


def clip_depth(h: xr.DataArray, hmin: float) -> xr.DataArray:
    """Ensures that depth values do not fall below a minimum threshold.

    This function replaces all depth values in `h` that are less than `hmin` with `hmin`,
    ensuring a minimum depth constraint.

    Parameters
    ----------
    h : xr.DataArray
        The depth (bathymetry) array.
    hmin : float
        The minimum allowable depth value.

    Returns
    -------
    xr.DataArray
        The modified depth array with values clipped at `hmin`.
    """
    return xr.where(h < hmin, hmin, h)


def _compute_rfactor(h):
    """Computes the slope parameter (r-factor) in both horizontal directions.

    The r-factor is calculated as |Δh| / (2h) in the eta and xi directions:
        - r_eta = |h_i - h_{i-1}| / (h_i + h_{i+1})
        - r_xi = |h_i - h_{i-1}| / (h_i + h_{i+1})

    Parameters
    ----------
    h : xarray.DataArray
        The topography data.

    Returns
    -------
    tuple of xarray.DataArray
        r_eta : r-factor in the eta direction.
        r_xi : r-factor in the xi direction.
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
    """Adds topography metadata to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to update.
    topography_source : dict
        Dictionary with topography source information (requires 'name' key).
    smooth_factor : float
        Smoothing factor (unused in this function).
    hmin : float
        Minimum height threshold for smoothing.
    rmax : float
        Maximum slope parameter (unused in this function).

    Returns
    -------
    xarray.Dataset
        Updated dataset with added metadata.
    """
    ds.attrs["topography_source_name"] = topography_source["name"]
    if "path" in topography_source:
        ds.attrs["topography_source_path"] = topography_source["path"]
    ds.attrs["hmin"] = hmin

    return ds


def nan_check(hraw):
    """Checks for NaN values in the topography data.

    Parameters
    ----------
    hraw : xarray.DataArray
        Input topography data to check for NaN values.

    Raises
    ------
    ValueError
        If NaN values are found in the data, raises an error with a descriptive message.
    """
    error_message = (
        "NaN values found in regridded topography. This likely occurs because the ROMS grid, including "
        "a small safety margin for interpolation, is not fully contained within the topography dataset's longitude/latitude range. Please ensure that the "
        "dataset covers the entire area required by the ROMS grid."
    )
    if hraw.isnull().any().values:
        raise ValueError(error_message)
