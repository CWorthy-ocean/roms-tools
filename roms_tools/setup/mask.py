import logging
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import regionmask
import xarray as xr
from scipy.ndimage import label

from roms_tools.setup.utils import (
    handle_boundaries,
)
from roms_tools.utils import (
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
)

logger = logging.getLogger(__name__)


def add_mask(ds: xr.Dataset, shapefile: str | Path | None = None) -> xr.Dataset:
    """Adds a land/water mask to the dataset at rho-points.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing latitude and longitude coordinates at rho-points.

    shapefile: str or Path | None
        Path to a coastal shapefile to determine the land mask. If None, NaturalEarth 10m is used.

    Returns
    -------
    xarray.Dataset
        The original dataset with an added 'mask_rho' variable, representing land/water mask.
    """
    # Suppress specific warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="No gridpoint belongs to any region.*"
        )

        if shapefile:
            coast = gpd.read_file(shapefile)

            try:
                # 3D method: returns a boolean array for each region, then take max along the region dimension
                # Pros: more memory-efficient for high-res grids if number of regions isn't extreme
                mask = ~regionmask.mask_3D_geopandas(
                    coast, ds["lon_rho"], ds["lat_rho"]
                ).max(dim="region")

            except MemoryError:
                logging.info(
                    "MemoryError encountered with 3D mask; falling back to 2D method."
                )
                # 2D method: returns a single array with integer codes for each region, using np.nan for points not in any region
                # Pros: works well for small/medium grids
                # Cons: can use a large float64 array internally for very high-resolution grids
                mask_2d = regionmask.mask_geopandas(coast, ds["lon_rho"], ds["lat_rho"])
                mask = mask_2d.isnull()

        else:
            # Use Natural Earth 10m land polygons if no shapefile is provided
            land = regionmask.defined_regions.natural_earth_v5_0_0.land_10
            land_mask = land.mask(ds["lon_rho"], ds["lat_rho"])
            mask = land_mask.isnull()

    ds = _add_coastlines_metadata(ds, shapefile)

    # fill enclosed basins with land
    mask = _fill_enclosed_basins(mask.values)
    # adjust mask boundaries by copying values from adjacent cells
    mask = handle_boundaries(mask)

    ds["mask_rho"] = xr.DataArray(mask.astype(np.int32), dims=("eta_rho", "xi_rho"))
    ds["mask_rho"].attrs = {
        "long_name": "Mask at rho-points",
        "units": "land/water (0/1)",
    }

    ds = add_velocity_masks(ds)

    return ds


def _add_coastlines_metadata(
    ds: xr.Dataset,
    shapefile: str | Path | None = None,
) -> xr.Dataset:
    """
    Add coastline metadata to a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be updated.
    shapefile : str or pathlib.Path or None, optional
        Path to the shapefile used for land/ocean masking.

    Returns
    -------
    xarray.Dataset
        Dataset with updated coastline-related metadata.
    """
    if shapefile is not None:
        ds.attrs["mask_shapefile"] = str(shapefile)

    return ds


def _fill_enclosed_basins(mask) -> np.ndarray:
    """Fills enclosed basins in the mask with land (value = 1).

    This function identifies the largest connected region in the mask, which is assumed to represent
    the land, and sets all other regions to water (value = 0).

    Parameters
    ----------
    mask : np.ndarray
        A binary array representing the land/water mask (land = 1, water = 0).

    Returns
    -------
    np.ndarray
        The modified mask with enclosed basins filled with land (1).
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


def add_velocity_masks(ds):
    """Adds velocity masks for u- and v-points based on the rho-point mask.

    This function generates masks for u- and v-points by interpolating the rho-point land/water mask.
    The interpolation method used is "multiplicative", which scales the rho-point mask to the respective
    u- and v-points.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the land/water mask at rho-points (`mask_rho`).

    Returns
    -------
    xarray.Dataset
        The input dataset with added velocity masks (`mask_u` and `mask_v`) for u- and v-points.
    """
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


def _close_narrow_channels(
    ds: xr.Dataset,
    mask_var: str = "mask_rho",
    max_iterations: int = 10,
    connectivity: int = 4,
    min_region_fraction: float = 0.1,
    inplace: bool = False,
) -> xr.Dataset:
    """Close narrow channels and holes in a ROMS mask (internal function).

    This function performs two main operations:
    1. Fills narrow 1-pixel passages in both north-south and east-west directions
    2. Fills holes by keeping only the largest connected region (unless a region
       is larger than a specified fraction of the domain)

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the mask variable.
    mask_var : str, optional
        Name of the mask variable in the dataset. Default is "mask_rho".
    max_iterations : int, optional
        Maximum number of iterations for closing narrow channels. Default is 10.
    connectivity : int, optional
        Connectivity for connected component labeling. Use 4 for 4-connectivity
        (north, south, east, west) or 8 for 8-connectivity (includes diagonals).
        Default is 4.
    min_region_fraction : float, optional
        Minimum fraction of domain size for a region to be preserved when filling
        holes. Regions smaller than this fraction will be removed unless they are
        the largest region. Default is 0.1 (10%).
    inplace : bool, optional
        If True, modify the dataset in place. If False, return a new dataset.
        Default is False.

    Returns
    -------
    xarray.Dataset
        Dataset with the modified mask. If `inplace=True`, returns the same
        dataset object.

    Notes
    -----
    The function first ensures mask values are non-negative (negative values are
    set to 0). Then it iteratively removes 1-pixel wide passages in both
    north-south and east-west directions. Finally, it identifies connected
    regions and keeps only the largest one, unless another region exceeds the
    minimum region fraction threshold.

    Examples
    --------
    >>> import xarray as xr
    >>> ds = xr.open_dataset("grid.nc")
    >>> ds_filled = _close_narrow_channels(ds)
    >>> ds_filled.to_netcdf("grid_filled.nc")
    """
    # Ensure we have the mask variable
    if mask_var not in ds.variables:
        raise ValueError(f"Mask variable '{mask_var}' not found in dataset.")

    # Get mask and ensure it's non-negative
    mask = ds[mask_var].values.copy()
    mask[mask < 0] = 0

    # Close narrow channels
    for it in range(max_iterations):
        # Fill 1-pixel passages in north-south direction
        fill = mask.copy()
        fill[1:, :] = fill[1:, :] + mask[:-1, :]
        fill[:-1, :] = fill[:-1, :] + mask[1:, :]
        fill[mask < 1] = 0

        nf = np.sum(fill == 1)
        if nf > 0:
            logger.info(
                f"Closing: {nf} points in 1-pixel NS channels (iteration {it + 1})"
            )
            mask[fill == 1] = 0
        else:
            break

        # Fill 1-pixel passages in east-west direction
        fill = mask.copy()
        fill[:, 1:] = fill[:, 1:] + mask[:, :-1]
        fill[:, :-1] = fill[:, :-1] + mask[:, 1:]
        fill[mask < 1] = 0

        nf = np.sum(fill == 1)
        if nf > 0:
            logger.info(
                f"Closing: {nf} points in 1-pixel EW channels (iteration {it + 1})"
            )
            mask[fill == 1] = 0
        else:
            break

    logger.info("Filling holes")

    # Create structure for connected component labeling
    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif connectivity == 8:
        structure = np.ones((3, 3), dtype=int)
    else:
        raise ValueError("connectivity must be 4 or 8")

    # Label connected regions
    reg, nreg = label(mask, structure=structure)

    # Find the largest region
    lint = 0  # size of largest region
    lreg = 0  # number of largest region
    for i in range(1, nreg + 1):
        region_size = np.sum(reg == i)
        if region_size > lint:
            lreg = i
            lint = region_size

    # Remove all regions except the largest one (unless they exceed min_region_fraction)
    ny, nx = mask.shape
    domain_size = nx * ny

    for ireg in range(1, nreg + 1):
        if ireg != lreg:
            region_size = np.sum(reg == ireg)
            if region_size > domain_size * min_region_fraction:
                logger.warning(
                    f"Region {ireg} is large ({region_size} points, "
                    f"{100 * region_size / domain_size:.1f}% of domain). Preserving it."
                )
            else:
                mask[reg == ireg] = 0

    # Update the dataset
    if inplace:
        ds[mask_var].values[:] = mask
        result = ds
    else:
        result = ds.copy(deep=False)
        result[mask_var].values[:] = mask

    return result
