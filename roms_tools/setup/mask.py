import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import regionmask
import xarray as xr
from scipy.ndimage import label

from roms_tools.setup.utils import (
    handle_boundaries,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
)


def add_mask(ds: xr.Dataset, shapefile: str | Path | None = None):
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
            # We use the "3D" version of the mask, which returns an array of booleans for each "region" (polygon)
            # in the original dataset. Then we take the boolean "max" along the region to include any point in any
            # polygon. Finally, we do a boolean inversion (~) to get the same convention as before.

            # There is an alternative "2D" method in regionmask, but it returns a single array that nominally contains
            # the integer of the region at each point, but it uses np.nan to indicate points that aren't in any region,
            # and in order to do that, it makes a huge float64 array that can blow out memory for a high-res grid. The
            # 3D method ends up being more memory efficient as long as the number of "regions" in your domain isn't
            # extreme.
            coast = gpd.read_file(shapefile)
            mask = ~regionmask.mask_3D_geopandas(
                coast, ds["lon_rho"], ds["lat_rho"]
            ).max(dim="region")

        else:
            land = regionmask.defined_regions.natural_earth_v5_0_0.land_10
            land_mask = land.mask(ds["lon_rho"], ds["lat_rho"])
            mask = land_mask.isnull()

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
