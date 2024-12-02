import xarray as xr
import numpy as np
import regionmask
from scipy.ndimage import label
from roms_tools.setup.utils import (
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    handle_boundaries,
)


def _add_mask(ds):

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
    ds = _add_velocity_masks(ds)

    return ds


def _fill_enclosed_basins(mask) -> np.ndarray:
    """Fills in enclosed basins with land."""

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
