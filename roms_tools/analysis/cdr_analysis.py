import logging

import numpy as np
import xarray as xr


def compute_cdr_metrics(ds: xr.Dataset, grid_ds: xr.Dataset) -> xr.Dataset:
    """
    Compute Carbon Dioxide Removal (CDR) metrics from model output.

    Calculates CDR uptake efficiency using two methods:
      1. Flux-based: from area-integrated CO2 flux differences.
      2. DIC difference-based: from volume-integrated DIC differences.

    Copies selected tracer and flux variables and computes grid cell areas
    and averaging window durations.

    Parameters
    ----------
    ds : xr.Dataset
        Model output with required variables:
        'avg_begin_time', 'avg_end_time', 'ALK_source', 'DIC_source',
        'FG_CO2', 'FG_ALT_CO2', 'hDIC', 'hDIC_ALT_CO2'.

    grid_ds : xr.Dataset
        Grid dataset with 'pm', 'pn' (inverse grid spacing).

    Returns
    -------
    ds_cdr : xr.Dataset
        Dataset containing:
        - 'area', 'window_length'
        - copied flux/tracer variables
        - 'cdr_efficiency' and 'cdr_efficiency_from_delta_diff' (dimensionless)

    Raises
    ------
    KeyError
        If required variables are missing from `ds` or `grid_ds`.
    """
    # Define required variables
    ds_vars = [
        "avg_begin_time",
        "avg_end_time",
        "ALK_source",
        "FG_CO2",
        "FG_ALT_CO2",
        "hDIC",
        "hDIC_ALT_CO2",
    ]
    grid_vars = ["pm", "pn"]

    # Check that all required variables exist
    missing_ds = [var for var in ds_vars if var not in ds]
    missing_grid = [var for var in grid_vars if var not in grid_ds]

    if missing_ds:
        raise KeyError(f"Missing required variables in ds: {missing_ds}")
    if missing_grid:
        raise KeyError(f"Missing required variables in grid_ds: {missing_grid}")

    ds_cdr = xr.Dataset()

    # Copy relevant variables
    vars_to_copy = ["FG_CO2", "FG_ALT_CO2", "hDIC", "hDIC_ALT_CO2"]
    for var_name in vars_to_copy:
        ds_cdr[var_name] = ds[var_name]

    # Grid cell area
    ds_cdr["area"] = 1 / (grid_ds["pm"] * grid_ds["pn"])
    ds_cdr["area"].attrs.update(
        long_name="Grid cell area",
        units="m^2",
    )

    # Duration of each averaging window
    ds_cdr["window_length"] = ds["avg_end_time"] - ds["avg_begin_time"]
    ds_cdr["window_length"].attrs.update(
        long_name="Duration of each averaging window",
        units="s",
    )

    _validate_source(ds)

    # Cumulative alkalinity source
    source = (
        (ds["ALK_source"] - ds["DIC_source"]).sum(dim=["s_rho", "eta_rho", "xi_rho"])
        * ds_cdr["window_length"]
    ).cumsum(dim="time")

    # Cumulative flux-based uptake (Method 1)
    flux = (
        ((ds["FG_CO2"] - ds["FG_ALT_CO2"]) * ds_cdr["area"]).sum(
            dim=["eta_rho", "xi_rho"]
        )
        * ds_cdr["window_length"]
    ).cumsum(dim="time")

    # DIC difference-based uptake (Method 2)
    diff_DIC = ((ds["hDIC"] - ds["hDIC_ALT_CO2"]) * ds_cdr["area"]).sum(
        dim=["s_rho", "eta_rho", "xi_rho"]
    )

    # Normalize by cumulative source with safe division (NaN where source=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        uptake_efficiency_flux = (flux / source).where(np.isfinite(flux / source))
        uptake_efficiency_diff = (diff_DIC / source).where(
            np.isfinite(diff_DIC / source)
        )

    _validate_uptake_efficiency(uptake_efficiency_flux, uptake_efficiency_diff)

    # Store results with metadata
    ds_cdr["cdr_efficiency"] = uptake_efficiency_flux
    ds_cdr["cdr_efficiency"].attrs.update(
        long_name="CDR uptake efficiency (from flux differences)",
        units="nondimensional",
        description="Carbon Dioxide Removal efficiency computed using area-integrated CO2 flux differences",
    )
    ds_cdr["cdr_efficiency_from_delta_diff"] = uptake_efficiency_diff
    ds_cdr["cdr_efficiency_from_delta_diff"].attrs.update(
        long_name="CDR uptake efficiency (from DIC differences)",
        units="nondimensional",
        description="Carbon Dioxide Removal efficiency computed using volume-integrated DIC differences",
    )

    return ds_cdr


def _validate_uptake_efficiency(
    uptake_efficiency_flux: xr.DataArray,
    uptake_efficiency_diff: xr.DataArray,
) -> float:
    """
    Compute and log the maximum absolute difference between two uptake efficiency estimates.

    Parameters
    ----------
    uptake_efficiency_flux : xr.DataArray
        Uptake computed from fluxes.
    uptake_efficiency_diff : xr.DataArray
        Uptake computed from DIC differences.

    Returns
    -------
    max_abs_diff : float
        Maximum absolute difference between uptake_flux and uptake_diff.
    """
    abs_diff = np.abs(uptake_efficiency_flux - uptake_efficiency_diff)
    max_abs_diff = float(abs_diff.max())

    logging.info("Max absolute difference in uptake efficiency: %.3e", max_abs_diff)

    return max_abs_diff


def _validate_source(ds: xr.Dataset):
    """
    Validate that ALK_source and DIC_source in a ROMS dataset respect release constraints.

    - 'ALK_source' must be non-negative (≥ 0).
    - 'DIC_source' must be non-positive (≤ 0).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset expected to contain 'ALK_source' and 'DIC_source'.

    Raises
    ------
    KeyError
        If 'ALK_source' or 'DIC_source' are missing from the dataset.
    ValueError
        If 'ALK_source' or 'DIC_source' violate the release constraints.
    """
    constraints = {
        "ALK_source": lambda x: x >= 0,
        "DIC_source": lambda x: x <= 0,
    }

    for var, check in constraints.items():
        if var not in ds.data_vars:
            raise KeyError(f"Dataset is missing required variable '{var}'.")
        if not check(ds[var]).all():
            sign = "negative" if var == "ALK_source" else "positive"
            raise ValueError(
                f"'{var}' contains {sign} values, which violates release constraints."
            )
