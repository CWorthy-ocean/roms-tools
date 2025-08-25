import numpy as np
import xarray as xr


def compute_uptake_efficiency(
    ds: xr.Dataset, grid_ds: xr.Dataset
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute CO2 uptake efficiency based on fluxes and DIC differences.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing model outputs, expected to include:
        - 'avg_begin_time', 'avg_end_time': Time bounds of averaging windows.
        - 'ALK_source': Alkalinity source term.
        - 'FG_CO2', 'FG_ALT_CO2': CO2 fluxes.
        - 'hDIC', 'hDIC_ALT_CO2': DIC (dissolved inorganic carbon) concentrations.

    grid_ds : xr.Dataset
        Grid dataset containing grid metrics:
        - 'pm', 'pn': Reciprocal grid spacing in xi and eta directions.

    Returns
    -------
    uptake_flux : xr.DataArray
        Cumulative CO2 uptake efficiency computed from fluxes, normalized by cumulative source.

    uptake_diff : xr.DataArray
        Cumulative CO2 uptake efficiency computed from DIC differences, normalized by cumulative source.
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

    # Duration of each averaging window
    window_length = ds["avg_end_time"] - ds["avg_begin_time"]

    # Grid cell area
    area = 1 / (grid_ds["pm"] * grid_ds["pn"])

    _validate_source(ds)

    # Cumulative alkalinity source
    source = (
        (ds["ALK_source"] - ds["DIC_source"]).sum(dim=["s_rho", "eta_rho", "xi_rho"])
        * window_length
    ).cumsum(dim="time")

    # Cumulative flux-based uptake (Method 1)
    flux = (
        ((ds["FG_CO2"] - ds["FG_ALT_CO2"]) * area).sum(dim=["eta_rho", "xi_rho"])
        * window_length
    ).cumsum(dim="time")

    # DIC difference-based uptake (Method 2)
    diff_DIC = ((ds["hDIC"] - ds["hDIC_ALT_CO2"]) * area).sum(
        dim=["s_rho", "eta_rho", "xi_rho"]
    )

    # Normalize by cumulative source with safe division (NaN where source=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        uptake_efficiency_flux = (flux / source).where(np.isfinite(flux / source))
        uptake_efficiency_diff = (diff_DIC / source).where(
            np.isfinite(diff_DIC / source)
        )

    return uptake_efficiency_flux, uptake_efficiency_diff


def validate_uptake_efficiency(
    uptake_efficiency_flux: xr.DataArray,
    uptake_efficiency_diff: xr.DataArray,
    tol: float = 1e-2,
) -> bool:
    """
    Validate that two uptake efficiency estimates are sufficiently close.

    Parameters
    ----------
    uptake_efficiency_flux : xr.DataArray
        Uptake computed from fluxes.
    uptake_efficiency_diff : xr.DataArray
        Uptake computed from DIC differences.
    tol : float
        Relative tolerance for agreement (default 1%).

    Returns
    -------
    valid : bool
        True if uptake_flux and uptake_diff agree within the specified tolerance.
    """
    # Compute relative difference
    relative_diff = np.abs(
        uptake_efficiency_flux - uptake_efficiency_diff
    ) / np.maximum(np.abs(uptake_efficiency_flux), 1e-10)

    # Check if maximum difference is below tolerance
    max_diff = float(relative_diff.max())
    if max_diff > tol:
        print(f"Validation failed: maximum relative difference = {max_diff:.3e}")
        return False
    return True


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
