import logging
import typing as t

import numpy as np
import xarray as xr

# Molar mass of CO2 (kg mol^-1), 44.0095 g mol^-1 (IUPAC-style value for conversions).
_MOLAR_MASS_CO2_KG_PER_MOL = 44.0095e-3

# "release" is the CDR intervention itself; "transfer" is a parent domain receiving
# the child's boundary perturbation (ucla-roms ``upscale_output.F90`` writes
# ``ALK_add = ALK_rate - ALK_alt_rate``, same for DIC, so transfer sources
# legitimately take both signs).
SourceRole = t.Literal["release", "transfer"]
SOURCE_ROLES: t.Final[tuple[str, ...]] = t.get_args(SourceRole)

# The outermost rho row and column on each side are boundary points whose values the
# boundary condition sets. What reaches them has left the domain: ucla-roms exports
# it to an upscaled parent across the faces inside them, so the parent counts it.
_INTERIOR: t.Final[dict[str, slice]] = {
    "eta_rho": slice(1, -1),
    "xi_rho": slice(1, -1),
}


def compute_cdr_metrics(
    ds: xr.Dataset, grid_ds: xr.Dataset, source_role: SourceRole = "release"
) -> xr.Dataset:
    """
    Compute Carbon Dioxide Removal (CDR) metrics from model output.

    Calculates CDR uptake efficiency and total carbon uptake using two methods:
      1. Flux-based: from area-integrated CO2 flux differences (time-cumulative
         for uptake).
      2. DIC difference-based: from volume-integrated DIC differences

    Both integrals cover the interior rho points only. The outermost row and
    column on each side are boundary points set by the boundary condition; a
    perturbation there has already left the domain (and, in a nested experiment,
    has been exported to the parent, which counts it).

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

    source_role : {"release", "transfer"}, optional
        Role of ``ALK_source``/``DIC_source`` in this domain. ``"release"`` (default)
        is the CDR intervention itself: the source must satisfy
        ``ALK_source >= 0`` and ``DIC_source <= 0``, and efficiencies are the uptake
        normalised by the cumulative source. ``"transfer"`` marks a parent domain
        that receives the child's boundary perturbation re-injected as a source
        (signed, not a new release): the sign checks are skipped and the efficiency
        variables are all-NaN, because the domain's source is not a release. Uptake
        (tonnes CO2) is computed identically for both roles.

    Returns
    -------
    ds_cdr : xr.Dataset
        Dataset containing:
        - 'area', 'window_length'
        - copied flux/tracer variables
        - 'cdr_efficiency_from_flux' and 'cdr_efficiency_from_DIC_difference'
          (dimensionless; all-NaN for ``source_role="transfer"``)
        - 'cdr_cumulative_source' (time-cumulative volume-integrated
          ``ALK_source - DIC_source``; the efficiency denominator for a release,
          the net perturbation received for a transfer)
        - 'cdr_carbon_uptake_from_flux' (cumulative flux-based uptake, tonnes CO2)
        - 'cdr_carbon_uptake_from_DIC_difference' (DIC-difference-based uptake, tonnes CO2)

        Both uptake variables carry a ``tonnes_co2_per_native_unit`` attribute (the
        scale applied to the native integrated carbon amount), and
        ``ds_cdr.attrs["cdr_source_role"]`` records `source_role`, so the dataset
        is self-describing for downstream combination.

    Raises
    ------
    ValueError
        If `source_role` is not one of ``SOURCE_ROLES``, or if ``source_role="release"``
        and the source violates the release sign constraints.
    KeyError
        If required variables are missing from `ds` or `grid_ds`.
    """
    if source_role not in SOURCE_ROLES:
        raise ValueError(
            f"source_role must be one of {SOURCE_ROLES}, got {source_role!r}"
        )

    # Define required variables
    ds_vars = [
        "avg_begin_time",
        "avg_end_time",
        "ALK_source",
        "DIC_source",
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

    _validate_source(ds, source_role)

    # Cumulative alkalinity source
    source = (
        (
            (ds["ALK_source"] - ds["DIC_source"]).sum(
                dim=["s_rho", "eta_rho", "xi_rho"]
            )
            * ds_cdr["window_length"]
        )
        .cumsum(dim="time")
        .compute()
    )

    # Cumulative flux-based uptake (Method 1)
    flux = (
        (
            ((ds["FG_CO2"] - ds["FG_ALT_CO2"]) * ds_cdr["area"])
            .isel(_INTERIOR)
            .sum(dim=["eta_rho", "xi_rho"])
            * ds_cdr["window_length"]
        )
        .cumsum(dim="time")
        .compute()
    )

    # DIC difference-based uptake (Method 2)
    diff_DIC = (
        ((ds["hDIC"] - ds["hDIC_ALT_CO2"]) * ds_cdr["area"])
        .isel(_INTERIOR)
        .sum(dim=["s_rho", "eta_rho", "xi_rho"])
        .compute()
    )

    if source_role == "release":
        # Normalize by cumulative source with safe division (NaN where source=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            uptake_efficiency_flux = (flux / source).where(np.isfinite(flux / source))
            uptake_efficiency_diff = (diff_DIC / source).where(
                np.isfinite(diff_DIC / source)
            )

        _validate_uptake_efficiency(uptake_efficiency_flux, uptake_efficiency_diff)
        description_suffix = ""
    else:
        uptake_efficiency_flux = xr.full_like(source, np.nan, dtype=float)
        uptake_efficiency_diff = xr.full_like(source, np.nan, dtype=float)
        description_suffix = (
            " (not defined for source_role='transfer': the domain's source is a "
            "transfer, not a release)"
        )

    flux_units = ds["FG_CO2"].attrs.get("units")
    hdic_units = ds["hDIC"].attrs.get("units")
    flux_scale = _native_carbon_amount_to_tonnes_co2_scale(flux_units)
    dic_scale = _native_carbon_amount_to_tonnes_co2_scale(hdic_units)
    flux_tonnes_co2 = flux * flux_scale
    diff_dic_tonnes_co2 = diff_DIC * dic_scale

    source_units = ds["ALK_source"].attrs.get("units") or "mmol"
    source_units = source_units.removesuffix("/s")

    # Store results with metadata
    ds_cdr["cdr_cumulative_source"] = source
    ds_cdr["cdr_cumulative_source"].attrs.update(
        long_name="Cumulative CDR source",
        units=source_units,
        description=(
            "Time-cumulative volume-integrated (ALK_source - DIC_source); for "
            "source_role='release' this is the denominator of the efficiency, for "
            "'transfer' it is the net perturbation received from a child domain"
        ),
    )

    ds_cdr["cdr_efficiency_from_flux"] = uptake_efficiency_flux
    ds_cdr["cdr_efficiency_from_flux"].attrs.update(
        long_name="CDR uptake efficiency (from flux differences)",
        units="nondimensional",
        description="Carbon Dioxide Removal efficiency computed using area-integrated CO2 flux differences"
        + description_suffix,
    )
    ds_cdr["cdr_efficiency_from_DIC_difference"] = uptake_efficiency_diff
    ds_cdr["cdr_efficiency_from_DIC_difference"].attrs.update(
        long_name="CDR uptake efficiency (from DIC differences)",
        units="nondimensional",
        description="Carbon Dioxide Removal efficiency computed using volume-integrated DIC differences"
        + description_suffix,
    )

    ds_cdr["cdr_carbon_uptake_from_flux"] = flux_tonnes_co2
    ds_cdr["cdr_carbon_uptake_from_flux"].attrs.update(
        long_name="CDR carbon uptake (cumulative flux differences)",
        units="tonnes CO2",
        description="Time-cumulative CO2 uptake mass from area-integrated CO2 flux differences",
        tonnes_co2_per_native_unit=flux_scale,
    )

    ds_cdr["cdr_carbon_uptake_from_DIC_difference"] = diff_dic_tonnes_co2
    ds_cdr["cdr_carbon_uptake_from_DIC_difference"].attrs.update(
        long_name="CDR carbon uptake (from DIC differences)",
        units="tonnes CO2",
        description=("CO2-equivalent mass from the volume-integrated DIC difference"),
        tonnes_co2_per_native_unit=dic_scale,
    )

    ds_cdr.attrs["cdr_source_role"] = source_role

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

    logging.info(
        "Max absolute difference between flux-based and DIC-based uptake efficiency: %.3e",
        max_abs_diff,
    )

    return max_abs_diff


def _validate_source(ds: xr.Dataset, source_role: SourceRole):
    """
    Validate ALK_source and DIC_source in a ROMS dataset for the given source role.

    The presence of both variables is always checked. For ``source_role="release"``
    the release sign constraints are also enforced:

    - 'ALK_source' must be non-negative (≥ 0).
    - 'DIC_source' must be non-positive (≤ 0).

    A ``"transfer"`` source is signed, so no sign check is applied.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset expected to contain 'ALK_source' and 'DIC_source'.
    source_role : {"release", "transfer"}
        Role of the source; the sign constraints apply only to ``"release"``.

    Raises
    ------
    KeyError
        If 'ALK_source' or 'DIC_source' are missing from the dataset.
    ValueError
        If ``source_role="release"`` and 'ALK_source' or 'DIC_source' violate the
        release constraints.
    """
    constraints = {
        "ALK_source": lambda x: x >= 0,
        "DIC_source": lambda x: x <= 0,
    }

    for var, check in constraints.items():
        if var not in ds.data_vars:
            raise KeyError(f"Dataset is missing required variable '{var}'.")
        if source_role == "release" and not check(ds[var]).all():
            sign = "negative" if var == "ALK_source" else "positive"
            raise ValueError(
                f"'{var}' contains {sign} values, which violates release constraints."
            )


def _native_carbon_amount_to_tonnes_co2_scale(units: str | None) -> float:
    """
    Scale factor from integrated ROMS-style DIC carbon amounts to tonnes CO2.

    For each mole of DIC carbon counted in the flux or inventory difference, we
    assign one mole of CO2 (air-sea CO2 exchange bookkeeping), then convert to
    mass using the molar mass of CO2:

        tonnes CO2 = n_C (mol) * M_CO2 (kg mol^-1) / (1000 kg tonne^-1)

    ``n_C`` is obtained from the native amount (e.g. mmol) using metadata units
    when present; missing units default to mmol (typical ROMS BGC).
    """
    mol_per_native = _native_dic_amount_to_mol_multiplier(units)
    kg_co2_per_native = mol_per_native * _MOLAR_MASS_CO2_KG_PER_MOL
    return kg_co2_per_native / 1000.0


def _native_dic_amount_to_mol_multiplier(units: str | None) -> float:
    """Convert one native unit of DIC carbon inventory to moles carbon."""
    if units is None:
        logging.warning(
            "DIC carbon amount units are missing; assuming mmol for conversion to tonnes CO2."
        )
        return 1e-3
    u = units.strip().lower().replace(" ", "").replace("µ", "u")
    if u.startswith("mmol") or "mmol/" in u:
        return 1e-3
    if u.startswith("umol") or "umol/" in u or "micromol" in u:
        logging.warning(
            "DIC carbon amount units %r are not mmol; using umol-to-mol conversion.",
            units,
        )
        return 1e-6
    if (u.startswith("mol") or u.startswith("mole")) and not u.startswith("mmol"):
        logging.warning(
            "DIC carbon amount units %r are not mmol; using mol-to-mol conversion.",
            units,
        )
        return 1.0
    logging.warning(
        "DIC carbon amount units %r are not recognized as mmol; assuming mmol.",
        units,
    )
    return 1e-3
