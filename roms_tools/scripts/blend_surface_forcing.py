"""Combine two ROMS surface-forcing products into one, with optional feathering.

Typical use: a regional product (e.g. CONUS404) leaves ``NaN`` outside its footprint
on the ROMS grid, and a global product (e.g. ERA5) covers the whole grid. Build one
:class:`~roms_tools.SurfaceForcing` for each and blend them::

    from roms_tools.scripts.blend_surface_forcing import (
        blend_surface_forcing,
        save_blended_surface_forcing,
    )

    blended = blend_surface_forcing(
        conus404_sf,  # primary: NaN outside its footprint
        era5_sf,  # fallback: covers the whole grid
        blend_width_km=150.0,
    )
    save_blended_surface_forcing(blended, "blended_surface_forcing")

Either argument may be a :class:`~roms_tools.SurfaceForcing`, an
``xarray.Dataset``, or a path, glob or list of paths to NetCDF files written by
:meth:`~roms_tools.SurfaceForcing.save` (or a single zarr store). Write the result with
:func:`save_blended_surface_forcing`, which splits it into monthly files and applies
the encodings ROMS needs, as ``SurfaceForcing.save`` does.

The layering model is "primary over fallback": wherever the primary has a value it
wins, and wherever it does not -- because the target point lies outside a
limited-extent source's footprint -- the fallback fills in. A NaN in the primary is
the only signal of missing coverage.

``blend_width_km`` carries a feathered transition across the seam: the primary's
weight ramps from 0 at the edge of its footprint to 1 that far inside it. The ramp
is one-sided, running inward through the primary's own territory. That is the right
way round for a nested product: the outer rim of a regional model's domain is its
lateral relaxation zone, where it is nudged toward the reanalysis that drove it and
is least independent of it, so preferring the global source there gives up almost
nothing, while letting the regional product bleed outward past its own domain would
extrapolate it into territory it never simulated. Feathering removes the
discontinuity caused by switching source mid-field; it does not make the boundary
invisible -- what remains is the two products' genuine disagreement.

The layering helpers below are deliberately free of any dependency on
``roms_tools.setup``: they operate on already-regridded ``xarray.DataArray``
objects on the target grid, so they can be tested with synthetic arrays and no I/O.
The weighted form reduces *exactly* to the hard edge when the weights are zero or
one, which keeps the two behaviours on one code path.

Neither product can be checked for physical consistency here. Compare the two over
their overlap before trusting a blend: an offset near 273 in ``Tair`` (Kelvin versus
Celsius), a factor of 10 in ``rain``, or a large wind difference with a near-zero
mean (a rotation error) are not ordinary reanalysis disagreement.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import xarray as xr

from roms_tools.utils import DEFAULT_NETCDF_FORMAT, NetCDFFormat

if TYPE_CHECKING:
    from roms_tools import Grid, SurfaceForcing

# What to do when the primary lacks a variable the fallback provides.
ON_MISSING_PRIMARY_VAR = ("error", "fallback")

# How to put the fallback onto the primary's time axis.
TIME_ALIGN_METHODS = ("exact", "nearest", "linear")

BLEND_PROFILES = ("smoothstep", "linear")

RADIATION_VARS = ("swrad", "lwrad")


def layer_field(
    primary: xr.DataArray,
    fallback: xr.DataArray,
    *,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Layer ``primary`` over ``fallback``.

    Parameters
    ----------
    primary : xr.DataArray
        The preferred field. NaN marks points it does not cover.
    fallback : xr.DataArray
        The field to use where ``primary`` is NaN. Must already be aligned with
        ``primary`` (see :func:`align_fallback_time`).
    weights : xr.DataArray, optional
        Per-point weight on the primary, in [0, 1]. ``None`` (the default) gives a
        hard edge: the primary wherever it is valid, the fallback elsewhere.

    Returns
    -------
    xr.DataArray
        The combined field. Lazy if the inputs are lazy.

    Notes
    -----
    With ``weights`` supplied, the weight is forced to zero wherever the primary
    is NaN, so a point with no primary value is pure fallback regardless of what
    the weight field says there, and a point of weight 1 is pure primary even
    where the fallback is NaN. That is what makes ``weights`` of exactly 0/1
    reproduce the ``weights=None`` result.
    """
    if weights is None:
        return xr.where(primary.notnull(), primary, fallback, keep_attrs=True)

    w = weights.where(primary.notnull(), 0.0)
    combined = w * primary.fillna(0.0) + (1.0 - w) * fallback
    # At full weight the primary stands alone, as on the hard edge, even where the
    # fallback is NaN (which `(1 - w) * fallback` would otherwise propagate).
    combined = xr.where(w >= 1.0, primary, combined)
    # A 2D weight field leads the multiplication, so the result comes out with the
    # horizontal dims first. Restore the primary's order so the weighted and
    # unweighted paths are interchangeable for the caller. The float64 weights
    # would also promote float32 fields; keep the dtype the unweighted path gives.
    combined = combined.transpose(*primary.dims).astype(
        np.result_type(primary.dtype, fallback.dtype)
    )
    combined.attrs = dict(primary.attrs)
    return combined


def align_fallback_time(
    fallback: xr.DataArray,
    target_time: xr.DataArray,
    *,
    method: Literal["exact", "nearest", "linear"] = "exact",
    tolerance: np.timedelta64 | float | None = None,
    var_name: str = "field",
) -> xr.DataArray:
    """Put ``fallback`` onto ``target_time``.

    Parameters
    ----------
    fallback : xr.DataArray
        The field to reindex. Must have a ``time`` dimension.
    target_time : xr.DataArray
        The time coordinate to select onto, normally the primary's.
    method : {"exact", "nearest", "linear"}, optional
        ``"exact"`` (default) requires every target stamp to be present in
        ``fallback`` and selects them, which is correct and free for two sources
        on the same hourly grid. ``"nearest"`` tolerates stamp jitter within
        ``tolerance``. ``"linear"`` interpolates.
    tolerance : np.timedelta64 or float, optional
        Maximum distance for ``method="nearest"``. A ``np.timedelta64`` is converted
        to days when the time axis is numeric, as in a ``SurfaceForcing`` dataset
        (days since the model reference date); a float is used as is, in the
        units of the time axis.
    var_name : str, optional
        Used in error messages.

    Returns
    -------
    xr.DataArray
        ``fallback`` on ``target_time``. Stays lazy.

    Raises
    ------
    ValueError
        If ``method="exact"`` and some target stamps are missing, or if ``method``
        is unknown.
    """
    if method not in TIME_ALIGN_METHODS:
        raise ValueError(
            f"Unknown time alignment method {method!r}; expected one of "
            f"{list(TIME_ALIGN_METHODS)}."
        )

    if "time" not in fallback.dims:
        return fallback

    if method == "exact":
        missing = np.setdiff1d(
            np.asarray(target_time.values), np.asarray(fallback["time"].values)
        )
        if missing.size:
            raise ValueError(
                f"The fallback source is missing {missing.size} time step(s) that "
                f"the primary source provides for {var_name!r}, so the two cannot "
                f"be layered on an exact time match. First missing: {missing[0]}; "
                f"last: {missing[-1]}. Widen the fallback's time range, or use "
                f'time_align="nearest".'
            )
        return fallback.sel(time=target_time)

    if method == "nearest":
        if isinstance(tolerance, np.timedelta64) and not np.issubdtype(
            fallback["time"].dtype, np.datetime64
        ):
            tolerance = float(tolerance / np.timedelta64(1, "D"))
        aligned = fallback.sel(time=target_time, method="nearest", tolerance=tolerance)
        # `sel(method="nearest")` keeps the *source's* time labels. Left as-is, the
        # combine would then align on a time index that does not match the
        # primary's and silently produce an empty intersection.
        return aligned.assign_coords(time=target_time)

    return fallback.interp(time=target_time)


def layer_fields(
    primary: dict[str, xr.DataArray],
    fallback: dict[str, xr.DataArray],
    *,
    weights: xr.DataArray | None = None,
    on_missing_primary_var: Literal["error", "fallback"] = "error",
    time_align: Literal["exact", "nearest", "linear"] = "exact",
    time_tolerance: np.timedelta64 | float | None = None,
    rechunk_to_primary: bool = True,
) -> dict[str, xr.DataArray]:
    """Layer a whole dict of primary fields over the matching fallback fields.

    Parameters
    ----------
    primary, fallback : dict[str, xr.DataArray]
        Regridded fields on the target grid, keyed by ROMS variable name.
    weights : xr.DataArray, optional
        Passed through to :func:`layer_field`.
    on_missing_primary_var : {"error", "fallback"}, optional
        What to do about a variable the fallback has and the primary does not.
        ``"error"`` (default) raises; ``"fallback"`` takes the variable wholly
        from the fallback. Note the latter has no seam, since the choice is
        spatially uniform -- it is strictly safer than layering that variable.
    time_align : {"exact", "nearest", "linear"}, optional
        How to put the fallback on the primary's time axis.
    time_tolerance : np.timedelta64 or float, optional
        Tolerance for ``time_align="nearest"``; see :func:`align_fallback_time`.
    rechunk_to_primary : bool, optional
        Rechunk the fallback to the primary's chunking before combining. On by
        default: the two fields come off different regridders and inherit
        different time chunking from their sources, and combining mismatched
        chunks forces a rechunk in the middle of the graph.

    Returns
    -------
    dict[str, xr.DataArray]
        The combined fields.

    Raises
    ------
    ValueError
        If the two dicts disagree on which variables they carry (subject to
        ``on_missing_primary_var``), or if their horizontal shapes or time axes
        cannot be reconciled.
    """
    _check_variable_sets(primary, fallback, on_missing_primary_var)

    combined: dict[str, xr.DataArray] = {}
    for var_name in primary:
        p = primary[var_name]
        if var_name not in fallback:
            raise ValueError(
                f"The primary source provides {var_name!r} but the fallback source "
                f"does not, so there is nothing to fill its gaps with. Both sources "
                f"must provide every variable the primary does."
            )
        f = fallback[var_name]

        _check_horizontal_shapes(p, f, var_name)

        if "time" in p.dims:
            f = align_fallback_time(
                f,
                p["time"],
                method=time_align,
                tolerance=time_tolerance,
                var_name=var_name,
            )

        if rechunk_to_primary and p.variable.chunks is not None:
            # Match the primary's chunking so the combine is chunk-aligned and
            # dask can run it block by block. Same pattern as the radiation
            # correction factors in SurfaceForcing.
            f = f.chunk({d: c for d, c in p.variable.chunksizes.items() if d in f.dims})

        combined[var_name] = layer_field(p, f, weights=weights)

    # Variables only the fallback has, taken wholesale (see
    # `on_missing_primary_var`); already validated above. They still need the
    # primary's time axis, or the final assign would reindex them to NaN.
    target_time = next((p["time"] for p in primary.values() if "time" in p.dims), None)
    for var_name in fallback:
        if var_name not in combined:
            f = fallback[var_name]
            if target_time is not None and "time" in f.dims:
                f = align_fallback_time(
                    f,
                    target_time,
                    method=time_align,
                    tolerance=time_tolerance,
                    var_name=var_name,
                )
            combined[var_name] = f

    return combined


def _check_variable_sets(
    primary: dict[str, xr.DataArray],
    fallback: dict[str, xr.DataArray],
    on_missing_primary_var: str,
) -> None:
    """Require the two sources to carry the same variables."""
    if on_missing_primary_var not in ON_MISSING_PRIMARY_VAR:
        raise ValueError(
            f"Unknown on_missing_primary_var {on_missing_primary_var!r}; expected "
            f"one of {list(ON_MISSING_PRIMARY_VAR)}."
        )

    missing_in_primary = sorted(set(fallback) - set(primary))
    if missing_in_primary:
        if on_missing_primary_var == "error":
            raise ValueError(
                f"The fallback source provides {missing_in_primary} but the primary "
                f"source does not. Taking those from the fallback everywhere is "
                f"often reasonable -- it is spatially uniform, so it introduces no "
                f"seam -- but say so explicitly with "
                f'on_missing_primary_var="fallback".'
            )
        logging.info(
            "Variables %s come from the fallback source alone; the primary source "
            "does not provide them.",
            missing_in_primary,
        )


def _check_horizontal_shapes(
    primary: xr.DataArray, fallback: xr.DataArray, var_name: str
) -> None:
    """Require matching horizontal sizes, with a message that names the cause.

    ``eta_rho``/``xi_rho`` carry no coordinates, so xarray aligns them
    positionally. That is what we want, but it means a coarse-vs-fine mismatch
    surfaces as an opaque broadcasting error instead of a useful one.
    """
    for dim in ("eta_rho", "xi_rho"):
        if dim in primary.dims and dim in fallback.dims:
            if primary.sizes[dim] != fallback.sizes[dim]:
                raise ValueError(
                    f"The primary and fallback sources produced different {dim} "
                    f"sizes for {var_name!r} ({primary.sizes[dim]} vs "
                    f"{fallback.sizes[dim]}). This means they disagree about "
                    f"`use_coarse_grid`; build both SurfaceForcing objects with "
                    f"the same `coarse_grid_mode` ('always' or 'never')."
                )


def coverage_fraction(field: xr.DataArray, mask: xr.DataArray | None = None) -> float:
    """Fraction of (wet) target points where ``field`` has a value.

    A cheap single-time-step diagnostic for the two failure modes a layered setup
    can hide: 0.0 means the primary regridded to all-NaN (wrong footprint, or a
    coordinate/projection problem), and 1.0 means the primary covers everything --
    so either the fallback is dead weight or extrapolation leaked into the
    primary. :func:`blend_surface_forcing` logs it, and warns at 0.0.

    Parameters
    ----------
    field : xr.DataArray
        A primary field, before any gap filling.
    mask : xr.DataArray, optional
        Wet mask; when given, only wet points are counted.

    Returns
    -------
    float
        Covered fraction in [0, 1].
    """
    if "time" in field.dims:
        field = field.isel(time=0)
    valid = field.notnull()
    if mask is not None:
        valid = valid.where(mask > 0)
    return float(valid.mean().compute())


def compute_blend_weights(
    primary_footprint: xr.DataArray,
    *,
    cell_size_m: tuple[float, float],
    width_km: float,
    profile: Literal["smoothstep", "linear"] = "smoothstep",
) -> xr.DataArray:
    """Ramp the primary's weight from 0 at its footprint edge to 1 inside it.

    The ramp is **one-sided**: it runs inward from the edge through the primary's
    own territory (see the module docstring). Distance is computed with
    ``scipy.ndimage.distance_transform_edt`` on the footprint mask, scaled by
    the grid's mean cell size. That is O(N) -- unlike
    ``roms_tools.setup.utils.min_dist_to_land``, whose every-point-to-every-point
    search would take minutes on a large domain. It is exact for a uniform grid and
    approximate where the spacing varies appreciably, which is well inside the
    tolerance of a smoothing profile.

    Widening the band does not help indefinitely: once the cell adjacent to the seam
    has a weight near zero it is already pure fallback, so widening only pushes the
    transition deeper into the primary's interior, where the primary is the better
    product. Measured across the real CONUS404/ERA5 boundary, the mean seam
    discontinuity improved most at a band of ~150 km and then flattened.

    Parameters
    ----------
    primary_footprint : xr.DataArray
        2D boolean ``(eta_rho, xi_rho)``, True where the primary source has data.
    cell_size_m : tuple[float, float]
        Mean grid spacing ``(dy, dx)`` in metres, along ``eta_rho`` and ``xi_rho``.
    width_km : float
        Distance over which the primary's weight rises from 0 to 1.
    profile : {"smoothstep", "linear"}, optional
        Shape of the ramp. ``"smoothstep"`` (default) is continuous in value and
        first derivative at both ends, so the blend introduces no kink in the
        field's gradient, which shows up in derived quantities like wind stress
        curl.

    Returns
    -------
    xr.DataArray
        Weights in [0, 1] on ``(eta_rho, xi_rho)``, time-invariant.

    Raises
    ------
    ValueError
        If ``width_km <= 0`` or ``profile`` is unknown.
    """
    from scipy.ndimage import distance_transform_edt

    if width_km <= 0:
        raise ValueError(f"`width_km` must be > 0, got {width_km}.")
    if profile not in BLEND_PROFILES:
        raise ValueError(
            f"Unknown blend_profile {profile!r}; expected one of {list(BLEND_PROFILES)}."
        )

    footprint = np.asarray(primary_footprint.values, dtype=bool)

    if footprint.all():
        # No seam anywhere, so nothing to feather. This case must be caught
        # explicitly: with no zeros in the input, `distance_transform_edt` measures
        # to the array border instead, which would paint a spurious feather band
        # around the whole domain.
        logging.info(
            "The primary source covers the entire domain, so blend_width_km has no "
            "effect."
        )
        return xr.ones_like(primary_footprint, dtype=float)

    # Array order is (eta, xi), hence sampling=(dy, dx). Points outside the
    # footprint come back as 0, i.e. weight 0, pure fallback.
    distance_m = distance_transform_edt(footprint, sampling=cell_size_m)
    w = np.clip(distance_m / (width_km * 1000.0), 0.0, 1.0)
    if profile == "smoothstep":
        w = w * w * (3.0 - 2.0 * w)

    weights = xr.DataArray(w, dims=primary_footprint.dims)
    weights.attrs["long_name"] = "primary source weight"
    return weights


def blend_surface_forcing(
    primary: SurfaceForcing | xr.Dataset | str | Path | list[str | Path],
    fallback: SurfaceForcing | xr.Dataset | str | Path | list[str | Path],
    *,
    grid: Grid | xr.Dataset | None = None,
    blend_width_km: float = 0.0,
    blend_profile: Literal["smoothstep", "linear"] = "smoothstep",
    time_align: Literal["exact", "nearest", "linear"] = "exact",
    time_tolerance: np.timedelta64 | float | None = None,
    on_missing_primary_var: Literal["error", "fallback"] = "error",
    fill_value: float | None = 0.0,
) -> xr.Dataset:
    """Layer a regional surface-forcing product over a global one.

    Each target point is taken from ``primary`` where it has data and from
    ``fallback`` everywhere else; with ``blend_width_km`` the two are averaged
    across a band inside the primary's footprint (see the module docstring).

    Parameters
    ----------
    primary : SurfaceForcing, xarray.Dataset, path or list of paths
        The preferred product. NaN marks the points it does not cover. A path may
        be a NetCDF file, a glob pattern, or a list of files (opened lazily with
        ``xarray.open_mfdataset`` in blocks of 24 records), or a single zarr store
        whose path ends in ``.zarr``.
    fallback : SurfaceForcing, xarray.Dataset, path or list of paths
        The product that fills the primary's gaps, in the same forms as
        ``primary``. It must cover every point the primary does not, carry every
        variable the primary does, and have the same grid shape, units and
        resolved ``use_coarse_grid`` (build both with the same explicit
        ``coarse_grid_mode``). Each product applies its own radiation correction.
    grid : Grid or xarray.Dataset, optional
        The ROMS grid the products were built on: a :class:`~roms_tools.Grid`, or
        a dataset with ``pm``, ``pn`` and ``mask_rho`` (and ``mask_coarse`` for
        coarse-grid products). Taken from ``primary`` when that is a
        ``SurfaceForcing``. Required when ``blend_width_km > 0``, to convert the
        band width to grid cells. When available it is also used to check the
        result for NaN at ocean points (see ``fill_value``).
    blend_width_km : float, optional
        Width of the transition, in km. ``0.0`` (default) is a hard edge.
    blend_profile : {"smoothstep", "linear"}, optional
        Shape of the transition when ``blend_width_km > 0``.
    time_align : {"exact", "nearest", "linear"}, optional
        How to put the fallback on the primary's time axis (see
        :func:`align_fallback_time`). With ``"exact"`` (default) the fallback must
        contain every time stamp of the primary; products built by
        ``SurfaceForcing`` for the same times and ``model_reference_date`` do.
        Independently of this, if the two products label radiation differently
        (different ``rad_time - time`` offsets), the fallback's ``swrad`` and
        ``lwrad`` are re-timed onto the primary's convention by linear
        interpolation.
    time_tolerance : np.timedelta64 or float, optional
        Tolerance for ``time_align="nearest"``. A ``np.timedelta64`` is converted
        to the days of a ``SurfaceForcing`` time axis; a float is in days.
    on_missing_primary_var : {"error", "fallback"}, optional
        What to do about a variable only the fallback has: raise (default), or
        take it from the fallback everywhere.
    fill_value : float or None, optional
        Value for the NaNs left after blending, to avoid blow-up of ROMS. When a
        grid is available, only land points are filled, as
        :class:`~roms_tools.SurfaceForcing` does, and a NaN at an ocean point
        raises (when the data are computed, if they are lazy): it means neither
        product reached that point at that time. Without a grid every NaN is
        filled. ``None`` leaves all NaNs in place.

    Returns
    -------
    xarray.Dataset
        On the primary's coordinates (``time`` and ``rad_time`` included), with the
        primary's attributes plus ``layered``, ``fallback_source``,
        ``fallback_correct_radiation``, ``blend_method``, ``blend_width_km``,
        ``blend_profile`` (feathered only) and ``primary_coverage_fraction``.
        Lazy if the inputs are. Write it with :func:`save_blended_surface_forcing`.

    Raises
    ------
    TypeError
        If ``primary`` or ``fallback`` is not a SurfaceForcing, Dataset or path(s).
    ValueError
        If ``blend_width_km`` is negative, or positive with no grid available
        (raised before any data are read); if the products disagree on grid
        shape, units or ``use_coarse_grid``; if a primary variable is missing
        from the fallback, or a fallback-only variable is present and
        ``on_missing_primary_var="error"``; if ``time_align="exact"`` and the
        fallback lacks some of the primary's time stamps; if ``time_align``,
        ``on_missing_primary_var`` or ``blend_profile`` is not a known value; or
        if a grid is available and a NaN is left at an ocean point.
    """
    if blend_width_km < 0:
        raise ValueError(f"`blend_width_km` must be >= 0, got {blend_width_km}.")

    if grid is None and hasattr(primary, "grid"):
        grid = primary.grid  # type: ignore[union-attr]
    if blend_width_km and grid is None:
        raise ValueError(
            "`blend_width_km > 0` needs the grid spacing: pass `grid=` (a roms_tools "
            "Grid, or a dataset with `pm` and `pn`), or give `primary` as a "
            "SurfaceForcing."
        )
    ds_p = _as_dataset(primary, "primary")
    ds_f = _as_dataset(fallback, "fallback")

    _check_dataset_compatibility(ds_p, ds_f)

    p_fields = _data_fields(ds_p)
    f_fields = _data_fields(ds_f)

    ds_f = _align_radiation_time(ds_p, ds_f, f_fields)
    f_fields = _data_fields(ds_f)  # rebuilt: radiation may have been re-timed

    weights = None
    if blend_width_km:
        footprint = _footprint(p_fields)
        weights = compute_blend_weights(
            footprint,
            cell_size_m=_cell_size_m(grid, ds_p),
            width_km=blend_width_km,
            profile=blend_profile,
        )

    mask = _wet_mask(grid, ds_p)
    coverage = coverage_fraction(next(iter(p_fields.values())), mask)
    logging.info(
        "Primary covers %.1f%% of the target %spoints; the remainder comes from the "
        "fallback.",
        100 * coverage,
        "ocean " if mask is not None else "",
    )
    if coverage == 0.0:
        logging.warning(
            "The primary product has no data on this grid, so the result is the "
            "fallback alone. Check that the primary was built on the same grid."
        )

    combined = layer_fields(
        p_fields,
        f_fields,
        weights=weights,
        on_missing_primary_var=on_missing_primary_var,
        time_align=time_align,
        time_tolerance=time_tolerance,
    )

    out = ds_p.drop_vars([v for v in ds_p.data_vars if v in combined]).assign(
        {name: da for name, da in combined.items()}
    )
    if fill_value is not None:
        from roms_tools.setup.utils import fill_nans_over_land

        for name in combined:
            if mask is None:
                out[name] = out[name].fillna(fill_value)
            else:
                out[name] = fill_nans_over_land(
                    out[name],
                    mask,
                    name,
                    fill_value=fill_value,
                    hint=(
                        "Neither product has a value there: check that the fallback "
                        "covers the whole grid and the primary's whole time range."
                    ),
                )
    out.attrs = dict(ds_p.attrs)
    # Describes the primary alone; the blended result has no such NaNs.
    out.attrs.pop("nan_outside_source_footprint", None)
    out.attrs["layered"] = "True"
    out.attrs["fallback_source"] = str(ds_f.attrs.get("source", "unknown"))
    if "correct_radiation" in ds_f.attrs:
        out.attrs["fallback_correct_radiation"] = str(ds_f.attrs["correct_radiation"])
    out.attrs["blend_method"] = "feathered" if blend_width_km else "hard_edge"
    out.attrs["blend_width_km"] = str(float(blend_width_km))
    if blend_width_km:
        out.attrs["blend_profile"] = blend_profile
    out.attrs["primary_coverage_fraction"] = f"{coverage:.4f}"
    return out


def save_blended_surface_forcing(
    ds: xr.Dataset,
    filepath: str | Path,
    *,
    group: bool = True,
    format: NetCDFFormat = DEFAULT_NETCDF_FORMAT,
) -> list[Path]:
    """Write a blended dataset to NetCDF file(s) that ROMS can read.

    Writes the files as :meth:`~roms_tools.SurfaceForcing.save` does. Prefer
    this over ``xarray.Dataset.to_netcdf``: it applies the same
    encodings as ``SurfaceForcing.save``, which ROMS needs: a datetime
    ``abs_time`` taken from an in-memory ``SurfaceForcing`` would otherwise be
    written as a 64-bit integer, which ParallelIO cannot read.

    Parameters
    ----------
    ds : xarray.Dataset
        The output of :func:`blend_surface_forcing`.
    filepath : str or Path
        Base path of the output file(s). A ``.nc`` suffix is dropped. With
        ``group=True`` the files are named ``<filepath>_YYYYMM.nc`` (or
        ``_YYYY.nc``), as ``SurfaceForcing.save`` names them.
    group : bool, optional
        Split the dataset into monthly or yearly files according to its time
        frequency. Defaults to True.
    format : {"NETCDF4", "NETCDF3_CLASSIC", "NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA"}, optional
        NetCDF file format. Defaults to ``"NETCDF4"``. ROMS built with ParallelIO
        needs ``"NETCDF3_64BIT_DATA"`` (CDF5).

    Returns
    -------
    list[Path]
        The files written.
    """
    from roms_tools.setup.utils import group_dataset
    from roms_tools.utils import save_datasets

    filepath = Path(filepath)
    if filepath.suffix == ".nc":
        filepath = filepath.with_suffix("")

    if group:
        dataset_list, output_filenames = group_dataset(ds, str(filepath))
    else:
        dataset_list, output_filenames = [ds], [str(filepath)]

    use_dask = any(v.chunks is not None for v in ds.variables.values())
    return save_datasets(
        dataset_list, output_filenames, use_dask=use_dask, format=format
    )


def _as_dataset(obj: Any, label: str) -> xr.Dataset:
    """Resolve a SurfaceForcing, Dataset or path(s) to a Dataset."""
    if isinstance(obj, xr.Dataset):
        return obj
    if hasattr(obj, "ds") and isinstance(obj.ds, xr.Dataset):
        return obj.ds
    if isinstance(obj, str | Path | list | tuple):
        paths = [obj] if isinstance(obj, str | Path) else list(obj)
        if len(paths) == 1 and str(paths[0]).endswith(".zarr"):
            return xr.open_zarr(paths[0])
        with warnings.catch_warnings():
            # Day-sized blocks rarely line up with a file's stored chunks, which
            # follow its own record count; xarray warns about that on every open,
            # but reading a day at a time is the point.
            warnings.filterwarnings(
                "ignore", message="The specified chunks separate the stored chunks"
            )
            return xr.open_mfdataset(
                [str(p) for p in paths] if len(paths) > 1 else str(paths[0]),
                combine="by_coords",
                decode_timedelta=False,
                # Lazy, and in blocks of a day: one chunk per file would be tens of GB.
                chunks={"time": 24},
            )
    raise TypeError(
        f"`{label}` must be a SurfaceForcing, an xarray.Dataset or path(s), got "
        f"{type(obj).__name__}."
    )


def _data_fields(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    return {name: ds[name] for name in ds.data_vars}


def _check_dataset_compatibility(ds_p: xr.Dataset, ds_f: xr.Dataset) -> None:
    """Raise on mismatches no blend can paper over; log the rest."""
    for name in sorted(set(ds_p.data_vars) & set(ds_f.data_vars)):
        p_units = ds_p[name].attrs.get("units")
        f_units = ds_f[name].attrs.get("units")
        if p_units and f_units and p_units != f_units:
            raise ValueError(
                f"The two products disagree on the units of {name!r}: the primary "
                f"reports {p_units!r} and the fallback {f_units!r}. Both must "
                f"carry the ROMS surface-forcing conventions."
            )
    p_coarse = ds_p.attrs.get("use_coarse_grid")
    f_coarse = ds_f.attrs.get("use_coarse_grid")
    if p_coarse is not None and f_coarse is not None and p_coarse != f_coarse:
        raise ValueError(
            f"The products disagree about `use_coarse_grid` ({p_coarse} vs "
            f"{f_coarse}); build both with the same `coarse_grid_mode`."
        )


def _rad_offset(ds: xr.Dataset) -> np.timedelta64 | None:
    """The ``rad_time - time`` offset of a product, or None if it has no rad_time."""
    if "rad_time" not in ds.coords:
        return None
    diff = (ds["rad_time"] - ds["time"]).isel(time=0).values
    if np.issubdtype(np.asarray(diff).dtype, np.timedelta64):
        return diff
    # Both are in (float) days relative to the model reference date.
    return np.timedelta64(round(float(diff) * 86400.0), "s")


def _align_radiation_time(
    ds_p: xr.Dataset, ds_f: xr.Dataset, f_fields: dict[str, xr.DataArray]
) -> xr.Dataset:
    """Put the fallback's radiation on the primary's ``rad_time`` convention.

    An hourly-mean product labels its flux at the end of the averaging interval and
    carries ``rad_time = time + offset``; an instantaneous one has no offset. The
    blended output has a single ``rad_time`` (the primary's), so if the fallback's
    offset differs its radiation is shifted to match before layering. Applying one
    shift to a field that is half hourly-mean and half instantaneous would be wrong
    over half the domain. Both products carrying the same offset (ERA5 and CONUS404
    are both backward hourly means) makes this a no-op.
    """
    off_p = _rad_offset(ds_p)
    off_f = _rad_offset(ds_f)
    if off_p is None and off_f is None:
        return ds_f
    zero = np.timedelta64(0, "s")
    off_p = zero if off_p is None else off_p
    off_f = zero if off_f is None else off_f
    if off_p == off_f:
        return ds_f

    shift = off_f - off_p
    logging.info(
        "The two products label radiation differently (offsets %s and %s); "
        "shifting the fallback's radiation by %s onto the primary's convention.",
        off_p,
        off_f,
        shift,
    )
    ds_f = ds_f.copy()
    # The flux a product labels at ``time`` is valid at its ``rad_time``. Re-sample
    # the fallback's flux, as a function of the instant it is valid at, onto the
    # primary's ``rad_time`` stamps, then label it with the primary's ``time``.
    f_rad = ds_f["rad_time"] if "rad_time" in ds_f.coords else ds_f["time"]
    p_rad = ds_p["rad_time"] if "rad_time" in ds_p.coords else ds_p["time"]
    for name in RADIATION_VARS:
        if name in f_fields and "time" in f_fields[name].dims:
            # NaN where the primary asks for an instant the fallback does not span;
            # the caller's time range should pad both products equally.
            shifted = f_fields[name].assign_coords(time=f_rad.values)
            resampled = shifted.interp(time=p_rad.values)
            ds_f[name] = resampled.assign_coords(time=ds_p["time"].values)
    return ds_f


def _footprint(p_fields: dict[str, xr.DataArray]) -> xr.DataArray:
    """True where *every* primary field has a value, at the first time step.

    The intersection is the conservative reading: a variable missing at a point
    would fall back anyway, so including the point in the feather band would only
    smooth toward a value the merge does not use.
    """
    footprint = None
    for da in p_fields.values():
        valid = (da.isel(time=0) if "time" in da.dims else da).notnull()
        footprint = valid if footprint is None else (footprint & valid)
    assert footprint is not None
    return footprint.compute()


def _wet_mask(grid: Grid | xr.Dataset | None, ds_p: xr.Dataset) -> xr.DataArray | None:
    """The grid's wet mask on the product's (fine or coarse) rho points, if known."""
    if grid is None:
        return None
    gds = cast("xr.Dataset", grid.ds if hasattr(grid, "ds") else grid)
    if str(ds_p.attrs.get("use_coarse_grid")) == "True":
        if "mask_coarse" not in gds:
            return None
        mask = gds["mask_coarse"].rename(
            {"eta_coarse": "eta_rho", "xi_coarse": "xi_rho"}
        )
    else:
        if "mask_rho" not in gds:
            return None
        mask = gds["mask_rho"]
    for dim in ("eta_rho", "xi_rho"):
        if dim in ds_p.dims and mask.sizes.get(dim) != ds_p.sizes[dim]:
            raise ValueError(
                f"The grid's {dim} size ({mask.sizes.get(dim)}) does not match the "
                f"products' ({ds_p.sizes[dim]}); pass the grid they were built on."
            )
    return mask


def _cell_size_m(
    grid: Grid | xr.Dataset | None, ds_p: xr.Dataset
) -> tuple[float, float]:
    """Mean ``(dy, dx)`` in metres of the grid the product was written on."""
    if grid is None:
        raise ValueError(
            "`blend_width_km > 0` needs the grid spacing: pass `grid=` (a roms_tools "
            "Grid, or a dataset with `pm` and `pn`), or give `primary` as a "
            "SurfaceForcing."
        )
    gds = cast("xr.Dataset", grid.ds if hasattr(grid, "ds") else grid)
    # The coarse grid doubles the spacing.
    factor = 2.0 if str(ds_p.attrs.get("use_coarse_grid")) == "True" else 1.0
    dy = factor * float((1 / gds["pn"]).mean())
    dx = factor * float((1 / gds["pm"]).mean())
    return dy, dx
