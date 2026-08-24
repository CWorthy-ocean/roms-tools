"""Combining fields from more than one source on the ROMS grid.

The functions here are deliberately source-agnostic and free of any dependency on
``roms_tools.setup``: they operate on already-regridded
:class:`xarray.DataArray` objects on the target grid, so they can be tested with
synthetic arrays and no I/O.

The layering model is "primary over fallback": wherever the primary source has a
value the primary wins, and wherever it does not -- because the target point lies
outside a limited-extent source's footprint -- the fallback fills in. A NaN in the
primary is the only signal of missing coverage, which is why the merge has to
happen before ``substitute_nans_by_fillvalue`` turns every remaining NaN into
zero.

``weights`` carries the feathered transition across the seam between the two
sources (built by ``SurfaceForcing._compute_blend_weights``). It is designed so
that the weighted form reduces *exactly* to the hard edge when the weights are
zero or one, which keeps the two behaviours on one code path rather than two that
have to be kept in step.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import xarray as xr

# What to do when the primary source lacks a variable the fallback provides.
ON_MISSING_PRIMARY_VAR = ("error", "fallback")

# How to put the fallback onto the primary's time axis.
TIME_ALIGN_METHODS = ("exact", "nearest", "linear")


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
    the weight field says there. That is what makes ``weights`` of exactly 0/1
    reproduce the ``weights=None`` result.
    """
    if weights is None:
        return xr.where(primary.notnull(), primary, fallback, keep_attrs=True)

    w = weights.where(primary.notnull(), 0.0)
    combined = w * primary.fillna(0.0) + (1.0 - w) * fallback
    # A 2D weight field leads the multiplication, so the result comes out with the
    # horizontal dims first. Restore the primary's order so the weighted and
    # unweighted paths are interchangeable for the caller.
    combined = combined.transpose(*primary.dims)
    combined.attrs = dict(primary.attrs)
    return combined


def align_fallback_time(
    fallback: xr.DataArray,
    target_time: xr.DataArray,
    *,
    method: Literal["exact", "nearest", "linear"] = "exact",
    tolerance: np.timedelta64 | None = None,
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
    tolerance : np.timedelta64, optional
        Maximum distance for ``method="nearest"``.
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
                f"last: {missing[-1]}. Widen the fallback's time range, or set "
                f'blend_options={{"time_align": "nearest"}}.'
            )
        return fallback.sel(time=target_time)

    if method == "nearest":
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
    time_tolerance: np.timedelta64 | None = None,
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
    time_tolerance : np.timedelta64, optional
        Tolerance for ``time_align="nearest"``.
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

        if rechunk_to_primary and p.chunks is not None:
            # Match the primary's chunking so the combine is chunk-aligned and
            # dask can run it block by block. Same pattern as the radiation
            # correction factors in SurfaceForcing.
            f = f.chunk({d: c for d, c in p.chunksizes.items() if d in f.dims})

        combined[var_name] = layer_field(p, f, weights=weights)

    # Variables only the fallback has, taken wholesale (see
    # `on_missing_primary_var`); already validated above.
    for var_name in fallback:
        if var_name not in combined:
            combined[var_name] = fallback[var_name]

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
                f"seam -- but say so explicitly with blend_options="
                f'{{"on_missing_primary_var": "fallback"}}.'
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
                    f"`use_coarse_grid`; set `coarse_grid_mode` explicitly to "
                    f"'always' or 'never'."
                )


def coverage_fraction(field: xr.DataArray, mask: xr.DataArray | None = None) -> float:
    """Fraction of (wet) target points where ``field`` has a value.

    A cheap single-time-step diagnostic that catches the two failure modes a
    layered setup fails silently on: 0.0 means the primary regridded to all-NaN
    (wrong footprint, or a coordinate/projection problem), and 1.0 means the
    primary covers everything -- so either the fallback is dead weight or
    extrapolation leaked into the primary.

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
