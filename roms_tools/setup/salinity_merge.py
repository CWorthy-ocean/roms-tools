"""Blend two BGC sources across a salinity transition band.

Why this exists
---------------
ESPER's neural nets are regressions on salinity, temperature and position, fitted to
GLODAP. Where salinity is low -- river plumes, monsoon rainfall, marginal seas -- the
``(S, T)`` combination falls outside the training distribution and the nets extrapolate.
Measured on a 12 km Pacific domain at 2010-01-01, against WOA at the same points:

* median ``|ESPER - WOA|`` nitrate is 22.6 mmol/m3 at ``S < 31``, against 0.9 domain-wide;
* 7.9% of wet cells had nitrate clipped at zero by :mod:`roms_tools.setup.esper`, and the
  pre-clip predictions reach -289 umol/kg;
* in brackish water ESPER runs 2.5x high on phosphate and 2.9x high on silicate.

**Why not weight by uncertainty.** The statistically principled combination of two
estimates is inverse-variance weighting (Gandin 1965; Bretherton, Davis and Fandry 1976),
and ESPER reports an uncertainty. It is not usable as a weight here: in that same
brackish region ESPER's reported sigma is 2.2 -- about a tenth of its actual error -- and
98% of those cells fall outside 1 sigma, against 56% domain-wide. Sigma also saturates
(p25/p50/p99 of 2.12/2.19/2.48), so it barely varies with how wrong the prediction is. An
inverse-variance blend would hand the extrapolated field near-equal weight exactly where
it is worst.

**What this does instead.** A deterministic transition to a second source across a
salinity band, in the spirit of the relaxation zones used at model open boundaries
(Davies 1976; Martinsen and Engedahl 1987). That lineage is a *numerical* justification --
avoid a seam and the adjustment shock it drives -- not a statistical one, and it is
documented as such. The weight is a raised-cosine (Tukey) taper rather than a linear ramp
because it is C1 continuous: a linear ramp leaves a kink in the gradient at each end of
the band, which is the discontinuity the taper exists to avoid.

**Known limitations, deliberately not hidden.** The blend conserves nothing. Blended
tracers are not internally consistent with tracers left unblended -- blending nutrients
but not alkalinity leaves the nutrient:alkalinity relationship altered inside the band.
Both are acceptable in an initial or boundary condition, whose fields are re-equilibrated
by the model, and both are recorded: every blended variable carries the band, the partner
source and the blended fraction in its attrs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr
from xarray.core.utils import is_duck_dask_array

#: Key on a ``bgc_sources`` item that requests (or declines) the merge. Unknown keys on
#: an item are ignored by ``build_bgc_companions`` and round-trip through YAML unchanged,
#: so adding this needs no schema change.
SALINITY_MERGE_KEY = "salinity_based_merge"

#: Source name the merge defaults to as the low-salinity partner.
DEFAULT_PARTNER = "WOA"

#: Source name the merge is applied to by default.
PRIMARY_SOURCE = "ESPER"

#: Variables merged by default. Deliberately the three nutrients ESPER gets badly wrong
#: in brackish water and *not* everything the two sources share:
#:
#: * ``ALK``/``DIC`` are within 3% of WOA at ``S < 31`` and are never clipped, so merging
#:   them would trade a good field for an interpolated climatology;
#: * ``O2`` agrees with WOA to 3% at ``S < 31``, and none of its clipped cells are at
#:   ``S < 33`` at all -- they are the eastern tropical Pacific oxygen minimum zone at
#:   ``S ~ 34.7``, which a salinity band cannot reach and a floor should handle instead.
DEFAULT_MERGE_VARIABLES: tuple[str, ...] = ("NO3", "PO4", "SiO3")

#: Default band in PSU: partner outright below, primary outright above.
DEFAULT_MERGE_BAND: tuple[float, float] = (31.0, 34.0)


def raised_cosine_weight(
    salinity: xr.DataArray, low: float, high: float
) -> xr.DataArray:
    """Weight for the *primary* source: 0 at or below ``low``, 1 at or above ``high``.

    ``0.5 - 0.5*cos(pi*x)`` on the normalised band coordinate ``x`` -- the raised-cosine
    (Tukey) taper. Its derivative vanishes at both ends, so the blended field has a
    continuous gradient across each edge of the band; a linear ramp would not.

    Parameters
    ----------
    salinity : xarray.DataArray
        Salinity on the target grid, broadcastable against the fields being blended.
    low, high : float
        Band edges in PSU. Below ``low`` the partner source is used outright; above
        ``high`` the primary source is used outright.

    Returns
    -------
    xarray.DataArray
        Weights in ``[0, 1]``, NaN wherever ``salinity`` is NaN.
    """
    if not high > low:
        raise ValueError(
            f"salinity merge band needs high > low, got low={low!r}, high={high!r}."
        )
    x = ((salinity - low) / (high - low)).clip(min=0.0, max=1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def resolve_merge_spec(
    item: dict[str, Any], available: set[str]
) -> dict[str, Any] | None:
    """Resolve one ``bgc_sources`` item's merge spec, applying defaults.

    The merge is **on by default for an ESPER source** whenever a usable partner is also
    configured, because using ESPER without one is the case the module docstring
    documents as unreliable. It is never turned on by conjuring a source the caller did
    not ask for: the partner has to be in ``bgc_sources`` already, since it needs to be
    regridded and (for ``WOA``) would otherwise trigger an unrequested download.

    Returns ``None`` when no merge applies. Accepted values of
    ``salinity_based_merge``:

    * absent -- default on for ``ESPER`` if a partner is available, else off;
    * ``False`` -- explicitly off, no warning;
    * a dict -- explicitly on, with any of ``with``/``variables``/``range`` defaulted.

    Parameters
    ----------
    item : dict
        One ``bgc_sources`` entry.
    available : set of str
        Names of every other configured BGC source, i.e. the partners that exist.
    """
    spec = item.get(SALINITY_MERGE_KEY, None)
    is_primary = (item.get("source") or {}).get("name") == PRIMARY_SOURCE

    if spec is False:
        return None
    if spec is None:
        if not is_primary:
            return None
        if DEFAULT_PARTNER not in available:
            logging.warning(
                "%s is configured without a %s BGC source, so the default "
                "salinity-based merge cannot run. ESPER extrapolates unreliably below "
                "about 33 PSU -- measured median error 22.6 mmol/m3 for nitrate against "
                "WOA at S < 31, with nitrate clipped at zero in 7.9%% of wet cells. Add "
                "a %s source carrying %s, or set %s=False to silence this.",
                PRIMARY_SOURCE,
                DEFAULT_PARTNER,
                DEFAULT_PARTNER,
                list(DEFAULT_MERGE_VARIABLES),
                SALINITY_MERGE_KEY,
            )
            return None
        spec = {}
    if not isinstance(spec, dict):
        raise ValueError(
            f"{SALINITY_MERGE_KEY!r} must be a dict or False, got "
            f"{type(spec).__name__}."
        )

    unknown = set(spec) - {"with", "variables", "range"}
    if unknown:
        raise ValueError(
            f"unknown key(s) in {SALINITY_MERGE_KEY}: {sorted(unknown)}; "
            "expected 'with', 'variables', 'range'."
        )
    partner = str(spec.get("with") or DEFAULT_PARTNER)
    variables = list(spec.get("variables") or DEFAULT_MERGE_VARIABLES)
    band = tuple(spec.get("range") or DEFAULT_MERGE_BAND)
    if len(band) != 2:
        raise ValueError(
            f"{SALINITY_MERGE_KEY} 'range' must be (low, high), got {band!r}."
        )
    low, high = float(band[0]), float(band[1])
    if not high > low:
        raise ValueError(
            f"{SALINITY_MERGE_KEY} 'range' needs high > low, got {band!r}."
        )
    return {"with": partner, "variables": variables, "range": (low, high)}


def _time_dim(a: xr.DataArray, b: xr.DataArray) -> str | None:
    """The time dimension the two arrays share, if any."""
    shared = set(a.dims) & set(b.dims)
    return next((d for d in shared if "time" in str(d)), None)


def align_partner_time(partner: xr.DataArray, primary: xr.DataArray) -> xr.DataArray:
    """Put ``partner`` on ``primary``'s time axis, cycling a climatology by time of year.

    Needed because the two sources rarely share a calendar: ESPER boundary output is
    daily, while WOA arrives as a 12-record climatology on mid-month days carrying
    ``cycle_length`` (365.25). Combining them without this is not an error but a silent
    one -- xarray outer-joins the mismatched coordinates and every value comes back NaN.

    A climatology is interpolated linearly in day of year, wrapping December into
    January by padding one record at each end, which is what ROMS itself does at runtime
    with ``cycle_length``. Identical axes are returned untouched.
    """
    # Guard the asymmetric case before anything else. A time axis the partner carries
    # and the primary does not is not merely unaligned -- xarray's arithmetic would
    # broadcast it in silently, giving the merged variable an axis it must never have.
    # For an initial condition (one time) against a 12-month climatology that is a
    # twelvefold blow-up of every merged variable: 2.1 GB becomes 26 GB on a 12 km
    # Pacific grid, which is a write that never finishes rather than an error anyone
    # sees. roms-tools reduces a climatology to the target time before this runs, so
    # in practice the axes do line up -- this is here so that if they ever stop lining
    # up it fails loudly.
    #
    # A length-1 axis is a degenerate leftover and is safe to drop. Anything longer
    # cannot be aligned here, because there is no target time to interpolate onto.
    for extra in [
        d for d in partner.dims if "time" in str(d) and d not in primary.dims
    ]:
        if partner.sizes[extra] == 1:
            partner = partner.squeeze(extra, drop=True)
        else:
            raise ValueError(
                f"the partner source carries a {extra!r} axis of length "
                f"{partner.sizes[extra]} that the primary does not have "
                f"(primary dims: {tuple(primary.dims)}). Merging them would broadcast "
                f"that axis into the result. Reduce the partner to the target time "
                f"first, or give both sources the same time dimension name so it can "
                "be interpolated by time of year."
            )

    dim = _time_dim(partner, primary)
    if dim is None or dim not in partner.coords or dim not in primary.coords:
        return partner  # nothing indexed to align (e.g. the synthetic test arrays)

    target = primary[dim]
    if partner[dim].size == target.size and np.array_equal(
        partner[dim].values, target.values
    ):
        return partner

    cycle = partner[dim].attrs.get("cycle_length")
    if cycle is None:
        raise ValueError(
            f"cannot merge sources on different {dim} axes: the partner has "
            f"{partner[dim].size} records and the primary {target.size}, and the "
            f"partner carries no 'cycle_length' attribute to cycle it by. Give both "
            "sources the same time range, or use a climatological partner."
        )
    cycle = float(cycle)

    # Pad one record at each end so December interpolates into January.
    times = partner[dim].values
    wrapped = xr.concat(
        [
            partner.isel({dim: [-1]}).assign_coords({dim: [times[-1] - cycle]}),
            partner,
            partner.isel({dim: [0]}).assign_coords({dim: [times[0] + cycle]}),
        ],
        dim=dim,
    )
    # The primary's axis is days since the model reference date; modulo the cycle turns
    # it into day of year, which is the climatology's own coordinate.
    day_of_year = np.asarray(target.values, dtype=float) % cycle
    out = wrapped.interp({dim: xr.DataArray(day_of_year, dims=dim)}, method="linear")
    return out.assign_coords({dim: target})


def _source_name(obj: Any) -> str | None:
    source = getattr(obj, "source", None)
    return source.get("name") if isinstance(source, dict) else None


def apply_salinity_based_merge(
    bgc_objs: list[Any],
    bgc_sources: list[dict[str, Any]],
    salinity_for,
    *,
    verbose: bool = True,
) -> None:
    """Blend each item's variables with its partner source, in place.

    Runs after the companions are built (so every source is already on the target grid)
    and before :meth:`BGCMarbl.process_bgc_fields`, which then sees one source per
    variable again -- the blend removes the overlap it needs to.

    Parameters
    ----------
    bgc_objs : list
        The built bgc-only source objects, positionally matching ``bgc_sources``.
    bgc_sources : list of dict
        The ``bgc_sources`` items, each optionally carrying a ``salinity_based_merge`` spec.
    salinity_for : callable
        ``salinity_for(var_name) -> xr.DataArray``: salinity on the target grid, already
        aligned to the variable being blended. Kept as a callable so the per-direction
        boundary case (``salt_south`` for ``NO3_south``) and the single-field initial
        condition case share this code without either leaking into it.
    verbose : bool
        Report the blended fraction per variable.
    """
    if len(bgc_objs) != len(bgc_sources):
        raise ValueError(
            f"{len(bgc_objs)} bgc objects against {len(bgc_sources)} source items."
        )
    by_name = {_source_name(obj): obj for obj in bgc_objs}
    available = {name for name in by_name if name}

    for obj, item in zip(bgc_objs, bgc_sources):
        spec = resolve_merge_spec(item, available - {_source_name(obj)})
        if spec is None:
            continue
        partner = by_name.get(spec["with"])
        if partner is None:
            raise ValueError(
                f"salinity_based_merge names partner source {spec['with']!r}, which is not "
                f"among the bgc_sources ({sorted(n for n in by_name if n)}). The partner "
                "has to be listed as a bgc source of its own so it is regridded too."
            )
        if partner is obj:
            raise ValueError(f"source {spec['with']!r} cannot blend with itself.")
        low, high = spec["range"]
        # An explicitly written spec is held to its word; a defaulted one skips
        # variables either side does not carry rather than failing the build.
        explicit = isinstance(item.get(SALINITY_MERGE_KEY), dict)

        for base in spec["variables"]:
            # One entry per dataset variable derived from `base`: the plain name for an
            # initial condition, one per open direction for boundary forcing.
            targets = [
                name
                for name in obj.ds.data_vars
                if name == base or name.startswith(f"{base}_")
            ]
            if not targets:
                if explicit:
                    raise ValueError(
                        f"{SALINITY_MERGE_KEY} variable {base!r} is not in the "
                        f"{_source_name(obj)!r} source's regridded variables: "
                        f"{sorted(obj.ds.data_vars)}."
                    )
                continue  # defaulted variable this source does not carry
            for name in targets:
                if name not in partner.ds:
                    if not explicit:
                        continue  # defaulted variable the partner does not carry
                    raise ValueError(
                        f"salinity_based_merge needs {name!r} from partner source "
                        f"{spec['with']!r}, which supplies "
                        f"{sorted(partner.ds.data_vars)}. Add it to that source's "
                        "use_vars."
                    )
                weight = raised_cosine_weight(salinity_for(name), low, high)
                primary = obj.ds[name]
                secondary = align_partner_time(partner.ds[name], primary)
                blended = weight * primary + (1.0 - weight) * secondary
                blended.attrs = dict(primary.attrs)
                blended.attrs.update(
                    {
                        "merge_partner": spec["with"],
                        "merge_coordinate": "salinity",
                        "merge_band_psu": f"{low:g}-{high:g}",
                        "merge_weight": "raised cosine on salinity; 0 = partner, 1 = this "
                        "source. Numerical regularisation, not an optimal "
                        "combination -- see roms_tools.setup.salinity_merge.",
                    }
                )
                obj.ds[name] = blended

                if verbose:
                    # Deliberately no statistics on a lazy weight. Reducing it here
                    # (a mean, a count) would force the whole salinity chain through
                    # at construction time -- once per variable per boundary
                    # direction -- ahead of the caller's own single compute at write
                    # time, which is exactly the laziness the rest of this pipeline
                    # is careful to preserve. The share of merged cells is a one-line
                    # diagnostic afterwards, and the band is in the variable's attrs.
                    share = ""
                    if not is_duck_dask_array(getattr(weight, "data", None)):
                        partly = float((weight < 1.0).mean()) * 100.0
                        wholly = float((weight == 0.0).mean()) * 100.0
                        share = f": {partly:.2f}% partly merged, {wholly:.2f}% wholly"
                    print(
                        f"  salinity merge {name} <- {spec['with']} over "
                        f"{low:g}-{high:g} PSU{share}"
                    )

                # The partner has handed this variable over; drop it so the downstream
                # merge still sees one source per variable.
                partner.ds = partner.ds.drop_vars(name)
                if getattr(partner, "variable_info_bgc", None):
                    partner.variable_info_bgc.pop(name, None)
