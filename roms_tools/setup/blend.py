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

from typing import Any

import numpy as np
import xarray as xr

#: Key on a ``bgc_sources`` item that requests a blend. Unknown keys on an item are
#: ignored by ``build_bgc_companions`` and round-trip through YAML unchanged, so adding
#: this needs no schema change.
BLEND_KEY = "salinity_blend"


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
            f"salinity blend band needs high > low, got low={low!r}, high={high!r}."
        )
    x = ((salinity - low) / (high - low)).clip(min=0.0, max=1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def parse_blend_spec(item: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and normalise one ``bgc_sources`` item's blend spec, or return None."""
    spec = item.get(BLEND_KEY)
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(f"{BLEND_KEY!r} must be a dict, got {type(spec).__name__}.")

    unknown = set(spec) - {"with", "variables", "range"}
    if unknown:
        raise ValueError(
            f"unknown key(s) in {BLEND_KEY}: {sorted(unknown)}; "
            "expected 'with', 'variables', 'range'."
        )
    partner = spec.get("with")
    if not partner:
        raise ValueError(f"{BLEND_KEY} needs 'with': the partner BGC source's name.")
    variables = spec.get("variables")
    if not variables:
        raise ValueError(
            f"{BLEND_KEY} needs 'variables': which tracers to blend. There is no "
            "sensible default -- blending every shared variable would blend alkalinity "
            "and DIC too, which do not need it (they are within 3% of WOA in brackish "
            "water and are never clipped)."
        )
    band = tuple(spec.get("range", (31.0, 34.0)))
    if len(band) != 2:
        raise ValueError(f"{BLEND_KEY} 'range' must be (low, high), got {band!r}.")
    low, high = float(band[0]), float(band[1])
    if not high > low:
        raise ValueError(f"{BLEND_KEY} 'range' needs high > low, got {band!r}.")
    return {"with": str(partner), "variables": list(variables), "range": (low, high)}


def _source_name(obj: Any) -> str | None:
    source = getattr(obj, "source", None)
    return source.get("name") if isinstance(source, dict) else None


def apply_salinity_blends(
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
        The ``bgc_sources`` items, each optionally carrying a ``salinity_blend`` spec.
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

    for obj, item in zip(bgc_objs, bgc_sources):
        spec = parse_blend_spec(item)
        if spec is None:
            continue
        partner = by_name.get(spec["with"])
        if partner is None:
            raise ValueError(
                f"salinity_blend names partner source {spec['with']!r}, which is not "
                f"among the bgc_sources ({sorted(n for n in by_name if n)}). The partner "
                "has to be listed as a bgc source of its own so it is regridded too."
            )
        if partner is obj:
            raise ValueError(f"source {spec['with']!r} cannot blend with itself.")
        low, high = spec["range"]

        for base in spec["variables"]:
            # One entry per dataset variable derived from `base`: the plain name for an
            # initial condition, one per open direction for boundary forcing.
            targets = [
                name
                for name in obj.ds.data_vars
                if name == base or name.startswith(f"{base}_")
            ]
            if not targets:
                raise ValueError(
                    f"salinity_blend variable {base!r} is not in the "
                    f"{_source_name(obj)!r} source's regridded variables: "
                    f"{sorted(obj.ds.data_vars)}."
                )
            for name in targets:
                if name not in partner.ds:
                    raise ValueError(
                        f"salinity_blend needs {name!r} from partner source "
                        f"{spec['with']!r}, which supplies "
                        f"{sorted(partner.ds.data_vars)}. Add it to that source's "
                        "use_vars."
                    )
                weight = raised_cosine_weight(salinity_for(name), low, high)
                primary, secondary = obj.ds[name], partner.ds[name]
                blended = weight * primary + (1.0 - weight) * secondary
                blended.attrs = dict(primary.attrs)
                blended.attrs.update(
                    {
                        "blend_partner": spec["with"],
                        "blend_coordinate": "salinity",
                        "blend_band_psu": f"{low:g}-{high:g}",
                        "blend_weight": "raised cosine on salinity; 0 = partner, 1 = this "
                        "source. Numerical regularisation, not an optimal "
                        "combination -- see roms_tools.setup.blend.",
                    }
                )
                obj.ds[name] = blended

                if verbose:
                    frac = float((weight < 1.0).mean()) * 100.0
                    full = float((weight == 0.0).mean()) * 100.0
                    print(
                        f"  salinity blend {name}: {frac:.2f}% of cells partly from "
                        f"{spec['with']} ({full:.2f}% wholly), band {low:g}-{high:g} PSU"
                    )

                # The partner has handed this variable over; drop it so the downstream
                # merge still sees one source per variable.
                partner.ds = partner.ds.drop_vars(name)
                if getattr(partner, "variable_info_bgc", None):
                    partner.variable_info_bgc.pop(name, None)
