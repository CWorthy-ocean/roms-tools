"""What each forcing product needs from a source dataset.

A *context* names one place a source dataset is used, for example the physics half of
initial conditions or topography. Each context lists the variable roles it requires and
the kinds of dataset it can read. Whether a catalog entry serves a context is worked out
from these rules and the entry's own metadata, not stored per entry; an entry may only
*restrict* the result through its ``contexts`` metadata key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LATLON = "latlon"
ROMS = "roms"


@dataclass(frozen=True)
class ContextSpec:
    """Requirements of one context."""

    required: tuple[str, ...]
    """Variable roles that must resolve in the dataset."""
    grids: tuple[str, ...]
    """Dataset kinds the context can read: ``"latlon"`` and/or ``"roms"``."""
    optional: tuple[str, ...] = ()
    """Variable roles used when present."""
    needs_depth: bool = False
    """Whether a vertical axis is required (lat/lon datasets only)."""


_PHYSICS = ("temp", "salt", "u", "v", "zeta")

CONTEXTS: dict[str, ContextSpec] = {
    "topography": ContextSpec(required=("topo",), grids=(LATLON,)),
    "ic_physics": ContextSpec(
        required=_PHYSICS, grids=(LATLON, ROMS), needs_depth=True
    ),
    "ic_bgc": ContextSpec(required=(), grids=(ROMS,)),
    "bc_physics": ContextSpec(required=_PHYSICS, grids=(LATLON,), needs_depth=True),
}


def grid_kind(meta: dict[str, Any]) -> str:
    """Return ``"roms"`` for ROMS model output entries and ``"latlon"`` otherwise."""
    return ROMS if meta.get("model") == "roms" else LATLON


def qualifies(meta: dict[str, Any], context: str) -> bool:
    """Whether an entry can serve ``context`` judging by its metadata alone.

    Checks the dataset kind, the vertical axis where the context needs one, and the
    entry's optional ``contexts`` restriction. Whether the required variable roles
    resolve can only be known once the data is open; see
    :func:`roms_tools.datasets.resolve.resolve_names`.

    Parameters
    ----------
    meta : dict
        The entry's metadata.
    context : str
        A key of :data:`CONTEXTS`.

    Returns
    -------
    bool
        True if the entry is eligible for the context.

    Raises
    ------
    KeyError
        If ``context`` is not a known context.
    """
    spec = CONTEXTS[context]
    if grid_kind(meta) not in spec.grids:
        return False
    if (
        grid_kind(meta) == LATLON
        and spec.needs_depth
        and "Z" not in meta.get("axes", {})
    ):
        return False
    restriction = meta.get("contexts")
    return restriction is None or context in restriction
