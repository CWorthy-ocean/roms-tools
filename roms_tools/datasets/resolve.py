"""Resolve variable roles and dimension names for a catalog entry.

The consumers of a source dataset index it through ``var_names`` and ``dim_names``,
which map a role (``"temp"``, ``"topo"``, ``"longitude"``) to the name used in the file.
Subclasses used to hard-code those maps. Here they are worked out from the data itself:

* variable roles are matched against ``vocabulary.json``, a role-keyed dictionary of
  anchored regular expressions over variable names and CF ``standard_name`` attributes,
  applied through cf-xarray's ``custom_criteria``;
* dimension names come from the entry's ``axes`` metadata, which every catalog entry
  written by ocean-skill carries.

Nothing is renamed: the dataset keeps its native names.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from typing import Any

import xarray as xr

from roms_tools.datasets.contexts import CONTEXTS, LATLON, grid_kind
from roms_tools.datasets.transforms import Names

_AXIS_ROLES = {"X": "longitude", "Y": "latitude", "Z": "depth", "T": "time"}


class RoleResolutionError(ValueError):
    """A variable role matched no variable, or more than one."""


def load_vocabulary() -> dict[str, dict[str, str]]:
    """Read the packaged role vocabulary.

    Returns
    -------
    dict
        ``{role: {attribute: anchored regex}}``, the shape ``cf_pandas.Vocab`` saves
        and ``cf_xarray.set_options(custom_criteria=...)`` accepts.

    Raises
    ------
    ValueError
        If a pattern is not anchored with ``^`` and ``$``. cf-xarray matches from the
        start of a name, so an unanchored ``z`` would also match ``zeta`` and ``zos``.
    """
    text = resources.files("roms_tools.datasets").joinpath("vocabulary.json")
    vocab = json.loads(text.read_text(encoding="utf-8"))
    for role, criteria in vocab.items():
        for attr, pattern in criteria.items():
            if not (pattern.startswith("^") and pattern.endswith("$")):
                msg = f"Vocabulary pattern for {role!r}/{attr!r} must be anchored: {pattern!r}"
                raise ValueError(msg)
    return vocab


def resolve_roles(
    ds: xr.Dataset,
    roles: tuple[str, ...] | list[str],
    vocabulary: dict[str, dict[str, str]] | None = None,
    *,
    required: bool = True,
) -> dict[str, str]:
    """Find the variable that plays each role.

    Parameters
    ----------
    ds : xr.Dataset
        The opened dataset.
    roles : sequence of str
        Roles to look for, keys of the vocabulary.
    vocabulary : dict, optional
        Overrides the packaged vocabulary.
    required : bool, optional
        If True (default) a role with no match raises; if False it is left out.

    Returns
    -------
    dict[str, str]
        Role to variable name.

    Raises
    ------
    RoleResolutionError
        If a required role matches nothing, or any role matches several variables.
    """
    import cf_xarray

    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    found: dict[str, str] = {}
    with cf_xarray.set_options(custom_criteria=vocab):
        for role in roles:
            try:
                found[role] = ds.cf[role].name
            except KeyError as err:
                message = str(err)
                if "multiple" in message.lower():
                    listing = re.search(r"\[(.*?)\]", message)
                    names = (
                        re.findall(r"'([^']+)'", listing.group(1)) if listing else []
                    )
                    raise RoleResolutionError(
                        f"Role {role!r} matches several variables ({names}); "
                        "add a `roles` override to the catalog entry."
                    ) from err
                if required:
                    raise RoleResolutionError(
                        f"No variable in the dataset plays the role {role!r}."
                    ) from err
    return found


def dim_names_from_axes(meta: dict[str, Any]) -> dict[str, str]:
    """Map the entry's ``axes`` (``X``, ``Y``, ``Z``, ``T``) to ROMS-Tools dim roles.

    Raises
    ------
    ValueError
        If the entry has no ``axes`` metadata.
    """
    axes = meta.get("axes")
    if not axes:
        raise ValueError(
            "Catalog entry has no `axes` metadata; cannot name dimensions."
        )
    return {_AXIS_ROLES[k]: v for k, v in axes.items() if k in _AXIS_ROLES}


def resolve_names(ds: xr.Dataset, meta: dict[str, Any], context: str) -> Names:
    """Work out the name maps for a lat/lon entry in a given context.

    Parameters
    ----------
    ds : xr.Dataset
        The opened dataset.
    meta : dict
        The entry's metadata. An optional ``roles`` mapping overrides the vocabulary
        for the roles it names.
    context : str
        A key of :data:`~roms_tools.datasets.contexts.CONTEXTS`.

    Returns
    -------
    Names
        ``var_names`` for the required roles, ``opt_var_names`` for optional roles that
        resolved, and ``dim_names`` from the entry's axes.
    """
    if grid_kind(meta) != LATLON:
        raise ValueError("resolve_names applies to lat/lon entries only.")
    spec = CONTEXTS[context]
    overrides = dict(meta.get("roles", {}))
    to_find = [r for r in spec.required if r not in overrides]
    var_names = {
        **resolve_roles(ds, to_find),
        **{r: overrides[r] for r in spec.required if r in overrides},
    }
    var_names = {r: var_names[r] for r in spec.required}
    opt_var_names = resolve_roles(ds, spec.optional, required=False)
    return Names(
        var_names=var_names,
        dim_names=dim_names_from_axes(meta),
        opt_var_names=opt_var_names,
    )
