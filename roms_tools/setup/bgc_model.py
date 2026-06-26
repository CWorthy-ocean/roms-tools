"""BGC model abstractions for roms-tools.

This module centralises everything roms-tools needs to know about a biogeochemical
(BGC) model's tracer vocabulary and how to complete a partially-specified tracer
set.  Previously this knowledge was scattered across ``utils.py`` (the
``derive_*``/``fill_*``/``compute_missing_bgc_variables`` helpers) and
``bgc_source.py`` (the ``_BGC_EXPECTED_OUTPUT`` list and ``BGC_VARIABLE_INFO``).

The design intentionally separates two concerns:

* :class:`BGCModel` — the abstract description of a BGC model (its output tracer
  set, the input variables it can interpret, per-variable metadata).
* :class:`BGCMarbl` — the concrete MARBL implementation, including the
  stoichiometric relationships used to derive missing tracers and the constant
  defaults used to fill the remainder.

The public entry point is :meth:`BGCMarbl.process_bgc_fields`, which operates on
one or more *already-built* ``type="bgc"`` :class:`~roms_tools.setup.boundary_forcing.BoundaryForcing`
or :class:`~roms_tools.setup.initial_conditions.InitialConditions` objects.  Because
the user now produces a separate file per source, completeness is evaluated across
the *union* of all supplied objects, and derived/filled tracers are written into
the object that holds the relevant key fields (``CHL``/``Fe``).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import xarray as xr

from roms_tools.setup.utils import get_variable_metadata


def bgc_variable_info(var_names) -> dict[str, dict]:
    """Generic per-variable metadata for a set of BGC variable names.

    All BGC variables are scalar, rho-point, 3-D fields; only ``ALK`` is
    validated (NaN check) by the forcing classes.  This is model-agnostic, so the
    forcing classes can describe their BGC variables without depending on any
    particular :class:`BGCModel` subclass.
    """
    return {
        var: {
            "location": "rho",
            "is_vector": False,
            "vector_pair": None,
            "is_3d": True,
            "validate": var == "ALK",
        }
        for var in var_names
    }


class BGCModel:
    """Abstract description of a ROMS biogeochemical model.

    Subclasses describe a particular BGC model (e.g. MARBL) by declaring which
    tracers it writes to ROMS (:meth:`tracer_vars`), which additional input
    variables it can interpret (:meth:`known_vars`), and how to complete a
    partially-specified tracer set (:meth:`process_bgc_fields`).
    """

    name: str = "generic"

    # Tracers written to ROMS output. Subclasses override.
    _TRACER_VARS: frozenset[str] = frozenset()
    # Interpretable inputs that are *not* themselves written to output
    # (e.g. total chlorophyll CHL, which is expanded into per-PFT tracers).
    _INTERPRETABLE_INPUTS: frozenset[str] = frozenset()

    def tracer_vars(self) -> frozenset[str]:
        """Return the set of tracer variables written to ROMS output."""
        return self._TRACER_VARS

    def known_vars(self) -> frozenset[str]:
        """Return every variable the model understands.

        This is :meth:`tracer_vars` plus the interpretable inputs (such as
        ``CHL``) that the model can read and expand but never writes verbatim.
        """
        return self._TRACER_VARS | self._INTERPRETABLE_INPUTS

    def variable_info(self) -> dict[str, dict]:
        """Per-variable metadata (grid location, vector flags, validation) for all
        :meth:`known_vars`.
        """
        return bgc_variable_info(self.known_vars())

    def warn_missing(self, present: set[str]) -> None:
        """Warn if any expected output tracers are still absent.

        Parameters
        ----------
        present : set[str]
            The set of tracer names available across all processed objects after
            derivation and default-filling.
        """
        missing = sorted(self.tracer_vars() - set(present))
        if missing:
            logging.warning(
                "BGC sourcing incomplete — %d tracer(s) have no source and will be "
                "absent from the output files:\n  %s\n"
                "Provide a source that supplies these tracers (or a key field such "
                "as CHL/Fe from which they can be derived).",
                len(missing),
                ", ".join(missing),
            )

    def process_bgc_fields(self, forcings, filepath=None):
        """Complete the BGC tracer set across one or more forcing objects.

        Abstract — implemented by concrete model subclasses such as
        :class:`BGCMarbl`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement process_bgc_fields()."
        )


class BGCMarbl(BGCModel):
    """MARBL biogeochemical model.

    Implements the MARBL tracer vocabulary and the relationships used to complete
    a partial tracer set:

    * ``Fe`` → ``Lig`` (ligand = iron × 3)
    * ``DIC`` → ``DIC_ALT_CO2`` and ``ALK`` → ``ALK_ALT_CO2`` (identity copies)
    * ``CHL`` → the full small-phytoplankton / diatom / diazotroph / zooplankton
      tracer set (fixed stoichiometric ratios); ``CHL`` itself is then dropped.
    * constant defaults for organic-matter tracers that have no other source.
    """

    name = "MARBL"

    _TRACER_VARS = frozenset({
        "PO4", "NO3", "SiO3", "NH4", "Fe", "Lig", "O2",
        "DIC", "DIC_ALT_CO2", "ALK", "ALK_ALT_CO2",
        "DOC", "DON", "DOP", "DOPr", "DONr", "DOCr",
        "spChl", "spC", "spP", "spFe", "spCaCO3",
        "diatChl", "diatC", "diatP", "diatFe", "diatSi",
        "diazChl", "diazC", "diazP", "diazFe",
        "zooC",
    })
    _INTERPRETABLE_INPUTS = frozenset({"CHL"})

    # CHL → per-PFT tracer stoichiometric factors (multiplicative on total CHL).
    _CHL_FACTORS: dict[str, float] = {
        "zooC":    1.35,      # mmol m-3
        "spChl":   0.675,     # mg m-3
        "spC":     3.375,     # mmol m-3
        "spP":     0.03,      # mmol m-3
        "spFe":    1.35e-5,   # mmol m-3
        "spCaCO3": 0.0675,    # mmol m-3
        "diatChl": 0.0675,    # mg m-3
        "diatC":   0.2025,    # mmol m-3
        "diatP":   0.02,      # mmol m-3
        "diatFe":  1.35e-6,   # mmol m-3
        "diatSi":  0.0675,    # mmol m-3
        "diazChl": 0.0075,    # mg m-3
        "diazC":   0.0375,    # mmol m-3
        "diazP":   0.01,      # mmol m-3
        "diazFe":  7.5e-7,    # mmol m-3
    }

    # Fe → Lig multiplicative factor.
    _FE_TO_LIG = 3.0

    # Constant defaults for tracers with no other source (mmol m-3).
    _DEFAULTS: dict[str, float] = {
        "NH4":  1e-6,
        "DOC":  1e-6,
        "DON":  1.0,
        "DOP":  0.1,
        "DOCr": 1e-6,
        "DONr": 0.8,
        "DOPr": 0.003,
    }

    # ------------------------------------------------------------------
    # Pure (dict-level) derivation helpers — dask-safe, individually testable.
    # ------------------------------------------------------------------
    @classmethod
    def derive_phytoplankton_from_chl(cls, chl: xr.DataArray) -> dict[str, xr.DataArray]:
        """Derive the per-PFT and zooplankton tracers from total chlorophyll.

        Parameters
        ----------
        chl : xr.DataArray
            Total chlorophyll concentration (mg m-3).

        Returns
        -------
        dict[str, xr.DataArray]
            The 15 derived phytoplankton/zooplankton tracers.
        """
        return {var: chl * factor for var, factor in cls._CHL_FACTORS.items()}

    @classmethod
    def derive_ligand_from_iron(cls, fe: xr.DataArray) -> xr.DataArray:
        """Derive ligand from iron: ``Lig = Fe × 3`` (mmol m-3)."""
        return fe * cls._FE_TO_LIG

    @staticmethod
    def derive_alt_co2(dic: xr.DataArray, alk: xr.DataArray) -> dict[str, xr.DataArray]:
        """Derive the alternative-CO2 tracers as identity copies of DIC/ALK."""
        return {"DIC_ALT_CO2": dic * 1, "ALK_ALT_CO2": alk * 1}

    def default_values(self) -> dict[str, float]:
        """Return the constant-default tracer values used to fill gaps."""
        return dict(self._DEFAULTS)

    # ------------------------------------------------------------------
    # Object-level completion across multiple forcing objects.
    # ------------------------------------------------------------------
    def process_bgc_fields(self, forcings, filepath=None):
        """Complete the MARBL tracer set across a list of forcing objects.

        Completeness is evaluated across the *union* of all supplied objects: a
        tracer is "missing" only if absent from every object.  Derived tracers
        are written into the object that holds their key field (``CHL``/``Fe``/
        ``DIC``/``ALK``); the remaining gaps are constant-filled into the
        *primary* object (the one holding the most key fields).

        The supplied objects must share consistent spatial (and, for initial
        conditions, depth) dimensions — they describe the same ROMS grid.  Time
        axes may differ freely (ROMS interpolates each file independently).

        Cross-file conflicts (a tracer present in more than one object) are
        resolved so each tracer is written exactly once: by default the first
        object in the list keeps it, but a ``(object, [fields])`` entry marks that
        object as the preferred source for those fields (dropped from the others).

        Parameters
        ----------
        forcings : forcing object | (object, [fields]) | list of either
            One or more already-built ``type="bgc"`` forcing objects, modified in
            place.  Wrap an object as ``(object, ["ALK", "DIC"])`` to make it the
            preferred source for those fields.
        filepath : str | Path | list[str | Path] | None
            If given, each (modified) object is saved.  Pass a single path when
            ``forcings`` is a single object, or a list of paths matching the
            objects (one per entry, in order).

        Returns
        -------
        forcing object | list of forcing objects
            The processed object(s): the single object when one was passed, or the
            (unwrapped) list of objects otherwise.
        """
        single = _is_forcing(forcings)
        entries = [forcings] if single else list(forcings)
        if not entries:
            raise ValueError("process_bgc_fields requires at least one forcing object.")
        objs, claims = _parse_forcing_entries(entries)

        _check_spatial_consistency(objs)

        adapters = [_ForcingBGCAdapter(o, self) for o in objs]

        # Resolve cross-file conflicts before union/derivation/write so each
        # tracer ends up in exactly one object (prefer claims win; else first wins).
        _resolve_field_conflicts(adapters, claims)

        # Union of tracer/input names present across all objects (post-dedupe).
        present: set[str] = set()
        for a in adapters:
            present |= a.present_vars()

        # Primary object = the one holding the most key fields.
        def _key_score(a: "_ForcingBGCAdapter") -> int:
            return sum(a.has(k) for k in ("CHL", "Fe"))

        primary = max(adapters, key=_key_score)

        missing = set(self.tracer_vars()) - present

        def _first_with(var: str):
            return next((a for a in adapters if a.has(var)), None)

        # --- Fe → Lig ---
        if "Lig" in missing:
            a = _first_with("Fe")
            if a is not None:
                a.assign_derived("Lig", "Fe", lambda fe: fe * self._FE_TO_LIG)
                missing.discard("Lig"); present.add("Lig")

        # --- DIC → DIC_ALT_CO2 ---
        if "DIC_ALT_CO2" in missing:
            a = _first_with("DIC")
            if a is not None:
                a.assign_derived("DIC_ALT_CO2", "DIC", lambda x: x * 1)
                missing.discard("DIC_ALT_CO2"); present.add("DIC_ALT_CO2")

        # --- ALK → ALK_ALT_CO2 ---
        if "ALK_ALT_CO2" in missing:
            a = _first_with("ALK")
            if a is not None:
                a.assign_derived("ALK_ALT_CO2", "ALK", lambda x: x * 1)
                missing.discard("ALK_ALT_CO2"); present.add("ALK_ALT_CO2")

        # --- CHL → per-PFT tracers (then drop CHL) ---
        a = _first_with("CHL")
        if a is not None:
            for var, factor in self._CHL_FACTORS.items():
                if var in missing:
                    a.assign_derived(var, "CHL", lambda x, f=factor: x * f)
                    missing.discard(var); present.add(var)
            a.drop("CHL")
            present.discard("CHL")

        # --- constant defaults into the primary object ---
        for var in sorted(missing):
            if var in self._DEFAULTS:
                primary.assign_const(var, self._DEFAULTS[var])
                missing.discard(var); present.add(var)

        self.warn_missing(present)

        if filepath is not None:
            paths = [filepath] if single else list(filepath)
            if len(paths) != len(objs):
                raise ValueError(
                    "filepath must provide one path per forcing object "
                    f"(got {len(paths)} path(s) for {len(objs)} object(s))."
                )
            for obj, p in zip(objs, paths):
                obj.save(p)

        return objs[0] if single else objs


# Time-like dimensions are allowed to differ across files; everything else is a
# spatial/depth dimension that must be consistent (same ROMS grid).
_TIME_DIMS = frozenset(
    {"time", "bry_time", "ocean_time", "abs_time", "month", "nv", "ntides"}
)


def _is_forcing(x) -> bool:
    """A forcing object is anything exposing an xarray ``ds`` attribute."""
    return hasattr(x, "ds")


def _parse_forcing_entries(entries):
    """Split entries into parallel ``(objs, claims)`` lists.

    Each entry is either a forcing object or an ``(object, [fields])`` pair.
    ``claims[i]`` is the set of field names that object ``i`` is the preferred
    (authoritative) source for.
    """
    objs, claims = [], []
    for e in entries:
        if _is_forcing(e):
            objs.append(e)
            claims.append(set())
        elif isinstance(e, (tuple, list)) and len(e) == 2 and _is_forcing(e[0]):
            obj, fields = e
            objs.append(obj)
            claims.append(set(fields))
        else:
            raise ValueError(
                "Each forcing entry must be a forcing object or a "
                "(forcing_object, [field, ...]) pair."
            )
    return objs, claims


def _resolve_field_conflicts(adapters, claims) -> None:
    """Drop duplicate tracers so each is written by exactly one object.

    ``claims[i]`` lists fields that object ``i`` is the preferred source for
    (these win any conflict).  Any other tracer present in more than one object
    is kept in the first object (lowest index) and dropped from the rest.
    Mutates the underlying datasets in place.
    """
    present = [a.present_vars() for a in adapters]

    # Explicit prefer claims -> owner index (validate uniqueness and presence).
    claimed_owner: dict[str, int] = {}
    for idx, fields in enumerate(claims):
        for f in fields:
            if f in claimed_owner and claimed_owner[f] != idx:
                raise ValueError(
                    f"Field '{f}' is claimed via `prefer` by more than one forcing object."
                )
            if f not in present[idx]:
                raise ValueError(
                    f"A forcing object is set as the preferred source for '{f}', "
                    f"but that object does not contain '{f}'."
                )
            claimed_owner[f] = idx

    all_fields: set[str] = set().union(*present) if present else set()
    for f in all_fields:
        holders = [i for i, pv in enumerate(present) if f in pv]
        owner = claimed_owner.get(f, holders[0])
        for i in holders:
            if i != owner:
                adapters[i].drop(f)


def _check_spatial_consistency(objs) -> None:
    """Ensure all forcing objects share consistent spatial/depth dimensions.

    Compares the size of every non-time dimension that appears in more than one
    object's dataset and raises if any disagree.  Time axes are intentionally
    *not* checked — ROMS interpolates each output file independently, so the
    files may legitimately carry different time records.
    """
    seen: dict[str, tuple[int, int]] = {}  # dim -> (size, first object index)
    for i, obj in enumerate(objs):
        for dim, size in obj.ds.sizes.items():
            if dim in _TIME_DIMS:
                continue
            size = int(size)
            if dim in seen and seen[dim][0] != size:
                j = seen[dim][1]
                raise ValueError(
                    f"Inconsistent spatial dimension '{dim}' across forcing objects: "
                    f"object {j} has {dim}={seen[dim][0]} but object {i} has {dim}={size}. "
                    "All BGC forcing objects passed to process_bgc_fields() must describe "
                    "the same ROMS grid (and vertical levels for initial conditions)."
                )
            seen.setdefault(dim, (size, i))


class _ForcingBGCAdapter:
    """Hide the per-object dataset layout from :class:`BGCMarbl`.

    :class:`~roms_tools.setup.boundary_forcing.BoundaryForcing` stores BGC tracers
    suffixed by boundary direction (``PO4_south`` ...), whereas
    :class:`~roms_tools.setup.initial_conditions.InitialConditions` stores them with
    bare names (``PO4``).  This adapter presents a uniform bare-name interface for
    reading, deriving, constant-filling, and dropping tracers.
    """

    def __init__(self, obj, model: BGCModel):
        self.obj = obj
        self.model = model
        self._meta = get_variable_metadata()
        self._known = model.known_vars()
        # BoundaryForcing exposes a `boundaries` dict; InitialConditions does not.
        self.is_boundary = hasattr(obj, "boundaries") and isinstance(
            getattr(obj, "boundaries"), dict
        )
        if self.is_boundary:
            self.suffixes = [d for d, on in obj.boundaries.items() if on]
        else:
            self.suffixes = [None]

    # -- name <-> bare-name mapping --
    def _ds_name(self, bare: str, suffix) -> str:
        return f"{bare}_{suffix}" if suffix is not None else bare

    def _bare_name(self, ds_var: str):
        if not self.is_boundary:
            return ds_var
        for s in self.suffixes:
            if ds_var.endswith(f"_{s}"):
                return ds_var[: -(len(s) + 1)]
        return None

    # -- queries --
    def present_vars(self) -> set[str]:
        """Bare BGC variable names present in this object's dataset."""
        out: set[str] = set()
        for v in self.obj.ds.data_vars:
            bare = self._bare_name(v)
            if bare in self._known:
                out.add(bare)
        return out

    def has(self, bare: str) -> bool:
        return any(
            self._ds_name(bare, s) in self.obj.ds.data_vars for s in self.suffixes
        )

    # -- mutations --
    def assign_derived(self, bare: str, src: str, fn: Callable):
        """Write ``bare = fn(src)`` for each active suffix where ``src`` exists."""
        for s in self.suffixes:
            src_name = self._ds_name(src, s)
            if src_name in self.obj.ds.data_vars:
                val = fn(self.obj.ds[src_name]).astype(np.float32).fillna(0.0)
                self._write(bare, s, val)

    def assign_const(self, bare: str, value: float):
        """Write a constant ``bare`` field for each active suffix."""
        for s in self.suffixes:
            template = self._template(s)
            if template is None:
                continue
            val = xr.full_like(template, value).astype(np.float32)
            self._write(bare, s, val)

    def drop(self, bare: str):
        names = [
            self._ds_name(bare, s)
            for s in self.suffixes
            if self._ds_name(bare, s) in self.obj.ds.data_vars
        ]
        if names:
            self.obj.ds = self.obj.ds.drop_vars(names)

    # -- internals --
    def _template(self, suffix):
        """First existing BGC tracer for ``suffix``, used as a fill template."""
        for v in self.obj.ds.data_vars:
            bare = self._bare_name(v)
            if bare in self._known and (
                not self.is_boundary or v.endswith(f"_{suffix}")
            ):
                return self.obj.ds[v]
        return None

    def _write(self, bare: str, suffix, val: xr.DataArray):
        name = self._ds_name(bare, suffix)
        meta = self._meta.get(bare, {})
        long_name = meta.get("long_name", bare)
        if self.is_boundary:
            long_name = f"{suffix}ern boundary {long_name}"
        self.obj.ds[name] = val
        self.obj.ds[name].attrs["long_name"] = long_name
        if "units" in meta:
            self.obj.ds[name].attrs["units"] = meta["units"]
