"""BGC model abstractions for roms-tools.

This module centralises everything roms-tools needs to know about a biogeochemical
(BGC) model's tracer vocabulary and how to complete a partially-specified tracer
set.  This replaces ``compute_missing_bgc_variables`` in ``setup/utils.py``, which hard-coded MARBL's tracer set inline.

The design intentionally separates two concerns:

* :class:`BGCModel` — the abstract description of a BGC model (its output tracer
  set, the input variables it can interpret, per-variable metadata).
* :class:`BGCMarbl` — the concrete MARBL implementation, including the
  stoichiometric relationships used to derive missing tracers and the constant
  defaults used to fill the remainder.

The public entry point is :meth:`BGCMarbl.process_bgc_fields`, which operates on
one or more *already-built* ``type="bgc"`` :class:`~roms_tools.setup.boundary_forcing.BoundaryForcingSource`
or :class:`~roms_tools.setup.initial_conditions.InitialConditionsSource` objects
(the ``BoundaryForcing``/``InitialConditions`` wrapper classes call this
internally on their own ``.bgc`` list). It makes
**no prioritization decisions**: the caller is responsible for arranging (via each
object's ``use_vars``) that variables do not overlap across files.  It only (1) derives
tracers from the key fields present in each object (``CHL``/``Fe``/``DIC``/``ALK``),
(2) fills any tracer still missing across the union with a constant default (written
into the first object — the values are spatially uniform, so which file is arbitrary),
and (3) warns if the tracer set is still incomplete.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import xarray as xr

from roms_tools.setup.utils import get_tracer_defaults, get_variable_metadata


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

    def process_bgc_fields(self, forcings, filepath=None, serialize_dask=None):
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

    # ------------------------------------------------------------------
    # Object-level completion across one or more forcing objects.
    # ------------------------------------------------------------------
    def process_bgc_fields(self, forcings, filepath=None, serialize_dask=None):
        """Complete the MARBL tracer set across one or more forcing objects.

        Makes **no prioritization decisions** — the caller is responsible (via each
        object's ``use_vars``) for arranging that variables do not overlap across
        files.  This method only:

        1. **Derives** tracers from the key fields present in each object, in place:
           ``Fe``→``Lig``, ``DIC``→``DIC_ALT_CO2``, ``ALK``→``ALK_ALT_CO2``, and
           ``CHL``→ the phytoplankton/zooplankton set (``CHL`` is then dropped). A
           derived tracer is only added where it is not already present.
        2. **Fills** any tracer still missing across the union of objects with its
           constant MARBL default (:func:`~roms_tools.setup.utils.get_tracer_defaults`),
           written into the *first* object (values are spatially uniform, so which
           file receives them is immaterial).
        3. **Warns** if the tracer set is still incomplete.

        Parameters
        ----------
        forcings : forcing object | list of forcing objects
            One or more already-built ``type="bgc"`` forcing objects, modified in
            place.
        filepath : str | Path | list[str | Path] | None
            If given, each (modified) object is saved.  Pass a single path when
            ``forcings`` is a single object, or a list of paths matching the
            objects (one per object, in order).
        serialize_dask : bool, optional
            See :func:`roms_tools.utils.save_datasets`; only relevant when
            ``filepath`` is given. Defaults to ``None``, which resolves to
            ``False`` (the ordinary concurrent write) on each object's own
            :meth:`save` -- pass ``True`` to force the serialized,
            one-task-at-a-time write onto every object instead.

        Returns
        -------
        forcing object | list of forcing objects
            The processed object(s): the single object when one was passed, or the
            list of objects otherwise.
        """
        single = _is_forcing(forcings)
        objs = [forcings] if single else list(forcings)
        if not objs:
            raise ValueError("process_bgc_fields requires at least one forcing object.")

        adapters = [_ForcingBGCAdapter(o, self) for o in objs]

        # 1. Derive tracers from each object's own key fields (no cross-file priority).
        for a in adapters:
            if a.has("Fe") and not a.has("Lig"):
                a.assign_derived("Lig", "Fe", lambda fe: fe * self._FE_TO_LIG)
            if a.has("DIC") and not a.has("DIC_ALT_CO2"):
                a.assign_derived("DIC_ALT_CO2", "DIC", lambda x: x * 1)
            if a.has("ALK") and not a.has("ALK_ALT_CO2"):
                a.assign_derived("ALK_ALT_CO2", "ALK", lambda x: x * 1)
            if a.has("CHL"):
                for var, factor in self._CHL_FACTORS.items():
                    if not a.has(var):
                        a.assign_derived(var, "CHL", lambda x, f=factor: x * f)
                a.drop("CHL")

        # 2. Fill any tracer still missing across the union with its constant default,
        #    into the first object (spatially-uniform, so the choice of file is arbitrary).
        present: set[str] = set()
        for a in adapters:
            present |= a.present_vars()
        defaults = get_tracer_defaults()
        for var in sorted(self.tracer_vars() - present):
            adapters[0].assign_const(var, float(defaults.get(var, 0.0)))
            present.add(var)

        # 3. Completeness check.
        self.warn_missing(present)

        if filepath is not None:
            paths = [filepath] if single else list(filepath)
            if len(paths) != len(objs):
                raise ValueError(
                    "filepath must provide one path per forcing object "
                    f"(got {len(paths)} path(s) for {len(objs)} object(s))."
                )
            for obj, p in zip(objs, paths):
                obj.save(p, serialize_dask=serialize_dask)

        return objs[0] if single else objs


# Name -> class registry for the ``BoundaryForcing``/``InitialConditions`` wrapper
# classes' ``bgc_model=`` field: a `BGCModel` subclass has to round-trip through
# YAML as a plain string (``to_dict``'s generic serialization would otherwise pass
# a raw Python class straight to ``yaml.dump()``, which isn't safe to read back with
# ``yaml.safe_load_all``). Add an entry here for every new `BGCModel` subclass.
_BGC_MODEL_REGISTRY: dict[str, type[BGCModel]] = {"BGCMarbl": BGCMarbl}


def bgc_model_to_name(cls: type[BGCModel] | None) -> str | None:
    """Return the registry name for a ``BGCModel`` subclass, for YAML output."""
    if cls is None:
        return None
    for name, registered in _BGC_MODEL_REGISTRY.items():
        if registered is cls:
            return name
    raise ValueError(
        f"{cls!r} is not in the BGC model registry ({sorted(_BGC_MODEL_REGISTRY)}) "
        "-- add it to `_BGC_MODEL_REGISTRY` in bgc_model.py so it can round-trip "
        "through YAML."
    )


def bgc_model_from_name(name: str | None) -> type[BGCModel] | None:
    """Look up a ``BGCModel`` subclass by its registry name, for YAML input."""
    if name is None:
        return None
    try:
        return _BGC_MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown bgc_model {name!r}; valid options: {sorted(_BGC_MODEL_REGISTRY)}."
        ) from None


def _is_forcing(x) -> bool:
    """A forcing object is anything exposing an xarray ``ds`` attribute."""
    return hasattr(x, "ds")


class _ForcingBGCAdapter:
    """Hide the per-object dataset layout from :class:`BGCMarbl`.

    :class:`~roms_tools.setup.boundary_forcing.BoundaryForcingSource` stores BGC
    tracers suffixed by boundary direction (``PO4_south`` ...), whereas
    :class:`~roms_tools.setup.initial_conditions.InitialConditionsSource` stores
    them with bare names (``PO4``).  This adapter presents a uniform bare-name
    interface for reading, deriving, constant-filling, and dropping tracers.
    """

    def __init__(self, obj, model: BGCModel):
        self.obj = obj
        self.model = model
        self._meta = get_variable_metadata()
        self._known = model.known_vars()
        # BoundaryForcingSource exposes a `boundaries` dict; InitialConditionsSource
        # does not.
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
