"""BGC model abstractions for roms-tools.

This module contains information that roms-tools needs to know about a biogeochemical
(BGC) model's tracers, including naming, transformations, and how to complete a
partially-specified tracer set.

* :class:`BGCModel` — a proto-class designed to be subclassed by particular
 BGC models.  It establishes what constitutes BGC model knowledge to roms-tools.
* :class:`BGCMarbl` — MARBL tracer and tracer processing for roms-tools,
  including the stoichiometric relationships used to derive missing tracers
  and the constant open-ocean values used to fill the remainder.

The public entry point is :meth:`BGCMarbl.process_bgc_fields`, which operates on
one or more *already-built* ``type="bgc"`` :class:`~roms_tools.setup.boundary_forcing.BoundaryForcingSource`
or :class:`~roms_tools.setup.initial_conditions.InitialConditionsSource` objects
(the ``BoundaryForcing``/``InitialConditions`` wrapper classes call this
internally on their own ``.bgc`` list). It **enforces** — rather than merely
assumes — that variables do not overlap across sources: the caller is
responsible for arranging (via each object's ``use_vars``) that each tracer
comes from exactly one source, and an overlap (after derivation) raises
``ValueError`` naming the tracer and every source that supplies it. It will
(1) derive missing tracers from these fields if present in each source
(``CHL``/``Fe``/``DIC``/``ALK``) and not already supplied by another source,
(2) raise on any post-derivation overlap, (3) fill any tracer still missing
across the union with its constant open-ocean background, and (4) report what
was filled and warn if any tracer had no background value to fall back on.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

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


class BGCModel(ABC):
    """Abstract description of a ROMS biogeochemical model.

    Subclasses describe a particular BGC model (e.g. MARBL) by declaring which
    tracers it writes to ROMS (:meth:`tracer_vars`), which additional input
    variables it can interpret (:meth:`known_vars`), and how to complete a
    partially-specified tracer set (:meth:`process_bgc_fields`).

    Two things are enforced at definition/instantiation time rather than at the
    first call: :meth:`process_bgc_fields` is abstract (``BGCModel()`` itself
    raises ``TypeError``), and every *concrete* subclass must declare a
    non-empty ``_TRACER_VARS`` (checked in ``__init_subclass__``) -- an empty
    tracer set would make the model silently derive and fill nothing.
    Intermediate subclasses that leave :meth:`process_bgc_fields` abstract are
    exempt from the tracer check.
    """

    name: str = "generic"

    # Tracers written to ROMS output. Subclasses override.
    _TRACER_VARS: frozenset[str] = frozenset()
    # Interpretable inputs that are *not* themselves written to output
    # (e.g. total chlorophyll CHL, which is expanded into per-PFT tracers).
    _INTERPRETABLE_INPUTS: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # `__abstractmethods__` is computed by ABCMeta *after* this hook runs, so
        # test the method itself: a subclass that still inherits the abstract
        # `process_bgc_fields` is an intermediate abstract class and may leave
        # the tracer set for its own subclasses to fill in.
        still_abstract = getattr(cls.process_bgc_fields, "__isabstractmethod__", False)
        if not still_abstract and not cls._TRACER_VARS:
            raise TypeError(
                f"{cls.__name__} must declare a non-empty `_TRACER_VARS` frozenset "
                "(the tracers it writes to ROMS); an empty set would make "
                "process_bgc_fields() derive and fill nothing."
            )

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

    @abstractmethod
    def process_bgc_fields(self, forcings):
        """Complete the BGC tracer set across one or more forcing objects.

        Abstract — implemented by concrete model subclasses such as
        :class:`BGCMarbl`.
        """


class BGCMarbl(BGCModel):
    """MARBL biogeochemical model.

    Implements the MARBL tracers and known pre-processing fields, and the
    relationships used to complete a partial tracer set:

    * ``Fe`` → ``Lig`` (ligand = iron * 3)
    * ``DIC`` → ``DIC_ALT_CO2`` and ``ALK`` → ``ALK_ALT_CO2`` (identity copies)
    * ``CHL`` → the full small-phytoplankton / diatom / diazotroph / zooplankton
      tracer set (fixed stoichiometric ratios); ``CHL`` itself is then dropped.
    * constant open-ocean background values for organic-matter tracers that have
      no other source (:attr:`_OCEAN_FILL`).

    MARBL tracers and units, for reference:

    - "PO4": Dissolved Inorganic Phosphate (mmol/m³),
    - "NO3": Dissolved Inorganic Nitrate (mmol/m³),
    - "SiO3": Dissolved Inorganic Silicate (mmol/m³),
    - "NH4": Dissolved Ammonia (mmol/m³),
    - "Fe": Dissolved Inorganic Iron (mmol/m³),
    - "Lig": Iron Binding Ligand (mmol/m³),
    - "O2": Dissolved Oxygen (mmol/m³),
    - "DIC": Dissolved Inorganic Carbon (mmol/m³),
    - "DIC_ALT_CO2": Dissolved Inorganic Carbon, Alternative CO2 (mmol/m³),
    - "ALK": Alkalinity (meq/m³),
    - "ALK_ALT_CO2": Alkalinity, Alternative CO2 (meq/m³),
    - "DOC": Dissolved Organic Carbon (mmol/m³),
    - "DON": Dissolved Organic Nitrogen (mmol/m³),
    - "DOP": Dissolved Organic Phosphorus (mmol/m³),
    - "DOPr": Refractory Dissolved Organic Phosphorus (mmol/m³),
    - "DONr": Refractory Dissolved Organic Nitrogen (mmol/m³),
    - "DOCr": Refractory Dissolved Organic Carbon (mmol/m³),
    - "zooC": Zooplankton Carbon (mmol/m³),
    - "spChl": Small Phytoplankton Chlorophyll (mg/m³),
    - "spC": Small Phytoplankton Carbon (mmol/m³),
    - "spP": Small Phytoplankton Phosphorous (mmol/m³),
    - "spFe": Small Phytoplankton Iron (mmol/m³),
    - "spCaCO3": Small Phytoplankton CaCO3 (mmol/m³),
    - "diatChl": Diatom Chlorophyll (mg/m³),
    - "diatC": Diatom Carbon (mmol/m³),
    - "diatP": Diatom Phosphorus (mmol/m³),
    - "diatFe": Diatom Iron (mmol/m³),
    - "diatSi": Diatom Silicate (mmol/m³),
    - "diazChl": Diazotroph Chlorophyll (mg/m³),
    - "diazC": Diazotroph Carbon (mmol/m³),
    - "diazP": Diazotroph Phosphorus (mmol/m³),
    - "diazFe": Diazotroph Iron (mmol/m³),

    """

    name = "MARBL"

    _TRACER_VARS = frozenset(
        {
            "PO4",
            "NO3",
            "SiO3",
            "NH4",
            "Fe",
            "Lig",
            "O2",
            "DIC",
            "DIC_ALT_CO2",
            "ALK",
            "ALK_ALT_CO2",
            "DOC",
            "DON",
            "DOP",
            "DOPr",
            "DONr",
            "DOCr",
            "spChl",
            "spC",
            "spP",
            "spFe",
            "spCaCO3",
            "diatChl",
            "diatC",
            "diatP",
            "diatFe",
            "diatSi",
            "diazChl",
            "diazC",
            "diazP",
            "diazFe",
            "zooC",
        }
    )
    _INTERPRETABLE_INPUTS = frozenset({"CHL"})

    # CHL → per-PFT tracer stoichiometric factors (multiplicative on total CHL).
    _CHL_FACTORS: ClassVar[dict[str, float]] = {
        "zooC": 1.35,  # mmol m-3
        "spChl": 0.675,  # mg m-3
        "spC": 3.375,  # mmol m-3
        "spP": 0.03,  # mmol m-3
        "spFe": 1.35e-5,  # mmol m-3
        "spCaCO3": 0.0675,  # mmol m-3
        "diatChl": 0.0675,  # mg m-3
        "diatC": 0.2025,  # mmol m-3
        "diatP": 0.02,  # mmol m-3
        "diatFe": 1.35e-6,  # mmol m-3
        "diatSi": 0.0675,  # mmol m-3
        "diazChl": 0.0075,  # mg m-3
        "diazC": 0.0375,  # mmol m-3
        "diazP": 0.01,  # mmol m-3
        "diazFe": 7.5e-7,  # mmol m-3
    }

    # Fe → Lig multiplicative factor.
    _FE_TO_LIG = 3.0

    # Open-ocean background concentrations for tracers that have neither a source
    # nor a parent field to derive from. These mirror the ``(None, factor)`` entries
    # of ``compute_missing_bgc_variables`` in ``setup/utils.py`` -- the ocean-side
    # fill this class replaced -- and are deliberately NOT the values in
    # ``river_tracer_defaults.nc``: those are river-mouth concentrations, so filling
    # an initial-condition or open-boundary field from them put 460 mmol m-3 of DOC
    # (roughly an order of magnitude above open-ocean DOC) uniformly through the
    # whole domain, at every depth.
    _OCEAN_FILL: ClassVar[dict[str, float]] = {
        "NH4": 1e-6,  # mmol m-3
        "DOC": 1e-6,  # mmol m-3
        "DON": 1.0,  # mmol m-3
        "DOP": 0.1,  # mmol m-3
        "DOCr": 1e-6,  # mmol m-3
        "DONr": 0.8,  # mmol m-3
        "DOPr": 0.003,  # mmol m-3
    }

    # ------------------------------------------------------------------
    # Derivation rules — the single definition of the tracer math.
    # ------------------------------------------------------------------
    @classmethod
    def derivation_rules(cls) -> tuple[tuple[str, str, Callable], ...]:
        """Every ``(target, source, transform)`` this model derives, in apply order.

        One definition of the math, read by both the public ``derive_*`` helpers below
        and by :meth:`process_bgc_fields`. Keeping them in step matters: the two used to
        carry independent copies of the same expressions, so a change to one silently
        diverged from the other.

        Each transform takes exactly one ``DataArray`` (the source tracer) and returns
        one, which is what ``_ForcingBGCAdapter.assign_derived`` applies per boundary
        suffix. Nothing here reads a *derived* tracer, so the order only fixes which
        variable appears first in the output.

        Returns
        -------
        tuple[tuple[str, str, Callable], ...]
            ``(target_name, source_name, transform)`` triples.
        """
        rules: list[tuple[str, str, Callable]] = [
            ("Lig", "Fe", lambda fe: fe * cls._FE_TO_LIG),
            # Alternative-CO2 tracers start as identity copies; ROMS/MARBL evolves them
            # separately once the run starts. The `* 1` produces a distinct object
            # rather than aliasing the source.
            ("DIC_ALT_CO2", "DIC", lambda dic: dic * 1),
            ("ALK_ALT_CO2", "ALK", lambda alk: alk * 1),
        ]
        rules += [
            (var, "CHL", lambda chl, f=factor: chl * f)
            for var, factor in cls._CHL_FACTORS.items()
        ]
        return tuple(rules)

    @classmethod
    def _transform_for(cls, target: str) -> Callable:
        """The single-source transform producing ``target``."""
        for name, _source, fn in cls.derivation_rules():
            if name == target:
                return fn
        raise KeyError(f"{cls.__name__} derives no tracer named {target!r}")

    # ------------------------------------------------------------------
    # Pure (dict-level) derivation helpers — dask-safe, individually testable.
    # Thin wrappers over `derivation_rules`, exposed for callers who want one
    # derivation without going through a forcing object.
    # ------------------------------------------------------------------
    @classmethod
    def derive_phytoplankton_from_chl(
        cls, chl: xr.DataArray
    ) -> dict[str, xr.DataArray]:
        """Derive the per-PFT and zooplankton tracers from total chlorophyll.

        Parameters
        ----------
        chl : xr.DataArray
            Total chlorophyll concentration (mg m-3).

        Returns
        -------
        dict[str, xr.DataArray]
            The derived phytoplankton/zooplankton tracers.
        """
        return {
            target: fn(chl)
            for target, source, fn in cls.derivation_rules()
            if source == "CHL"
        }

    @classmethod
    def derive_ligand_from_iron(cls, fe: xr.DataArray) -> xr.DataArray:
        """Derive ligand from iron: ``Lig = Fe * 3`` (mmol m-3)."""
        return cls._transform_for("Lig")(fe)

    @classmethod
    def derive_alt_co2(
        cls, dic: xr.DataArray, alk: xr.DataArray
    ) -> dict[str, xr.DataArray]:
        """Derive the alternative-CO2 tracers as identity copies of DIC/ALK."""
        return {
            "DIC_ALT_CO2": cls._transform_for("DIC_ALT_CO2")(dic),
            "ALK_ALT_CO2": cls._transform_for("ALK_ALT_CO2")(alk),
        }

    # ------------------------------------------------------------------
    # Object-level completion across one or more forcing objects.
    # ------------------------------------------------------------------
    def process_bgc_fields(self, forcings):
        """Complete the MARBL tracer set across one or more forcing objects.

        Enforces that variables do not overlap across sources — the caller is
        responsible (via each object's ``use_vars``) for partitioning sources so
        each tracer comes from exactly one of them; an overlap is a configuration
        error and raises. This method:

        1. **Derives** tracers from the key fields present in each object, in place:
           ``Fe``→``Lig``, ``DIC``→``DIC_ALT_CO2``, ``ALK``→``ALK_ALT_CO2``, and
           ``CHL``→ the phytoplankton/zooplankton set (``CHL`` is then dropped). A
           derived tracer is only added where it is not already present *anywhere*
           across the objects (an explicit tracer supplied by one source suppresses
           derivation of the same tracer from another source's key field).
        2. **Checks for overlap**: after derivation, if any tracer is present in more
           than one object, raises ``ValueError`` naming the tracer and every source
           that supplies it.
        3. **Fills** any tracer still missing across the union of objects with its
           constant open-ocean background (:attr:`_OCEAN_FILL`), written into the
           first object that can supply a same-shaped template to fill from.
           Tracers with no background value defined are set to zero.
        4. **Reports** every constant fill, and warns for any tracer that fell back
           to zero.

        Parameters
        ----------
        forcings : forcing object | list of forcing objects
            One or more already-built ``type="bgc"`` forcing objects, modified in
            place.

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

        # 0. The union of tracers already present *before* any derivation runs,
        #    frozen for the derivation step below: an explicit tracer supplied by
        #    one source suppresses derivation of the same tracer from another
        #    source's key field (e.g. an explicit Lig in source B means source A's
        #    Fe does not also derive a Lig).
        union_present: set[str] = set()
        for a in adapters:
            union_present |= a.present_vars()

        # 1. Derive tracers from each object's own key fields (no cross-file priority
        #    beyond the union check above), from the same rule table the public
        #    derive_* helpers use.
        for a in adapters:
            for target, source, fn in self.derivation_rules():
                if a.has(source) and not a.has(target) and target not in union_present:
                    a.assign_derived(target, source, fn)
            # CHL is an input to the phytoplankton split, not a MARBL tracer itself, so
            # it goes once its derivatives exist -- including when they were already
            # present and nothing was derived above.
            if a.has("CHL"):
                a.drop("CHL")

        # 2. Overlap check, recomputed after derivation/CHL-drop: process_bgc_fields
        #    makes no prioritization decisions, so a tracer present in more than one
        #    object is a caller configuration error, not something to resolve
        #    silently.
        owners: dict[str, list[str]] = {}
        for a in adapters:
            label = a.source_label()
            for var in a.present_vars():
                owners.setdefault(var, []).append(label)
        overlaps = {var: names for var, names in owners.items() if len(names) > 1}
        if overlaps:
            detail = "; ".join(
                f"{var}: {names}" for var, names in sorted(overlaps.items())
            )
            raise ValueError(
                "BGC sourcing overlap -- the following tracer(s) are supplied by "
                f"more than one source: {detail}. Partition the sources with "
                "use_vars so each tracer comes from exactly one source."
            )

        # 3. Fill any tracer still missing across the union with its constant ocean
        #    background, per boundary suffix, into the first object that can supply
        #    a same-shaped fill template for that suffix (spatially-uniform values,
        #    so which object receives them is otherwise immaterial).
        present: set[str] = set(owners)
        filled: list[tuple[str, float]] = []
        unsourced: list[str] = []
        for var in sorted(self.tracer_vars() - present):
            if var in self._OCEAN_FILL:
                value = self._OCEAN_FILL[var]
            else:
                # No source, no parent field, and no ocean background defined for
                # this tracer -- zero is the only defensible cold start, but it is a
                # modelling decision the caller has to be told about.
                value = 0.0
                unsourced.append(var)
            self._fill_constant(adapters, var, float(value))
            filled.append((var, value))
            present.add(var)

        # 4. Report what was filled, then check completeness. A constant fill is a
        #    silent modelling decision otherwise: it produces a valid-looking file
        #    whose tracer is uniform at every depth.
        if filled:
            logging.info(
                "BGC constant fill — %d tracer(s) had no source and were set to a "
                "uniform value:\n  %s",
                len(filled),
                ", ".join(f"{var}={value:g}" for var, value in filled),
            )
        if unsourced:
            logging.warning(
                "BGC sourcing incomplete — %d tracer(s) have no source, no field to "
                "derive from, and no ocean background value, so they were set to "
                "zero:\n  %s\n"
                "Provide a source that supplies these tracers (or a key field such "
                "as CHL/Fe from which they can be derived).",
                len(unsourced),
                ", ".join(unsourced),
            )
        self.warn_missing(present)

        return objs[0] if single else objs

    @staticmethod
    def _fill_constant(
        adapters: list[_ForcingBGCAdapter], var: str, value: float
    ) -> None:
        """Write a constant ``var`` field into the first adapter, per active
        boundary suffix, that can supply a fill template for that suffix.

        Different suffixes of a boundary layout may need to be filled from
        different adapters (e.g. one source only covers a subset of active
        boundaries), so this is resolved independently per suffix rather than
        picking one adapter for the whole call.
        """
        suffixes: list = []
        for a in adapters:
            for s in a.suffixes:
                if s not in suffixes:
                    suffixes.append(s)
        for suffix in suffixes:
            if not any(a.try_assign_const(var, suffix, value) for a in adapters):
                raise ValueError(
                    f"Cannot constant-fill {var!r}"
                    f"{f' for the {suffix} boundary' if suffix is not None else ''}"
                    ": no BGC source has any tracer present to use as a fill "
                    "template."
                )


# Name -> class registry for the ``BoundaryForcing``/``InitialConditions`` wrapper
# classes' ``bgc_model=`` field: a `BGCModel` subclass has to round-trip through
# YAML as a plain string (``to_dict``'s generic serialization would otherwise pass
# a raw Python class straight to ``yaml.dump()``, which isn't safe to read back with
# ``yaml.safe_load_all``). Add an entry here for every new `BGCModel` subclass.
_BGC_MODEL_REGISTRY: dict[str, type[BGCModel]] = {"BGCMarbl": BGCMarbl}


def validate_bgc_model(bgc_model) -> type[BGCModel]:
    """Check that ``bgc_model`` is a concrete :class:`BGCModel` *subclass* (the
    class itself, not an instance) and return it.

    Used by the ``BoundaryForcing``/``InitialConditions`` wrappers before any
    data is loaded, so a wrong ``bgc_model=`` fails immediately with a message
    naming the fix rather than after the physics regrid as an opaque
    ``TypeError`` ("object is not callable" for an instance, or the ABC's own
    instantiation error for the base class).
    """
    if isinstance(bgc_model, BGCModel):
        raise ValueError(
            f"`bgc_model` must be the class itself, not an instance -- pass "
            f"`bgc_model={type(bgc_model).__name__}`, not "
            f"`bgc_model={type(bgc_model).__name__}()`."
        )
    if not (isinstance(bgc_model, type) and issubclass(bgc_model, BGCModel)):
        raise ValueError(
            f"`bgc_model` must be a BGCModel subclass (e.g. `bgc_model=rt.BGCMarbl`); "
            f"got {bgc_model!r}."
        )
    if getattr(bgc_model.process_bgc_fields, "__isabstractmethod__", False):
        raise ValueError(
            f"`bgc_model={bgc_model.__name__}` is abstract (it does not implement "
            "process_bgc_fields); pass a concrete model such as `rt.BGCMarbl`."
        )
    return bgc_model


def bgc_model_to_name(cls: type[BGCModel] | None) -> str | None:
    """Return the registry name for a ``BGCModel`` subclass, for YAML output."""
    if cls is None:
        return None
    validate_bgc_model(cls)
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
    """Hides the per-object dataset layout from :class:`BGCMarbl`.

    This allows a single interface to underlying tracer sources, so that
    we do not need bespoke code to process BoundaryForcingSource and InitialConditionsSource
    objects.
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

    def source_label(self) -> str:
        """Human-readable identifier for this object's BGC source, for error
        messages (overlap reporting): its source name, plus ``use_vars`` when set.
        """
        label = self.obj.source["name"]
        use_vars = getattr(self.obj, "use_vars", None)
        if use_vars:
            label = f"{label}[use_vars={','.join(use_vars)}]"
        return label

    # -- mutations --
    def assign_derived(self, bare: str, src: str, fn: Callable):
        """Write ``bare = fn(src)`` for each active suffix where ``src`` exists."""
        for s in self.suffixes:
            src_name = self._ds_name(src, s)
            if src_name in self.obj.ds.data_vars:
                val = fn(self.obj.ds[src_name]).astype(np.float32).fillna(0.0)
                self._write(bare, s, val)

    def try_assign_const(self, bare: str, suffix, value: float) -> bool:
        """Write a constant ``bare`` field for one ``suffix`` if this adapter has a
        fill template for it (an existing BGC tracer, of that suffix, to shape the
        constant field after). Returns whether the write happened.
        """
        if suffix not in self.suffixes:
            return False
        template = self._template(suffix)
        if template is None:
            return False
        val = xr.full_like(template, value).astype(np.float32)
        self._write(bare, suffix, val)
        return True

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
