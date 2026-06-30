"""Single source of truth for the data-processing method vocabulary.

This leaf module defines the enums, per-method metadata registry, and the
validation/resolution helpers shared by the source-prefill, lateral-regrid-engine,
destination-extrapolation, and BGC vertical-interpolation options. It imports
nothing from ``roms_tools`` so it can be imported freely by ``roms_tools.regrid``,
``roms_tools.datasets.lat_lon_datasets``, and ``roms_tools.setup.utils`` without
introducing an import cycle (``setup.utils`` already imports ``regrid``).

The public constructor surface of the setup classes stays as plain strings (so the
lightweight YAML round-trip is unchanged); these enums are the *internal*
representation, and the :class:`RegridConfig` model is built behind the constructor.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class PrefillMethod(StrEnum):
    """Source-side fill applied to the whole-domain source before regridding."""

    lateral_fill_2d = "2d_lateral_fill"  # AMG Poisson fill; no xESMF
    nearest_neighbor = "nearest_neighbor"  # cheap distance transform; no xESMF
    nearest_s2d = "nearest_s2d"  # xESMF single nearest source point
    inverse_dist = "inverse_dist"  # xESMF inverse-distance-weighted
    creep_fill = "creep_fill"  # xESMF truncated Laplace-style diffusion


class ExtrapMethod(StrEnum):
    """xESMF destination-side extrapolation used on the no-prefill regrid path."""

    inverse_dist = "inverse_dist"  # default when extrapolation is active
    nearest_s2d = "nearest_s2d"
    creep_fill = "creep_fill"


class RegridEngine(StrEnum):
    """Lateral-regrid engine selector (resolved to a "use xESMF" boolean)."""

    auto = "auto"  # xESMF if installed, else scipy
    xesmf = "xesmf"  # force xESMF (error if unavailable)
    scipy = "scipy"  # force scipy interp


class BgcInterpMethod(StrEnum):
    """Vertical interpolation method for BGC tracers."""

    depth = "depth"
    density = "density"
    density_mld = "density_mld"


# ---------------------------------------------------------------------------
# Per-method metadata registry (single source of truth)
# ---------------------------------------------------------------------------


class MethodSpec(BaseModel):
    """Immutable per-method metadata, replacing the previously scattered tables."""

    model_config = ConfigDict(frozen=True)

    requires_xesmf: bool
    allowed_kwargs: frozenset[str]


#: Keyed by the shared *string value* so members that coincide across
#: ``PrefillMethod`` / ``ExtrapMethod`` (``inverse_dist``, ``nearest_s2d``,
#: ``creep_fill``) share one entry. Enum members look up here transparently because
#: ``StrEnum`` instances hash and compare equal to their string value.
METHOD_META: dict[str, MethodSpec] = {
    PrefillMethod.lateral_fill_2d: MethodSpec(
        requires_xesmf=False, allowed_kwargs=frozenset()
    ),
    PrefillMethod.nearest_neighbor: MethodSpec(
        requires_xesmf=False, allowed_kwargs=frozenset()
    ),
    PrefillMethod.nearest_s2d: MethodSpec(
        requires_xesmf=True, allowed_kwargs=frozenset()
    ),
    PrefillMethod.inverse_dist: MethodSpec(
        requires_xesmf=True, allowed_kwargs=frozenset({"num_src_pnts", "dist_exponent"})
    ),
    PrefillMethod.creep_fill: MethodSpec(
        requires_xesmf=True, allowed_kwargs=frozenset({"num_levels"})
    ),
}

# ---------------------------------------------------------------------------
# Constants derived from the registry / enums (single source of truth)
# ---------------------------------------------------------------------------

#: Per-method allowed keys for ``prefill_kwargs`` / ``extrap_kwargs``. A method not
#: listed here (e.g. ``None``) accepts no kwargs.
PREFILL_ALLOWED_KWARGS: dict[str, frozenset[str]] = {
    str(method): spec.allowed_kwargs for method, spec in METHOD_META.items()
}

#: Prefill methods implemented via an xESMF source-on-source regrid (need xESMF).
XESMF_PREFILL_METHODS: frozenset[str] = frozenset(
    str(method) for method, spec in METHOD_META.items() if spec.requires_xesmf
)

#: Valid xESMF destination-extrapolation methods (``extrap_method``).
EXTRAP_METHODS: frozenset[str] = frozenset(str(m) for m in ExtrapMethod)

#: Allowed values for the ``regrid_method`` selector.
REGRID_METHODS: frozenset[str] = frozenset(str(m) for m in RegridEngine)

#: Allowed values for the ``bgc_interpolation_method`` selector (kept as a tuple to
#: preserve the original error-message ordering: depth, density, density_mld).
BGC_INTERPOLATION_METHODS: tuple[str, ...] = tuple(str(m) for m in BgcInterpMethod)


def _xesmf_available() -> bool:
    """Return True if the optional xESMF dependency can be imported."""
    try:
        import xesmf  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Validation / resolution helpers (registry-driven)
# ---------------------------------------------------------------------------


def validate_extrap(extrap_method: str | None, extrap_kwargs: dict | None) -> None:
    """Validate a destination ``extrap_method`` and its keyword arguments.

    Parameters
    ----------
    extrap_method : str or None
        The xESMF destination-extrapolation method. ``None`` is treated as the
        default ``"inverse_dist"``.
    extrap_kwargs : dict or None
        Method-specific keyword arguments, validated against the chosen method
        using the registry (``PREFILL_ALLOWED_KWARGS``).

    Raises
    ------
    ValueError
        If ``extrap_method`` is unrecognized or ``extrap_kwargs`` contains keys
        the chosen method does not accept.
    """
    if extrap_method is not None and extrap_method not in EXTRAP_METHODS:
        allowed = ", ".join(repr(m) for m in sorted(EXTRAP_METHODS))
        raise ValueError(
            f"extrap_method={extrap_method!r} is not supported. Allowed values: "
            f"{allowed} (or None for the default 'inverse_dist')."
        )
    method = extrap_method or ExtrapMethod.inverse_dist
    allowed_keys = PREFILL_ALLOWED_KWARGS.get(method, frozenset())
    extra = set(extrap_kwargs or {}) - allowed_keys
    if extra:
        allowed_keys_str = ", ".join(sorted(allowed_keys)) or "(none)"
        raise ValueError(
            f"extrap_kwargs {sorted(extra)} are not valid for "
            f"extrap_method={str(method)!r}. Allowed keys: {allowed_keys_str}."
        )


def resolve_regrid_engine(regrid_method: str | None, *, xesmf_available: bool) -> bool:
    """Resolve a ``regrid_method`` selector to a boolean "use xESMF" decision.

    Parameters
    ----------
    regrid_method : str or None
        ``None`` or ``"auto"`` -> xESMF if available, else scipy; ``"xesmf"`` ->
        force xESMF (error if unavailable); ``"scipy"`` -> force scipy.
    xesmf_available : bool
        Whether xESMF can be imported.

    Returns
    -------
    bool
        ``True`` to use the xESMF regridder, ``False`` to use scipy ``interp``.

    Raises
    ------
    ValueError
        If ``regrid_method`` is not a recognized value.
    ImportError
        If ``regrid_method="xesmf"`` but xESMF is not installed.
    """
    method = regrid_method or RegridEngine.auto
    if method not in REGRID_METHODS:
        allowed = ", ".join(repr(m) for m in sorted(REGRID_METHODS))
        raise ValueError(
            f"regrid_method={regrid_method!r} is not supported. Allowed values: "
            f"{allowed} (or None for 'auto')."
        )
    if method == RegridEngine.scipy:
        return False
    if method == RegridEngine.xesmf:
        if not xesmf_available:
            raise ImportError(
                "regrid_method='xesmf' requires the optional xESMF dependency, "
                "which is not installed. Install `roms-tools` via conda, or use "
                "regrid_method='scipy' (or 'auto')."
            )
        return True
    # auto
    return xesmf_available


def validate_prefill(
    prefill: str | None,
    prefill_kwargs: dict | None,
    allowed: "frozenset[str | None] | set[str | None]",
    *,
    xesmf_available: bool,
) -> None:
    """Validate a ``prefill`` selection and its keyword arguments.

    Parameters
    ----------
    prefill : str or None
        The requested source-prefill method (``None`` means "no prefill").
    prefill_kwargs : dict or None
        Method-specific keyword arguments to validate against the chosen method.
    allowed : set or frozenset of (str or None)
        The prefill values this class accepts (e.g. ``{None, *PrefillMethod}`` for
        ``BoundaryForcing``; ``{"2d_lateral_fill"}`` for classes that only support
        the legacy fill).
    xesmf_available : bool
        Whether xESMF can be imported. Used to reject xESMF-only prefill methods
        on installs without it.

    Raises
    ------
    ValueError
        If ``prefill`` is not in ``allowed`` or ``prefill_kwargs`` contains keys
        the chosen method does not accept.
    ImportError
        If ``prefill`` is an xESMF-only method but xESMF is not installed.
    """
    if prefill not in allowed:
        allowed_str = ", ".join(
            repr(a) for a in sorted(allowed, key=lambda x: (x is not None, x or ""))
        )
        raise ValueError(
            f"prefill={prefill!r} is not supported here. Allowed values: {allowed_str}."
        )

    if prefill is None:
        if prefill_kwargs:
            raise ValueError(
                "prefill_kwargs were provided but prefill is None (no prefill). "
                "Set a prefill method or drop prefill_kwargs."
            )
        return

    if prefill in XESMF_PREFILL_METHODS and not xesmf_available:
        raise ImportError(
            f"prefill={prefill!r} requires the optional xESMF dependency, which is "
            "not installed. Install `roms-tools` via conda, or use "
            "prefill='nearest_neighbor' (cheap, no xESMF) or prefill='2d_lateral_fill'."
        )

    allowed_keys = PREFILL_ALLOWED_KWARGS.get(prefill, frozenset())
    extra = set(prefill_kwargs or {}) - allowed_keys
    if extra:
        allowed_keys_str = ", ".join(sorted(allowed_keys)) or "(none)"
        raise ValueError(
            f"prefill_kwargs {sorted(extra)} are not valid for prefill={prefill!r}. "
            f"Allowed keys: {allowed_keys_str}."
        )


# ---------------------------------------------------------------------------
# Resolved, validated configuration object (internal)
# ---------------------------------------------------------------------------


class RegridConfig(BaseModel):
    """Validated lateral-regrid configuration with all derived state in one place.

    Built once (e.g. in ``BoundaryForcing.__post_init__``) from the public string
    fields; all cross-field compatibility rules run in the validator, and the
    derived decisions (``use_xesmf``, ``effective_extrap``, ...) are read off this
    object instead of being recomputed inline. Never serialized — the public
    constructor surface stays as plain strings.
    """

    model_config = ConfigDict(frozen=True)

    prefill: PrefillMethod | None = None
    prefill_kwargs: dict | None = None
    regrid_engine: RegridEngine = RegridEngine.auto
    extrap_method: ExtrapMethod | None = None
    extrap_kwargs: dict | None = None
    #: Environment / per-class context (not user configuration).
    xesmf_available: bool = True
    allowed_prefill: frozenset[PrefillMethod] = Field(
        default_factory=lambda: frozenset(PrefillMethod)
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_inputs(cls, data):
        """Validate the raw (string) inputs before enum coercion.

        Running the shared ``validate_*`` / ``resolve_regrid_engine`` helpers here
        (rather than after coercion) preserves their precise, user-facing error
        messages — e.g. an invalid ``regrid_method`` reports against the public
        ``regrid_method`` name, not pydantic's internal ``regrid_engine`` field.
        """
        if isinstance(data, dict):
            xesmf_available = data.get("xesmf_available", True)
            allowed = data.get("allowed_prefill") or frozenset(PrefillMethod)
            # Stringify so an "unsupported prefill" message lists clean values
            # ('2d_lateral_fill', ...) rather than enum reprs.
            validate_prefill(
                data.get("prefill"),
                data.get("prefill_kwargs"),
                {str(m) for m in allowed} | {None},
                xesmf_available=xesmf_available,
            )
            validate_extrap(data.get("extrap_method"), data.get("extrap_kwargs"))
            # Resolve eagerly so a bad/unavailable engine raises now (and reports
            # against the public ``regrid_method`` name).
            resolve_regrid_engine(
                data.get("regrid_engine"), xesmf_available=xesmf_available
            )
        return data

    @property
    def use_xesmf(self) -> bool:
        """Whether the xESMF regridder is used (vs scipy ``interp``)."""
        return resolve_regrid_engine(
            self.regrid_engine, xesmf_available=self.xesmf_available
        )

    @property
    def effective_extrap(self) -> ExtrapMethod:
        """Destination extrapolation, defaulting to ``inverse_dist`` when unset."""
        return self.extrap_method or ExtrapMethod.inverse_dist

    @property
    def user_set_extrap(self) -> bool:
        """Whether the user explicitly set ``extrap_method`` or ``extrap_kwargs``."""
        return self.extrap_method is not None or self.extrap_kwargs is not None

    @property
    def extrap_is_active(self) -> bool:
        """Whether destination extrapolation actually runs.

        Only on the default (no-prefill) xESMF path; a set prefill makes the source
        NaN-free (plain bilinear) and the scipy path pre-fills instead.
        """
        return self.prefill is None and self.use_xesmf

    @property
    def regrid_extrap_method(self) -> str | None:
        """Extrapolation passed to the regridder (``None`` when a prefill is set).

        Returned as a plain ``str`` (not the enum) since it is forwarded straight to
        ``xesmf.Regridder``.
        """
        return None if self.prefill is not None else str(self.effective_extrap)

    @property
    def regrid_extrap_kwargs(self) -> dict | None:
        """Extrapolation kwargs passed to the regridder (``None`` when prefilled)."""
        return None if self.prefill is not None else self.extrap_kwargs
