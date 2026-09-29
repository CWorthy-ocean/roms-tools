"""Build source datasets from intake catalog entries instead of per-source subclasses.

A catalog entry (see ``catalogs/roms_tools.yaml``) says where a dataset is and how to
open it, using the metadata contract ocean-skill's entries already carry: ``axes``,
``featureType``, ``model``, ``transforms``. Everything ROMS-Tools needs beyond that is
worked out from the data:

* which variable plays which role, through ``vocabulary.json`` (:mod:`.resolve`);
* which forcing products the entry can serve, from :mod:`.contexts`;
* how to fix up the raw dataset, through the transforms the entry names.

:func:`from_catalog` returns an ordinary :class:`LatLonDataset` or :class:`ROMSDataset`,
built with ``from_dataset`` so the reader owns loading and the class keeps every
processing step. No per-source subclass is involved.

The catalog path is opt-in; see :func:`use_catalog` and :func:`enabled`. intake and
cf-xarray are imported only when it is used.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from difflib import get_close_matches
from functools import lru_cache
from importlib import resources
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import xarray as xr

from roms_tools.datasets.contexts import CONTEXTS, ROMS, grid_kind, qualifies
from roms_tools.datasets.lat_lon_datasets import (
    _DEFAULT_LAT_LON_LATERAL_CHUNK,
    LatLonDataset,
)
from roms_tools.datasets.resolve import resolve_names
from roms_tools.datasets.roms_dataset import (
    _DEFAULT_ROMS_LATERAL_DASK_CHUNK,
    ROMSDataset,
)
from roms_tools.datasets.transforms import load_transform
from roms_tools.utils import (
    _get_ds_combine_base_params,
    apply_initial_slice,
    finalize_loaded_dataset,
    get_dask_chunks,
    get_pkg_error_msg,
)

# Set to ``1`` to route supported sources through the catalog.
ENABLE_ENV = "ROMS_TOOLS_USE_CATALOG"
# ``os.pathsep``-separated directories of extra catalogs; later entries shadow earlier.
CATALOGS_ENV = "ROMS_TOOLS_CATALOGS"
# ocean-skill's catalog variable, read too so its user catalogs are visible.
SHARED_CATALOGS_ENV = "OCEAN_SKILL_CATALOGS"

_ROMS_DIM_NAMES = {"eta_rho": "eta_rho", "xi_rho": "xi_rho", "time": "time"}
_OCEAN_SKILL_COPERNICUS = "ocean_skill.readers:CopernicusMarineReader"
_LOCAL_COPERNICUS = "roms_tools.datasets.readers:CopernicusMarineReader"
_CHUNKED_READERS = ("XArrayDatasetReader", "PoochReader")

_forced: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "roms_tools_use_catalog", default=None
)


def enabled() -> bool:
    """Whether supported sources are currently built from the catalog.

    A :func:`use_catalog` block wins; otherwise the ``ROMS_TOOLS_USE_CATALOG``
    environment variable decides.
    """
    forced = _forced.get()
    if forced is not None:
        return forced
    return os.environ.get(ENABLE_ENV, "").lower() in {"1", "true", "yes"}


@contextlib.contextmanager
def use_catalog(on: bool = True) -> Iterator[None]:
    """Route supported sources through the catalog (or not) inside a ``with`` block."""
    token = _forced.set(on)
    try:
        yield
    finally:
        _forced.reset(token)


def _require_intake():
    try:
        import cf_xarray  # noqa: F401
        import intake
    except ImportError as err:
        raise RuntimeError(
            get_pkg_error_msg(
                "catalog-driven datasets", "intake and cf_xarray", "catalog"
            )
        ) from err
    return intake


def search_paths() -> list[Path]:
    """Catalog directories, lowest to highest precedence.

    The packaged catalogs, then ``$OCEAN_SKILL_CATALOGS``, then ``$ROMS_TOOLS_CATALOGS``.
    Only directories that exist are returned.
    """
    paths = [Path(str(resources.files("roms_tools.datasets") / "catalogs"))]
    for var in (SHARED_CATALOGS_ENV, CATALOGS_ENV):
        paths.extend(
            Path(p).expanduser() for p in os.environ.get(var, "").split(os.pathsep) if p
        )
    seen: set[Path] = set()
    unique = []
    for p in reversed(paths):  # keep each directory's highest-precedence position
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return [p for p in reversed(unique) if p.is_dir()]


@dataclass(frozen=True)
class Ref:
    """Where an alias lives."""

    alias: str
    entry: str
    path: Path
    meta: dict[str, Any]


@lru_cache(maxsize=32)
def _load(path: str, mtime_ns: int, size: int):
    """Parse one catalog file; keyed on mtime and size so edits are picked up."""
    return _require_intake().from_yaml_file(path)


def _open(path: Path):
    st = path.stat()
    cat = _load(str(path), st.st_mtime_ns, st.st_size)
    if find_spec("ocean_skill") is None:
        # An entry written by ocean-skill names its Copernicus reader; use our copy.
        for desc in cat.entries.values():
            if desc.reader == _OCEAN_SKILL_COPERNICUS:
                desc.reader = _LOCAL_COPERNICUS
    return cat


def discover() -> dict[str, Ref]:
    """Index every alias in every catalog file on the search path.

    Later files shadow earlier ones by alias. Files that fail to parse are skipped with
    a warning. Entries are not instantiated, so no reader is imported and nothing is
    opened.
    """
    index: dict[str, Ref] = {}
    for directory in search_paths():
        for path in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
            try:
                cat = _open(path)
            except Exception as err:
                logging.warning("Skipping unreadable catalog %s: %s", path, err)
                continue
            for alias, entry in cat.aliases.items():
                if entry in cat.entries:
                    index[alias] = Ref(
                        alias, entry, path, dict(cat.entries[entry].metadata)
                    )
    return index


def _alias(name: str, variant: str) -> str:
    return name if variant == "external" else f"{name}_{variant}"


def resolve(name: str, context: str, variant: str = "external") -> Ref:
    """Find the entry for a source name and check it can serve ``context``.

    Parameters
    ----------
    name : str
        ``source["name"]``, e.g. ``"SRTM15"``.
    context : str
        A key of :data:`~roms_tools.datasets.contexts.CONTEXTS`.
    variant : str, optional
        ``"external"`` (user-supplied files) or ``"default"`` (the entry aliased
        ``<name>_default``, e.g. streaming from Copernicus Marine).

    Raises
    ------
    KeyError
        If no such alias exists.
    ValueError
        If the entry cannot serve the context.
    """
    alias = _alias(name, variant)
    index = discover()
    if alias not in index:
        hint = get_close_matches(alias, index, n=3)
        raise KeyError(
            f"No catalog entry {alias!r}." + (f" Did you mean {hint}?" if hint else "")
        )
    ref = index[alias]
    if not qualifies(ref.meta, context):
        raise ValueError(f"Catalog entry {alias!r} cannot serve context {context!r}.")
    return ref


def has(name: str, context: str, variant: str = "external") -> bool:
    """Whether a catalog entry exists for ``name`` that can serve ``context``."""
    try:
        resolve(name, context, variant)
    except (KeyError, ValueError):
        return False
    return True


def _as_url(path: Any) -> Any:
    if isinstance(path, list | tuple):
        return [str(p) for p in path]
    return str(path)


def _file_kwargs(url: Any, time_dim: str, *, force_nested: bool) -> dict[str, Any]:
    """Reader kwargs for several files: intake uses ``open_mfdataset`` for lists/globs."""
    if isinstance(url, list) and len(url) == 1:
        return {}  # intake opens a one-element list with open_dataset; combine kwargs fail
    if isinstance(url, list):
        return {
            **_get_ds_combine_base_params(),
            "combine": "nested",
            "concat_dim": time_dim,
        }
    if any(c in url for c in "*?[]"):
        combine = (
            {"combine": "nested", "concat_dim": time_dim}
            if force_nested
            else {"combine": "by_coords"}
        )
        return {**_get_ds_combine_base_params(), **combine}
    return {}


def from_catalog(
    name: str,
    source: dict[str, Any],
    context: str,
    *,
    variant: str = "external",
    **runtime: Any,
) -> LatLonDataset | ROMSDataset:
    """Build a source dataset object from its catalog entry.

    Parameters
    ----------
    name : str
        ``source["name"]``.
    source : dict
        The user's ``source`` dictionary. ``path`` fills the entry's template; ``grid``
        supplies the ROMS grid for ``model: roms`` entries.
    context : str
        The forcing product, a key of :data:`~roms_tools.datasets.contexts.CONTEXTS`.
    variant : str, optional
        ``"external"`` or ``"default"``; see :func:`resolve`.
    **runtime
        Constructor fields the forcing class supplies: ``start_time``, ``end_time``,
        ``allow_flex_time``, ``use_dask``, ``chunks``, ``initial_slice_bounds``,
        ``climatology``; for ROMS output also ``var_names``,
        ``adjust_depth_for_sea_surface_height`` and ``model_reference_date``.

    Returns
    -------
    LatLonDataset or ROMSDataset
        Built through ``from_dataset``; behaves exactly like the class-built object.
    """
    ref = resolve(name, context, variant)
    meta = ref.meta
    cat = _open(ref.path)

    if "path" in cat.user_parameters and ref.entry in _template_entries(cat):
        path = source.get("path")
        if not path:
            raise ValueError(f"Catalog entry {ref.alias!r} needs source['path'].")
        url = _as_url(path)
        cat = cat(path=url)
    else:
        url = None

    entry = cat[ref.entry]
    use_dask = bool(runtime.get("use_dask", False))
    is_roms = grid_kind(meta) == ROMS
    is_cmems = "dataset_id" in meta
    kind = type(entry).__name__
    read_kwargs: dict[str, Any] = {}

    if is_cmems:
        start = runtime.get("start_time")
        end = runtime.get("end_time")
        read_kwargs.update(
            start_datetime=start,
            end_datetime=end if end is not None else start,
            coordinates_selection_method="outside",
            chunk_size_limit=-1,
        )
    elif kind in _CHUNKED_READERS:
        if use_dask:
            chunks = runtime.get("chunks")
            if chunks is None:
                dims = _ROMS_DIM_NAMES if is_roms else _axis_dims(meta)
                lateral = (
                    _DEFAULT_ROMS_LATERAL_DASK_CHUNK
                    if is_roms
                    else _DEFAULT_LAT_LON_LATERAL_CHUNK
                )
                chunks = get_dask_chunks(dims, lateral_chunk=lateral)
            read_kwargs["chunks"] = chunks
        else:
            read_kwargs["chunks"] = None
        if url is not None:
            time_dim = (
                _ROMS_DIM_NAMES["time"]
                if is_roms
                else meta.get("axes", {}).get("T", "time")
            )
            read_kwargs.update(_file_kwargs(url, time_dim, force_nested=is_roms))

    ds = entry(**read_kwargs).read()

    if is_roms:
        return _build_roms(ds, meta, source, runtime)
    return _build_lat_lon(
        ds, meta, context, runtime, is_cmems=is_cmems, use_dask=use_dask
    )


def _template_entries(cat) -> set[str]:
    """Entries whose data URL is the ``{path}`` placeholder."""
    templated = set()
    for key, desc in cat.entries.items():
        data_refs = [
            v
            for v in desc.kwargs.values()
            if isinstance(v, str) and v.startswith("{data(")
        ]
        for ref in data_refs:
            data_key = ref[len("{data(") : -2]
            data = cat.data.get(data_key)
            if data is not None and "{path}" in str(data.kwargs.get("url", "")):
                templated.add(key)
    return templated


def _axis_dims(meta: dict[str, Any]) -> dict[str, str]:
    from roms_tools.datasets.resolve import dim_names_from_axes

    return dim_names_from_axes(meta)


def _build_lat_lon(
    ds: xr.Dataset,
    meta: dict[str, Any],
    context: str,
    runtime: dict[str, Any],
    *,
    is_cmems: bool,
    use_dask: bool,
) -> LatLonDataset:
    names = resolve_names(ds, meta, context)
    if use_dask and not is_cmems:
        # Same lazy region selection the class path applies while opening with dask.
        ds = apply_initial_slice(ds, runtime.get("initial_slice_bounds"))
    ds = finalize_loaded_dataset(ds, names.dim_names, read_zarr=is_cmems)

    transforms = meta.get("transforms", {})
    fields: dict[str, Any] = {
        "var_names": names.var_names,
        "dim_names": names.dim_names,
        "opt_var_names": names.opt_var_names,
        "needs_lateral_fill": bool(meta.get("land_masked", True)),
        "climatology": bool(runtime.get("climatology", meta.get("climatology", False))),
        "clean_up_fn": load_transform(transforms["clean_up"])
        if "clean_up" in transforms
        else None,
        "post_process_fn": load_transform(transforms["post_process"])
        if "post_process" in transforms
        else None,
        "use_dask": True if is_cmems else use_dask,
        "read_zarr": is_cmems,
    }
    for key in (
        "start_time",
        "end_time",
        "allow_flex_time",
        "start_time_pad",
        "end_time_pad",
        "chunks",
        "initial_slice_bounds",
    ):
        if key in runtime:
            fields[key] = runtime[key]
    return LatLonDataset.from_dataset(ds, **fields)


def _build_roms(
    ds: xr.Dataset,
    meta: dict[str, Any],
    source: dict[str, Any],
    runtime: dict[str, Any],
) -> ROMSDataset:
    grid = source.get("grid")
    if grid is None and meta.get("grid"):
        from roms_tools import Grid

        grid = Grid(filename=meta["grid"])
    if grid is None:
        raise ValueError(
            "A ROMS output entry needs a grid: pass source['grid'] or give the entry a "
            "`grid` file in its metadata."
        )
    ds = finalize_loaded_dataset(ds, _ROMS_DIM_NAMES)

    fields: dict[str, Any] = {"grid": grid}
    for key in (
        "var_names",
        "start_time",
        "end_time",
        "allow_flex_time",
        "use_dask",
        "chunks",
        "adjust_depth_for_sea_surface_height",
    ):
        if key in runtime:
            fields[key] = runtime[key]
    if runtime.get("model_reference_date") is not None:
        fields["model_reference_date"] = runtime["model_reference_date"]
    elif meta.get("reference_date"):
        fields["model_reference_date"] = datetime.fromisoformat(
            str(meta["reference_date"])
        )
    return ROMSDataset.from_dataset(ds, **fields)


__all__ = [
    "CONTEXTS",
    "enabled",
    "from_catalog",
    "has",
    "resolve",
    "search_paths",
    "use_catalog",
]
