"""BGC data source abstraction for BoundaryForcing and InitialConditions.

This module provides :class:`BGCSource`, a unified dataclass for specifying biogeochemical
data sources, along with helpers for instantiating raw datasets and merging fields from
multiple sources with priority semantics.

The same ``BGCSource`` / ``merge_bgc_fields`` machinery is shared by
:class:`~roms_tools.setup.boundary_forcing.BoundaryForcing` (type="bgc") and
:class:`~roms_tools.setup.initial_conditions.InitialConditions` (bgc_source=...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import xarray as xr

from roms_tools.datasets.lat_lon_datasets import (
    CESMBGCDataset,
    GLODAPv2BGCDataset,
    UnifiedBGCDataset,
)
from roms_tools.datasets.roms_dataset import ROMSDataset


# Registry mapping source name → dataset class (Mode A sources).
BGC_DATASET_MAP: dict[str, type] = {
    "CESM_REGRIDDED": CESMBGCDataset,
    "UNIFIED": UnifiedBGCDataset,
    "GLODAP": GLODAPv2BGCDataset,
    "ROMS": ROMSDataset,
}

# Comprehensive metadata for every possible BGC output variable.
# All BGC variables share the same location / vector / dimensionality attributes.
# Only ALK is validated (validate=True) to match the existing convention.
_BGC_VARS = [
    "PO4", "NO3", "SiO3", "NH4", "Fe", "Lig", "O2",
    "DIC", "DIC_ALT_CO2", "ALK", "ALK_ALT_CO2",
    "DOC", "DON", "DOP", "DOPr", "DONr", "DOCr",
    "spChl", "spC", "spP", "spFe", "spCaCO3",
    "diatChl", "diatC", "diatP", "diatFe", "diatSi",
    "diazChl", "diazC", "diazP", "diazFe",
    "zooC", "CHL",
]

BGC_VARIABLE_INFO: dict[str, dict] = {
    var: {
        "location": "rho",
        "is_vector": False,
        "vector_pair": None,
        "is_3d": True,
        "validate": var == "ALK",
    }
    for var in _BGC_VARS
}


@dataclass
class BGCSource:
    """A single BGC data source for use in BoundaryForcing or InitialConditions.

    Exactly one of the three modes must be configured:

    **Mode A — lat/lon dataset** (``name`` is set):
        Wraps an existing :class:`~roms_tools.datasets.lat_lon_datasets.LatLonDataset`
        subclass (e.g. ``"UNIFIED"``, ``"CESM_REGRIDDED"``) or a
        :class:`~roms_tools.datasets.roms_dataset.ROMSDataset` (``"ROMS"``).

    **Mode B — depth-invariant constants** (``constants`` is set):
        Provides uniform scalar values for one or more BGC variables.  The constant is
        applied at all depths and grid points.

    **Mode C — physics-derived algorithm** (``physics_forcing`` is set):
        Derives BGC variables from a pre-computed physics
        :class:`~roms_tools.setup.boundary_forcing.BoundaryForcing` or
        :class:`~roms_tools.setup.initial_conditions.InitialConditions` object via a
        user-supplied callable.

    Parameters
    ----------
    name : str, optional
        Dataset name for Mode A.  Must be a key of :data:`BGC_DATASET_MAP`.
    path : str or Path or list of str/Path, optional
        File path(s) for Mode A.
    climatology : bool, optional
        Whether the Mode A dataset is a climatology (12 monthly values).
        Defaults to False.
    constants : dict[str, float], optional
        Variable → constant value mapping for Mode B.
    physics_forcing : object, optional
        Pre-computed physics forcing object for Mode C.
    algorithm : callable, optional
        For Mode C: ``algorithm(physics_ds, direction) -> dict[str, xr.DataArray]``.
        ``direction`` is a boundary name (``"south"`` etc.) for BoundaryForcing, or
        ``None`` for the full domain (InitialConditions).  The callable must return
        lazy (dask-backed) DataArrays already on the ROMS boundary or domain grid.
    variables : list[str], optional
        When set, these specific variable names are **force-overwritten** in the merged
        result even if a higher-priority source already provided them.  All other
        variables from this source follow normal fill-only semantics.
    """

    # --- Mode A: lat/lon dataset ---
    name: str | None = None
    path: str | Path | list[str | Path] | None = None
    climatology: bool = False
    grid: Any | None = None  # parent Grid object; required only when name="ROMS"

    # --- Mode B: uniform depth-invariant constants ---
    constants: dict[str, float] | None = None

    # --- Mode C: algorithm from pre-computed physics forcing ---
    physics_forcing: Any | None = None
    algorithm: Callable | None = None

    # --- Priority override ---
    variables: list[str] | None = None

    def __post_init__(self) -> None:
        modes = sum([
            self.name is not None,
            self.constants is not None,
            self.physics_forcing is not None,
        ])
        if modes != 1:
            raise ValueError(
                "BGCSource: exactly one of (name, constants, physics_forcing) must be set."
            )
        if self.name is not None and self.name not in BGC_DATASET_MAP:
            raise ValueError(
                f"BGCSource: unknown name '{self.name}'. "
                f"Valid options: {list(BGC_DATASET_MAP)}"
            )
        if self.physics_forcing is not None and self.algorithm is None:
            raise ValueError(
                "BGCSource: 'algorithm' must be provided when 'physics_forcing' is set."
            )


def instantiate_bgc_dataset(
    source: BGCSource,
    start_time: datetime | None,
    end_time: datetime | None,
    use_dask: bool,
    chunks: dict | None,
    initial_slice_bounds: dict | None,
    roms_var_names: list[str] | None = None,
    allow_flex_time: bool = False,
    adjust_depth_for_sea_surface_height: bool = False,
) -> CESMBGCDataset | UnifiedBGCDataset | ROMSDataset:
    """Instantiate the raw dataset object for a Mode A :class:`BGCSource`.

    Parameters
    ----------
    source : BGCSource
        A Mode A source (``source.name`` must be set).
    start_time, end_time : datetime or None
        Time range for data selection.
    use_dask : bool
        Whether to load data lazily with dask.
    chunks : dict or None
        Dask chunk sizes.
    initial_slice_bounds : dict or None
        Optional lat/lon bounding box for initial data loading.
    roms_var_names : list[str], optional
        Required variable names; used only for the ``"ROMS"`` source.
    allow_flex_time : bool, optional
        Allow flexible time matching; used only for the ``"ROMS"`` source.

    Returns
    -------
    LatLonDataset or ROMSDataset
        The instantiated dataset object.
    """
    assert source.name is not None, "instantiate_bgc_dataset requires a Mode A BGCSource"

    dataset_cls = BGC_DATASET_MAP[source.name]

    if source.name == "ROMS":
        return dataset_cls(
            path=source.path,
            grid=source.grid,
            var_names=roms_var_names or [],
            start_time=start_time,
            allow_flex_time=allow_flex_time,
            adjust_depth_for_sea_surface_height=adjust_depth_for_sea_surface_height,
            use_dask=use_dask,
            chunks=chunks,
        )

    return dataset_cls(
        filename=source.path,
        start_time=start_time,
        end_time=end_time,
        climatology=source.climatology,
        use_dask=use_dask,
        chunks=chunks,
        initial_slice_bounds=initial_slice_bounds,
    )


def merge_bgc_fields(
    source_fields: list[tuple[BGCSource, dict[str, xr.DataArray]]],
) -> dict[str, xr.DataArray]:
    """Merge BGC fields from multiple sources using priority semantics.

    Sources are processed in order; the **first source has the highest priority**.
    For each variable:

    * If the variable is not yet in the accumulator → add it (fill-only).
    * If the variable is already present **and** appears in ``source.variables`` →
      overwrite it (explicit override).
    * Otherwise → skip (higher-priority source already filled this variable).

    All operations work on :class:`xarray.DataArray` references; no ``.compute()``
    is triggered.

    Parameters
    ----------
    source_fields : list of (BGCSource, dict)
        Pairs of source metadata and their per-variable DataArray fields, in
        priority order (index 0 = highest priority).

    Returns
    -------
    dict[str, xr.DataArray]
        Merged variable dictionary.
    """
    merged: dict[str, xr.DataArray] = {}
    for src, fields in source_fields:
        for var, da in fields.items():
            if var not in merged:
                merged[var] = da
            elif src.variables and var in src.variables:
                merged[var] = da
    return merged
