"""Detect and fill implausible BGC tracer values in ROMS initial conditions and boundary
forcing.

Estimated BGC fields (ESPER above all) can leave cells that are out of range, at zero next
to ordinary water, inconsistent with salinity, or stepped against their neighbours.
:func:`bgc_check` flags them and :func:`bgc_groom` replaces them with a smooth infill from
the surrounding good water, then checks again.

Each tracer gets a :class:`BgcCheck`:

- ``bounds``: the plausible range;
- ``floor_jump``: a value at its floor (``floor``) next to a neighbour at or above this,
  i.e. the zero-next-to-normal artefact. Used for nutrients and O2, whose zeros can be
  real, so a plain gradient limit would also flag natural nutriclines;
- ``max_jump``: the largest allowed difference from the median of a cell's neighbours, on
  salinity-normalized values (``x * 35 / S``) when ``normalize_by_salinity`` is set, so
  river-plume fronts that follow salinity are not flagged. Using the neighbour median flags
  the outlier only, not the healthy cells around it;
- ``physical_reality``: an optional check returning where the value is plausible, e.g.
  ALK against the salinity mixing line, DIC against ALK.

Horizontal neighbours only count when their layer depths differ by less than ``dz_max``:
over steep topography one s-level spans very different depths, and the vertical profile
would otherwise show up as horizontal jumps.

Filling is a fixed sequence of local operations: a column guess (linear in depth between
the nearest good levels; bottom holes take the deepest good value), a same-depth horizontal
mean for columns with no good level, then Jacobi relaxation of the holes. Every step has a
fixed reach and none stops on a convergence test, so the result does not depend on how the
data is chunked or tiled: the tile overlap is set to that reach, and tiled, untiled and
eager runs agree bit for bit. Memory per dask task is bounded by the tile size, not the
domain; the number of concurrent tasks (dask workers) is the other memory control.

Inputs may be :class:`~roms_tools.InitialConditions`, :class:`~roms_tools.BoundaryForcing`
(each BGC source is groomed; salinity comes from its physics), a
``BoundaryForcingSource``, an ``xr.Dataset``, or NetCDF path(s). Everything stays lazy until
the result is written or computed; :func:`summarize_flags` is the explicit eager step.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask
import dask.array as dsa
import numpy as np
import xarray as xr

from roms_tools.setup.utils import get_boundary_coords
from roms_tools.vertical_coordinate import compute_depth_coordinates

if TYPE_CHECKING:
    from roms_tools import Grid

#: Boundary directions, in the order roms-tools writes them.
SIDES = ("south", "east", "north", "west")

#: Horizontal neighbours on the same s-level count only within this depth difference (m).
DZ_MAX = 50.0

#: Reference salinity for salinity-normalized values (x * S_REF / S).
S_REF = 35.0

#: Below this salinity, normalized jumps are not evaluated (fresh or land water).
S_MIN_NORMALIZE = 5.0

#: Open-ocean salinity-normalized alkalinity range (mmol/m3 at S = 35).
NALK_RANGE = (2150.0, 2550.0)

#: River TA endmembers bracketing the ALK-salinity mixing band (mmol/m3). PyESPER's
#: salinity conditioning mixes toward 770, inside this range.
TA_RIVER_RANGE = (0.0, 3000.0)

#: Plausible DIC/ALK ratio.
DIC_ALK_RATIO = (0.80, 1.05)

#: Largest plausible O2 as a multiple of saturation.
O2_SAT_MAX = 1.25

#: Seawater density used to convert O2 solubility from umol/kg to mmol/m3.
RHO0 = 1025.0

#: A value counts as "at its floor" within this tolerance.
FLOOR_TOL = 1.0e-6

#: Default fill parameters. They define the result, so a run is reproducible from them.
PASSES = 5
RELAX_SWEEPS = 10
COLUMN_REACH = 8

#: Default horizontal tile (cells) for initial conditions; an execution setting only.
TILE = 128

#: Default bry_time chunk for boundary forcing; an execution setting only.
BRY_TIME_CHUNK = 31

#: Flag bits returned by bgc_check.
FLAG_BOUNDS = 1
FLAG_FLOOR_JUMP = 2
FLAG_JUMP = 4
FLAG_PHYSICAL = 8
FLAG_NAMES = {
    FLAG_BOUNDS: "bounds",
    FLAG_FLOOR_JUMP: "floor_jump",
    FLAG_JUMP: "jump",
    FLAG_PHYSICAL: "physical",
}


@dataclass(frozen=True)
class Block:
    """One block of data handed to the checks; arrays are (time, s, y, x) float64."""

    values: Mapping[str, np.ndarray]
    salt: np.ndarray
    temp: np.ndarray
    wet: np.ndarray


def alk_reality(block: Block) -> np.ndarray:
    """ALK within the band between conservative mixing lines from the river endmember
    range (``TA_RIVER_RANGE``) to the open-ocean nALK range (``NALK_RANGE``).
    """
    s = block.salt / S_REF
    lo = TA_RIVER_RANGE[0] + (NALK_RANGE[0] - TA_RIVER_RANGE[0]) * s
    hi = TA_RIVER_RANGE[1] + (NALK_RANGE[1] - TA_RIVER_RANGE[1]) * s
    a = block.values["ALK"]
    return (a >= lo) & (a <= hi)


def dic_reality(block: Block) -> np.ndarray:
    """DIC/ALK within ``DIC_ALK_RATIO``."""
    alk = block.values["ALK"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = block.values["DIC"] / alk
    return (alk > 0) & (r >= DIC_ALK_RATIO[0]) & (r <= DIC_ALK_RATIO[1])


def o2_reality(block: Block) -> np.ndarray:
    """O2 at most ``O2_SAT_MAX`` times saturation (TEOS-10 solubility)."""
    import gsw

    with np.errstate(invalid="ignore"):
        sat = gsw.O2sol_SP_pt(block.salt, block.temp) * RHO0 / 1000.0
    o2 = block.values["O2"]
    return ~(o2 > O2_SAT_MAX * sat)  # NaN saturation (no physics) does not flag


@dataclass(frozen=True)
class BgcCheck:
    """Checks for one tracer; see the module docstring."""

    bounds: tuple[float, float] | None = None
    floor: float = 0.0
    floor_jump: float | None = None
    max_jump: float | None = None
    normalize_by_salinity: bool = False
    physical_reality: Callable[[Block], np.ndarray] | None = None
    requires: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        """JSON-able description, recorded in output attrs."""
        return {
            "bounds": list(self.bounds) if self.bounds else None,
            "floor": self.floor,
            "floor_jump": self.floor_jump,
            "max_jump": self.max_jump,
            "normalize_by_salinity": self.normalize_by_salinity,
            "physical_reality": getattr(self.physical_reality, "__name__", None),
            "requires": list(self.requires),
        }


DEFAULT_CHECKS: dict[str, BgcCheck] = {
    "ALK": BgcCheck(
        bounds=(400.0, 2700.0),
        max_jump=100.0,
        normalize_by_salinity=True,
        physical_reality=alk_reality,
    ),
    "DIC": BgcCheck(
        bounds=(400.0, 2700.0),
        max_jump=100.0,
        normalize_by_salinity=True,
        physical_reality=dic_reality,
        requires=("ALK",),
    ),
    "O2": BgcCheck(
        bounds=(0.0, 600.0), floor=5.0, floor_jump=20.0, physical_reality=o2_reality
    ),
    "NO3": BgcCheck(bounds=(0.0, 50.0), floor_jump=5.0),
    "PO4": BgcCheck(bounds=(0.0, 4.0), floor_jump=0.5),
    "SiO3": BgcCheck(bounds=(0.0, 250.0), floor_jump=5.0),
}

# Neighbour directions as (axis, step) on (..., s, y, x) arrays. The order is fixed: it
# is the summation order of every neighbour mean, which keeps results bit-reproducible.
_DIRS = ((-3, -1), (-3, 1), (-2, -1), (-2, 1), (-1, -1), (-1, 1))


def _shift(a: np.ndarray, axis: int, step: int, fill: Any) -> np.ndarray:
    """Return b with b[i] = a[i + step] along ``axis``; ``fill`` where that falls off."""
    out = np.full_like(a, fill)
    n = a.shape[axis]
    if n < 2:
        return out
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    if step == 1:
        dst[axis], src[axis] = slice(0, n - 1), slice(1, n)
    else:
        dst[axis], src[axis] = slice(1, n), slice(0, n - 1)
    out[tuple(dst)] = a[tuple(src)]
    return out


def _eligible(wet: np.ndarray, z: np.ndarray, dz_max: float) -> list[np.ndarray]:
    """Per direction: both cells wet, and (horizontally) within dz_max in depth."""
    out = []
    for axis, step in _DIRS:
        e = wet & _shift(wet, axis, step, False)
        if axis != -3:
            with np.errstate(invalid="ignore"):
                e &= np.abs(z - _shift(z, axis, step, np.nan)) < dz_max
        out.append(e)
    return out


def _detect(
    name: str, chk: BgcCheck, block: Block, elig: list[np.ndarray]
) -> np.ndarray:
    """Flag bits (uint8) for one tracer."""
    a = block.values[name]
    wet = block.wet
    flags = np.zeros(a.shape, dtype=np.uint8)
    with np.errstate(invalid="ignore"):
        if chk.bounds is not None:
            ok = (a >= chk.bounds[0]) & (a <= chk.bounds[1])
            flags |= np.uint8(FLAG_BOUNDS) * (wet & ~ok)
        if chk.floor_jump is not None:
            at_floor = wet & (a <= chk.floor + FLOOR_TOL)
            high = np.zeros(a.shape, dtype=bool)
            for e, (axis, step) in zip(elig, _DIRS, strict=True):
                high |= e & (_shift(a, axis, step, -np.inf) >= chk.floor_jump)
            flags |= np.uint8(FLAG_FLOOR_JUMP) * (at_floor & high)
        if chk.max_jump is not None:
            if chk.normalize_by_salinity:
                valid = wet & (block.salt >= S_MIN_NORMALIZE)
                x = np.where(
                    valid, a * S_REF / np.where(valid, block.salt, 1.0), np.nan
                )
            else:
                valid = wet
                x = a
            nbrs = np.stack(
                [
                    np.where(
                        e & _shift(valid, axis, step, False),
                        _shift(x, axis, step, np.nan),
                        np.nan,
                    )
                    for e, (axis, step) in zip(elig, _DIRS, strict=True)
                ]
            )
            n_ok = np.sum(~np.isnan(nbrs), axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", RuntimeWarning
                )  # all-NaN neighbour sets
                med = np.nanmedian(nbrs, axis=0)
            jump = valid & (n_ok >= 2) & (np.abs(x - med) > chk.max_jump)
            flags |= np.uint8(FLAG_JUMP) * jump
        if chk.physical_reality is not None:
            ok = chk.physical_reality(block)
            flags |= np.uint8(FLAG_PHYSICAL) * (wet & ~ok)
    return flags


def _neighbour_mean(
    v: np.ndarray, elig: list[np.ndarray], horizontal_only: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of eligible, finite neighbours, accumulated in _DIRS order."""
    acc = np.zeros_like(v)
    cnt = np.zeros(v.shape, dtype=np.int8)
    for e, (axis, step) in zip(elig, _DIRS, strict=True):
        if horizontal_only and axis == -3:
            continue
        vs = _shift(v, axis, step, np.nan)
        ok = e & ~np.isnan(vs)
        acc += np.where(ok, vs, 0.0)
        cnt += ok
    return acc, cnt


def _fill(
    a: np.ndarray,
    holes: np.ndarray,
    wet: np.ndarray,
    z: np.ndarray,
    elig: list[np.ndarray],
    column_reach: int,
    relax_sweeps: int,
) -> np.ndarray:
    """Fill ``holes`` in ``a``: column guess, same-depth reach, Jacobi relaxation.

    Cells that cannot be reached keep their original value (and stay flagged).
    """
    if not holes.any():
        return a
    good = wet & ~holes
    ns = a.shape[-3]
    idx = np.broadcast_to(np.arange(ns).reshape(ns, 1, 1), a.shape)
    # s = 0 is the bottom: "lower" is the nearest good level at or below, "upper" above.
    lower = np.maximum.accumulate(np.where(good, idx, -1), axis=-3)
    upper = np.flip(
        np.minimum.accumulate(np.flip(np.where(good, idx, ns), axis=-3), axis=-3),
        axis=-3,
    )
    has_l, has_u = lower >= 0, upper < ns
    li, ui = np.clip(lower, 0, ns - 1), np.clip(upper, 0, ns - 1)
    al, au = np.take_along_axis(a, li, axis=-3), np.take_along_axis(a, ui, axis=-3)
    zl, zu = np.take_along_axis(z, li, axis=-3), np.take_along_axis(z, ui, axis=-3)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (z - zl) / (zu - zl)
        guess = np.where(holes & has_l & has_u, al + w * (au - al), np.nan)
    guess = np.where(holes & has_u & ~has_l, au, guess)  # bottom: deepest good above
    guess = np.where(holes & has_l & ~has_u, al, guess)  # top: shallowest good below
    out = np.where(holes, guess, a)

    for _ in range(column_reach):
        missing = holes & np.isnan(out)
        if not missing.any():
            break  # further iterations would change nothing
        acc, cnt = _neighbour_mean(out, elig, horizontal_only=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(missing & (cnt > 0), acc / np.maximum(cnt, 1), out)

    for _ in range(relax_sweeps):
        acc, cnt = _neighbour_mean(out, elig, horizontal_only=False)
        upd = holes & ~np.isnan(out) & (cnt > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(upd, acc / np.maximum(cnt, 1), out)

    return np.where(holes & np.isnan(out), a, out)


def _stages(names: Sequence[str], checks: Mapping[str, BgcCheck]) -> list[list[str]]:
    """Group tracers so each one is groomed after the tracers its check requires."""
    remaining = list(names)
    done: set[str] = set()
    stages = []
    while remaining:
        stage = [
            n
            for n in remaining
            if all(r in done or r not in names for r in checks[n].requires)
        ]
        if not stage:
            raise ValueError(f"Circular 'requires' among BGC checks: {remaining}")
        stages.append(stage)
        done.update(stage)
        remaining = [n for n in remaining if n not in stage]
    return stages


def _kernel(
    combo: np.ndarray,
    *,
    names: tuple[str, ...],
    checks: Mapping[str, BgcCheck],
    mode: str,
    stage: tuple[str, ...] = (),
    dz_max: float = DZ_MAX,
    column_reach: int = COLUMN_REACH,
    relax_sweeps: int = RELAX_SWEEPS,
) -> np.ndarray:
    """Check (mode="check": uint8 flags) or groom ``stage`` (mode="groom": values).

    ``combo`` is (tracers..., salt, temp, z, wet) stacked on axis 0, each (time, s, y, x).
    The output has the same shape: groomed tracers followed by the unchanged aux rows
    (mode="groom"), or tracer flags followed by zeros (mode="check").
    """
    nv = len(names)
    stack, (salt, temp, z, wetf) = combo[:nv], combo[nv:]
    wet = wetf > 0.5
    values = {n: stack[i] for i, n in enumerate(names)}
    elig = _eligible(wet, z, dz_max)
    if mode == "check":
        block = Block(values, salt, temp, wet)
        flags = np.zeros(combo.shape, dtype=np.uint8)
        for i, n in enumerate(names):
            flags[i] = _detect(n, checks[n], block, elig)
        return flags
    out = combo.copy()
    for n in stage:
        block = Block(values, salt, temp, wet)
        holes = _detect(n, checks[n], block, elig) > 0
        values[n] = _fill(values[n], holes, wet, z, elig, column_reach, relax_sweeps)
        out[names.index(n)] = values[n]
    return out


# --------------------------------------------------------------------------- inputs


def _open(obj: Any, label: str) -> xr.Dataset:
    """Dataset from a Dataset, an object with ``.ds``, or NetCDF path(s), lazily."""
    if isinstance(obj, xr.Dataset):
        return obj
    if hasattr(obj, "ds") and isinstance(obj.ds, xr.Dataset):
        return obj.ds
    if isinstance(obj, str | Path):
        return xr.open_dataset(
            obj, chunks={}, decode_times=False, decode_timedelta=False
        )
    if isinstance(obj, list | tuple):
        return xr.open_mfdataset(
            [str(p) for p in obj],
            combine="by_coords",
            decode_times=False,
            decode_timedelta=False,
            parallel=False,
        )
    raise TypeError(
        f"{label}: expected a Dataset, an object with .ds, or path(s); got {type(obj)}"
    )


def _grid_ds(grid: Grid | xr.Dataset | str | Path | None, source: Any) -> xr.Dataset:
    if grid is None:
        grid = getattr(source, "grid", None)
    if grid is None:
        raise ValueError("A grid (Grid, grid Dataset, or grid file path) is required.")
    return _open(grid, "grid")


def _is_boundary(ds: xr.Dataset, checks: Mapping[str, BgcCheck]) -> bool:
    return any(f"{v}_{s}" in ds for v in checks for s in SIDES)


def _select(
    ds: xr.Dataset, checks: Mapping[str, BgcCheck], variables, suffix: str = ""
) -> tuple[str, ...]:
    present = tuple(v for v in checks if f"{v}{suffix}" in ds)
    if variables is None:
        return present
    missing = [v for v in variables if v not in checks]
    if missing:
        raise ValueError(f"No BgcCheck defined for {missing}; known: {list(checks)}")
    absent = [v for v in variables if f"{v}{suffix}" not in ds]
    if absent and not suffix:
        raise ValueError(f"Variables {absent} not found in the dataset.")
    return tuple(v for v in variables if v in present)


def _as_dask(da: xr.DataArray) -> dsa.Array:
    data = da.data
    if not isinstance(data, dsa.Array):
        data = dsa.from_array(np.asarray(data), chunks=-1)
    return data.astype(np.float64)


# --------------------------------------------------------------------------- engine


def _overlap_reach(column_reach: int, relax_sweeps: int) -> int:
    """Cells a groom pass can reach: detection (1) + column reach + relaxation."""
    return 1 + column_reach + relax_sweeps


def _run(
    stack: dsa.Array,
    aux: dsa.Array,
    names: tuple[str, ...],
    checks: Mapping[str, BgcCheck],
    *,
    mode: str,
    dz_max: float,
    passes: int,
    column_reach: int,
    relax_sweeps: int,
) -> dsa.Array:
    """Chain lazy map_overlap passes; returns (tracer, time, s, y, x) values or flags."""
    nv = len(names)
    combo = dsa.concatenate([stack, aux], axis=0).rechunk({0: -1})
    kernel = partial(
        _kernel,
        names=names,
        checks=checks,
        dz_max=dz_max,
        column_reach=column_reach,
        relax_sweeps=relax_sweeps,
    )

    def depth_for(reach: int) -> dict[int, int]:
        # Only axes split into several chunks need an overlap; a single chunk already
        # holds every neighbour, so the result is the same either way.
        return {
            ax: (reach if ax >= 3 and len(combo.chunks[ax]) > 1 else 0)
            for ax in range(5)
        }

    if mode == "check":
        flags = dsa.map_overlap(
            partial(kernel, mode="check"),
            combo,
            depth=depth_for(1),
            boundary="none",
            dtype=np.uint8,
            meta=np.array((), dtype=np.uint8),
        )
        return flags[:nv]
    depth = depth_for(_overlap_reach(column_reach, relax_sweeps))
    for _ in range(passes):
        for stage in _stages(names, checks):
            combo = dsa.map_overlap(
                partial(kernel, mode="groom", stage=tuple(stage)),
                combo,
                depth=depth,
                boundary="none",
                dtype=np.float64,
                meta=np.array((), dtype=np.float64),
            )
    return combo[:nv]


def _params(
    checks, names, dz_max, passes, column_reach, relax_sweeps
) -> dict[str, Any]:
    return {
        "passes": passes,
        "relax_sweeps": relax_sweeps,
        "column_reach": column_reach,
        "dz_max": dz_max,
        "checks": {n: checks[n].describe() for n in names},
    }


def _ic_arrays(ds, physics, grid_ds, names, tile):
    """(tracer,t,s,y,x) stack and (salt,temp,z,wet) aux for an initial-conditions dataset."""
    phys = ds if physics is None else _open(physics, "physics")
    for v in ("salt", "temp"):
        if v not in phys:
            raise ValueError(
                f"'{v}' is needed by the checks but is not in the IC/physics data."
            )
    tdim = next(d for d in ds[names[0]].dims if d not in ("s_rho", "eta_rho", "xi_rho"))
    shape = ds[names[0]].transpose(tdim, "s_rho", "eta_rho", "xi_rho").shape
    chunks = (1, -1, tile, tile)

    def arr(da):
        return _as_dask(da.transpose(tdim, "s_rho", "eta_rho", "xi_rho")).rechunk(
            chunks
        )

    stack = dsa.stack([arr(ds[n]) for n in names]).rechunk((-1, *chunks))
    # Chunk the grid first so the 3-D layer depths stay lazy, tile by tile.
    gch = grid_ds.chunk({"eta_rho": tile, "xi_rho": tile})
    z = compute_depth_coordinates(gch, zeta=0, depth_type="layer", location="rho")
    z = _as_dask(z.transpose("s_rho", "eta_rho", "xi_rho"))
    wet = _as_dask(grid_ds["mask_rho"].transpose("eta_rho", "xi_rho"))
    aux = dsa.stack(
        [
            arr(phys["salt"]),
            arr(phys["temp"]),
            dsa.broadcast_to(z, shape).rechunk(chunks),
            dsa.broadcast_to(wet, shape).rechunk(chunks),
        ]
    ).rechunk((-1, *chunks))
    dims = (tdim, "s_rho", "eta_rho", "xi_rho")
    return stack, aux, dims


def _bry_arrays(ds, phys, grid_ds, names, side, time_chunk):
    """Stack/aux for one boundary side, as (tracer, t, s, 1, along)."""
    sl = get_boundary_coords()["rho"][side]
    for v in ("salt", "temp"):
        if f"{v}_{side}" not in phys:
            raise ValueError(
                f"'{v}_{side}' is needed by the checks but is not in the physics data."
            )
    if not np.array_equal(np.asarray(ds["bry_time"]), np.asarray(phys["bry_time"])):
        raise ValueError("BGC and physics boundary data have different bry_time axes.")
    along = "xi_rho" if side in ("south", "north") else "eta_rho"
    dims = ("bry_time", "s_rho", along)
    nt, ns, nx = ds[f"{names[0]}_{side}"].transpose(*dims).shape
    chunks = (time_chunk, -1, 1, -1)

    def arr(da):
        return _as_dask(da.transpose(*dims))[:, :, None, :].rechunk(chunks)

    stack = dsa.stack([arr(ds[f"{n}_{side}"]) for n in names]).rechunk((-1, *chunks))
    idx = {"eta": sl["eta_rho"]} if "eta_rho" in sl else {"xi": sl["xi_rho"]}
    z = compute_depth_coordinates(
        grid_ds, zeta=0, depth_type="layer", location="rho", **idx
    )
    z = _as_dask(z.transpose("s_rho", along))[:, None, :]
    wet = _as_dask(grid_ds["mask_rho"].isel(**sl))[None, None, :]
    shape = (nt, ns, 1, nx)
    aux = dsa.stack(
        [
            arr(phys[f"salt_{side}"]),
            arr(phys[f"temp_{side}"]),
            dsa.broadcast_to(z, shape).rechunk(chunks),
            dsa.broadcast_to(wet, shape).rechunk(chunks),
        ]
    ).rechunk((-1, *chunks))
    return stack, aux, dims


def _process(
    source,
    grid,
    physics,
    *,
    mode,
    checks,
    variables,
    dz_max,
    passes,
    column_reach,
    relax_sweeps,
    tile,
    time_chunk,
):
    """Shared driver for one dataset (IC layout or boundary layout)."""
    checks = DEFAULT_CHECKS if checks is None else checks
    ds = _open(source, "source")
    grid_ds = _grid_ds(grid, source)
    params = _params(checks, (), dz_max, passes, column_reach, relax_sweeps)
    run = dict(
        mode=mode,
        dz_max=dz_max,
        passes=passes,
        column_reach=column_reach,
        relax_sweeps=relax_sweeps,
    )
    out = ds.copy() if mode == "groom" else xr.Dataset(coords=ds.coords, attrs=ds.attrs)

    if not _is_boundary(ds, checks):
        names = _select(ds, checks, variables)
        if not names:
            raise ValueError("No checkable BGC variables found in the dataset.")
        params = _params(checks, names, dz_max, passes, column_reach, relax_sweeps)
        stack, aux, dims = _ic_arrays(ds, physics, grid_ds, names, tile)
        res = _run(stack, aux, names, checks, **run)
        for i, n in enumerate(names):
            _store(out, ds, n, res[i], dims, mode, checks, n, params)
    else:
        if physics is None:
            physics = getattr(getattr(source, "physics_forcing", None), "ds", None)
        if physics is None:
            raise ValueError(
                "Boundary BGC data needs the matching physics boundary data (salt/temp)."
            )
        phys = _open(physics, "physics")
        all_names: set[str] = set()
        for side in SIDES:
            names = _select(ds, checks, variables, suffix=f"_{side}")
            if not names:
                continue
            all_names.update(names)
            stack, aux, dims = _bry_arrays(ds, phys, grid_ds, names, side, time_chunk)
            res = _run(stack, aux, names, checks, **run)
            for i, n in enumerate(names):
                _store(
                    out,
                    ds,
                    f"{n}_{side}",
                    res[i][:, :, 0, :],
                    dims,
                    mode,
                    checks,
                    n,
                    params,
                )
        if not all_names:
            raise ValueError(
                "No checkable BGC boundary variables found in the dataset."
            )
        params = _params(
            checks, tuple(sorted(all_names)), dz_max, passes, column_reach, relax_sweeps
        )
    if mode == "groom":
        out.attrs["bgc_groom"] = json.dumps(params)
    else:
        out.attrs["bgc_check"] = json.dumps(
            {k: v for k, v in params.items() if k in ("dz_max", "checks")}
        )
        out.attrs["flag_bits"] = json.dumps(FLAG_NAMES)
    return out


def _store(out, ds, var, data, dims, mode, checks, name, params):
    if mode == "groom":
        orig = ds[var]
        da = xr.DataArray(data.astype(orig.dtype), dims=dims).transpose(*orig.dims)
        out[var] = da.assign_attrs(orig.attrs).assign_attrs(
            bgc_groom=json.dumps(
                {
                    **{
                        k: params[k]
                        for k in ("passes", "relax_sweeps", "column_reach", "dz_max")
                    },
                    "check": checks[name].describe(),
                }
            )
        )
    else:
        out[var] = xr.DataArray(data, dims=dims).transpose(*ds[var].dims)


# --------------------------------------------------------------------------- public API


def bgc_check(
    source: Any,
    grid: Grid | xr.Dataset | str | Path | None = None,
    physics: Any = None,
    *,
    checks: Mapping[str, BgcCheck] | None = None,
    variables: Sequence[str] | None = None,
    dz_max: float = DZ_MAX,
    tile: int = TILE,
    time_chunk: int = BRY_TIME_CHUNK,
) -> xr.Dataset | list[xr.Dataset]:
    """Flag implausible BGC values (lazy).

    Returns a Dataset of uint8 flag bits per checked variable (see ``FLAG_NAMES``); 0 means
    the cell passed. For a ``BoundaryForcing`` object, one Dataset per BGC source.
    ``grid`` defaults to ``source.grid``; ``physics`` (salt/temp) defaults to the source's
    own physics (IC data carries it; boundary BGC data needs the physics boundary data).
    """
    kw = dict(
        mode="check",
        checks=checks,
        variables=variables,
        dz_max=dz_max,
        passes=0,
        column_reach=0,
        relax_sweeps=0,
        tile=tile,
        time_chunk=time_chunk,
    )
    if hasattr(source, "bgc") and hasattr(source, "physics"):  # BoundaryForcing wrapper
        return [
            _process(
                b.ds,
                source.grid if grid is None else grid,
                source.physics.ds if physics is None else physics,
                **kw,
            )
            for b in source.bgc
        ]
    return _process(source, grid, physics, **kw)


def bgc_groom(
    source: Any,
    grid: Grid | xr.Dataset | str | Path | None = None,
    physics: Any = None,
    *,
    checks: Mapping[str, BgcCheck] | None = None,
    variables: Sequence[str] | None = None,
    dz_max: float = DZ_MAX,
    passes: int = PASSES,
    relax_sweeps: int = RELAX_SWEEPS,
    column_reach: int = COLUMN_REACH,
    tile: int = TILE,
    time_chunk: int = BRY_TIME_CHUNK,
) -> xr.Dataset | list[xr.Dataset]:
    """Return a lazily groomed copy of the BGC data; the input is never modified.

    Flagged cells are replaced by a fill from the surrounding good water, ``passes`` times
    (each pass re-checks). ``passes``, ``relax_sweeps``, ``column_reach`` and ``dz_max``
    define the result and are recorded in the attrs; ``tile`` and ``time_chunk`` only set
    task size and do not change it. Run :func:`bgc_check` on the result for what remains.
    For a ``BoundaryForcing`` object, returns one Dataset per BGC source (assign to
    ``bf.bgc[i].ds`` to save); for initial conditions assign to ``ic.ds``.
    """
    for k, v in (
        ("passes", passes),
        ("relax_sweeps", relax_sweeps),
        ("column_reach", column_reach),
    ):
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"{k} must be a non-negative int, got {v!r}")
    kw = dict(
        mode="groom",
        checks=checks,
        variables=variables,
        dz_max=dz_max,
        passes=passes,
        column_reach=column_reach,
        relax_sweeps=relax_sweeps,
        tile=tile,
        time_chunk=time_chunk,
    )
    if hasattr(source, "bgc") and hasattr(source, "physics"):  # BoundaryForcing wrapper
        return [
            _process(
                b.ds,
                source.grid if grid is None else grid,
                source.physics.ds if physics is None else physics,
                **kw,
            )
            for b in source.bgc
        ]
    return _process(source, grid, physics, **kw)


def summarize_flags(flags: xr.Dataset) -> dict[str, dict[str, int]]:
    """Count flagged cells per variable and reason (computes)."""
    lazy = {}
    for var in flags.data_vars:
        f = flags[var].data
        lazy[var] = {
            "total": (f > 0).sum(),
            **{
                name: ((f & np.uint8(bit)) > 0).sum()
                for bit, name in FLAG_NAMES.items()
            },
        }
    (done,) = dask.compute(lazy)
    return {v: {k: int(c) for k, c in d.items()} for v, d in done.items()}
