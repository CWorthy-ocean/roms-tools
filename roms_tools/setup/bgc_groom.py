"""Detect and fill implausible BGC tracer values in ROMS initial conditions and boundary
forcing.

Estimated BGC fields (ESPER above all) can leave cells that are out of range, at zero next
to ordinary water, inconsistent with salinity, or stepped against their neighbours.
:func:`bgc_check` flags them and :func:`bgc_groom` replaces them with a smooth infill from
the surrounding good water, then checks again.

Each tracer gets a :class:`BgcCheck`:

- ``bounds``: the plausible range;
- ``floor_jump``: a value at its floor (``floor``) next to a same-depth neighbour at or
  above this, or between two vertical neighbours at or above it, i.e. the
  zero-next-to-normal artefact. Used for nutrients and O2, whose zeros can be real, so a
  plain gradient limit would also flag natural nutriclines. One high vertical neighbour
  does not count: on a coarse vertical grid a real nitracline or OMZ edge steps from the
  floor to above ``floor_jump`` between adjacent levels;
- ``max_jump``: a cell is flagged as a spike when it lies outside the range of its
  eligible horizontal neighbours (at least two) by more than this, or outside the range of
  its two vertical neighbours. Values are salinity-normalized (``E + (x - E) * 35 / S``,
  E the freshwater endmember) when ``normalize_by_salinity`` is set, so river-plume fronts
  that follow salinity pass. A cell inside a steep but smooth gradient, on one side of a
  front, or next to a bad cell is not a spike. Neither is the interior of a large bad
  region, which agrees with itself: bounds and ``physical_reality`` must catch that;
- ``physical_reality``: an optional check returning where the value is plausible, e.g.
  ALK against the salinity mixing line, DIC against ALK.
- ``ceiling``: an optional upper limit that grooming clips every wet cell to after
  filling, e.g. O2 at 1.25 times saturation, which filling alone cannot always reach.

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

:func:`bgc_groom` takes an ``xr.Dataset`` or NetCDF path(s); :func:`bgc_check` also takes
:class:`~roms_tools.InitialConditions`, :class:`~roms_tools.BoundaryForcing` (each BGC
source is checked; salinity comes from its physics) or a ``BoundaryForcingSource``. The
inputs roms-tools builds are groomed through the ``groom``/``groom_settings`` arguments of
``InitialConditions``/``BoundaryForcing``, so their YAML regenerates the file. Everything
stays lazy until the result is written or computed; :func:`summarize_flags` is the explicit
eager step.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask
import dask.array as dsa
import numpy as np
import xarray as xr

from roms_tools.setup.bgc_model import MARBL_PARM_O2_MIN
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

#: Open-ocean salinity-normalized alkalinity range (mmol/m3 at S = 35). The upper end
#: allows for in-situ density: tracers are per m3 at in-situ density, about 4-5% above
#: surface density at 4-5 km, so deep water near 2450 umol/kg reads about 2560 mmol/m3.
NALK_RANGE = (2150.0, 2650.0)

#: Upper bound (mmol/m3) for ALK and DIC: a gross-error net only, set well outside
#: anything plausible. The plausibility checks are ``NALK_RANGE`` and ``DIC_ALK_RATIO``.
#: At in-situ density, deep Eastern Mediterranean water (TA about 2610 umol/kg) reads
#: above 2700 mmol/m3, so a tighter bound would clamp correct water.
ALK_DIC_MAX = 3000.0

#: Freshwater ALK and DIC endmember (mmol/m3) for salinity normalization: the value
#: PyESPER's salinity conditioning dilutes TA and DIC toward. Normalizing against it
#: (``E + (x - E) * S_REF / S``) removes both dilution and evaporative concentration, so
#: conditioned plume and hypersaline water normalize to open-ocean values.
FRESHWATER_ENDMEMBER = 770.0

#: Plausible DIC/ALK ratio.
DIC_ALK_RATIO = (0.80, 1.05)

#: Largest plausible O2 as a multiple of saturation.
O2_SAT_MAX = 1.25

#: Seawater density used to convert O2 solubility from umol/kg to mmol/m3.
RHO0 = 1025.0

#: A value counts as "at its floor" within this tolerance.
FLOOR_TOL = 1.0e-6

#: Projected cells land this fraction of max_jump inside their envelope, so writing as
#: float32 (rounding ~1e-4 mmol/m3 at these values) cannot push them back outside it.
PROJECT_MARGIN = 1.0e-3

#: Clipped cells land this fraction below their ceiling, so writing as float32 (relative
#: rounding ~6e-8) cannot push them back above it.
CEILING_MARGIN = 1.0e-6

#: Default fill parameters. They define the result, so a run is reproducible from them.
PASSES = 8
RELAX_SWEEPS = 10
COLUMN_REACH = 8

#: Default horizontal tile (cells) for initial conditions; an execution setting only.
TILE = 128

#: Default bry_time chunk for boundary forcing; an execution setting only.
BRY_TIME_CHUNK = 4

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
    """Salinity-normalized ALK (against ``FRESHWATER_ENDMEMBER``) within ``NALK_RANGE``:
    ALK lies in the band of mixing lines from the freshwater endmember to open-ocean water.
    Water fresher than ``S_MIN_NORMALIZE`` is not judged.
    """
    n, valid = _normalize(block.values["ALK"], block.salt, block.wet)
    return ~valid | ((n >= NALK_RANGE[0]) & (n <= NALK_RANGE[1]))


def dic_reality(block: Block) -> np.ndarray:
    """DIC/ALK within ``DIC_ALK_RATIO``."""
    alk = block.values["ALK"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = block.values["DIC"] / alk
    return (alk > 0) & (r >= DIC_ALK_RATIO[0]) & (r <= DIC_ALK_RATIO[1])


def o2_ceiling(block: Block) -> np.ndarray:
    """``O2_SAT_MAX`` times O2 saturation (TEOS-10 solubility), in mmol/m3."""
    import gsw

    with np.errstate(invalid="ignore"):
        return O2_SAT_MAX * gsw.O2sol_SP_pt(block.salt, block.temp) * RHO0 / 1000.0


def o2_reality(block: Block) -> np.ndarray:
    """O2 at most ``O2_SAT_MAX`` times saturation."""
    return ~(block.values["O2"] > o2_ceiling(block))  # NaN (no physics) does not flag


@dataclass(frozen=True)
class BgcCheck:
    """Checks for one tracer; see the module docstring."""

    bounds: tuple[float, float] | None = None
    floor: float = 0.0
    floor_jump: float | None = None
    max_jump: float | None = None
    normalize_by_salinity: bool = False
    physical_reality: Callable[[Block], np.ndarray] | None = None
    ceiling: Callable[[Block], np.ndarray] | None = None
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
            "ceiling": getattr(self.ceiling, "__name__", None),
            "requires": list(self.requires),
        }


DEFAULT_CHECKS: dict[str, BgcCheck] = {
    "ALK": BgcCheck(
        bounds=(400.0, ALK_DIC_MAX),
        max_jump=100.0,
        normalize_by_salinity=True,
        physical_reality=alk_reality,
    ),
    "DIC": BgcCheck(
        bounds=(400.0, ALK_DIC_MAX),
        max_jump=100.0,
        normalize_by_salinity=True,
        physical_reality=dic_reality,
        requires=("ALK",),
    ),
    "O2": BgcCheck(
        bounds=(0.0, 600.0),
        floor=MARBL_PARM_O2_MIN,
        floor_jump=20.0,
        physical_reality=o2_reality,
        ceiling=o2_ceiling,
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
            # Same-depth neighbours: a floor value above or below ordinary water is a
            # nitracline / OMZ edge on a coarse vertical grid, not an artefact. A floor
            # value between two ordinary levels is never a natural profile, so it counts.
            v_high = []
            for e, (axis, step) in zip(elig, _DIRS, strict=True):
                hi = e & (_shift(a, axis, step, -np.inf) >= chk.floor_jump)
                if axis == -3:
                    v_high.append(hi)
                else:
                    high |= hi
            high |= v_high[0] & v_high[1]
            flags |= np.uint8(FLAG_FLOOR_JUMP) * (at_floor & high)
        if chk.max_jump is not None:
            if chk.normalize_by_salinity:
                x, valid = _normalize(a, block.salt, wet)
                x = np.where(valid, x, np.nan)
            else:
                valid = wet
                x = a
            lo, hi = _spike_envelope(x, valid, elig, chk.max_jump)
            jump = valid & ((x < lo) | (x > hi))
            flags |= np.uint8(FLAG_JUMP) * jump
        if chk.physical_reality is not None:
            ok = chk.physical_reality(block)
            flags |= np.uint8(FLAG_PHYSICAL) * (wet & ~ok)
    return flags


def _normalize(
    a: np.ndarray, salt: np.ndarray, wet: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``E + (a - E) * S_REF / S`` with E = FRESHWATER_ENDMEMBER where S >= S_MIN_NORMALIZE
    (raw elsewhere), and the mask of where it applies.
    """
    valid = wet & (salt >= S_MIN_NORMALIZE)
    e = FRESHWATER_ENDMEMBER
    return np.where(valid, e + (a - e) * S_REF / np.where(valid, salt, 1.0), a), valid


def _spike_envelope(
    x: np.ndarray, valid: np.ndarray, elig: list[np.ndarray], max_jump: float
) -> tuple[np.ndarray, np.ndarray]:
    """Range a cell may take without being a spike: within ``max_jump`` of the range of its
    eligible horizontal neighbours (when there are at least two) and of its two vertical
    neighbours (when both exist). Unconstrained directions give -inf/inf.

    A spike lies outside the range of its neighbours by more than max_jump; a cell inside a
    steep but smooth gradient, or beside a front, does not. Vertical (rows 0-1 of _DIRS)
    and horizontal (rows 2-5) are judged apart, so a bad column (whose vertical neighbours
    agree with it) is still a horizontal spike at every level, and a bad layer a vertical
    one.
    """
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
    ok = ~np.isnan(nbrs)
    top = np.where(ok, nbrs, -np.inf)
    bot = np.where(ok, nbrs, np.inf)
    h_on = ok[2:].sum(axis=0) >= 2
    v_on = ok[:2].sum(axis=0) == 2
    lo = np.maximum(
        np.where(h_on, bot[2:].min(axis=0) - max_jump, -np.inf),
        np.where(v_on, bot[:2].min(axis=0) - max_jump, -np.inf),
    )
    hi = np.minimum(
        np.where(h_on, top[2:].max(axis=0) + max_jump, np.inf),
        np.where(v_on, top[:2].max(axis=0) + max_jump, np.inf),
    )
    return lo, hi


def _grow_infeasible(
    a: np.ndarray,
    holes: np.ndarray,
    salt: np.ndarray,
    wet: np.ndarray,
    elig: list[np.ndarray],
    max_jump: float,
    normalize: bool,
) -> np.ndarray:
    """Add the vertical neighbours of holes that no value can fix to the holes.

    A hole whose horizontal and vertical ranges do not overlap sits in a column offset
    from its surroundings by more than twice ``max_jump``; its column neighbours share the
    offset, so they are filled too and the fill spreads along the column over the passes.
    Vertical only: the s axis is never split, so this adds no tile overlap.
    """
    x, valid = _normalize(a, salt, wet) if normalize else (a, wet)
    lo, hi = _spike_envelope(x, valid, elig, max_jump)
    stuck = holes & (lo > hi)
    up_down = _shift(stuck, -3, 1, False) | _shift(stuck, -3, -1, False)
    return holes | (up_down & valid)


def _project_into_envelope(
    x: np.ndarray,
    holes: np.ndarray,
    valid: np.ndarray,
    elig: list[np.ndarray],
    max_jump: float,
) -> np.ndarray:
    """Clip filled cells into their spike envelope (its midpoint when the horizontal and
    vertical ranges do not overlap, which minimises the larger violation).
    """
    lo, hi = _spike_envelope(x, valid, elig, max_jump)
    m = PROJECT_MARGIN * max_jump
    with np.errstate(invalid="ignore"):  # -inf + inf where nothing constrains a cell
        target = np.where(lo + 2 * m <= hi, np.clip(x, lo + m, hi - m), 0.5 * (lo + hi))
    return np.where(holes & valid & np.isfinite(target), target, x)


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


def _fill_within(
    a: np.ndarray,
    holes: np.ndarray,
    region: np.ndarray,
    z: np.ndarray,
    elig: list[np.ndarray],
    column_reach: int,
    relax_sweeps: int,
) -> np.ndarray:
    """:func:`_fill` restricted to ``region``: its holes are filled from its own cells
    only, and cells outside it are neither filled nor donors.
    """
    elig_in = [
        e & region & _shift(region, axis, step, False)
        for e, (axis, step) in zip(elig, _DIRS, strict=True)
    ]
    return _fill(a, holes & region, region, z, elig_in, column_reach, relax_sweeps)


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
    stack: np.ndarray,
    aux: np.ndarray,
    *,
    names: tuple[str, ...],
    targets: tuple[str, ...],
    checks: Mapping[str, BgcCheck],
    mode: str,
    stage: tuple[str, ...] = (),
    dz_max: float = DZ_MAX,
    column_reach: int = COLUMN_REACH,
    relax_sweeps: int = RELAX_SWEEPS,
) -> np.ndarray:
    """Check (mode="check": uint8 flags) or groom ``stage`` (mode="groom": values).

    ``stack`` is the tracers stacked on axis 0 and ``aux`` (salt, temp, z, wet), each
    (time, s, y, x). ``names`` are all tracers in the stack; ``targets`` the ones checked
    (the rest are inputs a check requires). The output has the shape of ``stack``: the
    groomed tracers (mode="groom") or flags, zero for non-targets (mode="check"). ``aux``
    is never returned, so no pass copies it.
    """
    salt, temp, z, wetf = aux
    wet = wetf > 0.5
    values = {n: stack[i] for i, n in enumerate(names)}
    elig = _eligible(wet, z, dz_max)
    if mode == "check":
        block = Block(values, salt, temp, wet)
        flags = np.zeros(stack.shape, dtype=np.uint8)
        for n in targets:
            flags[names.index(n)] = _detect(n, checks[n], block, elig)
        return flags
    out = stack.copy()
    for n in stage:
        block = Block(values, salt, temp, wet)
        holes = _detect(n, checks[n], block, elig) > 0
        if (jmp := checks[n].max_jump) is not None:
            holes = _grow_infeasible(
                values[n], holes, salt, wet, elig, jmp, checks[n].normalize_by_salinity
            )
        if checks[n].normalize_by_salinity:
            # Fill the salinity-normalized field, so the infill follows salinity and meets
            # the same normalized jump test. Water fresher than S_MIN_NORMALIZE is not
            # normalized, so it is filled apart, in raw values from fresh donors only: no
            # fill mixes the two spaces. Only the hole cells are replaced.
            x, valid = _normalize(values[n], salt, wet)
            x = _fill_within(x, holes, valid, z, elig, column_reach, relax_sweeps)
            if (jmp := checks[n].max_jump) is not None:
                x = _project_into_envelope(x, holes, valid, elig, jmp)
            e = FRESHWATER_ENDMEMBER
            back = e + (x - e) * np.where(valid, salt, S_REF) / S_REF
            fresh = wet & ~valid
            raw = _fill_within(
                values[n], holes, fresh, z, elig, column_reach, relax_sweeps
            )
            values[n] = np.where(holes & valid, back, np.where(holes, raw, values[n]))
        else:
            values[n] = _fill(
                values[n], holes, wet, z, elig, column_reach, relax_sweeps
            )
            if (jmp := checks[n].max_jump) is not None:
                values[n] = _project_into_envelope(values[n], holes, wet, elig, jmp)
        if (ceiling := checks[n].ceiling) is not None:
            # Cap every wet cell, not only holes: a value above the ceiling is not
            # plausible anywhere. A NaN value or ceiling (no physics) compares False,
            # so it is left as is.
            cap = ceiling(Block(values, salt, temp, wet)) * (1.0 - CEILING_MARGIN)
            with np.errstate(invalid="ignore"):
                over = wet & (values[n] > cap)
            values[n] = np.where(over, cap, values[n])
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


def _with_requires(
    ds: xr.Dataset,
    checks: Mapping[str, BgcCheck],
    targets: tuple[str, ...],
    suffix: str = "",
) -> tuple[str, ...]:
    """Targets plus the tracers their checks require (inputs only, never groomed)."""
    extra = []
    for t in targets:
        for r in checks[t].requires:
            if r in targets or r in extra:
                continue
            if f"{r}{suffix}" not in ds:
                raise ValueError(
                    f"The {t} check requires {r}{suffix}, which is not in the data."
                )
            extra.append(r)
    return targets + tuple(extra)


def _as_dask(da: xr.DataArray, chunks: Any = -1) -> dsa.Array:
    """float64 dask view of ``da``; numpy data is chunked as ``chunks`` when wrapped, so
    no task ever holds a whole-domain float64 copy.
    """
    data = da.data
    if not isinstance(data, dsa.Array):
        data = dsa.from_array(np.asarray(data), chunks=chunks)
    return data.astype(np.float64)


# --------------------------------------------------------------------------- engine


def _overlap_reach(column_reach: int, relax_sweeps: int) -> int:
    """Cells a groom pass can reach: detection (1) + column reach + relaxation +
    envelope projection (1).
    """
    return 2 + column_reach + relax_sweeps


def _run(
    stack: dsa.Array,
    aux: dsa.Array,
    names: tuple[str, ...],
    targets: tuple[str, ...],
    checks: Mapping[str, BgcCheck],
    *,
    mode: str,
    dz_max: float,
    passes: int,
    column_reach: int,
    relax_sweeps: int,
) -> dsa.Array:
    """Chain lazy map_overlap passes; returns (tracer, time, s, y, x) values or flags.

    The aux rows (salt, temp, z, wet) are a separate, unchanged input to every pass,
    so no pass copies them.
    """
    stack = stack.rechunk({0: -1})
    aux = aux.rechunk((-1, *stack.chunks[1:]))
    kernel = partial(
        _kernel,
        names=names,
        targets=targets,
        checks=checks,
        dz_max=dz_max,
        column_reach=column_reach,
        relax_sweeps=relax_sweeps,
    )

    def depth_for(reach: int) -> dict[int, int]:
        # Only axes split into several chunks need an overlap; a single chunk already
        # holds every neighbour, so the result is the same either way.
        return {
            ax: (reach if ax >= 3 and len(stack.chunks[ax]) > 1 else 0)
            for ax in range(5)
        }

    if mode == "check":
        return dsa.map_overlap(
            partial(kernel, mode="check"),
            stack,
            aux,
            depth=depth_for(1),
            boundary="none",
            align_arrays=False,
            dtype=np.uint8,
            meta=np.array((), dtype=np.uint8),
        )
    depth = depth_for(_overlap_reach(column_reach, relax_sweeps))
    for _ in range(passes):
        for stage in _stages(targets, checks):
            stack = dsa.map_overlap(
                partial(kernel, mode="groom", stage=tuple(stage)),
                stack,
                aux,
                depth=depth,
                boundary="none",
                align_arrays=False,
                dtype=np.float64,
                meta=np.array((), dtype=np.float64),
            )
    return stack


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
        return _as_dask(
            da.transpose(tdim, "s_rho", "eta_rho", "xi_rho"), chunks
        ).rechunk(chunks)

    stack = dsa.stack([arr(ds[n]) for n in names]).rechunk((-1, *chunks))
    # Chunk the grid first so the 3-D layer depths stay lazy, tile by tile.
    gch = grid_ds.chunk({"eta_rho": tile, "xi_rho": tile})
    z = compute_depth_coordinates(gch, zeta=0, depth_type="layer", location="rho")
    z = _as_dask(z.transpose("s_rho", "eta_rho", "xi_rho"))
    wet = _as_dask(grid_ds["mask_rho"].transpose("eta_rho", "xi_rho"), (tile, tile))
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
        return _as_dask(da.transpose(*dims), (time_chunk, -1, -1))[
            :, :, None, :
        ].rechunk(chunks)

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
        targets = _select(ds, checks, variables)
        if not targets:
            raise ValueError("No checkable BGC variables found in the dataset.")
        names = _with_requires(ds, checks, targets)
        params = _params(checks, targets, dz_max, passes, column_reach, relax_sweeps)
        stack, aux, dims = _ic_arrays(ds, physics, grid_ds, names, tile)
        res = _run(stack, aux, names, targets, checks, **run)
        for n in targets:
            _store(out, ds, n, res[names.index(n)], dims, mode, checks, n, params)
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
            targets = _select(ds, checks, variables, suffix=f"_{side}")
            if not targets:
                continue
            all_names.update(targets)
            names = _with_requires(ds, checks, targets, suffix=f"_{side}")
            stack, aux, dims = _bry_arrays(ds, phys, grid_ds, names, side, time_chunk)
            res = _run(stack, aux, names, targets, checks, **run)
            for n in targets:
                _store(
                    out,
                    ds,
                    f"{n}_{side}",
                    res[names.index(n)][:, :, 0, :],
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

#: ``groom_settings`` keys that set fill parameters (passed to :func:`bgc_groom`).
GROOM_PARAMETERS = ("passes", "relax_sweeps", "column_reach", "dz_max")

#: Per-check fields a ``groom_settings["checks"]`` entry may override. ``physical_reality``
#: and ``ceiling`` are functions and can only be switched off (``None``).
_CHECK_OVERRIDES = (
    "bounds",
    "floor",
    "floor_jump",
    "max_jump",
    "normalize_by_salinity",
    "physical_reality",
    "ceiling",
)


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    """A finite builtin int/float (not bool, not a numpy scalar, which a YAML cannot hold)."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(
            f"groom_settings: {label} must be a finite number, got {value!r}."
        )
    if positive and value <= 0:
        raise ValueError(f"groom_settings: {label} must be positive, got {value!r}.")
    return float(value)


def _check_count(name: str, value: Any, prefix: str = "") -> int:
    """A non-negative builtin int (not bool): ``passes``, ``relax_sweeps``,
    ``column_reach``.
    """
    if type(value) is not int or value < 0:
        raise ValueError(f"{prefix}{name} must be a non-negative int, got {value!r}.")
    return value


def resolve_groom_settings(
    settings: Mapping[str, Any] | None,
) -> tuple[dict[str, BgcCheck], dict[str, Any]]:
    """Validate a ``groom_settings`` dict; return the checks and fill parameters it gives.

    ``settings`` may hold ``passes``, ``relax_sweeps``, ``column_reach`` (non-negative
    ints), ``dz_max`` (a positive number) and ``checks``: ``{tracer: {field: value}}``,
    partial overrides of :data:`DEFAULT_CHECKS` for ``bounds`` (``[low, high]`` or
    ``None``), ``floor``, ``floor_jump``, ``max_jump`` (a number or ``None``),
    ``normalize_by_salinity`` (bool), and ``physical_reality``/``ceiling`` (``None`` only,
    to switch them off). ``None`` returns the defaults. Anything else raises.
    """
    checks = dict(DEFAULT_CHECKS)
    if settings is None:
        return checks, {}
    if not isinstance(settings, Mapping):
        raise ValueError(
            f"groom_settings must be a dict or None, got {type(settings).__name__}."
        )
    unknown = set(settings) - {*GROOM_PARAMETERS, "checks"}
    if unknown:
        raise ValueError(
            f"groom_settings: unknown key(s) {sorted(unknown)}; valid keys are "
            f"{[*GROOM_PARAMETERS, 'checks']}."
        )
    params: dict[str, Any] = {}
    for k in ("passes", "relax_sweeps", "column_reach"):
        if k in settings:
            params[k] = _check_count(k, settings[k], "groom_settings: ")
    if "dz_max" in settings:
        params["dz_max"] = _number(settings["dz_max"], "dz_max", positive=True)

    overrides = settings.get("checks")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ValueError("groom_settings: 'checks' must be a dict of {tracer: {...}}.")
    for var, fields in overrides.items():
        if var not in checks:
            raise ValueError(
                f"groom_settings: no check for {var!r}; known: {sorted(checks)}."
            )
        if not isinstance(fields, Mapping):
            raise ValueError(f"groom_settings: checks[{var!r}] must be a dict.")
        bad = set(fields) - set(_CHECK_OVERRIDES)
        if bad:
            raise ValueError(
                f"groom_settings: checks[{var!r}] has unknown field(s) {sorted(bad)}; "
                f"valid fields are {list(_CHECK_OVERRIDES)}."
            )
        upd: dict[str, Any] = {}
        label = f"checks[{var!r}]"
        for f, v in fields.items():
            if f == "bounds":
                if v is None:
                    upd[f] = None
                    continue
                if type(v) is not list or len(v) != 2:
                    raise ValueError(
                        f"groom_settings: {label}.bounds must be [low, high] or None."
                    )
                lo, hi = (_number(x, f"{label}.bounds") for x in v)
                if not lo < hi:
                    raise ValueError(
                        f"groom_settings: {label}.bounds needs low < high, got {v!r}."
                    )
                upd[f] = (lo, hi)
            elif f == "floor":
                upd[f] = _number(v, f"{label}.floor")
            elif f in ("floor_jump", "max_jump"):
                upd[f] = (
                    None if v is None else _number(v, f"{label}.{f}", positive=True)
                )
            elif f == "normalize_by_salinity":
                if type(v) is not bool:
                    raise ValueError(f"groom_settings: {label}.{f} must be a bool.")
                upd[f] = v
            else:  # physical_reality, ceiling
                if v is not None:
                    raise ValueError(
                        f"groom_settings: {label}.{f} can only be switched off (null)."
                    )
                upd[f] = None
                if f == "physical_reality":
                    upd["requires"] = ()  # requires only feeds physical_reality
        checks[var] = replace(checks[var], **upd)
    return checks, params


def _dispatch(
    source: Any, grid: Any, physics: Any, kw: dict[str, Any]
) -> xr.Dataset | list[xr.Dataset]:
    """One result per BGC source for a ``BoundaryForcing`` (its BGC data lives in
    ``bf.bgc[i].ds``, its salt/temp in ``bf.physics.ds``); one Dataset otherwise,
    including an ``InitialConditions``, whose merged ``.ds`` carries salt and temp.
    """
    from roms_tools.setup.boundary_forcing import BoundaryForcing

    if isinstance(source, BoundaryForcing):
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
    return _dispatch(source, grid, physics, kw)


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
    """Return a lazily groomed copy of BGC data in a Dataset or NetCDF file(s); the
    input is never modified.

    A tool for analysis and one-off repair of existing files. To groom the inputs roms-tools
    builds, use the ``groom`` and ``groom_settings`` arguments of
    :class:`~roms_tools.InitialConditions` and :class:`~roms_tools.BoundaryForcing`: they
    are recorded in the YAML, so the file can be regenerated from it. Those objects are
    therefore refused here (:func:`bgc_check` accepts them). Grooming a file directly does
    not touch derived tracers such as ``DIC_ALT_CO2``/``ALK_ALT_CO2``.

    Flagged cells are replaced by a fill from the surrounding good water, ``passes`` times
    (each pass re-checks). ``passes``, ``relax_sweeps``, ``column_reach`` and ``dz_max``
    define the result and are recorded in the attrs; ``tile`` and ``time_chunk`` only set
    task size and do not change it. Run :func:`bgc_check` on the result for what remains.
    Boundary data needs ``physics`` (the matching physics boundary data, for salt/temp).

    Peak memory grows with the domain, not only the tile.
    """
    from roms_tools.setup.boundary_forcing import BoundaryForcing
    from roms_tools.setup.initial_conditions import InitialConditions

    if isinstance(source, InitialConditions | BoundaryForcing):
        raise TypeError(
            f"bgc_groom does not take a {type(source).__name__} object: a groomed file "
            "its YAML cannot regenerate would result. Pass groom=True (and optionally "
            f"groom_settings) to {type(source).__name__} instead, or groom a Dataset or "
            "NetCDF file."
        )
    for k, v in (
        ("passes", passes),
        ("relax_sweeps", relax_sweeps),
        ("column_reach", column_reach),
    ):
        _check_count(k, v)
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
    return _dispatch(source, grid, physics, kw)


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
