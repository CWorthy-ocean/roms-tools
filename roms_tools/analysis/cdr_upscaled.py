import logging
import operator
from collections import Counter
from dataclasses import dataclass, field
from functools import reduce
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.path import Path as MplPath

from roms_tools.analysis.cdr_analysis import SOURCE_ROLES

_FLUX_UPTAKE = "cdr_carbon_uptake_from_flux"
_DIC_UPTAKE = "cdr_carbon_uptake_from_DIC_difference"
_CUMULATIVE_SOURCE = "cdr_cumulative_source"

# Time series that are aligned and combined.
_SERIES_VARS = (_CUMULATIVE_SOURCE, _FLUX_UPTAKE, _DIC_UPTAKE)
_REQUIRED_VARS = (*_SERIES_VARS, "area", "hDIC", "hDIC_ALT_CO2", "window_length")
_REQUIRED_COORDS = ("time", "lat_rho", "lon_rho")

_FLUX_EFFICIENCY = "cdr_efficiency_from_flux"
_DIC_EFFICIENCY = "cdr_efficiency_from_DIC_difference"
_CLOSURE_RESIDUAL = "closure_residual"
# Variables of the combined dataset whose names do not depend on the members.
_COMBINED_VARS = (*_SERIES_VARS, _FLUX_EFFICIENCY, _DIC_EFFICIENCY, _CLOSURE_RESIDUAL)


def _member_var(member: str, var: str) -> str:
    """Name of a member's own `var` series in the combined dataset."""
    return f"{member}_{var.removeprefix('cdr_')}"


def _footprint_var(parent: str, child: str) -> str:
    """Name of the DIC perturbation of `parent` inside `child`'s footprint."""
    return f"{parent}_DIC_difference_inside_{child}"


def _unwrap_lon(lon: np.ndarray, ref: float) -> np.ndarray:
    """Shift longitudes by multiples of 360 degrees into ``[ref - 180, ref + 180)``."""
    return (lon - ref + 180) % 360 - 180 + ref


@dataclass
class UpscaledCDR:
    """
    Combines the CDR metrics of a chain of nested domains into one total.

    In a nested (upscaled) CDR experiment the actual release happens in a child
    domain; each parent domain receives the child's boundary perturbation
    re-injected as a source. Every member is a ``ds_cdr`` as produced by
    :func:`roms_tools.analysis.cdr_analysis.compute_cdr_metrics` (or
    ``ROMSOutput.cdr_metrics``) for its own domain, computed with that domain's
    own grid and the matching ``source_role``.

    Methodology:

    - Members are ordered child -> parent, i.e. each member's footprint must lie
      strictly inside the next one's.
    - The total carbon uptake is the sum of the members' uptakes (flux-based and
      DIC-difference-based separately): each domain removes the part of the
      perturbation that is in it.
    - The only real source is the release. The denominator of the combined
      efficiency is therefore the cumulative source of the members with
      ``cdr_source_role == "release"`` only. The source of a ``"transfer"``
      member is the child's perturbation passed on to the parent (it is signed,
      ``ALK_add = ALK_rate - ALK_alt_rate``), not new alkalinity, so adding it to
      the denominator would count the same release twice.
    - The summed uptake is converted back to native units with the members'
      shared ``tonnes_co2_per_native_unit`` before dividing, so the efficiency is
      the native-unit uptake over the native-unit source.

    Members must share the same averaging window (``window_length``) and the same
    native carbon units (``tonnes_co2_per_native_unit``). Longitudes are compared
    after shifting them by multiples of 360 degrees to within 180 degrees of the
    child's centre, so domains may cross the Greenwich meridian or the dateline
    and members may store longitudes in either [0, 360) or [-180, 180).
    Members are aligned on the union of their absolute times; where any member
    lacks a record the combined series are NaN there, and a warning is logged.

    Two diagnostics help to judge the result:

    - ``closure_residual``: combined flux-based uptake minus combined
      DIC-difference-based uptake (tonnes CO2). The two are bookkeeping views of
      the same perturbation, so this should be small.
    - ``{parent}_DIC_difference_inside_{child}``: the DIC perturbation (tonnes CO2)
      that a parent carries in the cells whose centre lies inside its child's
      area (the child's rho points extended by half a cell, following the
      child's orientation). The child already accounts for that region, so a
      nonzero value indicates double counting.

    Parameters
    ----------
    members : dict[str, str | Path | xr.Dataset]
        Dictionary mapping member names to either file paths (NetCDF) or
        xarray.Dataset objects holding the member's ``ds_cdr``. Order matters:
        child first, then its parent, then the parent's parent, and so on. At
        least two members are required, and at least one must have the role
        ``"release"``. The datasets need the ``lat_rho``/``lon_rho`` coordinates
        and the variables and attributes written by ``compute_cdr_metrics``.

    Raises
    ------
    ValueError
        If fewer than two members are given, a member lacks required variables,
        coordinates or attributes, no member is a release, the members are not
        nested in the given order or share the same grid, the averaging windows
        differ, the release members' source units differ, or a member name would
        give a variable name that clashes with another one in the combined dataset
        (e.g. a member called ``"cdr"``).
    """

    members: dict[str, str | Path | xr.Dataset]
    """Dictionary mapping member names (child first) to their CDR metrics."""
    ds: xr.Dataset = field(init=False)
    """Combined uptake, efficiency, per-member series and diagnostics."""

    def __post_init__(self):
        """Load and validate the members, align their times, and build ``self.ds``."""
        datasets = self._load_members()
        self._validate_members(datasets)
        aligned = self._align_times(datasets)
        self.ds = self._build_dataset(datasets, aligned)

    def _load_members(self) -> dict[str, xr.Dataset]:
        """
        Load the member datasets.

        Converts any file paths in `self.members` to xarray Datasets. Members that
        are already xarray Datasets are left unchanged.

        Returns
        -------
        dict[str, xr.Dataset]
            Dictionary mapping member names to xarray.Dataset objects.
        """
        return {
            name: xr.open_dataset(path) if isinstance(path, str | Path) else path
            for name, path in self.members.items()
        }

    def _validate_members(self, datasets: dict[str, xr.Dataset]) -> None:
        """
        Check that the members form a consistent child -> parent chain.

        Raises
        ------
        ValueError
            On any misconfiguration; the message names the offending member(s).
        """
        if len(datasets) < 2:
            raise ValueError(
                "UpscaledCDR needs a child and at least one parent member, "
                f"got {len(datasets)}."
            )

        hint = (
            "recompute the member with `compute_cdr_metrics` from a roms-tools "
            "version that writes these"
        )
        for name, ds in datasets.items():
            missing = [v for v in _REQUIRED_VARS if v not in ds.data_vars]
            missing += [c for c in _REQUIRED_COORDS if c not in ds.coords]
            if "cdr_source_role" not in ds.attrs:
                missing.append("attribute 'cdr_source_role'")
            missing += [
                f"attribute 'tonnes_co2_per_native_unit' on '{v}'"
                for v in (_FLUX_UPTAKE, _DIC_UPTAKE)
                if v in ds and "tonnes_co2_per_native_unit" not in ds[v].attrs
            ]
            if missing:
                raise ValueError(f"Member {name!r} is missing {missing}; {hint}.")
            if ds.attrs["cdr_source_role"] not in SOURCE_ROLES:
                raise ValueError(
                    f"Member {name!r} has cdr_source_role="
                    f"{ds.attrs['cdr_source_role']!r}; expected one of {SOURCE_ROLES}."
                )

        if not any(
            ds.attrs["cdr_source_role"] == "release" for ds in datasets.values()
        ):
            raise ValueError(
                "At least one member must have cdr_source_role='release'; got "
                f"{ {n: d.attrs['cdr_source_role'] for n, d in datasets.items()} }."
            )

        names = list(datasets)
        var_names = Counter(
            [
                *_COMBINED_VARS,
                *(
                    _member_var(n, v)
                    for n in names
                    for v in (_FLUX_UPTAKE, _DIC_UPTAKE)
                ),
                *(_footprint_var(p, c) for c, p in pairwise(names)),
            ]
        )
        clashes = sorted(v for v, count in var_names.items() if count > 1)
        if clashes:
            raise ValueError(
                f"Member names {names} give clashing variable names {clashes} in "
                "the combined dataset; rename the members."
            )

        self._validate_geometry(datasets)

        window_lengths = {}
        for name, ds in datasets.items():
            values = ds["window_length"].values
            # A member read back from NetCDF may have been decoded to timedelta64.
            if np.issubdtype(values.dtype, np.timedelta64):
                values = values / np.timedelta64(1, "s")
            if not np.allclose(values, values[0]):
                raise ValueError(
                    f"Member {name!r} must have a single unique window_length, "
                    f"got {np.unique(values)}."
                )
            window_lengths[name] = float(values[0])
        first = next(iter(window_lengths.values()))
        if not np.allclose(list(window_lengths.values()), first):
            raise ValueError(
                f"Members must share the same averaging window; got {window_lengths}."
            )

        for var in (_FLUX_UPTAKE, _DIC_UPTAKE):
            scales = {
                n: ds[var].attrs["tonnes_co2_per_native_unit"]
                for n, ds in datasets.items()
            }
            if not np.allclose(list(scales.values()), next(iter(scales.values()))):
                raise ValueError(
                    f"Members must share the same native carbon units for {var!r} "
                    f"(tonnes_co2_per_native_unit); got {scales}."
                )

    @staticmethod
    def _centre_lon(ds: xr.Dataset) -> float:
        """Longitude of the rho point at the centre of a member's grid."""
        lon = ds["lon_rho"].values
        return float(lon[lon.shape[0] // 2, lon.shape[1] // 2])

    @staticmethod
    def _bbox(ds: xr.Dataset, ref: float) -> tuple[float, float, float, float]:
        """
        Return (lon_min, lon_max, lat_min, lat_max) of a member's rho points.

        Longitudes are unwrapped to within 180 degrees of `ref` first.
        """
        lon, lat = _unwrap_lon(ds["lon_rho"].values, ref), ds["lat_rho"].values
        return (
            float(np.nanmin(lon)),
            float(np.nanmax(lon)),
            float(np.nanmin(lat)),
            float(np.nanmax(lat)),
        )

    def _validate_geometry(self, datasets: dict[str, xr.Dataset]) -> None:
        """
        Check that members have distinct grids and are nested child -> parent.

        Longitudes are unwrapped to within 180 degrees of the (smaller) member's
        centre before comparing, so grids crossing 0E or 180E are handled.
        """
        names = list(datasets)
        for i, a in enumerate(names):
            ref = self._centre_lon(datasets[a])
            for b in names[i + 1 :]:
                da, db = datasets[a], datasets[b]
                same = (
                    da["lon_rho"].shape == db["lon_rho"].shape
                    and np.allclose(
                        _unwrap_lon(da["lon_rho"].values, ref),
                        _unwrap_lon(db["lon_rho"].values, ref),
                    )
                    and np.allclose(da["lat_rho"].values, db["lat_rho"].values)
                )
                if same:
                    raise ValueError(
                        f"Members {a!r} and {b!r} have identical lon_rho/lat_rho: "
                        "the same grid appears to have been used for both members. "
                        "Compute each member's metrics with its own grid."
                    )

        for child, parent in pairwise(names):
            ref = self._centre_lon(datasets[child])
            c_lon0, c_lon1, c_lat0, c_lat1 = self._bbox(datasets[child], ref)
            p_lon0, p_lon1, p_lat0, p_lat1 = self._bbox(datasets[parent], ref)
            if not (
                p_lon0 < c_lon0
                and p_lon1 > c_lon1
                and p_lat0 < c_lat0
                and p_lat1 > c_lat1
            ):
                raise ValueError(
                    f"Member {parent!r} must strictly contain the lon/lat bounding "
                    f"box of {child!r}; members must be ordered child -> parent. "
                    f"{parent!r}: lon [{p_lon0}, {p_lon1}], lat [{p_lat0}, {p_lat1}]; "
                    f"{child!r}: lon [{c_lon0}, {c_lon1}], lat [{c_lat0}, {c_lat1}]."
                )

    def _align_times(self, datasets: dict[str, xr.Dataset]) -> dict[str, xr.Dataset]:
        """
        Align the members' time series on the union of their absolute times.

        Records that some member lacks are NaN for that member; a single warning
        reports how many records are affected and which members are shorter.

        Returns
        -------
        dict[str, xr.Dataset]
            For each member, its cumulative source and uptake series on the common
            time axis.
        """
        all_times = np.unique(
            np.concatenate([ds["time"].values for ds in datasets.values()])
        )
        incomplete = np.zeros(all_times.size, dtype=bool)
        for ds in datasets.values():
            incomplete |= ~np.isin(all_times, ds["time"].values)
        n_incomplete = int(incomplete.sum())
        if n_incomplete:
            shorter = [
                n for n, ds in datasets.items() if ds["time"].size < all_times.size
            ]
            logging.warning(
                "%d time record(s) are missing from at least one member; combined "
                "series are NaN there. Members with fewer records: %s.",
                n_incomplete,
                ", ".join(shorter),
            )
        return {
            name: ds[list(_SERIES_VARS)].reindex(time=all_times)
            for name, ds in datasets.items()
        }

    @staticmethod
    def _footprint(ds: xr.Dataset, ref: float) -> MplPath:
        """
        Perimeter of a member's area as a lon/lat polygon.

        Longitudes are unwrapped to within 180 degrees of `ref`. The rho points
        are cell centres, so each row and column is extrapolated by half a step to
        reach the cell edges before the outer ring is traced. This follows the
        grid's own orientation, unlike a lon/lat bounding box.
        """

        def extend(a: np.ndarray) -> np.ndarray:
            a = np.concatenate(
                [a[:1] - 0.5 * (a[1:2] - a[:1]), a, a[-1:] + 0.5 * (a[-1:] - a[-2:-1])]
            )
            return np.concatenate(
                [
                    a[:, :1] - 0.5 * (a[:, 1:2] - a[:, :1]),
                    a,
                    a[:, -1:] + 0.5 * (a[:, -1:] - a[:, -2:-1]),
                ],
                axis=1,
            )

        lon = extend(_unwrap_lon(ds["lon_rho"].values, ref))
        lat = extend(ds["lat_rho"].values)

        def ring(a: np.ndarray) -> np.ndarray:
            return np.concatenate([a[0, :], a[1:, -1], a[-1, -2::-1], a[-2:0:-1, 0]])

        return MplPath(np.column_stack([ring(lon), ring(lat)]))

    def _footprint_dic_difference(
        self, child: xr.Dataset, parent: xr.Dataset, all_times: np.ndarray
    ) -> xr.DataArray:
        """
        DIC perturbation (tonnes CO2) of `parent` inside `child`'s footprint.

        Sums ``(hDIC - hDIC_ALT_CO2) * area`` over the parent cells whose centre
        lies within the child's area (see `_footprint`).
        """
        ref = self._centre_lon(child)
        plon = _unwrap_lon(parent["lon_rho"].values, ref)
        plat = parent["lat_rho"].values
        inside_values = self._footprint(child, ref).contains_points(
            np.column_stack([plon.ravel(), plat.ravel()])
        )
        inside = xr.DataArray(
            inside_values.reshape(plon.shape), dims=parent["lon_rho"].dims
        )
        diff = ((parent["hDIC"] - parent["hDIC_ALT_CO2"]) * parent["area"]).sum("s_rho")
        scale = parent[_DIC_UPTAKE].attrs["tonnes_co2_per_native_unit"]
        footprint = (
            diff.where(inside).sum(["eta_rho", "xi_rho"]).compute() * scale
        ).reindex(time=all_times)
        return footprint.reset_coords(drop=True)

    def _build_dataset(
        self, datasets: dict[str, xr.Dataset], aligned: dict[str, xr.Dataset]
    ) -> xr.Dataset:
        """Assemble the combined dataset from the aligned member series."""
        names = list(datasets)
        roles = {n: datasets[n].attrs["cdr_source_role"] for n in names}
        releases = [n for n in names if roles[n] == "release"]
        all_times = aligned[names[0]]["time"].values

        def total(var: str, members: list[str]) -> xr.DataArray:
            return reduce(operator.add, [aligned[n][var] for n in members])

        source_units = {
            datasets[n][_CUMULATIVE_SOURCE].attrs.get("units") for n in releases
        }
        if len(source_units) != 1:
            raise ValueError(
                f"Release members must share the same {_CUMULATIVE_SOURCE} units, "
                f"got {sorted(map(str, source_units))}."
            )

        out = xr.Dataset(coords={"time": all_times})

        out[_CUMULATIVE_SOURCE] = total(_CUMULATIVE_SOURCE, releases)
        out[_CUMULATIVE_SOURCE].attrs.update(
            long_name="Cumulative CDR source (release members)",
            description=(
                "Time-cumulative volume-integrated (ALK_source - DIC_source) of the "
                "release members only; the denominator of the combined efficiency"
            ),
        )
        units = source_units.pop()
        if units is not None:
            out[_CUMULATIVE_SOURCE].attrs["units"] = units

        out[_FLUX_UPTAKE] = total(_FLUX_UPTAKE, names)
        out[_FLUX_UPTAKE].attrs.update(
            long_name="CDR carbon uptake (cumulative flux differences, all members)",
            units="tonnes CO2",
            description=(
                "Time-cumulative CO2 uptake mass from area-integrated CO2 flux "
                "differences, summed over all members"
            ),
        )
        out[_DIC_UPTAKE] = total(_DIC_UPTAKE, names)
        out[_DIC_UPTAKE].attrs.update(
            long_name="CDR carbon uptake (from DIC differences, all members)",
            units="tonnes CO2",
            description=(
                "CO2-equivalent mass from the volume-integrated DIC difference, "
                "summed over all members"
            ),
        )
        # Validation guarantees all members share the scale, so the sum keeps it.
        for var in (_FLUX_UPTAKE, _DIC_UPTAKE):
            out[var].attrs["tonnes_co2_per_native_unit"] = datasets[names[0]][
                var
            ].attrs["tonnes_co2_per_native_unit"]

        # Back to native units, so the ratio to the native source holds.
        for eff_var, uptake_var in (
            (_FLUX_EFFICIENCY, _FLUX_UPTAKE),
            (_DIC_EFFICIENCY, _DIC_UPTAKE),
        ):
            native = (
                out[uptake_var] / out[uptake_var].attrs["tonnes_co2_per_native_unit"]
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = native / out[_CUMULATIVE_SOURCE]
                out[eff_var] = ratio.where(np.isfinite(ratio))
        out[_FLUX_EFFICIENCY].attrs.update(
            long_name="CDR uptake efficiency (from flux differences)",
            units="nondimensional",
            description=(
                "Total uptake over all members (area-integrated CO2 flux "
                "differences) divided by the cumulative source of the release members"
            ),
        )
        out[_DIC_EFFICIENCY].attrs.update(
            long_name="CDR uptake efficiency (from DIC differences)",
            units="nondimensional",
            description=(
                "Total uptake over all members (volume-integrated DIC differences) "
                "divided by the cumulative source of the release members"
            ),
        )

        for n in names:
            for var in (_FLUX_UPTAKE, _DIC_UPTAKE):
                out[_member_var(n, var)] = aligned[n][var]
                out[_member_var(n, var)].attrs = dict(datasets[n][var].attrs)

        out[_CLOSURE_RESIDUAL] = out[_FLUX_UPTAKE] - out[_DIC_UPTAKE]
        out[_CLOSURE_RESIDUAL].attrs.update(
            long_name="Closure residual (flux-based minus DIC-based uptake)",
            units="tonnes CO2",
            description=(
                "Combined flux-based minus DIC-difference-based uptake; the two are "
                "bookkeeping views of the same perturbation, so this should be small"
            ),
        )
        diagnostics = {_CLOSURE_RESIDUAL: out[_CLOSURE_RESIDUAL]}

        for child, parent in pairwise(names):
            var = _footprint_var(parent, child)
            out[var] = self._footprint_dic_difference(
                datasets[child], datasets[parent], all_times
            )
            out[var].attrs.update(
                long_name=f"DIC perturbation of {parent} inside {child} footprint",
                units="tonnes CO2",
                description=(
                    f"DIC perturbation that {parent} carries in cells inside the "
                    f"area of {child}; double-counted if nonzero"
                ),
            )
            diagnostics[var] = out[var]

        for var, da in diagnostics.items():
            valid = da.dropna("time")
            if valid.size:
                logging.info(
                    "Final-time %s: %.3e tonnes CO2", var, float(valid.isel(time=-1))
                )

        out.attrs["cdr_source_role"] = "release"
        out.attrs["members"] = ", ".join(f"{n}:{roles[n]}" for n in names)
        return out

    def plot(
        self,
        save_path: str | None = None,
    ) -> None:
        """
        Plot the combined CDR efficiency and CO2 uptake in two subplots.

        The top panel shows the combined efficiency (flux-based and DIC-based); the
        bottom panel the combined uptake (both methods) with each member's
        flux-based uptake as a thinner line.

        Parameters
        ----------
        save_path : str, optional
            Path to save the generated plot. If None, the plot is shown interactively.
            Default is None.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.
        """
        fig, (ax_eff, ax_uptake) = plt.subplots(
            nrows=2, ncols=1, figsize=(9, 8), sharex=True
        )
        time = self.ds["time"].values

        ax_eff.plot(time, self.ds[_FLUX_EFFICIENCY], lw=2, label="flux differences")
        ax_eff.plot(time, self.ds[_DIC_EFFICIENCY], lw=2, label="DIC differences")
        ax_eff.set_ylabel("CDR efficiency")
        ax_eff.set_title("Combined CDR efficiency")
        ax_eff.grid()
        ax_eff.legend(loc="best")

        ax_uptake.plot(
            time, self.ds[_FLUX_UPTAKE], lw=2, label="total (flux differences)"
        )
        ax_uptake.plot(
            time, self.ds[_DIC_UPTAKE], lw=2, label="total (DIC differences)"
        )
        for name in self.members:
            ax_uptake.plot(
                time,
                self.ds[_member_var(name, _FLUX_UPTAKE)],
                lw=1,
                alpha=0.8,
                label=f"{name} (flux differences)",
            )
        ax_uptake.set_ylabel(r"CO$_2$ uptake (tonnes CO$_2$)")
        ax_uptake.set_xlabel("Time")
        ax_uptake.set_title(r"Combined CO$_2$ uptake")
        ax_uptake.grid()
        ax_uptake.legend(loc="best")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
