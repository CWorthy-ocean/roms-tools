import logging
import operator
from collections import Counter
from dataclasses import dataclass, field
from functools import reduce
from itertools import pairwise
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.path import Path as MplPath

from roms_tools.analysis.cdr_analysis import SOURCE_ROLES
from roms_tools.plot import _add_boundary_to_ax, _add_gridlines, get_projection

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
# Variables of the combined dataset whose names do not depend on the domains.
_COMBINED_VARS = (*_SERIES_VARS, _FLUX_EFFICIENCY, _DIC_EFFICIENCY, _CLOSURE_RESIDUAL)


def _domain_var(domain: str, var: str) -> str:
    """Name of a domain's own `var` series in the combined dataset."""
    return f"{domain}_{var.removeprefix('cdr_')}"


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
    re-injected as a source. Every domain is described by a ``ds_cdr`` as produced
    by :func:`roms_tools.analysis.cdr_analysis.compute_cdr_metrics` (or
    ``ROMSOutput.cdr_metrics``), computed with that domain's own grid and the
    matching ``source_role``.

    Methodology:

    - Domains are ordered child -> parent, i.e. each domain's footprint must lie
      strictly inside the next one's.
    - The total carbon uptake is the sum of the domains' uptakes (flux-based and
      DIC-difference-based separately): each domain removes the part of the
      perturbation that is in it.
    - The only real source is the release. The denominator of the combined
      efficiency is therefore the cumulative source of the domains with
      ``cdr_source_role == "release"`` only. The source of a ``"transfer"``
      domain is the child's perturbation passed on to the parent (it is signed,
      ``ALK_add = ALK_rate - ALK_alt_rate``), not new alkalinity, so adding it to
      the denominator would count the same release twice.
    - The summed uptake is converted back to native units with the domains'
      shared ``tonnes_co2_per_native_unit`` before dividing, so the efficiency is
      the native-unit uptake over the native-unit source.

    Domains must share the same averaging window (``window_length``) and the same
    native carbon units (``tonnes_co2_per_native_unit``). Longitudes are compared
    after shifting them by multiples of 360 degrees to within 180 degrees of the
    child's centre, so domains may cross the Greenwich meridian or the dateline
    and domains may store longitudes in either [0, 360) or [-180, 180).
    Domains are aligned on the union of their absolute times; where any domain
    lacks a record the combined series are NaN there, and a warning is logged.

    The combined dataset also carries two diagnostics:

    - ``closure_residual``: combined flux-based uptake minus combined
      DIC-difference-based uptake (tonnes CO2).
    - ``{parent}_DIC_difference_inside_{child}``: the DIC perturbation (tonnes CO2)
      that a parent carries in the cells whose centre lies inside its child's
      area (the child's rho points extended by half a cell, following the
      child's orientation). The child already accounts for that region, so a
      nonzero value indicates double counting.

    Parameters
    ----------
    domains : dict[str, str | Path | xr.Dataset]
        Dictionary mapping domain names to either file paths (NetCDF) or
        xarray.Dataset objects holding the domain's ``ds_cdr``. Order matters:
        child first, then its parent, then the parent's parent, and so on. At
        least two domains are required, and at least one must have the role
        ``"release"``. The datasets need the ``lat_rho``/``lon_rho`` coordinates
        and the variables and attributes written by ``compute_cdr_metrics``.

    Raises
    ------
    ValueError
        If fewer than two domains are given, a domain lacks required variables,
        coordinates or attributes, no domain is a release, the domains are not
        nested in the given order or share the same grid, the averaging windows
        differ, the release domains' source units differ, or a domain name would
        give a variable name that clashes with another one in the combined dataset
        (e.g. a domain called ``"cdr"``).
    """

    domains: dict[str, str | Path | xr.Dataset]
    """Dictionary mapping domain names (child first) to their CDR metrics."""
    ds: xr.Dataset = field(init=False)
    """Combined uptake, efficiency, per-domain series and diagnostics."""

    def __post_init__(self):
        """Load and validate the domains, align their times, and build ``self.ds``."""
        datasets = self._load_domains()
        self._validate_domains(datasets)
        aligned = self._align_times(datasets)
        self.ds = self._build_dataset(datasets, aligned)

    def _load_domains(self) -> dict[str, xr.Dataset]:
        """
        Load the domain datasets.

        Converts any file paths in `self.domains` to xarray Datasets. Domains that
        are already xarray Datasets are left unchanged.

        Returns
        -------
        dict[str, xr.Dataset]
            Dictionary mapping domain names to xarray.Dataset objects.
        """
        return {
            name: xr.open_dataset(path) if isinstance(path, str | Path) else path
            for name, path in self.domains.items()
        }

    def _validate_domains(self, datasets: dict[str, xr.Dataset]) -> None:
        """
        Check that the domains form a consistent child -> parent chain.

        Raises
        ------
        ValueError
            On any misconfiguration; the message names the offending domain(s).
        """
        if len(datasets) < 2:
            raise ValueError(
                "UpscaledCDR needs a child and at least one parent domain, "
                f"got {len(datasets)}."
            )

        hint = (
            "recompute the domain's metrics with `compute_cdr_metrics` from a "
            "roms-tools version that writes these"
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
                raise ValueError(f"Domain {name!r} is missing {missing}; {hint}.")
            if ds.attrs["cdr_source_role"] not in SOURCE_ROLES:
                raise ValueError(
                    f"Domain {name!r} has cdr_source_role="
                    f"{ds.attrs['cdr_source_role']!r}; expected one of {SOURCE_ROLES}."
                )

        if not any(
            ds.attrs["cdr_source_role"] == "release" for ds in datasets.values()
        ):
            raise ValueError(
                "At least one domain must have cdr_source_role='release'; got "
                f"{ {n: d.attrs['cdr_source_role'] for n, d in datasets.items()} }."
            )

        names = list(datasets)
        var_names = Counter(
            [
                *_COMBINED_VARS,
                *(
                    _domain_var(n, v)
                    for n in names
                    for v in (_FLUX_UPTAKE, _DIC_UPTAKE)
                ),
                *(_footprint_var(p, c) for c, p in pairwise(names)),
            ]
        )
        clashes = sorted(v for v, count in var_names.items() if count > 1)
        if clashes:
            raise ValueError(
                f"Domain names {names} give clashing variable names {clashes} in "
                "the combined dataset; rename the domains."
            )

        self._validate_geometry(datasets)

        window_lengths = {}
        for name, ds in datasets.items():
            values = ds["window_length"].values
            # A domain read back from NetCDF may have been decoded to timedelta64.
            if np.issubdtype(values.dtype, np.timedelta64):
                values = values / np.timedelta64(1, "s")
            if not np.allclose(values, values[0]):
                raise ValueError(
                    f"Domain {name!r} must have a single unique window_length, "
                    f"got {np.unique(values)}."
                )
            window_lengths[name] = float(values[0])
        first = next(iter(window_lengths.values()))
        if not np.allclose(list(window_lengths.values()), first):
            raise ValueError(
                f"Domains must share the same averaging window; got {window_lengths}."
            )

        for var in (_FLUX_UPTAKE, _DIC_UPTAKE):
            scales = {
                n: ds[var].attrs["tonnes_co2_per_native_unit"]
                for n, ds in datasets.items()
            }
            if not np.allclose(list(scales.values()), next(iter(scales.values()))):
                raise ValueError(
                    f"Domains must share the same native carbon units for {var!r} "
                    f"(tonnes_co2_per_native_unit); got {scales}."
                )

    @staticmethod
    def _centre_lon(ds: xr.Dataset) -> float:
        """Longitude of the rho point at the centre of a domain's grid."""
        lon = ds["lon_rho"].values
        return float(lon[lon.shape[0] // 2, lon.shape[1] // 2])

    @staticmethod
    def _bbox(ds: xr.Dataset, ref: float) -> tuple[float, float, float, float]:
        """
        Return (lon_min, lon_max, lat_min, lat_max) of a domain's rho points.

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
        Check that domains have distinct grids and are nested child -> parent.

        Longitudes are unwrapped to within 180 degrees of the (smaller) domain's
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
                        f"Domains {a!r} and {b!r} have identical lon_rho/lat_rho: "
                        "the same grid appears to have been used for both domains. "
                        "Compute each domain's metrics with its own grid."
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
                    f"Domain {parent!r} must strictly contain the lon/lat bounding "
                    f"box of {child!r}; domains must be ordered child -> parent. "
                    f"{parent!r}: lon [{p_lon0}, {p_lon1}], lat [{p_lat0}, {p_lat1}]; "
                    f"{child!r}: lon [{c_lon0}, {c_lon1}], lat [{c_lat0}, {c_lat1}]."
                )

    def _align_times(self, datasets: dict[str, xr.Dataset]) -> dict[str, xr.Dataset]:
        """
        Align the domains' time series on the union of their absolute times.

        Records that some domain lacks are NaN for that domain; a single warning
        reports how many records are affected and which domains are shorter.

        Returns
        -------
        dict[str, xr.Dataset]
            For each domain, its cumulative source and uptake series on the common
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
                "%d time record(s) are missing from at least one domain; combined "
                "series are NaN there. Domains with fewer records: %s.",
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
        Perimeter of a domain's area as a lon/lat polygon.

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
        lies within the child's area (see `_footprint`). The parent is cropped to
        the rows and columns holding those cells before the arithmetic, so only
        that slab of its (lazy) 3-D fields is read, not the whole parent.
        """
        ref = self._centre_lon(child)
        plon = _unwrap_lon(parent["lon_rho"].values, ref)
        plat = parent["lat_rho"].values
        inside_values = (
            self._footprint(child, ref)
            .contains_points(np.column_stack([plon.ravel(), plat.ravel()]))
            .reshape(plon.shape)
        )
        dims = parent["lon_rho"].dims
        window = {
            dim: slice(int(idx.min()), int(idx.max()) + 1) if idx.size else slice(0, 0)
            for dim, idx in zip(dims, np.nonzero(inside_values), strict=True)
        }
        inside = xr.DataArray(
            inside_values[window[dims[0]], window[dims[1]]], dims=dims
        )
        cropped = parent[["hDIC", "hDIC_ALT_CO2", "area"]].isel(window)
        diff = ((cropped["hDIC"] - cropped["hDIC_ALT_CO2"]) * cropped["area"]).sum(
            "s_rho"
        )
        scale = parent[_DIC_UPTAKE].attrs["tonnes_co2_per_native_unit"]
        footprint = (
            diff.where(inside).sum(["eta_rho", "xi_rho"]).compute() * scale
        ).reindex(time=all_times)
        return footprint.reset_coords(drop=True)

    def _build_dataset(
        self, datasets: dict[str, xr.Dataset], aligned: dict[str, xr.Dataset]
    ) -> xr.Dataset:
        """Assemble the combined dataset from the aligned domain series."""
        names = list(datasets)
        roles = {n: datasets[n].attrs["cdr_source_role"] for n in names}
        releases = [n for n in names if roles[n] == "release"]
        all_times = aligned[names[0]]["time"].values

        def total(var: str, domains: list[str]) -> xr.DataArray:
            return reduce(operator.add, [aligned[n][var] for n in domains])

        source_units = {
            datasets[n][_CUMULATIVE_SOURCE].attrs.get("units") for n in releases
        }
        if len(source_units) != 1:
            raise ValueError(
                f"Release domains must share the same {_CUMULATIVE_SOURCE} units, "
                f"got {sorted(map(str, source_units))}."
            )

        out = xr.Dataset(coords={"time": all_times})

        out[_CUMULATIVE_SOURCE] = total(_CUMULATIVE_SOURCE, releases)
        out[_CUMULATIVE_SOURCE].attrs.update(
            long_name="Cumulative CDR source (release domains)",
            description=(
                "Time-cumulative volume-integrated (ALK_source - DIC_source) of the "
                "release domains only; the denominator of the combined efficiency"
            ),
        )
        units = source_units.pop()
        if units is not None:
            out[_CUMULATIVE_SOURCE].attrs["units"] = units

        out[_FLUX_UPTAKE] = total(_FLUX_UPTAKE, names)
        out[_FLUX_UPTAKE].attrs.update(
            long_name="CDR carbon uptake (cumulative flux differences, all domains)",
            units="tonnes CO2",
            description=(
                "Time-cumulative CO2 uptake mass from area-integrated CO2 flux "
                "differences, summed over all domains"
            ),
        )
        out[_DIC_UPTAKE] = total(_DIC_UPTAKE, names)
        out[_DIC_UPTAKE].attrs.update(
            long_name="CDR carbon uptake (from DIC differences, all domains)",
            units="tonnes CO2",
            description=(
                "CO2-equivalent mass from the volume-integrated DIC difference, "
                "summed over all domains"
            ),
        )
        # Validation guarantees all domains share the scale, so the sum keeps it.
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
                "Total uptake over all domains (area-integrated CO2 flux "
                "differences) divided by the cumulative source of the release domains"
            ),
        )
        out[_DIC_EFFICIENCY].attrs.update(
            long_name="CDR uptake efficiency (from DIC differences)",
            units="nondimensional",
            description=(
                "Total uptake over all domains (volume-integrated DIC differences) "
                "divided by the cumulative source of the release domains"
            ),
        )

        for n in names:
            for var in (_FLUX_UPTAKE, _DIC_UPTAKE):
                out[_domain_var(n, var)] = aligned[n][var]
                out[_domain_var(n, var)].attrs = dict(datasets[n][var].attrs)

        out[_CLOSURE_RESIDUAL] = out[_FLUX_UPTAKE] - out[_DIC_UPTAKE]
        out[_CLOSURE_RESIDUAL].attrs.update(
            long_name="Closure residual (flux-based minus DIC-based uptake)",
            units="tonnes CO2",
            description="Combined flux-based minus DIC-difference-based uptake",
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
        out.attrs["domains"] = ", ".join(f"{n}:{roles[n]}" for n in names)
        return out

    def plot(
        self,
        save_path: str | None = None,
    ) -> None:
        """
        Plot the combined CDR efficiency and CO2 uptake in two subplots.

        The top panel shows the combined efficiency (flux-based and DIC-based); the
        bottom panel the combined uptake (both methods) with each domain's
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
        for name in self.domains:
            ax_uptake.plot(
                time,
                self.ds[_domain_var(name, _FLUX_UPTAKE)],
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

    def plot_domains(self, margin: float = 10.0, save_path: str | None = None) -> None:
        """
        Plot the outline of every domain on a map with coastlines.

        Shows how the domains are nested and where in the world the experiment
        takes place. Each outline is labelled ``"<name> (<role>)"``, where the role
        is the domain's ``cdr_source_role``, and drawn in the colour cycle's next
        colour in the order of `domains`. The map is centred on the outermost
        (last) domain.

        The release site itself is not marked, because ``ds_cdr`` carries no source
        fields; the release domain is identified by its label.

        Parameters
        ----------
        margin : float, optional
            Extent of the map beyond the outermost domain, in degrees of longitude
            and latitude. The default is chosen large enough for nearby coastlines
            to show; ``0`` fits the outermost domain. Default is 10.
        save_path : str, optional
            Path to save the generated plot. If None, the plot is shown interactively.
            Default is None.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        Raises
        ------
        ValueError
            If `margin` is negative.
        NotImplementedError
            If the outermost domain contains the North or South pole.
        """
        if margin < 0:
            raise ValueError(f"margin must be non-negative; got {margin}.")

        datasets = self._load_domains()
        outer_name = list(datasets)[-1]
        outer = datasets[outer_name]
        # All outlines share the outermost domain's longitude convention.
        ref = self._centre_lon(outer)
        lon_min, lon_max, lat_min, lat_max = self._bbox(outer, ref)

        lons = {
            name: ds["lon_rho"].copy(data=_unwrap_lon(ds["lon_rho"].values, ref))
            for name, ds in datasets.items()
        }
        trans = get_projection(lons[outer_name], outer["lat_rho"])

        fig, ax = plt.subplots(1, 1, figsize=(13, 7), subplot_kw={"projection": trans})
        for i, (name, ds) in enumerate(datasets.items()):
            _add_boundary_to_ax(
                ax,
                lons[name],
                ds["lat_rho"],
                trans,
                c=f"C{i}",
                label=f"{name} ({ds.attrs['cdr_source_role']})",
            )
        _add_gridlines(ax)
        ax.coastlines()
        ax.set_extent(
            [
                lon_min - margin,
                lon_max + margin,
                max(lat_min - margin, -90.0),
                min(lat_max + margin, 90.0),
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.set_title("Nested domains")
        ax.legend(loc="best")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
