from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


@dataclass
class Ensemble:
    members: dict[str, str | xr.Dataset]  # name → path or dataset
    ds: xr.Dataset = field(init=False)

    def __post_init__(self):
        datasets = self._load_members()
        effs = {name: self._extract_efficiency(ds) for name, ds in datasets.items()}
        aligned = self._align_times(effs)
        self.ds = aligned
        # self.ds = self._compute_statistics(aligned)

    def _load_members(self) -> dict[str, xr.Dataset]:
        """Open NetCDF paths into xarray datasets if needed."""
        return {
            name: xr.open_dataset(path) if isinstance(path, str | Path) else path
            for name, path in self.members.items()
        }

    def _extract_efficiency(self, ds: xr.Dataset) -> xr.DataArray:
        """Extract cdr_efficiency and reindex to time since release start."""
        eff = ds["cdr_efficiency"]

        # Find first non-NaN time (release start)
        valid = eff.dropna(dim="time")
        if valid.time.size == 0:
            raise ValueError("No valid efficiency values found in dataset.")
        release_start = valid.abs_time.min()

        # Reindex time relative to release_start in days
        time = (eff.abs_time - release_start).astype("timedelta64[D]")
        eff_rel = eff.assign_coords(time=time)
        eff_rel.time.attrs.update({"long_name": "time since release start"})
        eff_rel = eff_rel.drop_vars("abs_time")

        return eff_rel

    def _align_times(self, effs: dict[str, xr.DataArray]) -> xr.Dataset:
        """Reindex all efficiencies to a common time axis."""
        all_times = np.unique(
            np.concatenate([eff.time.values for eff in effs.values()])
        )
        aligned = {
            name: eff.reindex(time=all_times, method="nearest")
            for name, eff in effs.items()
        }
        return xr.Dataset(aligned)

    def _compute_statistics(self, ds: xr.Dataset) -> xr.Dataset:
        """Compute mean and std across ensemble members."""
        arr = ds.to_array("member")  # stack into (member, time)
        ds["mean"] = arr.mean("member")
        ds["std"] = arr.std("member")
        return ds

    def plot(self):
        """Plot ensemble members with mean ± std shading."""
        all_eff = self.gather()
        mean, std = self.statistics()

        plt.figure(figsize=(8, 5))

        # Individual ensemble members
        for name in all_eff.ensemble.values:
            plt.plot(all_eff.time, all_eff.sel(ensemble=name), alpha=0.4, label=name)

        # Mean ± std
        plt.plot(mean.time, mean, color="black", lw=2, label="mean")
        plt.fill_between(mean.time, mean - std, mean + std, color="gray", alpha=0.3)

        plt.xlabel("Time since release start [days]")
        plt.ylabel("CDR Efficiency")
        plt.title("Ensemble of interventions")
        plt.legend()
        plt.tight_layout()
        plt.show()
