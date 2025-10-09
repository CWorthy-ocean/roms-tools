from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


@dataclass
class Ensemble:
    """
    Represents an ensemble of CDR (Carbon Dioxide Removal) experiments.

    Loads, aligns, and analyzes efficiency metrics across multiple members.

    Parameters
    ----------
    members : dict[str, str | xr.Dataset]
        Dictionary mapping member names to either file paths (NetCDF) or
        xarray.Dataset objects containing the CDR metrics.
    """

    members: dict[str, str | xr.Dataset]
    """Dictionary mapping member names to CDR metrics."""
    ds: xr.Dataset = field(init=False)
    """xarray Dataset containing aligned efficiencies for all ensemble members."""

    def __post_init__(self):
        """
        Loads datasets, extracts efficiencies, aligns times, stores in self.ds, and
        plots ensemble curves.
        """
        datasets = self._load_members()
        effs = {name: self._extract_efficiency(ds) for name, ds in datasets.items()}
        aligned = self._align_times(effs)
        self.ds = self._compute_statistics(aligned)

    def _load_members(self) -> dict[str, xr.Dataset]:
        """
        Loads ensemble member datasets.

        Converts any file paths in `self.members` to xarray Datasets.
        Members that are already xarray Datasets are left unchanged.

        Returns
        -------
        dict[str, xr.Dataset]
            Dictionary mapping member names to xarray.Dataset objects.
        """
        return {
            name: xr.open_dataset(path) if isinstance(path, str | Path) else path
            for name, path in self.members.items()
        }

    def _extract_efficiency(self, ds: xr.Dataset) -> xr.DataArray:
        """
        Extracts the CDR efficiency metric and reindex to time since release start.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing a "cdr_efficiency" variable and "abs_time" coordinate.

        Returns
        -------
        xr.DataArray
            Efficiency reindexed to a relative time axis in days since release start.
        """
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
        """
        Align all ensemble members to a common time axis.

        Each member is reindexed to the union of all time coordinates.
        Times outside the original range of a mamber are filled with NaN.

        Parameters
        ----------
        effs : dict[str, xr.DataArray]
            Dictionary mapping member names to efficiency DataArrays
            (reindexed relative to their release start).

        Returns
        -------
        xr.Dataset
            Dataset containing all members aligned to a shared time coordinate, with
            NaNs for missing times.
        """
        all_times = np.unique(
            np.concatenate([eff.time.values for eff in effs.values()])
        )
        aligned = {name: eff.reindex(time=all_times) for name, eff in effs.items()}
        return xr.Dataset(aligned)

    def _compute_statistics(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Computes ensemble statistics: mean and standard deviation.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing aligned ensemble member efficiencies.

        Returns
        -------
        xr.Dataset
            Dataset with additional variables "mean" and "std" representing
            the ensemble mean and standard deviation across members.
        """
        da = ds.to_dataarray("member")  # stack into (member, time)
        ds["ensemble_mean"] = da.mean(dim="member")
        ds["ensemble_std"] = da.std(dim="member")
        return ds

    def plot(
        self,
        save_path: str | None = None,
    ) -> None:
        """
        Plots ensemble members with mean ± standard deviation shading.

        Displays individual member efficiency time series along with the ensemble
        mean and ±1 standard deviation as a shaded region.

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
        fig, ax = plt.subplots(figsize=(8, 5))

        time = self.ds.time.values / np.timedelta64(1, "D")  # converts to float days

        # Individual ensemble members
        for name in self.members.keys():
            ax.plot(time, self.ds[name], lw=2, label=name)

        # Mean ± std
        ax.plot(time, self.ds.ensemble_mean, color="black", lw=2, label="ensemble mean")
        ax.fill_between(
            time,
            self.ds.ensemble_mean - self.ds.ensemble_std,
            self.ds.ensemble_mean + self.ds.ensemble_std,
            color="gray",
            alpha=0.3,
        )

        ax.set_xlabel("Time since release start [days]")
        ax.set_ylabel("CDR Efficiency")
        ax.set_title("Ensemble of interventions")
        ax.legend()
        ax.grid()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
