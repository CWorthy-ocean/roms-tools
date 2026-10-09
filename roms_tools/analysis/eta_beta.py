import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import PyCO2SYS as pyco2
import xarray as xr

from roms_tools import Grid
from roms_tools.setup.utils import group_by_month
from roms_tools.utils import load_data, save_datasets


@dataclass(kw_only=True)
class calculate_eta_beta:
    """Computes the eta and beta carbonate sensitivities from ROMS-MARBL control-run output.

    On initialization, the class:

    1. Loads the surface fields from the history and BGC files
       (`_get_surface_vals_control`), stored in ``ds_surface_control``.
    2. Computes eta and beta from those fields with PyCO2SYS
       (`_compute_eta_beta`), stored in ``ds``.

    where

    - ``beta = dDIC / dCO2``
    - ``eta  = dDIC / dALK``

    Call `save` to write ``ds`` to monthly files.

    Parameters
    ----------
    his_path : str | Path | list[str | Path]
        Filename, wildcard pattern, or list of filenames with ROMS history output
    bgc_path : str | Path | list[str | Path]
        Filename, wildcard pattern, or list of filenames with MARBL BGC output
    grid : Grid
        Object representing the grid information. Land points are masked with
        ``mask_rho`` before eta and beta are computed.
    model_reference_date : datetime, optional
        Reference date of ROMS simulation.
    dim_names : dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
    use_dask : bool, optional
        Indicates whether to use dask for processing. Defaults to False.
    """

    his_path: str | Path | list[str | Path]
    """Filename, wildcard pattern, or list of filenames with ROMS history output."""
    bgc_path: str | Path | list[str | Path]
    """Filename, wildcard pattern, or list of filenames with MARBL BGC output."""
    grid: Grid
    """Object representing the grid information, used for land masking."""
    model_reference_date: datetime | None = None
    """Reference date of ROMS simulation."""
    dim_names: dict[str, str] = field(
        default_factory=lambda: {
            "eta_rho": "eta_rho",
            "xi_rho": "xi_rho",
            "s_rho": "s_rho",
            "time": "time",
        }
    )
    """Dictionary specifying the names of dimensions in the dataset."""
    use_dask: bool = False
    """Whether to use dask for processing."""

    ds_surface_control: xr.Dataset = field(init=False, repr=False)
    """An xarray Dataset containing the control-run surface fields."""
    ds: xr.Dataset = field(init=False, repr=False)
    """An xarray Dataset containing the eta and beta fields."""

    def __post_init__(self):
        self.ds_surface_control = self._get_surface_vals_control()
        self._infer_model_reference_date_from_metadata(self.ds_surface_control)
        ds = self._compute_eta_beta(self.ds_surface_control)
        self.ds = self._add_absolute_time(ds)

    def save(self, filepath: str | Path) -> list[Path]:
        """Save the eta and beta fields to monthly netCDF4 files.

        The dataset is split by calendar month of ``abs_time`` and each month is
        written to its own file, named ``<filepath>_YYYYMM.nc``.

        Parameters
        ----------
        filepath : str | Path
            The base path and filename for the output files, e.g.
            ``".../BETA/carbonate_sensitivity"``. A ``.nc`` suffix, if given,
            is removed before the ``_YYYYMM`` label is added.

        Returns
        -------
        list[Path]
            A list of ``Path`` objects for the files that were saved.
        """
        filepath = Path(filepath)
        if filepath.suffix == ".nc":
            filepath = filepath.with_suffix("")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        dataset_list, output_filenames = group_by_month(self.ds, str(filepath))

        return save_datasets(dataset_list, output_filenames, use_dask=self.use_dask)

    def _get_surface_vals_control(self) -> xr.Dataset:
        """Load control-run surface fields from the history and BGC files.

        Load his and bgc variables, only retain the variables needed for eta beta
        calculations, and reduce to the surface layer. ``ocean_time`` values are
        checked to be matching between his and bgc variables, and all variables are
        merged into a single dataset.

        Returns
        -------
        xr.Dataset
            Dataset containing the needed variables to perform eta beta calculations,
            at the surface layer.

        Raises
        ------
        ValueError
            If the history and BGC times do not match, or if any needed variable
            is not found in either file group.
        """
        ds_his = self._load_surface(self.his_path, source="his")
        ds_bgc = self._load_surface(self.bgc_path, source="bgc")

        self._check_time_consistency(ds_his, ds_bgc)
        ds = self._merge(ds_his, ds_bgc)

        return ds

    def _compute_eta_beta(self, ds_surface: xr.Dataset) -> xr.Dataset:
        """Compute the eta and beta carbonate sensitivities with PyCO2SYS.

        Land points are masked with the grid's ``mask_rho``, the inputs are
        loaded into memory, and `_compute_sensitivities` is called with the
        control run to calculate the sensitivies.
        ``beta = dDIC / dCO2`` and ``eta = dDIC / dALK``, computed from the
        surface ALK, DIC, salinity, temperature, silicate, and phosphate.

        Parameters
        ----------
        ds_surface : xr.Dataset
            Surface dataset containing ALK, DIC, ``salt``, ``temp``, ``PO4``,
            and ``SiO3``.

        Returns
        -------
        xr.Dataset
            Dataset with ``eta`` and ``beta`` on the same dimensions as the inputs,
            with ``ocean_time`` as a coordinate.

        Raises
        ------
        KeyError
            If any required variable is missing from ``ds_surface``.
        """
        required_vars = (
            "ALK_ALT_CO2",
            "DIC_ALT_CO2",
            "salt",
            "temp",
            "PO4",
            "SiO3",
        )
        missing = [v for v in required_vars if v not in ds_surface]
        if missing:
            raise KeyError(f"`ds_surface` is missing required variables {missing}")

        time = ds_surface["ocean_time"] if "ocean_time" in ds_surface else None
        ds_surface = ds_surface[list(required_vars)]

        ds_surface = ds_surface.where(self.grid.ds.mask_rho)

        # PyCO2SYS works on in-memory numpy arrays
        ds_surface = ds_surface.load()

        # Skip near-empty cells, using MARBL's floors
        salt_min = 0.1
        ds_surface = ds_surface.where(
            (ds_surface.salt >= salt_min)
            & (ds_surface.ALK_ALT_CO2 >= salt_min / 35.0 * 2225.0)
            & (ds_surface.DIC_ALT_CO2 >= salt_min / 35.0 * 1944.0)
        )

        rho_factor = 1000.0 / 1025.0  # mmol/m3 → µmol/kg

        ALK = ds_surface["ALK_ALT_CO2"] * rho_factor
        DIC = ds_surface["DIC_ALT_CO2"] * rho_factor

        csys = pyco2.sys(
            par1=ALK.values,
            par2=DIC.values,
            par1_type=1,
            par2_type=2,
            salinity=ds_surface.salt.values,
            temperature=ds_surface.temp.values,
            total_silicate=ds_surface.SiO3.clip(min=0).values * rho_factor,
            total_phosphate=ds_surface.PO4.clip(min=0).values * rho_factor,
            opt_buffers_mode=2,
        )

        iso_q = csys["isocapnic_quotient"]

        beta = (csys["dic"] - (csys["HCO3"] + 2.0 * csys["CO3"]) / iso_q) / csys["CO2"]
        eta = 1.0 / iso_q

        # Assign dimensions
        ds = xr.Dataset(
            {
                "eta": (ALK.dims, np.asarray(eta)),
                "beta": (ALK.dims, np.asarray(beta)),
            },
            coords=ALK.coords,
        )
        ds["eta"].attrs = {"long_name": "dDIC / dALK", "units": "1"}
        ds["beta"].attrs = {"long_name": "dDIC / dCO2", "units": "1"}

        if time is not None:
            ds = ds.assign_coords({"ocean_time": time.load()})

        return ds

    def _infer_model_reference_date_from_metadata(self, ds: xr.Dataset) -> None:
        """Infer and validate the model reference date from ``ocean_time`` metadata.

        Similar to `roms_tools.datasets.roms_dataset.ROMSDataset`: the
        reference date is read from the ``long_name`` attribute of
        ``ocean_time``.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset with an `ocean_time` variable and a `long_name` attribute
            in the format `Time since YYYY/MM/DD`.
        Raises
        ------
        ValueError
            If the reference date cannot be inferred and ``model_reference_date``
            is not set, or if the inferred date does not match
            ``model_reference_date``.
        """
        if "long_name" not in ds["ocean_time"].attrs:
            if self.model_reference_date is None:
                raise ValueError(
                    "`long_name` attribute not found in `ocean_time`, so the model "
                    "reference date could not be inferred. Pass "
                    "`model_reference_date` explicitly."
                )
            logging.warning(
                "`long_name` attribute not found in `ocean_time`; using "
                "`model_reference_date`."
            )
            return

        long_name = ds["ocean_time"].attrs.get("long_name", "")
        match = re.search(r"(\d{4})/(\d{2})/(\d{2})", long_name)

        if match:
            year, month, day = (int(g) for g in match.groups())
            inferred_date = datetime(year, month, day)
            if self.model_reference_date is None:
                self.model_reference_date = inferred_date
            elif self.model_reference_date != inferred_date:
                raise ValueError(
                    f"Mismatch between `model_reference_date` "
                    f"({self.model_reference_date}) and the reference date in the "
                    f"output metadata ({inferred_date})."
                )
        elif self.model_reference_date is None:
            raise ValueError(
                f"Model reference date could not be inferred from the `ocean_time` "
                f"`long_name` ({long_name!r}). Pass `model_reference_date` explicitly."
            )
        else:
            logging.warning(
                f"Could not infer the model reference date from the `ocean_time` "
                f"`long_name` ({long_name!r}); using `model_reference_date`."
            )

    def _add_absolute_time(self, ds: xr.Dataset) -> xr.Dataset:
        """Add absolute time as a coordinate, used to group the output by month.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing ``ocean_time`` in seconds since the model reference date.

        Returns
        -------
        xr.Dataset
            Dataset with an ``abs_time`` coordinate along the time dimension.
        """
        model_reference_date = self.model_reference_date
        if model_reference_date is None:
            raise ValueError(
                "`model_reference_date` must be set before adding absolute time."
            )

        abs_time = np.array(
            [
                model_reference_date + timedelta(seconds=float(seconds))  # CHANGED
                for seconds in ds["ocean_time"].values
            ],
            dtype="datetime64[ns]",
        )
        abs_time = xr.DataArray(
            abs_time,
            dims=[self.dim_names["time"]],
            attrs={"long_name": "absolute time"},
        )

        return ds.assign_coords(abs_time=abs_time)

    def _load_surface(
        self, path: str | Path | list[str | Path], source: str
    ) -> xr.Dataset:
        """Load one group of ROMS files and reduce it to the surface layer.

        Parameters
        ----------
        path : str | Path | list[str | Path]
            Filename, wildcard pattern, or list of filenames.
        source : str
            Label for the file group (``"his"`` or ``"bgc"``), used in log messages.

        Returns
        -------
        xr.Dataset
            Dataset reduced to the variables in ds_vars and only at the surface layer.
        """
        ds = load_data(
            filename=path,
            dim_names=self.dim_names,
            use_dask=self.use_dask,
            decode_times=False,
            decode_timedelta=False,
            force_combine_nested=True,
        )

        # Define required variables
        if source == "bgc":
            ds_vars = [
                "ALK_ALT_CO2",
                "DIC_ALT_CO2",
                "PO4",
                "SiO3",
                "ocean_time",
            ]
        elif source == "his":
            ds_vars = [
                "temp",
                "salt",
                "ocean_time",
            ]

        missing_ds = [var for var in ds_vars if var not in ds]
        if missing_ds:
            raise KeyError(
                f"Missing required variables in {source} files: {missing_ds}"
            )

        keep_vars = [v for v in ds_vars if v in ds.variables]
        logging.info(f"Variables taken from {source} files: {keep_vars}")
        ds = ds[keep_vars]

        s_rho = self.dim_names["s_rho"]
        if s_rho in ds.dims:
            # In ROMS the last s_rho index is the layer closest to the surface.
            ds = ds.isel({s_rho: -1})

        return ds

    def _check_time_consistency(self, ds_his: xr.Dataset, ds_bgc: xr.Dataset) -> None:
        """Ensure the history and BGC files cover the same output times.

        Parameters
        ----------
        ds_his : xr.Dataset
            Surface dataset from the history files.
        ds_bgc : xr.Dataset
            Surface dataset from the BGC files.

        Raises
        ------
        ValueError
            If ``ocean_time`` is missing from either group, or if the two groups
            have a different number of time records or different time values.
        """
        time_var = "ocean_time"

        for source, ds in (("his", ds_his), ("bgc", ds_bgc)):
            if time_var not in ds:
                raise ValueError(
                    f"`{time_var}` not found in the {source} files; it is needed "
                    f"to align history and BGC output."
                )

        t_his = ds_his[time_var].values
        t_bgc = ds_bgc[time_var].values

        if t_his.shape != t_bgc.shape:
            raise ValueError(
                f"History and BGC files have a different number of time records: "
                f"his={t_his.size}, bgc={t_bgc.size}. Check that both wildcards "
                f"match the same set of output periods."
            )
        if not (t_his == t_bgc).all():
            raise ValueError(
                f"History and BGC `{time_var}` values do not match. Check that both "
                f"groups were written at the same output frequency."
            )

    def _merge(self, ds_his: xr.Dataset, ds_bgc: xr.Dataset) -> xr.Dataset:
        """Merge the history and BGC surface datasets into one.

        Variables found in both groups (including ``ocean_time``) are taken from
        the history files.

        Parameters
        ----------
        ds_his : xr.Dataset
            Surface dataset from the history files.
        ds_bgc : xr.Dataset
            Surface dataset from the BGC files.

        Returns
        -------
        xr.Dataset
            Merged dataset.
        """
        shared = [v for v in ds_bgc.data_vars if v in ds_his.data_vars]
        extra_shared = [v for v in shared if v != "ocean_time"]
        if extra_shared:
            logging.warning(
                f"Variables {extra_shared} found in both history and BGC files; "
                f"using the history values."
            )
        ds_bgc = ds_bgc.drop_vars(shared)

        return xr.merge([ds_his, ds_bgc], join="exact", combine_attrs="override")
