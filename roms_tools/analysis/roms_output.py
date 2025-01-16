import xarray as xr
import numpy as np
from roms_tools.utils import _load_data
from dataclasses import dataclass, field
from roms_tools.setup.grid import Grid
from typing import Union
from pathlib import Path
import os
import re
import logging
from datetime import datetime, timedelta


@dataclass(frozen=True, kw_only=True)
class ROMSOutput:
    """Represents ROMS model output.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
    path : Union[str, Path]
        Directory or filename with model output.
    type : str
        Specifies the type of model output. Options are:

          - "restart": for restart files.
          - "average": for time-averaged files.
          - "snapshot": for snapshot files.

    model_reference_time : datetime, optional
        If not specified, this is inferred from metadata of the model output
        If specified and does not coincide with metadata, a warning is raised.
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.
    """

    grid: Grid
    path: Union[str, Path]
    type: Union[str, Path]
    use_dask: bool = False

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        # Validate `type`
        if self.type not in {"restart", "average", "snapshot"}:
            raise ValueError(
                f"Invalid type '{self.type}'. Must be one of 'restart', 'average', or 'snapshot'."
            )

        ds = self._get_model_output()
        self._infer_model_reference_date_from_metadata(ds)
        ds = self._add_absolute_time(ds)

        object.__setattr__(self, "ds", ds)

    def _get_model_output(self) -> xr.Dataset:
        """Load the model output based on the type."""
        # Determine if the path is a file or a directory
        if Path(self.path).is_file():
            single_file = True
            filename = self.path
            force_combine_nested = False
        elif Path(self.path).is_dir():
            single_file = False
            force_combine_nested = True
        else:
            raise FileNotFoundError(
                f"The specified path '{self.path}' is neither a file nor a directory."
            )

        time_chunking = True
        # Match the type and adjust filename pattern
        if self.type == "restart":
            time_chunking = False
            if single_file:
                if "rst" not in os.path.basename(filename):
                    logging.warning(
                        f"The file '{filename}' does not appear to be a restart file (missing '*rst*' in the name)."
                    )
            else:
                filename = os.path.join(self.path, "*rst.*.nc")
        elif self.type == "average":
            if single_file:
                if "avg" not in os.path.basename(filename):
                    logging.warning(
                        f"The file '{filename}' does not appear to be an average file (missing '*avg*' in the name)."
                    )
            else:
                filename = os.path.join(self.path, "avg.*.nc")
        elif self.type == "snapshot":
            if single_file:
                if "his" not in os.path.basename(filename):
                    logging.warning(
                        f"The file '{filename}' does not appear to be a snapshot file (missing '*his*' in the name)."
                    )
            else:
                filename = os.path.join(self.path, "his.*.nc")
        else:
            raise ValueError(f"Unsupported type '{self.type}'.")

        # Load the dataset
        ds = _load_data(
            filename,
            dim_names={"time": "time"},
            use_dask=self.use_dask,
            time_chunking=time_chunking,
            force_combine_nested=force_combine_nested,
        )

        return ds

    def _infer_model_reference_date_from_metadata(self, ds: xr.Dataset) -> None:
        """Infer and validate the model reference date from `ocean_time` metadata.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset with an `ocean_time` variable and a `long_name` attribute
            in the format `Time since YYYY/MM/DD`.

        Raises
        ------
        ValueError
            If `self.model_reference_date` is not set and the reference date cannot
            be inferred, or if the inferred date does not match `self.model_reference_date`.

        Warns
        -----
        UserWarning
            If `self.model_reference_date` is set but the reference date cannot be inferred.
        """
        input_string = ds.ocean_time.attrs["long_name"]
        match = re.search(r"(\d{4})/(\d{2})/(\d{2})", input_string)

        if match:
            year, month, day = map(int, match.groups())
            inferred_date = datetime(year, month, day)

            if hasattr(self, "model_reference_date") and self.model_reference_date:
                if self.model_reference_date != inferred_date:
                    raise ValueError(
                        f"Mismatch between `self.model_reference_date` ({self.model_reference_date}) "
                        f"and inferred reference date ({inferred_date})."
                    )
            else:
                object.__setattr__(self, "model_reference_date", inferred_date)
        else:
            if hasattr(self, "model_reference_date") and self.model_reference_date:
                logging.warning(
                    "Could not infer the model reference date from the metadata. "
                    "`self.model_reference_date` will be used.",
                )
            else:
                raise ValueError(
                    "Model reference date could not be inferred from the metadata, "
                    "and `self.model_reference_date` is not set."
                )

    def _add_absolute_time(self, ds):

        ocean_time_seconds = ds["ocean_time"].values

        abs_time = np.array(
            [
                self.model_reference_date + timedelta(seconds=seconds)
                for seconds in ocean_time_seconds
            ]
        )

        abs_time = xr.DataArray(
            abs_time, dims=["time"], coords={"time": ds["ocean_time"]}
        )
        abs_time.attrs["long_name"] = "absolute time"
        ds = ds.assign_coords({"abs_time": abs_time})

        return ds
