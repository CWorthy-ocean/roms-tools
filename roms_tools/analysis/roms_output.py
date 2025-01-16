import xarray as xr
import numpy as np
from roms_tools.utils import _load_data
from dataclasses import dataclass, field
from roms_tools.setup.grid import Grid
from typing import Dict, Optional, Union, List
from pathlib import Path
import os

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

          - "restarts": for restart files.
          - "averages": for time-averaged files.
          - "snapshots": for snapshot files.

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
        if self.type not in {"restarts", "averages", "snapshots"}:
            raise ValueError(
                f"Invalid type '{self.type}'. Must be one of 'restarts', 'averages', or 'snapshots'."
            )

        ds = self._get_model_output()
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
            raise FileNotFoundError(f"The specified path '{self.path}' is neither a file nor a directory.")

        time_chunking = True
        # Match the type and adjust filename pattern
        if self.type == "restarts":
            if single_file:
                if "rst" not in os.path.basename(filename):
                    logger.warning(
                        f"The file '{filename}' does not appear to be a restart file (missing '*rst*' in the name)."
                    )
            else:
                filename = os.path.join(self.path, "*rst.*.nc")
                time_chunking = False
        elif self.type == "averages":
            if single_file:
                if "avg" not in os.path.basename(filename):
                    logger.warning(
                        f"The file '{filename}' does not appear to be an averages file (missing '*avg*' in the name)."
                    )
            else:
                filename = os.path.join(self.path, "avg.*.nc")
        elif self.type == "snapshots":
            if single_file:
                if "his" not in os.path.basename(filename):
                    logger.warning(
                        f"The file '{filename}' does not appear to be a snapshots file (missing '*his*' in the name)."
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
