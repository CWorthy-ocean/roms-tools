import xarray as xr
import numpy as np
from roms_tools.utils import _load_data
from dataclasses import dataclass, field
from typing import Union
from pathlib import Path
import os
import re
import logging
from datetime import datetime, timedelta
from roms_tools import Grid
from roms_tools.vertical_coordinate import retrieve_depth_coordinates


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
        self._check_vertical_coordinate(ds)
        ds = self._add_absolute_time(ds)
        ds = self._add_lat_lon_coords(ds)
        object.__setattr__(self, "ds", ds)

    def get_vertical_coordinates(self, type="layer", additional_locations=[]):
        """Retrieve layer and interface depth coordinates.

        This method computes and updates the layer and interface depth coordinates. It handles depth calculations for rho points and
        additional specified locations (u and v).

        Parameters
        ----------
        type : str, optional
            The type of depth coordinate to retrieve. Default is "layer". Valid options are:
            - "layer": Retrieves layer depth coordinates.
            - "interface": Retrieves interface depth coordinates.

        additional_locations : list of str, optional
            Specifies additional locations to compute depth coordinates for. Default is ["u", "v"].
            Valid options include:
            - "u": Computes depth coordinates for u points.
            - "v": Computes depth coordinates for v points.

        Updates
        -------
        self.ds : xarray.Dataset
            The dataset is updated with the following vertical depth coordinates:
            - f"{type}_depth_rho": Depth coordinates at rho points.
            - f"{type}_depth_u": Depth coordinates at u points (if applicable).
            - f"{type}_depth_v": Depth coordinates at v points (if applicable).
        """

        retrieve_depth_coordinates(self.ds, self.grid.ds, type, additional_locations)

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

    def _check_vertical_coordinate(self, ds: xr.Dataset) -> None:
        """Check that the vertical coordinate parameters in the dataset are consistent
        with the model grid.

        This method compares the vertical coordinate parameters (`theta_s`, `theta_b`, `hc`, `Cs_r`, `Cs_w`) in
        the provided dataset (`ds`) with those in the model grid (`self.grid`). The first three parameters are
        checked for exact equality, while the last two are checked for numerical closeness.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset containing vertical coordinate parameters in its attributes, such as `theta_s`, `theta_b`,
            `hc`, `Cs_r`, and `Cs_w`.

        Raises
        ------
        ValueError
            If the vertical coordinate parameters do not match the expected values (based on exact or approximate equality).

        Notes
        -----
        - `theta_s`, `theta_b`, and `hc` are checked for exact equality using `np.array_equal`.
        - `Cs_r` and `Cs_w` are checked for numerical closeness using `np.allclose`.
        """

        # Check exact equality for theta_s, theta_b, and hc
        if not np.array_equal(self.grid.theta_s, ds.attrs["theta_s"]):
            raise ValueError(
                f"theta_s from grid ({self.grid.theta_s}) does not match dataset ({ds.attrs['theta_s']})."
            )

        if not np.array_equal(self.grid.theta_b, ds.attrs["theta_b"]):
            raise ValueError(
                f"theta_b from grid ({self.grid.theta_b}) does not match dataset ({ds.attrs['theta_b']})."
            )

        if not np.array_equal(self.grid.hc, ds.attrs["hc"]):
            raise ValueError(
                f"hc from grid ({self.grid.hc}) does not match dataset ({ds.attrs['hc']})."
            )

        # Check numerical closeness for Cs_r and Cs_w
        if not np.allclose(self.grid.ds.Cs_r, ds.attrs["Cs_r"]):
            raise ValueError(
                f"Cs_r from grid ({self.grid.Cs_r}) is not close to dataset ({ds.attrs['Cs_r']})."
            )

        if not np.allclose(self.grid.ds.Cs_w, ds.attrs["Cs_w"]):
            raise ValueError(
                f"Cs_w from grid ({self.grid.Cs_w}) is not close to dataset ({ds.attrs['Cs_w']})."
            )

    def _add_absolute_time(self, ds: xr.Dataset) -> xr.Dataset:
        """Add absolute time as a coordinate to the dataset.

        Computes "abs_time" based on "ocean_time" and a reference date,
        and adds it as a coordinate.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing "ocean_time" in seconds since the model reference date.

        Returns
        -------
        xarray.Dataset
            Dataset with "abs_time" added and "time" removed.
        """
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
        ds = ds.drop_vars("time")

        return ds

    def _add_lat_lon_coords(self, ds: xr.Dataset) -> xr.Dataset:
        """Add latitude and longitude coordinates to the dataset.

        Adds "lat_rho" and "lon_rho" from the grid object to the dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to update.

        Returns
        -------
        xarray.Dataset
            Dataset with "lat_rho" and "lon_rho" coordinates added.
        """
        ds = ds.assign_coords(
            {"lat_rho": self.grid.ds["lat_rho"], "lon_rho": self.grid.ds["lon_rho"]}
        )

        return ds
