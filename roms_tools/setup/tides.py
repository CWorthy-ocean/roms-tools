import importlib.metadata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from roms_tools import Grid
from roms_tools.plot import _plot
from roms_tools.regrid import LateralRegridToROMS
from roms_tools.setup.datasets import TPXOManager
from roms_tools.setup.utils import (
    _from_yaml,
    _to_dict,
    _write_to_yaml,
    get_target_coords,
    get_variable_metadata,
    get_vector_pairs,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    nan_check,
    rotate_velocities,
    substitute_nans_by_fillvalue,
)
from roms_tools.utils import save_datasets


@dataclass(kw_only=True)
class TidalForcing:
    """Represents tidal forcing for ROMS.

    Parameters
    ----------
    grid : Grid
        The grid object representing the ROMS grid associated with the tidal forcing data.
    source : Dict[str, Union[str, Path, Dict[str, Union[str, Path]]]]
        Dictionary specifying the source of the tidal data. Keys include:

          - "name" (str): Name of the data source (e.g., "TPXO").
          - "path" (Union[str, Path, Dict[str, Union[str, Path]]]):

            - If a string or Path is provided, it represents a single file.
            - If "name" is "TPXO", "path" can also be a dictionary with the following keys:

              - "grid" (Union[str, Path]): Path to the TPXO grid file.
              - "h" (Union[str, Path]): Path to the TPXO h-file.
              - "u" (Union[str, Path]): Path to the TPXO u-file.

    ntides : int, optional
        Number of constituents to consider. Maximum number is 15. Default is 10.
    model_reference_date : datetime, optional
        The reference date for the ROMS simulation. Default is datetime(2000, 1, 1).
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.
    bypass_validation: bool, optional
        Indicates whether to skip validation checks in the processed data. When set to True,
        the validation process that ensures no NaN values exist at wet points
        in the processed dataset is bypassed. Defaults to False.

    Examples
    --------
    Using a TPXO dataset with separate grid, h, and u files:

    >>> tidal_forcing = TidalForcing(
    ...     grid=grid,
    ...     source={
    ...         "name": "TPXO",
    ...         "path": {"grid": "tpxo_grid.nc", "h": "tpxo_h.nc", "u": "tpxo_u.nc"},
    ...     },
    ... )

    Using a single file as a source:

    >>> tidal_forcing = TidalForcing(
    ...     grid=grid,
    ...     source={"name": "TPXO", "path": "tpxo_merged.nc"},
    ... )
    """

    grid: Grid
    source: Dict[str, Union[str, Path, List[Union[str, Path]]]]
    ntides: int = 10
    model_reference_date: datetime = datetime(2000, 1, 1)
    use_dask: bool = False
    bypass_validation: bool = False

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        self._input_checks()
        target_coords = get_target_coords(self.grid)

        tidal_data = self._get_data()

        for key, data in tidal_data.datasets.items():
            if key != "omega":
                data.choose_subdomain(
                    target_coords,
                    buffer_points=20,
                )
                # Enforce double precision to ensure reproducibility
                data.convert_to_float64()

        tidal_data.correct_tides(self.model_reference_date)

        self._set_variable_info()
        var_names = self.variable_info.keys()

        processed_fields = {}

        # lateral fill and regridding
        for key, data in tidal_data.datasets.items():
            if key != "omega":
                data.apply_lateral_fill()
                lateral_regrid = LateralRegridToROMS(target_coords, data.dim_names)

                for var_name in var_names:
                    if var_name in data.var_names.keys():
                        processed_fields[var_name] = lateral_regrid.apply(
                            data.ds[data.var_names[var_name]]
                        )

        # rotation of velocities and interpolation to u/v points
        vector_pairs = get_vector_pairs(self.variable_info)
        for pair in vector_pairs:
            u_component = pair[0]
            v_component = pair[1]
            if u_component in processed_fields and v_component in processed_fields:
                (
                    processed_fields[u_component],
                    processed_fields[v_component],
                ) = rotate_velocities(
                    processed_fields[u_component],
                    processed_fields[v_component],
                    target_coords["angle"],
                    interpolate=False,
                )

        # convert to barotropic velocity
        for var_name in ["u_Re", "v_Re", "u_Im", "v_Im"]:
            processed_fields[var_name] = processed_fields[var_name] / self.grid.ds.h

        # interpolate from rho- to velocity points
        for uname in ["u_Re", "u_Im"]:
            processed_fields[uname] = interpolate_from_rho_to_u(processed_fields[uname])
        for vname in ["v_Re", "v_Im"]:
            processed_fields[vname] = interpolate_from_rho_to_v(processed_fields[vname])

        d_meta = get_variable_metadata()
        ds = self._write_into_dataset(processed_fields, d_meta)

        ds = self._add_global_metadata(ds)

        if not self.bypass_validation:
            self._validate(ds)

        ds = ds.assign_coords({"omega": tidal_data.datasets["omega"]})
        ds["ntides"].attrs["long_name"] = "constituent label"

        # substitute NaNs over land by a fill value to avoid blow-up of ROMS
        for var_name in ds.data_vars:
            ds[var_name] = substitute_nans_by_fillvalue(ds[var_name])

        self.ds = ds

    def _input_checks(self):

        if "name" not in self.source.keys():
            raise ValueError("`source` must include a 'name'.")
        if "path" not in self.source.keys():
            raise ValueError("`source` must include a 'path'.")
        if self.ntides > 15:
            raise ValueError("`ntides` must be at most 15.")

    def _get_data(self):
        """Loads tidal forcing data based on the specified source."""

        if self.source["name"] == "TPXO":
            if isinstance(self.source["path"], dict):
                fname_dict = {
                    "grid": self.source["path"]["grid"],
                    "h": self.source["path"]["h"],
                    "u": self.source["path"]["u"],
                }

            elif isinstance(self.source["path"], (str, Path)):
                fname_dict = {
                    "grid": self.source["path"],
                    "h": self.source["path"],
                    "u": self.source["path"],
                }
            else:
                raise ValueError(
                    'For TPXO, source["path"] must be either a string, Path, or a dictionary with "grid", "h", and "u" keys.'
                )

            data = TPXOManager(
                filenames=fname_dict,
                ntides=self.ntides,
                use_dask=self.use_dask,
            )

        else:
            raise ValueError('Only "TPXO" is a valid option for source["name"].')

        return data

    def _set_variable_info(self):
        """Sets up a dictionary with metadata for variables based on the type.

        The dictionary contains the following information:
        - `location`: Where the variable resides in the grid (e.g., rho, u, or v points).
        - `is_vector`: Whether the variable is part of a vector (True for velocity components like 'u' and 'v').
        - `vector_pair`: For vector variables, this indicates the associated variable that forms the vector (e.g., 'u' and 'v').

        Returns
        -------
        None
            This method updates the instance attribute `variable_info` with the metadata dictionary for the variables.
        """
        default_info = {
            "location": "rho",
            "is_vector": False,
            "vector_pair": None,
        }

        # Define a dictionary for variable names and their associated information
        variable_info = {
            "ssh_Re": {**default_info, "validate": True},
            "ssh_Im": {**default_info, "validate": False},
            "pot_Re": {**default_info, "validate": False},
            "pot_Im": {**default_info, "validate": False},
            "u_Re": {
                "location": "u",
                "is_vector": True,
                "vector_pair": "v_Re",
                "validate": True,
            },
            "v_Re": {
                "location": "v",
                "is_vector": True,
                "vector_pair": "u_Re",
                "validate": True,
            },
            "u_Im": {
                "location": "u",
                "is_vector": True,
                "vector_pair": "v_Im",
                "validate": False,
            },
            "v_Im": {
                "location": "v",
                "is_vector": True,
                "vector_pair": "u_Im",
                "validate": False,
            },
        }

        self.variable_info = variable_info

    def _write_into_dataset(self, processed_fields, d_meta):

        # save in new dataset
        ds = xr.Dataset()

        for var_name in processed_fields.keys():
            ds[var_name] = processed_fields[var_name].astype(np.float32)
            ds[var_name].attrs["long_name"] = d_meta[var_name]["long_name"]
            ds[var_name].attrs["units"] = d_meta[var_name]["units"]

        ds = ds.drop_vars(["lat_rho", "lon_rho"])

        return ds

    def _add_global_metadata(self, ds):

        ds.attrs["title"] = "ROMS tidal forcing created by ROMS-Tools"
        # Include the version of roms-tools
        try:
            roms_tools_version = importlib.metadata.version("roms-tools")
        except importlib.metadata.PackageNotFoundError:
            roms_tools_version = "unknown"

        ds.attrs["roms_tools_version"] = roms_tools_version

        ds.attrs["source"] = self.source["name"]
        ds.attrs["model_reference_date"] = str(self.model_reference_date)

        return ds

    def _validate(self, ds):
        """Validates the dataset by checking for NaN values at wet points for specified
        variables, which would indicate missing raw data coverage over the target
        domain.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to validate, containing tidal variables and a mask for wet points.

        Raises
        ------
        ValueError
            If NaN values are found in any of the specified variables at wet points,
            indicating incomplete data coverage.

        Notes
        -----
        This check is applied to the first constituent (`ntides=0`) of each variable in the dataset.
        The method utilizes `self.grid.ds.mask_rho` to determine the wet points in the domain.
        """
        for var_name in ds.data_vars:
            if self.variable_info[var_name]["validate"]:
                if self.variable_info[var_name]["location"] == "rho":
                    mask = self.grid.ds.mask_rho
                elif self.variable_info[var_name]["location"] == "u":
                    mask = self.grid.ds.mask_u
                elif self.variable_info[var_name]["location"] == "v":
                    mask = self.grid.ds.mask_v

                da = ds[var_name].isel(ntides=0)
                nan_check(da, mask)

    def plot(self, var_name, ntides=0) -> None:
        """Plot the specified tidal forcing variable for a given tidal constituent.

        Parameters
        ----------
        var_name : str
            The tidal forcing variable to plot. Options include:

            - "ssh_Re": Real part of tidal elevation.
            - "ssh_Im": Imaginary part of tidal elevation.
            - "pot_Re": Real part of tidal potential.
            - "pot_Im": Imaginary part of tidal potential.
            - "u_Re": Real part of tidal velocity in the x-direction.
            - "u_Im": Imaginary part of tidal velocity in the x-direction.
            - "v_Re": Real part of tidal velocity in the y-direction.
            - "v_Im": Imaginary part of tidal velocity in the y-direction.

        ntides : int, optional
            The index of the tidal constituent to plot. Default is 0, which corresponds
            to the first constituent.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        Raises
        ------
        ValueError
            If the specified field is not one of the valid options.


        Examples
        --------
        >>> tidal_forcing = TidalForcing(grid)
        >>> tidal_forcing.plot("ssh_Re", nc=0)
        """

        if var_name not in self.ds:
            raise ValueError(f"Variable '{var_name}' is not found in dataset.")
        field = self.ds[var_name].isel(ntides=ntides)

        if self.use_dask:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                field = field.load()

        if all(dim in field.dims for dim in ["eta_rho", "xi_rho"]):
            lon_deg = self.grid.ds["lon_rho"]
            lat_deg = self.grid.ds["lat_rho"]
            mask = self.grid.ds["mask_rho"]

        elif all(dim in field.dims for dim in ["eta_rho", "xi_u"]):
            lon_deg = self.grid.ds["lon_u"]
            lat_deg = self.grid.ds["lat_u"]
            mask = self.grid.ds["mask_u"]

        elif all(dim in field.dims for dim in ["eta_v", "xi_rho"]):
            lon_deg = self.grid.ds["lon_v"]
            lat_deg = self.grid.ds["lat_v"]
            mask = self.grid.ds["mask_v"]

        else:
            ValueError("provided field does not have two horizontal dimension")

        field = field.where(mask)
        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)
        field = field.assign_coords({"lon": lon_deg, "lat": lat_deg})

        title = "%s, constituent: %s" % (
            field.long_name,
            self.ds[var_name].ntides[ntides].values.item().decode("utf-8"),
        )

        vmax = max(field.max(), -field.min())
        vmin = -vmax
        cmap = plt.colormaps.get_cmap("RdBu_r")
        cmap.set_bad(color="gray")

        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

        _plot(
            field=field,
            title=title,
            kwargs=kwargs,
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """Save the tidal forcing information to a netCDF4 file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path or filename where the dataset will be saved. If a directory is specified,
            the file will be saved with a default name within that directory.

        Returns
        -------
        Path
            A `Path` object representing the location of the saved file.
        """

        # Ensure filepath is a Path object
        filepath = Path(filepath)

        # Remove ".nc" suffix if present
        if filepath.suffix == ".nc":
            filepath = filepath.with_suffix("")

        dataset_list = [self.ds]
        output_filenames = [str(filepath)]

        saved_filenames = save_datasets(
            dataset_list, output_filenames, use_dask=self.use_dask
        )

        return saved_filenames

    def to_yaml(self, filepath: Union[str, Path]) -> None:
        """Export the parameters of the class to a YAML file, including the version of
        roms-tools.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file where the parameters will be saved.
        """

        forcing_dict = _to_dict(self)
        _write_to_yaml(forcing_dict, filepath)

    @classmethod
    def from_yaml(
        cls,
        filepath: Union[str, Path],
        use_dask: bool = False,
    ) -> "TidalForcing":
        """Create an instance of the TidalForcing class from a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file from which the parameters will be read.
        use_dask: bool, optional
            Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.

        Returns
        -------
        TidalForcing
            An instance of the TidalForcing class.
        """
        filepath = Path(filepath)

        grid = Grid.from_yaml(filepath)
        tidal_forcing_params = _from_yaml(cls, filepath)
        return cls(
            grid=grid,
            **tidal_forcing_params,
            use_dask=use_dask,
        )
