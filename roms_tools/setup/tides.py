import importlib.metadata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from roms_tools import Grid
from roms_tools.datasets.lat_lon_datasets import TPXOManager
from roms_tools.plot import plot
from roms_tools.processing_methods import RegridConfig, _xesmf_available
from roms_tools.regrid import (
    build_lateral_regridder,
    select_source_mask,
)
from roms_tools.setup.utils import (
    apply_scipy_fallback_fill,
    apply_source_prefill,
    check_source_coverage,
    from_yaml,
    get_target_coords,
    get_variable_metadata,
    get_vector_pairs,
    nan_check,
    substitute_nans_by_fillvalue,
    to_dict,
    write_to_yaml,
)
from roms_tools.utils import (
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    rotate_velocities,
    save_datasets,
)


@dataclass(kw_only=True)
class TidalForcing:
    """Represents tidal forcing for ROMS.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
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
    prefill : str or None, optional
        How to fill masked land cells in the *source* tidal data before
        regridding. (TPXO stores zeros over land; the source mask, not NaNs,
        marks those cells invalid.) The default (``None``) applies **no** source
        prefill: with xESMF installed, masked bilinear interpolation plus
        destination extrapolation (``extrap_method``) produces NaN-free tidal
        fields directly; without xESMF, the source is automatically pre-filled
        with a cheap nearest-neighbor fill before scipy interpolation. Set
        ``prefill`` to fill the whole-domain source first (the regrid is then
        plain bilinear and ``extrap_method`` is ignored). Options:

          - ``"2d_lateral_fill"`` -- legacy AMG Poisson fill (smoothest, slow;
            no xESMF required). Together with ``regrid_method="scipy"`` this
            reproduces the pre-v4 tidal output.
          - ``"inverse_dist"`` -- xESMF inverse-distance-weighted source fill
            (tunable via ``prefill_kwargs``; requires xESMF).
          - ``"nearest_s2d"`` -- xESMF nearest-source fill (requires xESMF).
          - ``"nearest_neighbor"`` -- cheap distance-transform fill (no xESMF;
            also the automatic fallback when xESMF is unavailable).
          - ``"creep_fill"`` -- xESMF truncated Laplace-style diffusion source
            fill (tunable via ``prefill_kwargs``; requires xESMF). **Not
            available in current released xESMF** -- provided for use once a
            supporting xESMF is installed.

        Defaults to ``None``.
    prefill_kwargs : dict, optional
        Method-specific options for ``prefill``: ``num_src_pnts`` /
        ``dist_exponent`` for ``"inverse_dist"``; ``num_levels`` for
        ``"creep_fill"``. Ignored by the other methods. Defaults to ``None``.
    regrid_method : str or None, optional
        Horizontal regrid engine, chosen independently of ``prefill``:

          - ``None`` / ``"auto"`` (default) -- use xESMF if it is installed
            (lazy, weight-reused, faster on large grids), otherwise scipy.
          - ``"xesmf"`` -- force the xESMF regridder (raises if xESMF is absent).
          - ``"scipy"`` -- force scipy ``interp``. Byte-reproducible with pre-v4
            outputs; when ``prefill`` is ``None`` a nearest-neighbor source
            pre-fill is applied automatically so scipy cannot propagate NaNs.

        Note that ``inverse_dist`` / ``nearest_s2d`` *prefills* still require
        xESMF for the fill step regardless of ``regrid_method``. Defaults to
        ``None``.
    extrap_method : str or None, optional
        xESMF *destination* extrapolation used on the default path
        (``prefill is None``) to fill target points whose source neighbors are
        all masked land or out of range, guaranteeing NaN-free output.
        ``"inverse_dist"`` (the effective default) gives an
        inverse-distance-weighted average of the nearest source points (smoothly
        varying); ``"nearest_s2d"`` uses the single nearest source point.
        Ignored when ``prefill`` is set. Defaults to ``None`` (treated as
        ``"inverse_dist"``).
    extrap_kwargs : dict, optional
        Method-specific options for ``extrap_method``: ``num_src_pnts`` /
        ``dist_exponent`` for ``"inverse_dist"``. Defaults to ``None``.

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
    """Object representing the grid information."""
    source: dict[str, str | Path | dict[str, str | Path]]
    """Dictionary specifying the source of the tidal data."""
    ntides: int = 10
    """Number of constituents to consider."""
    model_reference_date: datetime = datetime(2000, 1, 1)
    """The reference date for the ROMS simulation."""
    use_dask: bool = False
    """Whether to use dask for processing."""
    bypass_validation: bool = False
    """Whether to skip validation checks in the processed data."""
    prefill: str | None = None
    """Source-side fill applied before lateral regridding. ``None`` (default) applies
    **no** whole-domain source fill: with xESMF the masked-bilinear regrid plus
    destination extrapolation (``extrap_method``) produces NaN-free output directly;
    without xESMF a nearest-neighbor pre-fill is applied automatically before scipy
    interpolation. Set to ``"2d_lateral_fill"`` (legacy AMG Poisson fill),
    ``"nearest_neighbor"``, ``"inverse_dist"``, ``"nearest_s2d"``, or ``"creep_fill"``
    to fill the whole-domain source first (the last three require xESMF)."""
    prefill_kwargs: dict | None = None
    """Method-specific options for ``prefill`` (e.g. ``num_src_pnts`` /
    ``dist_exponent``)."""
    regrid_method: str | None = None
    """Horizontal regrid engine, chosen independently of ``prefill``: ``None``/``"auto"``
    uses xESMF when installed (else scipy), ``"xesmf"`` forces xESMF, ``"scipy"`` forces
    scipy."""
    extrap_method: str | None = None
    """xESMF destination extrapolation used on the default no-prefill path (``None`` is
    treated as ``"inverse_dist"``) to fill target points whose source neighbors are all
    masked. Ignored when ``prefill`` is set or on the scipy path."""
    extrap_kwargs: dict | None = None
    """Method-specific options for ``extrap_method``."""

    ds: xr.Dataset = field(init=False, repr=False)
    """An xarray Dataset containing post-processed variables ready for input into
    ROMS."""

    def __post_init__(self):
        self._resolve_prefill_options()
        self._input_checks()
        target_coords = get_target_coords(self.grid)

        tidal_data = self._get_data()

        for key, data in tidal_data.datasets.items():
            if key != "omega":
                data.choose_subdomain(
                    target_coords,
                    unchunk_lateral_dims=True,
                )
                # Enforce double precision to ensure reproducibility
                data.convert_to_float64()

        tidal_data.correct_tides(self.model_reference_date)

        self._set_variable_info()
        var_names = self.variable_info.keys()

        processed_fields = {}

        # source fill (or masked-regrid setup) and lateral regridding
        for key, data in tidal_data.datasets.items():
            if key != "omega":
                # On the no-prefill xESMF path, destination extrapolation would
                # silently fill points outside the source coverage. Guard against
                # a grid that outruns the data (masked land *within* coverage is
                # still filled by the masked regrid + extrapolation).
                if self._regrid.extrap_is_active:
                    check_source_coverage(data, target_coords, "TPXO")
                apply_source_prefill(data, self._regrid, self.prefill_kwargs)
                apply_scipy_fallback_fill(data, self._regrid)

                # Each TPXO sub-dataset carries its own location-correct "mask"
                # (h/u/v staggered grids), so no tracer/velocity mask split is
                # needed; the mask only applies on the no-prefill xESMF path.
                source_mask = select_source_mask(
                    data.ds,
                    is_vector=False,
                    use_xesmf=self._regrid.use_xesmf,
                    prefill=self._regrid.prefill,
                )
                lateral_regrid = build_lateral_regridder(
                    target_coords, data, self._regrid, source_mask
                )

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

    def _resolve_prefill_options(self) -> None:
        """Build the validated :class:`RegridConfig` from the public options.

        Delegates all prefill/extrap/regrid validation to
        :meth:`RegridConfig.from_options`, then writes the resolved ``prefill`` back
        to the public field (as a plain string or ``None``) so the YAML round-trip
        emits a clean ``prefill``. Derived state (``use_xesmf``,
        ``extrap_is_active``, ...) is read off ``self._regrid``.
        """
        self._regrid = RegridConfig.from_options(
            prefill=self.prefill,
            prefill_kwargs=self.prefill_kwargs,
            regrid_method=self.regrid_method,
            extrap_method=self.extrap_method,
            extrap_kwargs=self.extrap_kwargs,
            xesmf_available=_xesmf_available(),
        )
        self.prefill = (
            None if self._regrid.prefill is None else str(self._regrid.prefill)
        )

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

            elif isinstance(self.source["path"], str | Path):
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

        ds.attrs["prefill"] = str(self.prefill)
        ds.attrs["regrid_method"] = "xesmf" if self._regrid.use_xesmf else "scipy"
        ds.attrs["extrap_method"] = str(self._regrid.effective_extrap)

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

    def plot(
        self,
        var_name: str,
        ntides: int = 0,
        save_path: str | None = None,
    ) -> None:
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
            If the specified field is not one of the valid options.


        Examples
        --------
        >>> tidal_forcing = TidalForcing(grid)
        >>> tidal_forcing.plot("ssh_Re", ntides=0)
        """
        if var_name not in self.ds:
            raise ValueError(f"Variable '{var_name}' is not found in dataset.")

        field = self.ds[var_name].isel(ntides=ntides)

        if self.use_dask:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                field = field.load()

        plot(
            field=field,
            grid_ds=self.grid.ds,
            save_path=save_path,
            cmap_name="RdBu_r",
        )

    def save(self, filepath: str | Path) -> None:
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

    def to_yaml(self, filepath: str | Path) -> None:
        """Export the parameters of the class to a YAML file, including the version of
        roms-tools.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file where the parameters will be saved.
        """
        forcing_dict = to_dict(self, exclude=["use_dask"])
        write_to_yaml(forcing_dict, filepath)

    @classmethod
    def from_yaml(
        cls,
        filepath: str | Path,
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
        tidal_forcing_params = from_yaml(cls, filepath)
        return cls(
            grid=grid,
            **tidal_forcing_params,
            use_dask=use_dask,
        )
