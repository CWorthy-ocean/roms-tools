import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from scipy.interpolate import griddata, interp1d

from roms_tools import Grid
from roms_tools.plot import plot_nesting
from roms_tools.setup.topography import clip_depth
from roms_tools.setup.utils import (
    Timed,
    check_and_set_boundaries,
    from_yaml,
    get_boundary_coords,
    pop_grid_data,
    to_dict,
    write_to_yaml,
)
from roms_tools.utils import (
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    save_datasets,
    wrap_longitudes,
)


@dataclass(kw_only=True)
class ChildGrid(Grid):
    """Represents a ROMS child grid that is compatible with the provided parent grid.

    This class establishes the relationship between a parent grid and a child grid in ROMS simulations.
    It generates two datasets:

    1. `ds`: Contains child grid variables, ensuring compatibility with the parent grid.
       The child grid topography and mask are adjusted to match the parent grid at the boundaries.

    2. `ds_nesting`: Contains boundary mappings, linking the boundaries of the child grid
       to the corresponding parent grid indices.

    Parameters
    ----------
    parent_grid : Grid
        The parent grid object, providing the reference for the topography and mask of the child grid.
    boundaries : Dict[str, bool]
        Specifies which child grid boundaries ('south', 'east', 'north', 'west') should be adjusted for topography/mask
        and included in `ds_nesting`. If not provided, valid (non-land) boundaries are enabled automatically.
    metadata : Dict[str, Any]
        Dictionary configuring the boundary nesting process, including:

        - `"prefix"` (str): Prefix for variable names in `ds_nesting`. Defaults to `"child"`.
        - `"period"` (float): Temporal resolution for boundary outputs in seconds. Defaults to 3600 (hourly).
    verbose: bool, optional
        Indicates whether to print grid generation steps with timing. Defaults to False.
    """

    parent_grid: Grid
    """The parent grid object, providing the reference for the child topography
    and mask of the child grid."""
    boundaries: dict[str, bool] | None = None
    """Specifies which child grid boundaries (south, east, north, west) should be
    adjusted for topography/mask and included in `ds_nesting`."""
    metadata: dict[str, Any] = field(
        default_factory=lambda: {"prefix": "child", "period": 3600.0}
    )
    """Dictionary configuring the boundary nesting process."""
    verbose: bool = False
    """Whether to print grid generation steps with timing."""

    ds: xr.Dataset = field(init=False, repr=False)
    """An xarray Dataset containing child grid variables aligned with the
    topography and mask of the parent grid at the boundaries."""
    ds_nesting: xr.Dataset = field(init=False, repr=False)
    """An xarray Dataset containing boundary mappings, where child grid boundaries are
    mapped onto parent grid indices."""

    def _map_child_boundaries_onto_parent_grid_indices(self, verbose: bool = False):
        """Maps child grid boundary points onto absolute indices of the parent grid."""
        with Timed(
            "=== Mapping the child grid boundary points onto the indices of the parent grid ===",
            verbose=verbose,
        ):
            self.boundaries = check_and_set_boundaries(
                self.boundaries, self.ds.mask_rho
            )

            # Prepare parent and child grid datasets by adjusting longitudes for dateline crossing
            parent_grid_ds, child_grid_ds = self._prepare_grid_datasets()

            # Map child boundaries onto parent grid indices
            ds_nesting = map_child_boundaries_onto_parent_grid_indices(
                parent_grid_ds,
                child_grid_ds,
                self.boundaries,
                self.metadata["prefix"],
                self.metadata["period"],
            )

            self.ds_nesting = ds_nesting

    def _apply_child_modification(
        self,
        modifier: Callable,
        modifier_name: str,
        verbose: bool = False,
    ):
        """Shared logic for modifying child mask/topography."""
        with Timed(f"=== Modifying the child {modifier_name} ===", verbose=verbose):
            # Prepare datasets (fix dateline)
            parent_grid_ds, child_grid_ds = self._prepare_grid_datasets()

            # Apply modification function
            child_grid_ds = modifier(parent_grid_ds, child_grid_ds)

            # Restore longitudes to 0-360
            _, child_grid_ds = self._finalize_grid_datasets(
                parent_grid_ds, child_grid_ds
            )

            self.ds = child_grid_ds

    def _modify_child_mask(self, verbose: bool = False) -> None:
        """Adjust child grid mask to align with the parent grid."""
        self._apply_child_modification(
            modifier=lambda p, c: modify_child_mask(p, c, self.boundaries),  # type: ignore[arg-type]
            modifier_name="mask",
            verbose=verbose,
        )

    def _modify_child_topography(self, hmin: float, verbose: bool = False) -> None:
        """Adjust child grid topography to align with the parent grid."""
        self._apply_child_modification(
            modifier=lambda p, c: modify_child_topography(p, c, self.boundaries, hmin),
            modifier_name="topography",
            verbose=verbose,
        )

    def update_mask(
        self, mask_shapefile: str | Path | None = None, verbose: bool = False
    ) -> None:
        """
        Update the child grid mask and ensure consistency with the parent grid.

        This method performs the following steps:

        1. Derives the child mask from the provided ``mask_shapefile`` (or from the
           default Natural Earth 10m coastline if ``None``).
        2. Updates the mapping of child boundaries to parent-grid indices.
           This mapping depends on the updated mask, since masked (land) points may
           extend outside the parent grid.
        3. Adjusts the child mask to ensure consistency with the parent mask.

        Parameters
        ----------
        mask_shapefile : str or Path, optional
            Path to a coastal shapefile used to derive the land mask. If ``None``,
            a default coastline dataset is used.
        verbose : bool, default False
            If True, prints timing and progress information.

        Returns
        -------
        None
            Updates the internal datasets (``self.ds`` and ``self.ds_nesting``) in place,
            modifying the mask and ensuring consistent parent-child boundary mapping.
        """
        super().update_mask(mask_shapefile=mask_shapefile, verbose=verbose)
        self._map_child_boundaries_onto_parent_grid_indices(verbose=verbose)
        self._modify_child_mask(verbose=verbose)

    def update_topography(
        self,
        topography_source: dict | None = None,
        hmin: float | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Update the child grid topography and ensure consistency with the parent grid.

        This method performs the following operations:

        1. Regrids the topography from the specified source.
        2. Applies domain-wide and local smoothing.
        3. Enforces the minimum depth constraint ``hmin``.
        4. Adjusts the child grid topography to maintain consistency with the
           parent grid.
        5. Updates the internal dataset (``self.ds``) with the processed bathymetry.

        Parameters
        ----------
        topography_source : dict, optional
            Dictionary describing the topography data source. Expected keys:

            - ``"name"`` (str): Name of the source dataset (e.g., ``"SRTM15"``).
            - ``"path"`` (str or Path): Path to the dataset file.

            If ``None``, the previously configured topography source is used.

        hmin : float, optional
            Minimum allowable ocean depth (meters). If ``None``, the existing value
            is retained.

        verbose : bool, default False
            If ``True``, prints progress messages and timing information.

        Returns
        -------
        None
            Updates ``self.ds`` in place by modifying the topography field. Nothing
            is returned.
        """
        hmin = hmin or self.hmin
        super().update_topography(
            topography_source=topography_source,
            hmin=hmin,
            verbose=verbose,
        )
        self._modify_child_topography(hmin=hmin, verbose=verbose)

    def plot_nesting(self, with_dim_names=False) -> None:
        """Plot the parent and child grids in a single figure.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.
        """
        plot_nesting(
            self.parent_grid.ds,
            self.ds,
            self.parent_grid.straddle,
            with_dim_names,
        )

    def save_nesting(
        self,
        filepath: str | Path,
    ) -> None:
        """Save the nesting information to netCDF4 files.

        Parameters
        ----------
        filepath : Union[str, Path]
            The base path and filename for the output files. The filenames will include the specified path and the `.nc` extension.

        Returns
        -------
        List[Path]
            A list of `Path` objects for the saved files. Each element in the list corresponds to a file that was saved.
        """
        # Ensure filepath is a Path object
        filepath = Path(filepath)

        # Remove ".nc" suffix if present
        if filepath.suffix == ".nc":
            filepath = filepath.with_suffix("")

        dataset_list = [self.ds_nesting]
        output_filenames = [str(filepath)]

        saved_filenames = save_datasets(dataset_list, output_filenames)

        return saved_filenames

    def to_yaml(self, filepath: str | Path) -> None:
        """Export the parameters of the class to a YAML file, including the version of
        roms-tools.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file where the parameters will be saved.
        """
        forcing_dict = to_dict(self, exclude=["ds_nesting"])
        forcing_dict["ChildGrid"] = pop_grid_data(forcing_dict["ChildGrid"])
        write_to_yaml(forcing_dict, filepath)

    @classmethod
    def from_yaml(
        cls, filepath: str | Path, verbose: bool = False, **kwargs: Any
    ) -> "ChildGrid":
        """Create an instance of the ChildGrid class from a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file from which the parameters will be read.
        verbose : bool, optional
            Indicates whether to print grid generation steps with timing. Defaults to False.
        **kwargs : Any
            Additional keyword arguments passed to Grid.from_yaml.

        Returns
        -------
        Nesting
            An instance of the ChildGrid class.
        """
        filepath = Path(filepath)

        parent_grid = Grid.from_yaml(
            filepath, verbose=verbose, section_name="ParentGrid"
        )
        params = from_yaml(cls, filepath)

        return cls(parent_grid=parent_grid, **params, verbose=verbose)

    def _prepare_grid_datasets(self) -> tuple[xr.Dataset, xr.Dataset]:
        """Prepare parent and child grid datasets by adjusting longitudes for dateline
        crossing.

        This method ensures that longitudes are properly wrapped to avoid interpolation artifacts
        and returns the updated parent and child grid datasets.

        Returns
        -------
        Tuple[xr.Dataset, xr.Dataset]
            A tuple containing:
            - The modified parent grid dataset.
            - The modified child grid dataset.
        """
        parent_grid_ds = wrap_longitudes(
            self.parent_grid.ds.copy(), straddle=self.parent_grid.straddle
        )
        child_grid_ds = wrap_longitudes(
            self.ds.copy(), straddle=self.parent_grid.straddle
        )

        return parent_grid_ds, child_grid_ds

    def _finalize_grid_datasets(
        self, parent_grid_ds: xr.Dataset, child_grid_ds: xr.Dataset
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Finalize the grid datasets by converting longitudes back to the [0, 360]
        range.

        Parameters
        ----------
        parent_grid_ds : xr.Dataset
            The parent grid dataset after modifications.

        child_grid_ds : xr.Dataset
            The child grid dataset after modifications.

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset]
            The finalized parent and child grid datasets with longitudes wrapped to [0, 360].

        """
        parent_grid_ds = wrap_longitudes(parent_grid_ds, straddle=False)
        child_grid_ds = wrap_longitudes(child_grid_ds, straddle=False)

        return parent_grid_ds, child_grid_ds

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        theta_s: float | None = None,
        theta_b: float | None = None,
        hc: float | None = None,
        N: int | None = None,
        verbose: bool = False,
    ) -> "ChildGrid":
        """This method is disabled in this subclass.

        .. noindex::
        """
        raise NotImplementedError(
            "The 'from_file' method is disabled in this subclass."
        )


def map_child_boundaries_onto_parent_grid_indices(
    parent_grid_ds: xr.Dataset,
    child_grid_ds: xr.Dataset,
    boundaries: dict = {"south": True, "east": True, "north": True, "west": True},
    prefix: str = "child",
    period: float = 3600.0,
    update_land_indices: bool = True,
):
    """Maps child grid boundary points onto absolute indices of the parent grid.

    This function interpolates the spatial indices of the child grid boundaries onto
    the parent grid, ensuring alignment between the two grids. It supports all four
    boundaries (south, east, north, and west) and considers different grid locations
    (`rho`, `u`, and `v`). Additionally, it updates land indices if they fall onto
    land points in the parent grid.

    Parameters
    ----------
    parent_grid_ds : xarray.Dataset
        The parent grid dataset containing longitude, latitude, and mask variables.

    child_grid_ds : xarray.Dataset
        The child grid dataset containing longitude, latitude, mask, and angle variables.

    boundaries : dict, optional
        A dictionary specifying which child boundaries should be mapped onto the parent grid.
        Keys should be `"south"`, `"east"`, `"north"`, and `"west"`, with boolean values
        indicating whether to process each boundary. Defaults to mapping all boundaries.

    prefix : str, optional
        A string prefix for naming the output variables in the resulting dataset.
        Defaults to `"child"`.

    period : float, optional
        The output period (in seconds) to be assigned to the mapped boundary indices.
        Defaults to `3600.0` (1 hour).

    update_land_indices : bool, optional
        If `True`, updates indices that fall on land in the parent grid to nearby ocean points.
        Defaults to `True`.

    Returns
    -------
    xarray.Dataset
        A dataset containing the mapped boundary indices for `rho`, `u`, and `v` grid points.
        - For `rho` points: Contains mapped `xi` and `eta` indices.
        - For `u` and `v` points: Contains mapped `xi`, `eta`, and angle values.
        - Attributes include long names, output variable names, units, and output period.
    """
    bdry_coords_dict = get_boundary_coords()

    # add angles at u- and v-points

    child_grid_ds["angle_u"] = interpolate_from_rho_to_u(child_grid_ds["angle"])
    child_grid_ds["angle_v"] = interpolate_from_rho_to_v(child_grid_ds["angle"])

    ds = xr.Dataset()

    for direction in ["south", "east", "north", "west"]:
        if boundaries[direction]:
            for grid_location in ["rho", "u", "v"]:
                names = {
                    "latitude": f"lat_{grid_location}",
                    "longitude": f"lon_{grid_location}",
                    "mask": f"mask_{grid_location}",
                    "angle": f"angle_{grid_location}",
                }
                bdry_coords = bdry_coords_dict[grid_location]
                if grid_location == "rho":
                    suffix = "r"
                else:
                    suffix = grid_location

                lon_child = child_grid_ds[names["longitude"]].isel(
                    **bdry_coords[direction]
                )
                lat_child = child_grid_ds[names["latitude"]].isel(
                    **bdry_coords[direction]
                )

                mask_child = child_grid_ds[names["mask"]].isel(**bdry_coords[direction])

                i_eta, i_xi = interpolate_indices(
                    parent_grid_ds, lon_child, lat_child, mask_child, direction
                )

                if update_land_indices and mask_child.sum() > 0:
                    i_eta, i_xi = update_indices_if_on_parent_land(
                        i_eta, i_xi, grid_location, parent_grid_ds
                    )

                var_name = f"{prefix}_{direction}_{suffix}"
                if grid_location == "rho":
                    ds[var_name] = xr.concat([i_xi, i_eta], dim="two")
                    ds[var_name].attrs["long_name"] = (
                        f"{grid_location}-points of {direction}ern child boundary mapped onto parent (absolute) grid indices"
                    )
                    ds[var_name].attrs["units"] = "non-dimensional"
                    ds[var_name].attrs["output_vars"] = "zeta, temp, salt"
                else:
                    angle_child = child_grid_ds[names["angle"]].isel(
                        **bdry_coords[direction]
                    )
                    ds[var_name] = xr.concat([i_xi, i_eta, angle_child], dim="three")
                    ds[var_name].attrs["long_name"] = (
                        f"{grid_location}-points  of {direction}ern child boundary mapped onto parent grid (absolute) indices and angle"
                    )
                    ds[var_name].attrs["units"] = "non-dimensional and radian"

                    if grid_location == "u":
                        ds[var_name].attrs["output_vars"] = "ubar, u, up"
                    elif grid_location == "v":
                        ds[var_name].attrs["output_vars"] = "vbar, v, vp"

                ds[var_name].attrs["output_period"] = period

    vars_to_drop = ["lat_rho", "lon_rho", "lat_u", "lon_u", "lat_v", "lon_v"]
    vars_to_drop_existing = [var for var in vars_to_drop if var in ds]
    ds = ds.drop_vars(vars_to_drop_existing)

    # Rename dimensions
    dims_to_rename = {
        dim: f"{prefix}_{dim}" for dim in ds.dims if dim not in ["two", "three"]
    }
    ds = ds.rename(dims_to_rename)

    return ds


def interpolate_indices(
    parent_grid_ds: xr.Dataset,
    lon: xr.DataArray,
    lat: xr.DataArray,
    mask: xr.DataArray,
    direction: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Interpolate the parent indices to the child grid boundary.

    Uses the parent grid ``lon_rho``/``lat_rho`` coordinates to compute
    fractional i/j indices at child-boundary longitude/latitude points using
    linear interpolation. The function verifies that all ocean child points
    (based on ``mask``) lie within the parent grid and warns if child
    boundary points fall near the parent-grid edges. Land child points that fall
    outside the parent grid (i.e., interpolation returns NaN) are filled with
    ``-1e5``.

    Parameters
    ----------
    parent_grid_ds : xarray.Dataset
        Grid information of parent grid.
    lon : xarray.DataArray
        Longitudes of the child grid where interpolation is desired.
    lat : xarray.DataArray
        Latitudes of the child grid where interpolation is desired.
    mask: xarray.DataArray
        Mask for the child longitudes and latitudes under consideration.
    direction : str
        Boundary identifier (``"south"``, ``"north"``, ``"east"``, ``"west"``).
        Used for generating informative warnings or errors.
    Returns
    -------
    i : xarray.DataArray
        Interpolated i-indices for the child grid.
    j : xarray.DataArray
        Interpolated j-indices for the child grid.
    """
    i_eta = np.arange(-0.5, len(parent_grid_ds.eta_rho) + -0.5, 1)
    i_xi = np.arange(-0.5, len(parent_grid_ds.xi_rho) + -0.5, 1)

    parent_grid_ds = parent_grid_ds.assign_coords(
        i_eta=("eta_rho", i_eta)
    ).assign_coords(i_xi=("xi_rho", i_xi))

    lon_parent = parent_grid_ds.lon_rho
    lat_parent = parent_grid_ds.lat_rho
    i_parent = parent_grid_ds.i_eta
    j_parent = parent_grid_ds.i_xi

    # Create meshgrid
    j_parent, i_parent = np.meshgrid(j_parent.values, i_parent.values)

    # Flatten the input coordinates and indices for griddata
    points = np.column_stack((lon_parent.values.ravel(), lat_parent.values.ravel()))
    i_parent_flat = i_parent.ravel()
    j_parent_flat = j_parent.ravel()

    # Interpolate the i and j indices
    i = griddata(points, i_parent_flat, (lon.values, lat.values), method="linear")
    j = griddata(points, j_parent_flat, (lon.values, lat.values), method="linear")

    i = xr.DataArray(i, dims=lon.dims)
    j = xr.DataArray(j, dims=lon.dims)

    # Check whether ocean child points fall outside the parent grid
    if (
        i.where(mask, other=0.0).isnull().any()
        or j.where(mask, other=0.0).isnull().any()
    ):
        raise ValueError(
            f"Some wet points on the {direction}ern boundary of the child grid lie "
            "outside the parent grid. Please use a larger parent grid or a smaller child grid."
        )

    # Try to fix NaNs if there only a few per boundary. Fix with out-of-bounds points is not valid.
    nxp, nyp = lon_parent.shape
    nan_idx = (
        i.isnull() | j.isnull() | (i > nxp - 2) | (i < 0) | (j > nyp - 2) | (j < 0)
    )

    idx = xr.DataArray(np.arange(i.size), dims=i.dims)

    # Interpolate indices for points that are invalid (NaN or out-of-bounds)
    if nan_idx.any() and not nan_idx.all():
        idx_tmp = idx.where(~nan_idx, drop=True)  # valid poins
        i_tmp = i.where(~nan_idx, drop=True)  # valid points
        j_tmp = j.where(~nan_idx, drop=True)  # valid points

        interp_i = interp1d(
            idx_tmp.values, i_tmp.values, kind="nearest", fill_value="extrapolate"
        )
        interp_j = interp1d(
            idx_tmp.values, j_tmp.values, kind="nearest", fill_value="extrapolate"
        )

        i[nan_idx.values] = interp_i(idx[nan_idx].values)
        j[nan_idx.values] = interp_j(idx[nan_idx].values)

    # This should only occur in rare edge cases
    if i.isnull().any() or j.isnull().any():
        raise ValueError(
            f"Mapping failed: the {direction}ern boundary of the child grid could not be "
            "mapped onto parent-grid indices. Please adjust the parent/child grid configuration."
        )

    # Warn if child boundary points are near the edges of the parent grid
    if (
        i.where(mask).min() < 0
        or i.where(mask).max() > nxp - 2
        or j.where(mask).min() < 0
        or j.where(mask).max() > nyp - 2
    ):
        logging.warning(
            f"Some wet points on the {direction}ern boundary of the child grid lie very close to the edges of the parent grid."
        )

    return i, j


def update_indices_if_on_parent_land(
    i_eta: xr.DataArray,
    i_xi: xr.DataArray,
    grid_location: str,
    parent_grid_ds: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Finds points that are in the parent land mask but not land masked in the child
    and replaces parent indices with nearest neighbor wet points.

    Parameters
    ----------
    i_eta : xarray.DataArray
        Interpolated i_eta-indices for the child grid.
    i_xi : xarray.DataArray
        Interpolated i_xi-indices for the child grid.
    mask: xarray.DataArray
        Mask for the child longitudes and latitudes under consideration.
    grid_location : str
        Location type ('rho', 'u', 'v').
    parent_grid_ds : xarray.Dataset
        Grid information of parent grid.

    Returns
    -------
    i_eta : xarray.DataArray
        Updated i_eta-indices for the child grid.
    i_xi : xarray.DataArray
        Updated i_xi-indices for the child grid.
    """
    if grid_location == "rho":
        i_eta_rho = i_eta + 0.5
        i_xi_rho = i_xi + 0.5
        mask_rho = parent_grid_ds.mask_rho
        summed_mask = np.zeros_like(i_eta_rho)

        for i in range(len(i_eta_rho)):
            i_eta_lower = int(np.floor(i_eta_rho[i]))
            i_xi_lower = int(np.floor(i_xi_rho[i]))
            mask = mask_rho.isel(
                eta_rho=slice(i_eta_lower, i_eta_lower + 2),
                xi_rho=slice(i_xi_lower, i_xi_lower + 2),
            )
            summed_mask[i] = np.sum(mask)

    elif grid_location in ["u", "v"]:
        i_eta_u = i_eta + 0.5
        i_xi_u = i_xi

        mask_u = parent_grid_ds.mask_u
        summed_mask_u = np.zeros_like(i_eta_u)

        for i in range(len(i_eta_u)):
            i_eta_lower = int(np.floor(i_eta_u[i]))
            i_xi_lower = int(np.floor(i_xi_u[i]))
            mask = mask_u.isel(
                eta_rho=slice(i_eta_lower, i_eta_lower + 2),
                xi_u=slice(i_xi_lower, i_xi_lower + 2),
            )
            summed_mask_u[i] = np.sum(mask)

        i_eta_v = i_eta
        i_xi_v = i_xi + 0.5

        mask_v = parent_grid_ds.mask_v
        summed_mask_v = np.zeros_like(i_xi_v)

        for i in range(len(i_eta_v)):
            i_eta_lower = int(np.floor(i_eta_v[i]))
            i_xi_lower = int(np.floor(i_xi_v[i]))
            mask = mask_v.isel(
                eta_v=slice(i_eta_lower, i_eta_lower + 2),
                xi_rho=slice(i_xi_lower, i_xi_lower + 2),
            )
            summed_mask_v[i] = np.sum(mask)

        summed_mask = summed_mask_u * summed_mask_v

    # Filter out points where summed_mask is 0
    valid_points = summed_mask != 0
    x_mod = np.arange(len(summed_mask))[valid_points]
    i_eta_mod = i_eta[valid_points]
    i_xi_mod = i_xi[valid_points]

    # Handle indices where summed_mask is 0
    indx = np.where(summed_mask == 0)[0]
    if len(indx) > 0:
        i_eta_interp = interp1d(
            x_mod, i_eta_mod, kind="nearest", fill_value="extrapolate"
        )
        i_xi_interp = interp1d(
            x_mod, i_xi_mod, kind="nearest", fill_value="extrapolate"
        )

        i_eta[indx] = i_eta_interp(indx)
        i_xi[indx] = i_xi_interp(indx)

    return i_eta, i_xi


def _interpolate_parent(
    parent_da: xr.DataArray, child_da: xr.DataArray
) -> xr.DataArray:
    """
    Interpolate data from a parent grid onto a child grid using linear interpolation.

    Parameters
    ----------
    parent_da : xr.DataArray
        The data array on the parent grid. Must have coordinates `lon_rho` and `lat_rho`.
    child_da : xr.DataArray
        The target child grid data array. Must have coordinates `lon_rho` and `lat_rho`.

    Returns
    -------
    xr.DataArray
        The interpolated data on the child grid, with dimensions ("eta_rho", "xi_rho").
    """
    points = np.column_stack(
        (parent_da.lon_rho.values.ravel(), parent_da.lat_rho.values.ravel())
    )
    xi = (child_da.lon_rho.values, child_da.lat_rho.values)
    values = parent_da.values.ravel()

    parent_interpolated = griddata(points, values, xi, method="linear")
    return xr.DataArray(parent_interpolated, dims=("eta_rho", "xi_rho"))


def modify_child_mask(
    parent_grid_ds: xr.Dataset,
    child_grid_ds: xr.Dataset,
    boundaries: dict = {"south": True, "east": True, "north": True, "west": True},
):
    """Adjust the child gridmask to align with the parent grid.

    The mask of the child grid is adjusted using a weighted sum based on the boundary distance.

    Parameters
    ----------
    parent_grid_ds : xarray.Dataset
        The parent grid dataset containing `mask_rho` (land-sea mask).

    child_grid_ds : xarray.Dataset
        The child grid dataset whose `mask_rho` will be modified.

    boundaries : dict, optional
        A dictionary specifying which boundaries should be modified. Expected keys:
        - `"south"` (bool): Whether to adjust the southern boundary.
        - `"east"` (bool): Whether to adjust the eastern boundary.
        - `"north"` (bool): Whether to adjust the northern boundary.
        - `"west"` (bool): Whether to adjust the western boundary.
        Defaults to modifying all boundaries.

    Returns
    -------
    xarray.Dataset
        The updated child grid dataset with modified mask (`mask_rho`).
    """
    # regrid parent mask onto child grid
    mask_parent_interpolated = _interpolate_parent(
        parent_grid_ds["mask_rho"], child_grid_ds["mask_rho"]
    )

    # compute weight based on distance
    alpha = compute_boundary_distance(child_grid_ds["mask_rho"], boundaries)

    # update child mask to be weighted sum between original child and interpolated parent
    child_mask = (
        alpha * child_grid_ds["mask_rho"] + (1 - alpha) * mask_parent_interpolated
    )
    child_grid_ds["mask_rho"] = xr.where(child_mask >= 0.5, 1, 0)

    return child_grid_ds


def modify_child_topography(
    parent_grid_ds,
    child_grid_ds,
    boundaries={"south": True, "east": True, "north": True, "west": True},
    hmin=5.0,
):
    """Adjust the child grid topography to align with the parent grid.

    The topography of the child grid is adjusted using a weighted sum based on the boundary distance,
    and the depth values are clipped to enforce a minimum depth constraint.

    Parameters
    ----------
    parent_grid_ds : xarray.Dataset
        The parent grid dataset containing `h` (topography).

    child_grid_ds : xarray.Dataset
        The child grid dataset whose `h` will be modified.

    boundaries : dict, optional
        A dictionary specifying which boundaries should be modified. Expected keys:
        - `"south"` (bool): Whether to adjust the southern boundary.
        - `"east"` (bool): Whether to adjust the eastern boundary.
        - `"north"` (bool): Whether to adjust the northern boundary.
        - `"west"` (bool): Whether to adjust the western boundary.
        Defaults to modifying all boundaries.

    hmin : float, optional
        The minimum allowable ocean depth (in meters). Depth values in the modified
        child topography will be clipped to ensure they meet or exceed this value.
        Defaults to `5.0` meters.

    Returns
    -------
    xarray.Dataset
        The updated child grid dataset with modified topography (`h`).
    """
    # regrid parent mask onto child grid
    h_parent_interpolated = _interpolate_parent(parent_grid_ds["h"], child_grid_ds["h"])

    # compute weight based on distance
    alpha = compute_boundary_distance(child_grid_ds["mask_rho"], boundaries)

    # update child topography to be weighted sum between original child and interpolated parent
    child_grid_ds["h"] = (
        alpha * child_grid_ds["h"] + (1 - alpha) * h_parent_interpolated
    )

    # Clip depth on modified child topography
    child_grid_ds["h"] = clip_depth(child_grid_ds["h"], hmin)

    return child_grid_ds


def compute_boundary_distance(
    child_mask, boundaries={"south": True, "east": True, "north": True, "west": True}
):
    """Computes a normalized distance field from the boundaries of a grid, given a mask
    and boundary conditions. The normalized distance values range from 0 (boundary) to 1
    (inner grid).

    Parameters
    ----------
    child_mask : xr.DataArray
        A 2D xarray DataArray representing the land/sea mask of the grid (1 for sea, 0 for land),
        with dimensions ("eta_rho", "xi_rho").
    boundaries : dict, optional
        A dictionary specifying which boundaries are open. Keys are "south", "east", "north", "west",
        with boolean values indicating whether the boundary is open.

    Returns
    -------
    xr.DataArray
        A 2D DataArray with normalized distance values.
    """
    dist = np.full_like(child_mask, 1e6, dtype=float)
    nx, ny = child_mask.shape
    n = max(nx, ny)

    x = np.arange(nx) / n
    y = np.arange(ny) / n
    x, y = np.meshgrid(x, y, indexing="ij")

    trans = 0.05
    width = int(np.ceil(n * trans))

    if boundaries["south"]:
        bx = x[:, 0][child_mask[:, 0] > 0]
        by = y[:, 0][child_mask[:, 0] > 0]
        for i in range(len(bx)):
            dtmp = (x[:, :width] - bx[i]) ** 2 + (y[:, :width] - by[i]) ** 2
            dist[:, :width] = np.minimum(dist[:, :width], dtmp)

    if boundaries["east"]:
        bx = x[-1, :][child_mask[-1, :] > 0]
        by = y[-1, :][child_mask[-1, :] > 0]
        for i in range(len(bx)):
            dtmp = (x[nx - width : nx, :] - bx[i]) ** 2 + (
                y[nx - width : nx, :] - by[i]
            ) ** 2
            dist[nx - width : nx, :] = np.minimum(dist[nx - width : nx, :], dtmp)

    if boundaries["north"]:
        bx = x[:, -1][child_mask[:, -1] > 0]
        by = y[:, -1][child_mask[:, -1] > 0]
        for i in range(len(bx)):
            dtmp = (x[:, ny - width : ny] - bx[i]) ** 2 + (
                y[:, ny - width : ny] - by[i]
            ) ** 2
            dist[:, ny - width : ny] = np.minimum(dist[:, ny - width : ny], dtmp)

    if boundaries["west"]:
        bx = x[0, :][child_mask[0, :] > 0]
        by = y[0, :][child_mask[0, :] > 0]
        for i in range(len(bx)):
            dtmp = (x[:width, :] - bx[i]) ** 2 + (y[:width, :] - by[i]) ** 2
            dist[:width, :] = np.minimum(dist[:width, :], dtmp)

    dist = np.sqrt(dist)
    dist[dist > trans] = trans
    dist = dist / trans
    alpha = 0.5 - 0.5 * np.cos(np.pi * dist)

    alpha = xr.DataArray(alpha, dims=("eta_rho", "xi_rho"))

    return alpha
