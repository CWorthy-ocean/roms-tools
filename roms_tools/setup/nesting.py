import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from dataclasses import dataclass, field
from typing import Dict, Union
from pathlib import Path
import logging
from scipy.interpolate import interp1d
from roms_tools import Grid
from roms_tools.plot import _plot_nesting
from roms_tools.utils import save_datasets
from roms_tools.setup.utils import (
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    get_boundary_coords,
    wrap_longitudes,
    _to_yaml,
    _from_yaml,
)


@dataclass(frozen=True, kw_only=True)
class Nesting:
    """Represents the relationship between a parent and a child grid in nested ROMS
    simulations. This class facilitates mapping the boundaries of the child grid onto
    the parent grid indices and modifying the child grid topography such that it matches
    the parent topography at the boundaries.

    Parameters
    ----------
    parent_grid : Grid
        The parent grid object, containing information about the larger domain.
    child_grid : Grid
        The child grid object, containing information about the nested domain.
    boundaries : Dict[str, bool], optional
        Dictionary specifying which boundaries of the child grid are used
        in the nesting process. Keys are "south", "east", "north", and "west",
        with boolean values indicating inclusion. Defaults to all boundaries (True).
    child_prefix : str, optional
        Prefix added to variable names in the generated dataset to distinguish
        child grid information. Defaults to "child".
    period : float, optional
        The temporal resolution or output period for boundary variables in the child grid.
        Defaults to hourly.

    Attributes
    ----------
    ds : xr.Dataset
        An xarray Dataset containing the index mappings between the child and parent grids
        for each specified boundary. Includes metadata about grid points, angles,
        and boundary variable outputs.
    """

    parent_grid: Grid
    child_grid: Grid
    boundaries: Dict[str, bool] = field(
        default_factory=lambda: {
            "south": True,
            "east": True,
            "north": True,
            "west": True,
        }
    )
    child_prefix: str = "child"
    period: float = 3600.0

    def __post_init__(self):

        parent_grid_ds = self.parent_grid.ds
        child_grid_ds = self.child_grid.ds

        # Adjust longitude for dateline crossing to prevent interpolation artifacts
        for grid_ds in [parent_grid_ds, child_grid_ds]:
            grid_ds = wrap_longitudes(grid_ds, straddle=self.parent_grid.straddle)

        # Map child boundaries onto parent grid indices
        ds = map_child_boundaries_onto_parent_grid_indices(
            parent_grid_ds,
            child_grid_ds,
            self.boundaries,
            self.child_prefix,
            self.period,
        )
        object.__setattr__(self, "ds", ds)

        # Modify child topography and mask to match the parent grid
        child_grid_ds = modify_child_topography_and_mask(
            parent_grid_ds, child_grid_ds, self.boundaries
        )

        # Convert longitudes back to [0, 360] range
        for grid_ds in [parent_grid_ds, child_grid_ds]:
            grid_ds = wrap_longitudes(grid_ds, straddle=False)
        object.__setattr__(self.parent_grid, "ds", parent_grid_ds)
        object.__setattr__(self.child_grid, "ds", child_grid_ds)

    def plot(self, with_dim_names=False) -> None:
        """Plot the parent and child grids in a single figure.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.
        """

        _plot_nesting(
            self.parent_grid.ds,
            self.child_grid.ds,
            self.parent_grid.straddle,
            with_dim_names,
        )

    def save(
        self,
        filepath: Union[str, Path],
        filepath_child_grid: Union[str, Path],
    ) -> None:
        """Save the nesting and child grid file to netCDF4 files. The child grid file is
        required because the topography and mask of the child grid has been modified.

        Parameters
        ----------
        filepath : Union[str, Path]
            The base path and filename for the output files. The filenames will include the specified path and the `.nc` extension.

        filepath_child_grid : Union[str, Path]
            The base path and filename for saving the childe grid file.

        Returns
        -------
        List[Path]
            A list of `Path` objects for the saved files. Each element in the list corresponds to a file that was saved.
        """

        # Ensure filepath is a Path object
        filepath = Path(filepath)
        filepath_child_grid = Path(filepath_child_grid)

        # Remove ".nc" suffix if present
        if filepath.suffix == ".nc":
            filepath = filepath.with_suffix("")
        if filepath_child_grid.suffix == ".nc":
            filepath_child_grid = filepath_child_grid.with_suffix("")

        dataset_list = [self.ds, self.child_grid.ds]
        output_filenames = [str(filepath), str(filepath_child_grid)]

        saved_filenames = save_datasets(dataset_list, output_filenames)

        return saved_filenames

    def to_yaml(self, filepath: Union[str, Path]) -> None:
        """Export the parameters of the class to a YAML file, including the version of
        roms-tools.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file where the parameters will be saved.
        """

        _to_yaml(self, filepath)

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "Nesting":
        """Create an instance of the Nesting class from a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path to the YAML file from which the parameters will be read.

        Returns
        -------
        Nesting
            An instance of the Nesting class.
        """
        filepath = Path(filepath)

        parent_grid = Grid.from_yaml(filepath, "ParentGrid")
        child_grid = Grid.from_yaml(filepath, "ChildGrid")
        params = _from_yaml(cls, filepath)

        return cls(parent_grid=parent_grid, child_grid=child_grid, **params)


def map_child_boundaries_onto_parent_grid_indices(
    parent_grid_ds,
    child_grid_ds,
    boundaries={"south": True, "east": True, "north": True, "west": True},
    child_prefix="child",
    period=3600.0,
    update_land_indices=True,
):

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
                    parent_grid_ds, lon_child, lat_child, mask_child
                )

                if update_land_indices:
                    i_eta, i_xi = update_indices_if_on_parent_land(
                        i_eta, i_xi, grid_location, parent_grid_ds
                    )

                var_name = f"{child_prefix}_{direction}_{suffix}"
                if grid_location == "rho":
                    ds[var_name] = xr.concat([i_xi, i_eta], dim="two")
                    ds[var_name].attrs[
                        "long_name"
                    ] = f"{grid_location}-points of {direction}ern child boundary mapped onto parent (absolute) grid indices"
                    ds[var_name].attrs["units"] = "non-dimensional"
                    ds[var_name].attrs["output_vars"] = "zeta, temp, salt"
                else:
                    angle_child = child_grid_ds[names["angle"]].isel(
                        **bdry_coords[direction]
                    )
                    ds[var_name] = xr.concat([i_xi, i_eta, angle_child], dim="three")
                    ds[var_name].attrs[
                        "long_name"
                    ] = f"{grid_location}-points  of {direction}ern child boundary mapped onto parent grid (absolute) indices and angle"
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
        dim: f"{child_prefix}_{dim}" for dim in ds.dims if dim not in ["two", "three"]
    }
    ds = ds.rename(dims_to_rename)

    ds = ds.assign_coords(
        {
            "indices_rho": ("two", ["xi", "eta"]),
            "indices_vel": ("three", ["xi", "eta", "angle"]),
        }
    )

    return ds


def interpolate_indices(parent_grid_ds, lon, lat, mask):
    """Interpolate the parent indices to the child grid.

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

    # Check for NaN values
    if np.sum(np.isnan(i)) > 0 or np.sum(np.isnan(j)) > 0:
        raise ValueError(
            "Some points are outside the grid. Please choose either a bigger parent grid or a smaller child grid."
        )

    # Check whether indices are close to border of parent grid
    nxp, nyp = lon_parent.shape
    if np.min(i) < 0 or np.max(i) > nxp - 2:
        logging.warning(
            "Some boundary points of the child grid are very close to the boundary of the parent grid."
        )
    if np.min(j) < 0 or np.max(j) > nyp - 2:
        logging.warning(
            "Some boundary points of the child grid are very close to the boundary of the parent grid."
        )

    return i, j


def update_indices_if_on_parent_land(i_eta, i_xi, grid_location, parent_grid_ds):
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


def modify_child_topography_and_mask(
    parent_grid_ds,
    child_grid_ds,
    boundaries={"south": True, "east": True, "north": True, "west": True},
):

    # regrid parent topography and mask onto child grid
    points = np.column_stack(
        (parent_grid_ds.lon_rho.values.ravel(), parent_grid_ds.lat_rho.values.ravel())
    )
    xi = (child_grid_ds.lon_rho.values, child_grid_ds.lat_rho.values)

    values = parent_grid_ds["h"].values.ravel()
    h_parent_interpolated = griddata(points, values, xi, method="linear")
    h_parent_interpolated = xr.DataArray(
        h_parent_interpolated, dims=("eta_rho", "xi_rho")
    )

    values = parent_grid_ds["mask_rho"].values.ravel()
    mask_parent_interpolated = griddata(points, values, xi, method="linear")
    mask_parent_interpolated = xr.DataArray(
        mask_parent_interpolated, dims=("eta_rho", "xi_rho")
    )

    # compute weight based on distance
    alpha = compute_boundary_distance(child_grid_ds["mask_rho"], boundaries)
    # update child topography and mask to be weighted sum between original child and interpolated parent
    child_grid_ds["h"] = (
        alpha * child_grid_ds["h"] + (1 - alpha) * h_parent_interpolated
    )

    child_mask = (
        alpha * child_grid_ds["mask_rho"] + (1 - alpha) * mask_parent_interpolated
    )
    child_grid_ds["mask_rho"] = xr.where(child_mask >= 0.5, 1, 0)

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
