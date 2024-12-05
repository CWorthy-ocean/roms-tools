import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from dataclasses import dataclass, field
from typing import Dict
from roms_tools.setup.grid import Grid
from roms_tools.setup.utils import (
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
)
from roms_tools.setup.plot import _plot_nesting
import warnings
from scipy.interpolate import interp1d


@dataclass(frozen=True, kw_only=True)
class NestBoundaries:
    """
    Represents relation between parent and child grid for nested ROMS simulations.

    Parameters
    ----------
    parent_grid : Grid
        Object representing the parent grid information.
    child_grid :
        Object representing the child grid information.
    boundaries : Dict[str, bool], optional
        Dictionary specifying which boundaries of the child grid are to be forced (south, east, north, west). Default is all True.
    child_prefix : str
    Attributes
    ----------
    ds : xr.Dataset
        Xarray Dataset containing the index information for the child grid.
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

    def __post_init__(self):

        # Boundary coordinates for rho-points
        bdry_coords_rho = {
            "south": {"eta_rho": 0},
            "east": {"xi_rho": -1},
            "north": {"eta_rho": -1},
            "west": {"xi_rho": 0},
        }
        # How to rename the dimensions at rho-points
        rename_rho = {
            "south": {"xi_rho": "xi_rho_south"},
            "east": {"eta_rho": "eta_rho_east"},
            "north": {"xi_rho": "xi_rho_north"},
            "west": {"eta_rho": "eta_rho_west"},
        }

        # Boundary coordinates for u-points
        bdry_coords_u = {
            "south": {"eta_rho": 0},
            "east": {"xi_u": -1},
            "north": {"eta_rho": -1},
            "west": {"xi_u": 0},
        }
        # How to rename the dimensions at u-points
        rename_u = {
            "south": {"xi_u": "xi_u_south"},
            "east": {"eta_rho": "eta_u_east"},
            "north": {"xi_u": "xi_u_north"},
            "west": {"eta_rho": "eta_u_west"},
        }

        # Boundary coordinates for v-points
        bdry_coords_v = {
            "south": {"eta_v": 0},
            "east": {"xi_rho": -1},
            "north": {"eta_v": -1},
            "west": {"xi_rho": 0},
        }
        # How to rename the dimensions at v-points
        rename_v = {
            "south": {"xi_rho": "xi_v_south"},
            "east": {"eta_v": "eta_v_east"},
            "north": {"xi_rho": "xi_v_north"},
            "west": {"eta_v": "eta_v_west"},
        }

        # lon_parent = self.parent_grid.ds["lon_rho"]
        # lat_parent = self.parent_grid.ds["lat_rho"]
        parent_grid_ds = self.parent_grid.ds.copy()
        child_grid_ds = self.child_grid.ds.copy()

        # i_eta = np.arange(-0.5, len(lon_parent.eta_rho) + -0.5, 1)
        # i_xi = np.arange(-0.5, len(lon_parent.xi_rho) + -0.5, 1)
        i_eta = np.arange(-0.5, len(parent_grid_ds.eta_rho) + -0.5, 1)
        i_xi = np.arange(-0.5, len(parent_grid_ds.xi_rho) + -0.5, 1)

        # lon_parent = lon_parent.assign_coords(i_eta=("eta_rho", i_eta)).assign_coords(
        #    i_xi=("xi_rho", i_xi)
        # )
        # lat_parent = lat_parent.assign_coords(i_eta=("eta_rho", i_eta)).assign_coords(
        #    i_xi=("xi_rho", i_xi)
        # )
        parent_grid_ds = parent_grid_ds.assign_coords(
            i_eta=("eta_rho", i_eta)
        ).assign_coords(i_xi=("xi_rho", i_xi))

        if self.parent_grid.straddle:
            # lon_parent = xr.where(lon_parent > 180, lon_parent - 360, lon_parent)
            for grid_ds in [parent_grid_ds, child_grid_ds]:
                for lon_dim in ["lon_rho", "lon_u", "lon_v"]:
                    grid_ds[lon_dim] = xr.where(
                        grid_ds[lon_dim] > 180,
                        grid_ds[lon_dim] - 360,
                        grid_ds[lon_dim],
                    )
        else:
            # lon_parent = xr.where(lon_parent < 0, lon_parent + 360, lon_parent)
            for grid_ds in [parent_grid_ds, child_grid_ds]:
                for lon_dim in ["lon_rho", "lon_u", "lon_v"]:
                    grid_ds[lon_dim] = xr.where(
                        grid_ds[lon_dim] < 0, grid_ds[lon_dim] + 360, grid_ds[lon_dim]
                    )

        # add angles at u- and v-points
        child_grid_ds["angle_u"] = interpolate_from_rho_to_u(child_grid_ds["angle"])
        child_grid_ds["angle_v"] = interpolate_from_rho_to_v(child_grid_ds["angle"])

        ds = xr.Dataset()

        for direction in ["south", "east", "north", "west"]:

            if self.boundaries[direction]:
                print(direction)
                for grid_location in ["rho", "u", "v"]:
                    if grid_location == "rho":
                        names = {
                            "latitude": "lat_rho",
                            "longitude": "lon_rho",
                            "mask": "mask_rho",
                        }
                        bdry_coords = bdry_coords_rho
                        suffix = "r"
                    elif grid_location == "u":
                        names = {
                            "latitude": "lat_u",
                            "longitude": "lon_u",
                            "mask": "mask_u",
                            "angle": "angle_u",
                        }
                        bdry_coords = bdry_coords_u
                        suffix = "u"
                    elif grid_location == "v":
                        names = {
                            "latitude": "lat_v",
                            "longitude": "lon_v",
                            "mask": "mask_v",
                            "angle": "angle_v",
                        }
                        bdry_coords = bdry_coords_v
                        suffix = "v"

                    lon_child = child_grid_ds[names["longitude"]].isel(
                        **bdry_coords[direction]
                    )
                    lat_child = child_grid_ds[names["latitude"]].isel(
                        **bdry_coords[direction]
                    )

                    mask_child = child_grid_ds[names["mask"]].isel(
                        **bdry_coords[direction]
                    )
                    ds = ds.assign_coords(
                        {
                            f"{names['latitude']}_{direction}": lat_child,
                            f"{names['longitude']}_{direction}": lon_child,
                        }
                    )
                    i_eta, i_xi = interpolate_indices(
                        parent_grid_ds, lon_child, lat_child, mask_child
                    )

                    print(i_eta)
                    print(i_xi)
                    # i_eta, i_xi = update_indices_if_on_parent_land(
                    #    i_eta, i_xi, grid_location, parent_grid_ds
                    # )

                    if grid_location == "rho":
                        ds[f"{self.child_prefix}_{direction}_{suffix}"] = xr.concat(
                            [i_eta, i_xi], dim="two"
                        ).rename(
                            **rename_rho[direction]
                        )  # dimension name "two" is suboptimal but inherited from matlab scripts
                    else:
                        angle_child = child_grid_ds[names["angle"]].isel(
                            **bdry_coords[direction]
                        )
                        ds[f"{self.child_prefix}_{direction}_{suffix}"] = xr.concat(
                            [i_eta, i_xi, angle_child], dim="three"
                        )  # dimension name "three" is suboptimal but inherited from matlab scripts
                        if grid_location == "u":
                            ds[f"{self.child_prefix}_{direction}_{suffix}"].rename(
                                **rename_u[direction]
                            )
                        elif grid_location == "v":
                            ds[f"{self.child_prefix}_{direction}_{suffix}"].rename(
                                **rename_v[direction]
                            )

        ds = ds.drop_vars(
            [
                "lat_rho",
                "lon_rho",
                "lat_u",
                "lon_u",
                "lat_v",
                "lon_v",
            ]
        )

        # Rename dimensions
        dims_to_rename = {
            dim: f"{self.child_prefix}_{dim}"
            for dim in ds.dims
            if dim not in ["two", "three"]
        }
        ds = ds.rename(dims_to_rename)

        object.__setattr__(self, "ds", ds)

    def plot(self) -> None:
        """
        Plot the parent and child grids in a single figure.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        """

        _plot_nesting(
            self.parent_grid.ds, self.child_grid.ds, self.parent_grid.straddle
        )


def interpolate_indices(parent_grid_ds, lon, lat, mask):
    """
    Interpolate the parent indices to the child grid.

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

    # Crop parent grid to minimial size to avoid aliased interpolated indices
    # latitude_range = [lat.min().values, lat.max().values]
    # longitude_range = [lon.min().values, lon.max().values]
    # cropped_parent_grid_ds = refine_region(
    #    parent_grid_ds, latitude_range, longitude_range
    # )

    lon_parent = parent_grid_ds.lon_rho
    lat_parent = parent_grid_ds.lat_rho
    i_parent = parent_grid_ds.i_eta
    j_parent = parent_grid_ds.i_xi
    # lon_parent = cropped_parent_grid_ds.lon_rho
    # lat_parent = cropped_parent_grid_ds.lat_rho
    # i_parent = cropped_parent_grid_ds.i_eta
    # j_parent = cropped_parent_grid_ds.i_xi

    # Create meshgrid
    i_parent, j_parent = np.meshgrid(i_parent.values, j_parent.values)

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

    ## Check only unmasked i- and j-indices
    i_chk = i[mask == 1]
    j_chk = j[mask == 1]

    ## Check for NaN values
    # if np.sum(np.isnan(i_chk)) > 0 or np.sum(np.isnan(j_chk)) > 0:
    #    raise ValueError(
    #        "Some unmasked points are outside the grid. Please choose either a bigger parent grid or a smaller child grid."
    #    )

    nxp, nyp = lon_parent.shape
    # Check whether indices are close to border of parent grid
    if len(i_chk) > 0:
        if np.min(i_chk) < 0 or np.max(i_chk) > nxp - 2:
            warnings.warn(
                "Some boundary points of the child grid are very close to the boundary of the parent grid."
            )
    if len(j_chk) > 0:
        if np.min(j_chk) < 0 or np.max(j_chk) > nyp - 2:
            warnings.warn(
                "Some boundary points of the child grid are very close to the boundary of the parent grid."
            )

    return i, j


def pad_and_extend_condition(cond, dim):
    """
    Pad the condition array with a single False value at both ends,
    perform logical OR operation with shifted versions of the array,
    and then remove the padding.

    Parameters
    ----------
    cond : xarray.DataArray
        Condition array to be padded and extended.
    dim : str
        Dimension along which to pad and extend the condition.

    Returns
    -------
    xarray.DataArray
        Padded and extended condition array.
    """
    cond_padded = xr.concat(
        [False * cond.isel({dim: 0}), cond, False * cond.isel({dim: -1})], dim=dim
    )
    extended_cond = (
        cond_padded.shift({dim: -1}, fill_value=False)
        | cond_padded
        | cond_padded.shift({dim: 1}, fill_value=False)
    )
    return extended_cond.sel({dim: slice(1, -1)})


def refine_region(grid_ds, latitude_range, longitude_range):
    """
    Refine the region of the grid to match boundary conditions.

    Parameters
    ----------
    grid_ds : xarray.Dataset
        Grid information of parent grid.

    latitude_range : tuple
        A tuple (lat_min, lat_max) specifying the minimum and maximum latitude values of the subdomain.
    longitude_range : tuple
        A tuple (lon_min, lon_max) specifying the minimum and maximum longitude values of the subdomain.

    Returns
    -------
    xr.Dataset
        The subset of the original dataset representing the chosen subdomain.
    """
    lat_min, lat_max = latitude_range
    lon_min, lon_max = longitude_range

    lat_cond = (grid_ds.lat_rho >= lat_min) & (grid_ds.lat_rho <= lat_max)
    lon_cond = (grid_ds.lon_rho >= lon_min) & (grid_ds.lon_rho <= lon_max)

    combined_cond = lat_cond & lon_cond

    extended_cond = pad_and_extend_condition(combined_cond, "eta_rho")
    extended_cond = pad_and_extend_condition(extended_cond, "xi_rho")

    if not extended_cond.any():
        raise ValueError(
            "Some unmasked points are outside the grid. Please choose either a bigger parent grid or a smaller child grid."
        )

    subdomain = grid_ds.where(extended_cond, drop=True)

    return subdomain


def crop_parent(parent_grid_ds, latitude_range, longitude_range):
    """
    Refine the region of the grid to match boundary conditions.

    Parameters
    ----------
    parent_grid_ds : xarray.Dataset
        Grid information of parent grid.

    latitude_range : tuple
        A tuple (lat_min, lat_max) specifying the minimum and maximum latitude values of the subdomain.

    longitude_range : tuple
        A tuple (lon_min, lon_max) specifying the minimum and maximum longitude values of the subdomain.

    Returns
    -------
    xr.Dataset
        The subset of the original dataset representing the chosen subdomain.
    """
    lat_min, lat_max = latitude_range
    lon_min, lon_max = longitude_range

    # Find the indices of the parent grid that match the latitude and longitude ranges
    lat_cond = (parent_grid_ds.lat_rho >= lat_min) & (parent_grid_ds.lat_rho <= lat_max)
    lon_cond = (parent_grid_ds.lon_rho >= lon_min) & (parent_grid_ds.lon_rho <= lon_max)

    # Combined condition
    combined_cond = lat_cond & lon_cond

    # Check if any points satisfy the combined condition
    if not combined_cond.any():
        raise ValueError(
            "No points found within the specified latitude and longitude range."
        )

    print(combined_cond.where(combined_cond, drop=True).eta_rho.values)
    print(combined_cond.where(combined_cond, drop=True).xi_rho.values)
    # Find the minimum and maximum indices in both dimensions
    i0 = combined_cond.where(combined_cond, drop=True).eta_rho.values[0]
    i1 = combined_cond.where(combined_cond, drop=True).eta_rho.values[-1]
    j0 = combined_cond.where(combined_cond, drop=True).xi_rho.values[0]
    j1 = combined_cond.where(combined_cond, drop=True).xi_rho.values[-1]

    print(f"i0: {i0}, i1: {i1}, j0: {j0}, j1: {j1}")
    # Subset the original dataset based on the found indices
    cropped_ds = parent_grid_ds.isel(
        eta_rho=slice(i0 - 1, i1 + 2), xi_rho=slice(j0 - 1, j1 + 2)
    )

    return cropped_ds


# def crop_parent(parent_grid_ds, latitude_range, longitude_range):
#    """
#    Crop parent grid to minimal size.
#
#    Parameters
#    ----------
#    parent_grid_ds : xarray.Dataset
#        Grid information of parent grid.
#
#    latitude_range : tuple
#        A tuple (lat_min, lat_max) specifying the minimum and maximum latitude values of the subdomain.
#
#    longitude_range : tuple
#        A tuple (lon_min, lon_max) specifying the minimum and maximum longitude values of the subdomain.
#
#    Returns
#    -------
#    xr.Dataset
#        The subset of the original dataset representing the chosen subdomain.
#    """
#    lat_min, lat_max = latitude_range
#    lon_min, lon_max = longitude_range
#
#    cropped_ds = parent_grid_ds
#    for _ in range(5):
#        lon = cropped_ds.lon_rho
#        lat = cropped_ds.lat_rho
#        nxs, nys = lon.shape
#
#        parent_lon_min = lon.min(dim="eta_rho")
#        parent_lon_max = lon.max(dim="eta_rho")
#        parent_lat_min = lat.min(dim="xi_rho")
#        parent_lat_max = lat.max(dim="xi_rho")
#        print(parent_lon_min)
#        print(parent_lon_max)
#        print(parent_lat_min)
#        print(parent_lat_max)
#
#        i0 = (
#            np.where(parent_lon_max < lon_min)[0][-1]
#            if np.any(parent_lon_max < lon_min)
#            else 0
#        )
#        i1 = (
#            np.where(parent_lon_min > lon_max)[0][0]
#            if np.any(parent_lon_min > lon_max)
#            else nxs - 1
#        )
#        j0 = (
#            np.where(parent_lat_max < lat_min)[0][-1]
#            if np.any(parent_lat_max < lat_min)
#            else 0
#        )
#        j1 = (
#            np.where(parent_lat_min > lat_max)[0][0]
#            if np.any(parent_lat_min > lat_max)
#            else nys - 1
#        )
#
#        print(f"i0: {i0}, i1: {i1}, j0: {j0}, j1: {j1}")
#        cropped_ds = cropped_ds.isel(
#            eta_rho=slice(i0, i1 + 1), xi_rho=slice(j0, j1 + 1)
#        )
#
#    return cropped_ds


def update_indices_if_on_parent_land(i_eta, i_xi, grid_location, parent_grid_ds):
    """
    Finds points that are in the parent mask but not masked in the child and replaces
    parent indices with nearest neighbor points.

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

    print(i_eta)
    print(i_xi)
    if grid_location == "rho":
        # convert from [-0.5, len(eta_rho) - 1.5] to [0, len(eta_rho) - 1]
        # convert from [-0.5, len(xi_rho) - 1.5] to [0, len(xi_rho) - 1]
        i_eta_rho = i_eta + 0.5
        i_xi_rho = i_xi + 0.5
        print(i_eta_rho)
        print(i_xi_rho)
        mask_rho = parent_grid_ds.mask_rho.copy()
        print(mask_rho.values)
        summed_mask = np.zeros_like(i_eta_rho)

        for i in range(len(i_eta_rho)):
            i_eta_lower = int(np.floor(i_eta_rho[i]))
            i_xi_lower = int(np.floor(i_xi_rho[i]))
            mask = mask_rho.isel(
                eta_rho=slice(i_eta_lower, i_eta_lower + 2),
                xi_rho=slice(i_xi_lower, i_xi_lower + 2),
            )
            print(i_eta_lower)
            print(i_xi_lower)
            print(mask)
            summed_mask[i] = np.sum(mask)

        print(f"summed mask: {summed_mask}")

    elif grid_location in ["u", "v"]:
        # convert from [0, len(eta_rho) - 2] to [0, len(eta_rho) - 2]
        # convert from [-0.5, len(xi_rho) - 1.5] to [0, len(xi_rho) - 1]
        i_eta_u = i_eta
        i_xi_u = i_xi + 0.5

        mask_u = parent_grid_ds.mask_u.copy()
        print(mask_u.values)
        summed_mask_u = np.zeros_like(i_eta_u)

        for i in range(len(i_eta_u)):
            i_eta_lower = int(np.floor(i_eta_u[i]))
            i_xi_lower = int(np.floor(i_xi_u[i]))
            mask = mask_u.isel(
                eta_rho=slice(i_eta_lower, i_eta_lower + 2),
                xi_u=slice(i_xi_lower, i_xi_lower + 2),
            )
            summed_mask_u[i] = np.sum(mask)

        # convert from [-0.5, len(eta_rho) - 1.5] to [0, len(eta_rho) - 1]
        # convert from [0, len(xi_rho) - 2] to [0, len(xi_rho) - 2]
        i_eta_v = i_eta + 0.5
        i_xi_v = i_xi

        mask_v = parent_grid_ds.mask_v.copy()
        print(mask_v.values)
        summed_mask_v = np.zeros_like(i_xi_v)

        for i in range(len(i_eta_v)):
            i_eta_lower = int(np.floor(i_eta_v[i]))
            i_xi_lower = int(np.floor(i_xi_v[i]))
            # print(i_eta_lower)
            # print(i_xi_lower)
            mask = mask_v.isel(
                eta_v=slice(i_eta_lower, i_eta_lower + 2),
                xi_rho=slice(i_xi_lower, i_xi_lower + 2),
            )
            # print(mask)
            summed_mask_v[i] = np.sum(mask)

        summed_mask = summed_mask_u * summed_mask_v
        # print(f"summed mask u: {summed_mask_u}")
        # print(f"summed mask v: {summed_mask_v}")

    # Filter out points where summed_mask is 0
    valid_points = summed_mask != 0
    x_mod = np.arange(len(summed_mask))[valid_points]
    i_eta_mod = i_eta[valid_points]
    i_xi_mod = i_xi[valid_points]

    # Handle indices where summed_mask is 0
    indx = np.where(summed_mask == 0)[0]
    print(f"Fixing {len(indx)} points that are inside the parent mask")
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
