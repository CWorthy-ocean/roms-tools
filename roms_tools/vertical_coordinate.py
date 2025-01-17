import numpy as np
import xarray as xr
from roms_tools.utils import (
    transpose_dimensions,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
)


def compute_cs(sigma, theta_s, theta_b):
    """Compute the S-coordinate stretching curves according to Shchepetkin and
    McWilliams (2009).

    Parameters
    ----------
    sigma : np.ndarray or float
        The sigma-coordinate values.
    theta_s : float
        The surface control parameter.
    theta_b : float
        The bottom control parameter.

    Returns
    -------
    C : np.ndarray or float
        The stretching curve values.

    Raises
    ------
    ValueError
        If theta_s or theta_b are not within the valid range.
    """
    if not (0 < theta_s <= 10):
        raise ValueError("theta_s must be between 0 and 10.")
    if not (0 < theta_b <= 4):
        raise ValueError("theta_b must be between 0 and 4.")

    C = (1 - np.cosh(theta_s * sigma)) / (np.cosh(theta_s) - 1)
    C = (np.exp(theta_b * C) - 1) / (1 - np.exp(-theta_b))

    return C


def sigma_stretch(theta_s, theta_b, N, type):
    """Compute sigma and stretching curves based on the type and parameters.

    Parameters
    ----------
    theta_s : float
        The surface control parameter.
    theta_b : float
        The bottom control parameter.
    N : int
        The number of vertical levels.
    type : str
        The type of sigma ('w' for vertical velocity points, 'r' for rho-points).

    Returns
    -------
    cs : xr.DataArray
        The stretching curve values.
    sigma : xr.DataArray
        The sigma-coordinate values.

    Raises
    ------
    ValueError
        If the type is not 'w' or 'r'.
    """
    if type == "w":
        k = xr.DataArray(np.arange(N + 1), dims="s_w")
        sigma = (k - N) / N
    elif type == "r":
        k = xr.DataArray(np.arange(1, N + 1), dims="s_rho")
        sigma = (k - N - 0.5) / N
    else:
        raise ValueError(
            "Type must be either 'w' for vertical velocity points or 'r' for rho-points."
        )

    cs = compute_cs(sigma, theta_s, theta_b)

    return cs, sigma


def compute_depth(zeta, h, hc, cs, sigma):
    """Compute the depth at different sigma levels.

    Parameters
    ----------
    zeta : xr.DataArray or scalar
        The sea surface height.
    h : xr.DataArray
        The depth of the sea bottom.
    hc : float
        The critical depth.
    cs : xr.DataArray
        The stretching curve values.
    sigma : xr.DataArray
        The sigma-coordinate values.

    Returns
    -------
    z : xr.DataArray
        The depth at different sigma levels.
    """

    z = (hc * sigma + h * cs) / (hc + h)
    z = zeta + (zeta + h) * z

    z = -transpose_dimensions(z)

    return z


def retrieve_depth_coordinates(
    ds: "xr.Dataset",
    grid_ds: "xr.Dataset",
    type: str,
    additional_locations: list[str] = ["u", "v"],
) -> None:
    """Compute and update vertical depth coordinates for specified locations.

    This function calculates the vertical depth coordinates (layer or interface)
    for rho points and optionally for additional locations (e.g., u and v points)
    in the provided dataset. The computed depth coordinates are added as variables
    in the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The target dataset where the depth coordinates will be added. This dataset
        should contain the sea surface height (`zeta`) if available; otherwise,
        a default value of 0 will be used for calculations.

    grid_ds : xarray.Dataset
        The grid dataset containing essential information for depth calculations,
        such as bathymetry (`h`), stretching curves (`Cs_r` and `Cs_w`), and
        stretching parameters (`sigma_r` and `sigma_w`).

    type : str
        The type of depth coordinate to compute. Valid options are:
        - "layer": Compute layer depth coordinates.
        - "interface": Compute interface depth coordinates.

    additional_locations : list of str, optional
        Additional locations to compute depth coordinates for. Default is ["u", "v"].
        Valid options include:
        - "u": Compute depth coordinates for u points.
        - "v": Compute depth coordinates for v points.

    Updates
    -------
    ds : xarray.Dataset
        The dataset is updated with the following depth coordinate variables:
        - f"{type}_depth_rho": Depth coordinates at rho points.
        - f"{type}_depth_u": Depth coordinates at u points (if "u" is specified).
        - f"{type}_depth_v": Depth coordinates at v points (if "v" is specified).

    Notes
    -----
    - If depth coordinates for all specified locations are already present in
      the dataset, the function does nothing.
    - If only the rho-point depth coordinates are available, the function
      interpolates to calculate the depth coordinates for the additional locations.
    - If depth coordinates are not present, they are computed using the
      `compute_depth` function based on grid information and optional free-surface
      elevation (`zeta`).

    Examples
    --------
    >>> retrieve_depth_coordinates(
    ...     ds, grid_ds, type="layer", additional_locations=["u", "v"]
    ... )
    >>> print(ds["layer_depth_rho"])
    >>> print(ds["layer_depth_u"])
    >>> print(ds["layer_depth_v"])
    """

    layer_vars = []
    for location in ["rho"] + additional_locations:
        layer_vars.append(f"{type}_depth_{location}")

    if all(layer_var in ds for layer_var in layer_vars):
        # Vertical coordinate data already exists
        pass

    elif f"{type}_depth_rho" in ds:
        depth = ds[f"{type}_depth_rho"]

        if "u" in additional_locations or "v" in additional_locations:
            # interpolation
            if "u" in additional_locations:
                depth_u = interpolate_from_rho_to_u(depth)
                depth_u.attrs["long_name"] = f"{type} depth at u-points"
                depth_u.attrs["units"] = "m"
                ds[f"{type}_depth_u"] = depth_u
            if "v" in additional_locations:
                depth_v = interpolate_from_rho_to_v(depth)
                depth_v.attrs["long_name"] = f"{type} depth at v-points"
                depth_v.attrs["units"] = "m"
                ds[f"{type}_depth_v"] = depth_v
    else:
        h = grid_ds["h"]
        if "zeta" in ds.data_vars:
            eta = ds["zeta"]
        else:
            eta = 0
        if type == "layer":
            depth = compute_depth(
                eta, h, grid_ds.attrs["hc"], grid_ds["Cs_r"], grid_ds["sigma_r"]
            )
        else:
            depth = compute_depth(
                eta, h, grid_ds.attrs["hc"], grid_ds["Cs_w"], grid_ds["sigma_w"]
            )

        depth.attrs["long_name"] = f"{type} depth at rho-points"
        depth.attrs["units"] = "m"
        ds[f"{type}_depth_rho"] = depth

        if "u" in additional_locations or "v" in additional_locations:
            # interpolation
            depth_u = interpolate_from_rho_to_u(depth)
            depth_u.attrs["long_name"] = f"{type} depth at u-points"
            depth_u.attrs["units"] = "m"
            depth_v = interpolate_from_rho_to_v(depth)
            depth_v.attrs["long_name"] = f"{type} depth at v-points"
            depth_v.attrs["units"] = "m"
            ds[f"{type}_depth_u"] = depth_u
            ds[f"{type}_depth_v"] = depth_v
