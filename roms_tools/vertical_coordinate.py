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


def add_depth_coordinates_to_dataset(
    ds: "xr.Dataset",
    grid_ds: "xr.Dataset",
    depth_type: str,
    locations: list[str] = ["rho", "u", "v"],
) -> None:
    """Add computed vertical depth coordinates to a dataset for specified grid
    locations.

    This function computes vertical depth coordinates (layer or interface) and updates
    the provided dataset with these coordinates for the specified grid locations. If
    the dataset already contains depth coordinates for all specified locations, the function
    does nothing.

    Parameters
    ----------
    ds : xr.Dataset
        Target dataset to which computed depth coordinates will be added.
        If the `zeta` variable is not present, static vertical coordinates are used.

    grid_ds : xr.Dataset
        Grid dataset containing bathymetry, stretching curves, and parameters.

    depth_type : str
        Type of depth coordinates to compute. Options are:
        - "layer": Layer depth coordinates.
        - "interface": Interface depth coordinates.

    locations : list of str, optional
        List of locations for which to compute depth coordinates. Default is ["rho", "u", "v"].
    """
    required_vars = [f"{depth_type}_depth_{loc}" for loc in locations]

    if all(var in ds for var in required_vars):
        return  # Depth coordinates already exist

    # Compute or interpolate depth coordinates
    if f"{depth_type}_depth_rho" in ds:
        depth_rho = ds[f"{depth_type}_depth_rho"]
    else:
        h = grid_ds["h"]
        zeta = ds.get("zeta", 0)
        if depth_type == "layer":
            Cs = grid_ds["Cs_r"]
            sigma = grid_ds["sigma_r"]
        elif depth_type == "interface":
            Cs = grid_ds["Cs_w"]
            sigma = grid_ds["sigma_w"]
        depth_rho = compute_depth(zeta, h, grid_ds.attrs["hc"], Cs, sigma)
        depth_rho.attrs.update(
            {"long_name": f"{depth_type} depth at rho-points", "units": "m"}
        )
        ds[f"{depth_type}_depth_rho"] = depth_rho

    # Interpolate depth to other locations
    for loc in locations:
        if loc == "rho":
            continue

        interp_func = (
            interpolate_from_rho_to_u if loc == "u" else interpolate_from_rho_to_v
        )
        depth_loc = interp_func(depth_rho)
        depth_loc.attrs.update(
            {"long_name": f"{depth_type} depth at {loc}-points", "units": "m"}
        )
        ds[f"{depth_type}_depth_{loc}"] = depth_loc


def compute_depth_coordinates(
    ds: "xr.Dataset",
    grid_ds: "xr.Dataset",
    depth_type: str,
    location: str,
    s: int = None,
    eta: int = None,
    xi: int = None,
) -> "xr.DataArray":
    """Compute vertical depth coordinates efficiently for a specified grid location and
    optional indices.

    This function calculates vertical depth coordinates (layer or interface) for a given grid
    location (`rho`, `u`, or `v`). It performs spatial slicing (meridional or zonal) on the
    bathymetry and free-surface elevation (`zeta`) before computing depth coordinates. This
    approach minimizes computational overhead by reducing the dataset size before performing
    vertical coordinate calculations.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing optional `zeta` (free-surface elevation). If `zeta` is not present,
        static vertical coordinates are computed.

    grid_ds : xr.Dataset
        Grid dataset containing bathymetry (`h`), stretching curves (`Cs`), and sigma-layer
        parameters (`sigma`). The attributes of this dataset should include the critical depth (`hc`).

    depth_type : str
        Type of depth coordinates to compute:
        - `"layer"`: Depth at the center of layers.
        - `"interface"`: Depth at layer interfaces.

    location : str
        Grid location for the computation. Options are:
        - `"rho"`: Depth at rho points (cell centers).
        - `"u"`: Depth at u points (eastward velocity points).
        - `"v"`: Depth at v points (northward velocity points).

    s : int, optional
        Vertical index to extract a single layer or interface slice. If not provided, all vertical
        layers are included.

    eta : int, optional
        Meridional (north-south) index to extract a slice. If not provided, all meridional indices
        are included.

    xi : int, optional
        Zonal (east-west) index to extract a slice. If not provided, all zonal indices are included.

    Returns
    -------
    xr.DataArray
        A DataArray containing the computed depth coordinates. If no indices are specified, the
        array will have the full dimensionality of the depth coordinates. The dimensions of the
        output depend on the provided indices:
        - Full 3D (or 4D if `zeta` includes time) depth coordinates if no indices are provided.
        - Reduced dimensionality for specified slices (e.g., 2D for a single vertical slice).

    Notes
    -----
    - To ensure computational efficiency, spatial slicing (based on `eta` and `xi`) is performed
      before computing depth coordinates. This reduces memory usage and processing time.
    - Depth coordinates are interpolated to the specified grid location (`rho`, `u`, or `v`) if
      necessary.
    - If depth coordinates for the specified location and configuration already exist in `ds`,
      they are not recomputed.
    """

    h = grid_ds["h"]
    zeta = ds.get("zeta", None)

    # Interpolate h and zeta to the specified location
    if location == "u":
        h = interpolate_from_rho_to_u(h)
        if zeta is not None:
            zeta = interpolate_from_rho_to_u(zeta)
    elif location == "v":
        h = interpolate_from_rho_to_v(h)
        if zeta is not None:
            zeta = interpolate_from_rho_to_v(zeta)

    # Slice spatially based on the location's specific dimensions
    if eta is not None:
        if location == "v":
            h = h.isel(eta_v=eta)
            if zeta is not None:
                zeta = zeta.isel(eta_v=eta)
        else:  # Default to "rho" or "u"
            h = h.isel(eta_rho=eta)
            if zeta is not None:
                zeta = zeta.isel(eta_rho=eta)
    if xi is not None:
        if location == "u":
            h = h.isel(xi_u=xi)
            if zeta is not None:
                zeta = zeta.isel(xi_u=xi)
        else:  # Default to "rho" or "v"
            h = h.isel(xi_rho=xi)
            if zeta is not None:
                zeta = zeta.isel(xi_rho=xi)

    # Compute depth
    if depth_type == "layer":
        Cs = grid_ds["Cs_r"]
        sigma = grid_ds["sigma_r"]
    elif depth_type == "interface":
        Cs = grid_ds["Cs_w"]
        sigma = grid_ds["sigma_w"]
    depth = compute_depth(zeta, h, grid_ds.attrs["hc"], Cs, sigma)

    # Slice vertically
    if s is not None:
        vertical_dim = "s_rho" if "s_rho" in depth.dims else "s_w"
        depth = depth.isel({vertical_dim: s})

    return depth
