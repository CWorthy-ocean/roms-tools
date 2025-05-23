import numpy as np
import xarray as xr

from roms_tools.utils import (
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    transpose_dimensions,
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


def compute_depth_coordinates(
    grid_ds: "xr.Dataset",
    zeta: xr.DataArray | float = 0,
    depth_type: str = "layer",
    location: str = "rho",
    eta: int = None,
    xi: int = None,
) -> "xr.DataArray":
    """Compute vertical depth coordinates for a given ROMS grid location.

    This function calculates depth coordinates (layer or interface) at a specified grid
    location (`rho`, `u`, or `v`). It optimizes computations by slicing the bathymetry (`h`)
    and free-surface elevation (`zeta`) before performing vertical coordinate calculations,
    reducing memory usage and improving efficiency.

    Parameters
    ----------
    grid_ds : xr.Dataset
        ROMS grid dataset containing bathymetry (`h`), stretching curves (`Cs`),
        and sigma-layer parameters (`sigma`). The dataset's attributes should include
        the critical depth parameter (`hc`).

    zeta : xr.DataArray or float, optional
        Free-surface elevation. If set to `0` (default), the static sea level is assumed.

    depth_type : str, optional
        Type of depth coordinates to compute:
        - `"layer"` (default): Depth at the center of vertical layers.
        - `"interface"`: Depth at layer interfaces.

    location : str, optional
        Grid location for depth computation:
        - `"rho"` (default): Depth at cell centers (rho points).
        - `"u"`: Depth at eastward velocity points (u points).
        - `"v"`: Depth at northward velocity points (v points).

    eta : int, optional
        Meridional (north-south) index to extract a zonal slice. If not provided,
        all meridional indices are included.

    xi : int, optional
        Zonal (east-west) index to extract a meridional slice. If not provided,
        all zonal indices are included.

    Returns
    -------
    xr.DataArray
        Computed depth coordinates. The shape of the output depends on the given indices:
        - Full 3D (or 4D if `zeta` includes time) depth field if no indices are specified.
        - 2D (or 3D if `zeta` includes time) slice if either `eta` or `xi` are specified.
        - 1D (or 2D if `zeta` includes time) slice if both `eta` and `xi` are specified.

    Notes
    -----
    - The function first interpolates `h` and `zeta` to the specified grid location (`rho`, `u`, or `v`).
    - Spatial slicing (`eta`, `xi`) is performed before depth computation to optimize efficiency.
    - Depth calculations rely on the ROMS vertical stretching curves (`Cs`) and sigma-layers.
    """
    # Validate location
    valid_locations = {"rho", "u", "v"}
    if location not in valid_locations:
        raise ValueError(
            f"Invalid location: {location}. Must be one of {valid_locations}."
        )

    # Select the appropriate depth computation parameters
    if depth_type == "layer":
        Cs = grid_ds["Cs_r"]
        sigma = grid_ds["sigma_r"]
    elif depth_type == "interface":
        Cs = grid_ds["Cs_w"]
        sigma = grid_ds["sigma_w"]
    else:
        raise ValueError(
            f"Invalid depth_type: {depth_type}. Choose 'layer' or 'interface'."
        )

    h = grid_ds["h"]

    # Interpolate h and zeta to the specified location
    if location == "u":
        h = interpolate_from_rho_to_u(h)
        if isinstance(zeta, xr.DataArray):
            zeta = interpolate_from_rho_to_u(zeta)
    elif location == "v":
        h = interpolate_from_rho_to_v(h)
        if isinstance(zeta, xr.DataArray):
            zeta = interpolate_from_rho_to_v(zeta)

    # Slice spatially based on indices
    if eta is not None:
        if location == "v":
            h = h.isel(eta_v=eta)
            if isinstance(zeta, xr.DataArray):
                zeta = zeta.isel(eta_v=eta)
        else:  # Applies to "rho" or "u"
            h = h.isel(eta_rho=eta)
            if isinstance(zeta, xr.DataArray):
                zeta = zeta.isel(eta_rho=eta)

    if xi is not None:
        if location == "u":
            h = h.isel(xi_u=xi)
            if isinstance(zeta, xr.DataArray):
                zeta = zeta.isel(xi_u=xi)
        else:  # Applies to "rho" or "v"
            h = h.isel(xi_rho=xi)
            if isinstance(zeta, xr.DataArray):
                zeta = zeta.isel(xi_rho=xi)

    depth = compute_depth(zeta, h, grid_ds.attrs["hc"], Cs, sigma)

    # Add metadata
    depth.name = f"{depth_type}_depth_{location}"
    depth.attrs.update(
        {"long_name": f"{depth_type} depth at {location}-points", "units": "m"}
    )

    return depth
