import numpy as np
import xarray as xr

from roms_tools.constants import UPPER_BOUND_THETA_B, UPPER_BOUND_THETA_S
from roms_tools.utils import (
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    transpose_dimensions,
)


def compute_cs(sigma, theta_s, theta_b, vstretching=4):
    """Compute the S-coordinate stretching curves for the ROMS vertical stretching
    function ``Vstretching`` (see Rutgers ROMS ``set_scoord.F``). UCLA-ROMS uses
    ``Vstretching = 4`` (Shchepetkin and McWilliams, 2009).

    Parameters
    ----------
    sigma : np.ndarray or float
        The sigma-coordinate values.
    theta_s : float
        The surface control parameter.
    theta_b : float
        The bottom control parameter.
    vstretching : int, optional
        The vertical stretching function, 1 to 5. Defaults to 4.

    Returns
    -------
    C : np.ndarray or float
        The stretching curve values.

    Raises
    ------
    ValueError
        If theta_s or theta_b are not within the valid range.
    """
    if vstretching == 1:  # Song and Haidvogel (1994)
        if theta_s == 0:
            return sigma
        return (1 - theta_b) * np.sinh(theta_s * sigma) / np.sinh(theta_s) + theta_b * (
            0.5 * np.tanh(theta_s * (sigma + 0.5)) / np.tanh(0.5 * theta_s) - 0.5
        )
    if vstretching == 2:  # Shchepetkin (2005)
        if theta_s <= 0:
            return sigma
        C = (1 - np.cosh(theta_s * sigma)) / (np.cosh(theta_s) - 1)
        if theta_b <= 0:
            return C
        Cbot = np.sinh(theta_b * (sigma + 1)) / np.sinh(theta_b) - 1
        weight = (sigma + 1) * (1 - sigma)
        return weight * C + (1 - weight) * Cbot
    if vstretching == 3:  # Geyer, for shallow bottom boundary layers
        Cbot = np.log(np.cosh(3 * (sigma + 1) ** theta_b)) / np.log(np.cosh(3)) - 1
        Csur = -np.log(np.cosh(3 * abs(sigma) ** theta_s)) / np.log(np.cosh(3))
        weight = 0.5 * (1 - np.tanh(3 * (sigma + 0.5)))
        return weight * Cbot + (1 - weight) * Csur

    if vstretching not in [4, 5]:
        raise ValueError("vstretching must be 1 to 5.")
    if not (0 < theta_s <= UPPER_BOUND_THETA_S):
        raise ValueError(f"theta_s must be between 0 and {UPPER_BOUND_THETA_S}.")
    if not (0 < theta_b <= UPPER_BOUND_THETA_B):
        raise ValueError(f"theta_b must be between 0 and {UPPER_BOUND_THETA_B}.")

    C = (1 - np.cosh(theta_s * sigma)) / (np.cosh(theta_s) - 1)
    C = (np.exp(theta_b * C) - 1) / (1 - np.exp(-theta_b))

    return C


def sigma_stretch(theta_s, theta_b, N, type, vstretching=4):
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
    vstretching : int, optional
        The vertical stretching function, 1 to 5. Defaults to 4.

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

    if vstretching == 5:  # quadratic Legendre polynomial (Souza et al., 2015)
        k = k - 0.5 if type == "r" else k
        sigma = -(k * k - 2 * k * N + k + N * N - N) / (N * N - N) - 0.01 * (
            k * k - k * N
        ) / (1 - N)

    cs = compute_cs(sigma, theta_s, theta_b, vstretching)

    return cs, sigma


def compute_depth(zeta, h, hc, cs, sigma, vtransform=2):
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
    vtransform : int, optional
        The ROMS vertical transformation equation, 1 or 2. Defaults to 2, as in
        UCLA-ROMS.

    Returns
    -------
    z : xr.DataArray
        The depth at different sigma levels.
    """
    if vtransform == 1:
        z = hc * sigma + (h - hc) * cs
        z = z + zeta * (1 + z / h)
    else:
        z = (hc * sigma + h * cs) / (hc + h)
        z = zeta + (zeta + h) * z

    z = -transpose_dimensions(z)

    return z


def get_vertical_coordinate(grid_ds, depth_type="layer"):
    """Get the vertical coordinate parameters of a grid at layers or interfaces.

    Parameters
    ----------
    grid_ds : xr.Dataset
        ROMS grid dataset containing the stretching curves (`Cs`), sigma-layers (`sigma`)
        and the critical depth (`hc`) attribute.
    depth_type : str, optional
        `"layer"` (default) or `"interface"`.

    Returns
    -------
    Cs : xr.DataArray
        The stretching curve values.
    sigma : xr.DataArray
        The sigma-coordinate values.
    hc : float
        The critical depth.
    vtransform : int
        The ROMS vertical transformation equation.

    Raises
    ------
    ValueError
        If the grid has no vertical coordinate or `depth_type` is invalid.
    """
    if "hc" not in grid_ds.attrs or "Cs_r" not in grid_ds:
        raise ValueError(
            "The grid has no vertical coordinate. Provide N, theta_s, theta_b and hc "
            "(and vtransform and vstretching for Rutgers ROMS) when creating the Grid, "
            "or call Grid.update_vertical_coordinate."
        )
    if depth_type == "layer":
        Cs, sigma = grid_ds["Cs_r"], grid_ds["sigma_r"]
    elif depth_type == "interface":
        Cs, sigma = grid_ds["Cs_w"], grid_ds["sigma_w"]
    else:
        raise ValueError(
            f"Invalid depth_type: {depth_type}. Choose 'layer' or 'interface'."
        )
    return Cs, sigma, grid_ds.attrs["hc"], grid_ds.attrs.get("Vtransform", 2)


def compute_depth_coordinates(
    grid_ds: "xr.Dataset",
    zeta: xr.DataArray | float = 0,
    depth_type: str = "layer",
    location: str = "rho",
    eta: int | None = None,
    xi: int | None = None,
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

    Cs, sigma, hc, vtransform = get_vertical_coordinate(grid_ds, depth_type)
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

    depth = compute_depth(zeta, h, hc, Cs, sigma, vtransform)

    # Add metadata
    depth.name = f"{depth_type}_depth_{location}"
    depth.attrs.update(
        {"long_name": f"{depth_type} depth at {location}-points", "units": "m"}
    )

    return depth
