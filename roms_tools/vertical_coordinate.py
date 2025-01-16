import numpy as np
import xarray as xr
from roms_tools.setup.utils import transpose_dimensions


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
