import numpy as np
import xarray as xr
from dataclasses import dataclass, field
from roms_tools.setup.grid import Grid
from roms_tools.setup.utils import (
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
)
from roms_tools.setup.plot import _plot, _section_plot, _profile_plot, _line_plot
import matplotlib.pyplot as plt


@dataclass(frozen=True, kw_only=True)
class VerticalCoordinate:
    """
    Represents vertical coordinate for ROMS.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
    N : int
        The number of vertical levels.
    theta_s : float
        The surface control parameter. Must satisfy 0 < theta_s <= 10.
    theta_b : float
        The bottom control parameter. Must satisfy 0 < theta_b <= 4.
    hc : float
        The critical depth.

    Attributes
    ----------
    ds : xr.Dataset
        Xarray Dataset containing the atmospheric forcing data.
    """

    grid: Grid
    N: int
    theta_s: float
    theta_b: float
    hc: float

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        h = self.grid.ds.h

        cs_r, sigma_r = sigma_stretch(self.theta_s, self.theta_b, self.N, "r")
        zr = compute_depth(h * 0, h, self.hc, cs_r, sigma_r)
        cs_w, sigma_w = sigma_stretch(self.theta_s, self.theta_b, self.N, "w")
        zw = compute_depth(h * 0, h, self.hc, cs_w, sigma_w)

        ds = xr.Dataset()

        ds["theta_s"] = np.float32(self.theta_s)
        ds["theta_s"].attrs["long_name"] = "S-coordinate surface control parameter"
        ds["theta_s"].attrs["units"] = "nondimensional"

        ds["theta_b"] = np.float32(self.theta_b)
        ds["theta_b"].attrs["long_name"] = "S-coordinate bottom control parameter"
        ds["theta_b"].attrs["units"] = "nondimensional"

        ds["Tcline"] = np.float32(self.hc)
        ds["Tcline"].attrs["long_name"] = "S-coordinate surface/bottom layer width"
        ds["Tcline"].attrs["units"] = "m"

        ds["hc"] = np.float32(self.hc)
        ds["hc"].attrs["long_name"] = "S-coordinate parameter critical depth"
        ds["hc"].attrs["units"] = "m"

        ds["sc_r"] = sigma_r.astype(np.float32)
        ds["sc_r"].attrs["long_name"] = "S-coordinate at rho-points"
        ds["sc_r"].attrs["units"] = "nondimensional"

        ds["Cs_r"] = cs_r.astype(np.float32)
        ds["Cs_r"].attrs["long_name"] = "S-coordinate stretching curves at rho-points"
        ds["Cs_r"].attrs["units"] = "nondimensional"

        depth = -zr
        depth.attrs["long_name"] = "Layer depth at rho-points"
        depth.attrs["units"] = "m"
        ds = ds.assign_coords({"layer_depth_rho": depth.astype(np.float32)})

        depth_u = interpolate_from_rho_to_u(depth).astype(np.float32)
        depth_u.attrs["long_name"] = "Layer depth at u-points"
        depth_u.attrs["units"] = "m"
        ds = ds.assign_coords({"layer_depth_u": depth_u})

        depth_v = interpolate_from_rho_to_v(depth).astype(np.float32)
        depth_v.attrs["long_name"] = "Layer depth at v-points"
        depth_v.attrs["units"] = "m"
        ds = ds.assign_coords({"layer_depth_v": depth_v})

        depth = -zw
        depth.attrs["long_name"] = "Interface depth at rho-points"
        depth.attrs["units"] = "m"
        ds = ds.assign_coords({"interface_depth_rho": depth.astype(np.float32)})

        depth_u = interpolate_from_rho_to_u(depth).astype(np.float32)
        depth_u.attrs["long_name"] = "Interface depth at u-points"
        depth_u.attrs["units"] = "m"
        ds = ds.assign_coords({"interface_depth_u": depth_u})

        depth_v = interpolate_from_rho_to_v(depth).astype(np.float32)
        depth_v.attrs["long_name"] = "Interface depth at v-points"
        depth_v.attrs["units"] = "m"
        ds = ds.assign_coords({"interface_depth_v": depth_v})

        ds = ds.drop_vars(["eta_rho", "xi_rho"])

        ds.attrs["Title"] = "ROMS vertical coordinate produced by roms-tools"

        object.__setattr__(self, "ds", ds)

    def plot(
        self,
        varname="layer_depth_rho",
        s=None,
        eta=None,
        xi=None,
    ) -> None:
        """
        Plot the vertical coordinate system for a given eta-, xi-, or s-slice.

        Parameters
        ----------
        varname : str, optional
            The field to plot. Options are "depth_rho", "depth_u", "depth_v".
        s: int, optional
            The s-index to plot. Default is None.
        eta : int, optional
            The eta-index to plot. Default is None.
        xi : int, optional
            The xi-index to plot. Default is None.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        Raises
        ------
        ValueError
            If the specified varname is not one of the valid options.
            If none of s, eta, xi are specified.
        """

        if not any([s is not None, eta is not None, xi is not None]):
            raise ValueError("At least one of s, eta, or xi must be specified.")

        self.ds[varname].load()
        field = self.ds[varname].squeeze()

        if all(dim in field.dims for dim in ["eta_rho", "xi_rho"]):
            interface_depth = self.ds.interface_depth_rho
        elif all(dim in field.dims for dim in ["eta_rho", "xi_u"]):
            interface_depth = self.ds.interface_depth_u
        elif all(dim in field.dims for dim in ["eta_v", "xi_rho"]):
            interface_depth = self.ds.interface_depth_v

        # slice the field as desired
        title = field.long_name
        if s is not None:
            if "s_rho" in field.dims:
                title = title + f", s_rho = {field.s_rho[s].item()}"
                field = field.isel(s_rho=s)
            elif "s_w" in field.dims:
                title = title + f", s_w = {field.s_w[s].item()}"
                field = field.isel(s_w=s)
            else:
                raise ValueError(
                    f"None of the expected dimensions (s_rho, s_w) found in ds[{varname}]."
                )

        if eta is not None:
            if "eta_rho" in field.dims:
                title = title + f", eta_rho = {field.eta_rho[eta].item()}"
                field = field.isel(eta_rho=eta)
                interface_depth = interface_depth.isel(eta_rho=eta)
            elif "eta_v" in field.dims:
                title = title + f", eta_v = {field.eta_v[eta].item()}"
                field = field.isel(eta_v=eta)
                interface_depth = interface_depth.isel(eta_v=eta)
            else:
                raise ValueError(
                    f"None of the expected dimensions (eta_rho, eta_v) found in ds[{varname}]."
                )
        if xi is not None:
            if "xi_rho" in field.dims:
                title = title + f", xi_rho = {field.xi_rho[xi].item()}"
                field = field.isel(xi_rho=xi)
                interface_depth = interface_depth.isel(xi_rho=xi)
            elif "xi_u" in field.dims:
                title = title + f", xi_u = {field.xi_u[xi].item()}"
                field = field.isel(xi_u=xi)
                interface_depth = interface_depth.isel(xi_u=xi)
            else:
                raise ValueError(
                    f"None of the expected dimensions (xi_rho, xi_u) found in ds[{varname}]."
                )

        if eta is None and xi is None:
            vmax = field.max().values
            vmin = field.min().values
            cmap = plt.colormaps.get_cmap("YlGnBu")
            cmap.set_bad(color="gray")
            kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

            _plot(
                self.grid.ds,
                field=field,
                straddle=self.grid.straddle,
                depth_contours=True,
                title=title,
                kwargs=kwargs,
                c="g",
            )
        else:
            if len(field.dims) == 2:
                cmap = plt.colormaps.get_cmap("YlGnBu")
                cmap.set_bad(color="gray")
                kwargs = {"vmax": 0.0, "vmin": 0.0, "cmap": cmap, "add_colorbar": False}

                _section_plot(
                    xr.zeros_like(field),
                    interface_depth=interface_depth,
                    title=title,
                    kwargs=kwargs,
                )
            else:
                if "s_rho" in field.dims or "s_w" in field.dims:
                    _profile_plot(field, title=title)
                else:
                    _line_plot(field, title=title)

    def save(self, filepath: str) -> None:
        """
        Save the vertical coordinate information to a netCDF4 file.

        Parameters
        ----------
        filepath
        """
        self.ds.to_netcdf(filepath)

    @classmethod
    def from_file(cls, filepath: str) -> "VerticalCoordinate":
        """
        Create a VerticalCoordinate instance from an existing file.

        Parameters
        ----------
        filepath : str
            Path to the file containing the vertical coordinate information.

        Returns
        -------
        VerticalCoordinate
            A new instance of VerticalCoordinate populated with data from the file.
        """
        # Load the dataset from the file
        ds = xr.open_dataset(filepath)

        # Create a new VerticalCoordinate instance without calling __init__ and __post_init__
        vertical_coordinate = cls.__new__(cls)

        # Set the dataset for the vertical_corodinate instance
        object.__setattr__(vertical_coordinate, "ds", ds)

        # Manually set the remaining attributes by extracting parameters from dataset
        object.__setattr__(vertical_coordinate, "N", ds.sizes["s_rho"])
        object.__setattr__(vertical_coordinate, "theta_s", ds["theta_s"].values.item())
        object.__setattr__(vertical_coordinate, "theta_b", ds["theta_b"].values.item())
        object.__setattr__(vertical_coordinate, "hc", ds["hc"].values.item())
        object.__setattr__(vertical_coordinate, "grid", None)

        return vertical_coordinate


def compute_cs(sigma, theta_s, theta_b):
    """
    Compute the S-coordinate stretching curves according to Shchepetkin and McWilliams (2009).

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
    """
    Compute sigma and stretching curves based on the type and parameters.

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
    """
    Compute the depth at different sigma levels.

    Parameters
    ----------
    zeta : xr.DataArray
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

    Raises
    ------
    ValueError
        If theta_s or theta_b are less than or equal to zero.
    """

    # Expand dimensions
    sigma = sigma.expand_dims(dim={"eta_rho": h.eta_rho, "xi_rho": h.xi_rho})
    cs = cs.expand_dims(dim={"eta_rho": h.eta_rho, "xi_rho": h.xi_rho})

    s = (hc * sigma + h * cs) / (hc + h)
    z = zeta + (zeta + h) * s

    return z
