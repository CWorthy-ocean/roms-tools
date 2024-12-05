import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr


def _plot(
    grid_ds,
    field=None,
    depth_contours=False,
    straddle=False,
    c="red",
    title="",
    kwargs={},
):
    """Plots a grid or field on a map with optional depth contours.

    This function plots a map using Cartopy projections. It supports plotting a grid, a field, and adding depth contours if desired.
    The projection can be customized, and the grid can be adjusted for domains straddling the 180° meridian.

    Parameters
    ----------
    grid_ds : xarray.Dataset
        The grid dataset containing coordinates (`lon_rho`, `lat_rho`).
    field : xarray.DataArray, optional
        The field to plot. If None, only the grid is plotted.
    depth_contours : bool, optional
        If True, adds depth contours to the plot.
    straddle : bool, optional
        If True, adjusts longitude values to straddle across the 180° meridian.
    c : str, optional
        Color for the boundary plot (default is 'red').
    title : str, optional
        Title of the plot.
    kwargs : dict, optional
        Additional keyword arguments to pass to `pcolormesh` (e.g., colormap or color limits).

    Notes
    -----
    The function raises a `NotImplementedError` if the domain contains the North or South Pole.
    """

    if field is None:
        lon_deg = grid_ds["lon_rho"]
        lat_deg = grid_ds["lat_rho"]

    else:

        field = field.squeeze()
        lon_deg = field.lon
        lat_deg = field.lat

        # check if North or South pole are in domain
        if lat_deg.max().values > 89 or lat_deg.min().values < -89:
            raise NotImplementedError(
                "Plotting is not implemented for the case that the domain contains the North or South pole."
            )

    if straddle:
        lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)

    trans = _get_projection(lon_deg, lat_deg)

    lon_deg = lon_deg.values
    lat_deg = lat_deg.values

    fig, ax = plt.subplots(1, 1, figsize=(13, 7), subplot_kw={"projection": trans})

    if c is not None:
        _add_boundary_to_ax(ax, lon_deg, lat_deg, trans, c, kwargs)

    if field is not None:
        _add_field_to_ax(ax, lon_deg, lat_deg, field, depth_contours, kwargs=kwargs)

    ax.coastlines(
        resolution="50m", linewidth=0.5, color="black"
    )  # add map of coastlines
    ax.gridlines()
    ax.set_title(title)


def _add_boundary_to_ax(ax, lon_deg, lat_deg, trans, c="red", label=""):
    """Plots a grid or field on a map with optional depth contours.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes on which to plot the data (Cartopy axis with projection).

    lon_deg : np.ndarray
        Longitude values in degrees.

    lat_deg : np.ndarray
        Latitude values in degrees.

    trans : cartopy.crs.Projection
        The projection for transforming coordinates.

    c : str, optional
        Color of the grid boundary (default is 'red').
    """
    proj = ccrs.PlateCarree()

    # find corners
    corners = [
        (lon_deg[0, 0], lat_deg[0, 0]),
        (lon_deg[0, -1], lat_deg[0, -1]),
        (lon_deg[-1, -1], lat_deg[-1, -1]),
        (lon_deg[-1, 0], lat_deg[-1, 0]),
    ]

    # transform coordinates to projected space
    transformed_corners = [trans.transform_point(lo, la, proj) for lo, la in corners]
    transformed_lons, transformed_lats = zip(*transformed_corners)

    ax.plot(
        list(transformed_lons) + [transformed_lons[0]],
        list(transformed_lats) + [transformed_lats[0]],
        "o-",
        c=c,
        label=label,
    )


def _add_field_to_ax(
    ax,
    lon_deg,
    lat_deg,
    field,
    depth_contours=False,
    add_colorbar=True,
    kwargs={},
):
    """Plots a grid or field on a map with optional depth contours.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes on which to plot the data (Cartopy axis with projection).

    lon_deg : np.ndarray
        Longitude values in degrees.

    lat_deg : np.ndarray
        Latitude values in degrees.

    field : xarray.DataArray, optional
        Field data to plot (e.g., temperature, salinity). If None, only the grid is plotted.

    depth_contours : bool, optional
        If True, adds depth contours to the plot.

    add_colorbar : bool, optional
        If True, add colobar.

    kwargs : dict, optional
        Additional keyword arguments passed to `pcolormesh` (e.g., colormap, limits).

    Notes
    -----
    - If `depth_contours` is True, the field’s `layer_depth` is used to add contours.
    """
    proj = ccrs.PlateCarree()

    p = ax.pcolormesh(lon_deg, lat_deg, field, transform=proj, **kwargs)
    if hasattr(field, "long_name"):
        label = f"{field.long_name} [{field.units}]"
    elif hasattr(field, "Long_name"):
        # this is the case for matlab generated grids
        label = f"{field.Long_name} [{field.units}]"
    else:
        label = ""
    if add_colorbar:
        plt.colorbar(p, label=label)

    if depth_contours:
        cs = ax.contour(lon_deg, lat_deg, field.layer_depth, transform=proj, colors="k")
        ax.clabel(cs, inline=True, fontsize=10)


def _get_projection(lon, lat):

    return ccrs.NearsidePerspective(
        central_longitude=lon.mean().values, central_latitude=lat.mean().values
    )


def _section_plot(field, interface_depth=None, title="", kwargs={}, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    dims_to_check = ["eta_rho", "eta_u", "eta_v", "xi_rho", "xi_u", "xi_v"]
    try:
        xdim = next(
            dim
            for dim in field.dims
            if any(dim.startswith(prefix) for prefix in dims_to_check)
        )
    except StopIteration:
        raise ValueError(
            "None of the dimensions found in field.dims starts with (eta_rho, eta_u, eta_v, xi_rho, xi_u, xi_v)"
        )

    depths_to_check = [
        "layer_depth",
        "interface_depth",
    ]
    try:
        depth_label = next(
            depth_label
            for depth_label in field.coords
            if any(depth_label.startswith(prefix) for prefix in depths_to_check)
        )
    except StopIteration:
        raise ValueError(
            "None of the coordinates found in field.coords starts with (layer_depth_rho, layer_depth_u, layer_depth_v, interface_depth_rho, interface_depth_u, interface_depth_v)"
        )

    more_kwargs = {"x": xdim, "y": depth_label, "yincrease": False}
    field.plot(**kwargs, **more_kwargs, ax=ax)

    if interface_depth is not None:
        layer_key = "s_rho" if "s_rho" in interface_depth else "s_w"

        for i in range(len(interface_depth[layer_key])):
            ax.plot(
                interface_depth[xdim], interface_depth.isel({layer_key: i}), color="k"
            )

    ax.set_title(title)


def _profile_plot(field, title="", ax=None):
    """Plots a profile of the given field against depth.

    Parameters
    ----------
    field : xarray.DataArray
        Data to plot.
    title : str, optional
        Title of the plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.

    Raises
    ------
    ValueError
        If no expected depth coordinate is found in the field.
    """

    depths_to_check = [
        "layer_depth",
        "interface_depth",
    ]
    try:
        depth_label = next(
            depth_label
            for depth_label in field.coords
            if any(depth_label.startswith(prefix) for prefix in depths_to_check)
        )
    except StopIteration:
        raise ValueError(
            "None of the expected coordinates (layer_depth_rho, layer_depth_u, layer_depth_v, interface_depth_rho, interface_depth_u, interface_depth_v) found in field.coords"
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 7))
    kwargs = {"y": depth_label, "yincrease": False}
    field.plot(**kwargs)
    ax.set_title(title)
    ax.grid()


def _line_plot(field, title="", ax=None):
    """Plots a line graph of the given field.

    Parameters
    ----------
    field : xarray.DataArray
        Data to plot.
    title : str, optional
        Title of the plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    None
        Modifies the plot in-place.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    field.plot(ax=ax)
    ax.set_title(title)
    ax.grid()


def _plot_nesting(parent_grid_ds, child_grid_ds, parent_straddle):

    parent_lon_deg = parent_grid_ds["lon_rho"]
    parent_lat_deg = parent_grid_ds["lat_rho"]

    child_lon_deg = child_grid_ds["lon_rho"]
    child_lat_deg = child_grid_ds["lat_rho"]

    if parent_straddle:
        parent_lon_deg = xr.where(
            parent_lon_deg > 180, parent_lon_deg - 360, parent_lon_deg
        )
        child_lon_deg = xr.where(
            child_lon_deg > 180, child_lon_deg - 360, child_lon_deg
        )

    trans = _get_projection(parent_lon_deg, parent_lat_deg)

    parent_lon_deg = parent_lon_deg.values
    parent_lat_deg = parent_lat_deg.values
    child_lon_deg = child_lon_deg.values
    child_lat_deg = child_lat_deg.values

    fig, ax = plt.subplots(1, 1, figsize=(13, 7), subplot_kw={"projection": trans})

    _add_boundary_to_ax(
        ax, parent_lon_deg, parent_lat_deg, trans, c="r", label="parent grid"
    )

    _add_boundary_to_ax(
        ax, child_lon_deg, child_lat_deg, trans, c="g", label="child grid"
    )

    vmax = 3
    vmin = 0
    cmap = plt.colormaps.get_cmap("Blues")
    kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

    _add_field_to_ax(
        ax,
        parent_lon_deg,
        parent_lat_deg,
        parent_grid_ds.mask_rho,
        add_colorbar=False,
        kwargs=kwargs,
    )

    ax.coastlines(
        resolution="50m", linewidth=0.5, color="black"
    )  # add map of coastlines
    ax.gridlines()
    ax.legend()
