import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _plot(
    field,
    depth_contours=False,
    c="red",
    title="",
    with_dim_names=False,
    plot_data=True,
    add_colorbar=True,
    kwargs={},
    ax=None,
):
    """Plots a grid or field on a map with optional depth contours.

    This function plots a map using Cartopy projections. It supports plotting a grid, a field, and adding depth contours if desired.

    Parameters
    ----------
    field : xarray.DataArray
        The field to plot.
    depth_contours : bool, optional
        If True, adds depth contours to the plot.
    c : str, optional
        Color for the boundary plot (default is 'red').
    title : str, optional
        Title of the plot.
    plot_data : bool, optional
        If True, plots the provided field data on the map. If False, only the grid
        boundaries and optional depth contours are plotted. Default is True.
    add_colorbar : bool, optional
        If True, add colobar.
    kwargs : dict, optional
        Additional keyword arguments to pass to `pcolormesh` (e.g., colormap or color limits).
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw the plot on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.figure.Figure, optional
        The generated figure with the plotted data, only returned if `ax` is None.

    Raises
    ------
    NotImplementedError
        If the domain contains the North or South Pole.
    """

    field = field.squeeze()
    lon_deg = field.lon
    lat_deg = field.lat

    # check if North or South pole are in domain
    if lat_deg.max().values > 89 or lat_deg.min().values < -89:
        raise NotImplementedError(
            "Plotting is not implemented for the case that the domain contains the North or South pole."
        )

    trans = _get_projection(lon_deg, lat_deg)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(13, 7), subplot_kw={"projection": trans})

    lon_deg = lon_deg.values
    lat_deg = lat_deg.values

    if c is not None:
        _add_boundary_to_ax(
            ax, lon_deg, lat_deg, trans, c, with_dim_names=with_dim_names
        )

    if plot_data:
        _add_field_to_ax(
            ax,
            lon_deg,
            lat_deg,
            field,
            depth_contours,
            add_colorbar=add_colorbar,
            kwargs=kwargs,
        )

    ax.coastlines(
        resolution="50m", linewidth=0.5, color="black"
    )  # add map of coastlines

    # Add gridlines with labels for latitude and longitude
    gridlines = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.7, linestyle="--"
    )
    gridlines.top_labels = False  # Hide top labels
    gridlines.right_labels = False  # Hide right labels
    gridlines.xlabel_style = {
        "size": 10,
        "color": "black",
    }  # Customize longitude label style
    gridlines.ylabel_style = {
        "size": 10,
        "color": "black",
    }  # Customize latitude label style

    ax.set_title(title)

    # Only return fig if it was created inside the function (i.e., ax was not provided)
    if ax is None:
        return fig


def _plot_nesting(parent_grid_ds, child_grid_ds, parent_straddle, with_dim_names=False):
    """Plots nested parent and child grids with boundary overlays and grid masking.

    Parameters
    ----------
    parent_grid_ds : xarray.Dataset
        The parent grid dataset containing `lon_rho`, `lat_rho`, and `mask_rho` variables.
    child_grid_ds : xarray.Dataset
        The child grid dataset containing `lon_rho` and `lat_rho` variables.
    parent_straddle : bool
        Whether the parent grid straddles the 180-degree meridian. If True, longitudes
        greater than 180° are wrapped to the -180° to 180° range.
    with_dim_names : bool, optional
        Whether to include dimension names in the plotted grid boundaries. Defaults to False.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure displaying the parent and child grid boundaries, mask,
        and additional map features.
    """

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
        ax,
        parent_lon_deg,
        parent_lat_deg,
        trans,
        c="r",
        label="parent grid",
        with_dim_names=with_dim_names,
    )

    _add_boundary_to_ax(
        ax,
        child_lon_deg,
        child_lat_deg,
        trans,
        c="g",
        label="child grid",
        with_dim_names=with_dim_names,
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

    # Add gridlines with labels for latitude and longitude
    gridlines = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.7, linestyle="--"
    )
    gridlines.top_labels = False  # Hide top labels
    gridlines.right_labels = False  # Hide right labels
    gridlines.xlabel_style = {
        "size": 10,
        "color": "black",
    }  # Customize longitude label style
    gridlines.ylabel_style = {
        "size": 10,
        "color": "black",
    }  # Customize latitude label style

    ax.legend(loc="best")

    return fig


def _section_plot(field, interface_depth=None, title="", kwargs={}, ax=None):
    """Plots a vertical section of a field with optional interface depths.

    Parameters
    ----------
    field : xarray.DataArray
        The field to plot, typically representing a vertical section of ocean data.
    interface_depth : xarray.DataArray, optional
        Interface depth values to overlay on the plot, useful for visualizing vertical layers.
        Defaults to None.
    title : str, optional
        Title of the plot. Defaults to an empty string.
    kwargs : dict, optional
        Additional keyword arguments to pass to `xarray.plot`. Defaults to an empty dictionary.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw the plot on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.figure.Figure, optional
        The generated figure with the plotted section, only returned if `ax` is None.

    Raises
    ------
    ValueError
        If no dimension in `field.dims` starts with any of the recognized horizontal dimension
        prefixes (`eta_rho`, `eta_v`, `xi_rho`, `xi_u`, `lat`, `lon`).
    ValueError
        If no coordinate in `field.coords` starts with either `layer` or `interface`.

    Notes
    -----
    - NaN values at the horizontal ends are dropped before plotting.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    dims_to_check = ["eta_rho", "eta_v", "xi_rho", "xi_u", "lat", "lon"]
    try:
        xdim = next(
            dim
            for dim in field.dims
            if any(dim.startswith(prefix) for prefix in dims_to_check)
        )
    except StopIteration:
        raise ValueError(
            "None of the dimensions found in field.dims starts with (eta_rho, eta_v, xi_rho, xi_u, lat, lon)"
        )

    depths_to_check = [
        "layer",
        "interface",
    ]
    try:
        depth_label = next(
            depth_label
            for depth_label in field.coords
            if any(depth_label.startswith(prefix) for prefix in depths_to_check)
        )
    except StopIteration:
        raise ValueError(
            "None of the coordinates found in field.coords starts with (layer_depth, interface_depth)"
        )

    # Handle NaNs on either horizontal end
    field = field.where(~field[depth_label].isnull(), drop=True)

    more_kwargs = {"x": xdim, "y": depth_label, "yincrease": False}

    field.plot(**kwargs, **more_kwargs, ax=ax)

    if interface_depth is not None:
        layer_key = "s_rho" if "s_rho" in interface_depth.dims else "s_w"

        for i in range(len(interface_depth[layer_key])):
            ax.plot(
                interface_depth[xdim], interface_depth.isel({layer_key: i}), color="k"
            )

    ax.set_title(title)
    ax.set_ylabel("Depth [m]")

    if xdim == "lon":
        xlabel = "Longitude [°E]"
    elif xdim == "lat":
        xlabel = "Latitude [°N]"
    else:
        xlabel = xdim
    ax.set_xlabel(xlabel)

    if ax is None:
        return fig


def _profile_plot(field, title="", ax=None):
    """Plots a vertical profile of the given field against depth.

    This function generates a profile plot by plotting the field values against
    depth. It automatically detects the appropriate depth coordinate and
    reverses the y-axis to follow the convention of increasing depth downward.

    Parameters
    ----------
    field : xarray.DataArray
        The field to plot, typically representing vertical profile data.
    title : str, optional
        Title of the plot. Defaults to an empty string.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw the plot on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure with the plotted profile, only returned if `ax` is None.

    Raises
    ------
    ValueError
        If no coordinate in `field.coords` starts with either `layer_depth` or `interface_depth`.

    Notes
    -----
    - The y-axis is inverted to ensure that depth increases downward.
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
            "None of the coordinates found in field.coords starts with (layer_depth, interface_depth)"
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 7))
    kwargs = {"y": depth_label, "yincrease": False}
    field.plot(**kwargs, linewidth=2)
    ax.set_title(title)
    ax.set_ylabel("Depth [m]")
    ax.grid()

    if ax is None:
        return fig


def _line_plot(field, title="", ax=None):
    """Plots a line graph of the given field with grey vertical bars indicating NaN
    regions.

    Parameters
    ----------
    field : xarray.DataArray
        The field to plot, typically a 1D or 2D field with one spatial dimension.
    title : str, optional
        Title of the plot. Defaults to an empty string.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw the plot on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure with the plotted data and highlighted NaN regions, only
        returned if `ax` is None.

    Raises
    ------
    ValueError
        If none of the dimensions in `field.dims` starts with one of the expected
        prefixes: `eta_rho`, `eta_v`, `xi_rho`, `xi_u`, `lat`, or `lon`.

    Notes
    -----
    - NaN regions are identified and marked using `axvspan` with a grey shade.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    field.plot(ax=ax, linewidth=2)

    # Loop through the NaNs in the field and add grey vertical bars
    dims_to_check = ["eta_rho", "eta_v", "xi_rho", "xi_u", "lat", "lon"]
    try:
        xdim = next(
            dim
            for dim in field.dims
            if any(dim.startswith(prefix) for prefix in dims_to_check)
        )
    except StopIteration:
        raise ValueError(
            "None of the dimensions found in field.dims starts with (eta_rho, eta_v, xi_rho, xi_u, lat, lon)"
        )

    nan_mask = np.isnan(field.values)
    nan_indices = np.where(nan_mask)[0]

    if len(nan_indices) > 0:
        # Add grey vertical bars for each NaN region
        start_idx = nan_indices[0]
        for idx in range(1, len(nan_indices)):
            if nan_indices[idx] != nan_indices[idx - 1] + 1:
                ax.axvspan(
                    field[xdim][start_idx],
                    field[xdim][nan_indices[idx - 1] + 1],
                    color="gray",
                    alpha=0.3,
                )
                start_idx = nan_indices[idx]
        # Add the last region of NaNs, making sure we don't go out of bounds
        ax.axvspan(
            field[xdim][start_idx],
            field[xdim][nan_indices[-1]],
            color="gray",
            alpha=0.3,
        )

    # Set plot title and grid
    ax.set_title(title)
    ax.grid()
    ax.set_xlim([field[xdim][0], field[xdim][-1]])

    if xdim == "lon":
        xlabel = "Longitude [°E]"
    elif xdim == "lat":
        xlabel = "Latitude [°N]"
    else:
        xlabel = xdim
    ax.set_xlabel(xlabel)

    if ax is None:
        return fig


def _add_boundary_to_ax(
    ax, lon_deg, lat_deg, trans, c="red", label="", with_dim_names=False
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

    if with_dim_names:
        for i in range(len(corners)):
            if i in [0, 2]:
                dim_name = r"$\xi$"
            else:
                dim_name = r"$\eta$"
            # Define start and end points for each edge
            start_lon, start_lat = transformed_corners[i]
            end_lon, end_lat = transformed_corners[(i + 1) % len(corners)]

            # Compute midpoint
            mid_lon = (start_lon + end_lon) / 2
            mid_lat = (start_lat + end_lat) / 2

            # Compute vector direction for arrow
            arrow_dx = (end_lon - start_lon) * 0.4  # Scale arrow size
            arrow_dy = (end_lat - start_lat) * 0.4

            # Reverse arrow direction for edges 2 and 3
            if i in [2, 3]:
                arrow_dx *= -1
                arrow_dy *= -1

            # Add arrow
            ax.annotate(
                "",
                xy=(mid_lon + arrow_dx, mid_lat + arrow_dy),
                xytext=(mid_lon - arrow_dx, mid_lat - arrow_dy),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.5),
            )

            ax.text(
                mid_lon,
                mid_lat,
                dim_name,
                color=c,
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                    boxstyle="round,pad=0.2",
                ),
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
