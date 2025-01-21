import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from roms_tools.vertical_coordinate import retrieve_depth_coordinates


def _plot(
    field,
    depth_contours=False,
    c="red",
    title="",
    with_dim_names=False,
    plot_data=True,
    kwargs={},
):
    """Plots a grid or field on a map with optional depth contours.

    This function plots a map using Cartopy projections. It supports plotting a grid, a field, and adding depth contours if desired.

    Parameters
    ----------
    field : xarray.DataArray, optional
        The field to plot. If None, only the grid is plotted.
    depth_contours : bool, optional
        If True, adds depth contours to the plot.
    c : str, optional
        Color for the boundary plot (default is 'red').
    title : str, optional
        Title of the plot.
    plot_data : bool, optional
        If True, plots the provided field data on the map. If False, only the grid
        boundaries and optional depth contours are plotted. Default is True.
    kwargs : dict, optional
        Additional keyword arguments to pass to `pcolormesh` (e.g., colormap or color limits).

    Notes
    -----
    The function raises a `NotImplementedError` if the domain contains the North or South Pole.
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

    lon_deg = lon_deg.values
    lat_deg = lat_deg.values

    fig, ax = plt.subplots(1, 1, figsize=(13, 7), subplot_kw={"projection": trans})

    if c is not None:
        _add_boundary_to_ax(
            ax, lon_deg, lat_deg, trans, c, with_dim_names=with_dim_names
        )

    if plot_data:
        _add_field_to_ax(ax, lon_deg, lat_deg, field, depth_contours, kwargs=kwargs)

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


def _plot_nesting(parent_grid_ds, child_grid_ds, parent_straddle, with_dim_names=False):

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


def _plot_slice_of_field_with_two_horizontal_dims(
    field,
    mask,
    ds,
    ds_grid,
    s=None,
    eta=None,
    xi=None,
    depth_contours=False,
    layer_contours=False,
    ax=None,
    kwargs={},
) -> None:
    """Plot a ROMS output field for a given eta-, xi-, or s_rho- slice.

    Parameters
    ----------
    field : a 2D or 3D field with two horizontal dimension without time dimension
    mask : the corresponding mask
    ds : xarray.Dataset
        Dataset with vertical coordinate information and no time dimension.
        For static vertical coordinate, this is typically the grid file.
        For moving vertical coordinate, this is typically a time slice of a dataset with time dimension.
        A to which the computed depth coordinates will be looked for, and if not found, added.
        The dataset should not have a time dimension (it could be a time slice of another dataset,
        where time slice is same as for field).

        This dataset should ideally contain the sea surface height variable (`zeta`),
        which represents the free surface elevation. If `zeta` is not available,
        a default value of 0 will be used, resulting in static vertical coordinates.

        This dataset may be the same as `grid_ds` if you wish to compute depth
        coordinates without accounting for variations in `zeta`. If they are the
        same, the computation will use a static vertical coordinate system based
        solely on bathymetry and stretching parameters.

        After execution, the dataset will be updated to include the computed
        depth coordinates for the specified locations (e.g., rho, u, v points).

    grid_ds : xarray.Dataset
        The grid dataset containing essential information for depth calculations,
        such as bathymetry (`h`), stretching curves (`Cs_r` and `Cs_w`), and
        stretching parameters (`sigma_r` and `sigma_w`).
    s : int, optional
        The index of the vertical layer (`s_rho`) to plot. If not specified, the plot
        will represent a horizontal slice (eta- or xi- plane). Default is None.
    eta : int, optional
        The eta-index to plot. Used for vertical sections or horizontal slices.
        Default is None.
    xi : int, optional
        The xi-index to plot. Used for vertical sections or horizontal slices.
        Default is None.
    depth_contours : bool, optional
        If True, depth contours will be overlaid on the plot, showing lines of constant
        depth. This is typically used for plots that show a single vertical layer.
        Default is False.
    layer_contours : bool, optional
        If True, contour lines representing the boundaries between vertical layers will
        be added to the plot. This is useful in vertical sections to
        visualize the layering of the water column. For clarity, the number of layer
        contours displayed is limited to a maximum of 10. Default is False.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is created. Note that this argument does not work for horizontal plots that display the eta- and xi-dimensions at the same time.

    Returns
    -------
    None
        This method does not return any value. It generates and displays a plot.
    """

    if len(field.dims) == 3 and not any(
        [s is not None, eta is not None, xi is not None]
    ):
        raise ValueError(
            "For 3D fields, at least one of s, eta, or xi must be specified."
        )

    if len(field.dims) == 2 and all([eta is not None, xi is not None]):
        raise ValueError("For 2D fields, specify either eta or xi, not both.")

    compute_layer_depth = False
    compute_interface_depth = False

    if depth_contours or s is None:
        # depth coordinates are needed for any vertical section plot, or if the user wants to add depth contours to a plot of a single vertical layer
        if len(field.dims) > 2:
            compute_layer_depth = True
    if layer_contours and s is None:
        # layer contours are only used for vertical section plots and if specified by user
        compute_interface_depth = True

    if all(dim in field.dims for dim in ["eta_rho", "xi_rho"]):
        if compute_layer_depth:
            if "layer_depth_rho" not in ds:
                retrieve_depth_coordinates(
                    ds, ds_grid, type="layer", additional_locations=[]
                )
            layer_depth = ds.layer_depth_rho
        if compute_interface_depth:
            if "interface_depth_rho" not in ds:
                retrieve_depth_coordinates(
                    ds, ds_grid, type="interface", additional_locations=[]
                )
            interface_depth = ds.interface_depth_rho

    elif all(dim in field.dims for dim in ["eta_rho", "xi_u"]):
        if compute_layer_depth:
            if "layer_depth_u" not in ds:
                retrieve_depth_coordinates(
                    ds, ds_grid, type="layer", additional_locations=["u", "v"]
                )
            layer_depth = ds.layer_depth_u
        if compute_interface_depth:
            if "interface_depth_u" not in ds:
                retrieve_depth_coordinates(
                    ds, ds_grid, type="interface", additional_locations=["u", "v"]
                )
            interface_depth = ds.interface_depth_u

    elif all(dim in field.dims for dim in ["eta_v", "xi_rho"]):
        if compute_layer_depth:
            if "layer_depth_v" not in ds:
                retrieve_depth_coordinates(
                    ds, ds_grid, type="layer", additional_locations=["u", "v"]
                )
            layer_depth = ds.layer_depth_v
        if compute_interface_depth:
            if "interface_depth_v" not in ds:
                retrieve_depth_coordinates(
                    ds, ds_grid, type="interface", additional_locations=["u", "v"]
                )
            interface_depth = ds.interface_depth_v

    else:
        ValueError("provided field does not have two horizontal dimension")

    # slice the field as desired
    title = field.long_name
    if s is not None:
        title = title + f", s_rho = {field.s_rho[s].item()}"
        field = field.isel(s_rho=s)
        if compute_layer_depth:
            layer_depth = layer_depth.isel(s_rho=s)
            field = field.assign_coords({"layer_depth": layer_depth})
    else:
        depth_contours = False

    if eta is not None:
        if "eta_rho" in field.dims:
            title = title + f", eta_rho = {field.eta_rho[eta].item()}"
            field = field.isel(eta_rho=eta)
            if compute_layer_depth:
                layer_depth = layer_depth.isel(eta_rho=eta)
                if "s_rho" in field.dims:
                    field = field.assign_coords({"layer_depth": layer_depth})
            if compute_interface_depth:
                interface_depth = interface_depth.isel(eta_rho=eta)
        elif "eta_v" in field.dims:
            title = title + f", eta_v = {field.eta_v[eta].item()}"
            field = field.isel(eta_v=eta)
            if compute_layer_depth:
                layer_depth = layer_depth.isel(eta_v=eta)
                if "s_rho" in field.dims:
                    field = field.assign_coords({"layer_depth": layer_depth})
            if compute_interface_depth:
                interface_depth = interface_depth.isel(eta_v=eta)
        else:
            raise ValueError(
                "None of the expected dimensions (eta_rho, eta_v) found in field."
            )
    if xi is not None:
        if "xi_rho" in field.dims:
            title = title + f", xi_rho = {field.xi_rho[xi].item()}"
            field = field.isel(xi_rho=xi)
            if compute_layer_depth:
                layer_depth = layer_depth.isel(xi_rho=xi)
                if "s_rho" in field.dims:
                    field = field.assign_coords({"layer_depth": layer_depth})
            if compute_interface_depth:
                interface_depth = interface_depth.isel(xi_rho=xi)
        elif "xi_u" in field.dims:
            title = title + f", xi_u = {field.xi_u[xi].item()}"
            field = field.isel(xi_u=xi)
            if compute_layer_depth:
                layer_depth = layer_depth.isel(xi_u=xi)
                if "s_rho" in field.dims:
                    field = field.assign_coords({"layer_depth": layer_depth})
            if compute_interface_depth:
                interface_depth = interface_depth.isel(xi_u=xi)
        else:
            raise ValueError(
                "None of the expected dimensions (xi_rho, xi_u) found in field."
            )

    if eta is None and xi is None:
        _plot(
            field=field.where(mask),
            depth_contours=depth_contours,
            title=title,
            kwargs=kwargs,
            c="g",
        )
    else:
        if not layer_contours:
            interface_depth = None
        else:
            # restrict number of layer_contours to 10 for the sake of plot clearity
            nr_layers = len(interface_depth["s_w"])
            selected_layers = np.linspace(
                0, nr_layers - 1, min(nr_layers, 10), dtype=int
            )
            interface_depth = interface_depth.isel(s_w=selected_layers)

        if len(field.dims) == 2:
            _section_plot(
                field,
                interface_depth=interface_depth,
                title=title,
                kwargs=kwargs,
                ax=ax,
            )
        else:
            if "s_rho" in field.dims:
                _profile_plot(field, title=title, ax=ax)
            else:
                _line_plot(field, title=title, ax=ax)
