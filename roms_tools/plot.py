import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from roms_tools.regrid import LateralRegridFromROMS, VerticalRegridFromROMS
from roms_tools.utils import (
    _generate_coordinate_range,
    _remove_edge_nans,
    infer_nominal_horizontal_resolution,
    normalize_longitude,
)
from roms_tools.vertical_coordinate import compute_depth_coordinates


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


def plot(
    field: xr.DataArray,
    grid_ds: xr.DataArray,
    grid_straddle: bool,
    zeta: xr.DataArray | float = 0.0,
    s: int | None = None,
    eta: int | None = None,
    xi: int | None = None,
    depth: float | None = None,
    lat: float | None = None,
    lon: float | None = None,
    include_boundary: bool = False,
    depth_contours: bool = False,
    ax: Axes | None = None,
    save_path: str | None = None,
    cmap_name: str | None = "YlOrRd",
) -> None:
    """Generate a plot of a ROMS output field for a specified vertical or horizontal
    slice.

    Parameters
    ----------
    field : xr.DataArray
        ROMS output variable already selected at a single time index.

    s : int, optional
        The index of the vertical layer (`s_rho`) to plot. If specified, the plot
        will display a horizontal slice at that layer. Cannot be used simultaneously
        with `depth`. Default is None.

    eta : int, optional
        The eta-index to plot. Used for generating vertical sections or plotting
        horizontal slices along a constant eta-coordinate. Cannot be used simultaneously
        with `lat` or `lon`, but can be combined with `xi`. Default is None.

    xi : int, optional
        The xi-index to plot. Used for generating vertical sections or plotting
        horizontal slices along a constant xi-coordinate. Cannot be used simultaneously
        with `lat` or `lon`, but can be combined with `eta`. Default is None.

    depth : float, optional
        Depth (in meters) to plot a horizontal slice at a specific depth level.
        If specified, the plot will interpolate the field to the given depth.
        Cannot be used simultaneously with `s` or for fields that are inherently
        2D (such as "zeta"). Default is None.

    lat : float, optional
        Latitude (in degrees) to plot a vertical section at a specific
        latitude. This option is useful for generating zonal (west-east)
        sections. Cannot be used simultaneously with `eta` or `xi`, bu can be
        combined with `lon`. Default is None.

    lon : float, optional
        Longitude (in degrees) to plot a vertical section at a specific
        longitude. This option is useful for generating meridional (south-north) sections.
        Cannot be used simultaneously with `eta` or `xi`, but can be combined
        with `lat`. Default is None.

    include_boundary : bool, optional
        Whether to include the outermost grid cells along the `eta`- and `xi`-boundaries in the plot.
        In diagnostic ROMS output fields, these boundary cells are set to zero, so excluding them can improve visualization.
        Default is False.

    depth_contours : bool, optional
        If True, overlays contours representing lines of constant depth on the plot.
        This option is only relevant when the `s` parameter is provided (i.e., not None).
        By default, depth contours are not shown (False).

    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is created. Note that this argument is ignored for 2D horizontal plots. Default is None.

    save_path : str, optional
        Path to save the generated plot. If None, the plot is shown interactively.
        Default is None.

    cmap_name : str, optional
        Colormap name to use.

    Returns
    -------
    None
        This method does not return any value. It generates and displays a plot.

    Raises
    ------
    ValueError
        - If the specified `var_name` is not one of the valid options.
        - If the field specified by `var_name` is 3D and none of `s`, `eta`, `xi`, `depth`, `lat`, or `lon` are specified.
        - If the field specified by `var_name` is 2D and both `eta` and `xi` or both `lat` and `lon` are specified.
        - If conflicting dimensions are specified (e.g., specifying `eta`/`xi` with `lat`/`lon` or both `s` and `depth`).
        - If more than two dimensions are specified for a 3D field.
        - If `time` exceeds the bounds of the time dimension.
        - If `time` is specified for a field that does not have a time dimension.
        - If `eta` or `xi` indices are out of bounds.
        - If `eta` or `xi` lie on the boundary when `include_boundary=False`.
    """
    # Input checks
    _validate_plot_inputs(field, s, eta, xi, depth, lat, lon, include_boundary)

    # Get horizontal dimensions and grid location
    horizontal_dims_dict = {
        "rho": {"eta": "eta_rho", "xi": "xi_rho"},
        "u": {"eta": "eta_rho", "xi": "xi_u"},
        "v": {"eta": "eta_v", "xi": "xi_rho"},
    }
    for loc, horizontal_dims in horizontal_dims_dict.items():
        if all(dim in field.dims for dim in horizontal_dims.values()):
            break

    # Convert relative to absolute indices
    def _get_absolute_index(idx, field, dim_name):
        index = field[dim_name].isel(**{dim_name: idx}).item()
        return index

    if eta is not None and eta < 0:
        eta = _get_absolute_index(eta, field, horizontal_dims["eta"])
    if xi is not None and xi < 0:
        xi = _get_absolute_index(xi, field, horizontal_dims["xi"])
    if s is not None and s < 0:
        s = _get_absolute_index(s, field, "s_rho")

    # Set spatial coordinates
    lat_deg = grid_ds[f"lat_{loc}"]
    lon_deg = grid_ds[f"lon_{loc}"]
    if grid_straddle:
        lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)
    if lon is not None:
        lon = normalize_longitude(lon, grid_straddle)

    field = field.assign_coords({"lon": lon_deg, "lat": lat_deg})

    # Mask the field
    mask = grid_ds[f"mask_{loc}"]
    field = field.where(mask)

    # Assign eta and xi as coordinates
    coords_to_assign = {dim: field[dim] for dim in horizontal_dims.values()}
    field = field.assign_coords(**coords_to_assign)

    # Remove horizontal boundary if desired
    slice_dict = {
        "rho": {"eta_rho": slice(1, -1), "xi_rho": slice(1, -1)},
        "u": {"eta_rho": slice(1, -1), "xi_u": slice(1, -1)},
        "v": {"eta_v": slice(1, -1), "xi_rho": slice(1, -1)},
    }
    if not include_boundary:
        field = field.isel(**slice_dict[loc])

    # Compute layer depth for 3D fields when depth contours are requested or no vertical layer is specified.
    compute_layer_depth = len(field.dims) > 2 and (depth_contours or s is None)
    if compute_layer_depth:
        layer_depth = compute_depth_coordinates(
            grid_ds,
            zeta,
            depth_type="layer",
            location=loc,
            eta=eta,
            xi=xi,
        )

        if not include_boundary:
            # Apply valid slices only for dimensions that exist in layer_depth.dims
            layer_depth = layer_depth.isel(
                **{
                    dim: s
                    for dim, s in slice_dict.get(loc, {}).items()
                    if dim in layer_depth.dims
                }
            )
        layer_depth.load()

    # Prepare figure title
    formatted_time = np.datetime_as_string(field.abs_time.values, unit="m")
    title = f"time: {formatted_time}"

    # Slice the field horizontally as desired
    def _slice_along_dimension(field, title, dim_name, idx):
        field = field.sel(**{dim_name: idx})
        title = title + f", {dim_name} = {idx}"
        return field, title

    if eta is not None:
        field, title = _slice_along_dimension(field, title, horizontal_dims["eta"], eta)
    if xi is not None:
        field, title = _slice_along_dimension(field, title, horizontal_dims["xi"], xi)
    if s is not None:
        field, title = _slice_along_dimension(field, title, "s_rho", s)
        if compute_layer_depth:
            layer_depth = layer_depth.isel(s_rho=s)
    else:
        depth_contours = False

    # Regrid laterally
    if lat is not None or lon is not None:
        if lat is not None:
            lats = [lat]
            title = title + f", lat = {lat}°N"
        else:
            resolution = infer_nominal_horizontal_resolution(grid_ds)
            lats = _generate_coordinate_range(
                field.lat.min().values, field.lat.max().values, resolution
            )
        lats = xr.DataArray(lats, dims=["lat"], attrs={"units": "°N"})

        if lon is not None:
            lons = [lon]
            title = title + f", lon = {lon}°E"
        else:
            resolution = infer_nominal_horizontal_resolution(grid_ds, lat)
            lons = _generate_coordinate_range(
                field.lon.min().values, field.lon.max().values, resolution
            )
        lons = xr.DataArray(lons, dims=["lon"], attrs={"units": "°E"})

        target_coords = {"lat": lats, "lon": lons}
        lateral_regrid = LateralRegridFromROMS(field, target_coords)
        field = lateral_regrid.apply(field).squeeze()

        if compute_layer_depth:
            layer_depth = lateral_regrid.apply(layer_depth).squeeze()

    # Assign depth as coordinate
    if compute_layer_depth:
        field = field.assign_coords({"layer_depth": layer_depth})

    if lat is not None:
        field, layer_depth = _remove_edge_nans(
            field, "lon", layer_depth if "layer_depth" in locals() else None
        )
    if lon is not None:
        field, layer_depth = _remove_edge_nans(
            field, "lat", layer_depth if "layer_depth" in locals() else None
        )

    # Regrid vertically
    if depth is not None:
        ds = xr.Dataset()
        ds["s_rho"] = field["s_rho"]
        vertical_regrid = VerticalRegridFromROMS(ds)
        # Save attributes before vertical regridding
        attrs = field.attrs
        field = vertical_regrid.apply(field, layer_depth, np.array([depth])).squeeze()
        # Reset attributes
        field.attrs = attrs
        title = title + f", depth = {depth}m"

    # Plotting
    if cmap_name == "RdBu_r":
        vmax = max(field.max().values, -field.min().values)
        vmin = -vmax
    else:
        vmax = field.max().values
        vmin = field.min().values

    cmap = plt.colormaps.get_cmap(cmap_name)
    cmap.set_bad(color="gray")

    if (eta is None and xi is None) and (lat is None and lon is None):
        fig = _plot(
            field=field,
            depth_contours=depth_contours,
            title=title,
            kwargs={"vmax": vmax, "vmin": vmin, "cmap": cmap},
            c=None,
        )
    else:
        if len(field.dims) == 2:
            fig = _section_plot(
                field,
                interface_depth=None,
                title=title,
                kwargs={"vmax": vmax, "vmin": vmin, "cmap": cmap},
                ax=ax,
            )
        else:
            if "s_rho" in field.dims:
                fig = _profile_plot(field, title=title, ax=ax)
            else:
                fig = _line_plot(field, title=title, ax=ax)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def _validate_plot_inputs(field, s, eta, xi, depth, lat, lon, include_boundary):
    """Validate input parameters for the plot method.

    Parameters
    ----------
    field : xr.DataArray
        Input data to be plotted.
    s : int, float, or None
        Depth level index or value for the s-coordinate. Use None for surface plotting.
    eta : int or None
        Eta index for ROMS grid selection. Must be within bounds.
    xi : int or None
        Xi index for ROMS grid selection. Must be within bounds.
    depth : int, float, or None
        Depth value for slicing. Not yet implemented.
    lat : float or None
        Latitude value for slicing. Must be specified with `lon` if provided.
    lon : float or None
        Longitude value for slicing. Must be specified with `lat` if provided.
    include_boundary : bool
        Whether to include boundary points when selecting grid indices.

    Raises
    ------
    ValueError
        If conflicting dimensions are specified.
        If eta or xi indices are out of bounds.
        If eta or xi lie on the boundary when `include_boundary=False`.
    """

    # Check conflicting dimension choices
    if s is not None and depth is not None:
        raise ValueError(
            "Conflicting input: You cannot specify both 's' and 'depth' at the same time."
        )
    if any([eta is not None, xi is not None]) and any(
        [lat is not None, lon is not None]
    ):
        raise ValueError(
            "Conflicting input: You cannot specify 'lat' or 'lon' simultaneously with 'eta' or 'xi'."
        )

    # 3D fields: Check for valid dimension specification
    if len(field.dims) == 3:
        if not any(
            [
                s is not None,
                eta is not None,
                xi is not None,
                depth is not None,
                lat is not None,
                lon is not None,
            ]
        ):
            raise ValueError(
                "Invalid input: For 3D fields, you must specify at least one of the dimensions 's', 'eta', 'xi', 'depth', 'lat', or 'lon'."
            )
        if sum([dim is not None for dim in [s, eta, xi, depth, lat, lon]]) > 2:
            raise ValueError(
                "Ambiguous input: For 3D fields, specify at most two of 's', 'eta', 'xi', 'depth', 'lat', or 'lon'. Specifying more than two is not allowed."
            )

    # 2D fields: Check for conflicts in dimension choices
    if len(field.dims) == 2:
        if s is not None:
            raise ValueError("Vertical dimension 's' should be None for 2D fields.")
        if depth is not None:
            raise ValueError("Vertical dimension 'depth' should be None for 2D fields.")
        if all([eta is not None, xi is not None]):
            raise ValueError(
                "Conflicting input: For 2D fields, specify only one dimension, either 'eta' or 'xi', not both."
            )
        if all([lat is not None, lon is not None]):
            raise ValueError(
                "Conflicting input: For 2D fields, specify only one dimension, either 'lat' or 'lon', not both."
            )

    # Check that indices are within bounds
    if eta is not None:
        dim = "eta_rho" if "eta_rho" in field.dims else "eta_v"
        if not eta < len(field[dim]):
            raise ValueError(
                f"Invalid eta index: {eta} is out of bounds. Must be between 0 and {len(field[dim]) - 1}."
            )
        if not include_boundary:
            if eta == 0 or eta == len(field[dim]) - 1:
                raise ValueError(
                    f"Invalid eta index: {eta} lies on the boundary, which is excluded when `include_boundary = False`. "
                    "Either set `include_boundary = True`, or adjust eta to avoid boundary values."
                )

    if xi is not None:
        dim = "xi_rho" if "xi_rho" in field.dims else "xi_u"
        if not xi < len(field[dim]):
            raise ValueError(
                f"Invalid eta index: {xi} is out of bounds. Must be between 0 and {len(field[dim]) - 1}."
            )
        if not include_boundary:
            if xi == 0 or xi == len(field[dim]) - 1:
                raise ValueError(
                    f"Invalid xi index: {xi} lies on the boundary, which is excluded when `include_boundary = False`. "
                    "Either set `include_boundary = True`, or adjust eta to avoid boundary values."
                )
