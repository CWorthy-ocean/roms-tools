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

    # Define projections
    proj = ccrs.PlateCarree()

    trans = ccrs.NearsidePerspective(
        central_longitude=lon_deg.mean().values, central_latitude=lat_deg.mean().values
    )

    lon_deg = lon_deg.values
    lat_deg = lat_deg.values

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

    fig, ax = plt.subplots(1, 1, figsize=(13, 7), subplot_kw={"projection": trans})

    ax.plot(
        list(transformed_lons) + [transformed_lons[0]],
        list(transformed_lats) + [transformed_lats[0]],
        "o-",
        c=c,
    )

    ax.coastlines(
        resolution="50m", linewidth=0.5, color="black"
    )  # add map of coastlines
    ax.gridlines()
    ax.set_title(title)

    if field is not None:
        p = ax.pcolormesh(lon_deg, lat_deg, field, transform=proj, **kwargs)
        if hasattr(field, "long_name"):
            label = f"{field.long_name} [{field.units}]"
        elif hasattr(field, "Long_name"):
            # this is the case for matlab generated grids
            label = f"{field.Long_name} [{field.units}]"
        else:
            label = ""
        plt.colorbar(p, label=label)

    if depth_contours:
        cs = ax.contour(lon_deg, lat_deg, field.layer_depth, transform=proj, colors="k")
        ax.clabel(cs, inline=True, fontsize=10)

    return fig


def _section_plot(field, interface_depth=None, title="", kwargs={}):

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


def _profile_plot(field, title=""):

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

    fig, ax = plt.subplots(1, 1, figsize=(4, 7))
    kwargs = {"y": depth_label, "yincrease": False}
    field.plot(**kwargs)
    ax.set_title(title)
    ax.grid()


def _line_plot(field, title=""):

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    field.plot()
    ax.set_title(title)
    ax.grid()
