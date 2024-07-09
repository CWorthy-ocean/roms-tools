import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

def _plot(grid_ds, field=None, depth_contours=False, straddle=False, c='red', title="", kwargs={}):

    if field is None:

        lon_deg = grid_ds["lon_rho"]
        lat_deg = grid_ds["lat_rho"]

    else:

        field = field.squeeze()
        
        if all(dim in field.dims for dim in ['eta_rho', 'xi_rho']):
            field = field.where(grid_ds.mask_rho)
            lon_deg = grid_ds["lon_rho"]
            lat_deg = grid_ds["lat_rho"]
        elif all(dim in field.dims for dim in ['eta_rho', 'xi_u']):
            field = field.where(grid_ds.mask_u)
            lon_deg = grid_ds["lon_u"]
            lat_deg = grid_ds["lat_u"]
        elif all(dim in field.dims for dim in ['eta_v', 'xi_rho']):
            field = field.where(grid_ds.mask_v)
            lon_deg = grid_ds["lon_v"]
            lat_deg = grid_ds["lat_v"]
        else:
            ValueError("provided field does not have two horizontal dimension")
        
        # check if North or South pole are in domain
        if lat_deg.max().values > 89 or lat_deg.min().values < -89:
            raise NotImplementedError("Plotting is not implemented for the case that the domain contains the North or South pole.")

    if straddle:
        lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)

    # Define projections
    proj = ccrs.PlateCarree()

    trans = ccrs.NearsidePerspective(
            central_longitude=lon_deg.mean().values,
            central_latitude=lat_deg.mean().values
    )


    lon_deg = lon_deg.values
    lat_deg = lat_deg.values

    # find corners
    corners = [
        (lon_deg[0, 0], lat_deg[0, 0]),
        (lon_deg[0, -1], lat_deg[0, -1]),
        (lon_deg[-1, -1], lat_deg[-1, -1]),
        (lon_deg[-1, 0], lat_deg[-1, 0])
    ]

    # transform coordinates to projected space
    transformed_corners = [trans.transform_point(lo, la, proj) for lo, la in corners]
    transformed_lons, transformed_lats = zip(*transformed_corners)

    fig, ax = plt.subplots(1, 1, figsize=(13, 7), subplot_kw={'projection': trans})

    ax.plot(
        list(transformed_lons) + [transformed_lons[0]],
        list(transformed_lats) + [transformed_lats[0]],
        "o-", c=c
    )

    ax.coastlines(
        resolution="50m", linewidth=0.5, color="black"
    )  # add map of coastlines
    ax.gridlines()
    ax.set_title(title)
        
    if field is not None:
        p = ax.pcolormesh(
                lon_deg, lat_deg, field, transform=proj, **kwargs
        )
        plt.colorbar(p, label=f"{field.long_name} [{field.units}]")

    if depth_contours:
        if all(dim in field.dims for dim in ['eta_rho', 'xi_rho']):
            depth = field.depth_rho
        elif all(dim in field.dims for dim in ['eta_rho', 'xi_u']):
            depth = field.depth_u
        elif all(dim in field.dims for dim in ['eta_v', 'xi_rho']):
            depth = field.depth_v

        cs = ax.contour(
                lon_deg, lat_deg, depth, transform=proj, colors='k'
                )
        ax.clabel(cs, inline=True, fontsize=10)
        
    return fig


def _section_plot(field, layer_contours=False, title="", kwargs={}):

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    dims_to_check = ["eta_rho", "xi_rho", "eta_v", "xi_u"]
    try:
        xdim = next(dim for dim in dims_to_check if dim in field.dims)
    except StopIteration:
        raise ValueError("None of the expected dimensions (eta_rho, xi_rho, eta_v, xi_u) found in field.dims")
        
    depths_to_check = ["depth_rho", "depth_u", "depth_v"]
    try:
        depth_label = next(depth_label for depth_label in depths_to_check if depth_label in field.coords)
    except StopIteration:
        raise ValueError("None of the expected coordinates (depth_rho, depth_u, depth_v) found in field.coords")

    more_kwargs = {"x": xdim, "y": depth_label, "yincrease": False}
    field.plot(**kwargs, **more_kwargs, ax=ax)
    ax.set_title(title)

    if layer_contours:
        nr_layers = len(field.s_rho)
        # Restrict the number of layers to 10 for readability
        if nr_layers > 10:
            selected_layers = np.linspace(0, nr_layers - 1, 10, dtype=int)
        else:
            selected_layers = range(nr_layers)

        for i in selected_layers:
            ax.plot(field[xdim], field[depth_label].isel(s_rho=i), color='k')


def _profile_plot(field, title=""):
    
    depths_to_check = ["depth_rho", "depth_u", "depth_v"]
    try:
        depth_label = next(depth_label for depth_label in depths_to_check if depth_label in field.coords)
    except StopIteration:
        raise ValueError("None of the expected coordinates (depth_rho, depth_u, depth_v) found in field.coords")

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

