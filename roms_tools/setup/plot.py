import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr


def _plot(grid_ds, field=None, straddle=False, c="red", kwargs={}):
    lon_deg = grid_ds["lon_rho"]
    lat_deg = grid_ds["lat_rho"]

    if field is not None:
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

    if field is not None:
        p = ax.pcolormesh(lon_deg, lat_deg, field, transform=proj, **kwargs)
        plt.colorbar(p, label=f"{field.long_name} [{field.units}]")
