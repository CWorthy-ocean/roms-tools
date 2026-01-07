from dataclasses import dataclass

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from roms_tools.analysis.cdr_analysis import compute_cdr_metrics
from roms_tools.datasets.roms_dataset import ROMSDataset
from roms_tools.plot import plot, plot_uptake_efficiency
from roms_tools.regrid import LateralRegridFromROMS, VerticalRegridFromROMS
from roms_tools.utils import (
    generate_coordinate_range,
    infer_nominal_horizontal_resolution,
)


@dataclass(kw_only=True)
class ROMSOutput(ROMSDataset):
    """Represents ROMS model output.

    Parameters
    ----------
    path: str | Path | list[str | Path]
        Filename, or list of filenames with model output.
    grid : Grid
        Object representing the grid information.
    model_reference_date : datetime, optional
        Reference date of ROMS simulation.
        If not specified, this is inferred from metadata of the model output
        If specified and does not coincide with metadata, a warning is raised.
    adjust_depth_for_sea_surface_height : bool, optional
        Whether to account for sea surface height variations when computing depth coordinates.
        Defaults to `False`.
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.
    """

    def cdr_metrics(self) -> None:
        """
        Compute and plot Carbon Dioxide Removal (CDR) metrics.

        If the CDR metrics dataset (`self.ds_cdr`) does not already exist,
        it computes the metrics using model output and grid information.
        Afterwards, it generates a plot of the computed metrics.

        Notes
        -----
        Metrics include:
          - Grid cell area
          - Selected tracer and flux variables
          - Uptake efficiency computed from flux differences and DIC differences
        """
        if not hasattr(self, "ds_cdr"):
            # Compute metrics and store
            self.ds_cdr = compute_cdr_metrics(self.ds, self.grid.ds)

        plot_uptake_efficiency(self.ds_cdr)

    def plot(
        self,
        var_name: str,
        time: int = 0,
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
    ) -> None:
        """Generate a plot of a ROMS output field for a specified vertical or horizontal
        slice.

        Parameters
        ----------
        var_name : str
            Name of the variable to plot. Supported options include:

                - Oceanographic fields: "temp", "salt", "zeta", "u", "v", "w", etc.
                - Biogeochemical tracers: "PO4", "NO3", "O2", "DIC", "ALK", etc.

        time : int, optional
            Index of the time dimension to plot. Default is 0.

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
        # Check if variable exists
        if var_name not in self.ds:
            raise ValueError(f"Variable '{var_name}' is not found in self.ds.")

        # Pick the variable
        field = self.ds[var_name]

        # Check and pick time
        if "time" in field.dims:
            if time >= len(field.time):
                raise ValueError(
                    f"Invalid time index: The specified time index ({time}) exceeds the maximum index "
                    f"({len(field.time) - 1}) for the 'time' dimension."
                )
            field = field.isel(time=time)
        else:
            if time > 0:
                raise ValueError(
                    f"Invalid input: The field does not have a 'time' dimension, "
                    f"but a time index ({time}) greater than 0 was provided."
                )
        # Load the data
        if self.use_dask:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                field.load()

        if self.adjust_depth_for_sea_surface_height:
            zeta = self.ds.zeta.isel(time=time)
        else:
            zeta = 0

        # Choose colorbar
        if var_name in ["u", "v", "w", "ubar", "vbar", "zeta"]:
            cmap_name = "RdBu_r"
        elif var_name in ["temp", "salt"]:
            cmap_name = "YlOrRd"
        else:
            cmap_name = "YlGn"

        plot(
            field=field,
            grid_ds=self.grid.ds,
            zeta=zeta,
            s=s,
            eta=eta,
            xi=xi,
            depth=depth,
            lat=lat,
            lon=lon,
            include_boundary=include_boundary,
            depth_contours=depth_contours,
            layer_contours=False,
            ax=ax,
            save_path=save_path,
            cmap_name=cmap_name,
        )

    def regrid(self, var_names=None, horizontal_resolution=None, depth_levels=None):
        """Regrid the dataset both horizontally and vertically.

        This method selects the specified variables, interpolates them onto a lat-lon-z horizontal grid. The horizontal target resolution and vertical target depth levels are either specified or inferred dynamically.

        Parameters
        ----------
        var_names : list of str, optional
            List of variable names to be regridded. If None, all variables in the dataset
            are used.
        horizontal_resolution : float, optional
            Target horizontal resolution in degrees. If None, the nominal horizontal resolution is inferred from the grid.
        depth_levels : xarray.DataArray, numpy.ndarray, list, optional
            Target depth levels. If None, depth levels are determined dynamically.
            If provided as a list or numpy array, it is safely converted to an `xarray.DataArray`.

        Returns
        -------
        xarray.Dataset
            The regridded dataset.
        """
        if var_names is None:
            var_names = list(self.ds.data_vars)

        # Check that all var_names exist in self.ds
        missing_vars = [var for var in var_names if var not in self.ds.data_vars]
        if missing_vars:
            raise ValueError(
                f"The following variables are not found in the dataset: {', '.join(missing_vars)}"
            )

        # Retain only the variables in var_names and drop others
        ds = self.ds[var_names]

        # Prepare lateral target coords
        lat_deg = self.grid.ds["lat_rho"]
        lon_deg = self.grid.ds["lon_rho"]
        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)
        if horizontal_resolution is None:
            horizontal_resolution = infer_nominal_horizontal_resolution(self.grid.ds)
        lons = generate_coordinate_range(
            lon_deg.min().values, lon_deg.max().values, horizontal_resolution
        )
        lons = xr.DataArray(lons, dims=["lon"], attrs={"units": "°E"})
        lats = generate_coordinate_range(
            lat_deg.min().values, lat_deg.max().values, horizontal_resolution
        )
        lats = xr.DataArray(lats, dims=["lat"], attrs={"units": "°N"})
        target_coords = {"lat": lats, "lon": lons}

        # Prepare vertical target coords
        if depth_levels is None:
            depth_levels, _ = self.grid._compute_exponential_depth_levels()
        if not isinstance(depth_levels, xr.DataArray):
            depth_levels = xr.DataArray(
                np.asarray(depth_levels),
                dims=["depth"],
                attrs={"long_name": "Depth", "units": "m"},
            )
        depth_levels = depth_levels.astype(np.float32)

        # Interpolate velocities to rho-points and rotate to east/north directions
        self.rotate_velocities_to_east_and_north(
            velocity_pairs=(("u", "v"), ("ubar", "vbar"), ("u_slow", "v_slow"))
        )

        # Compute depth coordinates on source data
        self._get_depth_coordinates(depth_type="layer", locations=["rho"])
        layer_depth = self.ds_depth_coords["layer_depth_rho"]
        h = self.grid.ds.h

        # Rename coordinates
        ds = (
            self.ds[[var_names[var]["name"] for var in var_names]]
            .rename({"lat_rho": "lat", "lon_rho": "lon"})
            .where(self.grid.ds.mask_rho)
        )

        # Exclude the horizontal boundary cells since diagnostic variables may contain zeros there
        coords = {"eta_rho": slice(1, -1), "xi_rho": slice(1, -1)}
        ds = ds.isel(**coords)
        layer_depth.isel(**coords)
        h = h.isel(**coords)

        # Lateral regridding
        lateral_regrid = LateralRegridFromROMS(ds, target_coords)
        ds = lateral_regrid.apply(ds)
        layer_depth = lateral_regrid.apply(layer_depth)
        h = lateral_regrid.apply(h)

        # Vertical regridding
        if "s_rho" in ds.dims:
            vertical_regrid = VerticalRegridFromROMS(ds)
            for var_name in var_names:
                if "s_rho" in ds[var_name].dims:
                    attrs = ds[var_name].attrs
                    regridded = vertical_regrid.apply(
                        ds[var_name],
                        layer_depth,
                        depth_levels,
                        mask_edges=False,
                    )
                    regridded = regridded.where(regridded.depth < h)
                    ds[var_name] = regridded
                    ds[var_name].attrs = attrs

            ds = ds.assign_coords({"depth": depth_levels})

        ds["time"].attrs = {"long_name": "Time"}
        ds["lon"].attrs = {"long_name": "Longitude", "units": "Degrees East"}
        ds["lat"].attrs = {"long_name": "Latitude", "units": "Degrees North"}

        return ds
