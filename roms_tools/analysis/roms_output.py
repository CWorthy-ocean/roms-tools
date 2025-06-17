import logging
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from roms_tools import Grid
from roms_tools.analysis.utils import _validate_plot_inputs
from roms_tools.plot import _line_plot, _plot, _profile_plot, _section_plot
from roms_tools.regrid import LateralRegridFromROMS, VerticalRegridFromROMS
from roms_tools.utils import (
    _generate_coordinate_range,
    _load_data,
    _remove_edge_nans,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
)
from roms_tools.vertical_coordinate import (
    compute_depth_coordinates,
)


@dataclass(kw_only=True)
class ROMSOutput:
    """Represents ROMS model output.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information.
    path : Union[str, Path, List[Union[str, Path]]]
        Filename, or list of filenames with model output.
    model_reference_date : datetime, optional
        If not specified, this is inferred from metadata of the model output
        If specified and does not coincide with metadata, a warning is raised.
    adjust_depth_for_sea_surface_height : bool, optional
        Whether to account for sea surface height variations when computing depth coordinates.
        Defaults to `False`.
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.
    """

    grid: Grid
    path: Union[str, Path]
    use_dask: bool = False
    model_reference_date: Optional[datetime] = None
    adjust_depth_for_sea_surface_height: Optional[bool] = False
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        ds = self._load_model_output()
        self._infer_model_reference_date_from_metadata(ds)
        self._check_vertical_coordinate(ds)
        ds = self._add_absolute_time(ds)
        ds = self._add_lat_lon_coords(ds)
        self.ds = ds

        # Dataset for depth coordinates
        self.ds_depth_coords = xr.Dataset()

    def plot(
        self,
        var_name,
        time=0,
        s=None,
        eta=None,
        xi=None,
        depth=None,
        lat=None,
        lon=None,
        include_boundary=False,
        depth_contours=False,
        ax=None,
        save_path=None,
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
            The axes to plot on. If None, a new figure is created. Note that this argument does not work for 2D horizontal plots. Default is None.

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
            raise ValueError(f"Variable '{var_name}' is not found in the dataset.")

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
        lat_deg = self.grid.ds[f"lat_{loc}"]
        lon_deg = self.grid.ds[f"lon_{loc}"]
        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)
        field = field.assign_coords({"lon": lon_deg, "lat": lat_deg})

        # Mask the field
        mask = self.grid.ds[f"mask_{loc}"]
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

        # Load the data
        if self.use_dask:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                field.load()

        # Compute layer depth for 3D fields when depth contours are requested or no vertical layer is specified.
        compute_layer_depth = len(field.dims) > 2 and (depth_contours or s is None)
        if compute_layer_depth:
            if eta is not None or xi is not None:
                # Computing depth coordinates directly for the slice in question is more efficient
                # than using .ds_depth_coords, which computes depth coordinates for full field
                if self.adjust_depth_for_sea_surface_height:
                    zeta = self.ds.zeta.isel(time=time)
                else:
                    zeta = 0
                if compute_layer_depth:
                    layer_depth = compute_depth_coordinates(
                        self.grid.ds,
                        zeta,
                        depth_type="layer",
                        location=loc,
                        eta=eta,
                        xi=xi,
                    )
            else:
                self._get_depth_coordinates(depth_type="layer", locations=[loc])
                layer_depth = self.ds_depth_coords[f"layer_depth_{loc}"]
                if self.adjust_depth_for_sea_surface_height:
                    layer_depth = layer_depth.isel(time=time)

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
            field, title = _slice_along_dimension(
                field, title, horizontal_dims["eta"], eta
            )
        if xi is not None:
            field, title = _slice_along_dimension(
                field, title, horizontal_dims["xi"], xi
            )
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
                resolution = self.grid._infer_nominal_horizontal_resolution()
                lats = _generate_coordinate_range(
                    field.lat.min().values, field.lat.max().values, resolution
                )
            lats = xr.DataArray(lats, dims=["lat"], attrs={"units": "°N"})

            if lon is not None:
                lons = [lon]
                title = title + f", lon = {lon}°E"
            else:
                resolution = self.grid._infer_nominal_horizontal_resolution(lat)
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
            vertical_regrid = VerticalRegridFromROMS(self.ds)
            # Save attributes before vertical regridding
            attrs = field.attrs
            field = vertical_regrid.apply(
                field, layer_depth, np.array([depth])
            ).squeeze()
            # Reset attributes
            field.attrs = attrs
            title = title + f", depth = {depth}m"

        # Choose colorbar
        if var_name in ["u", "v", "w", "ubar", "vbar", "zeta"]:
            vmax = max(field.max().values, -field.min().values)
            vmin = -vmax
            cmap = plt.colormaps.get_cmap("RdBu_r")
        else:
            vmax = field.max().values
            vmin = field.min().values
            if var_name in ["temp", "salt"]:
                cmap = plt.colormaps.get_cmap("YlOrRd")
            else:
                cmap = plt.colormaps.get_cmap("YlGn")
        cmap.set_bad(color="gray")
        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

        # Plotting
        if (eta is None and xi is None) and (lat is None and lon is None):
            fig = _plot(
                field=field,
                depth_contours=depth_contours,
                title=title,
                kwargs=kwargs,
                c=None,
            )
        else:
            if len(field.dims) == 2:
                fig = _section_plot(
                    field,
                    interface_depth=None,
                    title=title,
                    kwargs=kwargs,
                    ax=ax,
                )
            else:
                if "s_rho" in field.dims:
                    fig = _profile_plot(field, title=title, ax=ax)
                else:
                    fig = _line_plot(field, title=title, ax=ax)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

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

        # Prepare lateral regrid
        lat_deg = self.grid.ds["lat_rho"]
        lon_deg = self.grid.ds["lon_rho"]
        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)

        if horizontal_resolution is None:
            horizontal_resolution = self.grid._infer_nominal_horizontal_resolution()
        lons = _generate_coordinate_range(
            lon_deg.min().values, lon_deg.max().values, horizontal_resolution
        )
        lons = xr.DataArray(lons, dims=["lon"], attrs={"units": "°E"})
        lats = _generate_coordinate_range(
            lat_deg.min().values, lat_deg.max().values, horizontal_resolution
        )
        lats = xr.DataArray(lats, dims=["lat"], attrs={"units": "°N"})
        target_coords = {"lat": lats, "lon": lons}

        # Prepare vertical regrid
        if depth_levels is None:
            depth_levels, _ = self.grid._compute_exponential_depth_levels()

        # Ensure depth_levels is an xarray.DataArray
        if not isinstance(depth_levels, xr.DataArray):
            depth_levels = xr.DataArray(
                np.asarray(depth_levels),
                dims=["depth"],
                attrs={"long_name": "Depth", "units": "m"},
            )

        depth_levels = depth_levels.astype(np.float32)

        # Initialize list to hold regridded datasets
        regridded_datasets = []

        for loc, dims in [
            ("rho", ("eta_rho", "xi_rho")),
            ("u", ("eta_rho", "xi_u")),
            ("v", ("eta_v", "xi_rho")),
        ]:
            var_names_loc = [
                var_name
                for var_name in var_names
                if all(dim in ds[var_name].dims for dim in dims)
            ]
            if var_names_loc:
                ds_loc = (
                    ds[var_names_loc]
                    .rename({f"lat_{loc}": "lat", f"lon_{loc}": "lon"})
                    .where(self.grid.ds[f"mask_{loc}"])
                )
                self._get_depth_coordinates(depth_type="layer", locations=[loc])
                layer_depth_loc = self.ds_depth_coords[f"layer_depth_{loc}"]
                h_loc = self.grid.ds.h
                if loc == "u":
                    h_loc = interpolate_from_rho_to_u(h_loc)
                elif loc == "v":
                    h_loc = interpolate_from_rho_to_v(h_loc)

                # Exclude the horizontal boundary cells since diagnostic variables may contain zeros there
                ds_loc = ds_loc.isel({dims[0]: slice(1, -1), dims[1]: slice(1, -1)})
                layer_depth_loc = layer_depth_loc.isel(
                    {dims[0]: slice(1, -1), dims[1]: slice(1, -1)}
                )
                h_loc = h_loc.isel({dims[0]: slice(1, -1), dims[1]: slice(1, -1)})

                # Lateral regridding
                lateral_regrid = LateralRegridFromROMS(ds_loc, target_coords)
                ds_loc = lateral_regrid.apply(ds_loc)
                layer_depth_loc = lateral_regrid.apply(layer_depth_loc)
                h_loc = lateral_regrid.apply(h_loc)
                # Vertical regridding
                vertical_regrid = VerticalRegridFromROMS(ds_loc)
                for var_name in var_names_loc:
                    if "s_rho" in ds_loc[var_name].dims:
                        attrs = ds_loc[var_name].attrs
                        regridded = vertical_regrid.apply(
                            ds_loc[var_name],
                            layer_depth_loc,
                            depth_levels,
                            mask_edges=False,
                        )
                        regridded = regridded.where(regridded.depth < h_loc)
                        ds_loc[var_name] = regridded
                        ds_loc[var_name].attrs = attrs

                ds_loc = ds_loc.assign_coords({"depth": depth_levels})

                # Collect regridded dataset for merging
                regridded_datasets.append(ds_loc)

        # Merge all regridded datasets
        if regridded_datasets:
            ds = xr.merge(regridded_datasets)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                ds = ds.rename({"abs_time": "time"}).set_index(time="time")
            ds["time"].attrs = {"long_name": "Time"}
            ds["lon"].attrs = {"long_name": "Longitude", "units": "Degrees East"}
            ds["lat"].attrs = {"long_name": "Latitude", "units": "Degrees North"}

            return ds

    def _get_depth_coordinates(self, depth_type="layer", locations=["rho"]):
        """Ensure depth coordinates are stored for a given location and depth type.

        Calculates vertical depth coordinates (layer or interface) for specified locations (e.g., rho, u, v points)
        and updates them in the dataset (`self.ds`).

        Parameters
        ----------
        depth_type : str
            The type of depth coordinate to compute. Valid options:
            - "layer": Compute layer depth coordinates.
            - "interface": Compute interface depth coordinates.
        locations : list[str], optional
            Locations for which to compute depth coordinates. Default is ["rho", "u", "v"].
            Valid options include:
            - "rho": Depth coordinates at rho points.
            - "u": Depth coordinates at u points.
            - "v": Depth coordinates at v points.

        Updates
        -------
        self.ds_depth_coords : xarray.Dataset

        Raises
        ------
        ValueError
            If `adjust_depth_for_sea_surface_height` is enabled but `zeta` is missing from `self.ds`.

        Notes
        -----
        - This method relies on the `compute_depth_coordinates` function to perform calculations.
        - If `adjust_depth_for_sea_surface_height` is `True`, the method accounts for variations
          in sea surface height (`zeta`).
        """

        if self.adjust_depth_for_sea_surface_height:
            if "zeta" not in self.ds:
                raise ValueError(
                    "`zeta` is required in provided ROMS output when `adjust_depth_for_sea_surface_height` is enabled."
                )
            zeta = self.ds.zeta
        else:
            zeta = 0

        for location in locations:
            var_name = f"{depth_type}_depth_{location}"
            if var_name not in self.ds_depth_coords:
                self.ds_depth_coords[var_name] = compute_depth_coordinates(
                    self.grid.ds, zeta, depth_type, location
                )

    def _load_model_output(self) -> xr.Dataset:
        """Load the model output."""

        # Load the dataset
        ds = _load_data(
            self.path,
            dim_names={"time": "time"},
            use_dask=self.use_dask,
            time_chunking=True,
            force_combine_nested=True,
        )

        return ds

    def _infer_model_reference_date_from_metadata(self, ds: xr.Dataset) -> None:
        """Infer and validate the model reference date from `ocean_time` metadata.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset with an `ocean_time` variable and a `long_name` attribute
            in the format `Time since YYYY/MM/DD`.

        Raises
        ------
        ValueError
            If `self.model_reference_date` is not set and the reference date cannot
            be inferred, or if the inferred date does not match `self.model_reference_date`.

        Warns
        -----
        UserWarning
            If `self.model_reference_date` is set but the reference date cannot be inferred.
        """
        # Check if 'long_name' exists in the attributes of 'ocean_time'
        if "long_name" in ds.ocean_time.attrs:
            input_string = ds.ocean_time.attrs["long_name"]
            match = re.search(r"(\d{4})/(\d{2})/(\d{2})", input_string)

            if match:
                # If a match is found, extract year, month, day and create the inferred date
                year, month, day = map(int, match.groups())
                inferred_date = datetime(year, month, day)

                if hasattr(self, "model_reference_date") and self.model_reference_date:
                    # Check if the inferred date matches the provided model reference date
                    if self.model_reference_date != inferred_date:
                        raise ValueError(
                            f"Mismatch between `self.model_reference_date` ({self.model_reference_date}) "
                            f"and inferred reference date ({inferred_date})."
                        )
                else:
                    # Set the model reference date if not already set
                    self.model_reference_date = inferred_date
            else:
                # Handle case where no match is found
                if hasattr(self, "model_reference_date") and self.model_reference_date:
                    logging.warning(
                        "Could not infer the model reference date from the metadata. "
                        "`self.model_reference_date` will be used.",
                    )
                else:
                    raise ValueError(
                        "Model reference date could not be inferred from the metadata, "
                        "and `self.model_reference_date` is not set."
                    )
        else:
            # Handle case where 'long_name' attribute doesn't exist
            if hasattr(self, "model_reference_date") and self.model_reference_date:
                logging.warning(
                    "`long_name` attribute not found in ocean_time. "
                    "`self.model_reference_date` will be used instead.",
                )
            else:
                raise ValueError(
                    "Model reference date could not be inferred from the metadata, "
                    "and `self.model_reference_date` is not set."
                )

    def _check_vertical_coordinate(self, ds: xr.Dataset) -> None:
        """Check that the vertical coordinate parameters in the dataset are consistent
        with the model grid.

        This method compares the vertical coordinate parameters (`theta_s`, `theta_b`, `hc`, `Cs_r`, `Cs_w`) in
        the provided dataset (`ds`) with those in the model grid (`self.grid`). The first three parameters are
        checked for exact equality, while the last two are checked for numerical closeness.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset containing vertical coordinate parameters in its attributes, such as `theta_s`, `theta_b`,
            `hc`, `Cs_r`, and `Cs_w`.

        Raises
        ------
        ValueError
            If the vertical coordinate parameters do not match the expected values (based on exact or approximate equality).

        Notes
        -----
        - `theta_s`, `theta_b`, and `hc` are checked for exact equality using `np.array_equal`.
        - `Cs_r` and `Cs_w` are checked for numerical closeness using `np.allclose`.
        """

        # Check exact equality for theta_s, theta_b, and hc
        if not np.array_equal(self.grid.theta_s, ds.attrs["theta_s"]):
            raise ValueError(
                f"theta_s from grid ({self.grid.theta_s}) does not match dataset ({ds.attrs['theta_s']})."
            )

        if not np.array_equal(self.grid.theta_b, ds.attrs["theta_b"]):
            raise ValueError(
                f"theta_b from grid ({self.grid.theta_b}) does not match dataset ({ds.attrs['theta_b']})."
            )

        if not np.array_equal(self.grid.hc, ds.attrs["hc"]):
            raise ValueError(
                f"hc from grid ({self.grid.hc}) does not match dataset ({ds.attrs['hc']})."
            )

        # Check numerical closeness for Cs_r and Cs_w
        if not np.allclose(self.grid.ds.Cs_r, ds.attrs["Cs_r"]):
            raise ValueError(
                f"Cs_r from grid ({self.grid.ds.Cs_r}) is not close to dataset ({ds.attrs['Cs_r']})."
            )

        if not np.allclose(self.grid.ds.Cs_w, ds.attrs["Cs_w"]):
            raise ValueError(
                f"Cs_w from grid ({self.grid.ds.Cs_w}) is not close to dataset ({ds.attrs['Cs_w']})."
            )

    def _add_absolute_time(self, ds: xr.Dataset) -> xr.Dataset:
        """Add absolute time as a coordinate to the dataset.

        Computes "abs_time" based on "ocean_time" and a reference date,
        and adds it as a coordinate.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing "ocean_time" in seconds since the model reference date.

        Returns
        -------
        xarray.Dataset
            Dataset with "abs_time" added and "time" removed.
        """
        ocean_time_seconds = ds["ocean_time"].values

        abs_time = np.array(
            [
                self.model_reference_date + timedelta(seconds=seconds)
                for seconds in ocean_time_seconds
            ]
        )

        abs_time = xr.DataArray(
            abs_time, dims=["time"], coords={"time": ds["ocean_time"]}
        )
        abs_time.attrs["long_name"] = "absolute time"
        ds = ds.assign_coords({"abs_time": abs_time})
        ds = ds.drop_vars("time")

        return ds

    def _add_lat_lon_coords(self, ds: xr.Dataset) -> xr.Dataset:
        """Add latitude and longitude coordinates to the dataset based on the grid.

        This method assigns latitude and longitude coordinates from the grid to the dataset.
        It always adds the "lat_rho" and "lon_rho" coordinates. If the dataset contains the
        "xi_u" or "eta_v" dimensions, it also adds the corresponding "lat_u", "lon_u",
        "lat_v", and "lon_v" coordinates.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset to which latitude and longitude coordinates will be added.

        Returns
        -------
        xarray.Dataset
            Updated dataset with the appropriate latitude and longitude coordinates
            assigned to "rho", "u", and "v" points if applicable.
        """
        coords_to_add = {
            "lat_rho": self.grid.ds["lat_rho"],
            "lon_rho": self.grid.ds["lon_rho"],
        }

        if "xi_u" in ds.dims:
            coords_to_add.update(
                {"lat_u": self.grid.ds["lat_u"], "lon_u": self.grid.ds["lon_u"]}
            )
        if "eta_v" in ds.dims:
            coords_to_add.update(
                {"lat_v": self.grid.ds["lat_v"], "lon_v": self.grid.ds["lon_v"]}
            )

        # Add all necessary coordinates in one go
        ds = ds.assign_coords(coords_to_add)
        return ds
