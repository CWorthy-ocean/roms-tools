import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from roms_tools.utils import _load_data
from dataclasses import dataclass, field
from typing import Union, Optional
from pathlib import Path
import re
import logging
from datetime import datetime, timedelta
from roms_tools import Grid
from roms_tools.plot import _plot, _section_plot, _profile_plot, _line_plot
from roms_tools.vertical_coordinate import (
    compute_depth_coordinates,
)


@dataclass(frozen=True, kw_only=True)
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
    use_dask: bool, optional
        Indicates whether to use dask for processing. If True, data is processed with dask; if False, data is processed eagerly. Defaults to False.
    """

    grid: Grid
    path: Union[str, Path]
    use_dask: bool = False
    model_reference_date: Optional[datetime] = None
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):

        ds = self._load_model_output()
        self._infer_model_reference_date_from_metadata(ds)
        self._check_vertical_coordinate(ds)
        ds = self._add_absolute_time(ds)
        ds = self._add_lat_lon_coords(ds)
        object.__setattr__(self, "ds", ds)

        # Dataset for depth coordinates
        object.__setattr__(self, "ds_depth_coords", xr.Dataset())

    def plot(
        self,
        var_name,
        time=0,
        s=None,
        eta=None,
        xi=None,
        include_boundary=False,
        depth_contours=False,
        layer_contours=False,
        ax=None,
    ) -> None:
        """Plot a ROMS output field for a given vertical (s_rho) or horizontal (eta, xi)
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
            The index of the vertical layer (`s_rho`) to plot. If not specified, the plot
            will represent a horizontal slice (eta- or xi- plane). Default is None.
        eta : int, optional
            The eta-index to plot. Used for vertical sections or horizontal slices.
            Default is None.
        xi : int, optional
            The xi-index to plot. Used for vertical sections or horizontal slices.
            Default is None.
        include_boundary : bool, optional
            Whether to include the outermost grid cells along the `eta`- and `xi`-boundaries in the plot.
            In diagnostic ROMS output fields, these boundary cells are set to zero, so excluding them can improve visualization.
            This option is only relevant for 2D horizontal plots (`eta=None`, `xi=None`).
            Default is False.
        depth_contours : bool, optional
            If True, depth contours will be overlaid on the plot, showing lines of constant
            depth. This is typically used for plots that show a single vertical layer.
            Default is False.
        layer_contours : bool, optional
            If True, contour lines representing the boundaries between vertical layers will
            be added to the plot. This is particularly useful in vertical sections to
            visualize the layering of the water column. For clarity, the number of layer
            contours displayed is limited to a maximum of 10. Default is False.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure is created. Note that this argument does not work for horizontal plots that display the eta- and xi-dimensions at the same time.

        Returns
        -------
        None
            This method does not return any value. It generates and displays a plot.

        Raises
        ------
        ValueError
            If the specified `var_name` is not one of the valid options.
            If the field specified by `var_name` is 3D and none of `s`, `eta`, or `xi` are specified.
            If the field specified by `var_name` is 2D and both `eta` and `xi` are specified.
        """

        # Input checks
        if var_name not in self.ds:
            raise ValueError(f"Variable '{var_name}' is not found in dataset.")

        if "time" in self.ds[var_name].dims:
            if time >= len(self.ds[var_name].time):
                raise ValueError(
                    f"Invalid time index: The specified time index ({time}) exceeds the maximum index "
                    f"({len(self.ds[var_name].time) - 1}) for the 'time' dimension in variable '{var_name}'."
                )
            field = self.ds[var_name].isel(time=time)
        else:
            if time > 0:
                raise ValueError(
                    f"Invalid input: The variable '{var_name}' does not have a 'time' dimension, "
                    f"but a time index ({time}) greater than 0 was provided."
                )
            field = self.ds[var_name]

        if len(field.dims) == 3:
            if not any([s is not None, eta is not None, xi is not None]):
                raise ValueError(
                    "Invalid input: For 3D fields, you must specify at least one of the dimensions 's', 'eta', or 'xi'."
                )
            if all([s is not None, eta is not None, xi is not None]):
                raise ValueError(
                    "Ambiguous input: For 3D fields, specify at most two of 's', 'eta', or 'xi'. Specifying all three is not allowed."
                )

        if len(field.dims) == 2 and all([eta is not None, xi is not None]):
            raise ValueError(
                "Conflicting input: For 2D fields, specify only one dimension, either 'eta' or 'xi', not both."
            )

        # Load the data
        if self.use_dask:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                field.load()

        # Get correct mask and spatial coordinates
        if all(dim in field.dims for dim in ["eta_rho", "xi_rho"]):
            loc = "rho"
        elif all(dim in field.dims for dim in ["eta_rho", "xi_u"]):
            loc = "u"
        elif all(dim in field.dims for dim in ["eta_v", "xi_rho"]):
            loc = "v"
        else:
            ValueError("provided field does not have two horizontal dimension")

        mask = self.grid.ds[f"mask_{loc}"]
        lat_deg = self.grid.ds[f"lat_{loc}"]
        lon_deg = self.grid.ds[f"lon_{loc}"]

        if self.grid.straddle:
            lon_deg = xr.where(lon_deg > 180, lon_deg - 360, lon_deg)

        field = field.assign_coords({"lon": lon_deg, "lat": lat_deg})

        # Retrieve depth coordinates
        compute_layer_depth = (depth_contours or s is None) and len(field.dims) > 2
        compute_interface_depth = layer_contours and s is None

        if compute_layer_depth:
            layer_depth = compute_depth_coordinates(
                self.grid.ds,
                self.ds.zeta.isel(time=time),
                depth_type="layer",
                location=loc,
                eta=eta,
                xi=xi,
            )
            if s is not None:
                layer_depth = layer_depth.isel(s_rho=s)
        if compute_interface_depth:
            interface_depth = compute_depth_coordinates(
                self.grid.ds,
                self.ds.zeta.isel(time=time),
                depth_type="interface",
                location=loc,
                eta=eta,
                xi=xi,
            )
            if s is not None:
                interface_depth = interface_depth.isel(s_w=s)

        # Slice the field as desired
        title = field.long_name
        if s is not None:
            title = title + f", s_rho = {field.s_rho[s].item()}"
            field = field.isel(s_rho=s)
        else:
            depth_contours = False

        def _process_dimension(field, mask, dim_name, dim_values, idx, title):
            if dim_name in field.dims:
                title = title + f", {dim_name} = {dim_values[idx].item()}"
                field = field.isel(**{dim_name: idx})
                mask = mask.isel(**{dim_name: idx})
            else:
                raise ValueError(
                    f"None of the expected dimensions ({dim_name}) found in field."
                )
            return field, mask, title

        if eta is not None:
            field, mask, title = _process_dimension(
                field,
                mask,
                "eta_rho" if "eta_rho" in field.dims else "eta_v",
                field.eta_rho if "eta_rho" in field.dims else field.eta_v,
                eta,
                title,
            )

        if xi is not None:
            field, mask, title = _process_dimension(
                field,
                mask,
                "xi_rho" if "xi_rho" in field.dims else "xi_u",
                field.xi_rho if "xi_rho" in field.dims else field.xi_u,
                xi,
                title,
            )

        # Format to exclude seconds
        formatted_time = np.datetime_as_string(field.abs_time.values, unit="m")
        title = title + f", time: {formatted_time}"

        if compute_layer_depth:
            field = field.assign_coords({"layer_depth": layer_depth})

        if not include_boundary:
            slice_dict = None

            if eta is None and xi is None:
                slice_dict = {
                    "rho": {"eta_rho": slice(1, -1), "xi_rho": slice(1, -1)},
                    "u": {"eta_rho": slice(1, -1), "xi_u": slice(1, -1)},
                    "v": {"eta_v": slice(1, -1), "xi_rho": slice(1, -1)},
                }
            elif eta is None:
                slice_dict = {
                    "rho": {"eta_rho": slice(1, -1)},
                    "u": {"eta_rho": slice(1, -1)},
                    "v": {"eta_v": slice(1, -1)},
                }
            elif xi is None:
                slice_dict = {
                    "rho": {"xi_rho": slice(1, -1)},
                    "u": {"xi_u": slice(1, -1)},
                    "v": {"xi_rho": slice(1, -1)},
                }
            if slice_dict is not None:
                if loc in slice_dict:
                    field = field.isel(**slice_dict[loc])
                    mask = mask.isel(**slice_dict[loc])

        # Choose colorbar
        if var_name in ["u", "v", "w", "ubar", "vbar", "zeta"]:
            vmax = max(field.where(mask).max().values, -field.where(mask).min().values)
            vmin = -vmax
            cmap = plt.colormaps.get_cmap("RdBu_r")
        else:
            vmax = field.where(mask).max().values
            vmin = field.where(mask).min().values
            if var_name in ["temp", "salt"]:
                cmap = plt.colormaps.get_cmap("YlOrRd")
            else:
                cmap = plt.colormaps.get_cmap("YlGn")
        cmap.set_bad(color="gray")
        kwargs = {"vmax": vmax, "vmin": vmin, "cmap": cmap}

        # Plotting
        if eta is None and xi is None:
            _plot(
                field=field.where(mask),
                depth_contours=depth_contours,
                title=title,
                kwargs=kwargs,
                c="g",
            )
        else:
            if len(field.dims) == 2:
                if not layer_contours:
                    interface_depth = None
                else:
                    # restrict number of layer_contours to 10 for the sake of plot clearity
                    nr_layers = len(interface_depth["s_w"])
                    selected_layers = np.linspace(
                        0, nr_layers - 1, min(nr_layers, 10), dtype=int
                    )
                    interface_depth = interface_depth.isel(s_w=selected_layers)
                _section_plot(
                    field.where(mask),
                    interface_depth=interface_depth,
                    title=title,
                    kwargs=kwargs,
                    ax=ax,
                )
            else:
                if "s_rho" in field.dims:
                    _profile_plot(field.where(mask), title=title, ax=ax)
                else:
                    _line_plot(field.where(mask), title=title, ax=ax)

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

        Notes
        -----
        This method uses the `compute_depth_coordinates` function to perform calculations and updates.
        """

        for location in locations:
            self.ds_depth_coords[
                f"{depth_type}_depth_{location}"
            ] = compute_depth_coordinates(
                self.grid.ds, self.ds.zeta, depth_type, location
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
                    object.__setattr__(self, "model_reference_date", inferred_date)
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
        """Add latitude and longitude coordinates to the dataset.

        Adds "lat_rho" and "lon_rho" from the grid object to the dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to update.

        Returns
        -------
        xarray.Dataset
            Dataset with "lat_rho" and "lon_rho" coordinates added.
        """
        ds = ds.assign_coords(
            {"lat_rho": self.grid.ds["lat_rho"], "lon_rho": self.grid.ds["lon_rho"]}
        )

        return ds
