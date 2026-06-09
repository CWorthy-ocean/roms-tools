import warnings

import numpy as np
import xarray as xr
import xgcm


def _xesmf_extrap_kwargs(method: str | None, kwargs: dict | None) -> dict:
    """Map friendly extrapolation kwargs to ``xe.Regridder`` ``extrap_*`` kwargs.

    Both the destination-side extrapolation in :class:`LateralRegridToROMS` and
    the source-on-source fill in ``LatLonDataset.apply_source_fill`` accept the
    same user-facing keys; this centralizes the translation to xESMF's
    ``extrap_num_levels`` / ``extrap_num_src_pnts`` / ``extrap_dist_exponent``.

    Parameters
    ----------
    method : str or None
        The xESMF extrapolation method (``"inverse_dist"``, ``"nearest_s2d"``,
        ``"creep_fill"``). ``None`` returns an empty mapping.
    kwargs : dict or None
        User-facing kwargs: ``num_levels`` (creep_fill), ``num_src_pnts`` /
        ``dist_exponent`` (inverse_dist). ``None``/missing values are skipped so
        xESMF defaults apply.

    Returns
    -------
    dict
        Keyword arguments to splat into ``xe.Regridder(...)``.
    """
    kwargs = kwargs or {}
    out: dict = {}
    if method == "creep_fill":
        if kwargs.get("num_levels") is not None:
            out["extrap_num_levels"] = kwargs["num_levels"]
    elif method == "inverse_dist":
        if kwargs.get("num_src_pnts") is not None:
            out["extrap_num_src_pnts"] = kwargs["num_src_pnts"]
        if kwargs.get("dist_exponent") is not None:
            out["extrap_dist_exponent"] = kwargs["dist_exponent"]
    return out


class LateralRegridToROMS:
    """Handles lateral regridding of data onto a ROMS grid.

    Two modes are supported:

    * **scipy interpolation (default, ``use_xesmf=False``):** uses
      :meth:`xarray.DataArray.interp` with the requested ``method``. This is the
      historical behavior and requires no optional dependencies. The caller is
      responsible for filling NaNs in the source beforehand (otherwise NaNs
      propagate into the regridded field).
    * **xESMF bilinear (``use_xesmf=True``):** uses ``xesmf`` bilinear
      interpolation with an ``extrap_method`` (default ``"inverse_dist"``) to
      fill targets whose source neighbors are all masked/out of range, producing
      NaN-free output without any separate fill step. When a ``source_mask`` is
      supplied (for
      sources that still contain land NaNs, e.g. GLORYS), it is applied as
      ``mask_in`` so the bilinear weights are renormalized over only the valid
      ocean cells; when omitted (for already NaN-free sources), plain bilinear
      is used. Requires ``xesmf`` (install ``roms-tools`` via conda).
    """

    def __init__(
        self,
        target_coords,
        source_dim_names,
        source_ds=None,
        use_xesmf=False,
        source_mask=None,
        extrap_method="inverse_dist",
        extrap_kwargs=None,
    ):
        """Initialize the regridder.

        Parameters
        ----------
        target_coords : dict
            Dictionary containing 'lon' and 'lat' as xarray.DataArrays representing
            the longitude and latitude values of the target grid.
        source_dim_names : dict
            Dictionary specifying names for the latitude and longitude dimensions,
            typically using keys "latitude" and "longitude".
        source_ds : xarray.Dataset, optional
            The source dataset. Required when ``use_xesmf=True`` (used for the
            source latitude/longitude coordinates).
        use_xesmf : bool, optional
            If True, use xESMF masked bilinear regridding with nearest-neighbor
            extrapolation. If False (default), use scipy interpolation via
            :meth:`xarray.DataArray.interp`.
        source_mask : xarray.DataArray, optional
            A 2D source ocean mask (1 = valid/ocean, 0 = masked/land) with the
            source latitude/longitude dimensions, used as ``mask_in`` in the
            xESMF path so bilinear weights are renormalized over valid ocean
            cells. Pass the field-appropriate mask (e.g. ``ds["mask"]`` for
            tracers, ``ds["mask_vel"]`` for velocities) for sources that still
            contain land NaNs. When omitted, plain (unmasked) bilinear is used —
            appropriate for sources that are already NaN-free (e.g. a source
            laterally filled during preprocessing).
        extrap_method : str, optional
            xESMF extrapolation method used to fill target points whose source
            neighbors are all masked/out of range, guaranteeing NaN-free output.
            Options are ``"inverse_dist"`` (default; inverse-distance-weighted
            average of the nearest source points, giving smoothly varying values)
            and ``"nearest_s2d"`` (single nearest source point). Only used in the
            xESMF path. Pass ``None`` to disable extrapolation (plain bilinear),
            e.g. when the source has already been pre-filled and is NaN-free.
        extrap_kwargs : dict, optional
            Method-specific tuning for ``extrap_method``: ``num_src_pnts`` /
            ``dist_exponent`` for ``"inverse_dist"``, ``num_levels`` for
            ``"creep_fill"``. Translated to xESMF's ``extrap_*`` kwargs via
            :func:`_xesmf_extrap_kwargs`. Only used in the xESMF path.

        Raises
        ------
        ImportError
            If ``use_xesmf=True`` but xESMF is not installed.
        ValueError
            If ``use_xesmf=True`` but ``source_ds`` is not provided.
        """
        self.use_xesmf = use_xesmf

        if not use_xesmf:
            # Backward-compatible scipy interpolation path.
            self.coords = {
                source_dim_names["latitude"]: target_coords["lat"],
                source_dim_names["longitude"]: target_coords["lon"],
            }
            return

        if source_ds is None:
            raise ValueError(
                "source_ds must be provided when use_xesmf=True (needed for the "
                "source latitude/longitude coordinates)."
            )
        try:
            import xesmf as xe
        except ImportError:
            raise ImportError(
                "xesmf is required for masked regridding. Please install `roms-tools` via conda, which includes xesmf."
            )

        lat_name = source_dim_names["latitude"]
        lon_name = source_dim_names["longitude"]
        source_lat = np.asarray(source_ds[lat_name].values)
        source_lon = np.asarray(source_ds[lon_name].values)

        ds_in = xr.Dataset(
            {
                "lat": xr.DataArray(source_lat, dims="lat"),
                "lon": xr.DataArray(source_lon, dims="lon"),
            }
        )

        # xESMF reads masking from a variable named "mask" in the input dataset
        # (1 = valid, 0 = masked); masked cells are excluded and the bilinear
        # weights are renormalized over the remaining valid cells. We add it only
        # when the caller supplies a ``source_mask`` (sources that still contain
        # land NaNs, e.g. GLORYS/CESM). Sources that are already NaN-free (e.g.
        # the pre-filled UNIFIED BGC dataset) pass ``source_mask=None`` and use
        # plain bilinear, which is equivalent to an all-ocean mask.
        if source_mask is not None:
            # Rebuild as a bare (lat, lon) DataArray so its coordinates can't
            # conflict with the explicit lat/lon axes assigned to ``ds_in``.
            ds_in["mask"] = xr.DataArray(
                source_mask.transpose(lat_name, lon_name).values.astype(np.int32),
                dims=("lat", "lon"),
            )

        # A 1D target (e.g. a ROMS boundary line where lat and lon share a single
        # dimension) is a line of points. We do NOT use ``locstream_out=True``
        # here: xESMF silently ignores ``extrap_method`` for locstream output, so
        # boundary points whose source neighbors are all masked would be left as
        # NaN. Instead we promote the line to a (1, N) structured grid (adding a
        # singleton ``_dummy_y`` dimension), which keeps nearest-neighbor
        # extrapolation working, and squeeze that dimension back out in ``apply``.
        # A 2D (curvilinear) target is already a structured grid.
        self._dummy_dim = None
        ds_out = xr.Dataset()
        if target_coords["lat"].ndim == 1:
            self._dummy_dim = "_dummy_y"
            line_dim = target_coords["lat"].dims[0]
            ds_out["lat"] = xr.DataArray(
                target_coords["lat"].values[None, :], dims=(self._dummy_dim, line_dim)
            )
            ds_out["lon"] = xr.DataArray(
                target_coords["lon"].values[None, :], dims=(self._dummy_dim, line_dim)
            )
        else:
            ds_out["lat"] = target_coords["lat"]
            ds_out["lon"] = target_coords["lon"]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="xesmf")
            self.regridder = xe.Regridder(
                ds_in,
                ds_out,
                method="bilinear",
                unmapped_to_nan=True,
                extrap_method=extrap_method,
                **_xesmf_extrap_kwargs(extrap_method, extrap_kwargs),
            )

    def apply(self, da, method="linear"):
        """Regrid the provided data array.

        Parameters
        ----------
        da : xarray.DataArray
            The data array to regrid.
        method : str, optional
            Interpolation method for the scipy path. Ignored when
            ``use_xesmf=True``. Default is "linear".

        Returns
        -------
        xarray.DataArray
            The regridded data array.
        """
        if self.use_xesmf:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="xesmf")
                regridded = self.regridder(da, keep_attrs=True)
            if self._dummy_dim is not None:
                # Remove the singleton dimension added to promote a 1D boundary
                # line to a (1, N) structured target grid.
                regridded = regridded.isel({self._dummy_dim: 0})
            return regridded

        return da.interp(self.coords, method=method).drop_vars(list(self.coords.keys()))


class LateralRegridFromROMS:
    """Regrids data from a curvilinear ROMS grid onto latitude-longitude coordinates
    using xESMF.

    It requires the `xesmf` library, which can be installed by installing `roms-tools` via conda.

    Parameters
    ----------
    source_grid_ds : xarray.Dataset
        The source dataset containing the curvilinear ROMS grid with 'lat_rho' and 'lon_rho'.

    target_coords : dict
        A dictionary containing 'lat' and 'lon' arrays representing the target
        latitude and longitude coordinates for regridding.

    method : str, optional
        The regridding method to use. Default is "bilinear". Other options include "nearest_s2d" and "conservative".

    Raises
    ------
    ImportError
        If xESMF is not installed.
    """

    def __init__(self, ds_in, target_coords, method="bilinear"):
        """Initializes the regridder with the source and target grids.

        Parameters
        ----------
        ds_in : xarray.Dataset or xarray.DataArray
            The source dataset or dataarray containing the curvilinear ROMS grid with coordinates 'lat' and 'lon'.

        target_coords : dict
            A dictionary containing 'lat' and 'lon' arrays representing the target latitude
            and longitude coordinates for regridding.

        method : str, optional
            The regridding method to use. Default is "bilinear". Other options include
            "nearest_s2d" and "conservative".

        Raises
        ------
        ImportError
            If xESMF is not installed.
        """
        try:
            import xesmf as xe

        except ImportError:
            raise ImportError(
                "xesmf is required for this regridding task. Please install `roms-tools` via conda, which includes xesmf."
            )

        ds_out = xr.Dataset()
        ds_out["lat"] = target_coords["lat"]
        ds_out["lon"] = target_coords["lon"]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="xesmf")
            self.regridder = xe.Regridder(
                ds_in, ds_out, method=method, unmapped_to_nan=True
            )

    def apply(self, da):
        """Applies the regridding to the provided data array.

        Parameters
        ----------
        da : xarray.DataArray
            The data array to regrid. This should have the same dimension names as the
            source grid (e.g., 'lat' and 'lon').

        Returns
        -------
        xarray.DataArray
            The regridded data array.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="xesmf")
            regridded = self.regridder(da, keep_attrs=True)
        return regridded

    def destroy(self):
        """Release memory held by the underlying regridder.

        Call this when the regridder is no longer needed to avoid accumulating
        weight matrices across many regridding operations.
        """
        try:
            del self.regridder.weights
        except Exception:
            pass
        try:
            del self.regridder
        except Exception:
            pass


class VerticalRegrid:
    """Regrid ROMS variables along the vertical.

    This class uses the `xgcm` package. Both the source and target coordinates can vary spatially
    (i.e., they can be 3D fields).

    Attributes
    ----------
    grid : xgcm.Grid
        The XGCM grid object used for vertical regridding, initialized with the input dataset `ds`.
    """

    def __init__(self, ds: "xr.Dataset", source_dim: str):
        """Initialize the VerticalRegrid object with a ROMS dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            A ROMS source dataset containing the vertical coordinate `s_rho`.
        source_dim : str
            Name of the vertical source dimension in the dataset.
        """
        self.grid = xgcm.Grid(
            ds,
            coords={source_dim: {"center": source_dim}},
            periodic=False,
            autoparse_metadata=False,
        )
        self.source_dim = source_dim

    def apply(
        self,
        da: "xr.DataArray",
        source_depth_coords: "xr.DataArray",
        target_depth_coords: "xr.DataArray",
        mask_edges: bool = False,
    ) -> "xr.DataArray":
        """Regrid a ROMS variable from source vertical coordinates to target vertical coordinates.

        This method supports spatially varying vertical coordinates for both source and target,
        meaning that the depth levels can vary across the horizontal grid.

        Parameters
        ----------
        da : xarray.DataArray
            The data array to regrid. Must have a vertical dimension corresponding to `self.source_dim.

        source_depth_coords : array-like (1D or 3D)
            Depth coordinates of the source data. Can be a 1D array (same for all horizontal points)
            or a 3D array (e.g., terrain-following coordinate).

        target_depth_coords : array-like (1D or 3D)
            Desired depth coordinates of the regridded data. Can also be 1D or 3D.

        mask_edges : bool, optional
            If True, target values outside the range of source depth coordinates are masked with NaN.
            Defaults to False.

        Returns
        -------
        xarray.DataArray
            A new `DataArray` containing the regridded variable at the target depth coordinates.
        """
        target_dim = None
        dims = ["s_w", "s_rho"]
        for dim in dims:
            if dim in target_depth_coords.dims:
                target_dim = dim

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="xgcm")
            transformed = self.grid.transform(
                da,
                self.source_dim,
                target=target_depth_coords,
                target_data=source_depth_coords,
                target_dim=target_dim,
                mask_edges=mask_edges,
            )

        return transformed
