import warnings

import xarray as xr
import xgcm


class LateralRegridToROMS:
    """Handles lateral regridding of data onto a new spatial grid."""

    def __init__(self, target_coords, source_dim_names):
        """Initialize target grid coordinates and names for grid dimensions.

        Parameters
        ----------
        target_coords : dict
            Dictionary containing 'lon' and 'lat' as xarray.DataArrays representing
            the longitude and latitude values of the target grid.
        source_dim_names : dict
            Dictionary specifying names for the latitude and longitude dimensions,
            typically using keys like "latitude" and "longitude" to align with the dataset conventions.

        Attributes
        ----------
        coords : dict
            Maps the dimension names to the corresponding latitude and longitude
            DataArrays, providing easy access to target grid coordinates.
        """
        self.coords = {
            source_dim_names["latitude"]: target_coords["lat"],
            source_dim_names["longitude"]: target_coords["lon"],
        }

    def apply(self, da, method="linear"):
        """Fills missing values and regrids the variable.

        Parameters
        ----------
        da : xarray.DataArray
            Input data to fill and regrid.
        method : str
            Interpolation method to use.

        Returns
        -------
        xarray.DataArray
            Regridded data with filled values.
        """
        regridded = da.interp(self.coords, method=method).drop_vars(
            list(self.coords.keys())
        )
        return regridded


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


class VerticalRegridToROMS:
    """Interpolates data onto new vertical (depth) coordinates.

    Parameters
    ----------
    target_depth_coords : xarray.DataArray
        Depth coordinates for the target grid.
    source_depth_coords : xarray.DataArray
        Depth coordinates for the source grid.
    """

    def __init__(self, target_depth_coords, source_depth_coords):
        """Initialize regridding factors for interpolation.

        Parameters
        ----------
        target_depth_coords : xarray.DataArray
            Depth coordinates for the target grid.
        source_depth_coords : xarray.DataArray
            Depth coordinates for the source grid.

        Attributes
        ----------
        coeff : xarray.Dataset
            Dataset containing:
            - `is_below` : Boolean mask for depths just below target.
            - `is_above` : Boolean mask for depths just above target.
            - `upper_mask`, `lower_mask` : Masks for valid interpolation bounds.
            - `factor` : Weight for blending values between levels.
        """
        self.depth_dim = source_depth_coords.dims[0]
        source_depth = source_depth_coords
        dims = {"dim": self.depth_dim}

        dlev = source_depth - target_depth_coords
        is_below = dlev == dlev.where(dlev >= 0).min(**dims)
        is_above = dlev == dlev.where(dlev <= 0).max(**dims)
        p_below = dlev.where(is_below).sum(**dims)
        p_above = -dlev.where(is_above).sum(**dims)
        denominator = p_below + p_above
        denominator = denominator.where(denominator > 1e-6, 1e-6)
        factor = p_below / denominator

        upper_mask = is_above.sum(**dims) > 0
        lower_mask = is_below.sum(**dims) > 0

        self.coeff = xr.Dataset(
            {
                "is_below": is_below,
                "is_above": is_above,
                "upper_mask": upper_mask,
                "lower_mask": lower_mask,
                "factor": factor,
            }
        )

    def apply(self, da, fill_nans=True):
        """Interpolates the variable onto the new depth grid using precomputed
        coefficients for linear interpolation between layers.

        Parameters
        ----------
        da : xarray.DataArray
            The input data to be regridded along the depth dimension. This should be
            an array with the same depth coordinates as the original grid.
        fill_nans : bool, optional
            Whether to fill NaN values in the regridded data. If True (default),
            forward-fill and backward-fill are applied along the 's_rho' dimension to
            ensure there are no NaNs after interpolation.

        Returns
        -------
        xarray.DataArray
            The regridded data array, interpolated onto the new depth grid. NaN values
            are replaced if `fill_nans=True`, with extrapolation allowed at the surface
            and bottom layers to minimize gaps.
        """
        dims = {"dim": self.depth_dim}

        da_below = da.where(self.coeff["is_below"]).sum(**dims)
        da_above = da.where(self.coeff["is_above"]).sum(**dims)

        result = da_below + (da_above - da_below) * self.coeff["factor"]
        if fill_nans:
            result = result.where(self.coeff["upper_mask"], da.isel({dims["dim"]: 0}))
            result = result.where(self.coeff["lower_mask"], da.isel({dims["dim"]: -1}))
        else:
            result = result.where(self.coeff["upper_mask"]).where(
                self.coeff["lower_mask"]
            )

        return result


class VerticalRegridFromROMS:
    """A class for regridding data from the ROMS vertical coordinate system to target
    depth levels.

    This class uses the `xgcm` package to perform the transformation from the ROMS depth coordinates to
    a user-defined set of target depth levels. It assumes that the input dataset `ds` contains the necessary
    vertical coordinate information (`s_rho`).

    Attributes
    ----------
    grid : xgcm.Grid
        The grid object used for regridding, initialized with the given dataset `ds`.
    """

    def __init__(self, ds):
        """Initializes the `VerticalRegridFromROMS` object by creating an `xgcm.Grid`
        instance.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset containing the ROMS output data, which must include the vertical coordinate `s_rho`.
        """
        self.grid = xgcm.Grid(
            ds,
            coords={"s_rho": {"center": "s_rho"}},
            periodic=False,
            autoparse_metadata=False,
        )

    def apply(self, da, depth_coords, target_depth_levels, mask_edges=True):
        """Applies vertical regridding from ROMS to the specified target depth levels.

        This method transforms the input data array `da` from the ROMS vertical coordinate (`s_rho`)
        to a set of target depth levels defined by `target_depth_levels`.

        Parameters
        ----------
        da : xarray.DataArray
            The data array containing the ROMS output field to be regridded. It must have a vertical
            dimension corresponding to `s_rho`.

        depth_coords : array-like
            The depth coordinates of the input data array `da` (typically the `s_rho` coordinate in ROMS).

        target_depth_levels : array-like
            The target depth levels to which the input data `da` will be regridded.

        mask_edges: bool, optional
            If activated, target values outside the range of depth_coords are masked with nan. Defaults to True.

        Returns
        -------
        xarray.DataArray
            A new `xarray.DataArray` containing the regridded data at the specified target depth levels.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="xgcm")
            transformed = self.grid.transform(
                da,
                "s_rho",
                target_depth_levels,
                target_data=depth_coords,
                mask_edges=mask_edges,
            )

        return transformed


class VerticalRegrid:
    """Regrid ROMS variables along the vertical, using spatially varying coordinates.

    This class uses the `xgcm` package to transform data from a ROMS vertical coordinate
    system (`s_rho`) to a user-defined set of target depth levels, where both the source
    and target coordinates can vary spatially (i.e., 2D fields in horizontal space).

    Attributes
    ----------
    grid : xgcm.Grid
        The XGCM grid object used for vertical regridding, initialized with the input dataset `ds`.
    """

    def __init__(self, ds: "xr.Dataset"):
        """Initialize the VerticalRegrid object with a ROMS dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            The ROMS dataset containing the vertical coordinate `s_rho` and the variable(s)
            to be regridded.
        """
        self.grid = xgcm.Grid(
            ds,
            coords={"s_rho": {"center": "s_rho"}},
            periodic=False,
            autoparse_metadata=False,
        )

    def apply(
        self,
        da: "xr.DataArray",
        source_depth_coords: "xr.DataArray",
        target_depth_coords: "xr.DataArray",
        mask_edges: bool = True,
    ) -> "xr.DataArray":
        """Regrid a ROMS variable from source vertical coordinates to target vertical coordinates.

        This method supports spatially varying vertical coordinates for both source and target,
        meaning that the depth levels can vary across the horizontal grid.

        Parameters
        ----------
        da : xarray.DataArray
            The data array to regrid. Must have a vertical dimension corresponding to `s_rho`.

        source_depth_coords : array-like (1D or 2D)
            Depth coordinates of the source data. Can be a 1D array (same for all horizontal points)
            or a 2D array (varying in horizontal space).

        target_depth_coords : array-like (1D or 2D)
            Desired depth coordinates of the regridded data. Can also be 1D or 2D.

        mask_edges : bool, optional
            If True, target values outside the range of source depth coordinates are masked with NaN.
            Defaults to True.

        Returns
        -------
        xarray.DataArray
            A new `DataArray` containing the regridded variable at the target depth coordinates.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="xgcm")
            transformed = self.grid.transform(
                da,
                "s_rho",
                target=target_depth_coords,
                target_data=source_depth_coords,
                target_dim="s_rho",
                mask_edges=mask_edges,
            )

        return transformed
