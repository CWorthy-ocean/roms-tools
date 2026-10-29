import logging

import numpy as np
import pyamg
import xarray as xr
from scipy import sparse

from roms_tools.constants import R_EARTH
from roms_tools.setup.utils import build_kdtree_from_latlon, latlon_to_xyz


class LateralFill:
    """
    Fill NaN values in a 2D field using an iterative lateral diffusion (Poisson) solver.

    The fill is performed along two horizontal dimensions. The **order of these
    dimensions is significant** and must match the order of the dimensions of the
    input data passed to :meth:`apply`.
    """

    def __init__(self, mask: xr.DataArray, dims: tuple[str, str], tol: float = 1.0e-4):
        """
        Initialize a lateral fill operator.

        Parameters
        ----------
        mask : xarray.DataArray
            A 2D boolean mask defining valid points (True) and masked points (False).
            The mask dimensions **must be ordered consistently with `dims`**.
        dims : tuple of str
            The two horizontal dimensions along which the fill is applied.
            **Order matters** and must match the dimension order of both `mask`
            and the data passed to :meth:`apply` (e.g., ``("eta_rho", "xi_rho")``).
        tol : float, optional
            Convergence tolerance for the iterative solver. Default is ``1.0e-4``.

        Raises
        ------
        ValueError
            If the mask dimensionality or dimension order is inconsistent with `dims`.
        NotImplementedError
            If the mask is not two-dimensional.
        """
        # Type check
        if not isinstance(dims, tuple):
            raise TypeError(
                f"LateralFill error: 'dims' must be a tuple of two strings, got {type(dims).__name__}."
            )
        if len(dims) != 2:
            raise ValueError(
                f"LateralFill error: 'dims' must contain exactly two dimension names, got {len(dims)}: {dims}"
            )

        if len(mask.shape) != 2:
            raise NotImplementedError("LateralFill currently supports only 2D masks.")

        _check_dimension_match(dims, mask)

        self.dims = dims
        self.mask = mask

        # Ensure the mask is 2D, copy it and set boundary values to True
        mask = mask.copy()
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        # Flatten the mask for use in the sparse matrix solver
        mask_flat = mask.values.flatten()

        # Create a sparse matrix representing the Laplacian operator for the diffusion process
        A = laplacian(mask.shape, mask_flat, format="csr")

        # Use algebraic multigrid solver for solving the Poisson equation with set seed to ensure reproducibility
        np.random.seed(123089)
        self.ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
        self.tol = tol

    def apply(self, var):
        """Fills NaN values in an xarray DataArray using iterative lateral diffusion.

        Parameters
        ----------
        var : xarray.DataArray
            Input DataArray with NaN values to be filled. The fill is performed
            across the dimensions specified by `dims`.

        Returns
        -------
        var_filled : xarray.DataArray
            A DataArray with NaN values filled by iterative smoothing, while preserving
            non-NaN values.
        """
        _check_dimension_match(self.dims, var)

        # Apply fill to anomaly field
        mean = var.where(self.mask).mean(dim=self.dims, skipna=True)
        var = var - mean

        # Setup the right-hand side (RHS): ocean points take their original values, land points are set to 0
        b = xr.where(self.mask, var, 0)

        # Initial guess: ocean points take their original values, land points are set to 0
        x0 = xr.where(self.mask, var, 0)

        # Apply the iterative solver using a custom NumPy function
        var_filled = xr.apply_ufunc(
            _lateral_fill_np_array,
            x0,
            b,
            input_core_dims=[self.dims, self.dims],
            output_core_dims=[self.dims],
            output_dtypes=[x0.dtype],
            dask="parallelized",
            vectorize=True,
            kwargs={"ml": self.ml, "tol": self.tol},
        )

        var_filled = var_filled + mean

        return var_filled


def _lateral_fill_np_array(x0, b, ml, tol=1.0e-4):
    """Fills all NaN values in a 2D NumPy array using an iterative solver, while
    preserving the existing non-NaN values. The filling process uses an AMG solver to
    efficiently perform smoothing based on the Laplace operator.

    Parameters
    ----------
    x0 : numpy.ndarray
        Initial guess for the fill operation.
    b : numpy.ndarray
        Right-hand side (RHS) of the equation representing the data values
        to be used in the fill process. Non-NaN values in `b` correspond to
        valid points, and zeros are used for masked (invalid) points.
    ml : pyamg.MultilevelSolver
        An algebraic multigrid (AMG) solver used to iteratively fill NaNs
        via a smoothing process.
    tol : float, optional, default=1.0e-4
        Convergence tolerance for the iterative solver. The filling process
        stops when the relative residual (change in values) is less than or
        equal to `tol`. Specifically, the process iterates until:
        ``||Ax - b|| / ||Ax0 - b|| < tol``, where `A` is the system matrix,
        `x` is the solution, and `x0` is the initial guess.

    Returns
    -------
    x_2d : numpy.ndarray
        The filled 2D array where NaN values have been replaced with iteratively
        computed values, and non-NaN values remain unchanged.
    """
    if np.isnan(x0).any():
        raise ValueError(
            "LateralFill error: The fill operation cannot proceed because the input field contains NaNs at grid points marked as valid by the mask."
        )
    b_flat = b.flatten()
    x0_flat = x0.flatten()
    x = ml.solve(b_flat, x0_flat, tol=tol)
    x_2d = x.reshape(b.shape)

    return x_2d


def laplacian(grid, mask, dtype=float, format=None):
    """Return a sparse matrix for solving a 2-dimensional Poisson problem.

    This function generates a finite difference approximation of the Laplacian operator
    on a 2-dimensional grid with unit grid spacing and Dirichlet boundary conditions.
    The matrix can be used to solve Poisson-like equations in grid-based numerical methods.

    The computation iterates over the last dimension first (z, then y, then x), and
    the output matrix should be compatible with `np.mgrid()` or `np.ndenumerate()`.

    Parameters
    ----------
    grid : tuple of int
        Dimensions of the grid, e.g., (100, 100).
    mask : 2D array of bool
        A boolean mask of the same size as the grid, indicating valid grid points
        (True for valid points, False for masked points).
    dtype : data-type, optional
        The desired data type of the resulting matrix. Default is `float`.
    format : str, optional
        The format of the sparse matrix to return, such as "csr", "coo", etc.
        Default is None.

    Returns
    -------
    sparse matrix
        A sparse matrix representing the finite difference Laplacian operator for
        the given grid.
    """
    grid = tuple(grid)

    # create 2-dimensional Laplacian stencil
    N = 2
    stencil = np.zeros((3,) * N, dtype=dtype)
    for i in range(N):
        stencil[(1,) * i + (0,) + (1,) * (N - i - 1)] = 1
        stencil[(1,) * i + (2,) + (1,) * (N - i - 1)] = 1
    stencil[(1,) * N] = -2 * N

    return stencil_grid_mod(stencil, grid, mask, format=format)


def stencil_grid_mod(S, grid, msk, dtype=None, format=None):
    """Construct a sparse matrix from a local matrix stencil.

    This function generates a sparse matrix that represents an operator
    by applying the given stencil `S` at each vertex of a regular grid with
    the specified dimensions. The matrix is modified according to the provided
    mask to ensure that masked points are not affected during matrix operations.

    Parameters
    ----------
    S : ndarray
        An N-dimensional array representing the local matrix stencil.
        All dimensions of `S` must be odd.
    grid : tuple of int
        A tuple specifying the dimensions of the grid. The length of the tuple
        should match the number of dimensions of the stencil `S`.
    msk : ndarray of bool
        A 1D boolean array where `True` indicates points that are masked
        (i.e., should not be affected by the matrix).
    dtype : data-type, optional
        The data type of the resulting sparse matrix. Default is `None`, which
        will infer the type from `S`.
    format : str, optional
        The sparse matrix format to return, such as "csr", "coo", etc. If not
        specified, the default is DIA (diagonal) format.

    Returns
    -------
    A : sparse matrix
        A sparse matrix representing the operator formed by applying the stencil
        `S` at each grid vertex. The matrix is modified based on the mask so that
        masked points are unaffected by the operator.

    Notes
    -----
    The grid vertices are enumerated as `arange(prod(grid)).reshape(grid)`.
    This means the last grid dimension cycles fastest, while the first dimension
    cycles slowest. For example, if `grid=(2,3)`, then the grid vertices are ordered
    as (0,0), (0,1), (0,2), (1,0), (1,1), (1,2).

    This ordering is consistent with the NumPy functions `ndenumerate()` and `mgrid()`.

    The stencil is applied in all directions, and boundary conditions are
    respected by zeroing out connections to boundary points.
    """
    S = np.asarray(S, dtype=dtype)
    grid = tuple(grid)

    if not (np.asarray(S.shape) % 2 == 1).all():
        raise ValueError("all stencil dimensions must be odd")

    if len(grid) != np.ndim(S):
        raise ValueError(
            "stencil dimension must equal number of grid\
                          dimensions"
        )

    if min(grid) < 1:
        raise ValueError("grid dimensions must be positive")

    N_v = np.prod(grid)  # number of vertices in the mesh
    N_s = (S != 0).sum()  # number of nonzero stencil entries

    # diagonal offsets
    diags = np.zeros(N_s, dtype=int)

    # compute index offset of each dof within the stencil
    strides = np.cumprod([1] + list(reversed(grid)))[:-1]  # noqa: RUF005
    indices = tuple(i.copy() for i in S.nonzero())
    for i, s in zip(indices, S.shape):
        i -= s // 2

    for stride, coords in zip(strides, reversed(indices)):
        diags += stride * coords

    data = S[S != 0].repeat(N_v).reshape(N_s, N_v)

    indices = np.vstack(indices).T

    # zero boundary connections
    for index, diag in zip(indices, data):
        diag = diag.reshape(grid)
        for n, i in enumerate(index):
            if i > 0:
                s = [slice(None)] * len(grid)
                s[n] = slice(0, i)
                s = tuple(s)
                diag[s] = 0
            elif i < 0:
                s = [slice(None)] * len(grid)
                s[n] = slice(i, None)
                s = tuple(s)
                diag[s] = 0

    # remove diagonals that lie outside matrix
    mask = abs(diags) < N_v
    if not mask.all():
        diags = diags[mask]
        data = data[mask]

    # sum duplicate diagonals
    if len(np.unique(diags)) != len(diags):
        new_diags = np.unique(diags)
        new_data = np.zeros((len(new_diags), data.shape[1]), dtype=data.dtype)

        for dia, dat in zip(diags, data):
            n = np.searchsorted(new_diags, dia)
            new_data[n, :] += dat

        diags = new_diags
        data = new_data

    # Modify the data vectors so that masked points are not affected by the matrix solve.
    # The modifications to the data vectors are offset by the elements of "diag" because
    # of the way sparse.dia_matrix sets the diagonals
    for i in range(N_v):
        if msk[i]:
            if (i + diags[0]) >= 0:
                data[0, i + diags[0]] = 0
            if (i + diags[1]) >= 0:
                data[1, i + diags[1]] = 0
            data[2, i] = 1
            if (i + diags[3]) < (N_v):
                data[3, i + diags[3]] = 0
            if (i + diags[4]) < (N_v):
                data[4, i + diags[4]] = 0

    return sparse.dia_matrix((data, diags), shape=(N_v, N_v)).asformat(format)


def _check_dimension_match(dims: tuple[str, str], da: xr.DataArray):
    """
    Validate that a DataArray contains the required horizontal dimensions
    in the correct order.

    Parameters
    ----------
    dims : tuple[str, str]
        Names of the two horizontal dimensions, in the required order
        (e.g., ``("eta_rho", "xi_rho")``).
    da : xarray.DataArray
        The DataArray to validate.

    Raises
    ------
    ValueError
        If the required dimensions are missing, extra, or their order in ``da``
        does not exactly match ``dims``. The error message includes guidance
        on how to reorder the DataArray using ``transpose``.
    """
    # Extract the horizontal dims from the DataArray
    var_horiz_dims = tuple(d for d in da.dims if d in dims)

    if set(var_horiz_dims) != set(dims):
        raise ValueError(
            "LateralFill error: DataArray does not contain the required horizontal dimensions.\n"
            f"  expected dims = {dims}\n"
            f"  found dims    = {tuple(da.dims)}\n"
            "Ensure the DataArray includes all required dimensions."
        )

    if var_horiz_dims != dims:
        raise ValueError(
            "LateralFill error: DataArray horizontal dimension order is incorrect.\n"
            f"  expected order = {dims}\n"
            f"  found order    = {var_horiz_dims}\n"
            "Reorder the DataArray before applying LateralFill, e.g.:\n"
            f"  var = var.transpose(..., {', '.join(dims)})"
        )


def nearest_neighbor_fill(
    da: xr.DataArray, mask: xr.DataArray, dims: tuple[str, str]
) -> xr.DataArray:
    """Fill masked (land/NaN) cells with the value of the nearest valid cell.

    This is a cheap alternative to the iterative Poisson :class:`LateralFill`.
    Using a static 2D ocean ``mask``, it computes the index of the nearest valid
    (ocean) cell for every grid cell once via
    :func:`scipy.ndimage.distance_transform_edt`, then remaps every cell to that
    nearest valid value. Valid cells map to themselves (distance 0) and are left
    unchanged; masked cells take the nearest valid neighbor's value. The same
    index map is applied to every horizontal slice (depth/time), so the cost is
    a single distance transform regardless of the number of slices.

    This guarantees a NaN-free field (as long as at least one valid cell exists),
    so a subsequent linear interpolation cannot propagate NaNs.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray to fill. Must contain both dimensions in ``dims``.
    mask : xarray.DataArray
        2D boolean/integer ocean mask (True/1 = valid, False/0 = masked) with
        dimensions matching ``dims``.
    dims : tuple of str
        The two horizontal dimensions, ordered ``(latitude, longitude)``.

    Returns
    -------
    xarray.DataArray
        A new DataArray with masked cells filled by nearest-neighbor lookup.
    """
    from scipy.ndimage import distance_transform_edt

    _check_dimension_match(dims, da)

    mask_bool = mask.transpose(*dims).values.astype(bool)

    # Index of the nearest valid (True) cell for every cell in the grid.
    ii, jj = distance_transform_edt(
        ~mask_bool, return_distances=False, return_indices=True
    )

    def _fill(arr):
        # arr has the core dims (lat, lon) as its trailing axes; advanced
        # indexing with the 2D index arrays broadcasts over any leading dims.
        return arr[..., ii, jj]

    filled = xr.apply_ufunc(
        _fill,
        da,
        input_core_dims=[list(dims)],
        output_core_dims=[list(dims)],
        output_dtypes=[da.dtype],
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {
                dims[0]: mask_bool.shape[0],
                dims[1]: mask_bool.shape[1],
            }
        },
    )

    return filled.transpose(*da.dims)


def one_dim_fill(da: xr.DataArray, dim: str, direction="forward") -> xr.DataArray:
    """Fill NaN values in a DataArray along a specified dimension.

    Parameters
    ----------
    da : xr.DataArray
        The input DataArray with NaN values to be filled, which must include the specified dimension.
    dim : str
        The name of the dimension along which to fill NaN values (e.g., 'depth' or 'time').
    direction : str, optional
        The filling direction; either "forward" to propagate non-NaN values downward or "backward" to propagate them upward.
        Defaults to "forward".

    Returns
    -------
    xr.DataArray
        A new DataArray with NaN values filled in the specified direction, leaving the original data unchanged.
    """
    if dim in da.dims:
        if direction == "forward":
            return da.ffill(dim=dim)
        elif direction == "backward":
            return da.bfill(dim=dim)
    return da


def _own_weight(dz: np.ndarray, blend_m: float) -> np.ndarray:
    """Weight of a cell's own value, ``dz`` metres below its source's bottom.

    ``w = clip(1 - dz / blend_m, 0, 1)``: 1 at the bottom, 0 from ``blend_m`` below it,
    and 0 throughout for ``blend_m = 0``, where the donor's value is taken right below
    the bottom. The donor's weight is ``1 - w``.
    """
    if blend_m == 0:
        return np.zeros_like(dz, dtype=float)
    return np.clip(1.0 - dz / blend_m, 0.0, 1.0)


class BelowBottomFill:
    """Fill the cells below a source column's bottom from the nearest column with water.

    A column of a lat/lon source ends at its local bottom, so a deeper model column
    has levels with no source data. Copying the column's deepest value down gives
    bottom water at any depth. Here a cell below the bottom takes instead the value of
    the nearest (great-circle) column of the same source that has water at that
    depth, blended from the column's own bottom value:
    ``value = w * own + (1 - w) * donor`` with ``w = max(0, 1 - dz / blend_m)`` and
    ``dz`` the depth below the column's bottom. With ``blend_m = 0`` the donor's value
    is taken right below the bottom; with ``blend_m = inf`` the result is the plain
    forward fill along depth.

    The cells to fill are those a forward fill along depth would touch (finite at a
    level above, NaN at the level), so land columns are never filled. A level where no
    column has water keeps the filled level above. The donor of every cell is found
    once, from a static 3-D wet mask (one KD-tree per level), and reused by
    :meth:`apply` for any variable and time record.
    """

    def __init__(
        self,
        wet: np.ndarray,
        depth: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        dims: tuple[str, str, str],
        blend_m: float,
    ):
        """
        Build the donor map of a fill.

        Parameters
        ----------
        wet : numpy.ndarray
            3D boolean array, ``True`` where the source has data, with dimensions
            ``dims`` (depth, latitude, longitude).
        depth : numpy.ndarray
            1D source depths (m, positive downward, increasing with index).
        lat, lon : numpy.ndarray
            Latitude and longitude (degrees) of the source columns: 1D (along
            ``dims[1]`` and ``dims[2]``) or 2D with shape ``(ny, nx)``.
        dims : tuple of str
            Names of the depth, latitude and longitude dimensions of the data passed
            to :meth:`apply`.
        blend_m : float
            Depth (m) below a column's bottom over which a cell goes from the
            column's own bottom value to its donor's value; ``0`` (donor right below
            the bottom) to ``inf`` (plain forward fill).

        Raises
        ------
        ValueError
            If the shapes of the inputs are inconsistent or the depths do not
            increase.
        """
        wet = np.asarray(wet, dtype=bool)
        depth = np.asarray(depth, dtype=float)
        if wet.ndim != 3 or depth.shape != (wet.shape[0],):
            raise ValueError(
                "BelowBottomFill error: 'wet' must be 3D with one depth per level, "
                f"got wet.shape={wet.shape} and depth.shape={depth.shape}."
            )
        if not np.all(np.diff(depth) > 0):
            raise ValueError("BelowBottomFill error: 'depth' must increase.")
        lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        if lat.ndim == 1 and lon.ndim == 1:
            lat, lon = np.meshgrid(lat, lon, indexing="ij")
        if lat.shape != wet.shape[1:] or lon.shape != wet.shape[1:]:
            raise ValueError(
                "BelowBottomFill error: 'lat' and 'lon' must cover the horizontal "
                f"shape {wet.shape[1:]} of 'wet', got {lat.shape} and {lon.shape}."
            )

        self.dims = dims
        self.shape = wet.shape
        self.blend_m = blend_m
        # Per level with cells to fill: (level, flat cell indices, flat donor indices,
        # weights of the cells' own value), the donors and weights ``None`` where the
        # level has no water at all.
        self._levels: list = []
        # With an infinite blend length a cell keeps its own (forward-filled) value,
        # so no donor is needed.
        if not np.isinf(blend_m):
            self._find_donors(
                wet.reshape(wet.shape[0], -1), depth, lat.ravel(), lon.ravel()
            )

    def _find_donors(self, wet, depth, lat, lon):
        """Find, level by level, the donor of every cell below its column's bottom."""
        n_level = wet.shape[0]
        # Level of each column's deepest wet cell (meaningless for land columns, which
        # have no cell to fill).
        bottom = n_level - 1 - np.argmax(wet[::-1], axis=0)
        seen = np.zeros(wet.shape[1], dtype=bool)
        distance = []
        for k in range(n_level):
            # int32 flat indices halve the map's size; a level never has 2**31 cells.
            cells = np.flatnonzero(seen & ~wet[k]).astype(np.int32)
            seen |= wet[k]
            if not cells.size:
                continue
            wet_k = np.flatnonzero(wet[k]).astype(np.int32)
            if not wet_k.size:
                self._levels.append((k, cells, None, None))
                continue
            tree = build_kdtree_from_latlon(lat[wet_k], lon[wet_k])
            chord, nearest = tree.query(
                latlon_to_xyz(lat[cells], lon[cells]), workers=-1
            )
            distance.append((chord * R_EARTH / 1000.0).astype(np.float32))
            # Cells above their column's deepest water (gaps inside a column) take
            # the donor's value outright.
            dz = depth[k] - depth[bottom[cells]]
            weight = np.where(dz > 0, _own_weight(dz, self.blend_m), 0.0)
            self._levels.append((k, cells, wet_k[nearest], weight))
        if distance:
            distance = np.concatenate(distance)
            logging.info(
                f"Filled {distance.size} source cell(s) below the source bottom from "
                f"the nearest column with water at that depth, up to "
                f"{distance.max():.0f} km away (median {np.median(distance):.0f} km)."
            )

    def apply(self, da: xr.DataArray) -> xr.DataArray:
        """Fill the cells below each column's bottom.

        Parameters
        ----------
        da : xarray.DataArray
            Data to fill, with the dimensions ``dims`` (in any position, any other
            dimension, e.g. time, is carried along) and NaN outside the wet mask the
            fill was built from. Dask-backed input stays lazy; the dimensions
            ``dims`` must each be a single chunk.

        Returns
        -------
        xarray.DataArray
            The data with the cells below each column's bottom filled; cells that
            were not NaN, and cells in columns without any data, are unchanged.

        Raises
        ------
        ValueError
            If ``da`` lacks a dimension of ``dims`` or its size differs from that of
            the wet mask.
        """
        for dim, size in zip(self.dims, self.shape, strict=True):
            if dim not in da.dims or da.sizes[dim] != size:
                raise ValueError(
                    f"BelowBottomFill error: DataArray must have dimension {dim!r} "
                    f"of size {size}, found dims {tuple(da.dims)} with sizes "
                    f"{dict(da.sizes)}."
                )
        filled = xr.apply_ufunc(
            _below_bottom_fill_np,
            da,
            input_core_dims=[list(self.dims)],
            output_core_dims=[list(self.dims)],
            output_dtypes=[da.dtype],
            dask="parallelized",
            keep_attrs=True,
            kwargs={"levels": self._levels},
        )
        return filled.transpose(*da.dims)


def _below_bottom_fill_np(arr: np.ndarray, levels: list) -> np.ndarray:
    """Fill the cells below the columns' bottoms of ``arr`` (depth, lat, lon last).

    The forward fill along depth gives every cell below a column's bottom the column's
    own bottom value; the cells in ``levels`` (see :class:`BelowBottomFill`) are then
    replaced by the blend of that value with the donor's. The levels come from the
    first record's NaN pattern, so a cell that has data in this record keeps it.
    """
    shape = arr.shape
    src = arr.reshape(-1, shape[-3], shape[-2] * shape[-1])
    out = src.copy()
    for k in range(1, out.shape[1]):
        np.copyto(out[:, k], out[:, k - 1], where=np.isnan(out[:, k]))
    for k, cells, donors, weight in levels:
        raw = src[:, k, cells]
        if donors is None:
            # No water at this level: take the (filled) level above.
            value = out[:, k - 1, cells]
        else:
            own = out[:, k, cells]
            value = weight * own + (1.0 - weight) * src[:, k, donors]
            value = np.where(np.isfinite(value), value, own)
        out[:, k, cells] = np.where(np.isnan(raw), value, raw)
    return out.reshape(shape)


def fill_below_source_bottom(
    *,
    regridded: dict[str, xr.DataArray],
    target_depth: xr.DataArray,
    source_bottom: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    mask: xr.DataArray,
    parent_fields: dict[str, xr.DataArray],
    parent_depth: xr.DataArray,
    parent_h: xr.DataArray,
    parent_mask: xr.DataArray,
    parent_lon: xr.DataArray,
    parent_lat: xr.DataArray,
    blend_m: float,
) -> dict[str, xr.DataArray]:
    """Fill target levels below a parent's local bottom from deeper parent columns.

    Vertical regridding clamps every target level below the parent's deepest layer to
    that layer's value. For each wet target cell deeper than the regridded parent
    bottom, this takes the parent value at the same depth from the nearest
    (great-circle) wet column of the *native* parent grid whose depth ``h`` exceeds the
    cell's depth, interpolating linearly in that column's layer depths. The fill
    blends in from the clamped value: with ``dz`` the cell's depth below the regridded
    parent bottom, the donor's value has weight ``min(dz / blend_m, 1)``. So a cell at
    the parent bottom keeps the clamped value (the local bottom water is the better
    estimate there) and takes more of the deeper column's water at the same depth the
    further it lies below the bottom, all of it from ``blend_m`` on. With
    ``blend_m = 0`` the donor's value is taken right below the bottom; with
    ``blend_m = inf`` nothing is filled. Cells between the deepest layer and the
    bottom keep the clamped value, as do cells deeper than every parent column of
    the loaded parent subset (counted in one warning; a deeper column may exist
    outside the subset). The nearest eligible column is found exactly: from the
    nearest few columns of a KD-tree over all wet parent columns, retried with more,
    and finally by brute force over the deep-enough columns. Depths are positive
    downward, deepest level first. The arrays are computed (one record). Velocities
    are not filled.

    Parameters
    ----------
    regridded : dict of str to xarray.DataArray
        Tracers regridded to the target grid, ``(..., s_rho, eta_rho, xi_rho)``,
        keyed by name.
    target_depth : xarray.DataArray
        Layer depths (m) of the target grid, ``(s_rho, eta_rho, xi_rho)``.
    source_bottom : xarray.DataArray
        The parent's ``h`` regridded to the target grid, ``(eta_rho, xi_rho)``.
    lon, lat, mask : xarray.DataArray
        Longitude, latitude and land mask of the target grid, ``(eta_rho, xi_rho)``.
    parent_fields : dict of str to xarray.DataArray
        The tracers of the native parent grid, ``(s_rho, eta_rho, xi_rho)`` (one time
        record), under the names of ``regridded``. Land points hold laterally filled
        values; they are not used as donors.
    parent_depth : xarray.DataArray
        Layer depths (m) of the native parent grid, ``(s_rho, eta_rho, xi_rho)``.
    parent_h, parent_mask, parent_lon, parent_lat : xarray.DataArray
        Depth, land mask and coordinates of the native parent grid,
        ``(eta_rho, xi_rho)``.
    blend_m : float
        Depth (m) below the parent bottom over which a cell goes from the clamped
        value to the donor's; ``0`` (donor right below the bottom) to ``inf`` (no
        fill).

    Returns
    -------
    dict of str to xarray.DataArray
        The filled tracers, keyed like ``regridded``.

    Raises
    ------
    ValueError
        If the native parent fields do not share the horizontal shape of the parent
        grid.
    """
    if np.isinf(blend_m):
        return dict(regridded)

    dims = ("s_rho", "eta_rho", "xi_rho")
    tdepth = _single_record(target_depth, dims)
    bottom = source_bottom.transpose(*dims[1:]).values
    wet = mask.transpose(*dims[1:]).values > 0.5
    n_tgt, ny, nx = tdepth.shape
    pdepth = _single_record(parent_depth, dims)
    n_src = pdepth.shape[0]
    parent_2d = {
        name: da.transpose(*dims[1:]).values
        for name, da in (
            ("h", parent_h),
            ("mask", parent_mask),
            ("lat", parent_lat),
            ("lon", parent_lon),
        )
    }
    mismatched = [
        n
        for n, f in parent_fields.items()
        if tuple(f.sizes.get(d) for d in dims) != pdepth.shape
    ]
    mismatched += [n for n, a in parent_2d.items() if a.shape != pdepth.shape[1:]]
    if mismatched:
        raise ValueError(
            "The native parent fields and the parent grid differ in shape; expected "
            f"{pdepth.shape} (layer depths) but {mismatched} do not match."
        )
    parent_2d = {name: a.ravel() for name, a in parent_2d.items()}
    profiles = pdepth.reshape(n_src, -1)

    # Wet target cells below the parent bottom (NaN compares False).
    k, j, i = np.nonzero((tdepth > bottom) & wet)
    cells, column, depth = (k * ny + j) * nx + i, j * nx + i, tdepth[k, j, i]
    dz = depth - bottom[j, i]
    if not cells.size:
        return dict(regridded)

    # Parent columns that can donate: wet, with a finite depth and layer depths.
    pool = np.flatnonzero(
        (parent_2d["mask"] > 0.5)
        & np.isfinite(parent_2d["h"])
        & np.isfinite(profiles).all(0)
    )
    donor, distance = _find_deep_donors(
        depth,
        column,
        lat.transpose(*dims[1:]).values.ravel(),
        lon.transpose(*dims[1:]).values.ravel(),
        parent_2d["lat"][pool],
        parent_2d["lon"][pool],
        parent_2d["h"][pool],
    )
    found = donor >= 0
    if not found.all():
        logging.warning(
            f"{(~found).sum()} target cell(s) lie deeper than every parent column "
            "within the loaded parent subset and keep the parent's deepest value."
        )
    if not found.any():
        return dict(regridded)
    logging.info(
        f"Filled {found.sum()} target cell(s) below the source bottom from source "
        f"columns up to {np.nanmax(distance):.0f} km away "
        f"(median {np.nanmedian(distance):.0f} km)."
    )
    cells, depth, dz, donor = cells[found], depth[found], dz[found], pool[donor[found]]

    # Where each cell's depth falls in its donor's layer depths: linear interpolation
    # between two layers, clamped outside.
    deeper = np.empty(cells.size, dtype=np.int32)
    shallower = np.empty(cells.size, dtype=np.int32)
    frac = np.empty(cells.size)
    for start in range(0, cells.size, 50_000):
        sel = slice(start, start + 50_000)
        profile, z = profiles[:, donor[sel]], depth[sel]
        k0 = np.clip((profile >= z).sum(axis=0) - 1, 0, max(n_src - 2, 0))
        k1 = np.minimum(k0 + 1, n_src - 1)
        z0 = np.take_along_axis(profile, k0[None], 0)[0]
        z1 = np.take_along_axis(profile, k1[None], 0)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.clip((z0 - z) / (z0 - z1), 0.0, 1.0)
        deeper[sel], shallower[sel] = k0, k1
        frac[sel] = np.where(np.isfinite(w), w, 0.0)

    # Blend from the clamped value at the parent bottom to the donor's value, one
    # tracer at a time.
    own = _own_weight(dz, blend_m)
    filled = {}
    for name, field in regridded.items():
        pf = _single_record(parent_fields[name], dims).reshape(n_src, -1)
        low, high = pf[deeper, donor], pf[shallower, donor]
        value = low + frac * (high - low)
        target = field.transpose(..., *dims)
        out = np.array(target.values, order="C", copy=True)
        flat = out.reshape(-1, n_tgt * ny * nx)
        clamped = flat[:, cells]
        blended = own * clamped + (1.0 - own) * value
        flat[:, cells] = np.where(np.isfinite(value), blended, clamped)
        filled[name] = target.copy(data=out).transpose(*field.dims)
    return filled


def _single_record(da: xr.DataArray, dims: tuple[str, ...]) -> np.ndarray:
    """Return ``da`` as an array ordered ``dims``, after dropping a length-1 time axis.

    Raises
    ------
    ValueError
        If ``da`` has a dimension other than ``dims`` that is not of length 1.
    """
    extra = {d: 0 for d in da.dims if d not in dims}
    if any(da.sizes[d] != 1 for d in extra):
        raise ValueError(
            f"Expected a single record, got dimensions {dict(da.sizes)} for dims {dims}."
        )
    return da.isel(extra).transpose(*dims).values


def _find_deep_donors(
    depth: np.ndarray,
    column: np.ndarray,
    column_lat: np.ndarray,
    column_lon: np.ndarray,
    pool_lat: np.ndarray,
    pool_lon: np.ndarray,
    pool_h: np.ndarray,
    neighbors: tuple[int, ...] = (32, 128, 512),
) -> tuple[np.ndarray, np.ndarray]:
    """Find, for each cell, the nearest pool column deeper than the cell.

    Exact: the nearest ``neighbors[0]`` pool columns of a cell's column come from one
    KD-tree over the whole pool and the cell takes the first deep enough. Cells with
    none retry with the next count, then, if still unresolved, take the nearest
    deep-enough column by brute force (chord distance on the unit sphere, monotone in
    great-circle distance).

    Parameters
    ----------
    depth : numpy.ndarray
        Depth (m) of each cell.
    column : numpy.ndarray
        Id of the column of each cell (an index into ``column_lat``/``column_lon``).
    column_lat, column_lon : numpy.ndarray
        Coordinates (degrees) of the columns.
    pool_lat, pool_lon, pool_h : numpy.ndarray
        Coordinates (degrees) and depth (m) of the candidate donor columns.
    neighbors : tuple of int, optional
        Numbers of nearest pool columns to search, in turn.

    Returns
    -------
    donor : numpy.ndarray
        Index into the pool of each cell's donor column, ``-1`` where no pool column
        is deeper than the cell.
    distance : numpy.ndarray
        Distance (km) to the donor, NaN where there is none.
    """
    donor = np.full(depth.size, -1)
    distance = np.full(depth.size, np.nan)
    if not pool_h.size:
        return donor, distance
    # A cell at least as deep as every pool column can never find a donor.
    todo = np.flatnonzero(depth < pool_h.max())
    if not todo.size:
        return donor, distance
    tree = build_kdtree_from_latlon(pool_lat, pool_lon)
    block = 4_000_000  # entries of an (columns or cells, k) array held at once
    for count in neighbors:
        k = min(count, pool_h.size)
        # Sorted by column, a chunk of cells needs one query per distinct column.
        todo = todo[np.argsort(column[todo], kind="stable")]
        per_chunk = max(1, block // k)
        unresolved = []
        for start in range(0, todo.size, per_chunk):
            cells = todo[start : start + per_chunk]
            columns, rows = np.unique(column[cells], return_inverse=True)
            chord, nearest = tree.query(
                latlon_to_xyz(column_lat[columns], column_lon[columns]), k=k, workers=-1
            )
            chord, nearest = chord.reshape(-1, k), nearest.reshape(-1, k)
            neighbor = nearest[rows]
            deep = pool_h[neighbor] > depth[cells, None]
            nearest_deep = deep.argmax(axis=1)
            ok = deep.any(axis=1)
            taken = np.arange(cells.size)
            donor[cells[ok]] = neighbor[taken, nearest_deep][ok]
            distance[cells[ok]] = (
                chord[rows][taken, nearest_deep][ok] * R_EARTH / 1000.0
            )
            unresolved.append(cells[~ok])
        todo = np.concatenate(unresolved)
        # Every pool column was among the k nearest: the rest have no donor.
        if not todo.size or k >= pool_h.size:
            return donor, distance

    xyz = latlon_to_xyz(pool_lat, pool_lon)
    step = max(1, 20_000_000 // pool_h.size)
    for start in range(0, todo.size, step):
        cells = todo[start : start + step]
        xyz_cells = latlon_to_xyz(column_lat[column[cells]], column_lon[column[cells]])
        d2 = 2.0 - 2.0 * (xyz_cells @ xyz.T)
        d2 = np.where(pool_h[None, :] > depth[cells, None], d2, np.inf)
        nearest_col = d2.argmin(axis=1)
        best = d2[np.arange(cells.size), nearest_col]
        ok = np.isfinite(best)
        donor[cells[ok]] = nearest_col[ok]
        distance[cells[ok]] = np.sqrt(np.clip(best[ok], 0.0, None)) * R_EARTH / 1000.0
    return donor, distance
