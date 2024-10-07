import numpy as np
import xarray as xr
import pyamg
from scipy import sparse


class LateralFill:
    def __init__(self, mask, dims, tol=1.0e-4):
        """
        Initializes the LateralFill class, which fills NaN values in a DataArray
        by iteratively solving a Poisson equation using a lateral diffusion approach.

        Parameters
        ----------
        mask : xarray.DataArray or ndarray of bool
            A 2D boolean mask indicating valid points (True) and land points (False).
            Boundary points are automatically set to land (True).
        dims : list of str
            Dimensions along which to perform the lateral fill.
        tol : float, optional
            Tolerance for the iterative solver, determining convergence. Default is 1.0e-4.

        Raises
        ------
        NotImplementedError
            If the input mask has more than two dimensions, which is not supported by the current implementation.
        """

        if len(mask.shape) > 2:
            raise NotImplementedError("LateralFill currently supports only 2D masks.")

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
        self.dims = dims
        self.tol = tol

    def apply(self, var):
        """
        Fills NaN values in an xarray DataArray using iterative lateral diffusion.

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
        # Apply fill to anomaly field
        mean = var.mean(dim=self.dims, skipna=True)
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
    """
    Fills all NaN values in a 2D NumPy array using an iterative solver,
    while preserving the existing non-NaN values.
    The filling process uses an AMG solver to efficiently perform smoothing
    based on the Laplace operator.

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

    b_flat = b.flatten()
    x0_flat = x0.flatten()
    x = ml.solve(b_flat, x0_flat, tol=tol)
    x_2d = x.reshape(b.shape)

    return x_2d


def laplacian(grid, mask, dtype=float, format=None):
    """
    Return a sparse matrix for solving a 2-dimensional Poisson problem.

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
    """
    Construct a sparse matrix from a local matrix stencil.

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
