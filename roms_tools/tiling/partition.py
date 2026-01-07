from collections.abc import Sequence
from numbers import Integral
from pathlib import Path

import numpy as np
import xarray as xr

from roms_tools.utils import save_datasets

DIM_INFO = {
    # eta-direction
    "eta_rho": dict(axis="eta", ghost=2, edge="both"),
    "eta_u": dict(axis="eta", ghost=2, edge="both"),
    "eta_v": dict(axis="eta", ghost=1, edge="upper"),
    "eta_psi": dict(axis="eta", ghost=3, edge="both"),
    "eta_coarse": dict(axis="eta", ghost=2, edge="both"),
    # xi-direction
    "xi_rho": dict(axis="xi", ghost=2, edge="both"),
    "xi_u": dict(axis="xi", ghost=1, edge="upper"),
    "xi_v": dict(axis="xi", ghost=2, edge="both"),
    "xi_psi": dict(axis="xi", ghost=3, edge="both"),
    "xi_coarse": dict(axis="xi", ghost=2, edge="both"),
}


def _exact_division(size: int, nparts: int, dim: str) -> int:
    if size % nparts != 0:
        raise ValueError(
            f"Dimension '{dim}' of size {size} cannot be evenly divided into {nparts} partitions."
        )
    return size // nparts


def _compute_partition_sizes(
    total_size: int,
    nparts: int,
    ghost: int,
    edge: str,
    dim: str,
) -> list[int]:
    """Compute per-tile sizes including ghost cells."""
    if nparts == 1:
        return [total_size]

    core = _exact_division(total_size - ghost, nparts, dim)

    sizes = [core] * nparts

    if edge == "both":
        sizes[0] += ghost // 2
        sizes[-1] += ghost - ghost // 2
    elif edge == "upper":
        sizes[-1] += ghost
    else:
        raise ValueError(f"Unknown edge rule '{edge}' for {dim}")

    return sizes


def _cumsum(sizes: list[int]) -> np.ndarray:
    out = np.zeros(len(sizes) + 1, dtype=int)
    out[1:] = np.cumsum(sizes)
    return out


def partition(
    ds: xr.Dataset, np_eta: int = 1, np_xi: int = 1, include_coarse_dims: bool = True
) -> tuple[list[int], list[xr.Dataset]]:
    """Partition a ROMS (Regional Ocean Modeling System) dataset into smaller spatial
    tiles.

    This function divides the input dataset into `np_eta` by `np_xi` tiles, where each tile
    represents a subdomain of the original dataset. The partitioning is performed along
    the spatial dimensions `eta_rho`, `eta_u`, `eta_v`, `eta_coarse`, `xi_rho`, `xi_u`, `xi_v`,
    and `xi_coarse`, depending on which dimensions are present in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input ROMS dataset that is to be partitioned.

    np_eta : int, optional
        The number of partitions along the `eta` direction. Must be a positive integer. Default is 1.

    np_xi : int, optional
        The number of partitions along the `xi` direction. Must be a positive integer. Default is 1.

    include_coarse_dims : bool, optional
        Whether to include coarse grid dimensions (`eta_coarse`, `xi_coarse`) in the partitioning.
        If False, these dimensions will not be split. Relevant if none of the coarse resolution variables are actually used by ROMS.
        Default is True.

    Returns
    -------
    tuple[list[int], list[xr.Dataset]]
        A tuple containing two elements:

        - A list of integers representing the file numbers associated with each partition.
        - A list of `xarray.Dataset` objects, each representing a partitioned subdomain of the original dataset.

    Raises
    ------
    ValueError
        If `np_eta` or `np_xi` is not a positive integer, or if the dataset cannot be evenly partitioned
        into the specified number of tiles.


    Example
    -------
    >>> partitioned_file_numbers, partitioned_datasets = partition(
    ...     ds, np_eta=2, np_xi=3
    ... )
    >>> print(partitioned_file_numbers)
    [0, 1, 2, 3, 4, 5]
    >>> print([ds.sizes for ds in partitioned_datasets])
    [{'eta_rho': 50, 'xi_rho': 50}, {'eta_rho': 50, 'xi_rho': 50}, ...]

    This example partitions the dataset into 2 tiles along the `eta` direction and 3 tiles
    along the `xi` direction, resulting in a total of 6 partitions.
    """
    if (
        not isinstance(np_eta, Integral)
        or np_eta < 1
        or not isinstance(np_xi, Integral)
        or np_xi < 1
    ):
        raise ValueError("np_eta and np_xi must be positive integers")

    # Select applicable dimensions
    dims_to_partition = [
        d
        for d in DIM_INFO
        if d in ds.dims and (include_coarse_dims or "coarse" not in d)
    ]

    partitioned_sizes = {}

    for dim in dims_to_partition:
        info = DIM_INFO[dim]
        nparts = np_eta if info["axis"] == "eta" else np_xi

        partitioned_sizes[dim] = _compute_partition_sizes(
            total_size=ds.sizes[dim],
            nparts=nparts,
            ghost=info["ghost"],
            edge=info["edge"],
            dim=dim,
        )

    file_numbers = []
    partitioned_datasets = []

    for i in range(np_eta):
        for j in range(np_xi):
            file_numbers.append(j + i * np_xi)
            indexers = {}

            for dim, sizes in partitioned_sizes.items():
                info = DIM_INFO[dim]
                idx = i if info["axis"] == "eta" else j
                bounds = _cumsum(sizes)
                indexers[dim] = slice(bounds[idx], bounds[idx + 1])

            partitioned_datasets.append(ds.isel(**indexers))

    return file_numbers, partitioned_datasets


def partition_netcdf(
    filepath: str | Path | Sequence[str | Path],
    np_eta: int = 1,
    np_xi: int = 1,
    output_dir: str | Path | None = None,
    include_coarse_dims: bool = True,
) -> list[Path]:
    """Partition one or more ROMS NetCDF files into smaller spatial tiles and save them to disk.

    This function divides each dataset into `np_eta` by `np_xi` tiles.
    Each tile is saved as a separate NetCDF file.

    Parameters
    ----------
    filepath : str | Path | Sequence[str | Path]
        A path or list of paths to input NetCDF files.

    np_eta : int, optional
        The number of partitions along the `eta` direction. Must be a positive integer. Default is 1.

    np_xi : int, optional
        The number of partitions along the `xi` direction. Must be a positive integer. Default is 1.

    output_dir : str | Path | None, optional
        Directory or base path to save partitioned files.
        If None, files are saved alongside the input file.

    include_coarse_dims : bool, optional
        Whether to include coarse grid dimensions (`eta_coarse`, `xi_coarse`) in the partitioning.
        If False, these dimensions will not be split. Relevant if none of the coarse resolution variables are actually used by ROMS.
        Default is True.

    Returns
    -------
    list[Path]
        A list of Path objects for the filenames that were saved.
    """
    if isinstance(filepath, str | Path):
        filepaths = [Path(filepath)]
    else:
        filepaths = [Path(fp) for fp in filepath]

    all_saved_filenames = []

    for fp in filepaths:
        input_file = fp.with_suffix(".nc")
        ds = xr.open_dataset(input_file, decode_timedelta=False)

        file_numbers, partitioned_datasets = partition(
            ds, np_eta=np_eta, np_xi=np_xi, include_coarse_dims=include_coarse_dims
        )

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            base_filepath = output_dir / fp.stem
        else:
            base_filepath = fp.with_suffix("")

        ndigits = len(str(max(file_numbers)))
        paths_to_partitioned_files = [
            Path(f"{base_filepath}.{num:0{ndigits}d}") for num in file_numbers
        ]

        saved = save_datasets(
            partitioned_datasets, paths_to_partitioned_files, verbose=False
        )
        all_saved_filenames.extend(saved)

    return all_saved_filenames
