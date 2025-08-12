from collections.abc import Sequence
from pathlib import Path

import xarray as xr


def open_partitions(pattern: str) -> xr.Dataset:
    """
    Open partitioned ROMS netCDF files as a single dataset.

    Parameters
    ----------
    pattern: str
        Glob pattern for partitioned files, e.g. "roms_rst.20121209133435.*.nc"

    Returns
    -------
    xarray.Dataset
        Dataset containing unified partitioned datasets
    """
    filepaths = sorted(Path(".").glob(pattern))
    if not filepaths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    joined = join_datasets(filepaths)
    return joined


def join_netcdf(pattern: str, output_path: Path | None = None) -> Path:
    """
    Join partitioned NetCDFs into a single dataset.

    Parameters
    ----------
    pattern : str
        Glob pattern for partitioned files, e.g. "roms_rst.20121209133435.*.nc"

    output_path : Path, optional
        If provided, the joined dataset will be saved to this path.
        Otherwise, the common base of pattern (e.g. roms_rst.20121209133435.nc)
        will be used.

    Returns
    -------
    Path
        The path of the saved file
    """
    filepaths = sorted(Path(".").glob(pattern))
    if not filepaths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
        # Determine output path if not provided
    if output_path is None:
        # e.g. roms_rst.20120101120000.023.nc -> roms_rst.20120101120000.nc
        output_path = filepaths[0].with_suffix("").with_suffix(".nc")

    joined = open_partitions(pattern)
    joined.to_netcdf(output_path)
    print(f"Saved joined dataset to: {output_path}")

    return output_path


def _count_transitions(dim_sizes: list[int]) -> int:
    """Counts the number of transitions in a list of dimension sizes.

    A transition is a point where the dimension size changes from the previous one.
    This is used to determine the number of partitions (e.g., np_eta or np_xi).

    Parameters
    ----------
    dim_sizes : list[int]
        A list of integer sizes for a given dimension across multiple datasets.

    Returns
    -------
    int
        The number of transitions detected.

    >>> # Example for np_xi = 3, np_eta = 2
    >>> # Datasets are ordered first by eta, then by xi: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    >>> # The 'eta' dimension size will be consistent for a row (np_xi=3), but change at the next row.
    >>> eta_dim_sizes = [50, 50, 50, 51, 51, 51]
    >>> _count_transitions(eta_dim_sizes) # transition from 50 to 51
    1
    """
    if not dim_sizes:
        return 0
    transitions = 0
    for i in range(1, len(dim_sizes)):
        if dim_sizes[i] != dim_sizes[i - 1]:
            transitions += 1
    return transitions


def find_common_dim(direction: str, datasets: Sequence[xr.Dataset]) -> str:
    """Finds a common dimension along the xi or eta direction amongst a list of Datasets.

    Parameters
    ----------
    direction: str ("xi" or "eta")
        The direction in which to seek a common dimension
    datasets: Sequence[xr.Dataset]:
        The datasets in which to look

    Returns
    -------
    common_dim: str
        The dimension common to all specified datasets along 'direction'
    """
    if direction not in ["xi", "eta"]:
        raise ValueError("'direction' must be 'xi' or 'eta'")
    for point in ["rho", "u", "v"]:
        if all(f"{direction}_{point}" in d.dims for d in datasets):
            return f"{direction}_{point}"
    raise ValueError(f"No common point found along direction {direction}")


def infer_partition_layout_from_datasets(
    datasets: Sequence[xr.Dataset],
) -> tuple[int, int]:
    """Infer np_eta, np_xi from datasets."""
    dims_list = [
        {
            "eta_rho": ds.sizes.get("eta_rho", 0),
            "xi_rho": ds.sizes.get("xi_rho", 0),
            "eta_v": ds.sizes.get("eta_v", 0),
            "xi_u": ds.sizes.get("xi_u", 0),
        }
        for ds in datasets
    ]
    eta_dim = find_common_dim("eta", datasets)
    eta_sizes = [d[eta_dim] for d in dims_list]
    eta_transitions = _count_transitions(eta_sizes)

    xi_dim = find_common_dim("xi", datasets)
    xi_sizes = [d[xi_dim] for d in dims_list]
    xi_transitions = _count_transitions(xi_sizes)

    nd = len(datasets)
    if eta_transitions == 0:
        if xi_transitions == nd - 1:
            np_xi, np_eta = 1, nd
        else:
            np_xi = xi_transitions + 1
            np_eta = nd // np_xi
    elif xi_transitions == 0:
        if eta_transitions == nd - 1:
            np_xi, np_eta = nd, 1
        else:
            np_eta = eta_transitions + 1
            np_xi = nd // np_eta
    return np_xi, np_eta


def join_datasets(filepaths: Sequence[Path]) -> xr.Dataset:
    datasets = [xr.open_dataset(p, decode_timedelta=False) for p in sorted(filepaths)]

    np_xi, np_eta = infer_partition_layout_from_datasets(datasets)
    # info = get_dims_from_datasets(datasets, np_xi=np_xi, np_eta=np_eta)

    # Arrange into grid
    grid = [[datasets[j + i * np_xi] for j in range(np_xi)] for i in range(np_eta)]

    # Join each row (along xi_*)
    rows_joined = []
    for row in grid:
        all_vars = set().union(*(ds.data_vars for ds in row))
        row_dataset = xr.Dataset()

        for varname in all_vars:
            var_slices = [ds[varname] for ds in row if varname in ds]
            xi_dims = [dim for dim in var_slices[0].dims if dim.startswith("xi_")]

            if not xi_dims:
                row_dataset[varname] = var_slices[0]
            else:
                xi_dim = xi_dims[0]
                row_dataset[varname] = xr.concat(
                    var_slices, dim=xi_dim, combine_attrs="override"
                )

        rows_joined.append(row_dataset)

    # Join all rows (along eta_*)
    final_dataset = xr.Dataset()
    all_vars = set().union(*(ds.data_vars for ds in rows_joined))

    for varname in all_vars:
        var_slices = [ds[varname] for ds in rows_joined if varname in ds]
        eta_dims = [dim for dim in var_slices[0].dims if dim.startswith("eta_")]

        if not eta_dims:
            final_dataset[varname] = var_slices[0]
        else:
            eta_dim = eta_dims[0]
            final_dataset[varname] = xr.concat(
                var_slices, dim=eta_dim, combine_attrs="override"
            )

    return final_dataset
