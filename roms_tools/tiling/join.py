from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import xarray as xr

from roms_tools.utils import FilePaths, _path_list_from_input


def open_partitions(files: FilePaths) -> xr.Dataset:
    """
    Open partitioned ROMS netCDF files as a single dataset.

    Parameters
    ----------
    files: str | List[str | Path]
        List or wildcard pattern describing files to join,
        e.g. "roms_rst.20121209133435.*.nc"

    Returns
    -------
    xarray.Dataset
        Dataset containing unified partitioned datasets
    """
    filepaths = _path_list_from_input(files)
    datasets = [xr.open_dataset(p, decode_timedelta=True) for p in sorted(filepaths)]
    joined = join_datasets(datasets)
    return joined


def join_netcdf(files: FilePaths, output_path: Path | None = None) -> Path:
    """
    Join partitioned NetCDFs into a single dataset.

    Parameters
    ----------
    files : str | List[str | Path]
        List or wildcard pattern describing files to join,
        e.g. "roms_rst.20121209133435.*.nc"

    output_path : Path, optional
        If provided, the joined dataset will be saved to this path.
        Otherwise, the common base of pattern (e.g. roms_rst.20121209133435.nc)
        will be used.

    Returns
    -------
    Path
        The path of the saved file
    """
    filepaths = _path_list_from_input(files)
    # Determine output path if not provided
    if output_path is None:
        # e.g. roms_rst.20120101120000.023.nc -> roms_rst.20120101120000.nc
        output_path = filepaths[0].with_suffix("").with_suffix(".nc")

    joined = open_partitions(cast(FilePaths, filepaths))
    joined.to_netcdf(output_path)
    print(f"Saved joined dataset to: {output_path}")

    return output_path


def _find_transitions(dim_sizes: list[int]) -> list[int]:
    """Finds the indices of all transitions in a list of dimension sizes.

    A transition is a point where the dimension size changes from the previous one.
    This function is used to determine the number of partitions (e.g., np_eta or np_xi).

    Parameters
    ----------
    dim_sizes : list[int]
        A list of integer sizes for a given dimension across multiple datasets.

    Returns
    -------
    List[int]
        A list of indices where a transition was detected.
    """
    transitions: list[int] = []
    if len(dim_sizes) < 2:
        return transitions

    for i in range(1, len(dim_sizes)):
        if dim_sizes[i] != dim_sizes[i - 1]:
            transitions.append(i)
    return transitions


def _find_common_dims(
    direction: Literal["xi", "eta"], datasets: Sequence[xr.Dataset]
) -> list[str]:
    """Finds all common dimensions along the xi or eta direction amongst a list of Datasets.

    Parameters
    ----------
    direction: str ("xi" or "eta")
        The direction in which to seek a common dimension
    datasets: Sequence[xr.Dataset]:
        The datasets in which to look

    Returns
    -------
    common_dim: list[str]
        The dimensions common to all specified datasets along 'direction'
    """
    if direction not in ["xi", "eta"]:
        raise ValueError("'direction' must be 'xi' or 'eta'")
    dims = []
    for point in ["rho", "u", "v"]:
        if all(f"{direction}_{point}" in d.dims for d in datasets):
            dims.append(f"{direction}_{point}")
    if not dims:
        raise ValueError(f"No common point found along direction {direction}")
    return dims


def _infer_partition_layout_from_datasets(
    datasets: Sequence[xr.Dataset],
) -> tuple[int, int]:
    """Infer np_eta, np_xi from datasets."""
    nd = len(datasets)
    if nd == 1:
        return 1, 1

    eta_dims = _find_common_dims("eta", datasets)
    first_eta_transition = nd

    for eta_dim in eta_dims:
        dim_sizes = [ds.sizes.get(eta_dim, 0) for ds in datasets]
        eta_transitions = _find_transitions(dim_sizes)
        if eta_transitions and (min(eta_transitions) < first_eta_transition):
            first_eta_transition = min(eta_transitions)
    if first_eta_transition < nd:
        np_xi = first_eta_transition
        np_eta = nd // np_xi
        return np_xi, np_eta
    # If we did not successfully find np_xi,np_eta using eta points
    # then we have a single-column grid:

    return nd, 1


def join_datasets(datasets: Sequence[xr.Dataset]) -> xr.Dataset:
    """Take a sequence of partitioned Datasets and return a joined Dataset."""
    np_xi, np_eta = _infer_partition_layout_from_datasets(datasets)

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
    # Copy attributes from first dataset
    final_dataset.attrs = datasets[0].attrs

    return final_dataset
