from numbers import Integral

import numpy as np
import xarray as xr
from typing import Union
from pathlib import Path
import re
import glob


def partition(
    ds: xr.Dataset, np_eta: int = 1, np_xi: int = 1
) -> tuple[list[int], list[xr.Dataset]]:
    """Partition a ROMS (Regional Ocean Modeling System) dataset into smaller spatial
    tiles.

    This function divides the input dataset into `np_eta` by `np_xi` tiles, where each tile
    represents a subdomain of the original dataset. The partitioning is performed along
    the spatial dimensions `eta_rho`, `xi_rho`, `eta_v`, `xi_u`, `eta_psi`, `xi_psi`, `eta_coarse`, and `xi_coarse`,
    depending on which dimensions are present in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input ROMS dataset that is to be partitioned.

    np_eta : int, optional
        The number of partitions along the `eta` direction. Must be a positive integer. Default is 1.

    np_xi : int, optional
        The number of partitions along the `xi` direction. Must be a positive integer. Default is 1.

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

    partitionable_dims_maybe_present = [
        "eta_rho",
        "xi_rho",
        "eta_v",
        "xi_u",
        "eta_psi",
        "xi_psi",
        "eta_coarse",
        "xi_coarse",
    ]
    dims_to_partition = [d for d in partitionable_dims_maybe_present if d in ds.dims]

    # if eta is periodic there are no ghost cells along those dimensions
    if "eta_v" in ds.sizes and ds.sizes["eta_rho"] == ds.sizes["eta_v"]:
        # TODO how are we supposed to know if eta is periodic if eta_v doesn't appear? partit.F doesn't say...
        n_eta_ghost_cells = 0
    else:
        n_eta_ghost_cells = 1

    # if xi is periodic there are no ghost cells along those dimensions
    if "xi_u" in ds.sizes and ds.sizes["xi_rho"] == ds.sizes["xi_u"]:
        n_xi_ghost_cells = 0
    else:
        n_xi_ghost_cells = 1

    def integer_division_or_raise(a: int, b: int, dimension: str) -> int:
        """Perform integer division and ensure that the division is exact.

        Parameters
        ----------
        a : int
            The numerator for the division.
        b : int
            The denominator for the division.
        dimension : str
            The name of the dimension being partitioned, used for error reporting.

        Returns
        -------
        int
            The result of the integer division.

        Raises
        ------
        ValueError
            If the division is not exact, indicating that the domain cannot be evenly divided
            along the specified dimension.
        """
        remainder = a % b
        if remainder == 0:
            return a // b
        else:
            raise ValueError(
                f"Dimension '{dimension}' of size {a} cannot be evenly divided into {b} partitions."
            )

    if "eta_rho" in dims_to_partition:
        eta_rho_domain_size = integer_division_or_raise(
            ds.sizes["eta_rho"] - 2 * n_eta_ghost_cells, np_eta, "eta_rho"
        )

    if "xi_rho" in dims_to_partition:
        xi_rho_domain_size = integer_division_or_raise(
            ds.sizes["xi_rho"] - 2 * n_xi_ghost_cells, np_xi, "xi_rho"
        )

    if "eta_v" in dims_to_partition:
        eta_v_domain_size = integer_division_or_raise(
            ds.sizes["eta_v"] - 1 * n_eta_ghost_cells, np_eta, "eta_v"
        )

    if "xi_u" in dims_to_partition:
        xi_u_domain_size = integer_division_or_raise(
            ds.sizes["xi_u"] - 1 * n_xi_ghost_cells, np_xi, "xi_u"
        )

    if "eta_psi" in dims_to_partition:
        eta_psi_domain_size = integer_division_or_raise(
            ds.sizes["eta_psi"] - 3 * n_eta_ghost_cells, np_eta, "eta_psi"
        )

    if "xi_psi" in dims_to_partition:
        xi_psi_domain_size = integer_division_or_raise(
            ds.sizes["xi_psi"] - 3 * n_xi_ghost_cells, np_xi, "xi_psi"
        )

    if "eta_coarse" in dims_to_partition:
        eta_coarse_domain_size = integer_division_or_raise(
            ds.sizes["eta_coarse"] - 2 * n_eta_ghost_cells, np_eta, "eta_coarse"
        )
    if "xi_coarse" in dims_to_partition:
        xi_coarse_domain_size = integer_division_or_raise(
            ds.sizes["xi_coarse"] - 2 * n_xi_ghost_cells, np_xi, "xi_coarse"
        )

    # unpartitioned dimensions should have sizes unchanged
    partitioned_sizes = {
        dim: [size] for dim, size in ds.sizes.items() if dim in dims_to_partition
    }

    # TODO refactor to use two functions for odd- and even-length dimensions
    if "eta_v" in dims_to_partition:
        partitioned_sizes["eta_v"] = [eta_v_domain_size] * (np_eta - 1) + [
            eta_v_domain_size + n_eta_ghost_cells
        ]
    if "xi_u" in dims_to_partition:
        partitioned_sizes["xi_u"] = [xi_u_domain_size] * (np_xi - 1) + [
            xi_u_domain_size + n_xi_ghost_cells
        ]

    if np_eta > 1:
        if "eta_rho" in dims_to_partition:
            partitioned_sizes["eta_rho"] = (
                [eta_rho_domain_size + n_eta_ghost_cells]
                + [eta_rho_domain_size] * (np_eta - 2)
                + [eta_rho_domain_size + n_eta_ghost_cells]
            )
        if "eta_psi" in dims_to_partition:
            partitioned_sizes["eta_psi"] = (
                [n_eta_ghost_cells + eta_psi_domain_size]
                + [eta_psi_domain_size] * (np_eta - 2)
                + [eta_psi_domain_size + 2 * n_eta_ghost_cells]
            )
        if "eta_coarse" in dims_to_partition:
            partitioned_sizes["eta_coarse"] = (
                [eta_coarse_domain_size + n_eta_ghost_cells]
                + [eta_coarse_domain_size] * (np_eta - 2)
                + [eta_coarse_domain_size + n_eta_ghost_cells]
            )

    if np_xi > 1:
        if "xi_rho" in dims_to_partition:
            partitioned_sizes["xi_rho"] = (
                [xi_rho_domain_size + n_xi_ghost_cells]
                + [xi_rho_domain_size] * (np_xi - 2)
                + [xi_rho_domain_size + n_xi_ghost_cells]
            )
        if "xi_psi" in dims_to_partition:
            partitioned_sizes["xi_psi"] = (
                [n_xi_ghost_cells + xi_psi_domain_size]
                + [xi_psi_domain_size] * (np_xi - 2)
                + [xi_psi_domain_size + 2 * n_xi_ghost_cells]
            )
        if "xi_coarse" in dims_to_partition:
            partitioned_sizes["xi_coarse"] = (
                [xi_coarse_domain_size + n_xi_ghost_cells]
                + [xi_coarse_domain_size] * (np_xi - 2)
                + [xi_coarse_domain_size + n_xi_ghost_cells]
            )

    def cumsum(pmf):
        """Implementation of cumsum which ensures the result starts with zero."""
        cdf = np.empty(len(pmf) + 1, dtype=int)
        cdf[0] = 0
        np.cumsum(pmf, out=cdf[1:])
        return cdf

    file_numbers = []
    partitioned_datasets = []
    for i in range(np_eta):
        for j in range(np_xi):
            file_number = j + (i * np_xi)
            file_numbers.append(file_number)

            indexers = {}

            if "eta_rho" in dims_to_partition:
                eta_rho_partition_indices = cumsum(partitioned_sizes["eta_rho"])
                indexers["eta_rho"] = slice(
                    int(eta_rho_partition_indices[i]),
                    int(eta_rho_partition_indices[i + 1]),
                )
            if "xi_rho" in dims_to_partition:
                xi_rho_partition_indices = cumsum(partitioned_sizes["xi_rho"])
                indexers["xi_rho"] = slice(
                    int(xi_rho_partition_indices[j]),
                    int(xi_rho_partition_indices[j + 1]),
                )

            if "eta_v" in dims_to_partition:
                eta_v_partition_indices = cumsum(partitioned_sizes["eta_v"])
                indexers["eta_v"] = slice(
                    int(eta_v_partition_indices[i]),
                    int(eta_v_partition_indices[i + 1]),
                )
            if "xi_u" in dims_to_partition:
                xi_u_partition_indices = cumsum(partitioned_sizes["xi_u"])
                indexers["xi_u"] = slice(
                    int(xi_u_partition_indices[j]), int(xi_u_partition_indices[j + 1])
                )
            if "eta_psi" in dims_to_partition:
                eta_psi_partition_indices = cumsum(partitioned_sizes["eta_psi"])
                indexers["eta_psi"] = slice(
                    int(eta_psi_partition_indices[i]),
                    int(eta_psi_partition_indices[i + 1]),
                )
            if "xi_psi" in dims_to_partition:
                xi_psi_partition_indices = cumsum(partitioned_sizes["xi_psi"])
                indexers["xi_psi"] = slice(
                    int(xi_psi_partition_indices[j]),
                    int(xi_psi_partition_indices[j + 1]),
                )

            if "eta_coarse" in dims_to_partition:
                eta_coarse_partition_indices = cumsum(partitioned_sizes["eta_coarse"])
                indexers["eta_coarse"] = slice(
                    int(eta_coarse_partition_indices[i]),
                    int(eta_coarse_partition_indices[i + 1]),
                )

            if "xi_coarse" in dims_to_partition:
                xi_coarse_partition_indices = cumsum(partitioned_sizes["xi_coarse"])
                indexers["xi_coarse"] = slice(
                    int(xi_coarse_partition_indices[j]),
                    int(xi_coarse_partition_indices[j + 1]),
                )

            partitioned_ds = ds.isel(**indexers)

            partitioned_datasets.append(partitioned_ds)

    return file_numbers, partitioned_datasets


def partition_netcdf(
    filepath: Union[str, Path], np_eta: int = 1, np_xi: int = 1
) -> None:
    """Partition a ROMS NetCDF file into smaller spatial tiles and save them to disk.

    This function divides the dataset in the specified NetCDF file into `np_eta` by `np_xi` tiles.
    Each tile is saved as a separate NetCDF file.

    Parameters
    ----------
    filepath : Union[str, Path]
        The path to the input NetCDF file.

    np_eta : int, optional
        The number of partitions along the `eta` direction. Must be a positive integer. Default is 1.

    np_xi : int, optional
        The number of partitions along the `xi` direction. Must be a positive integer. Default is 1.

    Returns
    -------
    List[Path]
        A list of Path objects for the filenames that were saved.
    """

    # Ensure filepath is a Path object
    filepath = Path(filepath)

    # Open the dataset
    ds = xr.open_dataset(filepath.with_suffix(".nc"))

    # Partition the dataset
    file_numbers, partitioned_datasets = partition(ds, np_eta=np_eta, np_xi=np_xi)

    # Generate paths to the partitioned files
    base_filepath = filepath.with_suffix("")
    paths_to_partitioned_files = [
        Path(f"{base_filepath}.{file_number}.nc") for file_number in file_numbers
    ]

    # Save the partitioned datasets to files
    xr.save_mfdataset(partitioned_datasets, paths_to_partitioned_files)

    return paths_to_partitioned_files


def _load_data(
    filename,
    dim_names,
    use_dask,
    time_chunking=True,
    decode_times=True,
    force_combine_nested=False,
):
    """Load dataset from the specified file.

    Parameters
    ----------
    filename : Union[str, Path, List[Union[str, Path]]]
        The path to the data file(s). Can be a single string (with or without wildcards), a single Path object,
        or a list of strings or Path objects containing multiple files.
    dim_names : Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
        Required only for lat-lon datasets to map dimension names like "latitude" and "longitude".
        For ROMS datasets, this parameter can be omitted, as default ROMS dimensions ("eta_rho", "xi_rho", "s_rho") are assumed.
    use_dask: bool
        Indicates whether to use dask for chunking. If True, data is loaded with dask; if False, data is loaded eagerly. Defaults to False.
    time_chunking : bool, optional
        If True and `use_dask=True`, the data will be chunked along the time dimension with a chunk size of 1.
        If False, the data will not be chunked explicitly along the time dimension, but will follow the default auto chunking scheme. This option is useful for ROMS restart files.
        Defaults to True.
    decode_times: bool, optional
        If True, decode times encoded in the standard NetCDF datetime format into datetime objects. Otherwise, leave them encoded as numbers.
        Defaults to True.
    force_combine_nested: bool, optional
        If True, forces the use of nested combination (`combine_nested`) regardless of whether wildcards are used.
        Defaults to False.

    Returns
    -------
    ds : xr.Dataset
        The loaded xarray Dataset containing the forcing data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If a list of files is provided but dim_names["time"] is not available or use_dask=False.
    """
    if dim_names is None:
        dim_names = {}

    # Precompile the regex for matching wildcard characters
    wildcard_regex = re.compile(r"[\*\?\[\]]")

    # Convert Path objects to strings
    if isinstance(filename, (str, Path)):
        filename_str = str(filename)
    elif isinstance(filename, list):
        filename_str = [str(f) for f in filename]
    else:
        raise ValueError("filename must be a string, Path, or a list of strings/Paths.")

    # Handle the case when filename is a string
    contains_wildcard = False
    if isinstance(filename_str, str):
        contains_wildcard = bool(wildcard_regex.search(filename_str))
        if contains_wildcard:
            matching_files = glob.glob(filename_str)
            if not matching_files:
                raise FileNotFoundError(
                    f"No files found matching the pattern '{filename_str}'."
                )
        else:
            matching_files = [filename_str]

    # Handle the case when filename is a list
    elif isinstance(filename_str, list):
        contains_wildcard = any(wildcard_regex.search(f) for f in filename_str)
        if contains_wildcard:
            matching_files = []
            for f in filename_str:
                files = glob.glob(f)
                if not files:
                    raise FileNotFoundError(
                        f"No files found matching the pattern '{f}'."
                    )
                matching_files.extend(files)
        else:
            matching_files = filename_str

    # Sort the matching files
    matching_files = sorted(matching_files)

    # Check if time dimension is available when multiple files are provided
    if isinstance(filename_str, list) and "time" not in dim_names:
        raise ValueError(
            "A list of files is provided, but time dimension is not available. "
            "A time dimension must be available to concatenate the files."
        )

    # Determine the kwargs for combining datasets
    if force_combine_nested:
        kwargs = {"combine": "nested", "concat_dim": dim_names["time"]}
    elif contains_wildcard or len(matching_files) == 1:
        kwargs = {"combine": "by_coords"}
    else:
        kwargs = {"combine": "nested", "concat_dim": dim_names["time"]}

    # Base kwargs used for dataset combination
    combine_kwargs = {
        "coords": "minimal",
        "compat": "override",
        "combine_attrs": "override",
    }

    if use_dask:

        if "latitude" in dim_names and "longitude" in dim_names:
            # for lat-lon datasets
            chunks = {
                dim_names["latitude"]: -1,
                dim_names["longitude"]: -1,
            }
        else:
            # For ROMS datasets
            chunks = {
                "eta_rho": -1,
                "eta_v": -1,
                "xi_rho": -1,
                "xi_u": -1,
                "s_rho": -1,
            }

        if "depth" in dim_names:
            chunks[dim_names["depth"]] = -1
        if "time" in dim_names and time_chunking:
            chunks[dim_names["time"]] = 1

        ds = xr.open_mfdataset(
            matching_files,
            decode_times=decode_times,
            chunks=chunks,
            **combine_kwargs,
            **kwargs,
        )

        # Rechunk the dataset along the tidal constituent dimension ("ntides") after loading
        # because the original dataset does not have a chunk size of 1 along this dimension.
        if "ntides" in dim_names:
            ds = ds.chunk({dim_names["ntides"]: 1})

    else:
        ds_list = []
        for file in matching_files:
            ds = xr.open_dataset(file, decode_times=decode_times, chunks=None)
            ds_list.append(ds)

        if kwargs["combine"] == "by_coords":
            ds = xr.combine_by_coords(ds_list, **combine_kwargs)
        elif kwargs["combine"] == "nested":
            ds = xr.combine_nested(
                ds_list, concat_dim=kwargs["concat_dim"], **combine_kwargs
            )

    if "time" in dim_names and dim_names["time"] not in ds.dims:
        ds = ds.expand_dims(dim_names["time"])

    return ds


def interpolate_from_rho_to_u(field, method="additive"):
    """Interpolates the given field from rho points to u points.

    This function performs an interpolation from the rho grid (cell centers) to the u grid
    (cell edges in the xi direction). Depending on the chosen method, it either averages
    (additive) or multiplies (multiplicative) the field values between adjacent rho points
    along the xi dimension. It also handles the removal of unnecessary coordinate variables
    and updates the dimensions accordingly.

    Parameters
    ----------
    field : xr.DataArray
        The input data array on the rho grid to be interpolated. It is assumed to have a dimension
        named "xi_rho".

    method : str, optional, default='additive'
        The method to use for interpolation. Options are:
        - 'additive': Average the field values between adjacent rho points.
        - 'multiplicative': Multiply the field values between adjacent rho points. Appropriate for
          binary masks.

    Returns
    -------
    field_interpolated : xr.DataArray
        The interpolated data array on the u grid with the dimension "xi_u".
    """

    if method == "additive":
        field_interpolated = 0.5 * (field + field.shift(xi_rho=1)).isel(
            xi_rho=slice(1, None)
        )
    elif method == "multiplicative":
        field_interpolated = (field * field.shift(xi_rho=1)).isel(xi_rho=slice(1, None))
    else:
        raise NotImplementedError(f"Unsupported method '{method}' specified.")

    vars_to_drop = ["lat_rho", "lon_rho", "eta_rho", "xi_rho"]
    for var in vars_to_drop:
        if var in field_interpolated.coords:
            field_interpolated = field_interpolated.drop_vars(var)

    field_interpolated = field_interpolated.swap_dims({"xi_rho": "xi_u"})

    return field_interpolated


def interpolate_from_rho_to_v(field, method="additive"):
    """Interpolates the given field from rho points to v points.

    This function performs an interpolation from the rho grid (cell centers) to the v grid
    (cell edges in the eta direction). Depending on the chosen method, it either averages
    (additive) or multiplies (multiplicative) the field values between adjacent rho points
    along the eta dimension. It also handles the removal of unnecessary coordinate variables
    and updates the dimensions accordingly.

    Parameters
    ----------
    field : xr.DataArray
        The input data array on the rho grid to be interpolated. It is assumed to have a dimension
        named "eta_rho".

    method : str, optional, default='additive'
        The method to use for interpolation. Options are:
        - 'additive': Average the field values between adjacent rho points.
        - 'multiplicative': Multiply the field values between adjacent rho points. Appropriate for
          binary masks.

    Returns
    -------
    field_interpolated : xr.DataArray
        The interpolated data array on the v grid with the dimension "eta_v".
    """

    if method == "additive":
        field_interpolated = 0.5 * (field + field.shift(eta_rho=1)).isel(
            eta_rho=slice(1, None)
        )
    elif method == "multiplicative":
        field_interpolated = (field * field.shift(eta_rho=1)).isel(
            eta_rho=slice(1, None)
        )
    else:
        raise NotImplementedError(f"Unsupported method '{method}' specified.")

    vars_to_drop = ["lat_rho", "lon_rho", "eta_rho", "xi_rho"]
    for var in vars_to_drop:
        if var in field_interpolated.coords:
            field_interpolated = field_interpolated.drop_vars(var)

    field_interpolated = field_interpolated.swap_dims({"eta_rho": "eta_v"})

    return field_interpolated


def transpose_dimensions(da: xr.DataArray) -> xr.DataArray:
    """Transpose the dimensions of an xarray.DataArray to ensure that 'time', any
    dimension starting with 's_', 'eta_', and 'xi_' are ordered first, followed by the
    remaining dimensions in their original order.

    Parameters
    ----------
    da : xarray.DataArray
        The input DataArray whose dimensions are to be reordered.

    Returns
    -------
    xarray.DataArray
        The DataArray with dimensions reordered so that 'time', 's_*', 'eta_*',
        and 'xi_*' are first, in that order, if they exist.
    """

    # List of preferred dimension patterns
    preferred_order = ["time", "s_", "eta_", "xi_"]

    # Get the existing dimensions in the DataArray
    dims = list(da.dims)

    # Collect dimensions that match any of the preferred patterns
    matched_dims = []
    for pattern in preferred_order:
        # Find dimensions that start with the pattern
        matched_dims += [dim for dim in dims if dim.startswith(pattern)]

    # Create a new order: first the matched dimensions, then the rest
    remaining_dims = [dim for dim in dims if dim not in matched_dims]
    new_order = matched_dims + remaining_dims

    # Transpose the DataArray to the new order
    transposed_da = da.transpose(*new_order)

    return transposed_da
