import glob
import logging
import re
import warnings
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import xarray as xr

from roms_tools.constants import R_EARTH


def _load_data(
    filename,
    dim_names,
    use_dask,
    time_chunking=True,
    decode_times=True,
    force_combine_nested=False,
    read_zarr: bool = False,
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
        If True, decode times and timedeltas encoded in the standard NetCDF datetime format into datetime objects. Otherwise, leave them encoded as numbers.
        Defaults to True.
    force_combine_nested: bool, optional
        If True, forces the use of nested combination (`combine_nested`) regardless of whether wildcards are used.
        Defaults to False.
    read_zarr: bool, optional
        If True, use the zarr engine to read the dataset, and don't use mfdataset.
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

    if use_dask:
        if not _has_dask():
            raise RuntimeError(
                "Dask is required but not installed. Install it with:\n"
                "  • `pip install roms-tools[dask]` or\n"
                "  • `conda install dask`\n"
                "Alternatively, install `roms-tools` with conda to include all dependencies."
            )
    if read_zarr:
        if isinstance(filename, list):
            raise ValueError("read_zarr requires a single path, not a list of paths")
        if not use_dask:
            raise ValueError("read_zarr must be used with use_dask")

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
        if "ntides" in dim_names:
            chunks[dim_names["ntides"]] = 1

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=r"^The specified chunks separate.*",
            )

            if read_zarr:
                ds = xr.open_zarr(
                    matching_files[0],
                    decode_times=decode_times,
                    chunks=chunks,
                    consolidated=None,
                    storage_options=dict(token="anon"),
                )
            else:
                ds = xr.open_mfdataset(
                    matching_files,
                    decode_times=decode_times,
                    decode_timedelta=decode_times,
                    chunks=chunks,
                    **combine_kwargs,
                    **kwargs,
                )

    else:
        ds_list = []
        for file in matching_files:
            ds = xr.open_dataset(
                file,
                decode_times=decode_times,
                decode_timedelta=decode_times,
                chunks=None,
            )
            ds_list.append(ds)

        if kwargs["combine"] == "by_coords":
            ds = xr.combine_by_coords(ds_list, **combine_kwargs)
        elif kwargs["combine"] == "nested":
            ds = xr.combine_nested(
                ds_list, concat_dim=kwargs["concat_dim"], **combine_kwargs
            )

    if "time" in dim_names and dim_names["time"] not in ds.dims:
        ds = ds.expand_dims(dim_names["time"])

    if "time" in dim_names and not read_zarr:
        ds = ds.drop_duplicates(dim=dim_names["time"])

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


def save_datasets(dataset_list, output_filenames, use_dask=False, verbose=True):
    """Save the list of datasets to netCDF4 files.

    Parameters
    ----------
    dataset_list : list
        List of datasets to be saved.
    output_filenames : list
        List of filenames for the output files.
    use_dask : bool, optional
        Whether to use Dask diagnostics (e.g., progress bars) when saving the datasets, by default False.
    verbose : bool, optional
        Whether to log information about the files being written. If True, logs the output filenames.
        Defaults to True.

    Returns
    -------
    List[Path]
        A list of Path objects for the filenames that were saved.
    """

    saved_filenames = []

    output_filenames = [f"{filename}.nc" for filename in output_filenames]
    if verbose:
        logging.info(
            "Writing the following NetCDF files:\n%s", "\n".join(output_filenames)
        )

    if use_dask:
        from dask.diagnostics import ProgressBar

        with ProgressBar():
            xr.save_mfdataset(dataset_list, output_filenames)
    else:
        xr.save_mfdataset(dataset_list, output_filenames)

    saved_filenames.extend(Path(f) for f in output_filenames)

    return saved_filenames


def get_dask_chunks(location, chunk_size):
    """Returns the appropriate Dask chunking dictionary based on grid location.

    Parameters
    ----------
    location : str
        The grid location, one of "rho", "u", or "v".
    chunk_size : int
        The chunk size to apply.

    Returns
    -------
    dict
        Dictionary specifying the chunking strategy.
    """
    chunk_mapping = {
        "rho": {"eta_rho": chunk_size, "xi_rho": chunk_size},
        "u": {"eta_rho": chunk_size, "xi_u": chunk_size},
        "v": {"eta_v": chunk_size, "xi_rho": chunk_size},
    }
    return chunk_mapping.get(location, {})


def _generate_coordinate_range(min_val: float, max_val: float, resolution: float):
    """Generate an array of target coordinates (e.g., latitude or longitude) within a
    specified range, with a resolution that is rounded to the nearest value of the form
    `1/n` (or integer).

    This method generates an array of target coordinates between the provided `min_val` and `max_val`
    values, ensuring that both `min_val` and `max_val` are included in the resulting range. The resolution
    is rounded to the nearest fraction of the form `1/n` or an integer, based on the input.

    Parameters
    ----------
    min_val : float
        The minimum value (in degrees) of the coordinate range (inclusive).

    max_val : float
        The maximum value (in degrees) of the coordinate range (inclusive).

    resolution : float
        The spacing (in degrees) between each coordinate in the array. The resolution will
        be rounded to the nearest value of the form `1/n` or an integer, depending on the size
        of the resolution value.

    Returns
    -------
    numpy.ndarray
        An array of target coordinates generated from the specified range, with the resolution
        rounded to a suitable fraction (e.g., `1/n`) or integer, depending on the input resolution.
    """

    # Find the closest fraction of the form 1/n or integer to match the resolution
    resolution_rounded = None
    min_diff = float("inf")  # Initialize the minimum difference as infinity

    # Search for the best fraction or integer approximation to the resolution
    for n in range(1, 1000):  # Try fractions 1/n, where n ranges from 1 to 999
        if resolution <= 1:
            fraction = (
                1 / n
            )  # For small resolutions (<= 1), consider fractions of the form 1/n
        else:
            fraction = n  # For larger resolutions (>1), consider integers (n)

        diff = abs(
            fraction - resolution
        )  # Calculate the difference between the fraction and the resolution

        if diff < min_diff:  # If the current fraction is a better approximation
            min_diff = diff
            resolution_rounded = fraction  # Update the best fraction (or integer) found

    # Adjust the start and end of the range to include integer values
    start_int = np.floor(min_val)  # Round the minimum value down to the nearest integer
    end_int = np.ceil(max_val)  # Round the maximum value up to the nearest integer

    # Generate the array of target coordinates, including both the min and max values
    target = np.arange(start_int, end_int + resolution_rounded, resolution_rounded)

    # Truncate any values that exceed max (including small floating point errors)
    target = target[target <= end_int + 1e-10]

    return target.astype(np.float32)


def _generate_focused_coordinate_range(
    center: float,
    sc: float,
    min_val: float,
    max_val: float,
    N: int,
    stretch_factor: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an array of target coordinates within [min_val, max_val] with two.

    resolution zones: a higher resolution region centered near `center` ± stretch_factor
    * `sc`, and coarser resolution outside.

    Parameters
    ----------
    min_val : float
        Minimum coordinate value (inclusive).
        Units: degrees or meters.
    max_val : float
        Maximum coordinate value (inclusive).
        Units: degrees or meters.
    center : float
        Coordinate around which to increase resolution.
        Units: degrees or meters.
    sc : float
        Width controlling the size of the high-resolution region.
        If 0, treated as a small default to avoid zero-width window.
        Units: degrees or meters.
    N : int
        Number of desired coordinate points.
    stretch_factor : float, default 3.0
        Multiplier for `sc` to define the width of the high-resolution window.

    Returns
    -------
    tuple of np.ndarray
        Tuple of two arrays:
        - centers: cell centers.
        - faces: cell faces.

    Notes
    -----
    - About 20% of the faces are placed in the high-resolution region.
    - Remaining faces are split evenly in the coarse resolution regions
    - Faces and centers are returned sorted and unique.
    """

    # Define the bounds of the high-res region
    if sc == 0.0:
        sc = 1
    low_bound = max(center - stretch_factor * sc, min_val)
    high_bound = min(center + stretch_factor * sc, max_val)

    # Decide how many points for high and low resolution
    high_res_points = int(N * 0.2)  # 20% of the points in high-res region
    low_res_points = N + 1 - high_res_points * 2  # rest split

    # First coarse region
    first_coarse_res_faces = (
        np.linspace(min_val, low_bound, low_res_points // 2, endpoint=False)
        if low_bound > 0
        else np.array([])
    )
    # High-res region faces (low_bound to high_bound)
    high_res_faces = np.linspace(low_bound, high_bound, high_res_points, endpoint=False)
    # Second coarse region
    second_coarse_res_faces = (
        np.linspace(
            high_bound,
            max_val,
            low_res_points - len(first_coarse_res_faces),
            endpoint=True,
        )
        if high_bound < max_val
        else np.array([])
    )

    faces = np.unique(
        np.concatenate(
            [first_coarse_res_faces, high_res_faces, second_coarse_res_faces]
        )
    )
    centers = (faces[:-1] + faces[1:]) / 2

    return centers, faces


def _remove_edge_nans(
    field: xr.DataArray, xdim: str, layer_depth: xr.DataArray | None = None
) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Remove NaN-only slices at the edges of a specified dimension.

    This function trims leading and trailing slices along the specified `xdim` where all values
    are NaN. It assumes that the data has only one other relevant dimension (typically depth).
    If `layer_depth` is provided, it is used instead of `field` to determine which slices are NaN-only.

    Parameters
    ----------
    field : xr.DataArray
        Input array to trim. May contain NaNs at the edges along `xdim`.
    xdim : str
        The dimension along which to remove NaN-only slices.
    layer_depth : xr.DataArray, optional
        Optional array to evaluate NaN positions. If not provided, `field` is used instead.

    Returns
    -------
    field : xr.DataArray
        Trimmed `field`, with leading and trailing NaN-only slices removed along `xdim`.
    layer_depth : xr.DataArray or None
        Trimmed `layer_depth`, if provided. Otherwise, returns None.

    Raises
    ------
    ValueError
        If `field` has more than one additional dimension besides `xdim`.

    Notes
    -----
    - If `xdim` is not in `field.dims`, no trimming is performed.
    - This is typically used for visualizing or extracting clean sections from 2D slices
      (e.g., vertical sections) that have NaNs at the spatial boundaries.
    """
    if xdim in field.dims:
        other_dims = [dim for dim in field.dims if dim != xdim]
        if len(other_dims) == 0:
            if layer_depth is not None:
                nan_mask = layer_depth.isnull()
            else:
                nan_mask = field.isnull()

        elif len(other_dims) == 1:

            depth_dim = other_dims[0]

            if layer_depth is not None:
                if depth_dim in layer_depth.dims:
                    nan_mask = layer_depth.isnull().sum(dim=depth_dim)
                else:
                    nan_mask = layer_depth.isnull()
            else:
                N = field.sizes[depth_dim]
                nan_mask = field.isnull().sum(dim=depth_dim) == N

        else:
            raise ValueError(
                f"Cannot trim along dimension '{xdim}': expected at most one other dimension, "
                f"but got {len(other_dims)} ({other_dims})."
            )

        valid_indices = np.where(nan_mask.values == 0)[0]

        if len(valid_indices) > 0:
            first_valid = valid_indices[0]
            last_valid = valid_indices[-1]
            field = field.isel({xdim: slice(first_valid, last_valid + 1)})
            if layer_depth is not None:
                layer_depth = layer_depth.isel(
                    {xdim: slice(first_valid, last_valid + 1)}
                )

    return field, layer_depth


def _has_dask() -> bool:
    return find_spec("dask") is not None


def _has_gcsfs() -> bool:
    return find_spec("gcsfs") is not None


def normalize_longitude(lon: float, straddle: bool) -> float:
    """Normalize longitude to the appropriate range depending on whether the grid
    straddles the dateline.

    Parameters
    ----------
    lon : float
        Longitude in degrees (can be any value, including multiples of 360 or negative values).
    straddle : bool
        Whether the grid straddles the dateline. If True, output will be in (-180, 180];
        if False, output will be in [0, 360).
    Returns
    -------
    float
        Normalized longitude.
    """
    lon = lon % 360
    if straddle:
        return lon - 360 if lon > 180 else lon
    else:
        return lon + 360 if lon < 0 else lon


def infer_nominal_horizontal_resolution(
    grid_ds: xr.Dataset, lat: float | None = None
) -> float:
    """Estimate the nominal horizontal resolution of the ROMS grid in degrees at a
    specified latitude.

    This function calculates the average horizontal grid spacing (based on `pm` and `pn`),
    then converts that spacing from meters to degrees longitude, accounting for Earth's curvature
    and the cosine of the latitude.

    Parameters
    ----------
    grid_ds : xr.Dataset
        ROMS grid dataset containing `pm`, `pn`, and `lat_rho` coordinates.
    lat : float, optional
        Latitude (in degrees) at which to estimate resolution. If None, the average latitude
        over the grid is used.

    Returns
    -------
    float
        Estimated nominal horizontal resolution in degrees longitude.
    """
    if lat is None:
        # Use center of the domain in latitude
        lat = float((grid_ds.lat_rho.max() + grid_ds.lat_rho.min()) / 2)

    lat_rad = np.deg2rad(lat)

    # Mean grid spacing in meters (pm and pn are in 1/m)
    resolution_in_m = ((1 / grid_ds.pm).mean() + (1 / grid_ds.pn).mean()) / 2

    # Meters per degree at the equator
    meters_per_degree = 2 * np.pi * R_EARTH / 360

    # Adjust resolution to degrees longitude at the specified latitude
    resolution_in_degrees = resolution_in_m / (meters_per_degree * np.cos(lat_rad))

    return float(resolution_in_degrees)
