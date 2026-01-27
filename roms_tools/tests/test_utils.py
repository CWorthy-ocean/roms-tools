from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from roms_tools.datasets.download import download_test_data
from roms_tools.datasets.lat_lon_datasets import ERA5Correction
from roms_tools.utils import (
    _interpolate_generic,
    _path_list_from_input,
    generate_focused_coordinate_range,
    get_dask_chunks,
    has_copernicus,
    has_dask,
    has_gcsfs,
    interpolate_cyclic_time,
    interpolate_from_climatology,
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    interpolate_from_u_to_rho,
    interpolate_from_v_to_rho,
    load_data,
    rotate_velocities,
)


@pytest.mark.parametrize(
    "min_val, max_val, center, sc, N",
    [
        (-20.0, 5.5, -3.1, 1.0, 100),
        (100.0, 200.0, 150.0, 30.0, 100),
        (0.0, 2000.0, 150.0, 0.0, 100),
        (0.0, 2000.0, 150.0, 30.0, 100),
    ],
)
def test_coordinate_range_monotonicity(min_val, max_val, center, sc, N):
    centers, faces = generate_focused_coordinate_range(
        min_val=min_val, max_val=max_val, center=center, sc=sc, N=N
    )
    assert np.all(np.diff(faces) > 0), "faces is not strictly increasing"
    assert np.all(np.diff(centers) > 0), "centers is not strictly increasing"


class TestPathListFromInput:
    """A collection of tests for the _path_list_from_input function."""

    # Test cases that don't require I/O
    def test_list_of_strings(self):
        """Test with a list of file paths as strings."""
        files_list = ["path/to/file1.txt", "path/to/file2.txt"]
        result = _path_list_from_input(files_list)
        assert len(result) == 2
        assert result[0] == Path("path/to/file1.txt")
        assert result[1] == Path("path/to/file2.txt")

    def test_list_of_path_objects(self):
        """Test with a list of pathlib.Path objects."""
        files_list = [Path("file_a.txt"), Path("file_b.txt")]
        result = _path_list_from_input(files_list)
        assert len(result) == 2
        assert result[0] == Path("file_a.txt")
        assert result[1] == Path("file_b.txt")

    def test_single_path_object(self):
        """Test with a single pathlib.Path object."""
        file_path = Path("a_single_file.csv")
        result = _path_list_from_input(file_path)
        assert len(result) == 1
        assert result[0] == file_path

    def test_invalid_input_type_raises(self):
        """Test that an invalid input type raises a TypeError."""
        with pytest.raises(TypeError, match="'files' should be str, Path, or List"):
            _path_list_from_input(123)

    # Test cases that require I/O and `tmp_path`
    def test_single_file_as_str(self, tmp_path):
        """Test with a single file given as a string, requiring a file to exist."""
        p = tmp_path / "test_file.txt"
        p.touch()
        result = _path_list_from_input(str(p))
        assert len(result) == 1
        assert result[0] == p

    def test_wildcard_pattern(self, tmp_path, monkeypatch):
        """Test with a wildcard pattern, requiring files to exist, using monkeypatch."""
        # Setup
        d = tmp_path / "data"
        d.mkdir()
        (d / "file1.csv").touch()
        (d / "file2.csv").touch()
        (d / "other_file.txt").touch()

        # Action: Temporarily change the current working directory
        monkeypatch.chdir(tmp_path)

        result = _path_list_from_input("data/*.csv")

        # Assertion
        assert len(result) == 2
        assert result[0].name == "file1.csv"
        assert result[1].name == "file2.csv"

    def test_non_matching_pattern_raises(self, tmp_path):
        """Test that a non-matching pattern raises a FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No files matched"):
            _path_list_from_input(str(tmp_path / "non_existent_file_*.txt"))


def test_has_dask() -> None:
    """Verify that dask existence is correctly reported when found."""
    with mock.patch("roms_tools.utils.find_spec", return_value=mock.MagicMock):
        assert has_dask()


def test_has_dask_error_when_missing() -> None:
    """Verify that dask existence is correctly reported when not found."""
    with mock.patch("roms_tools.utils.find_spec", return_value=None):
        assert not has_dask()


def test_has_gcfs() -> None:
    """Verify that GCFS existence is correctly reported when found."""
    with mock.patch("roms_tools.utils.find_spec", return_value=mock.MagicMock):
        assert has_gcsfs()


def test_has_gcfs_error_when_missing() -> None:
    """Verify that GCFS existence is correctly reported when not found."""
    with mock.patch("roms_tools.utils.find_spec", return_value=None):
        assert not has_gcsfs()


def test_has_copernicus() -> None:
    """Verify that copernicus existence is correctly reported when found."""
    with mock.patch("roms_tools.utils.find_spec", return_value=mock.MagicMock):
        assert has_copernicus()


def test_has_copernicus_error_when_missing() -> None:
    """Verify that copernicus existence is correctly reported when not found."""
    with mock.patch("roms_tools.utils.find_spec", return_value=None):
        assert not has_copernicus()


def test_load_data_dask_not_found() -> None:
    """Verify that load data raises an exception when dask is requested and missing."""
    with (
        mock.patch("roms_tools.utils.has_dask", return_value=False),
        pytest.raises(RuntimeError),
    ):
        load_data("foo.zarr", {"a": "a"}, use_dask=True)


def test_load_data_open_zarr_without_dask() -> None:
    """Verify that load data raises an exception when zarr is requested without dask."""
    with (
        mock.patch("roms_tools.utils.has_dask", return_value=False),
        pytest.raises(ValueError),
    ):
        # read_zarr should require use_dask to be True
        load_data("foo.zarr", {"a": ""}, use_dask=False, read_zarr=True)


@pytest.mark.parametrize(
    ("dataset_name", "expected_dim"),
    [
        ("surface_forcing", "time"),
        ("bgc_surface_forcing", "time"),
        ("tidal_forcing", "eta_rho"),
        ("coarse_surface_forcing", "eta_rho"),
    ],
)
def test_load_data_open_dataset(
    dataset_name: str,
    expected_dim: str,
    get_test_data_path: Callable[[str], Path],
) -> None:
    """Verify that a zarr file is correctly loaded when not using Dask.

    This must use xr.open_dataset
    """
    ds_path = get_test_data_path(dataset_name)

    with mock.patch(
        "roms_tools.utils.xr.open_dataset",
        wraps=xr.open_dataset,
    ) as fn_od:
        ds = load_data(
            ds_path,
            {"latitude": "latitude"},
            use_dask=False,
        )
        assert fn_od.called

    assert expected_dim in ds.dims


# test get_dask_chunks


def test_latlon_default_chunks():
    dim_names = {"latitude": "lat", "longitude": "lon"}
    expected = {"lat": -1, "lon": -1}
    result = get_dask_chunks(dim_names)
    assert result == expected


def test_latlon_with_depth_and_time():
    dim_names = {"latitude": "lat", "longitude": "lon", "depth": "z", "time": "t"}
    expected = {"lat": -1, "lon": -1, "z": -1, "t": 1}
    result = get_dask_chunks(dim_names)
    assert result == expected


def test_latlon_with_time_chunking_false():
    dim_names = {"latitude": "lat", "longitude": "lon", "time": "t"}
    expected = {"lat": -1, "lon": -1}
    result = get_dask_chunks(dim_names, time_chunking=False)
    assert result == expected


def test_roms_default_chunks():
    dim_names = {}
    expected_keys = {"eta_rho", "eta_v", "xi_rho", "xi_u", "s_rho"}
    result = get_dask_chunks(dim_names)
    assert set(result.keys()) == expected_keys
    assert all(v == -1 for v in result.values())


def test_roms_with_depth_and_time():
    dim_names = {"depth": "s_rho", "time": "ocean_time"}
    result = get_dask_chunks(dim_names)
    # ROMS default keys + depth + time
    expected_keys = {"eta_rho", "eta_v", "xi_rho", "xi_u", "s_rho", "ocean_time"}
    assert set(result.keys()) == expected_keys
    assert result["ocean_time"] == 1
    assert result["s_rho"] == -1


def test_roms_with_ntides():
    dim_names = {"ntides": "nt"}
    result = get_dask_chunks(dim_names)
    assert result["nt"] == 1


def test_time_chunking_false_roms():
    dim_names = {"time": "ocean_time"}
    result = get_dask_chunks(dim_names, time_chunking=False)
    assert "ocean_time" not in result


# test interpolate_from_climatology


@pytest.fixture
def climatology_data():
    """Create a simple annual cycle dataset with 12 time points."""
    time_coord = np.arange(1, 13)  # months as day_of_year approximation
    da = xr.DataArray(np.arange(12), dims=("time",), coords={"time": time_coord})
    ds = xr.Dataset({"var1": da, "var2": da * 2})
    return da, ds, "time", "time"


def test_interpolate_dataarray_single_time(climatology_data):
    da, _, time_dim, time_coord = climatology_data
    target_time = pd.Timestamp("2000-03-15")  # day_of_year ~ 75
    interpolated = interpolate_from_climatology(da, time_dim, time_coord, target_time)
    assert isinstance(interpolated, xr.DataArray)
    assert interpolated.sizes[time_dim] == 1


def test_interpolate_dataset_multiple_times(climatology_data):
    _, ds, time_dim, time_coord = climatology_data
    target_times = pd.date_range("2000-01-01", periods=3, freq="ME")
    interpolated = interpolate_from_climatology(ds, time_dim, time_coord, target_times)
    assert isinstance(interpolated, xr.Dataset)
    assert all(interpolated[var].sizes[time_dim] == 3 for var in interpolated.data_vars)


def test_interpolate_dataarray_time_dim_not_equal_time_coord():
    time_values = np.arange(1, 13)
    da = xr.DataArray(
        np.arange(12),
        dims=("time_dim",),
        coords={"time_coord": ("time_dim", time_values)},
    )
    target_time = pd.Timestamp("2000-06-15")
    interpolated = interpolate_from_climatology(
        da, time_dim="time_dim", time_coord="time_coord", time=target_time
    )
    assert interpolated.sizes["time_dim"] == 1
    assert np.issubdtype(interpolated.dtype, np.number)


def test_interpolate_cyclic_time_basic():
    time_values = np.arange(1, 13)
    da = xr.DataArray(np.arange(12), dims=("time",), coords={"time": time_values})
    target_days = [0.5, 6.5, 12.5]  # fractional days, include cyclic behavior
    interpolated = interpolate_cyclic_time(
        da, time_dim="time", time_coord="time", day_of_year=target_days
    )
    assert isinstance(interpolated, xr.DataArray)
    assert interpolated.sizes["time"] == len(target_days)


def test_interpolate_from_climatology_invalid_input():
    with pytest.raises(TypeError):
        interpolate_from_climatology(
            "not a dataset", "time", "time", pd.Timestamp("2000-01-01")
        )


def test_interpolate_from_real_climatology(use_dask):
    fname = download_test_data("ERA5_regional_test_data.nc")
    era5_times = xr.open_dataset(fname).time

    climatology = ERA5Correction(use_dask=use_dask)
    field = climatology.ds["ssr_corr"]
    field["time"] = field["time"].dt.days

    interpolated_field = interpolate_from_climatology(field, "time", "time", era5_times)
    assert len(interpolated_field.time) == len(era5_times)


# test _interpolate_generic and its wrappers

# -------------------------
# Fixtures
# -------------------------


@pytest.fixture
def sample_rho_field() -> xr.DataArray:
    """Create a simple rho-point field for testing."""
    data = np.arange(12, dtype=float).reshape(3, 4)
    eta = np.arange(3)
    xi = np.arange(4)

    return xr.DataArray(
        data,
        dims=("eta_rho", "xi_rho"),
        coords={
            "lat_rho": (("eta_rho", "xi_rho"), eta[:, None] * np.ones((1, 4))),
            "lon_rho": (("eta_rho", "xi_rho"), np.ones((3, 1)) * xi[None, :]),
        },
    )


@pytest.fixture
def sample_u_field() -> xr.DataArray:
    """Create a simple u-point field for testing."""
    data = np.arange(9, dtype=float).reshape(3, 3)
    eta = np.arange(3)
    xi = np.arange(3)

    return xr.DataArray(
        data,
        dims=("eta_rho", "xi_u"),
        coords={
            "lat_u": (("eta_rho", "xi_u"), eta[:, None] * np.ones((1, 3))),
            "lon_u": (("eta_rho", "xi_u"), np.ones((3, 1)) * xi[None, :]),
        },
    )


@pytest.fixture
def sample_v_field() -> xr.DataArray:
    """Create a simple v-point field for testing."""
    data = np.arange(8, dtype=float).reshape(2, 4)
    eta = np.arange(2)
    xi = np.arange(4)

    return xr.DataArray(
        data,
        dims=("eta_v", "xi_rho"),
        coords={
            "lat_v": (("eta_v", "xi_rho"), eta[:, None] * np.ones((1, 4))),
            "lon_v": (("eta_v", "xi_rho"), np.ones((2, 1)) * xi[None, :]),
        },
    )


# -------------------------
# Generic interpolation tests
# -------------------------


def test_interpolate_from_rho_to_u_additive(sample_rho_field: xr.DataArray):
    result = _interpolate_generic(
        sample_rho_field, dim_in="xi_rho", dim_out="xi_u", method="additive"
    )

    # One fewer point along xi
    assert result.shape[1] == sample_rho_field.shape[1] - 1

    expected = 0.5 * (sample_rho_field.values[:, 1:] + sample_rho_field.values[:, :-1])
    np.testing.assert_allclose(result.values, expected)


def test_interpolate_from_rho_to_u_multiplicative(sample_rho_field: xr.DataArray):
    result = _interpolate_generic(
        sample_rho_field, dim_in="xi_rho", dim_out="xi_u", method="multiplicative"
    )

    expected = sample_rho_field.values[:, 1:] * sample_rho_field.values[:, :-1]
    np.testing.assert_allclose(result.values, expected)


# -------------------------
# Wrapper tests
# -------------------------


def test_rho_to_u_wrapper_additive(sample_rho_field: xr.DataArray):
    result = interpolate_from_rho_to_u(sample_rho_field, method="additive")

    # Dimension swap
    assert "xi_u" in result.dims
    assert "xi_rho" not in result.dims

    # Coordinates dropped
    for coord in ("lat_rho", "lon_rho"):
        assert coord not in result.coords

    # Shape check
    assert result.sizes["xi_u"] == sample_rho_field.sizes["xi_rho"] - 1


def test_rho_to_v_wrapper_additive(sample_rho_field: xr.DataArray):
    result = interpolate_from_rho_to_v(sample_rho_field, method="additive")

    # Dimension swap
    assert "eta_v" in result.dims
    assert "eta_rho" not in result.dims

    # Coordinates dropped
    for coord in ("lat_rho", "lon_rho"):
        assert coord not in result.coords

    # Shape check
    assert result.sizes["eta_v"] == sample_rho_field.sizes["eta_rho"] - 1


def test_u_to_rho_wrapper_additive(sample_u_field: xr.DataArray):
    result = interpolate_from_u_to_rho(sample_u_field, method="additive")

    # Dimension swap
    assert "xi_rho" in result.dims
    assert "xi_u" not in result.dims

    # Coordinates dropped
    for coord in ("lat_u", "lon_u"):
        assert coord not in result.coords

    # Shape: one more along xi due to padding
    assert result.sizes["xi_rho"] == sample_u_field.sizes["xi_u"] + 1


def test_v_to_rho_wrapper_additive(sample_v_field: xr.DataArray):
    result = interpolate_from_v_to_rho(sample_v_field, method="additive")

    # Dimension swap
    assert "eta_rho" in result.dims
    assert "eta_v" not in result.dims

    # Coordinates dropped
    for coord in ("lat_v", "lon_v"):
        assert coord not in result.coords

    # Shape: one more along eta due to padding
    assert result.sizes["eta_rho"] == sample_v_field.sizes["eta_v"] + 1


# -------------------------
# Error handling
# -------------------------


def test_invalid_method_raises(
    sample_rho_field: xr.DataArray,
    sample_u_field: xr.DataArray,
    sample_v_field: xr.DataArray,
):
    with pytest.raises(NotImplementedError):
        interpolate_from_rho_to_u(sample_rho_field, method="unsupported")

    with pytest.raises(NotImplementedError):
        interpolate_from_rho_to_v(sample_rho_field, method="unsupported")

    with pytest.raises(NotImplementedError):
        interpolate_from_u_to_rho(sample_u_field, method="unsupported")

    with pytest.raises(NotImplementedError):
        interpolate_from_v_to_rho(sample_v_field, method="unsupported")


# Test rotate_velocities
@pytest.fixture
def sample_velocities_centered():
    """Create a centered-grid velocity field with random values and grid angle."""
    np.random.seed(42)  # For reproducibility

    eta_rho, xi_rho = 10, 15

    u = xr.DataArray(
        np.random.rand(eta_rho, xi_rho),
        dims=("eta_rho", "xi_rho"),
        coords={
            "eta_rho": np.arange(eta_rho),
            "xi_rho": np.arange(xi_rho),
        },
    )

    v = xr.DataArray(
        np.random.rand(eta_rho, xi_rho),
        dims=("eta_rho", "xi_rho"),
        coords={
            "eta_rho": np.arange(eta_rho),
            "xi_rho": np.arange(xi_rho),
        },
    )

    angle = xr.DataArray(
        np.random.rand(eta_rho, xi_rho) * np.pi / 2
        - np.pi / 4,  # random angles in [-45°, 45°]
        dims=("eta_rho", "xi_rho"),
        coords={
            "eta_rho": np.arange(eta_rho),
            "xi_rho": np.arange(xi_rho),
        },
    )

    return u, v, angle


def test_rotate_velocities_roundtrip(sample_velocities_centered):
    """Test rotation to grid and back recovers original velocities."""
    u, v, angle = sample_velocities_centered

    # Rotate forward: lat-lon → model grid
    u_rot, v_rot = rotate_velocities(
        u, v, angle, interpolate_before=False, interpolate_after=False
    )

    # Rotate backward: model grid → lat-lon
    u_back, v_back = rotate_velocities(
        u_rot, v_rot, -angle, interpolate_before=False, interpolate_after=False
    )

    np.testing.assert_allclose(u.values, u_back.values)
    np.testing.assert_allclose(v.values, v_back.values)
