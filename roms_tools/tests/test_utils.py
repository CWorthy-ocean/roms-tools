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
    _path_list_from_input,
    generate_focused_coordinate_range,
    get_dask_chunks,
    has_copernicus,
    has_dask,
    has_gcsfs,
    interpolate_cyclic_time,
    interpolate_from_climatology,
    load_data,
    wrap_longitudes,
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
    target_times = pd.date_range("2000-01-01", periods=3, freq="M")
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


def test_wrap_longitudes_staggered():
    # Dimensions
    eta_rho, xi_rho, xi_u = 3, 4, 5
    eta_v = 2

    # Create 2D coordinates
    lon_rho = xr.DataArray(
        np.linspace(0, 360, eta_rho * xi_rho).reshape(eta_rho, xi_rho),
        dims=("eta_rho", "xi_rho"),
        attrs={"units": "degrees_east"},
    )
    lon_u = xr.DataArray(
        np.linspace(-190, 190, eta_rho * xi_u).reshape(eta_rho, xi_u),
        dims=("eta_rho", "xi_u"),
        attrs={"units": "degrees_east"},
    )
    lon_v = xr.DataArray(
        np.linspace(-180, 180, eta_v * xi_rho).reshape(eta_v, xi_rho),
        dims=("eta_v", "xi_rho"),
        attrs={"units": "degrees_east"},
    )

    # Dummy variables
    ds = xr.Dataset(
        {
            "dummy_rho": (("eta_rho", "xi_rho"), np.zeros((eta_rho, xi_rho))),
            "dummy_u": (("eta_rho", "xi_u"), np.zeros((eta_rho, xi_u))),
            "dummy_v": (("eta_v", "xi_rho"), np.zeros((eta_v, xi_rho))),
        },
        coords={"lon_rho": lon_rho, "lon_u": lon_u, "lon_v": lon_v},
    )

    # Wrap to [-180, 180]
    ds_wrapped = wrap_longitudes(ds, straddle=True)

    # Check values: all >180 should be shifted
    assert ds_wrapped.lon_rho.max().values <= 180
    assert ds_wrapped.lon_u.max().values <= 180
    assert ds_wrapped.lon_v.max().values <= 180

    # Wrap to [0, 360]
    ds_wrapped2 = wrap_longitudes(ds, straddle=False)
    assert ds_wrapped2.lon_rho.min().values >= 0
    assert ds_wrapped2.lon_u.min().values >= 0
    assert ds_wrapped2.lon_v.min().values >= 0

    # Check attributes preserved
    for name in ["lon_rho", "lon_u", "lon_v"]:
        assert ds.coords[name].attrs == ds_wrapped.coords[name].attrs
