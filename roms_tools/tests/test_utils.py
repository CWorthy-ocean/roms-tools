from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from roms_tools.utils import (
    generate_focused_coordinate_range,
    get_dask_chunks,
    has_copernicus,
    has_dask,
    has_gcsfs,
    load_data,
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
