import logging
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from roms_tools.download import download_test_data
from roms_tools.setup.datasets import (
    CESMBGCDataset,
    Dataset,
    ERA5Correction,
    GLORYSDataset,
    TPXODataset,
)


@pytest.fixture
def global_dataset():
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 90, 180)
    depth = np.linspace(0, 2000, 10)
    time = [
        np.datetime64("2022-01-01T00:00:00"),
        np.datetime64("2022-02-01T00:00:00"),
        np.datetime64("2022-03-01T00:00:00"),
        np.datetime64("2022-04-01T00:00:00"),
    ]
    data = np.random.rand(4, 10, 180, 360)
    ds = xr.Dataset(
        {"var": (["time", "depth", "latitude", "longitude"], data)},
        coords={
            "time": (["time"], time),
            "depth": (["depth"], depth),
            "latitude": (["latitude"], lat),
            "longitude": (["longitude"], lon),
        },
    )
    return ds


@pytest.fixture
def global_dataset_with_noon_times():
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 90, 180)
    time = [
        np.datetime64("2022-01-01T12:00:00"),
        np.datetime64("2022-02-01T12:00:00"),
        np.datetime64("2022-03-01T12:00:00"),
        np.datetime64("2022-04-01T12:00:00"),
    ]
    data = np.random.rand(4, 180, 360)
    ds = xr.Dataset(
        {"var": (["time", "latitude", "longitude"], data)},
        coords={
            "time": (["time"], time),
            "latitude": (["latitude"], lat),
            "longitude": (["longitude"], lon),
        },
    )
    return ds


@pytest.fixture
def global_dataset_with_multiple_times_per_day():
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 90, 180)
    time = [
        np.datetime64("2022-01-01T00:00:00"),
        np.datetime64("2022-01-01T12:00:00"),
        np.datetime64("2022-02-01T00:00:00"),
        np.datetime64("2022-02-01T12:00:00"),
        np.datetime64("2022-03-01T00:00:00"),
        np.datetime64("2022-03-01T12:00:00"),
        np.datetime64("2022-04-01T00:00:00"),
        np.datetime64("2022-04-01T12:00:00"),
    ]
    data = np.random.rand(8, 180, 360)
    ds = xr.Dataset(
        {"var": (["time", "latitude", "longitude"], data)},
        coords={
            "time": (["time"], time),
            "latitude": (["latitude"], lat),
            "longitude": (["longitude"], lon),
        },
    )
    return ds


@pytest.fixture
def non_global_dataset():
    lon = np.linspace(0, 180, 181)
    lat = np.linspace(-90, 90, 180)
    data = np.random.rand(180, 181)
    ds = xr.Dataset(
        {"var": (["latitude", "longitude"], data)},
        coords={"latitude": (["latitude"], lat), "longitude": (["longitude"], lon)},
    )
    return ds


@pytest.mark.parametrize(
    "data_fixture, expected_time_values",
    [
        (
            "global_dataset",
            [
                np.datetime64("2022-02-01T00:00:00"),
                np.datetime64("2022-03-01T00:00:00"),
            ],
        ),
        (
            "global_dataset_with_noon_times",
            [
                np.datetime64("2022-01-01T12:00:00"),
                np.datetime64("2022-02-01T12:00:00"),
                np.datetime64("2022-03-01T12:00:00"),
            ],
        ),
        (
            "global_dataset_with_multiple_times_per_day",
            [
                np.datetime64("2022-02-01T00:00:00"),
                np.datetime64("2022-02-01T12:00:00"),
                np.datetime64("2022-03-01T00:00:00"),
            ],
        ),
    ],
)
def test_select_times(data_fixture, expected_time_values, request, tmp_path, use_dask):
    """Test selecting times with different datasets."""
    start_time = datetime(2022, 2, 1)
    end_time = datetime(2022, 3, 1)

    # Get the fixture dynamically based on the parameter
    dataset = request.getfixturevalue(data_fixture)

    filepath = tmp_path / "test.nc"
    dataset.to_netcdf(filepath)
    dataset = Dataset(
        filename=filepath,
        var_names={"var": "var"},
        start_time=start_time,
        end_time=end_time,
        use_dask=use_dask,
    )

    assert dataset.ds is not None
    assert len(dataset.ds.time) == len(expected_time_values)
    for expected_time in expected_time_values:
        assert expected_time in dataset.ds.time.values


@pytest.mark.parametrize(
    "data_fixture, expected_time_values",
    [
        ("global_dataset", [np.datetime64("2022-02-01T00:00:00")]),
        ("global_dataset_with_noon_times", [np.datetime64("2022-02-01T12:00:00")]),
    ],
)
def test_select_times_valid_start_no_end_time(
    data_fixture, expected_time_values, request, tmp_path, use_dask
):
    """Test selecting times with only start_time specified."""
    start_time = datetime(2022, 2, 1)

    # Get the fixture dynamically based on the parameter
    dataset = request.getfixturevalue(data_fixture)

    # Create a temporary file
    filepath = tmp_path / "test.nc"
    dataset.to_netcdf(filepath)

    # Instantiate Dataset object using the temporary file
    dataset = Dataset(
        filename=filepath,
        var_names={"var": "var"},
        start_time=start_time,
        use_dask=use_dask,
    )

    assert dataset.ds is not None
    assert len(dataset.ds.time) == len(expected_time_values)
    for expected_time in expected_time_values:
        assert expected_time in dataset.ds.time.values


@pytest.mark.parametrize(
    "data_fixture, expected_time_values",
    [
        ("global_dataset", [np.datetime64("2022-02-01T00:00:00")]),
        ("global_dataset_with_noon_times", [np.datetime64("2022-02-01T12:00:00")]),
    ],
)
def test_select_times_invalid_start_no_end_time(
    data_fixture, expected_time_values, request, tmp_path, use_dask
):
    """Test selecting times with only start_time specified."""
    # Get the fixture dynamically based on the parameter
    dataset = request.getfixturevalue(data_fixture)

    # Create a temporary file
    filepath = tmp_path / "test.nc"
    dataset.to_netcdf(filepath)

    with pytest.raises(
        ValueError,
        match="The dataset does not contain any time entries between the specified start_time",
    ):
        dataset = Dataset(
            filename=filepath,
            var_names={"var": "var"},
            start_time=datetime(2022, 5, 1),
            use_dask=use_dask,
        )


def test_multiple_matching_times(
    global_dataset_with_multiple_times_per_day, tmp_path, use_dask
):
    """Test handling when multiple matching times are found when end_time is not
    specified."""
    filepath = tmp_path / "test.nc"
    global_dataset_with_multiple_times_per_day.to_netcdf(filepath)
    dataset = Dataset(
        filename=filepath,
        var_names={"var": "var"},
        start_time=datetime(2022, 1, 31, 22, 0),
        use_dask=use_dask,
    )

    assert dataset.ds["time"].values == np.datetime64(datetime(2022, 2, 1, 0, 0))


def test_warnings_times(global_dataset, tmp_path, caplog, use_dask):
    """Test handling when no matching times are found."""
    # Create a temporary file
    filepath = tmp_path / "test.nc"
    global_dataset.to_netcdf(filepath)
    with caplog.at_level(logging.WARNING):
        start_time = datetime(2021, 1, 1)
        end_time = datetime(2021, 2, 1)

        Dataset(
            filename=filepath,
            var_names={"var": "var"},
            start_time=start_time,
            end_time=end_time,
            use_dask=use_dask,
        )
    # Verify the warning message in the log
    assert "No records found at or before the start_time." in caplog.text

    with caplog.at_level(logging.WARNING):
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 2, 1)

        Dataset(
            filename=filepath,
            var_names={"var": "var"},
            start_time=start_time,
            end_time=end_time,
            use_dask=use_dask,
        )
    # Verify the warning message in the log
    assert "No records found at or after the end_time." in caplog.text


def test_from_ds(global_dataset, global_dataset_with_noon_times, use_dask, tmp_path):
    """Test the from_ds method of the Dataset class."""

    start_time = datetime(2022, 1, 1)

    filepath = tmp_path / "test.nc"
    global_dataset.to_netcdf(filepath)

    dataset = Dataset(
        filename=filepath,
        var_names={"var": "var"},
        dim_names={
            "latitude": "latitude",
            "longitude": "longitude",
            "time": "time",
            "depth": "depth",
        },
        start_time=start_time,
        use_dask=use_dask,
    )

    new_dataset = Dataset.from_ds(dataset, global_dataset_with_noon_times)

    assert isinstance(new_dataset, Dataset)  # Check that the new instance is a Dataset
    assert new_dataset.ds.equals(
        global_dataset_with_noon_times
    )  # Verify the new ds attribute is set correctly

    # Ensure other attributes are copied correctly
    for attr in vars(dataset):
        if attr != "ds":
            assert getattr(new_dataset, attr) == getattr(
                dataset, attr
            ), f"Attribute {attr} not copied correctly."


def test_reverse_latitude_reverse_depth_choose_subdomain(
    global_dataset, tmp_path, use_dask
):
    """Test reversing latitude when it is not ascending, the choose_subdomain method,
    and the convert_to_negative_depth method of the Dataset class."""
    start_time = datetime(2022, 1, 1)

    filepath = tmp_path / "test.nc"
    global_dataset["latitude"] = global_dataset["latitude"][::-1]
    global_dataset["depth"] = global_dataset["depth"][::-1]
    global_dataset.to_netcdf(filepath)

    dataset = Dataset(
        filename=filepath,
        var_names={"var": "var"},
        dim_names={
            "latitude": "latitude",
            "longitude": "longitude",
            "time": "time",
            "depth": "depth",
        },
        start_time=start_time,
        use_dask=use_dask,
    )

    assert np.all(np.diff(dataset.ds["latitude"]) > 0)
    assert np.all(np.diff(dataset.ds["depth"]) > 0)

    # test choosing subdomain for domain that straddles the dateline
    target_coords = {
        "lat": xr.DataArray(np.linspace(-10, 10, 100)),
        "lon": xr.DataArray(np.linspace(-10, 10, 100)),
        "straddle": True,
    }
    sub_dataset = dataset.choose_subdomain(
        target_coords,
        buffer_points=1,
        return_copy=True,
    )

    assert -11 <= sub_dataset.ds["latitude"].min() <= 11
    assert -11 <= sub_dataset.ds["latitude"].max() <= 11
    assert -11 <= sub_dataset.ds["longitude"].min() <= 11
    assert -11 <= sub_dataset.ds["longitude"].max() <= 11

    target_coords = {
        "lat": xr.DataArray(np.linspace(-10, 10, 100)),
        "lon": xr.DataArray(np.linspace(10, 20, 100)),
        "straddle": False,
    }
    sub_dataset = dataset.choose_subdomain(
        target_coords,
        buffer_points=1,
        return_copy=True,
    )

    assert -11 <= sub_dataset.ds["latitude"].min() <= 11
    assert -11 <= sub_dataset.ds["latitude"].max() <= 11
    assert 9 <= sub_dataset.ds["longitude"].min() <= 21
    assert 9 <= sub_dataset.ds["longitude"].max() <= 21


def test_check_if_global_with_global_dataset(global_dataset, tmp_path, use_dask):

    filepath = tmp_path / "test.nc"
    global_dataset.to_netcdf(filepath)
    dataset = Dataset(filename=filepath, var_names={"var": "var"}, use_dask=use_dask)
    is_global = dataset.check_if_global(dataset.ds)
    assert is_global


def test_check_if_global_with_non_global_dataset(
    non_global_dataset, tmp_path, use_dask
):

    filepath = tmp_path / "test.nc"
    non_global_dataset.to_netcdf(filepath)
    dataset = Dataset(filename=filepath, var_names={"var": "var"}, use_dask=use_dask)
    is_global = dataset.check_if_global(dataset.ds)

    assert not is_global


def test_check_dataset(global_dataset, tmp_path, use_dask):

    ds = global_dataset.copy()
    ds = ds.drop_vars("var")

    filepath = tmp_path / "test.nc"
    ds.to_netcdf(filepath)

    start_time = datetime(2022, 2, 1)
    end_time = datetime(2022, 3, 1)
    with pytest.raises(
        ValueError, match="Dataset does not contain all required variables."
    ):

        Dataset(
            filename=filepath,
            var_names={"var": "var"},
            start_time=start_time,
            end_time=end_time,
            use_dask=use_dask,
        )

    ds = global_dataset.copy()
    ds = ds.rename({"latitude": "lat", "longitude": "long"})

    filepath = tmp_path / "test2.nc"
    ds.to_netcdf(filepath)

    start_time = datetime(2022, 2, 1)
    end_time = datetime(2022, 3, 1)
    with pytest.raises(
        ValueError, match="Dataset does not contain all required dimensions."
    ):

        Dataset(
            filename=filepath,
            var_names={"var": "var"},
            start_time=start_time,
            end_time=end_time,
            use_dask=use_dask,
        )


def test_era5_correction_choose_subdomain(use_dask):

    data = ERA5Correction(use_dask=use_dask)
    lats = data.ds.latitude[10:20]
    lons = data.ds.longitude[10:20]
    target_coords = {"lat": lats, "lon": lons}
    data.choose_subdomain(target_coords, straddle=False)
    assert (data.ds["latitude"] == lats).all()
    assert (data.ds["longitude"] == lons).all()


def test_data_concatenation(use_dask):

    fname = download_test_data("GLORYS_NA_2012.nc")
    data = GLORYSDataset(
        filename=fname,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2013, 1, 1),
        use_dask=use_dask,
    )

    # Concatenating the datasets at fname0 and fname1 should result in the dataset at fname
    fname0 = download_test_data("GLORYS_NA_20120101.nc")
    fname1 = download_test_data("GLORYS_NA_20121231.nc")

    # Test concatenation based on wildcards
    directory_path = Path(fname0).parent
    data_concatenated = GLORYSDataset(
        filename=str(directory_path) + "/GLORYS_NA_2012????.nc",
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2013, 1, 1),
        use_dask=use_dask,
    )
    assert data.ds.equals(data_concatenated.ds)

    # Test concatenation based on lists
    data_concatenated = GLORYSDataset(
        filename=[fname0, fname1],
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2013, 1, 1),
        use_dask=use_dask,
    )
    assert data.ds.equals(data_concatenated.ds)


def test_time_validation(use_dask):

    fname = download_test_data("GLORYS_NA_2012.nc")

    with pytest.raises(TypeError, match="start_time must be a datetime object"):
        GLORYSDataset(
            filename=fname,
            start_time="dummy",
            end_time=datetime(2013, 1, 1),
            use_dask=use_dask,
        )
    with pytest.raises(TypeError, match="end_time must be a datetime object"):

        GLORYSDataset(
            filename=fname,
            start_time=datetime(2012, 1, 1),
            end_time="dummy",
            use_dask=use_dask,
        )


def test_climatology_error(use_dask):

    fname = download_test_data("GLORYS_NA_2012.nc")

    with pytest.raises(
        ValueError,
        match="The dataset contains 2 time steps, but the climatology flag is set to True, which requires exactly 12 time steps.",
    ):
        GLORYSDataset(
            filename=fname,
            start_time=datetime(2012, 1, 1),
            end_time=datetime(2013, 1, 1),
            climatology=True,
            use_dask=use_dask,
        )

    fname_bgc = download_test_data("CESM_regional_coarse_test_data_climatology.nc")

    with pytest.raises(
        ValueError,
        match="The dataset contains integer time values, which are only supported when the climatology flag is set to True. However, your climatology flag is set to False.",
    ):

        CESMBGCDataset(
            filename=fname_bgc,
            start_time=datetime(2012, 1, 1),
            end_time=datetime(2013, 1, 1),
            climatology=False,
            use_dask=use_dask,
        )


@pytest.mark.parametrize(
    "data_fixture, expected_resolution",
    [
        ("era5_data", 0.25),
        ("glorys_data", 1 / 12),
        # ("tpxo_data", 1 / 6),
        ("cesm_bgc_data", 1.0),
        ("cesm_surface_bgc_data", 1.0),
        ("unified_bgc_data", 2.0),
        ("unified_surface_bgc_data", 2.0),
    ],
)
def test_horizontal_resolution(data_fixture, expected_resolution, request):

    data = request.getfixturevalue(data_fixture)
    assert np.isclose(data.resolution, expected_resolution)


class TestTPXODataset:
    @pytest.fixture
    def regional_tpxo_dataset(self, use_dask):
        """TPXO dataset with regional coverage and 25 constituents: M2, S2, N2, K2, K1, O1, P1, Q1, MM, Mf, MSF, M4, Mn4, Ms4, ..."""
        fname_grid = Path(download_test_data("regional_grid_tpxo10v2.nc"))
        fname_h = Path(download_test_data("regional_h_tpxo10v2.nc"))

        return TPXODataset(
            filename=fname_h,
            grid_filename=fname_grid,
            location="h",
            var_names={"ssh_Re": "hRe", "ssh_Im": "hIm"},
            use_dask=use_dask,
        )

    @pytest.fixture
    def global_tpxo_dataset(self, use_dask):
        """TPXO dataset with global coverage and 1 constituent: M2"""
        fname_grid = Path(download_test_data("global_grid_tpxo10.v2.nc"))
        fname_h = Path(download_test_data("global_h_tpxo10.v2.nc"))

        return TPXODataset(
            filename=fname_h,
            grid_filename=fname_grid,
            location="h",
            var_names={"ssh_Re": "hRe", "ssh_Im": "hIm"},
            use_dask=use_dask,
        )

    @pytest.fixture
    def omega(self):
        return {
            "m2": 1.405189e-04,
            "s2": 1.454441e-04,
            "n2": 1.378797e-04,
            "k2": 1.458423e-04,
            "k1": 7.292117e-05,
            "o1": 6.759774e-05,
            "p1": 7.252295e-05,
            "q1": 6.495854e-05,
            "mm": 0.026392e-04,
            "mf": 0.053234e-04,
            "m4": 2.810377e-04,
            "mn4": 2.783984e-04,
            "ms4": 2.859630e-04,
            "2n2": 1.352405e-04,
            "s1": 7.2722e-05,
        }

    @pytest.mark.parametrize(
        "tpxo_dataset_fixture",
        [
            "regional_tpxo_dataset",
            "global_tpxo_dataset",
        ],
    )
    def test_initialization_and_clean_up(self, tpxo_dataset_fixture, request):

        tpxo_dataset = request.getfixturevalue(tpxo_dataset_fixture)
        assert tpxo_dataset.location == "h"

        assert "hRe" in tpxo_dataset.ds.data_vars
        assert "hIm" in tpxo_dataset.ds.data_vars
        assert "mask" in tpxo_dataset.ds.data_vars
        assert "longitude" in tpxo_dataset.ds.dims
        assert "latitude" in tpxo_dataset.ds.dims
        assert "longitude" in tpxo_dataset.ds.variables
        assert "latitude" in tpxo_dataset.ds.variables

    def test_select_fewer_constituents(self, regional_tpxo_dataset, omega):

        regional_tpxo_dataset.select_constituents(2, omega)
        assert (
            regional_tpxo_dataset.ds["ntides"].values.astype("U3") == ["m2", "s2"]
        ).all()

    def test_select_constituents_with_reordering(self, regional_tpxo_dataset, omega):

        regional_tpxo_dataset.select_constituents(11, omega)

        assert len(regional_tpxo_dataset.ds["ntides"]) == 11
        # check that m4 has been moved from 12th to 11th position to follow TPXO9 order
        assert (
            regional_tpxo_dataset.ds["ntides"].isel(ntides=10).item().decode("utf-8")
            == "m4"
        )

    def test_select_constituents_omega_mismatch(self, regional_tpxo_dataset, omega):

        omega = OrderedDict(
            list(omega.items())[:3] + [("fake", 6.495854e-05)] + list(omega.items())[3:]
        )
        with pytest.raises(ValueError, match="The dataset contains tidal constituents"):
            regional_tpxo_dataset.select_constituents(11, omega)

    def test_select_constituents_too_few(self, global_tpxo_dataset, omega):
        with pytest.raises(ValueError, match="The dataset contains tidal constituents"):
            global_tpxo_dataset.select_constituents(11, omega)
