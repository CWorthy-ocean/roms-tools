import pytest
from datetime import datetime
import numpy as np
import xarray as xr
from roms_tools.setup.datasets import Dataset
import tempfile
import os


@pytest.fixture
def mock_dataset():
    """
    Fixture to provide a mock Dataset object with relevant mock data.
    """
    # Mock dataset with time, latitude, and longitude dimensions
    mock_data = xr.Dataset(
        {
            "time": (
                "time",
                [
                    np.datetime64("2022-01-01T00:00:00"),
                    np.datetime64("2022-02-01T00:00:00"),
                    np.datetime64("2022-03-01T00:00:00"),
                    np.datetime64("2022-04-01T00:00:00"),
                ],
            ),
            "latitude": ("latitude", np.linspace(-90, 90, 180)),
            "longitude": ("longitude", np.linspace(-180, 180, 360)),
            "depth": ("depth", np.linspace(0.5, 4000, 10)),
        }
    )

    return mock_data


@pytest.fixture
def mock_dataset_with_noon_times():
    """
    Fixture to provide a mock Dataset object with different time values.
    """
    mock_data = xr.Dataset(
        {
            "time": (
                "time",
                [
                    np.datetime64("2022-01-01T12:00:00"),
                    np.datetime64("2022-02-01T12:00:00"),
                    np.datetime64("2022-03-01T12:00:00"),
                    np.datetime64("2022-04-01T12:00:00"),
                ],
            ),
            "latitude": ("latitude", np.linspace(-90, 90, 180)),
            "longitude": ("longitude", np.linspace(-180, 180, 360)),
        }
    )

    return mock_data


@pytest.fixture
def mock_dataset_with_multiple_times_per_day():
    """
    Fixture to provide a mock Dataset object with different time values.
    """
    mock_data = xr.Dataset(
        {
            "time": (
                "time",
                [
                    np.datetime64("2022-01-01T00:00:00"),
                    np.datetime64("2022-01-01T12:00:00"),
                    np.datetime64("2022-02-01T00:00:00"),
                    np.datetime64("2022-02-01T12:00:00"),
                    np.datetime64("2022-03-01T00:00:00"),
                    np.datetime64("2022-03-01T12:00:00"),
                    np.datetime64("2022-04-01T00:00:00"),
                    np.datetime64("2022-04-01T12:00:00"),
                ],
            ),
            "latitude": ("latitude", np.linspace(-90, 90, 180)),
            "longitude": ("longitude", np.linspace(-180, 180, 360)),
        }
    )

    return mock_data


@pytest.mark.parametrize(
    "mock_data_fixture, expected_time_values",
    [
        ("mock_dataset", [np.datetime64("2022-02-01T00:00:00")]),
        ("mock_dataset_with_noon_times", [np.datetime64("2022-02-01T12:00:00")]),
        (
            "mock_dataset_with_multiple_times_per_day",
            [
                np.datetime64("2022-02-01T00:00:00"),
                np.datetime64("2022-02-01T12:00:00"),
            ],
        ),
    ],
)
def test_select_times(mock_data_fixture, expected_time_values, request):
    """
    Test selecting times with different mock datasets.
    """
    start_time = datetime(2022, 2, 1)
    end_time = datetime(2022, 3, 1)

    # Get the fixture dynamically based on the parameter
    mock_dataset = request.getfixturevalue(mock_data_fixture)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
        mock_dataset.to_netcdf(filepath)
    try:
        # Instantiate Dataset object using the temporary file
        dataset = Dataset(filename=filepath, start_time=start_time, end_time=end_time)

        assert dataset.ds is not None
        assert len(dataset.ds.time) == len(expected_time_values)
        for expected_time in expected_time_values:
            assert expected_time in dataset.ds.time.values
    finally:
        os.remove(filepath)


@pytest.mark.parametrize(
    "mock_data_fixture, expected_time_values",
    [
        ("mock_dataset", [np.datetime64("2022-02-01T00:00:00")]),
        ("mock_dataset_with_noon_times", [np.datetime64("2022-02-01T12:00:00")]),
    ],
)
def test_select_times_no_end_time(mock_data_fixture, expected_time_values, request):
    """
    Test selecting times with only start_time specified.
    """
    start_time = datetime(2022, 2, 1)

    # Get the fixture dynamically based on the parameter
    mock_dataset = request.getfixturevalue(mock_data_fixture)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
        mock_dataset.to_netcdf(filepath)
    try:
        # Instantiate Dataset object using the temporary file
        dataset = Dataset(filename=filepath, start_time=start_time)

        assert dataset.ds is not None
        assert len(dataset.ds.time) == len(expected_time_values)
        for expected_time in expected_time_values:
            assert expected_time in dataset.ds.time.values
    finally:
        os.remove(filepath)


def test_multiple_matching_times(mock_dataset_with_multiple_times_per_day):
    """
    Test handling when multiple matching times are found when end_time is not specified.
    """
    start_time = datetime(2022, 1, 1)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
        mock_dataset_with_multiple_times_per_day.to_netcdf(filepath)
    try:
        # Instantiate Dataset object using the temporary file
        with pytest.raises(
            ValueError,
            match="There must be exactly one time matching the start_time. Found 2 matching times.",
        ):
            Dataset(filename=filepath, start_time=start_time)
    finally:
        os.remove(filepath)


def test_no_matching_times(mock_dataset):
    """
    Test handling when no matching times are found.
    """
    start_time = datetime(2021, 1, 1)
    end_time = datetime(2021, 2, 1)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
        mock_dataset.to_netcdf(filepath)
    try:
        # Instantiate Dataset object using the temporary file
        with pytest.raises(ValueError, match="No matching times found."):
            Dataset(filename=filepath, start_time=start_time, end_time=end_time)
    finally:
        os.remove(filepath)


def test_reverse_latitude_choose_subdomain_negative_depth(mock_dataset):
    """
    Test reversing latitude when it is not ascending, the choose_subdomain method, and the convert_to_negative_depth method of the Dataset class.
    """
    start_time = datetime(2022, 1, 1)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
        mock_dataset["latitude"] = mock_dataset["latitude"][::-1]
        mock_dataset.to_netcdf(filepath)
    try:
        # Instantiate Dataset object using the temporary file
        dataset = Dataset(filename=filepath, start_time=start_time)

        assert np.all(np.diff(dataset.ds["latitude"]) > 0)

        # test choosing subdomain for domain that straddles the dateline
        dataset.choose_subdomain(
            latitude_range=(-10, 10), longitude_range=(-10, 10), margin=1, straddle=True
        )

        assert -11 <= dataset.ds["latitude"].min() <= 11
        assert -11 <= dataset.ds["latitude"].max() <= 11
        assert -11 <= dataset.ds["longitude"].min() <= 11
        assert -11 <= dataset.ds["longitude"].max() <= 11

        # test choosing subdomain for domain that does not straddle the dateline
        dataset = Dataset(filename=filepath, start_time=start_time)
        dataset.choose_subdomain(
            latitude_range=(-10, 10), longitude_range=(10, 20), margin=1, straddle=False
        )

        assert -11 <= dataset.ds["latitude"].min() <= 11
        assert -11 <= dataset.ds["latitude"].max() <= 11
        assert 9 <= dataset.ds["longitude"].min() <= 21
        assert 9 <= dataset.ds["longitude"].max() <= 21

        dataset.convert_to_negative_depth()

        assert (dataset.ds["depth"] < 0).all()

    finally:
        os.remove(filepath)
