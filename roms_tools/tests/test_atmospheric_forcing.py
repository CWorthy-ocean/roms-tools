import pytest
from datetime import datetime
from roms_tools import Grid, AtmosphericForcing, SWRCorrection
from roms_tools.setup.datasets import download_test_data
import xarray as xr
import tempfile
import os
import pooch
import numpy as np


@pytest.fixture
def grid_that_straddles_dateline():
    """
    Fixture for creating a domain that straddles the dateline and lies within the bounds of the regional ERA5 data.
    """
    grid = Grid(
        nx=5,
        ny=5,
        size_x=1800,
        size_y=2400,
        center_lon=-10,
        center_lat=61,
        rot=20,
    )

    return grid


@pytest.fixture
def grid_that_straddles_dateline_but_is_too_big_for_regional_test_data():
    """
    Fixture for creating a domain that straddles the dateline but exceeds the bounds of the regional ERA5 data.
    Centered east of dateline.
    """
    grid = Grid(
        nx=5,
        ny=5,
        size_x=2000,
        size_y=2400,
        center_lon=10,
        center_lat=61,
        rot=20,
    )

    return grid


@pytest.fixture
def another_grid_that_straddles_dateline_but_is_too_big_for_regional_test_data():
    """
    Fixture for creating a domain that straddles the dateline but exceeds the bounds of the regional ERA5 data.
    Centered west of dateline. This one was hard to catch for the nan_check for a long time, but should work now.
    """
    grid = Grid(
        nx=5,
        ny=5,
        size_x=1950,
        size_y=2400,
        center_lon=-30,
        center_lat=61,
        rot=25,
    )

    return grid


@pytest.fixture
def grid_that_lies_east_of_dateline_less_than_five_degrees_away():
    """
    Fixture for creating a domain that lies east of Greenwich meridian, but less than 5 degrees away.
    We care about the 5 degree mark because it decides whether the code handles the longitudes as straddling the dateline or not.
    """

    grid = Grid(
        nx=5,
        ny=5,
        size_x=500,
        size_y=2000,
        center_lon=10,
        center_lat=61,
        rot=0,
    )

    return grid


@pytest.fixture
def grid_that_lies_east_of_dateline_more_than_five_degrees_away():
    """
    Fixture for creating a domain that lies east of Greenwich meridian, more than 5 degrees away.
    We care about the 5 degree mark because it decides whether the code handles the longitudes as straddling the dateline or not.
    """
    grid = Grid(
        nx=5,
        ny=5,
        size_x=500,
        size_y=2400,
        center_lon=15,
        center_lat=61,
        rot=0,
    )

    return grid


@pytest.fixture
def grid_that_lies_west_of_dateline_less_than_five_degrees_away():
    """
    Fixture for creating a domain that lies west of Greenwich meridian, less than 5 degrees away.
    We care about the 5 degree mark because it decides whether the code handles the longitudes as straddling the dateline or not.
    """

    grid = Grid(
        nx=5,
        ny=5,
        size_x=700,
        size_y=2400,
        center_lon=-15,
        center_lat=61,
        rot=0,
    )

    return grid


@pytest.fixture
def grid_that_lies_west_of_dateline_more_than_five_degrees_away():
    """
    Fixture for creating a domain that lies west of Greenwich meridian, more than 5 degrees away.
    We care about the 5 degree mark because it decides whether the code handles the longitudes as straddling the dateline or not.
    """

    grid = Grid(
        nx=5,
        ny=5,
        size_x=1000,
        size_y=2400,
        center_lon=-25,
        center_lat=61,
        rot=0,
    )

    return grid

@pytest.fixture
def grid_that_straddles_180_degree_meridian():
    """
    Fixture for creating a domain that straddles 180 degree meridian. This is a good test grid for the global ERA5 data, which comes on an [-180, 180] longitude grid.
    """

    grid = Grid(
        nx=5,
        ny=5,
        size_x=1800,
        size_y=2400,
        center_lon=180,
        center_lat=61,
        rot=20,
    )

    return grid


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid_that_straddles_dateline",
        "grid_that_lies_east_of_dateline_less_than_five_degrees_away",
        "grid_that_lies_east_of_dateline_more_than_five_degrees_away",
        "grid_that_lies_west_of_dateline_less_than_five_degrees_away",
        "grid_that_lies_west_of_dateline_more_than_five_degrees_away",
    ],
)
def test_successful_initialization_with_regional_data(grid_fixture, request):
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    atm_forcing = AtmosphericForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        source="era5",
        filename=fname,
    )

    assert atm_forcing.ds is not None

    grid.coarsen()

    atm_forcing = AtmosphericForcing(
        grid=grid,
        use_coarse_grid=True,
        start_time=start_time,
        end_time=end_time,
        source="era5",
        filename=fname,
    )

    assert isinstance(atm_forcing.ds, xr.Dataset)
    assert "uwnd" in atm_forcing.ds
    assert "vwnd" in atm_forcing.ds
    assert "swrad" in atm_forcing.ds
    assert "lwrad" in atm_forcing.ds
    assert "Tair" in atm_forcing.ds
    assert "qair" in atm_forcing.ds
    assert "rain" in atm_forcing.ds

    assert atm_forcing.start_time == start_time
    assert atm_forcing.end_time == end_time
    assert atm_forcing.filename == fname
    assert atm_forcing.source == "era5"


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
        "another_grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
    ],
)
def test_nan_detection_initialization_with_regional_data(grid_fixture, request):
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    with pytest.raises(ValueError, match="NaN values found"):

        AtmosphericForcing(
            grid=grid,
            start_time=start_time,
            end_time=end_time,
            source="era5",
            filename=fname,
        )

    grid.coarsen()

    with pytest.raises(ValueError, match="NaN values found"):
        AtmosphericForcing(
            grid=grid,
            use_coarse_grid=True,
            start_time=start_time,
            end_time=end_time,
            source="era5",
            filename=fname,
        )

def test_no_longitude_intersection_initialization_with_regional_data(grid_that_straddles_180_degree_meridian):
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")

    with pytest.raises(ValueError, match="Selected longitude range does not intersect with dataset"):

        AtmosphericForcing(
            grid=grid_that_straddles_180_degree_meridian,
            start_time=start_time,
            end_time=end_time,
            source="era5",
            filename=fname,
        )

    grid_that_straddles_180_degree_meridian.coarsen()

    with pytest.raises(ValueError, match="Selected longitude range does not intersect with dataset"):
        AtmosphericForcing(
            grid=grid_that_straddles_180_degree_meridian,
            use_coarse_grid=True,
            start_time=start_time,
            end_time=end_time,
            source="era5",
            filename=fname,
        )

@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid_that_straddles_dateline",
        "grid_that_lies_east_of_dateline_less_than_five_degrees_away",
        "grid_that_lies_east_of_dateline_more_than_five_degrees_away",
        "grid_that_lies_west_of_dateline_less_than_five_degrees_away",
        "grid_that_lies_west_of_dateline_more_than_five_degrees_away",
        "grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
        "another_grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
        "grid_that_straddles_180_degree_meridian"
    ],
)
def test_successful_initialization_with_global_data(grid_fixture, request):
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    atm_forcing = AtmosphericForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        source="era5",
        filename=fname,
    )

    assert isinstance(atm_forcing.ds, xr.Dataset)
    assert "uwnd" in atm_forcing.ds
    assert "vwnd" in atm_forcing.ds
    assert "swrad" in atm_forcing.ds
    assert "lwrad" in atm_forcing.ds
    assert "Tair" in atm_forcing.ds
    assert "qair" in atm_forcing.ds
    assert "rain" in atm_forcing.ds

    grid.coarsen()

    atm_forcing = AtmosphericForcing(
        grid=grid,
        use_coarse_grid=True,
        start_time=start_time,
        end_time=end_time,
        source="era5",
        filename=fname,
    )

    assert isinstance(atm_forcing.ds, xr.Dataset)
    assert "uwnd" in atm_forcing.ds
    assert "vwnd" in atm_forcing.ds
    assert "swrad" in atm_forcing.ds
    assert "lwrad" in atm_forcing.ds
    assert "Tair" in atm_forcing.ds
    assert "qair" in atm_forcing.ds
    assert "rain" in atm_forcing.ds

    assert atm_forcing.start_time == start_time
    assert atm_forcing.end_time == end_time
    assert atm_forcing.filename == fname
    assert atm_forcing.source == "era5"


@pytest.fixture
def atmospheric_forcing(grid_that_straddles_dateline):
    """
    Fixture for creating a AtmosphericForcing object.
    """

    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")

    return AtmosphericForcing(
        grid=grid_that_straddles_dateline,
        start_time=start_time,
        end_time=end_time,
        source="era5",
        filename=fname,
    )


def test_plot_method(atmospheric_forcing):
    """
    Test the plot method of the AtmosphericForcing object.
    """
    atmospheric_forcing.plot(varname="uwnd", time=0)


def test_save_method(atmospheric_forcing, tmp_path):
    """
    Test the save method of the AtmosphericForcing object.
    """

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    atmospheric_forcing.save(filepath)
    extended_filepath = filepath + ".20200201-01.nc"

    try:
        assert os.path.exists(extended_filepath)
    finally:
        os.remove(extended_filepath)


# SWRCorrection checks


@pytest.fixture
def swr_correction():

    correction_filename = pooch.retrieve(
        url="https://github.com/CWorthy-ocean/roms-tools-data/raw/main/SSR_correction.nc",
        known_hash="a170c1698e6cc2765b3f0bb51a18c6a979bc796ac3a4c014585aeede1f1f8ea0",
    )
    correction_filename

    return SWRCorrection(
        filename=correction_filename,
        varname="ssr_corr",
        dim_names={"time": "time", "latitude": "latitude", "longitude": "longitude"},
        temporal_resolution="climatology",
    )


def test_check_dataset(swr_correction):

    ds = swr_correction.ds.copy()
    ds = ds.drop_vars("ssr_corr")
    with pytest.raises(ValueError):
        swr_correction._check_dataset(ds)

    ds = swr_correction.ds.copy()
    ds = ds.rename({"latitude": "lat", "longitude": "long"})
    with pytest.raises(ValueError):
        swr_correction._check_dataset(ds)


def test_ensure_latitude_ascending(swr_correction):

    ds = swr_correction.ds.copy()

    ds["latitude"] = ds["latitude"][::-1]
    ds = swr_correction._ensure_latitude_ascending(ds)
    assert np.all(np.diff(ds["latitude"]) > 0)


def test_handle_longitudes(swr_correction):
    swr_correction.ds["longitude"] = (
        (swr_correction.ds["longitude"] + 180) % 360
    ) - 180  # Convert to [-180, 180]
    swr_correction._handle_longitudes(straddle=False)
    assert np.all(
        (swr_correction.ds["longitude"] >= 0) & (swr_correction.ds["longitude"] <= 360)
    )


def test_choose_subdomain(swr_correction):
    lats = swr_correction.ds.latitude[10:20]
    lons = swr_correction.ds.longitude[10:20]
    coords = {"latitude": lats, "longitude": lons}
    subdomain = swr_correction._choose_subdomain(coords)
    assert (subdomain["latitude"] == lats).all()
    assert (subdomain["longitude"] == lons).all()


def test_interpolate_temporally(swr_correction):
    field = swr_correction.ds["ssr_corr"]

    fname = download_test_data("ERA5_regional_test_data.nc")
    era5_times = xr.open_dataset(fname).time
    interpolated_field = swr_correction._interpolate_temporally(field, era5_times)
    assert len(interpolated_field.time) == len(era5_times)
