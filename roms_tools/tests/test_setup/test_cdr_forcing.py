import pytest
from copy import deepcopy
from datetime import datetime
from roms_tools import CDRPointSource, Grid
import xarray as xr
import numpy as np
import logging
from roms_tools.setup.utils import get_river_tracer_defaults

# Fixtures
@pytest.fixture
def iceland_test_grid():
    return Grid(
        nx=18, ny=18, size_x=800, size_y=800, center_lon=-18, center_lat=65, rot=0, N=3
    )


@pytest.fixture
def test_grid_that_straddles():
    return Grid(
        nx=18, ny=18, size_x=800, size_y=800, center_lon=0, center_lat=65, rot=0, N=3
    )


@pytest.fixture
def start_end_times():
    start_time = datetime(2022, 1, 1)
    end_time = datetime(2022, 12, 31)
    return start_time, end_time


@pytest.fixture
def empty_cdr_point_source_without_grid(start_end_times):
    start_time, end_time = start_end_times
    return CDRPointSource(
        start_time=start_time,
        end_time=end_time,
    )


@pytest.fixture
def empty_cdr_point_source_with_grid(iceland_test_grid, start_end_times):
    start_time, end_time = start_end_times
    return CDRPointSource(
        grid=iceland_test_grid,
        start_time=start_time,
        end_time=end_time,
    )


@pytest.fixture
def empty_cdr_point_source_with_grid_that_straddles(
    test_grid_that_straddles, start_end_times
):
    start_time, end_time = start_end_times
    return CDRPointSource(
        grid=test_grid_that_straddles,
        start_time=start_time,
        end_time=end_time,
    )


@pytest.fixture
def valid_release_params():
    return {
        "lat": 66.0,
        "lon": -25.0,
        "depth": 50.0,
        "volumes": 100.0,
    }


# Tests
@pytest.mark.parametrize(
    "cdr_forcing_fixture",
    [
        "empty_cdr_point_source_without_grid",
        "empty_cdr_point_source_with_grid",
    ],
)
def test_cdr_point_source_init(cdr_forcing_fixture, start_end_times, request):
    cdr = request.getfixturevalue(cdr_forcing_fixture)
    start_time, end_time = start_end_times
    assert cdr.start_time == start_time
    assert cdr.end_time == end_time

    # Check that dataset is empty but has right dimensions and variables
    assert isinstance(cdr.ds, xr.Dataset)
    assert "time" in cdr.ds
    assert "release_name" in cdr.ds
    assert "tracer_name" in cdr.ds
    assert "tracer_unit" in cdr.ds
    assert "tracer_long_name" in cdr.ds
    assert "cdr_time" in cdr.ds
    assert "cdr_volume" in cdr.ds
    assert "cdr_tracer" in cdr.ds
    assert len(cdr.ds.time) == 0
    assert len(cdr.ds.ncdr) == 0
    assert len(cdr.ds.ntracers) == 34
    assert cdr.ds.cdr_time.size == 0
    assert cdr.ds.cdr_volume.size == 0
    assert cdr.ds.cdr_tracer.size == 0

    # Check that release dictionary is empty
    assert not cdr.releases


def test_cdr_point_source_init_invalid_times():
    start_time = datetime(2022, 5, 1)
    end_time = datetime(2022, 5, 1)
    with pytest.raises(
        ValueError, match="`start_time` must be earlier than `end_time`"
    ):
        CDRPointSource(start_time=start_time, end_time=end_time)


@pytest.mark.parametrize(
    "cdr_forcing_fixture",
    [
        "empty_cdr_point_source_without_grid",
        "empty_cdr_point_source_with_grid",
    ],
)
def test_add_release(cdr_forcing_fixture, valid_release_params, request):
    
    cdr = request.getfixturevalue(cdr_forcing_fixture)
    release_params = deepcopy(valid_release_params)
    times = [
        datetime(2022, 1, 1),
        datetime(2022, 1, 5),
        datetime(2022, 1, 10),
    ]
    release_params["times"] = times

    volumes = [100.0, 200.0, 150.0]
    release_params["volumes"] = volumes

    tracer_concentrations = {
        "ALK": [2300.0, 2350.0, 2400.0],
        "DIC": [2100.0, 2150.0, 2200.0],
        "temp": 10.0,
        "salt": 35.0,
    }
    release_params["tracer_concentrations"] = tracer_concentrations

    # Add release
    cdr.add_release(name="release", **release_params)
    
    # Check dataset structure
    assert "release" in cdr.ds["release_name"].values
    assert len(cdr.ds.time) == len(release_params["times"])
    assert len(cdr.ds.ncdr) == 1
    assert len(cdr.ds.ntracers) == 34
    assert cdr.ds.cdr_time.size > 0
    
    # Check cdr_volume shape and values
    assert cdr.ds.cdr_volume.shape == (len(times), 1)
    np.testing.assert_allclose(cdr.ds.cdr_volume[:, 0], volumes, rtol=1e-3)
    
    # Check tracer concentration shape
    assert cdr.ds.cdr_tracer.shape == (len(times), 34, 1)

    # Check tracer concentration values for known tracers
    tracer_index = {name: i for i, name in enumerate(cdr.ds.tracer_name.values)}
    for tracer, expected in tracer_concentrations.items():
        i = tracer_index[tracer]
        if isinstance(expected, list):
            np.testing.assert_allclose(cdr.ds.cdr_tracer[:, i, 0], expected, rtol=1e-3)
        else:
            np.testing.assert_allclose(cdr.ds.cdr_tracer[:, i, 0], expected, rtol=1e-3)
    
    assert "release" in cdr.releases.keys()

@pytest.mark.parametrize(
    "cdr_forcing_fixture",
    [
        "empty_cdr_point_source_without_grid",
        "empty_cdr_point_source_with_grid",
    ],
)
def test_add_duplicate_release(cdr_forcing_fixture, valid_release_params, request):
    cdr = request.getfixturevalue(cdr_forcing_fixture)
    release_params = deepcopy(valid_release_params)
    cdr.add_release(name="release_1", **release_params)
    with pytest.raises(
        ValueError, match="A release with the name 'release_1' already exists."
    ):
        cdr.add_release(name="release_1", **release_params)


@pytest.mark.parametrize(
    "cdr_forcing_fixture",
    [
        "empty_cdr_point_source_without_grid",
        "empty_cdr_point_source_with_grid",
    ],
)
def test_invalid_release_params(cdr_forcing_fixture, valid_release_params, request):
    cdr = request.getfixturevalue(cdr_forcing_fixture)

    # Test invalid latitude
    invalid_params = deepcopy(valid_release_params)
    invalid_params["lat"] = 100.0
    with pytest.raises(ValueError, match="Latitude must be between -90 and 90."):
        cdr.add_release(name="release_1", **invalid_params)

    # Test invalid depth (negative value)
    invalid_params = deepcopy(valid_release_params)
    invalid_params["depth"] = -10.0
    with pytest.raises(ValueError, match="Depth must be a non-negative number."):
        cdr.add_release(name="release_1", **invalid_params)

    # Test times not being datetime objects
    invalid_params = deepcopy(valid_release_params)
    invalid_params["times"] = [
        "2023-01-01",
        "2023-02-01",
    ]  # Invalid times format (strings)
    with pytest.raises(
        ValueError,
        match="If 'times' is provided, all entries must be datetime objects.",
    ):
        cdr.add_release(name="release_1", **invalid_params)

    # Test times being too short (less than 2 datetime objects)
    invalid_params = deepcopy(valid_release_params)
    invalid_params["times"] = [datetime(2022, 1, 1)]  # Only one datetime object
    with pytest.raises(
        ValueError,
        match="If 'times' is provided, it must contain at least two datetime objects.",
    ):
        cdr.add_release(name="release_1", **invalid_params)

    # Test times being not monotonically increasing
    invalid_params = deepcopy(valid_release_params)
    invalid_params["times"] = [
        datetime(2022, 1, 1),
        datetime(2022, 2, 1),
        datetime(2022, 1, 15),  # Out of order date
        datetime(2022, 3, 1),
    ]
    with pytest.raises(
        ValueError, match="The 'times' list must be monotonically increasing."
    ):
        cdr.add_release(name="release_1", **invalid_params)

    # Test first time earlier than self.start_time
    invalid_params = deepcopy(valid_release_params)
    invalid_params["times"] = [
        datetime(2000, 1, 1),
        datetime(2022, 2, 1),
    ]  # Earlier than self.start_time
    with pytest.raises(ValueError, match="First entry"):
        cdr.add_release(name="release_1", **invalid_params)

    # Test last time later than self.end_time
    invalid_params = deepcopy(valid_release_params)
    invalid_params["times"] = [
        datetime(2022, 1, 1),
        datetime(2025, 1, 1),
    ]  # Later than self.end_time
    with pytest.raises(ValueError, match="Last entry"):
        cdr.add_release(name="release_1", **invalid_params)

    # Test mismatch between times and volume length
    invalid_params = deepcopy(valid_release_params)
    invalid_params["times"] = [datetime(2022, 1, 1), datetime(2022, 1, 2)]  # Two times
    invalid_params["volumes"] = [100]  # Only one volume entry
    with pytest.raises(ValueError, match="The length of `volumes` "):
        cdr.add_release(name="release_1", **invalid_params)

    # Test mismatch between times and tracer_concentrations length
    invalid_params = deepcopy(valid_release_params)
    invalid_params["times"] = [datetime(2022, 1, 1), datetime(2022, 1, 2)]  # Two times
    invalid_params["tracer_concentrations"] = {
        "ALK": [1]
    }  # Only one tracer concentration
    with pytest.raises(ValueError, match="The length of tracer 'ALK'"):
        cdr.add_release(name="release_1", **invalid_params)

    # Test invalid volume (negative)
    invalid_params = deepcopy(valid_release_params)
    invalid_params["volumes"] = -100  # Invalid volume
    with pytest.raises(ValueError, match="Volume must be non-negative"):
        cdr.add_release(name="release_1", **invalid_params)

    # Test volume as list with negative values
    invalid_params = deepcopy(valid_release_params)
    invalid_params["volumes"] = [10, -5]  # Invalid volume in list
    with pytest.raises(
        ValueError, match="All entries in `volumes` must be non-negative"
    ):
        cdr.add_release(name="release_1", **invalid_params)

    # Test invalid tracer concentration (negative)
    invalid_params = deepcopy(valid_release_params)
    invalid_params["tracer_concentrations"] = {"ALK": -1}
    with pytest.raises(ValueError, match="The concentration of tracer"):
        cdr.add_release(name="release_1", **invalid_params)

    # Test tracer_concentration as list with negative values
    invalid_params = deepcopy(valid_release_params)
    invalid_params["tracer_concentrations"] = {
        "ALK": [10, -5]
    }  # Invalid concentration in list
    with pytest.raises(ValueError, match="All entries in "):
        cdr.add_release(name="release_1", **invalid_params)


def test_warning_no_grid(
    empty_cdr_point_source_without_grid, valid_release_params, caplog
):

    with caplog.at_level(logging.WARNING):
        empty_cdr_point_source_without_grid.add_release(
            name="release_1", **valid_release_params
        )

    assert "Grid not provided"


@pytest.mark.parametrize(
    "cdr_forcing_fixture",
    [
        "empty_cdr_point_source_with_grid",
        "empty_cdr_point_source_with_grid_that_straddles",
    ],
)
def test_invalid_release_longitude(cdr_forcing_fixture, valid_release_params, request):
    cdr = request.getfixturevalue(cdr_forcing_fixture)

    # Release location outside of domain
    invalid_params = deepcopy(valid_release_params)
    invalid_params["lon"] = -30
    invalid_params["lat"] = 60
    with pytest.raises(ValueError, match="outside of the grid domain"):
        cdr.add_release(name="release_1", **invalid_params)

    # Release location outside of domain
    invalid_params = deepcopy(valid_release_params)
    invalid_params["lon"] = 360 - 30
    invalid_params["lat"] = 60
    with pytest.raises(ValueError, match="outside of the grid domain"):
        cdr.add_release(name="release_1", **invalid_params)


def test_invalid_release_location(
    empty_cdr_point_source_with_grid, valid_release_params
):

    # Release location too close to boundary of Iceland domain; lat_rho[0, 0] = 60.97, lon_rho[0, 0] = 334.17
    invalid_params = deepcopy(valid_release_params)
    invalid_params["lon"] = 334.17
    invalid_params["lat"] = 60.97
    with pytest.raises(ValueError, match="too close to the grid boundary"):
        empty_cdr_point_source_with_grid.add_release(name="release_1", **invalid_params)

    # Release location lies on land
    invalid_params = deepcopy(valid_release_params)
    invalid_params["lon"] = -20.0
    invalid_params["lat"] = 64.5
    with pytest.raises(ValueError, match="on land"):
        empty_cdr_point_source_with_grid.add_release(name="release_1", **invalid_params)

    # Release location lies below seafloor
    invalid_params = deepcopy(valid_release_params)
    invalid_params["depth"] = 4000
    with pytest.raises(ValueError, match="below the seafloor"):
        empty_cdr_point_source_with_grid.add_release(name="release_1", **invalid_params)


def test_add_release_tracer_zero_fill(start_end_times, valid_release_params):

    start_time, end_time = start_end_times
    cdr = CDRPointSource(start_time=start_time, end_time=end_time)
    release_params = deepcopy(valid_release_params)
    release_params["fill_values"] = "zero_fill"
    cdr.add_release(name="filled_release", **release_params)
    defaults = get_river_tracer_defaults()
    # temp
    assert (cdr.ds["cdr_tracer"].isel(ntracers=0) == defaults["temp"]).all()
    # salt
    assert (cdr.ds["cdr_tracer"].isel(ntracers=1) == defaults["salt"]).all()
    # all other tracers should be zero
    assert (cdr.ds["cdr_tracer"].isel(ntracers=slice(2, None)) == 0.0).all()


def test_add_release_tracer_auto_fill(start_end_times, valid_release_params):

    start_time, end_time = start_end_times
    # Check that the tracer concentrations are auto-filled where missing
    cdr = CDRPointSource(start_time=start_time, end_time=end_time)
    release_params = deepcopy(valid_release_params)
    release_params["fill_values"] = "auto_fill"
    cdr.add_release(name="filled_release", **release_params)

    defaults = get_river_tracer_defaults()
    # temp
    assert (cdr.ds["cdr_tracer"].isel(ntracers=0) == defaults["temp"]).all()
    # salt
    assert (cdr.ds["cdr_tracer"].isel(ntracers=1) == defaults["salt"]).all()
    # ALK
    assert (cdr.ds["cdr_tracer"].isel(ntracers=11) == defaults["ALK"]).all()
    # all other tracers should also be equal to the tracer default values, so not equal to zero
    assert (cdr.ds["cdr_tracer"].isel(ntracers=slice(2, None)) > 0.0).all()

def test_conserve_tracers(start_end_times, valid_release_params):

    start_time, end_time = start_end_times
    cdr = CDRPointSource(start_time=start_time, end_time=end_time)
    
    release_params = deepcopy(valid_release_params)
    release_params["times"] = [
        datetime(2022, 1, 1),
        datetime(2022, 1, 3),
        datetime(2022, 1, 5),
    ]
    release_params["volumes"] = [1.0, 2.0, 3.0]
    release_params["tracer_concentrations"] = {"DIC": [10.0, 20.0, 30.0]}
    cdr.add_release(name="release1", **release_params)

    release_params = deepcopy(valid_release_params)
    release_params["times"] = [
        datetime(2022, 1, 2),
        datetime(2022, 1, 4),
    ]
    release_params["volumes"] = [2.0, 4.0]
    release_params["tracer_concentrations"] = {"DIC": [20.0, 40.0]}
    cdr.add_release(name="release2", **release_params)

    expected_times = [
            datetime(2022, 1, 1), # overall start time 
            datetime(2022, 1, 2),
            datetime(2022, 1, 3),
            datetime(2022, 1, 4),
            datetime(2022, 1, 5),
            datetime(2022, 12, 31) # overall end time
    ]
    assert cdr.ds["time"].values == expected_times

    dic_index = 9

    # check first release
    ncdr_index = 0
    assert cdr.ds["cdr_volume"].isel(ncdr=ncdr_index).values == [1.0, 1.5, 2.0, 2.5, 3.0, 0.0]
    assert cdr.ds["cdr_tracer"].isel(ncdr=ncdr_index, ntracers=dic_index) == [10.0, 15.0, 20.0, 25.0, 30.0, 0.0]

    # Check tracer conservation: tracer * volume should match interpolation of original tracer * volume
    union_times = ds["time"].values.astype("datetime64[ns]")
    times_float = np.array(times, dtype="datetime64[ns]").astype(np.float64)
    union_float = union_times.astype(np.float64)

    expected_total_tracer = np.interp(
        union_float,
        times_float,
        np.array(tracer_concentrations["dic"]) * np.array(volumes),
    )

    actual_total_tracer = (volume * tracer).values

    np.testing.assert_allclose(actual_total_tracer, expected_total_tracer, rtol=1e-10)


