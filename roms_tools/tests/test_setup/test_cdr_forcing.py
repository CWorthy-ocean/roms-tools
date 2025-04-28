import pytest
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from roms_tools import CDRVolumePointSource, Grid
from roms_tools.constants import NUM_TRACERS
import xarray as xr
import numpy as np
import logging
from roms_tools.setup.utils import get_tracer_defaults
from conftest import calculate_file_hash

try:
    import xesmf  # type: ignore
except ImportError:
    xesmf = None

# Fixtures
@pytest.fixture
def iceland_test_grid():
    """Returns a grid surrouding Iceland."""
    return Grid(
        nx=18, ny=18, size_x=800, size_y=800, center_lon=-18, center_lat=65, rot=0, N=3
    )


@pytest.fixture
def test_grid_that_straddles():
    """Returns a grid that straddles the prime meridian."""
    return Grid(
        nx=18, ny=18, size_x=800, size_y=800, center_lon=0, center_lat=65, rot=0, N=3
    )


@pytest.fixture
def start_end_times():
    """Returns test start and end times."""
    start_time = datetime(2022, 1, 1)
    end_time = datetime(2022, 12, 31)
    return start_time, end_time


@pytest.fixture
def empty_cdr_point_source_without_grid(start_end_times):
    """Returns an empty CDR point source without a grid."""
    start_time, end_time = start_end_times
    return CDRVolumePointSource(
        start_time=start_time,
        end_time=end_time,
    )


@pytest.fixture
def empty_cdr_point_source_with_grid(iceland_test_grid, start_end_times):
    """Returns an empty CDR point source with the Iceland test grid."""
    start_time, end_time = start_end_times
    return CDRVolumePointSource(
        grid=iceland_test_grid,
        start_time=start_time,
        end_time=end_time,
    )


@pytest.fixture
def empty_cdr_point_source_with_grid_that_straddles(
    test_grid_that_straddles, start_end_times
):
    """Returns an empty CDR point source with a grid straddling the prime meridian."""
    start_time, end_time = start_end_times
    return CDRVolumePointSource(
        grid=test_grid_that_straddles,
        start_time=start_time,
        end_time=end_time,
    )


@pytest.fixture
def valid_release_params():
    """Returns a dictionary with valid parameters for a CDR point source release within
    the Iceland test domain."""
    return {
        "lat": 66.0,
        "lon": -25.0,
        "depth": 50.0,
        "volume_fluxes": 100.0,
    }


@pytest.fixture
def cdr_point_source_with_two_releases(
    iceland_test_grid, start_end_times, valid_release_params
):
    """Returns a CDR point source with one release."""
    start_time, end_time = start_end_times
    cdr = CDRVolumePointSource(
        grid=iceland_test_grid, start_time=start_time, end_time=end_time
    )
    cdr.add_release(name="release1", **valid_release_params)

    release_params = deepcopy(valid_release_params)
    release_params["times"] = [
        datetime(2022, 1, 1),
        datetime(2022, 1, 3),
        datetime(2022, 1, 5),
    ]
    release_params["volume_fluxes"] = [1.0, 2.0, 3.0]
    release_params["tracer_concentrations"] = {"DIC": [10.0, 20.0, 30.0]}
    cdr.add_release(name="release2", **release_params)

    return cdr


# Tests
@pytest.mark.parametrize(
    "cdr_forcing_fixture",
    [
        "empty_cdr_point_source_without_grid",
        "empty_cdr_point_source_with_grid",
    ],
)
def test_cdr_point_source_init(cdr_forcing_fixture, start_end_times, request):
    """Tests the initialization of CDR point source fixtures."""
    cdr = request.getfixturevalue(cdr_forcing_fixture)
    start_time, end_time = start_end_times
    assert cdr.start_time == start_time
    assert cdr.end_time == end_time

    # Check that dataset is empty but has right dimensions and variables
    assert isinstance(cdr.ds, xr.Dataset)

    # Check dimension lengths
    assert cdr.ds.time.size == 0
    assert cdr.ds.ncdr.size == 0
    assert cdr.ds.ntracers.size == NUM_TRACERS

    # Check coordinate and variable lengths
    assert cdr.ds.release_name.size == 0
    assert cdr.ds.tracer_name.size == NUM_TRACERS
    assert cdr.ds.tracer_unit.size == NUM_TRACERS
    assert cdr.ds.tracer_long_name.size == NUM_TRACERS
    assert cdr.ds.cdr_time.size == 0
    assert cdr.ds.cdr_lon.size == 0
    assert cdr.ds.cdr_lat.size == 0
    assert cdr.ds.cdr_dep.size == 0
    assert cdr.ds.cdr_hsc.size == 0
    assert cdr.ds.cdr_vsc.size == 0
    assert cdr.ds.cdr_volume.size == 0
    assert cdr.ds.cdr_tracer.size == 0

    # Check that release dictionary is empty except for tracer metadata
    assert set(cdr.releases.keys()) == {"_tracer_metadata"}


@pytest.mark.parametrize(
    "cdr_forcing_fixture",
    [
        "empty_cdr_point_source_without_grid",
        "empty_cdr_point_source_with_grid",
    ],
)
def test_add_release(cdr_forcing_fixture, valid_release_params, request):
    """Test some basic features of the `add_release` method for updating forcing dataset
    and dictionary."""

    cdr = request.getfixturevalue(cdr_forcing_fixture)
    release_params = deepcopy(valid_release_params)
    times = [
        cdr.start_time,
        datetime(2022, 1, 5),
        cdr.end_time,
    ]
    release_params["times"] = times

    volume_fluxes = [100.0, 200.0, 150.0]
    release_params["volume_fluxes"] = volume_fluxes

    tracer_concentrations = {
        "ALK": [2300.0, 2350.0, 2400.0],
        "DIC": [2100.0, 2150.0, 2200.0],
        "temp": 10.0,
        "salt": 35.0,
    }
    release_params["tracer_concentrations"] = tracer_concentrations

    # Add release
    cdr.add_release(name="release", **release_params)

    # Check dimension lengths
    assert cdr.ds.time.size == len(times)
    assert cdr.ds.ncdr.size == 1
    assert cdr.ds.ntracers.size == NUM_TRACERS

    # Check coordinate and variable lengths
    assert cdr.ds.release_name.size == 1
    assert "release" in cdr.ds["release_name"].values
    assert cdr.ds.tracer_name.size == NUM_TRACERS
    assert cdr.ds.tracer_unit.size == NUM_TRACERS
    assert cdr.ds.tracer_long_name.size == NUM_TRACERS
    assert cdr.ds.cdr_time.size == len(times)
    assert cdr.ds.cdr_lon.size == 1
    assert cdr.ds.cdr_lat.size == 1
    assert cdr.ds.cdr_dep.size == 1
    assert cdr.ds.cdr_hsc.size == 1
    assert cdr.ds.cdr_vsc.size == 1

    # Check cdr_volume shape and values
    assert cdr.ds.cdr_volume.shape == (len(times), 1)
    np.testing.assert_allclose(cdr.ds.cdr_volume[:, 0], volume_fluxes, rtol=1e-3)

    # Check tracer concentration shape
    assert cdr.ds.cdr_tracer.shape == (len(times), NUM_TRACERS, 1)

    # Check tracer concentration values for known tracers
    tracer_index = {name: i for i, name in enumerate(cdr.ds.tracer_name.values)}
    for tracer, expected in tracer_concentrations.items():
        i = tracer_index[tracer]
        if isinstance(expected, list):
            np.testing.assert_allclose(cdr.ds.cdr_tracer[:, i, 0], expected)
        else:
            np.testing.assert_allclose(cdr.ds.cdr_tracer[:, i, 0], expected)

    assert "release" in cdr.releases.keys()


def test_merge_multiple_releases(start_end_times, valid_release_params):
    """Test merging multiple releases in the dataset, including endpoint filling,
    timestamp adjustment, and interpolation."""

    start_time, end_time = start_end_times
    cdr = CDRVolumePointSource(start_time=start_time, end_time=end_time)
    dic_index = 9

    # add first release
    release_params1 = deepcopy(valid_release_params)
    release_params1["times"] = [
        datetime(2022, 1, 1),  # overall start time
        datetime(2022, 1, 3),
        datetime(2022, 1, 5),
    ]
    release_params1["volume_fluxes"] = [1.0, 2.0, 3.0]
    release_params1["tracer_concentrations"] = {"DIC": [10.0, 20.0, 30.0]}
    cdr.add_release(name="release1", **release_params1)

    # check time
    expected_times = [
        datetime(2022, 1, 1),  # overall start time
        datetime(2022, 1, 3),
        datetime(2022, 1, 5),
        datetime(2022, 12, 31),  # overall end time
    ]
    assert np.array_equal(
        cdr.ds["time"].values, np.array(expected_times, dtype="datetime64[ns]")
    )

    # check first release
    ncdr_index = 0

    assert cdr.ds["cdr_lon"].isel(ncdr=ncdr_index).values == release_params1["lon"]
    assert cdr.ds["cdr_lat"].isel(ncdr=ncdr_index).values == release_params1["lat"]
    assert cdr.ds["cdr_dep"].isel(ncdr=ncdr_index).values == release_params1["depth"]
    assert cdr.ds["cdr_hsc"].isel(ncdr=ncdr_index).values == 0.0
    assert cdr.ds["cdr_vsc"].isel(ncdr=ncdr_index).values == 0.0

    expected_volume_fluxes = [
        1.0,
        2.0,
        3.0,
        0.0,  # volume flux set to zero at endpoint
    ]

    assert np.allclose(
        cdr.ds["cdr_volume"].isel(ncdr=ncdr_index).values,
        np.array(expected_volume_fluxes),
    )

    expected_dics = [
        10.0,
        20.0,
        30.0,
        30.0,  # tracer concenctration extrapolated to endpoint
    ]
    assert np.allclose(
        cdr.ds["cdr_tracer"].isel(ncdr=ncdr_index, ntracers=dic_index).values,
        np.array(expected_dics),
    )

    # add second release
    release_params2 = deepcopy(valid_release_params)
    release_params2["lon"] = release_params2["lon"] - 1
    release_params2["lat"] = release_params2["lat"] - 1
    release_params2["depth"] = release_params2["depth"] - 1

    release_params2["times"] = [
        datetime(2022, 1, 2),
        datetime(2022, 1, 4),
        datetime(2022, 1, 5),
    ]
    release_params2["volume_fluxes"] = [2.0, 4.0, 10.0]
    release_params2["tracer_concentrations"] = {"DIC": [20.0, 40.0, 100.0]}
    cdr.add_release(name="release2", **release_params2)

    # check time again
    expected_times = [
        datetime(2022, 1, 1),  # overall start time
        datetime(2022, 1, 2),
        datetime(2022, 1, 3),
        datetime(2022, 1, 4),
        datetime(2022, 1, 5),
        datetime(2022, 12, 31),  # overall end time
    ]
    assert np.array_equal(
        cdr.ds["time"].values, np.array(expected_times, dtype="datetime64[ns]")
    )

    # check first release again
    ncdr_index = 0

    assert cdr.ds["cdr_lon"].isel(ncdr=ncdr_index).values == release_params1["lon"]
    assert cdr.ds["cdr_lat"].isel(ncdr=ncdr_index).values == release_params1["lat"]
    assert cdr.ds["cdr_dep"].isel(ncdr=ncdr_index).values == release_params1["depth"]
    assert cdr.ds["cdr_hsc"].isel(ncdr=ncdr_index).values == 0.0
    assert cdr.ds["cdr_vsc"].isel(ncdr=ncdr_index).values == 0.0

    expected_volume_fluxes = [
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        0.0,  # volume flux set to zero at endpoint
    ]

    assert np.allclose(
        cdr.ds["cdr_volume"].isel(ncdr=ncdr_index).values,
        np.array(expected_volume_fluxes),
    )

    expected_dics = [
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        30.0,  # tracer concenctration extrapolated to endpoint
    ]
    assert np.allclose(
        cdr.ds["cdr_tracer"].isel(ncdr=ncdr_index, ntracers=dic_index).values,
        np.array(expected_dics),
    )
    # check second release
    ncdr_index = 1

    assert cdr.ds["cdr_lon"].isel(ncdr=ncdr_index).values == release_params2["lon"]
    assert cdr.ds["cdr_lat"].isel(ncdr=ncdr_index).values == release_params2["lat"]
    assert cdr.ds["cdr_dep"].isel(ncdr=ncdr_index).values == release_params2["depth"]
    assert cdr.ds["cdr_hsc"].isel(ncdr=ncdr_index).values == 0.0
    assert cdr.ds["cdr_vsc"].isel(ncdr=ncdr_index).values == 0.0

    expected_volume_fluxes = [
        0.0,  # volume flux set to zero at startpoint
        2.0,
        3.0,
        4.0,
        10.0,
        0.0,  # volume flux set to zero at endpoint
    ]
    assert np.allclose(
        cdr.ds["cdr_volume"].isel(ncdr=ncdr_index).values,
        np.array(expected_volume_fluxes),
    )

    expected_dics = [
        20.0,  # tracer concenctration extrapolated to startpoint
        20.0,
        30.0,
        40.0,
        100.0,
        100.0,  # tracer concenctration extrapolated to endpoint
    ]
    assert np.allclose(
        cdr.ds["cdr_tracer"].isel(ncdr=ncdr_index, ntracers=dic_index).values,
        np.array(expected_dics),
    )


def test_cdr_point_source_init_invalid_times():
    """Test that initializing a CDR point source with the same start and end time raises
    a ValueError."""
    start_time = datetime(2022, 5, 1)
    end_time = datetime(2022, 5, 1)
    with pytest.raises(
        ValueError, match="`start_time` must be earlier than `end_time`"
    ):
        CDRVolumePointSource(start_time=start_time, end_time=end_time)


@pytest.mark.parametrize(
    "cdr_forcing_fixture",
    [
        "empty_cdr_point_source_without_grid",
        "empty_cdr_point_source_with_grid",
    ],
)
def test_add_duplicate_release(cdr_forcing_fixture, valid_release_params, request):
    """Test that adding a duplicate release raises a ValueError."""
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
    """Test that invalid release parameters raise the appropriate ValueErrors."""
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

    # Test times being not monotonically increasing
    invalid_params = deepcopy(valid_release_params)
    invalid_params["times"] = [
        datetime(2022, 1, 1),
        datetime(2022, 2, 1),
        datetime(2022, 1, 15),  # Out of order date
        datetime(2022, 3, 1),
    ]
    with pytest.raises(
        ValueError, match="The 'times' list must be strictly monotonically increasing."
    ):
        cdr.add_release(name="release_1", **invalid_params)

    # Test times being not strictly monotonically increasing
    invalid_params = deepcopy(valid_release_params)
    invalid_params["times"] = [
        datetime(2022, 1, 1),
        datetime(2022, 2, 1),
        datetime(2022, 2, 1),  # Duplicated time
        datetime(2022, 3, 1),
    ]
    with pytest.raises(
        ValueError, match="The 'times' list must be strictly monotonically increasing."
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

    # Test invalid volume_fluxes: not a float/int or list of float/int
    invalid_params = deepcopy(valid_release_params)
    invalid_params["volume_fluxes"] = ["not", "valid"]
    with pytest.raises(ValueError, match="Invalid 'volume_fluxes' input"):
        cdr.add_release(name="release_invalid_volume", **invalid_params)

    # Test invalid tracer_concentrations: not a float/int or list of float/int
    invalid_params = deepcopy(valid_release_params)
    invalid_params["tracer_concentrations"] = {"ALK": ["not", "valid"]}
    with pytest.raises(ValueError, match="Invalid tracer concentration for 'ALK'"):
        cdr.add_release(name="release_invalid_tracer", **invalid_params)

    # Test mismatch between times and volume fluxes length
    invalid_params = deepcopy(valid_release_params)
    invalid_params["times"] = [datetime(2022, 1, 1), datetime(2022, 1, 2)]  # Two times
    invalid_params["volume_fluxes"] = [100]  # Only one volume flux entry
    with pytest.raises(ValueError, match="The length of `volume_fluxes` "):
        cdr.add_release(name="release_1", **invalid_params)

    # Test mismatch between times and tracer_concentrations length
    invalid_params = deepcopy(valid_release_params)
    invalid_params["times"] = [datetime(2022, 1, 1), datetime(2022, 1, 2)]  # Two times
    invalid_params["tracer_concentrations"] = {
        "ALK": [1]
    }  # Only one tracer concentration
    with pytest.raises(ValueError, match="The length of tracer 'ALK'"):
        cdr.add_release(name="release_1", **invalid_params)

    # Test invalid volume flux (negative)
    invalid_params = deepcopy(valid_release_params)
    invalid_params["volume_fluxes"] = -100  # Invalid volume flux
    with pytest.raises(ValueError, match="Volume flux must be non-negative"):
        cdr.add_release(name="release_1", **invalid_params)

    # Test volume flux as list with negative values
    invalid_params = deepcopy(valid_release_params)
    invalid_params["times"] = [cdr.start_time, cdr.end_time]
    invalid_params["volume_fluxes"] = [10, -5]  # Invalid volume fluxes in list
    with pytest.raises(
        ValueError, match="All entries in `volume_fluxes` must be non-negative"
    ):
        cdr.add_release(name="release_1", **invalid_params)

    # Test invalid tracer concentration (negative)
    invalid_params = deepcopy(valid_release_params)
    invalid_params["tracer_concentrations"] = {"ALK": -1}
    with pytest.raises(ValueError, match="The concentration of tracer"):
        cdr.add_release(name="release_1", **invalid_params)

    # Test tracer_concentration as list with negative values
    invalid_params = deepcopy(valid_release_params)
    invalid_params["times"] = [cdr.start_time, cdr.end_time]
    invalid_params["tracer_concentrations"] = {
        "ALK": [10, -5]
    }  # Invalid concentration in list
    with pytest.raises(ValueError, match="All entries in "):
        cdr.add_release(name="release_1", **invalid_params)


def test_warning_no_grid(
    empty_cdr_point_source_without_grid, valid_release_params, caplog
):
    """Test warning if no grid is provided."""
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
    """Test that error is raised if release location is outside grid."""

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
    """Test that error is raised if release location is outside grid or on land."""
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
    """Test that zero fill of tracer concentrations works as expected."""
    start_time, end_time = start_end_times
    cdr = CDRVolumePointSource(start_time=start_time, end_time=end_time)
    release_params = deepcopy(valid_release_params)
    release_params["fill_values"] = "zero"
    cdr.add_release(name="filled_release", **release_params)
    defaults = get_tracer_defaults()
    # temp
    assert (cdr.ds["cdr_tracer"].isel(ntracers=0) == defaults["temp"]).all()
    # salt
    assert (cdr.ds["cdr_tracer"].isel(ntracers=1) == defaults["salt"]).all()
    # all other tracers should be zero
    assert (cdr.ds["cdr_tracer"].isel(ntracers=slice(2, None)) == 0.0).all()


def test_add_release_tracer_auto_fill(start_end_times, valid_release_params):
    """Test that auto fill of tracer concentrations works as expected."""
    start_time, end_time = start_end_times
    # Check that the tracer concentrations are auto-filled where missing
    cdr = CDRVolumePointSource(start_time=start_time, end_time=end_time)
    release_params = deepcopy(valid_release_params)
    release_params["fill_values"] = "auto"
    cdr.add_release(name="filled_release", **release_params)

    defaults = get_tracer_defaults()
    # temp
    assert (cdr.ds["cdr_tracer"].isel(ntracers=0) == defaults["temp"]).all()
    # salt
    assert (cdr.ds["cdr_tracer"].isel(ntracers=1) == defaults["salt"]).all()
    # ALK
    assert (cdr.ds["cdr_tracer"].isel(ntracers=11) == defaults["ALK"]).all()
    # all other tracers should also be equal to the tracer default values, so not equal to zero
    assert (cdr.ds["cdr_tracer"].isel(ntracers=slice(2, None)) > 0.0).all()


def test_add_release_invalid_fill(start_end_times, valid_release_params):
    """Test that invalid fill method of tracer concentrations raises error."""
    start_time, end_time = start_end_times
    cdr = CDRVolumePointSource(start_time=start_time, end_time=end_time)
    release_params = deepcopy(valid_release_params)
    release_params["fill_values"] = "zero_fill"

    with pytest.raises(ValueError, match="Invalid fill_values option"):

        cdr.add_release(name="filled_release", **release_params)


def test_plot_error_when_no_grid(start_end_times, valid_release_params):
    """Test that error is raised if plotting without a grid."""
    start_time, end_time = start_end_times
    cdr = CDRVolumePointSource(start_time=start_time, end_time=end_time)
    release_params = deepcopy(valid_release_params)
    cdr.add_release(name="release1", **release_params)

    with pytest.raises(ValueError, match="A grid must be provided for plotting"):
        cdr.plot_location_top_view("all")

    with pytest.raises(ValueError, match="A grid must be provided for plotting"):
        cdr.plot_location_side_view("release1")


def test_plot(cdr_point_source_with_two_releases):
    """Test that plotting method run without error."""

    cdr_point_source_with_two_releases.plot_volume_flux()
    cdr_point_source_with_two_releases.plot_tracer_concentration("ALK")
    cdr_point_source_with_two_releases.plot_tracer_concentration("DIC")

    cdr_point_source_with_two_releases.plot_location_top_view()


@pytest.mark.skipif(xesmf is None, reason="xesmf required")
def test_plot_side_view(cdr_point_source_with_two_releases):
    """Test that plotting method run without error."""

    cdr_point_source_with_two_releases.plot_location_side_view("release1")


def test_plot_more_errors(cdr_point_source_with_two_releases):
    """Test that error is raised on bad plot args or ambiguous release."""

    with pytest.raises(ValueError, match="Multiple releases found"):
        cdr_point_source_with_two_releases.plot_location_side_view()

    with pytest.raises(ValueError, match="Invalid release"):
        cdr_point_source_with_two_releases.plot_location_side_view(release="fake")

    with pytest.raises(ValueError, match="Invalid releases"):
        cdr_point_source_with_two_releases.plot_location_top_view(releases=["fake"])

    with pytest.raises(ValueError, match="should be a string"):
        cdr_point_source_with_two_releases.plot_location_top_view(releases=4)

    with pytest.raises(ValueError, match="list must be strings"):
        cdr_point_source_with_two_releases.plot_location_top_view(releases=[4])


def test_cdr_forcing_save(cdr_point_source_with_two_releases, tmp_path):
    """Test save method."""

    for file_str in ["test_cdr_forcing", "test_cdr_forcing.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [tmp_path / file_str, str(tmp_path / file_str)]:

            saved_filenames = cdr_point_source_with_two_releases.save(filepath)
            # Check if the .nc file was created
            filepath = Path(filepath).with_suffix(".nc")
            assert saved_filenames == [filepath]
            assert filepath.exists()
            # Clean up the .nc file
            filepath.unlink()


def test_roundtrip_yaml(cdr_point_source_with_two_releases, tmp_path):
    """Test that creating a CDRVolumePointSource object, saving its parameters to yaml
    file, and re-opening yaml file creates the same object."""

    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str

        cdr_point_source_with_two_releases.to_yaml(filepath)

        cdr_forcing_from_file = CDRVolumePointSource.from_yaml(filepath)

        assert cdr_point_source_with_two_releases == cdr_forcing_from_file

        filepath = Path(filepath)
        filepath.unlink()


def test_files_have_same_hash(cdr_point_source_with_two_releases, tmp_path):
    """Test that saving the same CDR forcing configuration to NetCDF twice results in
    reproducible file hashes."""

    yaml_filepath = tmp_path / "test_yaml.yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"

    cdr_point_source_with_two_releases.to_yaml(yaml_filepath)
    cdr_point_source_with_two_releases.save(filepath1)
    cdr_from_file = CDRVolumePointSource.from_yaml(yaml_filepath)
    cdr_from_file.save(filepath2)

    hash1 = calculate_file_hash(filepath1)
    hash2 = calculate_file_hash(filepath2)

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    filepath1.unlink()
    filepath2.unlink()
