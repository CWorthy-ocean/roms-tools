import pytest
from roms_tools import Grid, TidalForcing
import xarray as xr
from roms_tools.setup.datasets import download_test_data


@pytest.fixture
def grid_that_lies_within_bounds_of_regional_tpxo_data():
    grid = Grid(
        nx=5, ny=5, size_x=1800, size_y=1500, center_lon=235, center_lat=25, rot=-20
    )

    return grid


@pytest.fixture
def grid_that_straddles_dateline():
    """
    Fixture for creating a domain that straddles the dateline.
    """
    grid = Grid(
        nx=5,
        ny=5,
        size_x=1800,
        size_y=2400,
        center_lon=-10,
        center_lat=30,
        rot=20,
    )

    return grid


@pytest.fixture
def grid_that_straddles_180_degree_meridian():
    """
    Fixture for creating a domain that straddles 180 degree meridian.
    """

    grid = Grid(
        nx=5,
        ny=5,
        size_x=1800,
        size_y=2400,
        center_lon=180,
        center_lat=30,
        rot=20,
    )

    return grid


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid_that_lies_within_bounds_of_regional_tpxo_data",
        "grid_that_straddles_dateline",
        "grid_that_straddles_180_degree_meridian",
    ],
)
def test_successful_initialization_with_global_data(grid_fixture, request):

    fname = download_test_data("TPXO_global_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    tidal_forcing = TidalForcing(grid=grid, filename=fname, source="tpxo", ntides=2)

    assert isinstance(tidal_forcing.ds, xr.Dataset)
    assert "omega" in tidal_forcing.ds
    assert "ssh_Re" in tidal_forcing.ds
    assert "ssh_Im" in tidal_forcing.ds
    assert "pot_Re" in tidal_forcing.ds
    assert "pot_Im" in tidal_forcing.ds
    assert "u_Re" in tidal_forcing.ds
    assert "u_Im" in tidal_forcing.ds
    assert "v_Re" in tidal_forcing.ds
    assert "v_Im" in tidal_forcing.ds

    assert tidal_forcing.filename == fname
    assert tidal_forcing.source == "tpxo"
    assert tidal_forcing.ntides == 2


def test_successful_initialization_with_regional_data(
    grid_that_lies_within_bounds_of_regional_tpxo_data,
):

    fname = download_test_data("TPXO_regional_test_data.nc")

    tidal_forcing = TidalForcing(
        grid=grid_that_lies_within_bounds_of_regional_tpxo_data,
        filename=fname,
        source="tpxo",
        ntides=10,
    )

    assert isinstance(tidal_forcing.ds, xr.Dataset)
    assert "omega" in tidal_forcing.ds
    assert "ssh_Re" in tidal_forcing.ds
    assert "ssh_Im" in tidal_forcing.ds
    assert "pot_Re" in tidal_forcing.ds
    assert "pot_Im" in tidal_forcing.ds
    assert "u_Re" in tidal_forcing.ds
    assert "u_Im" in tidal_forcing.ds
    assert "v_Re" in tidal_forcing.ds
    assert "v_Im" in tidal_forcing.ds

    assert tidal_forcing.filename == fname
    assert tidal_forcing.source == "tpxo"
    assert tidal_forcing.ntides == 10


@pytest.mark.parametrize(
    "grid_fixture",
    ["grid_that_straddles_dateline", "grid_that_straddles_180_degree_meridian"],
)
def test_unsuccessful_initialization_with_regional_data(grid_fixture, request):

    fname = download_test_data("TPXO_regional_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    with pytest.raises(
        ValueError, match="Selected longitude range does not intersect with dataset"
    ):
        TidalForcing(grid=grid, filename=fname, source="tpxo", ntides=10)


def test_insufficient_number_of_consituents(grid_that_straddles_dateline):

    fname = download_test_data("TPXO_global_test_data.nc")

    with pytest.raises(ValueError, match="The dataset contains fewer"):
        TidalForcing(
            grid=grid_that_straddles_dateline, filename=fname, source="tpxo", ntides=10
        )
