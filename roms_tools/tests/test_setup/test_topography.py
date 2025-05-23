import numpy as np
import numpy.testing as npt
import pytest
from scipy.ndimage import label

from roms_tools import Grid
from roms_tools.download import download_test_data
from roms_tools.setup.topography import _compute_rfactor


def test_enclosed_regions():
    """Test that there are only two connected regions, one dry and one wet."""

    grid = Grid(
        nx=100,
        ny=100,
        size_x=1800,
        size_y=2400,
        center_lon=30,
        center_lat=61,
        rot=20,
    )

    reg, nreg = label(grid.ds.mask_rho)
    npt.assert_equal(nreg, 2)


def test_rmax_criterion():
    grid = Grid(
        nx=100,
        ny=100,
        size_x=1800,
        size_y=2400,
        center_lon=30,
        center_lat=61,
        rot=20,
    )
    r_eta, r_xi = _compute_rfactor(grid.ds.h)
    rmax0 = np.max([r_eta.max(), r_xi.max()])
    npt.assert_array_less(rmax0, 0.2)


def test_hmin_criterion():
    grid = Grid(
        nx=100,
        ny=100,
        size_x=1800,
        size_y=2400,
        center_lon=30,
        center_lat=61,
        rot=20,
        hmin=5.0,
    )

    assert grid.hmin == 5.0
    assert np.less_equal(grid.hmin, grid.ds.h.min())

    grid.update_topography(hmin=10.0)

    assert grid.hmin == 10.0
    assert np.less_equal(grid.hmin, grid.ds.h.min())

    # this should not do anything
    grid.update_topography()

    assert grid.hmin == 10.0
    assert np.less_equal(grid.hmin, grid.ds.h.min())


def test_mask_topography_boundary():
    """Test that the mask and topography along the grid boundaries (north, south, east,
    west) are identical to the adjacent inland cells."""

    # Create a grid with some land along the northern boundary
    grid = Grid(
        nx=10, ny=10, size_x=1000, size_y=1000, center_lon=-20, center_lat=60, rot=0
    )

    # Toopography
    np.testing.assert_array_equal(
        grid.ds.h.isel(eta_rho=0).data, grid.ds.h.isel(eta_rho=1).data
    )
    np.testing.assert_array_equal(
        grid.ds.h.isel(eta_rho=-1).data, grid.ds.h.isel(eta_rho=-2).data
    )
    np.testing.assert_array_equal(
        grid.ds.h.isel(xi_rho=0).data, grid.ds.h.isel(xi_rho=1).data
    )
    np.testing.assert_array_equal(
        grid.ds.h.isel(xi_rho=-1).data, grid.ds.h.isel(xi_rho=-2).data
    )

    # Mask
    np.testing.assert_array_equal(
        grid.ds.mask_rho.isel(eta_rho=0).data, grid.ds.mask_rho.isel(eta_rho=1).data
    )
    np.testing.assert_array_equal(
        grid.ds.mask_rho.isel(eta_rho=-1).data, grid.ds.mask_rho.isel(eta_rho=-2).data
    )
    np.testing.assert_array_equal(
        grid.ds.mask_rho.isel(xi_rho=0).data, grid.ds.mask_rho.isel(xi_rho=1).data
    )
    np.testing.assert_array_equal(
        grid.ds.mask_rho.isel(xi_rho=-1).data, grid.ds.mask_rho.isel(xi_rho=-2).data
    )


@pytest.fixture()
def grid_that_straddles_dateline_with_shifted_global_etopo_data():

    grid = Grid(
        nx=5,
        ny=5,
        size_x=1000,
        size_y=1000,
        center_lon=0,
        center_lat=0,
        rot=20,
        topography_source={
            "name": "ETOPO5",
            "path": download_test_data("etopo5_coarsened_and_shifted.nc"),
        },
    )

    return grid


@pytest.fixture()
def grid_that_straddles_dateline_with_global_srtm15_data():

    grid = Grid(
        nx=5,
        ny=5,
        size_x=1000,
        size_y=1000,
        center_lon=0,
        center_lat=0,
        rot=20,
        topography_source={
            "name": "SRTM15",
            "path": download_test_data("srtm15_coarsened.nc"),
        },
    )

    return grid


@pytest.fixture()
def grid_that_straddles_180_degree_meridian_with_shifted_global_etopo_data():

    grid = Grid(
        nx=5,
        ny=5,
        size_x=1000,
        size_y=1000,
        center_lon=180,
        center_lat=0,
        rot=20,
        topography_source={
            "name": "ETOPO5",
            "path": download_test_data("etopo5_coarsened_and_shifted.nc"),
        },
    )

    return grid


@pytest.fixture()
def grid_that_straddles_180_degree_meridian_with_global_srtm15_data():

    grid = Grid(
        nx=5,
        ny=5,
        size_x=1000,
        size_y=1000,
        center_lon=180,
        center_lat=0,
        rot=20,
        topography_source={
            "name": "SRTM15",
            "path": download_test_data("srtm15_coarsened.nc"),
        },
    )

    return grid


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid_that_straddles_dateline",
        "grid_that_straddles_180_degree_meridian",
        "grid_that_straddles_dateline_with_shifted_global_etopo_data",
        "grid_that_straddles_180_degree_meridian_with_shifted_global_etopo_data",
        "grid_that_straddles_dateline_with_global_srtm15_data",
        "grid_that_straddles_180_degree_meridian_with_global_srtm15_data",
    ],
)
def test_successful_initialization(grid_fixture, request):

    grid = request.getfixturevalue(grid_fixture)
    assert grid is not None
