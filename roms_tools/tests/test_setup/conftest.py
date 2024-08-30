import pytest
from datetime import datetime
from roms_tools import (
    Grid,
    TidalForcing,
    InitialConditions,
    BoundaryForcing,
    SurfaceForcing,
)
from roms_tools.setup.download import download_test_data


@pytest.fixture(scope="session")
def simple_grid():

    grid = Grid(nx=1, ny=1, size_x=100, size_y=100, center_lon=-20, center_lat=0, rot=0)

    return grid


@pytest.fixture(scope="session")
def simple_grid_that_straddles_dateline():

    grid = Grid(nx=1, ny=1, size_x=100, size_y=100, center_lon=0, center_lat=0, rot=20)

    return grid


@pytest.fixture(scope="session")
def tidal_forcing():

    grid = Grid(
        nx=3, ny=3, size_x=1500, size_y=1500, center_lon=235, center_lat=25, rot=-20
    )
    fname = download_test_data("TPXO_regional_test_data.nc")

    return TidalForcing(
        grid=grid,
        source={"name": "TPXO", "path": fname},
        ntides=1,
    )


@pytest.fixture(scope="session")
def initial_conditions():
    """
    Fixture for creating an InitialConditions object.
    """

    grid = Grid(
        nx=2,
        ny=2,
        size_x=500,
        size_y=1000,
        center_lon=0,
        center_lat=55,
        rot=10,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    fname = download_test_data("GLORYS_coarse_test_data.nc")

    return InitialConditions(
        grid=grid,
        ini_time=datetime(2021, 6, 29),
        source={"path": fname, "name": "GLORYS"},
    )


@pytest.fixture(scope="session")
def initial_conditions_with_bgc():
    """
    Fixture for creating an InitialConditions object.
    """

    grid = Grid(
        nx=2,
        ny=2,
        size_x=500,
        size_y=1000,
        center_lon=0,
        center_lat=55,
        rot=10,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    fname = download_test_data("GLORYS_coarse_test_data.nc")
    fname_bgc = download_test_data("CESM_regional_test_data_one_time_slice.nc")

    return InitialConditions(
        grid=grid,
        ini_time=datetime(2021, 6, 29),
        source={"path": fname, "name": "GLORYS"},
        bgc_source={"path": fname_bgc, "name": "CESM_REGRIDDED"},
    )


@pytest.fixture(scope="session")
def initial_conditions_with_bgc_from_climatology():
    """
    Fixture for creating an InitialConditions object.
    """

    grid = Grid(
        nx=2,
        ny=2,
        size_x=500,
        size_y=1000,
        center_lon=0,
        center_lat=55,
        rot=10,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    fname = download_test_data("GLORYS_coarse_test_data.nc")
    fname_bgc = download_test_data("CESM_regional_test_data_climatology.nc")

    return InitialConditions(
        grid=grid,
        ini_time=datetime(2021, 6, 29),
        source={"path": fname, "name": "GLORYS"},
        bgc_source={
            "path": fname_bgc,
            "name": "CESM_REGRIDDED",
            "climatology": True,
        },
    )


@pytest.fixture(scope="session")
def boundary_forcing():
    """
    Fixture for creating a BoundaryForcing object.
    """

    grid = Grid(
        nx=2,
        ny=2,
        size_x=500,
        size_y=1000,
        center_lon=0,
        center_lat=55,
        rot=10,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    fname = download_test_data("GLORYS_coarse_test_data.nc")

    return BoundaryForcing(
        grid=grid,
        start_time=datetime(2021, 6, 29),
        end_time=datetime(2021, 6, 30),
        source={"name": "GLORYS", "path": fname},
    )


@pytest.fixture(scope="session")
def bgc_boundary_forcing_from_climatology():
    """
    Fixture for creating a BoundaryForcing object.
    """

    grid = Grid(
        nx=2,
        ny=2,
        size_x=500,
        size_y=1000,
        center_lon=0,
        center_lat=55,
        rot=10,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    fname_bgc = download_test_data("CESM_regional_coarse_test_data_climatology.nc")

    return BoundaryForcing(
        grid=grid,
        start_time=datetime(2021, 6, 29),
        end_time=datetime(2021, 6, 30),
        source={"path": fname_bgc, "name": "CESM_REGRIDDED", "climatology": True},
        type="bgc",
    )


@pytest.fixture(scope="session")
def surface_forcing():
    """
    Fixture for creating a SurfaceForcing object.
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

    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")

    return SurfaceForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        source={"name": "ERA5", "path": fname},
    )


@pytest.fixture(scope="session")
def coarse_surface_forcing():
    """
    Fixture for creating a SurfaceForcing object.
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

    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")

    return SurfaceForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        use_coarse_grid=True,
        source={"name": "ERA5", "path": fname},
    )


@pytest.fixture(scope="session")
def corrected_surface_forcing():
    """
    Fixture for creating a SurfaceForcing object with shortwave radiation correction.
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

    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")

    return SurfaceForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        source={"name": "ERA5", "path": fname},
        correct_radiation=True,
    )


@pytest.fixture(scope="session")
def bgc_surface_forcing():
    """
    Fixture for creating a SurfaceForcing object with BGC.
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

    start_time = datetime(2020, 2, 1)
    end_time = datetime(2020, 2, 1)

    fname_bgc = download_test_data("CESM_surface_global_test_data.nc")

    return SurfaceForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        source={"name": "CESM_REGRIDDED", "path": fname_bgc},
        type="bgc",
    )


@pytest.fixture(scope="session")
def bgc_surface_forcing_from_climatology():
    """
    Fixture for creating a SurfaceForcing object with BGC from climatology.
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

    start_time = datetime(2020, 2, 1)
    end_time = datetime(2020, 2, 1)

    fname_bgc = download_test_data("CESM_surface_global_test_data_climatology.nc")

    return SurfaceForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        source={"name": "CESM_REGRIDDED", "path": fname_bgc, "climatology": True},
        type="bgc",
    )
