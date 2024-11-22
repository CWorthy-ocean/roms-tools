import pytest
from datetime import datetime
from roms_tools import (
    Grid,
    TidalForcing,
    InitialConditions,
    BoundaryForcing,
    SurfaceForcing,
    RiverForcing,
)
from roms_tools.setup.datasets import (
    GLORYSDataset,
    ERA5Dataset,
    CESMBGCDataset,
    CESMBGCSurfaceForcingDataset,
    TPXODataset,
)
from roms_tools.setup.download import download_test_data
import hashlib


def pytest_addoption(parser):
    parser.addoption(
        "--overwrite",
        action="append",
        default=[],
        help="Specify which fixtures to overwrite. Use 'all' to overwrite all fixtures.",
    )
    parser.addoption(
        "--use_dask", action="store_true", default=False, help="Run tests with Dask"
    )


def pytest_configure(config):
    if "all" in config.getoption("--overwrite"):
        # If 'all' is specified, overwrite everything
        config.option.overwrite = ["all"]


@pytest.fixture(scope="session")
def use_dask(request):
    return request.config.getoption("--use_dask")


@pytest.fixture(scope="session")
def grid():

    grid = Grid(nx=1, ny=1, size_x=100, size_y=100, center_lon=-20, center_lat=0, rot=0)

    return grid


@pytest.fixture(scope="session")
def grid_that_straddles_dateline():

    grid = Grid(nx=1, ny=1, size_x=100, size_y=100, center_lon=0, center_lat=0, rot=20)

    return grid


@pytest.fixture(scope="session")
def tidal_forcing(request, use_dask):

    grid = Grid(
        nx=3, ny=3, size_x=1500, size_y=1500, center_lon=235, center_lat=25, rot=-20
    )
    fname = download_test_data("TPXO_regional_test_data.nc")

    return TidalForcing(
        grid=grid, source={"name": "TPXO", "path": fname}, ntides=1, use_dask=use_dask
    )


@pytest.fixture(scope="session")
def initial_conditions(request, use_dask):
    """Fixture for creating an InitialConditions object."""

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
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def initial_conditions_with_bgc(request, use_dask):
    """Fixture for creating an InitialConditions object."""

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
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def initial_conditions_with_bgc_from_climatology(request, use_dask):
    """Fixture for creating an InitialConditions object."""

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
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def boundary_forcing(request, use_dask):
    """Fixture for creating a BoundaryForcing object."""

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
        apply_2d_horizontal_fill=True,
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def bgc_boundary_forcing_from_climatology(request, use_dask):
    """Fixture for creating a BoundaryForcing object."""

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
        apply_2d_horizontal_fill=True,
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def surface_forcing(request, use_dask):
    """Fixture for creating a SurfaceForcing object."""

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
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def coarse_surface_forcing(request, use_dask):
    """Fixture for creating a SurfaceForcing object."""

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
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def corrected_surface_forcing(request, use_dask):
    """Fixture for creating a SurfaceForcing object with shortwave radiation
    correction."""

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
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def bgc_surface_forcing(request, use_dask):
    """Fixture for creating a SurfaceForcing object with BGC."""
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
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def bgc_surface_forcing_from_climatology(request, use_dask):
    """Fixture for creating a SurfaceForcing object with BGC from climatology."""
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
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def river_forcing():
    """Fixture for creating a RiverForcing object from the global Dai river dataset."""
    grid = Grid(
        nx=18, ny=18, size_x=800, size_y=800, center_lon=-18, center_lat=65, rot=20, N=3
    )

    start_time = datetime(1998, 1, 1)
    end_time = datetime(1998, 3, 1)

    return RiverForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
    )


@pytest.fixture(scope="session")
def river_forcing_no_climatology():
    """Fixture for creating a RiverForcing object from the global Dai river dataset."""
    grid = Grid(
        nx=18, ny=18, size_x=800, size_y=800, center_lon=-18, center_lat=65, rot=20, N=3
    )

    start_time = datetime(1998, 1, 1)
    end_time = datetime(1998, 3, 1)

    return RiverForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        convert_to_climatology="never",
    )


@pytest.fixture(scope="session")
def era5_data(request, use_dask):
    fname = download_test_data("ERA5_regional_test_data.nc")
    data = ERA5Dataset(
        filename=fname,
        start_time=datetime(2020, 1, 31),
        end_time=datetime(2020, 2, 2),
        use_dask=use_dask,
    )

    return data


@pytest.fixture(scope="session")
def glorys_data(request, use_dask):
    # the following GLORYS data has a wide enough domain
    # to have different masks for tracers vs. velocities
    fname = download_test_data("GLORYS_test_data.nc")

    data = GLORYSDataset(
        filename=fname,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2013, 1, 1),
        use_dask=use_dask,
    )

    ds = data.ds.isel(depth=[0, 10, 30])
    object.__setattr__(data, "ds", ds)

    data.extrapolate_deepest_to_bottom()

    return data


@pytest.fixture(scope="session")
def tpxo_data(request, use_dask):
    fname = download_test_data("TPXO_regional_test_data.nc")

    data = TPXODataset(
        filename=fname,
        use_dask=use_dask,
    )

    return data


@pytest.fixture(scope="session")
def cesm_bgc_data(request, use_dask):
    fname = download_test_data("CESM_regional_test_data_one_time_slice.nc")

    data = CESMBGCDataset(
        filename=fname,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2013, 1, 1),
        climatology=False,
        use_dask=use_dask,
    )

    return data


@pytest.fixture(scope="session")
def coarsened_cesm_bgc_data(request, use_dask):
    fname = download_test_data("CESM_BGC_2012.nc")

    data = CESMBGCDataset(
        filename=fname,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2013, 1, 1),
        climatology=False,
        use_dask=use_dask,
    )

    data.extrapolate_deepest_to_bottom()

    return data


@pytest.fixture(scope="session")
def cesm_surface_bgc_data(request, use_dask):
    fname = download_test_data("CESM_BGC_SURFACE_2012.nc")

    data = CESMBGCSurfaceForcingDataset(
        filename=fname,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2013, 1, 1),
        climatology=False,
        use_dask=use_dask,
    )
    data.post_process()

    return data


def calculate_file_hash(filepath, hash_algorithm="sha256"):
    """Calculate the hash of a file using the specified hash algorithm."""
    hash_func = hashlib.new(hash_algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()
