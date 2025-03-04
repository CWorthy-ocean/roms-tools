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
    TPXODataset,
    GLORYSDataset,
    ERA5Dataset,
    CESMBGCDataset,
    CESMBGCSurfaceForcingDataset,
)
from roms_tools.download import download_test_data
import hashlib
import h5py
from pathlib import Path


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

    grid = Grid(
        nx=1, ny=1, size_x=1000, size_y=1000, center_lon=0.5, center_lat=0, rot=20
    )

    return grid


@pytest.fixture(scope="session")
def grid_that_straddles_180_degree_meridian():
    """Fixture for creating a domain that straddles 180 degree meridian.

    This is a good test grid for the global ERA5 data, which comes on an [-180, 180]
    longitude grid.
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


@pytest.fixture(scope="session")
def tidal_forcing(request, use_dask):

    grid = Grid(
        nx=3, ny=3, size_x=1500, size_y=1500, center_lon=235, center_lat=25, rot=-20
    )
    fname_grid = Path(download_test_data("regional_grid_tpxo10v2.nc"))
    fname_h = Path(download_test_data("regional_h_tpxo10v2.nc"))
    fname_u = Path(download_test_data("regional_u_tpxo10v2.nc"))
    fname_dict = {"grid": fname_grid, "h": fname_h, "u": fname_u}

    return TidalForcing(
        grid=grid,
        source={"name": "TPXO", "path": fname_dict},
        ntides=1,
        use_dask=use_dask,
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

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))

    return InitialConditions(
        grid=grid,
        ini_time=datetime(2021, 6, 29),
        source={"path": fname, "name": "GLORYS"},
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def initial_conditions_adjusted_for_zeta(request, use_dask):
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

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))

    return InitialConditions(
        grid=grid,
        ini_time=datetime(2021, 6, 29),
        source={"path": fname, "name": "GLORYS"},
        adjust_depth_for_sea_surface_height=True,
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

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    fname_bgc = Path(download_test_data("CESM_regional_test_data_one_time_slice.nc"))

    return InitialConditions(
        grid=grid,
        ini_time=datetime(2021, 6, 29),
        source={"path": fname, "name": "GLORYS"},
        bgc_source={"path": fname_bgc, "name": "CESM_REGRIDDED"},
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def initial_conditions_with_bgc_adjusted_for_zeta(request, use_dask):
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

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    fname_bgc = Path(download_test_data("CESM_regional_test_data_one_time_slice.nc"))

    return InitialConditions(
        grid=grid,
        ini_time=datetime(2021, 6, 29),
        source={"path": fname, "name": "GLORYS"},
        bgc_source={"path": fname_bgc, "name": "CESM_REGRIDDED"},
        adjust_depth_for_sea_surface_height=True,
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

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    fname_bgc = Path(download_test_data("CESM_regional_test_data_climatology.nc"))

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
def boundary_forcing(use_dask):
    """Fixture for creating a BoundaryForcing object."""
    grid = Grid(
        nx=3,
        ny=3,
        size_x=400,
        size_y=400,
        center_lon=-8,
        center_lat=58,
        rot=0,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))
    return BoundaryForcing(
        grid=grid,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2012, 12, 31),
        source={"name": "GLORYS", "path": [fname1, fname2]},
        apply_2d_horizontal_fill=False,
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def boundary_forcing_adjusted_for_zeta(use_dask):
    """Fixture for creating a BoundaryForcing object."""
    grid = Grid(
        nx=3,
        ny=3,
        size_x=400,
        size_y=400,
        center_lon=-8,
        center_lat=58,
        rot=0,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))
    return BoundaryForcing(
        grid=grid,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2012, 12, 31),
        source={"name": "GLORYS", "path": [fname1, fname2]},
        apply_2d_horizontal_fill=False,
        adjust_depth_for_sea_surface_height=True,
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def boundary_forcing_with_2d_fill(use_dask):
    """Fixture for creating a BoundaryForcing object."""
    grid = Grid(
        nx=3,
        ny=3,
        size_x=400,
        size_y=400,
        center_lon=-8,
        center_lat=58,
        rot=0,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))
    return BoundaryForcing(
        grid=grid,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2012, 12, 31),
        source={"name": "GLORYS", "path": [fname1, fname2]},
        apply_2d_horizontal_fill=True,
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def boundary_forcing_with_2d_fill_adjusted_for_zeta(use_dask):
    """Fixture for creating a BoundaryForcing object."""
    grid = Grid(
        nx=3,
        ny=3,
        size_x=400,
        size_y=400,
        center_lon=-8,
        center_lat=58,
        rot=0,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))
    return BoundaryForcing(
        grid=grid,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2012, 12, 31),
        source={"name": "GLORYS", "path": [fname1, fname2]},
        apply_2d_horizontal_fill=True,
        adjust_depth_for_sea_surface_height=True,
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

    fname_bgc = Path(
        download_test_data("CESM_regional_coarse_test_data_climatology.nc")
    )

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

    fname = Path(download_test_data("ERA5_global_test_data.nc"))

    return SurfaceForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        source={"name": "ERA5", "path": fname},
        correct_radiation=False,
        coarse_grid_mode="never",
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

    fname = Path(download_test_data("ERA5_global_test_data.nc"))

    return SurfaceForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        coarse_grid_mode="always",
        source={"name": "ERA5", "path": fname},
        correct_radiation=False,
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

    fname = Path(download_test_data("ERA5_global_test_data.nc"))

    return SurfaceForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        source={"name": "ERA5", "path": fname},
        correct_radiation=True,
        coarse_grid_mode="never",
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

    fname_bgc = Path(download_test_data("CESM_surface_global_test_data.nc"))

    return SurfaceForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        source={"name": "CESM_REGRIDDED", "path": fname_bgc},
        type="bgc",
        coarse_grid_mode="never",
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

    fname_bgc = Path(download_test_data("CESM_surface_global_test_data_climatology.nc"))

    return SurfaceForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        source={"name": "CESM_REGRIDDED", "path": fname_bgc, "climatology": True},
        type="bgc",
        coarse_grid_mode="never",
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


@pytest.fixture
def river_forcing_with_bgc(scope="session"):
    """Fixture for creating a RiverForcing object with BGC tracers."""
    grid = Grid(
        nx=18, ny=18, size_x=800, size_y=800, center_lon=-18, center_lat=65, rot=20, N=3
    )

    start_time = datetime(1998, 1, 1)
    end_time = datetime(1998, 3, 1)

    return RiverForcing(
        grid=grid, start_time=start_time, end_time=end_time, include_bgc=True
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
    fname = download_test_data("regional_h_tpxo10v2a.nc")
    fname_grid = download_test_data("regional_grid_tpxo10v2a.nc")

    data = TPXODataset(
        filename=fname,
        grid_filename=fname_grid,
        location="h",
        var_names={"ssh_Re": "hRe", "ssh_Im": "hIm"},
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


def calculate_data_hash(filepath):
    """Calculate the hash of an HDF5 file's datasets, ignoring certain metadata.

    This function computes a SHA-256 hash based on the actual data stored in the
    datasets of an HDF5 file. It excludes metadata attributes such as `STORAGE_LAYOUT`
    to ensure consistency in hashing when metadata differences do not affect data values.

    Parameters:
        filepath (str): Path to the HDF5 file.

    Returns:
        str: The computed SHA-256 hash as a hexadecimal string.
    """
    with h5py.File(filepath, "r") as f:
        # Create a hash object
        hash_obj = hashlib.sha256()

        # Iterate over datasets in the file
        for dataset_name in f:
            dataset = f[dataset_name]

            # Skip metadata like STORAGE_LAYOUT or any other non-data attributes
            # You can skip the dataset attributes you don't care about
            dataset_attrs = list(dataset.attrs)
            for attr in dataset_attrs:
                if attr == "STORAGE_LAYOUT":
                    del dataset.attrs[attr]  # Remove this attribute

            # Update the hash with the actual data (ignoring non-data metadata)
            data = dataset[()]
            hash_obj.update(data.tobytes())  # Convert data to bytes

        # Return the computed hash
        return hash_obj.hexdigest()
