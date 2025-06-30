import importlib.metadata
import logging
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from scipy.ndimage import label

from conftest import calculate_file_hash
from roms_tools import Grid
from roms_tools.constants import (
    MAXIMUM_GRID_SIZE,
    UPPER_BOUND_THETA_B,
    UPPER_BOUND_THETA_S,
)
from roms_tools.download import download_test_data
from roms_tools.setup.topography import _compute_rfactor


@pytest.fixture()
def counter_clockwise_rotated_grid():

    grid = Grid(
        nx=1, ny=1, size_x=100, size_y=100, center_lon=-20, center_lat=0, rot=20
    )

    return grid


@pytest.fixture()
def clockwise_rotated_grid():

    grid = Grid(
        nx=1, ny=1, size_x=100, size_y=100, center_lon=-20, center_lat=0, rot=-20
    )

    return grid


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


def test_grid_creation(grid):

    assert grid.nx == 1
    assert grid.ny == 1
    assert grid.size_x == 100
    assert grid.size_y == 100
    assert grid.center_lon == -20
    assert grid.center_lat == 0
    assert grid.rot == 0
    assert isinstance(grid.ds, xr.Dataset)


@pytest.mark.parametrize(
    "grid_fixture",
    ["grid", "counter_clockwise_rotated_grid", "clockwise_rotated_grid"],
)
def test_coords_relation(grid_fixture, request):
    """Test that the coordinates satisfy the expected relations on a C-grid."""
    grid = request.getfixturevalue(grid_fixture)

    # psi versus rho
    # assert grid.ds.lon_psi.min() < grid.ds.lon_rho.min()
    # assert grid.ds.lon_psi.max() > grid.ds.lon_rho.max()
    # assert grid.ds.lat_psi.min() < grid.ds.lat_rho.min()
    # assert grid.ds.lat_psi.max() > grid.ds.lat_rho.max()

    # Assertion with tolerance is necessary for non-rotated grids
    def assert_larger_equal_than_with_tolerance(value1, value2, tolerance=1e-5):
        assert value1 >= value2 - tolerance

    def assert_smaller_equal_than_with_tolerance(value1, value2, tolerance=1e-5):
        assert value1 <= value2 + tolerance

    # u versus rho
    assert_larger_equal_than_with_tolerance(grid.ds.lon_u.min(), grid.ds.lon_rho.min())
    assert_larger_equal_than_with_tolerance(grid.ds.lat_u.min(), grid.ds.lat_rho.min())
    assert_smaller_equal_than_with_tolerance(grid.ds.lon_u.max(), grid.ds.lon_rho.max())
    assert_smaller_equal_than_with_tolerance(grid.ds.lon_u.max(), grid.ds.lon_rho.max())

    # v versus rho
    assert_larger_equal_than_with_tolerance(grid.ds.lon_v.min(), grid.ds.lon_rho.min())
    assert_larger_equal_than_with_tolerance(grid.ds.lat_v.min(), grid.ds.lat_rho.min())
    assert_smaller_equal_than_with_tolerance(grid.ds.lon_v.max(), grid.ds.lon_rho.max())
    assert_smaller_equal_than_with_tolerance(grid.ds.lon_v.max(), grid.ds.lon_rho.max())


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
def test_successful_initialization_with_topography(grid_fixture, request):

    grid = request.getfixturevalue(grid_fixture)
    assert grid is not None


def test_plot():

    grid = Grid(
        nx=20, ny=20, size_x=100, size_y=100, center_lon=-20, center_lat=0, rot=0
    )

    grid.plot(bathymetry=True)
    grid.plot(bathymetry=False)


def test_save(tmp_path):

    grid = Grid(
        nx=20, ny=20, size_x=100, size_y=100, center_lon=-20, center_lat=0, rot=0
    )

    for file_str in ["test_grid", "test_grid.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            saved_filenames = grid.save(filepath)
            # Check if the .nc file was created
            filepath = Path(filepath).with_suffix(".nc")

            assert saved_filenames == [filepath]
            assert filepath.exists()
            # Clean up the .nc file
            filepath.unlink()


def test_raise_if_domain_too_large():
    with pytest.raises(ValueError, match="Domain size exceeds"):
        Grid(
            nx=3,
            ny=3,
            size_x=MAXIMUM_GRID_SIZE + 10,
            size_y=1000,
            center_lon=0,
            center_lat=51.5,
        )
    with pytest.raises(ValueError, match="Domain size exceeds"):
        Grid(
            nx=3,
            ny=3,
            size_x=1000,
            size_y=MAXIMUM_GRID_SIZE + 10,
            center_lon=0,
            center_lat=51.5,
        )

    # test grid with reasonable domain size
    grid = Grid(
        nx=3,
        ny=3,
        size_x=1800,
        size_y=2400,
        center_lon=-21,
        center_lat=61,
        rot=20,
    )
    assert isinstance(grid, Grid)


def test_grid_straddle_crosses_meridian():
    grid = Grid(
        nx=3,
        ny=3,
        size_x=100,
        size_y=100,
        center_lon=0,
        center_lat=61,
        rot=20,
    )
    assert grid.straddle

    grid = Grid(
        nx=3,
        ny=3,
        size_x=100,
        size_y=100,
        center_lon=180,
        center_lat=61,
        rot=20,
    )
    assert not grid.straddle


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid",
        "grid_that_straddles_dateline_with_shifted_global_etopo_data",
        "grid_that_straddles_dateline_with_global_srtm15_data",
    ],
)
def test_roundtrip_netcdf(grid_fixture, tmp_path, request):
    """Test that creating a grid, saving it to file, and re-opening it is the same as
    just creating it."""

    grid = request.getfixturevalue(grid_fixture)

    for file_str in ["test_grid", "test_grid.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            grid.save(filepath)

            filepath = Path(filepath)

            # Load the grid from the file
            grid_from_file = Grid.from_file(filepath.with_suffix(".nc"))

            # Assert that the initial grid and the loaded grid are equivalent (including the 'ds' attribute)
            assert grid == grid_from_file

            # Clean up the .nc file
            (filepath.with_suffix(".nc")).unlink()


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid",
        "grid_that_straddles_dateline_with_shifted_global_etopo_data",
        "grid_that_straddles_dateline_with_global_srtm15_data",
    ],
)
def test_roundtrip_yaml(grid_fixture, tmp_path, request):
    """Test that creating a grid, saving its parameters to yaml file, and re-opening
    yaml file creates the same grid."""

    grid = request.getfixturevalue(grid_fixture)

    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str

        grid.to_yaml(filepath)

        grid_from_file = Grid.from_yaml(filepath)

        # Assert that the initial grid and the loaded grid are equivalent (including the 'ds' attribute)
        assert grid == grid_from_file

        filepath = Path(filepath)
        filepath.unlink()


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid",
        "grid_that_straddles_dateline_with_shifted_global_etopo_data",
        "grid_that_straddles_dateline_with_global_srtm15_data",
    ],
)
def test_roundtrip_from_file_yaml(grid_fixture, tmp_path, request):
    """Test that reading a grid from file and then saving it to yaml works."""

    grid = request.getfixturevalue(grid_fixture)

    filepath = Path(tmp_path / "test.nc")
    grid.save(filepath)

    grid_from_file = Grid.from_file(filepath)

    filepath_yaml = Path(tmp_path / "test.yaml")
    grid_from_file.to_yaml(filepath_yaml)

    filepath.unlink()
    filepath_yaml.unlink()


def test_files_have_same_hash(tmp_path):

    # Initialize a Grid object using the initializer
    grid_init = Grid(
        nx=10,
        ny=15,
        size_x=100.0,
        size_y=150.0,
        center_lon=0.0,
        center_lat=0.0,
        rot=0.0,
        topography_source={"name": "ETOPO5"},
        hmin=5.0,
    )

    yaml_filepath = tmp_path / "test_yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"

    grid_init.to_yaml(yaml_filepath)
    grid_init.save(filepath1)

    grid_from_file = Grid.from_yaml(yaml_filepath)
    grid_from_file.save(filepath2)

    hash1 = calculate_file_hash(filepath1)
    hash2 = calculate_file_hash(filepath2)

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    filepath1.unlink()
    filepath2.unlink()


def test_from_yaml_missing_version(tmp_path):

    yaml_content = textwrap.dedent(
        """\
    Grid:
      nx: 100
      ny: 100
      size_x: 1800
      size_y: 2400
      center_lon: -10
      center_lat: 61
      rot: -20
      topography_source:
        name: ETOPO5
      hmin: 5.0
    """
    )
    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for yaml_filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str

        # Write YAML content to file
        if isinstance(yaml_filepath, Path):
            yaml_filepath.write_text(yaml_content)
        else:
            with open(yaml_filepath, "w") as f:
                f.write(yaml_content)

        with pytest.raises(
            ValueError, match="Version of ROMS-Tools not found in the YAML file."
        ):
            Grid.from_yaml(yaml_filepath)

        yaml_filepath = Path(yaml_filepath)
        yaml_filepath.unlink()


def test_from_yaml_missing_grid(tmp_path):
    roms_tools_version = importlib.metadata.version("roms-tools")

    yaml_content = f"---\nroms_tools_version: {roms_tools_version}\n---\n"

    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for yaml_filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str

        # Write YAML content to file
        if isinstance(yaml_filepath, Path):
            yaml_filepath.write_text(yaml_content)
        else:
            with open(yaml_filepath, "w") as f:
                f.write(yaml_content)

        with pytest.raises(
            ValueError, match="No Grid configuration found in the YAML file."
        ):
            Grid.from_yaml(yaml_filepath)

        yaml_filepath = Path(yaml_filepath)
        yaml_filepath.unlink()


def test_from_yaml_version_mismatch(tmp_path, caplog):
    yaml_content = textwrap.dedent(
        """\
    ---
    roms_tools_version: 0.0.0
    ---
    Grid:
      nx: 100
      ny: 100
      size_x: 1800
      size_y: 2400
      center_lon: -10
      center_lat: 61
      rot: -20
      topography_source:
        name: ETOPO5
      hmin: 5.0
    """
    )

    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for yaml_filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str

        # Write YAML content to file
        if isinstance(yaml_filepath, Path):
            yaml_filepath.write_text(yaml_content)
        else:
            with open(yaml_filepath, "w") as f:
                f.write(yaml_content)

        with caplog.at_level(logging.WARNING):
            Grid.from_yaml(yaml_filepath)

        # Verify the warning message in the log
        assert "Current roms-tools version" in caplog.text

        yaml_filepath = Path(yaml_filepath)
        yaml_filepath.unlink()


def test_invalid_theta_s_value():
    """Test the validation of the theta_s value."""
    with pytest.raises(ValueError):

        Grid(
            nx=2,
            ny=2,
            size_x=500,
            size_y=1000,
            center_lon=0,
            center_lat=55,
            rot=10,
            N=3,
            theta_s=UPPER_BOUND_THETA_S + 1,
        )


def test_invalid_theta_b_value():
    """Test the validation of the theta_b value."""
    with pytest.raises(ValueError):
        Grid(
            nx=2,
            ny=2,
            size_x=500,
            size_y=1000,
            center_lon=0,
            center_lat=55,
            rot=10,
            N=3,
            theta_b=UPPER_BOUND_THETA_B + 1,
        )


def test_update_vertical_coordinate():

    grid = Grid(
        nx=2, ny=2, size_x=500, size_y=1000, center_lon=0, center_lat=55, rot=10
    )

    assert grid.N == 100
    assert grid.theta_s == 5.0
    assert grid.theta_b == 2.0
    assert grid.hc == 300.0
    assert len(grid.ds.s_rho) == 100

    grid.update_vertical_coordinate(N=3, theta_s=10.0, theta_b=1.0, hc=400.0)

    assert grid.N == 3
    assert grid.theta_s == 10.0
    assert grid.theta_b == 1.0
    assert grid.hc == 400.0
    assert len(grid.ds.s_rho) == 3

    grid.update_vertical_coordinate(N=5)

    assert grid.N == 5
    assert grid.theta_s == 10.0
    assert grid.theta_b == 1.0
    assert grid.hc == 400.0
    assert len(grid.ds.s_rho) == 5

    grid.update_vertical_coordinate()

    assert grid.N == 5
    assert grid.theta_s == 10.0
    assert grid.theta_b == 1.0
    assert grid.hc == 400.0
    assert len(grid.ds.s_rho) == 5


def test_plot_vertical_coordinate():
    grid = Grid(
        nx=2,
        ny=2,
        size_x=500,
        size_y=1000,
        center_lon=0,
        center_lat=55,
        rot=10,
        N=3,
        theta_s=5.0,
        theta_b=2.0,
        hc=250.0,
    )
    grid.plot_vertical_coordinate(s=-1)
    grid.plot_vertical_coordinate(eta=0)
    grid.plot_vertical_coordinate(xi=0)
    # Test that passing a matplotlib.axes.Axes works
    fig, ax = plt.subplots(1, 1)
    grid.plot_vertical_coordinate(s=-1, ax=ax)
    grid.plot_vertical_coordinate(eta=0, ax=ax)

    with pytest.raises(ValueError, match="Exactly one of"):
        grid.plot_vertical_coordinate(s=-1, eta=0)
    with pytest.raises(ValueError, match="Exactly one of"):
        grid.plot_vertical_coordinate(s=-1, xi=0)
    with pytest.raises(ValueError, match="Exactly one of"):
        grid.plot_vertical_coordinate(eta=-1, xi=0)
    with pytest.raises(ValueError, match="Exactly one of"):
        grid.plot_vertical_coordinate(eta=-1, xi=0, s=-1)


# More Grid.from_file() tests


def test_compatability_with_matlab_grid(tmp_path):

    fname = download_test_data("grid_created_with_matlab.nc")

    grid = Grid.from_file(fname)

    assert not grid.straddle
    assert grid.theta_s == 5.0
    assert grid.theta_b == 2.0
    assert grid.hc == 300.0
    assert grid.N == 100
    assert grid.nx == 24
    assert grid.ny == 24
    assert grid.center_lon == -4.1
    assert grid.center_lat == 52.4
    assert grid.rot == 0.0

    expected_coords = set(
        [
            "lat_rho",
            "lon_rho",
            "lat_u",
            "lon_u",
            "lat_v",
            "lon_v",
            "lat_coarse",
            "lon_coarse",
        ]
    )
    actual_coords = set(grid.ds.coords.keys())
    assert actual_coords == expected_coords

    grid.plot(bathymetry=True)

    for file_str in ["test_grid", "test_grid.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            # Test saving without partitioning
            _ = grid.save(filepath)

            filepath = Path(filepath)

            # Load the grid from the file
            grid_from_file = Grid.from_file(filepath.with_suffix(".nc"))

            # Assert that the initial grid and the loaded grid are equivalent (including the 'ds' attribute)
            assert grid == grid_from_file

            # Clean up the .nc file
            (filepath.with_suffix(".nc")).unlink()


def test_from_file_with_vertical_coords(grid, tmp_path):

    theta_s = 6.0
    theta_b = 4.0
    hc = 300.0
    N = 10

    grid.update_vertical_coordinate(theta_s=theta_s, theta_b=theta_b, hc=hc, N=N)
    Cs_r = grid.ds.Cs_r
    Cs_w = grid.ds.Cs_w
    path = tmp_path / "grid.nc"
    grid.save(path)

    grid_from_file = Grid.from_file(path, theta_s=theta_s, theta_b=theta_b, hc=hc, N=N)
    assert np.allclose(grid_from_file.ds.Cs_r, Cs_r)
    assert np.allclose(grid_from_file.ds.Cs_w, Cs_w)
    assert grid_from_file.theta_s == theta_s
    assert grid_from_file.theta_b == theta_b
    assert grid_from_file.hc == hc
    assert grid_from_file.N == N


def test_from_file_with_conflicting_vertical_coords(grid, tmp_path):

    theta_s = 6.0
    theta_b = 4.0
    hc = 300.0
    N = 10

    grid.update_vertical_coordinate(theta_s=theta_s, theta_b=theta_b, hc=hc, N=N)

    path = tmp_path / "grid.nc"
    grid.save(path)

    with pytest.raises(ValueError, match="inconsistent with the provided N"):
        Grid.from_file(path, theta_s=5.0, theta_b=2.0, hc=300.0, N=100)

    with pytest.raises(ValueError, match="inconsistent with the provided theta_s, "):
        Grid.from_file(path, theta_s=5.0, theta_b=2.0, hc=300.0, N=10)


def test_from_file_missing_attributes(grid, tmp_path):

    del grid.ds.attrs["theta_b"]

    path = tmp_path / "grid.nc"
    grid.save(path)

    with pytest.raises(ValueError, match="Missing vertical coordinate attributes"):
        Grid.from_file(path)


def test_from_file_partial_parameters_raises_error(grid, tmp_path):

    path = tmp_path / "grid.nc"
    grid.save(path)

    with pytest.raises(ValueError, match="must provide all of"):
        Grid.from_file(path, theta_s=5.0)


# Topography tests
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
