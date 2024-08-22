import pytest
import numpy as np
import numpy.testing as npt
import xarray as xr
from roms_tools import Grid
import os
import tempfile
import importlib.metadata
import textwrap
from roms_tools.setup.download import download_test_data


@pytest.fixture
def simple_grid():

    grid = Grid(nx=1, ny=1, size_x=100, size_y=100, center_lon=-20, center_lat=0, rot=0)

    return grid


def test_grid_creation(simple_grid):

    assert simple_grid.nx == 1
    assert simple_grid.ny == 1
    assert simple_grid.size_x == 100
    assert simple_grid.size_y == 100
    assert simple_grid.center_lon == -20
    assert simple_grid.center_lat == 0
    assert simple_grid.rot == 0
    assert isinstance(simple_grid.ds, xr.Dataset)


def test_plot_save_methods(simple_grid, tmp_path):

    simple_grid.plot(bathymetry=True)
    filepath = tmp_path / "grid.nc"
    simple_grid.save(filepath)
    assert filepath.exists()


@pytest.fixture
def simple_grid_that_straddles_dateline():

    grid = Grid(nx=1, ny=1, size_x=100, size_y=100, center_lon=0, center_lat=0, rot=20)

    return grid


def test_simple_regression(simple_grid):

    expected_lat = np.array(
        [
            [-8.99249453e-01, -8.99249453e-01, -8.99249453e-01],
            [0.0, 0.0, 0.0],
            [8.99249453e-01, 8.99249453e-01, 8.99249453e-01],
        ]
    )
    expected_lon = np.array(
        [
            [339.10072286, 340.0, 340.89927714],
            [339.10072286, 340.0, 340.89927714],
            [339.10072286, 340.0, 340.89927714],
        ]
    )

    expected_lat_u = np.array(
        [
            [-8.99249453e-01, -8.99249453e-01],
            [-2.39610306e-15, -2.28602368e-15],
            [8.99249453e-01, 8.99249453e-01],
        ]
    )

    expected_lon_u = np.array(
        [
            [339.55036143, 340.44963857],
            [339.55036143, 340.44963857],
            [339.55036143, 340.44963857],
        ]
    )

    expected_lat_v = np.array(
        [[-0.44962473, -0.44962473, -0.44962473], [0.44962473, 0.44962473, 0.44962473]]
    )

    expected_lon_v = np.array(
        [[339.10072286, 340.0, 340.89927714], [339.10072286, 340.0, 340.89927714]]
    )

    expected_lat_coarse = np.array(
        [[-1.34887418, -1.34887418], [1.34887418, 1.34887418]]
    )

    expected_lon_coarse = np.array(
        [[338.65108429, 341.34891571], [338.65108429, 341.34891571]]
    )

    expected_angle = np.array(
        [
            [0.00000000e00, 0.00000000e00, 0.00000000e00],
            [-3.83707366e-17, -3.83707366e-17, -3.83707366e-17],
            [0.00000000e00, 0.00000000e00, 0.00000000e00],
        ]
    )

    expected_angle_coarse = np.array(
        [[1.91853683e-17, 1.91853683e-17], [1.91853683e-17, 1.91853683e-17]]
    )

    # TODO: adapt tolerances according to order of magnitude of respective fields
    npt.assert_allclose(simple_grid.ds["lat_rho"], expected_lat, atol=1e-8)
    npt.assert_allclose(simple_grid.ds["lon_rho"], expected_lon, atol=1e-8)
    npt.assert_allclose(simple_grid.ds["lat_u"], expected_lat_u, atol=1e-8)
    npt.assert_allclose(simple_grid.ds["lon_u"], expected_lon_u, atol=1e-8)
    npt.assert_allclose(simple_grid.ds["lat_v"], expected_lat_v, atol=1e-8)
    npt.assert_allclose(simple_grid.ds["lon_v"], expected_lon_v, atol=1e-8)
    npt.assert_allclose(simple_grid.ds["lat_coarse"], expected_lat_coarse, atol=1e-8)
    npt.assert_allclose(simple_grid.ds["lon_coarse"], expected_lon_coarse, atol=1e-8)
    npt.assert_allclose(simple_grid.ds["angle"], expected_angle, atol=1e-8)
    npt.assert_allclose(
        simple_grid.ds["angle_coarse"], expected_angle_coarse, atol=1e-8
    )


def test_simple_regression_dateline(simple_grid_that_straddles_dateline):

    expected_lat = np.array(
        [
            [-1.15258151e00, -8.45014017e-01, -5.37470876e-01],
            [-3.07559747e-01, -2.14815958e-15, 3.07559747e-01],
            [5.37470876e-01, 8.45014017e-01, 1.15258151e00],
        ]
    )
    expected_lon = np.array(
        [
            [3.59462527e02, 3.07583728e-01, 1.15258257e00],
            [3.59154948e02, 7.81866146e-16, 8.45052212e-01],
            [3.58847417e02, 3.59692416e02, 5.37473154e-01],
        ]
    )
    expected_lat_u = np.array(
        [
            [-0.99879776, -0.69124245],
            [-0.15377987, 0.15377987],
            [0.69124245, 0.99879776],
        ]
    )
    expected_lon_u = np.array(
        [
            [3.59885055e02, 7.30083149e-01],
            [3.59577474e02, 4.22526106e-01],
            [3.59269917e02, 1.14944713e-01],
        ]
    )
    expected_lat_v = np.array(
        [[-0.73007063, -0.42250701, -0.11495556], [0.11495556, 0.42250701, 0.73007063]]
    )
    expected_lon_v = np.array(
        [
            [3.59308737e02, 1.53791864e-01, 9.98817391e-01],
            [3.59001183e02, 3.59846208e02, 6.91262683e-01],
        ]
    )
    expected_lat_coarse = np.array(
        [[-1.72887807, -0.80621877], [0.80621877, 1.72887807]]
    )
    expected_lon_coarse = np.array(
        [[359.19378677, 1.72883383], [358.27116617, 0.80621323]]
    )
    expected_angle = np.array(
        [
            [0.34910175, 0.34910175, 0.34910175],
            [0.34906217, 0.34906217, 0.34906217],
            [0.34910175, 0.34910175, 0.34910175],
        ]
    )
    expected_angle_coarse = np.array(
        [[0.34912155, 0.34912155], [0.34912155, 0.34912155]]
    )

    # TODO: adapt tolerances according to order of magnitude of respective fields
    npt.assert_allclose(
        simple_grid_that_straddles_dateline.ds["lat_rho"], expected_lat, atol=1e-8
    )
    npt.assert_allclose(
        simple_grid_that_straddles_dateline.ds["lon_rho"], expected_lon, atol=1e-8
    )
    npt.assert_allclose(
        simple_grid_that_straddles_dateline.ds["lat_u"], expected_lat_u, atol=1e-8
    )
    npt.assert_allclose(
        simple_grid_that_straddles_dateline.ds["lon_u"], expected_lon_u, atol=1e-8
    )
    npt.assert_allclose(
        simple_grid_that_straddles_dateline.ds["lat_v"], expected_lat_v, atol=1e-8
    )
    npt.assert_allclose(
        simple_grid_that_straddles_dateline.ds["lon_v"], expected_lon_v, atol=1e-8
    )
    npt.assert_allclose(
        simple_grid_that_straddles_dateline.ds["lat_coarse"],
        expected_lat_coarse,
        atol=1e-8,
    )
    npt.assert_allclose(
        simple_grid_that_straddles_dateline.ds["lon_coarse"],
        expected_lon_coarse,
        atol=1e-8,
    )
    npt.assert_allclose(
        simple_grid_that_straddles_dateline.ds["angle"], expected_angle, atol=1e-8
    )
    npt.assert_allclose(
        simple_grid_that_straddles_dateline.ds["angle_coarse"],
        expected_angle_coarse,
        atol=1e-8,
    )


def test_raise_if_domain_too_large():
    with pytest.raises(ValueError, match="Domain size has to be smaller"):
        Grid(nx=3, ny=3, size_x=30000, size_y=30000, center_lon=0, center_lat=51.5)

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


def test_compatability_with_matlab_grid():

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
            "layer_depth_rho",
            "layer_depth_u",
            "layer_depth_v",
            "interface_depth_rho",
            "interface_depth_u",
            "interface_depth_v",
        ]
    )
    actual_coords = set(grid.ds.coords.keys())
    assert actual_coords == expected_coords

    grid.plot(bathymetry=True)
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name
    try:
        # Save the grid to a file
        grid.save(filepath)

        # Load the grid from the file
        grid_from_file = Grid.from_file(filepath)

        # Assert that the initial grid and the loaded grid are equivalent (including the 'ds' attribute)
        assert grid == grid_from_file

    finally:
        os.remove(filepath)


def test_roundtrip_netcdf():
    """Test that creating a grid, saving it to file, and re-opening it is the same as just creating it."""

    # Initialize a Grid object using the initializer
    grid_init = Grid(
        nx=10,
        ny=15,
        size_x=100.0,
        size_y=150.0,
        center_lon=0.0,
        center_lat=0.0,
        rot=0.0,
        topography_source="ETOPO5",
        hmin=5.0,
    )

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    try:
        # Save the grid to a file
        grid_init.save(filepath)

        # Load the grid from the file
        grid_from_file = Grid.from_file(filepath)

        # Assert that the initial grid and the loaded grid are equivalent (including the 'ds' attribute)
        assert grid_init == grid_from_file

    finally:
        os.remove(filepath)


def test_roundtrip_yaml():
    """Test that creating a grid, saving its parameters to yaml file, and re-opening yaml file creates the same grid."""

    # Initialize a Grid object using the initializer
    grid_init = Grid(
        nx=10,
        ny=15,
        size_x=100.0,
        size_y=150.0,
        center_lon=0.0,
        center_lat=0.0,
        rot=0.0,
        topography_source="ETOPO5",
        hmin=5.0,
    )

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    try:
        grid_init.to_yaml(filepath)

        grid_from_file = Grid.from_yaml(filepath)

        # Assert that the initial grid and the loaded grid are equivalent (including the 'ds' attribute)
        assert grid_init == grid_from_file

    finally:
        os.remove(filepath)


def test_from_yaml_missing_version():

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
      topography_source: ETOPO5
      hmin: 5.0
    """
    )
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        yaml_filepath = tmp_file.name
        tmp_file.write(yaml_content.encode())

    try:
        with pytest.raises(
            ValueError, match="Version of ROMS-Tools not found in the YAML file."
        ):
            Grid.from_yaml(yaml_filepath)
    finally:
        os.remove(yaml_filepath)


def test_from_yaml_missing_grid():
    roms_tools_version = importlib.metadata.version("roms-tools")

    yaml_content = f"---\nroms_tools_version: {roms_tools_version}\n---\n"

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        yaml_filepath = tmp_file.name
        tmp_file.write(yaml_content.encode())

    try:
        with pytest.raises(
            ValueError, match="No Grid configuration found in the YAML file."
        ):
            Grid.from_yaml(yaml_filepath)
    finally:
        os.remove(yaml_filepath)


def test_from_yaml_version_mismatch():
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
      topography_source: ETOPO5
      hmin: 5.0
    """
    )

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        yaml_filepath = tmp_file.name
        tmp_file.write(yaml_content.encode())

    try:
        with pytest.warns(
            UserWarning,
            match="Current roms-tools version.*does not match the version in the YAML header.*",
        ):
            Grid.from_yaml(yaml_filepath)
    finally:
        os.remove(yaml_filepath)
