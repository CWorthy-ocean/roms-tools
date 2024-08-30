import pytest
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
