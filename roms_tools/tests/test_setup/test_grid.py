import pytest
import xarray as xr
from roms_tools import Grid
import importlib.metadata
import textwrap
from roms_tools.setup.download import download_test_data
from conftest import calculate_file_hash
from pathlib import Path


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
    """
    Test that the coordinates satisfy the expected relations on a C-grid.
    """
    grid = request.getfixturevalue(grid_fixture)

    # psi versus rho
    assert grid.ds.lon_psi.min() < grid.ds.lon_rho.min()
    assert grid.ds.lon_psi.max() > grid.ds.lon_rho.max()
    assert grid.ds.lat_psi.min() < grid.ds.lat_rho.min()
    assert grid.ds.lat_psi.max() > grid.ds.lat_rho.max()

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


def test_plot_save_methods(tmp_path):

    grid = Grid(
        nx=20, ny=20, size_x=100, size_y=100, center_lon=-20, center_lat=0, rot=0
    )

    grid.plot(bathymetry=True)

    for file_str in ["test_grid", "test_grid.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            # Test saving without partitioning
            saved_filenames = grid.save(filepath)
            # Check if the .nc file was created
            filepath = Path(filepath).with_suffix(".nc")

            assert saved_filenames == [filepath]
            assert filepath.exists()
            # Clean up the .nc file
            filepath.unlink()

            # Test saving with partitioning
            saved_filenames = grid.save(filepath, np_eta=2, np_xi=5)

            filepath_str = str(filepath.with_suffix(""))
            expected_filepath_list = [
                Path(filepath_str + f".{index}.nc") for index in range(10)
            ]
            assert saved_filenames == expected_filepath_list
            for expected_filepath in expected_filepath_list:
                assert expected_filepath.exists()
                expected_filepath.unlink()


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


def test_roundtrip_netcdf(tmp_path):
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

    for file_str in ["test_grid", "test_grid.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            grid_init.save(filepath)

            filepath = Path(filepath)

            # Load the grid from the file
            grid_from_file = Grid.from_file(filepath.with_suffix(".nc"))

            # Assert that the initial grid and the loaded grid are equivalent (including the 'ds' attribute)
            assert grid_init == grid_from_file

            # Clean up the .nc file
            (filepath.with_suffix(".nc")).unlink()


def test_roundtrip_yaml(tmp_path):
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

    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str

        grid_init.to_yaml(filepath)

        grid_from_file = Grid.from_yaml(filepath)

        # Assert that the initial grid and the loaded grid are equivalent (including the 'ds' attribute)
        assert grid_init == grid_from_file

        filepath = Path(filepath)
        filepath.unlink()


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
        topography_source="ETOPO5",
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
      topography_source: ETOPO5
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


def test_from_yaml_version_mismatch(tmp_path):
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

        with pytest.warns(
            UserWarning,
            match="Current roms-tools version.*does not match the version in the YAML header.*",
        ):
            Grid.from_yaml(yaml_filepath)

        yaml_filepath = Path(yaml_filepath)
        yaml_filepath.unlink()
