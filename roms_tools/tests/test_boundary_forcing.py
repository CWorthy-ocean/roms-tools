import pytest
from datetime import datetime
from roms_tools import BoundaryForcing, Grid, VerticalCoordinate
import xarray as xr
import numpy as np
import tempfile
import os
import textwrap
from roms_tools.setup.datasets import download_test_data


@pytest.fixture
def example_grid():
    """
    Fixture for creating a Grid object.
    """
    grid = Grid(
        nx=2, ny=2, size_x=500, size_y=1000, center_lon=0, center_lat=55, rot=10
    )

    return grid


@pytest.fixture
def example_vertical_coordinate(example_grid):
    """
    Fixture for creating a VerticalCoordinate object.
    """
    vertical_coordinate = VerticalCoordinate(
        grid=example_grid,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    return vertical_coordinate


@pytest.fixture
def boundary_forcing(example_grid, example_vertical_coordinate):
    """
    Fixture for creating a BoundaryForcing object.
    """

    fname = download_test_data("GLORYS_test_data.nc")

    return BoundaryForcing(
        grid=example_grid,
        vertical_coordinate=example_vertical_coordinate,
        start_time=datetime(2021, 6, 29),
        end_time=datetime(2021, 6, 30),
        source="glorys",
        filename=fname,
    )


def test_boundary_forcing_creation(boundary_forcing):
    """
    Test the creation of the BoundaryForcing object.
    """
    assert boundary_forcing.start_time == datetime(2021, 6, 29)
    assert boundary_forcing.end_time == datetime(2021, 6, 30)
    assert boundary_forcing.filename == download_test_data("GLORYS_test_data.nc")
    assert boundary_forcing.source == "glorys"


def test_boundary_forcing_ds_attribute(boundary_forcing):
    """
    Test the ds attribute of the BoundaryForcing object.
    """
    assert isinstance(boundary_forcing.ds, xr.Dataset)
    for direction in ["south", "east", "north", "west"]:
        assert f"temp_{direction}" in boundary_forcing.ds
        assert f"salt_{direction}" in boundary_forcing.ds
        assert f"u_{direction}" in boundary_forcing.ds
        assert f"v_{direction}" in boundary_forcing.ds
        assert f"zeta_{direction}" in boundary_forcing.ds


def test_boundary_forcing_data_consistency_plot_save(boundary_forcing):
    """
    Test that the data within the BoundaryForcing object remains consistent.
    Also test plot and save methods in the same test since we dask arrays are already computed.
    """
    boundary_forcing.ds.load()

    # Define the expected data
    expected_zeta_south = np.array(
        [[-0.30468762, -0.29416865, -0.30391693, -0.32985148]], dtype=np.float32
    )
    expected_zeta_east = np.array(
        [[-0.32985148, -0.36176518, -0.40663475, -0.40699923]], dtype=np.float32
    )
    expected_zeta_north = np.array(
        [[-0.5534979, -0.5270749, -0.45107934, -0.40699923]], dtype=np.float32
    )
    expected_zeta_west = np.array(
        [[-0.30468762, -0.34336275, -0.3699948, -0.5534979]], dtype=np.float32
    )

    expected_temp_south = np.array(
        [
            [
                [16.84414, 16.905312, 16.967817],
                [18.088203, 18.121834, 18.315424],
                [18.431192, 18.496748, 18.718002],
                [19.294329, 19.30358, 19.439777],
            ]
        ],
        dtype=np.float32,
    )
    expected_temp_east = np.array(
        [
            [
                [19.294329, 19.30358, 19.439777],
                [18.633307, 18.637077, 18.667465],
                [8.710737, 11.25943, 13.111585],
                [9.20282, 10.667074, 11.752404],
            ]
        ],
        dtype=np.float32,
    )
    expected_temp_north = np.array(
        [
            [
                [10.233599, 10.546486, 10.671082],
                [10.147332, 10.502733, 10.68275],
                [10.458557, 11.209945, 11.377164],
                [9.20282, 10.667074, 11.752404],
            ]
        ],
        dtype=np.float32,
    )
    expected_temp_west = np.array(
        [
            [
                [16.84414, 16.905312, 16.967817],
                [12.639833, 13.479691, 14.426711],
                [11.027701, 11.650267, 12.200586],
                [10.233599, 10.546486, 10.671082],
            ]
        ],
        dtype=np.float32,
    )

    expected_u_south = np.array(
        [[[-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0], [0.0, -0.0, -0.0]]], dtype=np.float32
    )
    expected_u_east = np.array(
        [
            [
                [0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0],
                [0.06979556, 0.06167743, -0.02247071],
                [0.0211786, 0.03679834, 0.0274788],
            ]
        ],
        dtype=np.float32,
    )
    expected_u_north = np.array(
        [
            [
                [0.04268532, 0.03889201, 0.03351666],
                [0.04645353, 0.04914769, 0.03673013],
                [0.0211786, 0.03679834, 0.0274788],
            ]
        ],
        dtype=np.float32,
    )
    expected_u_west = np.array(
        [
            [
                [-0.0, -0.0, -0.0],
                [0.0, -0.0, -0.0],
                [0.0, 0.0, -0.0],
                [0.04268532, 0.03889201, 0.03351666],
            ]
        ],
        dtype=np.float32,
    )

    expected_v_south = np.array(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, -0.0], [-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0]]],
        dtype=np.float32,
    )
    expected_v_east = np.array(
        [
            [
                [-0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0],
                [-0.06720348, -0.08354441, -0.13835917],
            ]
        ],
        dtype=np.float32,
    )
    expected_v_north = np.array(
        [
            [
                [-0.00951457, -0.00576979, -0.02147919],
                [-0.0, -0.0, -0.0],
                [0.01915873, 0.02625698, 0.01757628],
                [-0.06720348, -0.08354441, -0.13835917],
            ]
        ],
        dtype=np.float32,
    )
    expected_v_west = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [-0.0, -0.0, -0.0],
                [-0.00951457, -0.00576979, -0.02147919],
            ]
        ],
        dtype=np.float32,
    )

    expected_ubar_south = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
    expected_ubar_east = np.array(
        [[0.0], [0.0], [0.04028399], [0.02812303]], dtype=np.float32
    )
    expected_ubar_north = np.array(
        [[0.03866891], [0.04446249], [0.02812303]], dtype=np.float32
    )
    expected_ubar_west = np.array([[0.0], [0.0], [0.0], [0.03866891]], dtype=np.float32)

    expected_vbar_south = np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    expected_vbar_east = np.array([[0.0], [0.0], [-0.09326097]], dtype=np.float32)
    expected_vbar_north = np.array(
        [[-0.01189703], [0.0], [0.02102064], [-0.09326097]], dtype=np.float32
    )
    expected_vbar_west = np.array([[0.0], [0.0], [-0.01189703]], dtype=np.float32)

    # Check the values in the dataset
    assert np.allclose(boundary_forcing.ds["zeta_south"].values, expected_zeta_south)
    assert np.allclose(boundary_forcing.ds["zeta_east"].values, expected_zeta_east)
    assert np.allclose(boundary_forcing.ds["zeta_north"].values, expected_zeta_north)
    assert np.allclose(boundary_forcing.ds["zeta_west"].values, expected_zeta_west)
    assert np.allclose(boundary_forcing.ds["temp_south"].values, expected_temp_south)
    assert np.allclose(boundary_forcing.ds["temp_east"].values, expected_temp_east)
    assert np.allclose(boundary_forcing.ds["temp_north"].values, expected_temp_north)
    assert np.allclose(boundary_forcing.ds["temp_west"].values, expected_temp_west)
    assert np.allclose(boundary_forcing.ds["u_south"].values, expected_u_south)
    assert np.allclose(boundary_forcing.ds["u_east"].values, expected_u_east)
    assert np.allclose(boundary_forcing.ds["u_north"].values, expected_u_north)
    assert np.allclose(boundary_forcing.ds["u_west"].values, expected_u_west)
    assert np.allclose(boundary_forcing.ds["v_south"].values, expected_v_south)
    assert np.allclose(boundary_forcing.ds["v_east"].values, expected_v_east)
    assert np.allclose(boundary_forcing.ds["v_north"].values, expected_v_north)
    assert np.allclose(boundary_forcing.ds["v_west"].values, expected_v_west)
    assert np.allclose(boundary_forcing.ds["ubar_south"].values, expected_ubar_south)
    assert np.allclose(boundary_forcing.ds["ubar_east"].values, expected_ubar_east)
    assert np.allclose(boundary_forcing.ds["ubar_north"].values, expected_ubar_north)
    assert np.allclose(boundary_forcing.ds["ubar_west"].values, expected_ubar_west)
    assert np.allclose(boundary_forcing.ds["vbar_south"].values, expected_vbar_south)
    assert np.allclose(boundary_forcing.ds["vbar_east"].values, expected_vbar_east)
    assert np.allclose(boundary_forcing.ds["vbar_north"].values, expected_vbar_north)
    assert np.allclose(boundary_forcing.ds["vbar_west"].values, expected_vbar_west)

    boundary_forcing.plot(varname="temp_south")
    boundary_forcing.plot(varname="temp_east")
    boundary_forcing.plot(varname="temp_north")
    boundary_forcing.plot(varname="temp_west")
    boundary_forcing.plot(varname="zeta_south")
    boundary_forcing.plot(varname="zeta_east")
    boundary_forcing.plot(varname="zeta_north")
    boundary_forcing.plot(varname="zeta_west")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    boundary_forcing.save(filepath)
    extended_filepath = filepath + ".20210629-29.nc"

    try:
        assert os.path.exists(extended_filepath)
    finally:
        os.remove(extended_filepath)


def test_roundtrip_yaml(boundary_forcing):
    """Test that creating a BoundaryForcing object, saving its parameters to yaml file, and re-opening yaml file creates the same object."""

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    try:
        boundary_forcing.to_yaml(filepath)

        boundary_forcing_from_file = BoundaryForcing.from_yaml(filepath)

        assert boundary_forcing == boundary_forcing_from_file

    finally:
        os.remove(filepath)


def test_from_yaml_missing_boundary_forcing():
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
      smooth_factor: 8
      hmin: 5.0
      rmax: 0.2
    """
    )

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        yaml_filepath = tmp_file.name
        tmp_file.write(yaml_content.encode())

    try:
        with pytest.raises(
            ValueError, match="No BoundaryForcing configuration found in the YAML file."
        ):
            BoundaryForcing.from_yaml(yaml_filepath)
    finally:
        os.remove(yaml_filepath)
