import pytest
from datetime import datetime
from roms_tools import InitialConditions, Grid, VerticalCoordinate
import xarray as xr
import numpy as np
import tempfile
import os
import textwrap
from roms_tools.setup.download import download_test_data
from roms_tools.setup.datasets import CESMBGCDataset


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
def initial_conditions(example_grid, example_vertical_coordinate):
    """
    Fixture for creating a dummy InitialConditions object.
    """

    fname = download_test_data("GLORYS_test_data.nc")

    return InitialConditions(
        grid=example_grid,
        vertical_coordinate=example_vertical_coordinate,
        ini_time=datetime(2021, 6, 29),
        physics_source={"path": fname, "name": "GLORYS"},
    )


@pytest.fixture
def initial_conditions_with_bgc(example_grid, example_vertical_coordinate):
    """
    Fixture for creating a dummy InitialConditions object.
    """

    fname = download_test_data("GLORYS_test_data.nc")
    fname_bgc = download_test_data("CESM_regional_test_data_one_time_slice.nc")

    return InitialConditions(
        grid=example_grid,
        vertical_coordinate=example_vertical_coordinate,
        ini_time=datetime(2021, 6, 29),
        physics_source={"path": fname, "name": "GLORYS"},
        bgc_source={"path": fname_bgc, "name": "CESM_REGRIDDED"},
    )


@pytest.fixture
def initial_conditions_with_bgc_from_climatology(
    example_grid, example_vertical_coordinate
):
    """
    Fixture for creating a dummy InitialConditions object.
    """

    fname = download_test_data("GLORYS_test_data.nc")
    fname_bgc = download_test_data("CESM_regional_test_data_climatology.nc")

    return InitialConditions(
        grid=example_grid,
        vertical_coordinate=example_vertical_coordinate,
        ini_time=datetime(2021, 6, 29),
        physics_source={"path": fname, "name": "GLORYS"},
        bgc_source={
            "path": fname_bgc,
            "name": "CESM_REGRIDDED",
            "climatology": True,
        },
    )


@pytest.mark.parametrize(
    "ic_fixture",
    [
        "initial_conditions",
        "initial_conditions_with_bgc",
        "initial_conditions_with_bgc_from_climatology",
    ],
)
def test_initial_conditions_creation(ic_fixture, request):
    """
    Test the creation of the InitialConditions object.
    """

    ic = request.getfixturevalue(ic_fixture)

    assert ic.ini_time == datetime(2021, 6, 29)
    assert ic.physics_source == {
        "name": "GLORYS",
        "path": download_test_data("GLORYS_test_data.nc"),
        "climatology": False,
    }
    assert isinstance(ic.ds, xr.Dataset)
    assert "temp" in ic.ds
    assert "salt" in ic.ds
    assert "u" in ic.ds
    assert "v" in ic.ds
    assert "zeta" in ic.ds


# Test initialization with missing 'name' in physics_source
def test_initial_conditions_missing_physics_name(
    example_grid, example_vertical_coordinate
):
    with pytest.raises(ValueError, match="`physics_source` must include a 'name'."):
        InitialConditions(
            grid=example_grid,
            vertical_coordinate=example_vertical_coordinate,
            ini_time=datetime(2021, 6, 29),
            physics_source={"path": "physics_data.nc"},
        )


# Test initialization with missing 'path' in physics_source
def test_initial_conditions_missing_physics_path(
    example_grid, example_vertical_coordinate
):
    with pytest.raises(ValueError, match="`physics_source` must include a 'path'."):
        InitialConditions(
            grid=example_grid,
            vertical_coordinate=example_vertical_coordinate,
            ini_time=datetime(2021, 6, 29),
            physics_source={"name": "GLORYS"},
        )


# Test initialization with missing 'name' in bgc_source
def test_initial_conditions_missing_bgc_name(example_grid, example_vertical_coordinate):

    fname = download_test_data("GLORYS_test_data.nc")
    with pytest.raises(
        ValueError, match="`bgc_source` must include a 'name' if it is provided."
    ):
        InitialConditions(
            grid=example_grid,
            vertical_coordinate=example_vertical_coordinate,
            ini_time=datetime(2021, 6, 29),
            physics_source={"name": "GLORYS", "path": fname},
            bgc_source={"path": "bgc_data.nc"},
        )


# Test initialization with missing 'path' in bgc_source
def test_initial_conditions_missing_bgc_path(example_grid, example_vertical_coordinate):

    fname = download_test_data("GLORYS_test_data.nc")
    with pytest.raises(
        ValueError, match="`bgc_source` must include a 'path' if it is provided."
    ):
        InitialConditions(
            grid=example_grid,
            vertical_coordinate=example_vertical_coordinate,
            ini_time=datetime(2021, 6, 29),
            physics_source={"name": "GLORYS", "path": fname},
            bgc_source={"name": "CESM_REGRIDDED"},
        )


# Test default climatology value
def test_initial_conditions_default_climatology(
    example_grid, example_vertical_coordinate
):

    fname = download_test_data("GLORYS_test_data.nc")

    initial_conditions = InitialConditions(
        grid=example_grid,
        vertical_coordinate=example_vertical_coordinate,
        ini_time=datetime(2021, 6, 29),
        physics_source={"name": "GLORYS", "path": fname},
    )

    assert initial_conditions.physics_source["climatology"] is False
    assert initial_conditions.bgc_source is None


def test_initial_conditions_default_bgc_climatology(
    example_grid, example_vertical_coordinate
):

    fname = download_test_data("GLORYS_test_data.nc")
    fname_bgc = download_test_data("CESM_regional_test_data_one_time_slice.nc")

    initial_conditions = InitialConditions(
        grid=example_grid,
        vertical_coordinate=example_vertical_coordinate,
        ini_time=datetime(2021, 6, 29),
        physics_source={"name": "GLORYS", "path": fname},
        bgc_source={"name": "CESM_REGRIDDED", "path": fname_bgc},
    )

    assert initial_conditions.bgc_source["climatology"] is False


def test_interpolation_from_climatology(initial_conditions_with_bgc_from_climatology):

    fname_bgc = download_test_data("CESM_regional_test_data_climatology.nc")
    ds = xr.open_dataset(fname_bgc)

    # check if interpolated value for Jan 15 is indeed January value from climatology
    bgc_data = CESMBGCDataset(
        filename=fname_bgc, start_time=datetime(2012, 1, 15), climatology=True
    )
    assert np.allclose(ds["ALK"].sel(month=1), bgc_data.ds["ALK"], equal_nan=True)

    # check if interpolated value for Jan 30 is indeed average of January and February value from climatology
    bgc_data = CESMBGCDataset(
        filename=fname_bgc, start_time=datetime(2012, 1, 30), climatology=True
    )
    assert np.allclose(
        0.5 * (ds["ALK"].sel(month=1) + ds["ALK"].sel(month=2)),
        bgc_data.ds["ALK"],
        equal_nan=True,
    )


def test_initial_conditions_data_consistency_plot_save(
    initial_conditions_with_bgc_from_climatology, tmp_path
):
    """
    Test that the data within the InitialConditions object remains consistent.
    Also test plot and save methods in the same test since we dask arrays are already computed.
    """
    initial_conditions_with_bgc_from_climatology.ds.load()

    # Define the expected data
    expected_temp = np.array(
        [
            [
                [
                    [16.84414, 16.905312, 16.967817],
                    [18.088203, 18.121834, 18.315424],
                    [18.431192, 18.496748, 18.718002],
                    [19.294329, 19.30358, 19.439777],
                ],
                [
                    [12.639833, 13.479691, 14.426711],
                    [15.712767, 15.920951, 16.10028],
                    [13.02848, 14.5227165, 15.05175],
                    [18.633307, 18.637077, 18.667465],
                ],
                [
                    [11.027701, 11.650267, 12.200586],
                    [12.302642, 12.646921, 13.150708],
                    [8.143677, 11.435992, 13.356925],
                    [8.710737, 11.25943, 13.111585],
                ],
                [
                    [10.233599, 10.546486, 10.671082],
                    [10.147331, 10.502733, 10.68275],
                    [10.458557, 11.209945, 11.377164],
                    [9.20282, 10.667074, 11.752404],
                ],
            ]
        ],
        dtype=np.float32,
    )
    expected_salt = np.array(
        [
            [
                [
                    [33.832672, 33.77759, 33.633846],
                    [32.50002, 32.48105, 32.154694],
                    [30.922323, 30.909824, 30.508572],
                    [28.337738, 28.335176, 28.067144],
                ],
                [
                    [34.8002, 34.691143, 34.382282],
                    [32.43797, 32.369576, 32.027843],
                    [34.885834, 34.82964, 34.775684],
                    [29.237692, 29.232145, 29.09444],
                ],
                [
                    [34.046825, 33.950684, 33.87148],
                    [33.892323, 33.84102, 33.69169],
                    [34.964134, 34.91892, 34.91941],
                    [34.975933, 34.48586, 32.729057],
                ],
                [
                    [35.21593, 35.209476, 35.20767],
                    [35.224304, 35.209522, 35.20715],
                    [35.299217, 35.31244, 35.31555],
                    [34.25124, 33.828175, 33.234303],
                ],
            ]
        ],
        dtype=np.float32,
    )

    expected_zeta = np.array(
        [
            [
                [-0.30468762, -0.29416865, -0.30391693, -0.32985148],
                [-0.34336275, -0.29455253, -0.3718359, -0.36176518],
                [-0.3699948, -0.34693155, -0.41338325, -0.40663475],
                [-0.5534979, -0.5270749, -0.45107934, -0.40699923],
            ]
        ],
        dtype=np.float32,
    )

    expected_u = np.array(
        [
            [
                [[-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0], [0.0, -0.0, -0.0]],
                [[0.0, -0.0, -0.0], [-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0]],
                [
                    [0.0, 0.0, -0.0],
                    [0.0, 0.0, -0.0],
                    [0.06979556, 0.06167743, -0.02247071],
                ],
                [
                    [0.04268532, 0.03889201, 0.03351666],
                    [0.04645353, 0.04914769, 0.03673013],
                    [0.0211786, 0.03679834, 0.0274788],
                ],
            ]
        ],
        dtype=np.float32,
    )

    expected_v = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, -0.0],
                    [-0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0],
                ],
                [
                    [-0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0],
                    [-0.03831354, -0.02400788, -0.03179555],
                    [-0.0, -0.0, -0.0],
                ],
                [
                    [-0.00951457, -0.00576979, -0.02147919],
                    [-0.0, -0.0, -0.0],
                    [0.01915873, 0.02625698, 0.01757628],
                    [-0.06720348, -0.08354441, -0.13835917],
                ],
            ]
        ],
        dtype=np.float32,
    )

    expected_ubar = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.04028399],
                [0.03866891, 0.04446249, 0.02812303],
            ]
        ],
        dtype=np.float32,
    )

    expected_vbar = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.03169237, 0.0],
                [-0.01189703, 0.0, 0.02102064, -0.09326097],
            ]
        ],
        dtype=np.float32,
    )

    expected_alk = np.array(
        [
            [
                [
                    [2341.926, 2340.8894, 2340.557],
                    [2317.8875, 2315.86, 2315.2148],
                    [2297.689, 2285.8933, 2284.404],
                    [2276.4216, 2258.4436, 2256.1062],
                ],
                [
                    [2330.5837, 2329.8225, 2329.5264],
                    [2317.878, 2316.6787, 2316.3088],
                    [2278.9314, 2269.464, 2268.904],
                    [2259.975, 2247.7456, 2246.0632],
                ],
                [
                    [2376.7534, 2373.4402, 2372.9192],
                    [2362.5308, 2360.5066, 2360.2224],
                    [2350.3384, 2344.3135, 2343.6768],
                    [2310.4275, 2287.6785, 2281.5872],
                ],
                [
                    [2384.8064, 2386.2126, 2386.632],
                    [2383.737, 2385.1553, 2385.6685],
                    [2380.2297, 2381.4849, 2381.8616],
                    [2350.0762, 2342.5403, 2339.2244],
                ],
            ]
        ],
        dtype=np.float32,
    )

    # Check the values in the dataset
    assert np.allclose(
        initial_conditions_with_bgc_from_climatology.ds["temp"].values, expected_temp
    )
    assert np.allclose(
        initial_conditions_with_bgc_from_climatology.ds["salt"].values, expected_salt
    )
    assert np.allclose(
        initial_conditions_with_bgc_from_climatology.ds["zeta"].values, expected_zeta
    )
    assert np.allclose(
        initial_conditions_with_bgc_from_climatology.ds["u"].values, expected_u
    )
    assert np.allclose(
        initial_conditions_with_bgc_from_climatology.ds["v"].values, expected_v
    )
    assert np.allclose(
        initial_conditions_with_bgc_from_climatology.ds["ubar"].values, expected_ubar
    )
    assert np.allclose(
        initial_conditions_with_bgc_from_climatology.ds["vbar"].values, expected_vbar
    )
    assert np.allclose(
        initial_conditions_with_bgc_from_climatology.ds["ALK"].values, expected_alk
    )

    initial_conditions_with_bgc_from_climatology.plot(varname="temp", s=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", eta=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", s=0, xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", eta=0, xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="zeta")
    initial_conditions_with_bgc_from_climatology.plot(varname="ALK", s=0, xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="ALK", eta=0, xi=0)

    filepath = tmp_path / "initial_conditions.nc"
    initial_conditions_with_bgc_from_climatology.save(filepath)
    assert filepath.exists()


def test_roundtrip_yaml(initial_conditions):
    """Test that creating an InitialConditions object, saving its parameters to yaml file, and re-opening yaml file creates the same object."""

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    try:
        initial_conditions.to_yaml(filepath)

        initial_conditions_from_file = InitialConditions.from_yaml(filepath)

        assert initial_conditions == initial_conditions_from_file

    finally:
        os.remove(filepath)


def test_from_yaml_missing_initial_conditions():
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
            ValueError,
            match="No InitialConditions configuration found in the YAML file.",
        ):
            InitialConditions.from_yaml(yaml_filepath)
    finally:
        os.remove(yaml_filepath)
