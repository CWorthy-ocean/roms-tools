import pytest
from datetime import datetime
from roms_tools import InitialConditions, Grid
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

    return grid


@pytest.fixture
def initial_conditions(example_grid):
    """
    Fixture for creating a dummy InitialConditions object.
    """

    fname = download_test_data("GLORYS_test_data.nc")

    return InitialConditions(
        grid=example_grid,
        ini_time=datetime(2021, 6, 29),
        source={"path": fname, "name": "GLORYS"},
    )


@pytest.fixture
def initial_conditions_with_bgc(example_grid):
    """
    Fixture for creating a dummy InitialConditions object.
    """

    fname = download_test_data("GLORYS_test_data.nc")
    fname_bgc = download_test_data("CESM_regional_test_data_one_time_slice.nc")

    return InitialConditions(
        grid=example_grid,
        ini_time=datetime(2021, 6, 29),
        source={"path": fname, "name": "GLORYS"},
        bgc_source={"path": fname_bgc, "name": "CESM_REGRIDDED"},
    )


@pytest.fixture
def initial_conditions_with_bgc_from_climatology(example_grid):
    """
    Fixture for creating a dummy InitialConditions object.
    """

    fname = download_test_data("GLORYS_test_data.nc")
    fname_bgc = download_test_data("CESM_regional_test_data_climatology.nc")

    return InitialConditions(
        grid=example_grid,
        ini_time=datetime(2021, 6, 29),
        source={"path": fname, "name": "GLORYS"},
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
    assert ic.source == {
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


# Test initialization with missing 'name' in source
def test_initial_conditions_missing_physics_name(example_grid):
    with pytest.raises(ValueError, match="`source` must include a 'name'."):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"path": "physics_data.nc"},
        )


# Test initialization with missing 'path' in source
def test_initial_conditions_missing_physics_path(example_grid):
    with pytest.raises(ValueError, match="`source` must include a 'path'."):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS"},
        )


# Test initialization with missing 'name' in bgc_source
def test_initial_conditions_missing_bgc_name(example_grid):

    fname = download_test_data("GLORYS_test_data.nc")
    with pytest.raises(
        ValueError, match="`bgc_source` must include a 'name' if it is provided."
    ):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            bgc_source={"path": "bgc_data.nc"},
        )


# Test initialization with missing 'path' in bgc_source
def test_initial_conditions_missing_bgc_path(example_grid):

    fname = download_test_data("GLORYS_test_data.nc")
    with pytest.raises(
        ValueError, match="`bgc_source` must include a 'path' if it is provided."
    ):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            bgc_source={"name": "CESM_REGRIDDED"},
        )


# Test default climatology value
def test_initial_conditions_default_climatology(example_grid):

    fname = download_test_data("GLORYS_test_data.nc")

    initial_conditions = InitialConditions(
        grid=example_grid,
        ini_time=datetime(2021, 6, 29),
        source={"name": "GLORYS", "path": fname},
    )

    assert initial_conditions.source["climatology"] is False
    assert initial_conditions.bgc_source is None


def test_initial_conditions_default_bgc_climatology(example_grid):

    fname = download_test_data("GLORYS_test_data.nc")
    fname_bgc = download_test_data("CESM_regional_test_data_one_time_slice.nc")

    initial_conditions = InitialConditions(
        grid=example_grid,
        ini_time=datetime(2021, 6, 29),
        source={"name": "GLORYS", "path": fname},
        bgc_source={"name": "CESM_REGRIDDED", "path": fname_bgc},
    )

    assert initial_conditions.bgc_source["climatology"] is True


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


def test_coordinates_existence_and_values(initial_conditions_with_bgc_from_climatology):
    """
    Test that the dataset contains the expected coordinates with the correct values.
    """

    # Expected coordinates and their values
    expected_coords = {
        "abs_time": np.array(["2021-06-29T00:00:00.000000000"], dtype="datetime64[ns]"),
        "ocean_time": np.array([6.7824e08], dtype=float),
    }
    # Check that the dataset contains exactly the expected coordinates and no others
    actual_coords = set(initial_conditions_with_bgc_from_climatology.ds.coords.keys())
    expected_coords_set = set(expected_coords.keys())

    assert actual_coords == expected_coords_set, (
        f"Unexpected coordinates found. Expected only {expected_coords_set}, "
        f"but found {actual_coords}."
    )

    # Check that the coordinate values match the expected values
    np.testing.assert_array_equal(
        initial_conditions_with_bgc_from_climatology.ds.coords["abs_time"].values,
        expected_coords["abs_time"],
    )
    np.testing.assert_allclose(
        initial_conditions_with_bgc_from_climatology.ds.coords["ocean_time"].values,
        expected_coords["ocean_time"],
        rtol=1e-9,
        atol=0,
    )


def test_initial_conditions_data_consistency_plot_save(
    initial_conditions_with_bgc_from_climatology, tmp_path
):
    """
    Test that the data within the InitialConditions object remains consistent.
    Also test plot and save methods in the same test since we dask arrays are already computed.
    """

    initial_conditions_with_bgc_from_climatology.plot(varname="temp", s=0)
    initial_conditions_with_bgc_from_climatology.plot(
        varname="temp", s=0, depth_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(
        varname="temp", eta=0, layer_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(
        varname="temp", xi=0, layer_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", eta=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", s=0, xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", eta=0, xi=0)
    initial_conditions_with_bgc_from_climatology.plot(
        varname="u", s=0, layer_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(varname="u", s=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="u", eta=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="u", xi=0)
    initial_conditions_with_bgc_from_climatology.plot(
        varname="v", s=0, layer_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(varname="v", s=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="v", eta=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="v", xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="zeta")
    initial_conditions_with_bgc_from_climatology.plot(varname="ubar")
    initial_conditions_with_bgc_from_climatology.plot(varname="vbar")
    initial_conditions_with_bgc_from_climatology.plot(varname="ALK", s=0, xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="ALK", eta=0, xi=0)

    # Define the expected data
    expected_temp = np.array(
        [
            [
                [
                    [16.888039, 18.113976, 18.491693, 19.302889],
                    [13.35342, 15.896477, 14.248364, 18.63616],
                    [11.584284, 12.586861, 10.864316, 10.860589],
                    [10.516198, 10.465848, 11.167091, 10.501563],
                ],
                [
                    [16.96757, 18.292757, 18.528805, 19.310738],
                    [13.915829, 16.067533, 14.980555, 18.656109],
                    [11.922902, 12.84863, 12.886733, 12.576546],
                    [10.625185, 10.623089, 11.301617, 11.2624445],
                ],
                [
                    [16.968897, 18.347046, 18.774857, 19.45075],
                    [14.607185, 16.14147, 15.065658, 18.65017],
                    [12.252905, 13.331364, 13.432277, 13.1259985],
                    [10.70972, 10.72327, 11.414437, 11.879851],
                ],
            ]
        ],
        dtype=np.float32,
    )

    expected_salt = np.array(
        [
            [
                [
                    [33.80441, 32.486362, 30.912357, 28.335825],
                    [34.713787, 32.379322, 34.84995, 29.233776],
                    [33.961357, 33.86234, 34.923454, 34.666214],
                    [35.20989, 35.210526, 35.310867, 33.894505],
                ],
                [
                    [33.700096, 32.28268, 30.881216, 28.322077],
                    [34.5851, 32.149498, 34.775352, 29.208817],
                    [33.90608, 33.76901, 34.91334, 33.391956],
                    [35.20767, 35.207634, 35.31487, 33.565704],
                ],
                [
                    [33.607388, 32.072247, 30.331127, 27.99018],
                    [34.310513, 31.941208, 34.77581, 29.013182],
                    [33.868027, 33.625626, 34.92061, 32.718384],
                    [35.20767, 35.206123, 35.316204, 33.09605],
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
                [
                    [-0.02990346, -0.00527473, -0.00209207],
                    [-0.00619788, -0.02969519, -0.08197243],
                    [0.00090594, 0.01207017, 0.06334759],
                    [0.04009043, 0.05006053, 0.03555659],
                ],
                [
                    [-0.03373717, -0.0064072, -0.00393128],
                    [-0.03140489, -0.04504679, -0.09550258],
                    [-0.00865677, -0.00062148, 0.0236859],
                    [0.0349412, 0.04345757, 0.03713445],
                ],
                [
                    [-0.0401831, -0.01946738, -0.02024308],
                    [-0.08688492, -0.07161549, -0.11909306],
                    [-0.03151456, -0.02571391, -0.02354126],
                    [0.03305725, 0.03291277, 0.02392998],
                ],
            ]
        ],
        dtype=np.float32,
    )

    expected_v = np.array(
        [
            [
                [
                    [0.04143963, 0.00235313, -0.04121511, -0.00271586],
                    [-0.01356939, -0.01282511, -0.02829472, -0.03477801],
                    [-0.00779556, -0.01769326, 0.0237356, -0.08123003],
                ],
                [
                    [0.05191367, 0.00191267, -0.03233896, -0.00375686],
                    [-0.00451732, -0.01308978, -0.01365391, -0.02148061],
                    [-0.00770537, -0.0187724, 0.03198036, -0.08685598],
                ],
                [
                    [0.01049447, -0.00655571, -0.040242, -0.00815171],
                    [-0.04645749, -0.016339, -0.04035791, -0.06388983],
                    [-0.02848312, -0.01897671, 0.00935264, -0.15538278],
                ],
            ]
        ],
        dtype=np.float32,
    )

    expected_ubar = np.array(
        [
            [
                [-0.03445563, -0.01013886, -0.00840288],
                [-0.04031063, -0.04803473, -0.09810777],
                [-0.01240468, -0.00379001, 0.02369577],
                [0.036187, 0.04258458, 0.03251844],
            ]
        ],
        dtype=np.float32,
    )

    expected_vbar = np.array(
        [
            [
                [0.03502939, -0.00063722, -0.03797533, -0.00476785],
                [-0.02095497, -0.01402165, -0.02719686, -0.03939667],
                [-0.01424322, -0.01845166, 0.02204982, -0.10577434],
            ]
        ],
        dtype=np.float32,
    )

    expected_alk = np.array(
        [
            [
                [
                    [2340.938, 2315.954, 2286.2207, 2258.9512],
                    [2329.8655, 2316.7327, 2269.6416, 2248.1035],
                    [2373.7942, 2360.7102, 2344.5688, 2289.7437],
                    [2386.0618, 2385.0015, 2381.408, 2343.4858],
                ],
                [
                    [2340.7126, 2315.517, 2285.092, 2257.186],
                    [2329.665, 2316.4822, 2269.1628, 2246.8403],
                    [2373.1604, 2360.354, 2343.725, 2282.6587],
                    [2386.4377, 2385.4307, 2381.7134, 2340.097],
                ],
                [
                    [2340.4998, 2315.104, 2284.149, 2255.7056],
                    [2329.4756, 2316.2454, 2268.808, 2245.7747],
                    [2372.83, 2360.174, 2343.6592, 2281.1965],
                    [2386.7036, 2385.756, 2381.9155, 2338.9065],
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
