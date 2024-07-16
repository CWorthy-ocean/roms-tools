import pytest
from datetime import datetime
from roms_tools import InitialConditions, Grid
import xarray as xr
import numpy as np
from roms_tools.setup.datasets import download_test_data


@pytest.fixture
def example_grid():
    """
    Fixture for creating a dummy Grid object.
    """
    grid = Grid(
        nx=2, ny=2, size_x=500, size_y=1000, center_lon=0, center_lat=55, rot=10
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
        filename=fname,
        N=3,
        theta_s=5.0,
        theta_b=2.0,
        hc=250.0,
    )


def test_initial_conditions_creation(initial_conditions):
    """
    Test the creation of the InitialConditions object.
    """
    assert initial_conditions.ini_time == datetime(2021, 6, 29)
    assert initial_conditions.filename == download_test_data("GLORYS_test_data.nc")
    assert initial_conditions.N == 3
    assert initial_conditions.theta_s == 5.0
    assert initial_conditions.theta_b == 2.0
    assert initial_conditions.hc == 250.0
    assert initial_conditions.source == "glorys"


def test_initial_conditions_ds_attribute(initial_conditions):
    """
    Test the ds attribute of the InitialConditions object.
    """
    assert isinstance(initial_conditions.ds, xr.Dataset)
    assert "temp" in initial_conditions.ds
    assert "salt" in initial_conditions.ds
    assert "u" in initial_conditions.ds
    assert "v" in initial_conditions.ds
    assert "zeta" in initial_conditions.ds


def test_initial_conditions_data_consistency_plot_save(initial_conditions, tmp_path):
    """
    Test that the data within the InitialConditions object remains consistent.
    Also test plot and save methods in the same test since we dask arrays are already computed.
    """
    initial_conditions.ds.load()

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

    # Check the values in the dataset
    assert np.allclose(initial_conditions.ds["temp"].values, expected_temp)
    assert np.allclose(initial_conditions.ds["salt"].values, expected_salt)
    assert np.allclose(initial_conditions.ds["zeta"].values, expected_zeta)
    assert np.allclose(initial_conditions.ds["u"].values, expected_u)
    assert np.allclose(initial_conditions.ds["v"].values, expected_v)
    assert np.allclose(initial_conditions.ds["ubar"].values, expected_ubar)
    assert np.allclose(initial_conditions.ds["vbar"].values, expected_vbar)

    initial_conditions.plot(varname="temp", s_rho=0)
    initial_conditions.plot(varname="temp", eta=0)
    initial_conditions.plot(varname="temp", xi=0)
    initial_conditions.plot(varname="temp", s_rho=0, xi=0)
    initial_conditions.plot(varname="temp", eta=0, xi=0)
    initial_conditions.plot(varname="zeta")

    filepath = tmp_path / "initial_conditions.nc"
    initial_conditions.save(filepath)
    assert filepath.exists()


def test_invalid_theta_s_value(example_grid):
    """
    Test the validation of the theta_s value.
    """
    with pytest.raises(ValueError):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            filename=download_test_data("GLORYS_test_data.nc"),
            N=5,
            theta_s=11.0,  # Invalid value, should be 0 < theta_s <= 10
            theta_b=2.0,
            hc=250.0,
        )


def test_invalid_theta_b_value(example_grid):
    """
    Test the validation of the theta_b value.
    """
    with pytest.raises(ValueError):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            filename=download_test_data("GLORYS_test_data.nc"),
            N=5,
            theta_s=5.0,
            theta_b=5.0,  # Invalid value, should be 0 < theta_b <= 4
            hc=250.0,
        )
