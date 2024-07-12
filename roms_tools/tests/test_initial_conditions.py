import pytest
from datetime import datetime
from roms_tools import InitialConditions, Grid
import xarray as xr
from roms_tools.setup.datasets import download_test_data


@pytest.fixture
def example_grid():
    """
    Fixture for creating a dummy Grid object.
    """
    grid = Grid(
        nx=5, ny=5, size_x=500, size_y=1000, center_lon=0, center_lat=55, rot=10
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
        N=5,
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
    assert initial_conditions.N == 5
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


def test_plot_method(initial_conditions):
    """
    Test the plot method of the InitialConditions object.
    """
    initial_conditions.plot(varname="temp", s_rho=0)
    initial_conditions.plot(varname="temp", eta=0)
    initial_conditions.plot(varname="temp", xi=0)
    initial_conditions.plot(varname="temp", s_rho=0, xi=0)
    initial_conditions.plot(varname="temp", eta=0, xi=0)
    initial_conditions.plot(varname="zeta")


def test_save_method(initial_conditions, tmp_path):
    """
    Test the save method of the InitialConditions object.
    """
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
