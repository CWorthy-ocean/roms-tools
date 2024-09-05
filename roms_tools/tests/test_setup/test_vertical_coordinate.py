import pytest
from roms_tools import Grid


def test_invalid_theta_s_value():
    """
    Test the validation of the theta_s value.
    """
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
            theta_s=11.0,  # Invalid value, should be 0 < theta_s <= 10
            theta_b=2.0,
            hc=250.0,
        )


def test_invalid_theta_b_value():
    """
    Test the validation of the theta_b value.
    """
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
            theta_s=5.0,
            theta_b=5.0,  # Invalid value, should be 0 < theta_b <= 4
            hc=250.0,
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


def test_plot():
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
    grid.plot_vertical_coordinate("layer_depth_u", s=0)
    grid.plot_vertical_coordinate("layer_depth_rho", s=-1)
    grid.plot_vertical_coordinate("interface_depth_v", s=-1)
    grid.plot_vertical_coordinate("layer_depth_rho", eta=0)
    grid.plot_vertical_coordinate("layer_depth_u", eta=0)
    grid.plot_vertical_coordinate("layer_depth_v", eta=0)
    grid.plot_vertical_coordinate("interface_depth_rho", eta=0)
    grid.plot_vertical_coordinate("interface_depth_u", eta=0)
    grid.plot_vertical_coordinate("interface_depth_v", eta=0)
    grid.plot_vertical_coordinate("layer_depth_rho", xi=0)
    grid.plot_vertical_coordinate("layer_depth_u", xi=0)
    grid.plot_vertical_coordinate("layer_depth_v", xi=0)
    grid.plot_vertical_coordinate("interface_depth_rho", xi=0)
    grid.plot_vertical_coordinate("interface_depth_u", xi=0)
    grid.plot_vertical_coordinate("interface_depth_v", xi=0)
