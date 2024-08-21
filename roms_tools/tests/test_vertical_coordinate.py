import pytest
import numpy as np
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


def test_vertical_coordinate_data_consistency():
    """
    Test that the data within the VerticalCoordinate object remains consistent.
    """

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

    # Define the expected data
    expected_sc_r = np.array([-0.8333333, -0.5, -0.16666667], dtype=np.float32)
    expected_Cs_r = np.array([-0.6641397, -0.15129805, -0.01156188], dtype=np.float32)
    expected_layer_depth_rho = np.array(
        [
            [
                [17.235603, 17.235603, 24.206884, 24.206884],
                [17.235603, 17.235603, 24.206884, 24.206884],
                [25.667452, 25.667452, 35.966946, 35.966946],
                [25.667452, 25.667452, 35.966946, 35.966946],
            ],
            [
                [9.938617, 9.938617, 13.7451725, 13.7451725],
                [9.938617, 9.938617, 13.7451725, 13.7451725],
                [14.528268, 14.528268, 19.916193, 19.916193],
                [14.528268, 14.528268, 19.916193, 19.916193],
            ],
            [
                [3.2495391, 3.2495391, 4.4592304, 4.4592304],
                [3.2495391, 3.2495391, 4.4592304, 4.4592304],
                [4.7055993, 4.7055993, 6.377065, 6.377065],
                [4.7055993, 4.7055993, 6.377065, 6.377065],
            ],
        ],
        dtype=np.float32,
    )
    expected_layer_depth_u = np.array(
        [
            [
                [17.235603, 20.721243, 24.206884],
                [17.235603, 20.721243, 24.206884],
                [25.667452, 30.817198, 35.966946],
                [25.667452, 30.817198, 35.966946],
            ],
            [
                [9.938617, 11.841895, 13.7451725],
                [9.938617, 11.841895, 13.7451725],
                [14.528268, 17.22223, 19.916193],
                [14.528268, 17.22223, 19.916193],
            ],
            [
                [3.2495391, 3.854385, 4.4592304],
                [3.2495391, 3.854385, 4.4592304],
                [4.7055993, 5.5413322, 6.377065],
                [4.7055993, 5.5413322, 6.377065],
            ],
        ],
        dtype=np.float32,
    )
    expected_layer_depth_v = np.array(
        [
            [
                [17.235603, 17.235603, 24.206884, 24.206884],
                [21.451529, 21.451529, 30.086914, 30.086914],
                [25.667452, 25.667452, 35.966946, 35.966946],
            ],
            [
                [9.938617, 9.938617, 13.7451725, 13.7451725],
                [12.233442, 12.233442, 16.830683, 16.830683],
                [14.528268, 14.528268, 19.916193, 19.916193],
            ],
            [
                [3.2495391, 3.2495391, 4.4592304, 4.4592304],
                [3.977569, 3.977569, 5.418148, 5.418148],
                [4.7055993, 4.7055993, 6.377065, 6.377065],
            ],
        ],
        dtype=np.float32,
    )
    expected_interface_depth_rho = np.array(
        [
            [
                [21.013529, 21.013529, 29.688076, 29.688076],
                [21.013529, 21.013529, 29.688076, 29.688076],
                [31.51735, 31.51735, 44.52708, 44.52708],
                [31.51735, 31.51735, 44.52708, 44.52708],
            ],
            [
                [13.487298, 13.487298, 18.78298, 18.78298],
                [13.487298, 13.487298, 18.78298, 18.78298],
                [19.881702, 19.881702, 27.529188, 27.529188],
                [19.881702, 19.881702, 27.529188, 27.529188],
            ],
            [
                [6.5489607, 6.5489607, 9.014939, 9.014939],
                [6.5489607, 6.5489607, 9.014939, 9.014939],
                [9.519225, 9.519225, 12.960222, 12.960222],
                [9.519225, 9.519225, 12.960222, 12.960222],
            ],
            [
                [-0.0, -0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0, -0.0],
            ],
        ],
        dtype=np.float32,
    )
    expected_interface_depth_u = np.array(
        [
            [
                [21.013529, 25.350803, 29.688076],
                [21.013529, 25.350803, 29.688076],
                [31.51735, 38.022217, 44.52708],
                [31.51735, 38.022217, 44.52708],
            ],
            [
                [13.487298, 16.13514, 18.78298],
                [13.487298, 16.13514, 18.78298],
                [19.881702, 23.705446, 27.529188],
                [19.881702, 23.705446, 27.529188],
            ],
            [
                [6.5489607, 7.78195, 9.014939],
                [6.5489607, 7.78195, 9.014939],
                [9.519225, 11.239724, 12.960222],
                [9.519225, 11.239724, 12.960222],
            ],
            [
                [-0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0],
            ],
        ],
        dtype=np.float32,
    )
    expected_interface_depth_v = np.array(
        [
            [
                [21.013529, 21.013529, 29.688076, 29.688076],
                [26.26544, 26.26544, 37.10758, 37.10758],
                [31.51735, 31.51735, 44.52708, 44.52708],
            ],
            [
                [13.487298, 13.487298, 18.78298, 18.78298],
                [16.684502, 16.684502, 23.156084, 23.156084],
                [19.881702, 19.881702, 27.529188, 27.529188],
            ],
            [
                [6.5489607, 6.5489607, 9.014939, 9.014939],
                [8.034093, 8.034093, 10.987581, 10.987581],
                [9.519225, 9.519225, 12.960222, 12.960222],
            ],
            [
                [-0.0, -0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0, -0.0],
            ],
        ],
        dtype=np.float32,
    )

    # Check the values in the dataset
    assert np.allclose(grid.ds["sc_r"].values, expected_sc_r)
    assert np.allclose(grid.ds["Cs_r"].values, expected_Cs_r)
    assert np.allclose(grid.ds["layer_depth_rho"].values, expected_layer_depth_rho)
    assert np.allclose(grid.ds["layer_depth_u"].values, expected_layer_depth_u)
    assert np.allclose(grid.ds["layer_depth_v"].values, expected_layer_depth_v)
    assert np.allclose(
        grid.ds["interface_depth_rho"].values,
        expected_interface_depth_rho,
    )
    assert np.allclose(grid.ds["interface_depth_u"].values, expected_interface_depth_u)
    assert np.allclose(grid.ds["interface_depth_v"].values, expected_interface_depth_v)


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
