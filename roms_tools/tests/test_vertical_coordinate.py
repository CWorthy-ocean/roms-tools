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
                [17.235603, 9.938617, 3.2495391],
                [17.235603, 9.938617, 3.2495391],
                [24.206884, 13.7451725, 4.4592304],
                [24.206884, 13.7451725, 4.4592304],
            ],
            [
                [17.235603, 9.938617, 3.2495391],
                [17.235603, 9.938617, 3.2495391],
                [24.206884, 13.7451725, 4.4592304],
                [24.206884, 13.7451725, 4.4592304],
            ],
            [
                [25.667452, 14.528268, 4.7055993],
                [25.667452, 14.528268, 4.7055993],
                [35.966946, 19.916193, 6.377065],
                [35.966946, 19.916193, 6.377065],
            ],
            [
                [25.667452, 14.528268, 4.7055993],
                [25.667452, 14.528268, 4.7055993],
                [35.966946, 19.916193, 6.377065],
                [35.966946, 19.916193, 6.377065],
            ],
        ],
        dtype=np.float32,
    )
    expected_layer_depth_u = np.array(
        [
            [
                [17.235603, 9.938617, 3.2495391],
                [20.721243, 11.841895, 3.854385],
                [24.206884, 13.7451725, 4.4592304],
            ],
            [
                [17.235603, 9.938617, 3.2495391],
                [20.721243, 11.841895, 3.854385],
                [24.206884, 13.7451725, 4.4592304],
            ],
            [
                [25.667452, 14.528268, 4.7055993],
                [30.817198, 17.22223, 5.5413322],
                [35.966946, 19.916193, 6.377065],
            ],
            [
                [25.667452, 14.528268, 4.7055993],
                [30.817198, 17.22223, 5.5413322],
                [35.966946, 19.916193, 6.377065],
            ],
        ],
        dtype=np.float32,
    )
    expected_layer_depth_v = np.array(
        [
            [
                [17.235603, 9.938617, 3.2495391],
                [17.235603, 9.938617, 3.2495391],
                [24.206884, 13.7451725, 4.4592304],
                [24.206884, 13.7451725, 4.4592304],
            ],
            [
                [21.451529, 12.233442, 3.977569],
                [21.451529, 12.233442, 3.977569],
                [30.086914, 16.830683, 5.418148],
                [30.086914, 16.830683, 5.418148],
            ],
            [
                [25.667452, 14.528268, 4.7055993],
                [25.667452, 14.528268, 4.7055993],
                [35.966946, 19.916193, 6.377065],
                [35.966946, 19.916193, 6.377065],
            ],
        ],
        dtype=np.float32,
    )
    expected_interface_depth_rho = np.array(
        [
            [
                [21.013529, 13.487298, 6.5489607, -0.0],
                [21.013529, 13.487298, 6.5489607, -0.0],
                [29.688076, 18.78298, 9.014939, -0.0],
                [29.688076, 18.78298, 9.014939, -0.0],
            ],
            [
                [21.013529, 13.487298, 6.5489607, -0.0],
                [21.013529, 13.487298, 6.5489607, -0.0],
                [29.688076, 18.78298, 9.014939, -0.0],
                [29.688076, 18.78298, 9.014939, -0.0],
            ],
            [
                [31.51735, 19.881702, 9.519225, -0.0],
                [31.51735, 19.881702, 9.519225, -0.0],
                [44.52708, 27.529188, 12.960222, -0.0],
                [44.52708, 27.529188, 12.960222, -0.0],
            ],
            [
                [31.51735, 19.881702, 9.519225, -0.0],
                [31.51735, 19.881702, 9.519225, -0.0],
                [44.52708, 27.529188, 12.960222, -0.0],
                [44.52708, 27.529188, 12.960222, -0.0],
            ],
        ],
        dtype=np.float32,
    )
    expected_interface_depth_u = np.array(
        [
            [
                [21.013529, 13.487298, 6.5489607, -0.0],
                [25.350803, 16.13514, 7.78195, -0.0],
                [29.688076, 18.78298, 9.014939, -0.0],
            ],
            [
                [21.013529, 13.487298, 6.5489607, -0.0],
                [25.350803, 16.13514, 7.78195, -0.0],
                [29.688076, 18.78298, 9.014939, -0.0],
            ],
            [
                [31.51735, 19.881702, 9.519225, -0.0],
                [38.022217, 23.705446, 11.239724, -0.0],
                [44.52708, 27.529188, 12.960222, -0.0],
            ],
            [
                [31.51735, 19.881702, 9.519225, -0.0],
                [38.022217, 23.705446, 11.239724, -0.0],
                [44.52708, 27.529188, 12.960222, -0.0],
            ],
        ],
        dtype=np.float32,
    )
    expected_interface_depth_v = np.array(
        [
            [
                [21.013529, 13.487298, 6.5489607, -0.0],
                [21.013529, 13.487298, 6.5489607, -0.0],
                [29.688076, 18.78298, 9.014939, -0.0],
                [29.688076, 18.78298, 9.014939, -0.0],
            ],
            [
                [26.26544, 16.684502, 8.034093, -0.0],
                [26.26544, 16.684502, 8.034093, -0.0],
                [37.10758, 23.156084, 10.987581, -0.0],
                [37.10758, 23.156084, 10.987581, -0.0],
            ],
            [
                [31.51735, 19.881702, 9.519225, -0.0],
                [31.51735, 19.881702, 9.519225, -0.0],
                [44.52708, 27.529188, 12.960222, -0.0],
                [44.52708, 27.529188, 12.960222, -0.0],
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
