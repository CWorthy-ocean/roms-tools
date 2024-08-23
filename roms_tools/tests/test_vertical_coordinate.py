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
                [10.912094, 10.912094, 15.100041, 15.100041],
                [10.912094, 10.912094, 15.100041, 15.100041],
                [16.006283, 16.006283, 22.118557, 22.118557],
                [16.006283, 16.006283, 22.118557, 22.118557],
            ],
            [
                [6.382904, 6.382904, 8.749026, 8.749026],
                [6.382904, 6.382904, 8.749026, 8.749026],
                [9.255217, 9.255217, 12.616944, 12.616944],
                [9.255217, 9.255217, 12.616944, 12.616944],
            ],
            [
                [2.1017897, 2.1017897, 2.8674364, 2.8674364],
                [2.1017897, 2.1017897, 2.8674364, 2.8674364],
                [3.030261, 3.030261, 4.1027746, 4.1027746],
                [3.030261, 3.030261, 4.1027746, 4.1027746],
            ],
        ],
        dtype=np.float32,
    )

    expected_layer_depth_u = np.array(
        [
            [
                [10.912094, 13.006067, 15.100041],
                [10.912094, 13.006067, 15.100041],
                [16.006283, 19.06242, 22.118557],
                [16.006283, 19.06242, 22.118557],
            ],
            [
                [6.382904, 7.5659647, 8.749026],
                [6.382904, 7.5659647, 8.749026],
                [9.255217, 10.936081, 12.616944],
                [9.255217, 10.936081, 12.616944],
            ],
            [
                [2.1017897, 2.484613, 2.8674364],
                [2.1017897, 2.484613, 2.8674364],
                [3.030261, 3.5665178, 4.1027746],
                [3.030261, 3.5665178, 4.1027746],
            ],
        ],
        dtype=np.float32,
    )

    expected_layer_depth_v = np.array(
        [
            [
                [10.912094, 10.912094, 15.100041, 15.100041],
                [13.459188, 13.459188, 18.609299, 18.609299],
                [16.006283, 16.006283, 22.118557, 22.118557],
            ],
            [
                [6.382904, 6.382904, 8.749026, 8.749026],
                [7.8190603, 7.8190603, 10.682985, 10.682985],
                [9.255217, 9.255217, 12.616944, 12.616944],
            ],
            [
                [2.1017897, 2.1017897, 2.8674364, 2.8674364],
                [2.5660253, 2.5660253, 3.4851055, 3.4851055],
                [3.030261, 3.030261, 4.1027746, 4.1027746],
            ],
        ],
        dtype=np.float32,
    )

    expected_interface_depth_rho = np.array(
        [
            [
                [13.229508, 13.229508, 18.375496, 18.375496],
                [13.229508, 13.229508, 18.375496, 18.375496],
                [19.493834, 19.493834, 27.079603, 27.079603],
                [19.493834, 19.493834, 27.079603, 27.079603],
            ],
            [
                [8.606768, 8.606768, 11.847459, 11.847459],
                [8.606768, 8.606768, 11.847459, 11.847459],
                [12.544369, 12.544369, 17.205624, 17.205624],
                [12.544369, 12.544369, 17.205624, 17.205624],
            ],
            [
                [4.223935, 4.223935, 5.7733917, 5.7733917],
                [4.223935, 4.223935, 5.7733917, 5.7733917],
                [6.103692, 6.103692, 8.286574, 8.286574],
                [6.103692, 6.103692, 8.286574, 8.286574],
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
                [13.229508, 15.802503, 18.375496],
                [13.229508, 15.802503, 18.375496],
                [19.493834, 23.286718, 27.079603],
                [19.493834, 23.286718, 27.079603],
            ],
            [
                [8.606768, 10.227114, 11.847459],
                [8.606768, 10.227114, 11.847459],
                [12.544369, 14.874996, 17.205624],
                [12.544369, 14.874996, 17.205624],
            ],
            [
                [4.223935, 4.9986634, 5.7733917],
                [4.223935, 4.9986634, 5.7733917],
                [6.103692, 7.195133, 8.286574],
                [6.103692, 7.195133, 8.286574],
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
                [13.229508, 13.229508, 18.375496, 18.375496],
                [16.36167, 16.36167, 22.727549, 22.727549],
                [19.493834, 19.493834, 27.079603, 27.079603],
            ],
            [
                [8.606768, 8.606768, 11.847459, 11.847459],
                [10.575568, 10.575568, 14.526541, 14.526541],
                [12.544369, 12.544369, 17.205624, 17.205624],
            ],
            [
                [4.223935, 4.223935, 5.7733917, 5.7733917],
                [5.1638136, 5.1638136, 7.029983, 7.029983],
                [6.103692, 6.103692, 8.286574, 8.286574],
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
