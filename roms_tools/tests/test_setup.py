import pytest
import numpy as np
import numpy.testing as npt
from roms_tools import Grid


class TestCreateGrid:
    def test_simple_regression(self):
        grid = Grid(nx=1, ny=1, size_x=100, size_y=100, center_lon=-20, center_lat=0, rot=0)

        expected_lat = np.array(
            [
                [1.79855429e00, 1.79855429e00, 1.79855429e00],
                [0.0, 0.0, 0.0],
                [-1.79855429e00, -1.79855429e00, -1.79855429e00],
            ]
        )
        expected_lon = np.array(
            [
                [339.10072286, 340.0, 340.89927714],
                [339.10072286, 340.0, 340.89927714],
                [339.10072286, 340.0, 340.89927714],
            ]
        )
        expected_angle = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        expected_f0 = np.array(
            [
                [4.56484162e-06,  4.56484162e-06,  4.56484162e-06],
                [0.0, 0.0, 0.0],
                [-4.56484162e-06, -4.56484162e-06, -4.56484162e-06],
            ]
        )
        expected_pm = np.array(
            [
                [5.0e-06,  5.0e-06, 5.0e-06],
                [5.0e-06,  5.0e-06, 5.0e-06],
                [5.0e-06,  5.0e-06, 5.0e-06],
            ]
        )
        expected_pn = np.array(
            [
                [1.0004929e-05, 1.0004929e-05, 1.0004929e-05],
                [1.0e-05,  1.0e-05, 1.0e-05],
                [1.0004929e-05, 1.0004929e-05, 1.0004929e-05],
            ]
        )

        npt.assert_allclose(grid.ds["lat_rho"], expected_lat, atol=1e-8)
        npt.assert_allclose(grid.ds["lon_rho"], expected_lon, atol=1e-8)
        npt.assert_allclose(grid.ds["angle"], expected_angle, atol=1e-8)
        npt.assert_allclose(grid.ds["f0"], expected_f0, atol=1e-8)
        npt.assert_allclose(grid.ds["pm"], expected_pm, atol=1e-8)
        npt.assert_allclose(grid.ds["pn"], expected_pn, atol=1e-8)

    def test_raise_if_crossing_dateline(self):
        with pytest.raises(ValueError, match="cannot cross Greenwich Meridian"):
            # test grid centered over London
            Grid(nx=3, ny=3, size_x=100, size_y=100, center_lon=0, center_lat=51.5)

        # test Iceland grid which is rotated specifically to avoid Greenwich Meridian
        grid = Grid(
            nx=100,
            ny=100,
            size_x=1800,
            size_y=2400,
            center_lon=-21,
            center_lat=61,
            rot=20,
        )
        assert isinstance(grid, Grid)


class TestGridFromFile:
    def test_equal_to_from_init(self):
        ...

    def test_roundtrip(self):
        """Test that creating a grid, saving it to file, and re-opening it is the same as just creating it."""
        ...
