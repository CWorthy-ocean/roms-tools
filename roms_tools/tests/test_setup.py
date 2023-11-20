import pytest
import numpy as np
import numpy.testing as npt

from roms_tools import Grid


class TestCreateGrid:
    def test_simple_regression(self):
        grid = Grid(nx=1, ny=1, size_x=100, size_y=100, center_lon=-20, center_lat=0)

        expected_lat = np.array(
            [
                [1.79855429e00, 1.79855429e00, 1.79855429e00],
                [1.72818690e-14, 1.70960078e-14, 1.70960078e-14],
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

        npt.assert_allclose(grid.ds["lat_rho"], expected_lat)
        npt.assert_allclose(grid.ds["lon_rho"], expected_lon)

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
