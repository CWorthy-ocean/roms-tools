import pytest

from roms_tools.setup import Grid


class TestCreateGrid:
    def test_simple(self):
        ...

    def test_raise_if_crossing_dateline(self):
        with pytest.raises(ValueError, match="cannot cross Greenwich Meridian"):
            # test grid centered over London
            Grid(nx=3, ny=3, size_x=100, size_y=100, center_lon=0, center_lat=51.5)

        # test Iceland grid which is rotated specifically to avoid Greenwich Meridian
        grid = Grid(
            nx=3, ny=3, size_x=1800, size_y=2400, center_lon=-21, center_lat=61, rot=20
        )
        assert isinstance(grid, Grid)


class TestGridFromFile:
    def test_roundtrip(self):
        """Test that creating a grid, saving it to file, and re-opening it is the same as just creating it."""
        ...
