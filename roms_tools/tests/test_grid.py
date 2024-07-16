import pytest
import numpy as np
import numpy.testing as npt
from roms_tools import Grid
import os
import tempfile


class TestCreateGrid:
    def test_simple_regression(self):
        grid = Grid(
            nx=1, ny=1, size_x=100, size_y=100, center_lon=-20, center_lat=0, rot=0
        )

        expected_lat = np.array(
            [
                [-8.99249453e-01, -8.99249453e-01, -8.99249453e-01],
                [0.0, 0.0, 0.0],
                [8.99249453e-01, 8.99249453e-01, 8.99249453e-01],
            ]
        )
        expected_lon = np.array(
            [
                [339.10072286, 340.0, 340.89927714],
                [339.10072286, 340.0, 340.89927714],
                [339.10072286, 340.0, 340.89927714],
            ]
        )

        # TODO: adapt tolerances according to order of magnitude of respective fields
        npt.assert_allclose(grid.ds["lat_rho"], expected_lat, atol=1e-8)
        npt.assert_allclose(grid.ds["lon_rho"], expected_lon, atol=1e-8)

    def test_raise_if_domain_too_large(self):
        with pytest.raises(ValueError, match="Domain size has to be smaller"):
            Grid(nx=3, ny=3, size_x=30000, size_y=30000, center_lon=0, center_lat=51.5)

        # test grid with reasonable domain size
        grid = Grid(
            nx=3,
            ny=3,
            size_x=1800,
            size_y=2400,
            center_lon=-21,
            center_lat=61,
            rot=20,
        )
        assert isinstance(grid, Grid)

    def test_grid_straddle_crosses_meridian(self):
        grid = Grid(
            nx=3,
            ny=3,
            size_x=100,
            size_y=100,
            center_lon=0,
            center_lat=61,
            rot=20,
        )
        assert grid.straddle

        grid = Grid(
            nx=3,
            ny=3,
            size_x=100,
            size_y=100,
            center_lon=180,
            center_lat=61,
            rot=20,
        )
        assert not grid.straddle


class TestGridFromFile:
    def test_roundtrip(self):
        """Test that creating a grid, saving it to file, and re-opening it is the same as just creating it."""

        # Initialize a Grid object using the initializer
        grid_init = Grid(
            nx=10,
            ny=15,
            size_x=100.0,
            size_y=150.0,
            center_lon=0.0,
            center_lat=0.0,
            rot=0.0,
            topography_source="etopo5",
            smooth_factor=2,
            hmin=5.0,
            rmax=0.2,
        )

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            filepath = tmpfile.name

        try:
            # Save the grid to a file
            grid_init.save(filepath)

            # Load the grid from the file
            grid_from_file = Grid.from_file(filepath)

            # Assert that the initial grid and the loaded grid are equivalent (including the 'ds' attribute)
            assert grid_init == grid_from_file

        finally:
            os.remove(filepath)
