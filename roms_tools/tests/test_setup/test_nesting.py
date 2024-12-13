import pytest
import xarray as xr
import numpy as np
import logging
from roms_tools import Grid, Nesting
from roms_tools.setup.utils import get_boundary_coords
from roms_tools.setup.nesting import interpolate_indices


@pytest.fixture()
def parent_grid():
    return Grid(
        nx=5, ny=7, center_lon=-23, center_lat=61, rot=20, size_x=1800, size_y=2400
    )


@pytest.fixture()
def child_grid():
    return Grid(
        nx=10, ny=10, center_lon=-23, center_lat=61, rot=-20, size_x=500, size_y=500
    )


@pytest.fixture()
def parent_grid_that_straddles():
    return Grid(
        nx=5, ny=7, center_lon=10, center_lat=61, rot=20, size_x=1800, size_y=2400
    )


@pytest.fixture()
def child_grid_that_straddles():
    return Grid(
        nx=10, ny=10, center_lon=10, center_lat=61, rot=-20, size_x=500, size_y=500
    )


class TestMapChildBoundaries:
    @pytest.mark.parametrize(
        "grid",
        [
            "parent_grid",
            "parent_grid_that_straddles",
        ],
    )
    def test_correct_indices_of_same_grid(self, grid, caplog, request):
        """Test that the boundaries of the same grid are mapped correctly."""

        grid = request.getfixturevalue(grid)

        bdry_coords_dict = get_boundary_coords()
        location = "rho"
        for direction in ["south", "east", "north", "west"]:
            bdry_coords = bdry_coords_dict[location][direction]
            lon = grid.ds[f"lon_{location}"].isel(**bdry_coords)
            lat = grid.ds[f"lat_{location}"].isel(**bdry_coords)
            mask = grid.ds[f"mask_{location}"].isel(**bdry_coords)

            with caplog.at_level(logging.WARNING):
                i_eta, i_xi = interpolate_indices(grid.ds, lon, lat, mask)

            # Verify the warning message in the log
            assert (
                "Some boundary points of the child grid are very close to the boundary of the parent grid."
                in caplog.text
            )

            if direction == "south":
                expected_i_eta = -0.5 * xr.ones_like(grid.ds.xi_rho)
                expected_i_xi = np.arange(-0.5, grid.ds.xi_rho[-1] + 0.5)
            elif direction == "east":
                expected_i_eta = np.arange(-0.5, grid.ds.eta_rho[-1] + 0.5)
                expected_i_xi = (grid.ds.xi_rho[-1] - 0.5) * xr.ones_like(
                    grid.ds.eta_rho
                )
            elif direction == "north":
                expected_i_eta = (grid.ds.eta_rho[-1] - 0.5) * xr.ones_like(
                    grid.ds.xi_rho
                )
                expected_i_xi = np.arange(-0.5, grid.ds.xi_rho[-1] + 0.5)
            elif direction == "west":
                expected_i_eta = np.arange(-0.5, grid.ds.eta_rho[-1] + 0.5)
                expected_i_xi = -0.5 * xr.ones_like(grid.ds.eta_rho)

            np.testing.assert_allclose(i_eta.values, expected_i_eta)
            np.testing.assert_allclose(i_xi.values, expected_i_xi)

    @pytest.mark.parametrize(
        "parent_grid_fixture, child_grid_fixture",
        [
            ("parent_grid", "child_grid"),
            ("parent_grid_that_straddles", "child_grid_that_straddles"),
        ],
    )
    def test_indices_are_within_range_of_parent_grid(
        self, parent_grid_fixture, child_grid_fixture, request
    ):

        parent_grid = request.getfixturevalue(parent_grid_fixture)
        child_grid = request.getfixturevalue(child_grid_fixture)
        bdry_coords_dict = get_boundary_coords()
        for location in ["rho", "u", "v"]:
            for direction in ["south", "east", "north", "west"]:
                bdry_coords = bdry_coords_dict[location][direction]
                lon = child_grid.ds[f"lon_{location}"].isel(**bdry_coords)
                lat = child_grid.ds[f"lat_{location}"].isel(**bdry_coords)
                mask = child_grid.ds[f"mask_{location}"].isel(**bdry_coords)

                i_eta, i_xi = interpolate_indices(parent_grid.ds, lon, lat, mask)

                expected_i_eta_min = -0.5
                expected_i_eta_max = parent_grid.ds.eta_rho[-1] - 0.5
                expected_i_xi_min = -0.5
                expected_i_xi_max = parent_grid.ds.xi_rho[-1] - 0.5

                assert (i_eta >= expected_i_eta_min).all()
                assert (i_eta <= expected_i_eta_max).all()
                assert (i_xi >= expected_i_xi_min).all()
                assert (i_xi <= expected_i_xi_max).all()


class TestNesting:
    @pytest.mark.parametrize(
        "parent_grid_fixture, child_grid_fixture",
        [
            ("parent_grid", "child_grid_that_straddles"),
            ("parent_grid_that_straddles", "child_grid"),
        ],
    )
    def test_error_if_child_grid_beyond_parent_grid(
        self, parent_grid_fixture, child_grid_fixture, request
    ):
        parent_grid = request.getfixturevalue(parent_grid_fixture)
        child_grid = request.getfixturevalue(child_grid_fixture)

        with pytest.raises(ValueError, match="Some points are outside the grid."):
            Nesting(parent_grid=parent_grid, child_grid=child_grid)


# Test map_child_boundaries_onto_parent_grid_indices
# - if parent grid has no land along boundaries, update_indices doesn't do anything
# - if you use update_indices, the indices indeed map to wet points
# - check that variable names, period and variable attributes are correctly set
# - check that indices are monotonically increasing

# Test modify_child_topography_and_mask
# - check that topography and mask are only modified near boundary
# - check that mask is not modified if there is no parent land along boundaries
# - check that this does nothing if the parent_grid and child_grid coincide

# Test compute_boundary_distance
# - check that this is 1 at boundaries at 0 in interior if no land
# - check that there are some 1 along boundaries if there is land along boundary
