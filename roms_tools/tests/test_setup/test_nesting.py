import pytest
import xarray as xr
import numpy as np
import logging
from pathlib import Path
from roms_tools import Grid, ChildGrid
from roms_tools.setup.utils import get_boundary_coords
from conftest import calculate_file_hash
from roms_tools.setup.nesting import (
    interpolate_indices,
    map_child_boundaries_onto_parent_grid_indices,
    compute_boundary_distance,
    modify_child_topography_and_mask,
)


@pytest.fixture()
def big_grid():
    return Grid(
        nx=5, ny=7, center_lon=-23, center_lat=61, rot=20, size_x=1800, size_y=2400
    )


@pytest.fixture()
def small_grid():
    return Grid(
        nx=10, ny=10, center_lon=-23, center_lat=61, rot=-20, size_x=500, size_y=500
    )


@pytest.fixture()
def baby_grid():
    return Grid(
        nx=3, ny=5, center_lon=-23, center_lat=61, rot=0, size_x=200, size_y=200
    )


@pytest.fixture()
def big_grid_that_straddles():
    return Grid(
        nx=5, ny=7, center_lon=10, center_lat=61, rot=20, size_x=1800, size_y=2400
    )


@pytest.fixture()
def small_grid_that_straddles():
    return Grid(
        nx=10, ny=10, center_lon=10, center_lat=61, rot=-20, size_x=500, size_y=500
    )


@pytest.fixture()
def child_grid(big_grid):
    return ChildGrid(
        parent_grid=big_grid,
        nx=10,
        ny=10,
        center_lon=-23,
        center_lat=61,
        rot=-20,
        size_x=500,
        size_y=500,
    )


@pytest.fixture()
def child_grid_that_straddles(big_grid_that_straddles):
    return ChildGrid(
        parent_grid=big_grid_that_straddles,
        nx=10,
        ny=10,
        center_lon=10,
        center_lat=61,
        rot=-20,
        size_x=500,
        size_y=500,
    )


class TestInterpolateIndices:
    @pytest.mark.parametrize(
        "grid",
        [
            "big_grid",
            "big_grid_that_straddles",
        ],
    )
    def test_correct_indices_of_same_grid(self, grid, caplog, request):
        """Verify boundary indices are correctly interpolated for the same grid."""

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
        "big_grid_fixture, small_grid_fixture",
        [
            ("big_grid", "small_grid"),
            ("big_grid_that_straddles", "small_grid_that_straddles"),
        ],
    )
    def test_indices_are_within_range_of_parent_grid(
        self, big_grid_fixture, small_grid_fixture, request
    ):
        """Ensure interpolated indices fall within the parent grid's bounds."""

        big_grid = request.getfixturevalue(big_grid_fixture)
        small_grid = request.getfixturevalue(small_grid_fixture)

        bdry_coords_dict = get_boundary_coords()
        for location in ["rho", "u", "v"]:
            for direction in ["south", "east", "north", "west"]:
                bdry_coords = bdry_coords_dict[location][direction]
                lon = small_grid.ds[f"lon_{location}"].isel(**bdry_coords)
                lat = small_grid.ds[f"lat_{location}"].isel(**bdry_coords)
                mask = small_grid.ds[f"mask_{location}"].isel(**bdry_coords)

                i_eta, i_xi = interpolate_indices(big_grid.ds, lon, lat, mask)

                expected_i_eta_min = -0.5
                expected_i_eta_max = big_grid.ds.eta_rho[-1] - 0.5
                expected_i_xi_min = -0.5
                expected_i_xi_max = big_grid.ds.xi_rho[-1] - 0.5

                assert (i_eta >= expected_i_eta_min).all()
                assert (i_eta <= expected_i_eta_max).all()
                assert (i_xi >= expected_i_xi_min).all()
                assert (i_xi <= expected_i_xi_max).all()


class TestMapChildBoundaries:
    def test_update_indices_does_nothing_if_no_parent_land(self, small_grid, baby_grid):
        """Verify no change in indices when parent grid has no land at boundaries."""

        ds_without_updated_indices = map_child_boundaries_onto_parent_grid_indices(
            small_grid.ds, baby_grid.ds, update_land_indices=False
        )
        ds_with_updated_indices = map_child_boundaries_onto_parent_grid_indices(
            small_grid.ds, baby_grid.ds, update_land_indices=True
        )

        xr.testing.assert_allclose(ds_without_updated_indices, ds_with_updated_indices)

    @pytest.mark.parametrize(
        "big_grid_fixture, small_grid_fixture",
        [
            ("big_grid", "small_grid"),
            ("big_grid_that_straddles", "small_grid_that_straddles"),
        ],
    )
    def test_updated_indices_map_to_wet_points(
        self, big_grid_fixture, small_grid_fixture, request
    ):
        """Check updated indices map to wet points on the parent grid."""

        big_grid = request.getfixturevalue(big_grid_fixture)
        small_grid = request.getfixturevalue(small_grid_fixture)

        ds = map_child_boundaries_onto_parent_grid_indices(big_grid.ds, small_grid.ds)
        for direction in ["south", "east", "north", "west"]:
            for location in ["rho", "u", "v"]:
                if location == "rho":
                    dim = "two"
                    location = "r"
                    # convert from absolute indices [-0.5, ...] to [0, ...]
                    i_xi = ds[f"child_{direction}_{location}"].isel({dim: 0}) + 0.5
                    i_eta = ds[f"child_{direction}_{location}"].isel({dim: 1}) + 0.5
                    for i in range(len(i_xi)):
                        i_eta_lower = int(np.floor(i_eta[i]))
                        i_xi_lower = int(np.floor(i_xi[i]))
                        mask = big_grid.ds.mask_rho.isel(
                            eta_rho=slice(i_eta_lower, i_eta_lower + 2),
                            xi_rho=slice(i_xi_lower, i_xi_lower + 2),
                        )
                        assert np.sum(mask) > 0
                # TODO: check also u and v locations

    @pytest.mark.parametrize(
        "big_grid_fixture, small_grid_fixture",
        [
            ("big_grid", "small_grid"),
            ("big_grid_that_straddles", "small_grid_that_straddles"),
        ],
    )
    def test_indices_are_monotonically_increasing(
        self, big_grid_fixture, small_grid_fixture, request
    ):
        """Test that child boundary indices are monotonically increasing or decreasing
        in both the xi and eta directions, for all boundaries and locations."""

        big_grid = request.getfixturevalue(big_grid_fixture)
        small_grid = request.getfixturevalue(small_grid_fixture)

        for update_land_indices in [False, True]:
            ds = map_child_boundaries_onto_parent_grid_indices(
                big_grid.ds, small_grid.ds, update_land_indices=update_land_indices
            )

            for direction in ["south", "east", "north", "west"]:
                for location in ["rho", "u", "v"]:
                    if location == "rho":
                        dim = "two"
                        location = "r"
                    else:
                        dim = "three"

                    for coord in [0, 1]:  # 0 for xi, 1 for eta
                        index_values = ds[f"child_{direction}_{location}"].isel(
                            {dim: coord}
                        )
                        assert np.all(np.diff(index_values) >= 0) or np.all(
                            np.diff(index_values) <= 0
                        )


class TestBoundaryDistance:
    @pytest.mark.parametrize(
        "grid_fixture",
        [
            "small_grid",
            "baby_grid",
        ],
    )
    def test_boundary_distance_for_grid_without_land_along_boundary(
        self, grid_fixture, request
    ):
        """Ensure boundary distance is zero for grids without land along boundaries."""

        grid = request.getfixturevalue(grid_fixture)
        alpha = compute_boundary_distance(grid.ds.mask_rho)

        # check that all boundaries are zero
        assert (alpha.isel(eta_rho=0) == 0).all()
        assert (alpha.isel(eta_rho=-1) == 0).all()
        assert (alpha.isel(xi_rho=0) == 0).all()
        assert (alpha.isel(xi_rho=-1) == 0).all()

        # check that inner values are 1
        assert (
            alpha.isel(
                eta_rho=alpha.sizes["eta_rho"] // 2, xi_rho=alpha.sizes["xi_rho"] // 2
            )
            == 1
        )

    def test_boundary_distance_for_grid_with_land_along_boundary(self, big_grid):
        """Test that there are 1s along the boundary of alpha if the grid has land along
        the boundary."""
        alpha = compute_boundary_distance(big_grid.ds.mask_rho)
        assert (alpha.isel(eta_rho=0) == 1).any()
        assert (alpha.isel(eta_rho=-1) == 1).any()
        assert (alpha.isel(xi_rho=0) == 1).any()
        assert (alpha.isel(xi_rho=-1) == 1).any()


class TestModifyChid:
    def test_mask_is_not_modified_if_no_parent_land_along_boundaries(
        self, small_grid, baby_grid
    ):
        """Confirm child mask remains unchanged if no parent land is at boundaries."""

        mask_original = baby_grid.ds.mask_rho.copy()
        modified_baby_grid_ds = modify_child_topography_and_mask(
            small_grid.ds, baby_grid.ds
        )
        xr.testing.assert_allclose(modified_baby_grid_ds.mask_rho, mask_original)

    @pytest.mark.parametrize(
        "grid_fixture",
        [
            "big_grid",
            "small_grid",
            "baby_grid",
        ],
    )
    def test_no_modification_if_parent_and_child_coincide(self, grid_fixture, request):
        """Ensure no changes occur when parent and child grids coincide."""

        grid = request.getfixturevalue(grid_fixture)

        h_original = grid.ds.h.copy()
        mask_original = grid.ds.mask_rho.copy()
        modified_grid_ds = modify_child_topography_and_mask(grid.ds, grid.ds)

        xr.testing.assert_allclose(modified_grid_ds.h, h_original)
        xr.testing.assert_allclose(modified_grid_ds.mask_rho, mask_original)

    def test_modification_only_along_boundaries(self, big_grid, small_grid):
        """Test that modifications to the child grid's topography and mask occur only
        along the boundaries, leaving the interior unchanged."""

        # Make copies of original data for comparison
        h_original = small_grid.ds.h.copy()
        mask_original = small_grid.ds.mask_rho.copy()

        # Apply the modification function
        modified_ds = modify_child_topography_and_mask(big_grid.ds, small_grid.ds)

        # Calculate the center indices for the grid
        eta_center = h_original.sizes["eta_rho"] // 2
        xi_center = h_original.sizes["xi_rho"] // 2

        # Assert that the center values remain the same
        assert mask_original.isel(
            eta_rho=eta_center, xi_rho=xi_center
        ) == modified_ds.mask_rho.isel(
            eta_rho=eta_center, xi_rho=xi_center
        ), "Mask at the grid center was modified."

        assert h_original.isel(
            eta_rho=eta_center, xi_rho=xi_center
        ) == modified_ds.h.isel(
            eta_rho=eta_center, xi_rho=xi_center
        ), "Topography at the grid center was modified."


class TestNesting:
    @pytest.mark.parametrize(
        "child_grid_fixture",
        ["child_grid", "child_grid_that_straddles"],
    )
    def test_successful_initialization(self, child_grid_fixture, request):
        child_grid = request.getfixturevalue(child_grid_fixture)

        assert child_grid.boundaries == {
            "south": True,
            "east": True,
            "north": True,
            "west": True,
        }
        assert child_grid.metadata["prefix"] == "child"
        assert child_grid.metadata["period"] == 3600.0
        assert isinstance(child_grid.ds, xr.Dataset)
        assert isinstance(child_grid.ds_nesting, xr.Dataset)

        ds = child_grid.ds_nesting
        for direction in ["south", "east", "north", "west"]:
            for location in ["r", "u", "v"]:
                assert f"child_{direction}_{location}" in ds.data_vars
                assert (
                    ds[f"child_{direction}_{location}"].attrs["output_period"] == 3600.0
                )
                if location == "r":
                    assert (
                        ds[f"child_{direction}_{location}"].attrs["output_vars"]
                        == "zeta, temp, salt"
                    )
                elif location == "u":
                    assert (
                        ds[f"child_{direction}_{location}"].attrs["output_vars"]
                        == "ubar, u, up"
                    )
                elif location == "v":
                    assert (
                        ds[f"child_{direction}_{location}"].attrs["output_vars"]
                        == "vbar, v, vp"
                    )

    @pytest.mark.parametrize(
        "big_grid_fixture, small_grid_fixture",
        [
            ("big_grid", "small_grid_that_straddles"),
            ("big_grid_that_straddles", "small_grid"),
        ],
    )
    def test_error_if_child_grid_beyond_parent_grid(
        self, big_grid_fixture, small_grid_fixture, request
    ):
        big_grid = request.getfixturevalue(big_grid_fixture)
        small_grid = request.getfixturevalue(small_grid_fixture)

        import dataclasses

        params = dataclasses.asdict(small_grid)
        del params["ds"], params["straddle"]

        with pytest.raises(ValueError, match="Some points are outside the grid."):
            ChildGrid(parent_grid=big_grid, **params)

    @pytest.mark.parametrize(
        "child_grid_fixture",
        ["child_grid", "child_grid_that_straddles"],
    )
    def test_plot(self, child_grid_fixture, request):
        """Test plot method."""
        child_grid = request.getfixturevalue(child_grid_fixture)

        child_grid.plot()
        child_grid.plot_nesting()
        child_grid.plot_nesting(with_dim_names=True)

    def test_save(self, child_grid, tmp_path):
        """Test save methods."""

        for file_str in ["test_grid", "test_grid.nc"]:
            # Create a temporary filepath using the tmp_path fixture
            for filepath in [tmp_path / file_str, str(tmp_path / file_str)]:

                saved_filenames = child_grid.save(filepath)
                # Check if the .nc file was created
                filepath = Path(filepath).with_suffix(".nc")
                assert saved_filenames == [filepath]
                assert filepath.exists()
                # Clean up the .nc file
                filepath.unlink()

        for file_str in ["test_nesting", "test_nesting.nc"]:
            # Create a temporary filepath using the tmp_path fixture
            for filepath in [tmp_path / file_str, str(tmp_path / file_str)]:

                saved_filenames = child_grid.save_nesting(filepath)
                # Check if the .nc file was created
                filepath = Path(filepath).with_suffix(".nc")
                assert saved_filenames == [filepath]
                assert filepath.exists()
                # Clean up the .nc file
                filepath.unlink()

    def test_disabled_from_file_method(self):
        """Test that parent from_file method is indeed disabled
        """
        with pytest.raises(NotImplementedError, match="The 'from_file' method is disabled in this subclass."):
            ChildGrid.from_file(filepath="some_file.nc")

    def test_roundtrip_yaml(self, child_grid, tmp_path):
        """Test that creating a ChildGrid object, saving its parameters to yaml file,
        and re-opening yaml file creates the same object."""

        # Create a temporary filepath using the tmp_path fixture
        file_str = "test_yaml"
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            child_grid.to_yaml(filepath)

            child_grid_from_file = ChildGrid.from_yaml(filepath)

            assert child_grid == child_grid_from_file

            filepath = Path(filepath)
            filepath.unlink()

    def test_files_have_same_hash(self, child_grid, tmp_path):

        yaml_filepath = tmp_path / "test_yaml.yaml"
        filepath1 = tmp_path / "test1.nc"
        filepath2 = tmp_path / "test2.nc"
        grid_filepath1 = tmp_path / "grid_test1.nc"
        grid_filepath2 = tmp_path / "grid_test2.nc"

        child_grid.to_yaml(yaml_filepath)
        child_grid.save(grid_filepath1)
        child_grid.save_nesting(filepath1)

        child_grid_from_file = ChildGrid.from_yaml(yaml_filepath)
        child_grid_from_file.save(grid_filepath2)
        child_grid_from_file.save_nesting(filepath2)

        hash1 = calculate_file_hash(filepath1)
        hash2 = calculate_file_hash(filepath2)
        assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

        hash1 = calculate_file_hash(grid_filepath1)
        hash2 = calculate_file_hash(grid_filepath2)
        assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

        yaml_filepath.unlink()
        filepath1.unlink()
        filepath2.unlink()
        grid_filepath1.unlink()
        grid_filepath2.unlink()
