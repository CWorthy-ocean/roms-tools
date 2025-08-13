from pathlib import Path

import pytest
import xarray as xr

from roms_tools import Grid
from roms_tools.tiling.join import (
    _find_common_dims,
    _find_transitions,
    _infer_partition_layout_from_datasets,
    join_netcdf,
    open_partitions,
)
from roms_tools.tiling.partition import partition_netcdf


@pytest.fixture
def partitioned_grid_factory(tmp_path, large_grid) -> tuple[Grid, list[Path]]:
    """
    A fixture factory that returns a function to generate partitioned files
    with a configurable layout.
    """

    def _partitioned_files(np_xi: int, np_eta: int):
        partable_grid = large_grid
        partable_grid.save(tmp_path / "test_grid.nc")
        parted_files = partition_netcdf(
            tmp_path / "test_grid.nc", np_xi=np_xi, np_eta=np_eta
        )
        return partable_grid, parted_files

    return _partitioned_files


@pytest.fixture
def partitioned_ic_factory(
    tmp_path, initial_conditions_on_large_grid
) -> tuple[Path, list[Path]]:
    def _partitioned_files(np_xi: int, np_eta: int):
        whole_ics = initial_conditions_on_large_grid
        whole_path = whole_ics.save(tmp_path / "test_ic.nc")[0]
        parted_paths = partition_netcdf(whole_path, np_xi=np_xi, np_eta=np_eta)
        return whole_path, parted_paths

    return _partitioned_files


class TestHelperFunctions:
    def test_find_common_dims(self):
        """Test _find_common_dims with different datasets and directions.

        Tests for a valid common dimension, a valid dimension that is not common,
        an invalid direction, and a ValueError when no common dimension is found.
        """
        # Create mock xarray.Dataset objects for testing
        ds1 = xr.Dataset(coords={"xi_rho": [0], "xi_u": [0]})
        ds2 = xr.Dataset(coords={"xi_rho": [0], "xi_u": [0]})
        ds3 = xr.Dataset(coords={"xi_rho": [0], "xi_v": [0]})
        datasets_common = [ds1, ds2]
        datasets_not_common = [ds1, ds3]

        # Test case with a common dimension ("xi_rho" and "xi_u")
        # The function should find 'xi_rho' and 'xi_u' and return them in a list.
        assert _find_common_dims("xi", datasets_common) == ["xi_rho", "xi_u"]

        # Test case where a dimension is not common to all datasets
        # The function should find only "xi_rho" as "xi_u" is not in ds3.
        assert _find_common_dims("xi", datasets_not_common) == ["xi_rho"]

        # Test case for an invalid direction, should raise a ValueError
        with pytest.raises(ValueError, match="'direction' must be 'xi' or 'eta'"):
            _find_common_dims("zeta", datasets_common)

        # Test case where no common dimensions exist
        ds_no_common1 = xr.Dataset(coords={"xi_rho": [0]})
        ds_no_common2 = xr.Dataset(coords={"xi_v": [0]})
        with pytest.raises(
            ValueError, match="No common point found along direction xi"
        ):
            _find_common_dims("xi", [ds_no_common1, ds_no_common2])

    def test_find_transitions(self):
        """Test _find_transitions with various input lists.

        Test cases include lists with no transitions, a single transition,
        multiple transitions, and edge cases like empty and single-element lists.
        """
        # Test case with no transitions
        assert _find_transitions([10, 10, 10, 10]) == []

        # Test case with a single transition
        assert _find_transitions([10, 10, 12, 12]) == [2]

        # Test case with multiple transitions
        assert _find_transitions([10, 12, 12, 14, 14, 14]) == [1, 3]

        # Test case with transitions on every element
        assert _find_transitions([10, 12, 14, 16]) == [1, 2, 3]

        # Edge case: empty list
        assert _find_transitions([]) == []

        # Edge case: single-element list
        assert _find_transitions([10]) == []

    def test_infer_partition_layout_from_datasets(self):
        """Test _infer_partition_layout_from_datasets with various layouts.

        Tests include a single dataset, a 2x2 grid, a 4x1 grid (single row),
        and a 1x4 grid (single column).
        """
        # Test case 1: Single dataset (1x1 partition)
        ds1 = xr.Dataset(coords={"eta_rho": [0], "xi_rho": [0]})
        assert _infer_partition_layout_from_datasets([ds1]) == (1, 1)

        # Test case 2: 2x2 grid partition.
        # The eta dimension will transition after the second dataset (np_xi=2).
        ds_2x2_1 = xr.Dataset(coords={"eta_rho": [0] * 20, "xi_rho": [0] * 10})
        ds_2x2_2 = xr.Dataset(coords={"eta_rho": [0] * 20, "xi_rho": [0] * 10})
        ds_2x2_3 = xr.Dataset(coords={"eta_rho": [0] * 10, "xi_rho": [0] * 10})
        ds_2x2_4 = xr.Dataset(coords={"eta_rho": [0] * 10, "xi_rho": [0] * 10})
        datasets_2x2 = [ds_2x2_1, ds_2x2_2, ds_2x2_3, ds_2x2_4]
        assert _infer_partition_layout_from_datasets(datasets_2x2) == (2, 2)

        # Test case 3: 4x1 grid partition (single row).
        # The eta dimension sizes are all the same, so no transition is detected.
        # The function falls back to returning nd, 1.
        ds_4x1_1 = xr.Dataset(coords={"eta_rho": [0] * 10, "xi_rho": [0] * 5})
        ds_4x1_2 = xr.Dataset(coords={"eta_rho": [0] * 10, "xi_rho": [0] * 5})
        ds_4x1_3 = xr.Dataset(coords={"eta_rho": [0] * 10, "xi_rho": [0] * 5})
        ds_4x1_4 = xr.Dataset(coords={"eta_rho": [0] * 10, "xi_rho": [0] * 5})
        datasets_4x1 = [ds_4x1_1, ds_4x1_2, ds_4x1_3, ds_4x1_4]
        assert _infer_partition_layout_from_datasets(datasets_4x1) == (4, 1)

        # Test case 4: 1x4 grid partition (single column).
        # The xi dimension is partitioned, so the eta dimensions must change at every step.
        ds_1x4_1 = xr.Dataset(coords={"eta_rho": [0] * 10, "xi_rho": [0] * 20})
        ds_1x4_2 = xr.Dataset(coords={"eta_rho": [0] * 12, "xi_rho": [0] * 20})
        ds_1x4_3 = xr.Dataset(coords={"eta_rho": [0] * 14, "xi_rho": [0] * 20})
        ds_1x4_4 = xr.Dataset(coords={"eta_rho": [0] * 16, "xi_rho": [0] * 20})
        datasets_1x4 = [ds_1x4_1, ds_1x4_2, ds_1x4_3, ds_1x4_4]
        # In this case, `_find_transitions` for eta will find a transition at index 1, so np_xi=1.
        # This will correctly return (1, 4).
        assert _infer_partition_layout_from_datasets(datasets_1x4) == (1, 4)


class TestJoinROMSData:
    @pytest.mark.parametrize(
        "np_xi, np_eta",
        [
            (1, 1),
            (1, 6),
            (6, 1),
            (2, 2),
            (3, 3),
            (3, 4),
            (4, 3),  # (12,24)
            # # All possible:
            # # Single-partition grid
            # (1,1)
            # # Single-row grids
            # (2, 1), (3, 1), (4, 1), (6, 1), (12, 1),
            # # Single-column grids
            # (1, 2), (1, 3), (1, 4), (1, 6), (1, 8), (1, 12), (1, 24),
            # # Multi-row, multi-column grids
            # (2, 2), (2, 3), (2, 4), (2, 6), (2, 8), (2, 12), (2, 24),
            # (3, 2), (3, 3), (3, 4), (3, 6), (3, 8), (3, 12), (3, 24),
            # (4, 2), (4, 3), (4, 4), (4, 6), (4, 8), (4, 12), (4, 24),
            # (6, 2), (6, 3), (6, 4), (6, 6), (6, 8), (6, 12), (6, 24),
            # (12, 2), (12, 3), (12, 4), (12, 6), (12, 8), (12, 12), (12, 24)
        ],
    )
    def test_open_grid_partitions(self, partitioned_grid_factory, np_xi, np_eta):
        grid, partitions = partitioned_grid_factory(np_xi=np_xi, np_eta=np_eta)
        joined_grid = open_partitions(partitions)

        for v in grid.ds.variables:
            assert (grid.ds[v].values == joined_grid[v].values).all(), (
                f"{v} does not match in joined dataset"
            )

    def test_join_grid_netcdf(self, partitioned_grid_factory):
        grid, partitions = partitioned_grid_factory(np_xi=3, np_eta=4)
        joined_netcdf = join_netcdf(
            partitions, output_path=partitions[0].parent / "joined_grid.nc"
        )
        assert joined_netcdf.exists()
        joined_grid = xr.open_dataset(joined_netcdf)

        for v in grid.ds.variables:
            assert (grid.ds[v].values == joined_grid[v].values).all(), (
                f"{v} does not match in joined dataset"
            )

    @pytest.mark.parametrize(
        "np_xi, np_eta",
        [
            (1, 1),
            (1, 6),
            (6, 1),
            (2, 2),
            (3, 3),
            (3, 4),
            (4, 3),  # (12,24)
        ],
    )
    def test_open_initial_condition_partitions(
        self, partitioned_ic_factory, np_xi, np_eta
    ):
        whole_file, partitioned_files = partitioned_ic_factory(
            np_xi=np_xi, np_eta=np_eta
        )
        joined_ics = open_partitions(partitioned_files)
        whole_ics = xr.open_dataset(whole_file, decode_times=True)

        for v in whole_ics.variables:
            assert (whole_ics[v].values == joined_ics[v].values).all(), (
                f"{v} does not match in joined dataset: {joined_ics[v].values} vs {whole_ics[v].values}"
            )

    def test_join_initial_condition_netcdf(self, tmp_path, partitioned_ic_factory):
        whole_file, partitioned_files = partitioned_ic_factory(np_xi=3, np_eta=4)
        whole_ics = xr.open_dataset(whole_file, decode_times=True)

        joined_netcdf = join_netcdf(
            partitioned_files, output_path=partitioned_files[0].parent / "joined_ics.nc"
        )
        assert joined_netcdf.exists()
        joined_ics = xr.open_dataset(joined_netcdf, decode_times=True)

        for v in whole_ics.variables:  #
            assert (whole_ics[v].values == joined_ics[v].values).all(), (
                f"{v} does not match in joined dataset: {joined_ics[v].values} vs {whole_ics[v].values}"
            )
