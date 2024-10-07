import pytest
from pathlib import Path
import xarray.testing as xrt

from roms_tools.utils import partition, partition_netcdf
from roms_tools import Grid


@pytest.fixture
def grid():

    grid = Grid(nx=30, ny=30, size_x=80, size_y=80, center_lon=-20, center_lat=0, rot=0)

    return grid


class TestPartitionGrid:
    def test_partition_grid_along_x(self, grid):
        _, [ds1, ds2, ds3] = partition(grid.ds, np_eta=3, np_xi=1)

        assert ds1.sizes == {
            "eta_rho": 11,
            "xi_rho": 32,
            "xi_u": 31,
            "eta_v": 10,
            "eta_psi": 11,
            "xi_psi": 33,
            "eta_coarse": 6,
            "xi_coarse": 17,
            "s_rho": 100,
            "s_w": 101,
        }
        assert ds2.sizes == {
            "eta_rho": 10,
            "xi_rho": 32,
            "xi_u": 31,
            "eta_v": 10,
            "eta_psi": 10,
            "xi_psi": 33,
            "eta_coarse": 5,
            "xi_coarse": 17,
            "s_rho": 100,
            "s_w": 101,
        }
        assert ds3.sizes == {
            "eta_rho": 11,
            "xi_rho": 32,
            "xi_u": 31,
            "eta_v": 11,
            "eta_psi": 12,
            "xi_psi": 33,
            "eta_coarse": 6,
            "xi_coarse": 17,
            "s_rho": 100,
            "s_w": 101,
        }

    def test_partition_grid_along_y(self, grid):
        _, [ds1, ds2, ds3] = partition(grid.ds, np_eta=1, np_xi=3)

        assert ds1.sizes == {
            "eta_rho": 32,
            "xi_rho": 11,
            "xi_u": 10,
            "eta_v": 31,
            "eta_psi": 33,
            "xi_psi": 11,
            "eta_coarse": 17,
            "xi_coarse": 6,
            "s_rho": 100,
            "s_w": 101,
        }
        assert ds2.sizes == {
            "eta_rho": 32,
            "xi_rho": 10,
            "xi_u": 10,
            "eta_v": 31,
            "eta_psi": 33,
            "xi_psi": 10,
            "eta_coarse": 17,
            "xi_coarse": 5,
            "s_rho": 100,
            "s_w": 101,
        }
        assert ds3.sizes == {
            "eta_rho": 32,
            "xi_rho": 11,
            "xi_u": 11,
            "eta_v": 31,
            "eta_psi": 33,
            "xi_psi": 12,
            "eta_coarse": 17,
            "xi_coarse": 6,
            "s_rho": 100,
            "s_w": 101,
        }

    def test_partition_grid_along_xy(self, grid):
        # decomposition is increasing eta to the right, increasing xi down
        # fmt: off
        _, [ds1, ds2, ds3,
         ds4, ds5, ds6,
         ds7, ds8, ds9] = partition(grid.ds, np_eta=3, np_xi=3)
        # fmt: on

        assert ds1.sizes == {
            "eta_rho": 11,
            "xi_rho": 11,
            "xi_u": 10,
            "eta_v": 10,
            "eta_psi": 11,
            "xi_psi": 11,
            "eta_coarse": 6,
            "xi_coarse": 6,
            "s_rho": 100,
            "s_w": 101,
        }
        assert ds4.sizes == {
            "eta_rho": 10,
            "xi_rho": 11,
            "xi_u": 10,
            "eta_v": 10,
            "eta_coarse": 5,
            "eta_psi": 10,
            "xi_psi": 11,
            "xi_coarse": 6,
            "s_rho": 100,
            "s_w": 101,
        }
        assert ds7.sizes == {
            "eta_rho": 11,
            "xi_rho": 11,
            "xi_u": 10,
            "eta_v": 11,
            "eta_psi": 12,
            "xi_psi": 11,
            "eta_coarse": 6,
            "xi_coarse": 6,
            "s_rho": 100,
            "s_w": 101,
        }
        assert ds2.sizes == {
            "eta_rho": 11,
            "xi_rho": 10,
            "xi_u": 10,
            "eta_v": 10,
            "eta_psi": 11,
            "xi_psi": 10,
            "eta_coarse": 6,
            "xi_coarse": 5,
            "s_rho": 100,
            "s_w": 101,
        }
        assert ds5.sizes == {
            "eta_rho": 10,
            "xi_rho": 10,
            "xi_u": 10,
            "eta_v": 10,
            "eta_psi": 10,
            "xi_psi": 10,
            "eta_coarse": 5,
            "xi_coarse": 5,
            "s_rho": 100,
            "s_w": 101,
        }
        assert ds8.sizes == {
            "eta_rho": 11,
            "xi_rho": 10,
            "xi_u": 10,
            "eta_v": 11,
            "eta_psi": 12,
            "xi_psi": 10,
            "eta_coarse": 6,
            "xi_coarse": 5,
            "s_rho": 100,
            "s_w": 101,
        }
        assert ds3.sizes == {
            "eta_rho": 11,
            "xi_rho": 11,
            "xi_u": 11,
            "eta_v": 10,
            "eta_psi": 11,
            "xi_psi": 12,
            "eta_coarse": 6,
            "xi_coarse": 6,
            "s_rho": 100,
            "s_w": 101,
        }
        assert ds6.sizes == {
            "eta_rho": 10,
            "xi_rho": 11,
            "xi_u": 11,
            "eta_v": 10,
            "eta_psi": 10,
            "xi_psi": 12,
            "eta_coarse": 5,
            "xi_coarse": 6,
            "s_rho": 100,
            "s_w": 101,
        }
        assert ds9.sizes == {
            "eta_rho": 11,
            "xi_rho": 11,
            "xi_u": 11,
            "eta_v": 11,
            "eta_psi": 12,
            "xi_psi": 12,
            "eta_coarse": 6,
            "xi_coarse": 6,
            "s_rho": 100,
            "s_w": 101,
        }

    def test_partition_grid_no_op(self, grid):
        _, partitioned_datasets = partition(grid.ds, np_eta=1, np_xi=1)

        xrt.assert_identical(partitioned_datasets[0], grid.ds)

    def test_invalid_partitioning(self, grid):
        with pytest.raises(
            ValueError, match="np_eta and np_xi must be positive integers"
        ):
            partition(grid.ds, np_eta=3.0, np_xi=1)

        with pytest.raises(
            ValueError, match="np_eta and np_xi must be positive integers"
        ):
            partition(grid.ds, np_eta=-3, np_xi=1)

        with pytest.raises(ValueError, match="cannot be evenly divided"):
            partition(grid.ds, np_eta=4, np_xi=1)


class TestPartitionMissingDims:
    def test_partition_missing_dims(self, grid):
        dims_to_drop = ["xi_u", "eta_v", "eta_coarse", "xi_coarse"]

        ds_missing_dims = grid.ds.drop_dims(dims_to_drop)

        _, partitioned_datasets = partition(ds_missing_dims, np_eta=1, np_xi=1)

        xrt.assert_identical(partitioned_datasets[0], ds_missing_dims)

    def test_partition_missing_all_dims(self, grid):
        # this is all the partitionable dims, so in this case the file will just be copied np_eta * np_xi times
        dims_to_drop = ["eta_rho", "xi_rho", "xi_u", "eta_v", "eta_coarse", "xi_coarse"]

        ds_missing_dims = grid.ds.drop_dims(dims_to_drop)

        _, partitioned_datasets = partition(ds_missing_dims, np_eta=1, np_xi=1)

        xrt.assert_identical(partitioned_datasets[0], ds_missing_dims)


class TestFileNumbers:
    def test_partition_file_numbers(self, grid):
        np_eta = 3
        np_xi = 5
        file_numbers, _ = partition(grid.ds, np_eta=np_eta, np_xi=np_xi)

        # Generate the expected file numbers
        expected_file_numbers = list(range(np_eta * np_xi))

        # Check if file_numbers is a continuous range without gaps
        assert set(file_numbers) == set(expected_file_numbers)


class TestPartitionNetcdf:
    def test_partition_netcdf(self, grid, tmp_path):
        filepath = tmp_path / "test_grid.nc"
        grid.save(filepath)

        saved_filenames = partition_netcdf(filepath, np_eta=3, np_xi=3)

        filepath_str = str(filepath.with_suffix(""))
        expected_filepath_list = [
            Path(filepath_str + f".{index}.nc") for index in range(9)
        ]

        assert saved_filenames == expected_filepath_list

        for expected_filepath in expected_filepath_list:
            assert expected_filepath.exists()
            expected_filepath.unlink()
