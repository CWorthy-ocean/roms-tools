from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

from roms_tools import Grid
from roms_tools.tiling.partition import partition, partition_netcdf


@pytest.fixture
def grid():
    grid = Grid(nx=30, ny=30, size_x=80, size_y=80, center_lon=-20, center_lat=0, rot=0)

    return grid


class TestPartitionGrid:
    def test_partition_grid_along_x(self, grid):
        _, [ds1, ds2, ds3] = partition(grid.ds, np_eta=3, np_xi=1)

        assert dict(ds1.sizes) == {
            "eta_rho": 11,
            "xi_rho": 32,
            "xi_u": 31,
            "eta_v": 10,
            # "eta_psi": 11,
            # "xi_psi": 33,
            "eta_coarse": 6,
            "xi_coarse": 17,
            "s_rho": 100,
            "s_w": 101,
        }
        assert dict(ds2.sizes) == {
            "eta_rho": 10,
            "xi_rho": 32,
            "xi_u": 31,
            "eta_v": 10,
            # "eta_psi": 10,
            # "xi_psi": 33,
            "eta_coarse": 5,
            "xi_coarse": 17,
            "s_rho": 100,
            "s_w": 101,
        }
        assert dict(ds3.sizes) == {
            "eta_rho": 11,
            "xi_rho": 32,
            "xi_u": 31,
            "eta_v": 11,
            # "eta_psi": 12,
            # "xi_psi": 33,
            "eta_coarse": 6,
            "xi_coarse": 17,
            "s_rho": 100,
            "s_w": 101,
        }

    def test_partition_grid_along_y(self, grid):
        _, [ds1, ds2, ds3] = partition(grid.ds, np_eta=1, np_xi=3)

        assert dict(ds1.sizes) == {
            "eta_rho": 32,
            "xi_rho": 11,
            "xi_u": 10,
            "eta_v": 31,
            # "eta_psi": 33,
            # "xi_psi": 11,
            "eta_coarse": 17,
            "xi_coarse": 6,
            "s_rho": 100,
            "s_w": 101,
        }
        assert dict(ds2.sizes) == {
            "eta_rho": 32,
            "xi_rho": 10,
            "xi_u": 10,
            "eta_v": 31,
            # "eta_psi": 33,
            # "xi_psi": 10,
            "eta_coarse": 17,
            "xi_coarse": 5,
            "s_rho": 100,
            "s_w": 101,
        }
        assert dict(ds3.sizes) == {
            "eta_rho": 32,
            "xi_rho": 11,
            "xi_u": 11,
            "eta_v": 31,
            # "eta_psi": 33,
            # "xi_psi": 12,
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

        assert dict(ds1.sizes) == {
            "eta_rho": 11,
            "xi_rho": 11,
            "xi_u": 10,
            "eta_v": 10,
            # "eta_psi": 11,
            # "xi_psi": 11,
            "eta_coarse": 6,
            "xi_coarse": 6,
            "s_rho": 100,
            "s_w": 101,
        }
        assert dict(ds4.sizes) == {
            "eta_rho": 10,
            "xi_rho": 11,
            "xi_u": 10,
            "eta_v": 10,
            "eta_coarse": 5,
            # "eta_psi": 10,
            # "xi_psi": 11,
            "xi_coarse": 6,
            "s_rho": 100,
            "s_w": 101,
        }
        assert dict(ds7.sizes) == {
            "eta_rho": 11,
            "xi_rho": 11,
            "xi_u": 10,
            "eta_v": 11,
            # "eta_psi": 12,
            # "xi_psi": 11,
            "eta_coarse": 6,
            "xi_coarse": 6,
            "s_rho": 100,
            "s_w": 101,
        }
        assert dict(ds2.sizes) == {
            "eta_rho": 11,
            "xi_rho": 10,
            "xi_u": 10,
            "eta_v": 10,
            # "eta_psi": 11,
            # "xi_psi": 10,
            "eta_coarse": 6,
            "xi_coarse": 5,
            "s_rho": 100,
            "s_w": 101,
        }
        assert dict(ds5.sizes) == {
            "eta_rho": 10,
            "xi_rho": 10,
            "xi_u": 10,
            "eta_v": 10,
            # "eta_psi": 10,
            # "xi_psi": 10,
            "eta_coarse": 5,
            "xi_coarse": 5,
            "s_rho": 100,
            "s_w": 101,
        }
        assert dict(ds8.sizes) == {
            "eta_rho": 11,
            "xi_rho": 10,
            "xi_u": 10,
            "eta_v": 11,
            # "eta_psi": 12,
            # "xi_psi": 10,
            "eta_coarse": 6,
            "xi_coarse": 5,
            "s_rho": 100,
            "s_w": 101,
        }
        assert dict(ds3.sizes) == {
            "eta_rho": 11,
            "xi_rho": 11,
            "xi_u": 11,
            "eta_v": 10,
            # "eta_psi": 11,
            # "xi_psi": 12,
            "eta_coarse": 6,
            "xi_coarse": 6,
            "s_rho": 100,
            "s_w": 101,
        }
        assert dict(ds6.sizes) == {
            "eta_rho": 10,
            "xi_rho": 11,
            "xi_u": 11,
            "eta_v": 10,
            # "eta_psi": 10,
            # "xi_psi": 12,
            "eta_coarse": 5,
            "xi_coarse": 6,
            "s_rho": 100,
            "s_w": 101,
        }
        assert dict(ds9.sizes) == {
            "eta_rho": 11,
            "xi_rho": 11,
            "xi_u": 11,
            "eta_v": 11,
            # "eta_psi": 12,
            # "xi_psi": 12,
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

        with pytest.raises(ValueError, match="cannot be evenly divided"):
            partition(grid.ds, np_eta=10, np_xi=10)

    def test_skip_coarse_dims(self, grid):
        """Test that coarse dimensions remain unchanged when excluded from
        partitioning.
        """
        _, partitioned_datasets = partition(
            grid.ds, np_eta=10, np_xi=10, include_coarse_dims=False
        )
        for ds in partitioned_datasets:
            assert ds.sizes["eta_coarse"] == grid.ds.sizes["eta_coarse"]
            assert ds.sizes["xi_coarse"] == grid.ds.sizes["xi_coarse"]


class TestPartitionGridWithExtraDims:
    @pytest.fixture
    def ds_with_extra_dims(self):
        # Base dims
        eta_rho = 10
        xi_rho = 12
        s_rho = 5
        eta_v = eta_rho - 1
        xi_u = xi_rho - 1

        # Extra dims
        eta_u = eta_rho
        xi_v = xi_rho

        ds = xr.Dataset(
            {
                "zeta": (("eta_rho", "xi_rho"), np.zeros((eta_rho, xi_rho))),
                "u": (("s_rho", "eta_u", "xi_u"), np.zeros((s_rho, eta_u, xi_u))),
                "v": (("s_rho", "eta_v", "xi_v"), np.zeros((s_rho, eta_v, xi_v))),
            },
            coords={
                "eta_rho": np.arange(eta_rho),
                "eta_u": np.arange(eta_u),
                "eta_v": np.arange(eta_v),
                "xi_rho": np.arange(xi_rho),
                "xi_u": np.arange(xi_u),
                "xi_v": np.arange(xi_v),
                "s_rho": np.arange(s_rho),
            },
        )

        return ds

    def test_partition_with_extra_dims(self, ds_with_extra_dims):
        file_numbers, parts = partition(ds_with_extra_dims, np_eta=2, np_xi=2)

        # Test that partitioned datasets contain eta_u and xi_v
        for ds_part in parts:
            assert "eta_u" in ds_part.dims
            assert "xi_v" in ds_part.dims
            assert ds_part.sizes["eta_u"] < ds_with_extra_dims.sizes["eta_u"]
            assert ds_part.sizes["xi_v"] < ds_with_extra_dims.sizes["xi_v"]
            assert ds_part.sizes["eta_u"] > 0
            assert ds_part.sizes["xi_v"] > 0


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

        saved_filenames = partition_netcdf(filepath, np_eta=3, np_xi=5)

        filepath_str = str(filepath.with_suffix(""))
        expected_filepath_list = [
            Path(filepath_str + f".{index:02d}.nc") for index in range(15)
        ]

        assert saved_filenames == expected_filepath_list

        for expected_filepath in expected_filepath_list:
            assert expected_filepath.exists()
            expected_filepath.unlink()

    def test_partition_netcdf_with_output_dir(self, grid, tmp_path):
        # Save the input file
        input_file = tmp_path / "input_grid.nc"
        grid.save(input_file)

        # Create a custom output directory
        output_dir = tmp_path / "custom_output"
        output_dir.mkdir()

        saved_filenames = partition_netcdf(
            input_file, np_eta=3, np_xi=5, output_dir=output_dir
        )

        base_name = input_file.stem  # "input_grid"
        expected_filenames = [output_dir / f"{base_name}.{i:02d}.nc" for i in range(15)]

        assert saved_filenames == expected_filenames

        for f in expected_filenames:
            assert f.exists()
            f.unlink()

    def test_partition_netcdf_multiple_files(self, grid, tmp_path):
        # Create two test input files
        file1 = tmp_path / "grid1.nc"
        file2 = tmp_path / "grid2.nc"
        grid.save(file1)
        grid.save(file2)

        # Run partitioning with 2x2 tiles on both files
        saved_filenames = partition_netcdf([file1, file2], np_eta=3, np_xi=5)

        # Expect 4 tiles per file → 8 total output files
        expected_filepaths = []
        for file in [file1, file2]:
            base = file.with_suffix("")
            expected_filepaths += [Path(f"{base}.{i:02d}.nc") for i in range(15)]

        assert len(saved_filenames) == 30
        assert saved_filenames == expected_filepaths

        for path in expected_filepaths:
            assert path.exists()
            path.unlink()
