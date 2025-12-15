import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from roms_tools import Grid
from roms_tools.datasets.download import download_test_data
from roms_tools.datasets.roms_dataset import ROMSDataset, choose_subdomain
from roms_tools.setup.utils import get_target_coords

try:
    import xesmf  # type: ignore
except ImportError:
    xesmf = None


@pytest.fixture
def roms_dataset_from_restart_file(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # Single file
    return ROMSDataset(
        grid=grid,
        path=Path(download_test_data("eastpac25km_rst.19980106000000.nc")),
        use_dask=use_dask,
    )


@pytest.fixture
def roms_dataset_from_restart_file_adjusted_for_zeta(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # Single file
    return ROMSDataset(
        grid=grid,
        path=Path(download_test_data("eastpac25km_rst.19980106000000.nc")),
        adjust_depth_for_sea_surface_height=True,
        use_dask=use_dask,
    )


@pytest.fixture
def roms_dataset_from_restart_file_with_straddling_grid(use_dask):
    # Make fake grid that straddles the dateline and that has consistent sizes with test data below
    grid = Grid(
        nx=8, ny=13, center_lon=0, center_lat=60, rot=32, size_x=244, size_y=365
    )

    return ROMSDataset(
        grid=grid,
        path=Path(download_test_data("eastpac25km_rst.19980106000000.nc")),
        use_dask=use_dask,
    )


@pytest.mark.parametrize(
    "roms_dataset_fixture",
    [
        "roms_dataset_from_restart_file",
        "roms_dataset_from_restart_file_adjusted_for_zeta",
        "roms_dataset_from_restart_file_with_straddling_grid",
    ],
)
def test_load_model_output_file(roms_dataset_fixture, request):
    roms_dataset = request.getfixturevalue(roms_dataset_fixture)

    assert isinstance(roms_dataset.ds, xr.Dataset)


@pytest.fixture
def roms_dataset_from_two_restart_files(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # List of files
    file1 = Path(download_test_data("eastpac25km_rst.19980106000000.nc"))
    file2 = Path(download_test_data("eastpac25km_rst.19980126000000.nc"))
    return ROMSDataset(grid=grid, path=[file1, file2], use_dask=use_dask)


def test_load_model_output_file_list(roms_dataset_from_two_restart_files):
    assert isinstance(roms_dataset_from_two_restart_files.ds, xr.Dataset)


def test_load_model_output_with_wildcard(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # Download at least two files, so these will be found within the pooch directory
    Path(download_test_data("eastpac25km_rst.19980106000000.nc"))
    Path(download_test_data("eastpac25km_rst.19980126000000.nc"))
    directory = Path(
        os.path.dirname(download_test_data("eastpac25km_rst.19980106000000.nc"))
    )

    output = ROMSDataset(grid=grid, path=directory / "*rst*.nc", use_dask=use_dask)
    assert isinstance(output.ds, xr.Dataset)


def test_invalid_path(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        ROMSDataset(
            grid=grid,
            path=Path("/path/to/nonexistent/file.nc"),
            use_dask=use_dask,
        )


def test_check_consistency_data_grid(use_dask):
    grid_params = {
        "nx": 5,
        "ny": 5,
        "center_lon": -128,
        "center_lat": 9,
        "size_x": 100,
        "size_y": 100,
    }
    grid = Grid(**grid_params)

    with pytest.raises(ValueError, match="Inconsistent dataset dimensions"):
        ROMSDataset(
            grid=grid,
            path=Path(download_test_data("eastpac25km_rst.19980106000000.nc")),
            use_dask=use_dask,
        )


def test_set_correct_model_reference_date(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    output = ROMSDataset(
        grid=grid,
        path=Path(download_test_data("eastpac25km_rst.19980106000000.nc")),
        use_dask=use_dask,
    )
    assert output.model_reference_date == datetime(1995, 1, 1)


def test_model_reference_date_mismatch(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # Create a ROMSDataset with a specified model_reference_date
    model_ref_date = datetime(2020, 1, 1)
    with pytest.raises(
        ValueError, match="Mismatch between `self.model_reference_date`"
    ):
        ROMSDataset(
            grid=grid,
            path=Path(download_test_data("eastpac25km_rst.19980106000000.nc")),
            model_reference_date=model_ref_date,
            use_dask=use_dask,
        )


def test_model_reference_date_no_metadata(use_dask, tmp_path, caplog):
    # Helper function to handle the test logic for cases where metadata is missing or invalid
    def test_no_metadata(faulty_ocean_time_attr, expected_exception, log_message=None):
        ds = xr.open_dataset(fname)
        ds["ocean_time"].attrs = faulty_ocean_time_attr

        # Write modified dataset to a new file
        fname_mod = tmp_path / "eastpac25km_rst.19980106000000_without_metadata.nc"
        ds.to_netcdf(fname_mod)

        # Test case 1: Expecting a ValueError when metadata is missing or invalid
        with pytest.raises(
            expected_exception,
            match="Model reference date could not be inferred from the metadata",
        ):
            ROMSDataset(grid=grid, path=fname_mod, use_dask=use_dask)

        # Test case 2: When a model reference date is explicitly set, verify the warning
        with caplog.at_level(logging.WARNING):
            ROMSDataset(
                grid=grid,
                path=fname_mod,
                model_reference_date=datetime(1995, 1, 1),
                use_dask=use_dask,
            )

            if log_message:
                # Verify the warning message in the log
                assert log_message in caplog.text

        fname_mod.unlink()

    # Load grid and test data
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)
    fname = download_test_data("eastpac25km_rst.19980106000000.nc")

    # Test 1: Ocean time attribute 'long_name' is missing
    test_no_metadata({}, ValueError)

    # Test 2: Ocean time attribute 'long_name' contains invalid information
    test_no_metadata(
        {"long_name": "some random text"},
        ValueError,
        "Could not infer the model reference date from the metadata.",
    )


def test_compute_depth_coordinates(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)
    fname_restart1 = Path(download_test_data("eastpac25km_rst.19980106000000.nc"))

    for adjust_depth_for_sea_surface_height in [True, False]:
        output = ROMSDataset(
            grid=grid,
            path=fname_restart1,
            use_dask=use_dask,
            adjust_depth_for_sea_surface_height=adjust_depth_for_sea_surface_height,
        )

        # Before calling get_vertical_coordinates, check if the dataset doesn't already have depth coordinates
        assert "layer_depth_rho" not in output.ds_depth_coords.data_vars

        # Call the method to get vertical coordinates
        output._get_depth_coordinates(depth_type="layer")

        # Check if the depth coordinates were added
        assert "layer_depth_rho" in output.ds_depth_coords.data_vars


def test_missing_zeta_gets_raised(use_dask):
    """Test that a ValueError is raised when `zeta` is missing from the dataset and
    `adjust_depth_for_sea_surface_height` is enabled.
    """
    # Load the grid
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # Load the ROMS output
    fname_restart1 = Path(download_test_data("eastpac25km_rst.19980106000000.nc"))
    roms_output = ROMSDataset(
        grid=grid,
        path=fname_restart1,
        use_dask=use_dask,
        adjust_depth_for_sea_surface_height=True,
    )

    # Remove `zeta` from the dataset
    object.__setattr__(
        roms_output, "ds", roms_output.ds.drop_vars("zeta", errors="ignore")
    )

    # Expect ValueError when calling `_get_depth_coordinates`
    with pytest.raises(
        ValueError,
        match="`zeta` is required in provided ROMS output when `adjust_depth_for_sea_surface_height` is enabled.",
    ):
        roms_output._get_depth_coordinates()


def test_check_vertical_coordinate_mismatch(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    fname_restart1 = Path(download_test_data("eastpac25km_rst.19980106000000.nc"))
    output = ROMSDataset(grid=grid, path=fname_restart1, use_dask=use_dask)

    # create a mock dataset with inconsistent vertical coordinate parameters
    ds_mock = output.ds.copy()

    # Modify one of the vertical coordinate attributes to cause a mismatch
    ds_mock.attrs["theta_s"] = 999

    # Check if ValueError is raised due to mismatch
    with pytest.raises(ValueError, match="theta_s from grid"):
        output._check_vertical_coordinate(ds_mock)

    # create a mock dataset with inconsistent vertical coordinate parameters
    ds_mock = output.ds.copy()

    # Modify one of the vertical coordinate attributes to cause a mismatch
    ds_mock.attrs["Cs_w"] = ds_mock.attrs["Cs_w"] + 0.01

    # Check if ValueError is raised due to mismatch
    with pytest.raises(ValueError, match="Cs_w from grid"):
        output._check_vertical_coordinate(ds_mock)


def test_that_coordinates_and_masks_are_added(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    fname_restart1 = Path(download_test_data("eastpac25km_rst.19980106000000.nc"))
    output = ROMSDataset(grid=grid, path=fname_restart1, use_dask=use_dask)

    assert "time" in output.ds.coords
    assert "lat_rho" in output.ds.coords
    assert "lon_rho" in output.ds.coords
    assert "mask_rho" in output.ds
    assert "mask_u" in output.ds
    assert "mask_v" in output.ds


# Test applying lateral fill


def make_roms_dataset(ds, grid):
    roms = ROMSDataset.__new__(ROMSDataset)
    roms.ds = ds
    roms.grid = grid
    return roms


def test_apply_lateral_fill_rho_success():
    grid = Grid(
        nx=10, ny=10, size_x=2500, size_y=3000, center_lon=-30, center_lat=57, rot=-20
    )

    ds = xr.Dataset()
    ds["field"] = 5 * grid.ds.mask_rho.copy()
    ds["mask_rho"] = grid.ds.mask_rho.copy()

    roms = make_roms_dataset(ds, grid)

    # land exists initially
    assert (roms.ds["field"] == 0).any()

    roms.apply_lateral_fill()

    assert (roms.ds["field"] == 5).all()


def test_apply_lateral_fill_rho_missing_mask_raises():
    grid = Grid(
        nx=10, ny=10, size_x=2500, size_y=3000, center_lon=-30, center_lat=57, rot=-20
    )

    ds = xr.Dataset()
    ds["field"] = grid.ds.mask_rho.copy()

    roms = make_roms_dataset(ds, grid)

    with pytest.raises(ValueError, match="mask_rho"):
        roms.apply_lateral_fill()


def test_apply_lateral_fill_u_success():
    grid = Grid(
        nx=10, ny=10, size_x=2500, size_y=3000, center_lon=-30, center_lat=57, rot=-20
    )

    ds = xr.Dataset()
    ds["u_field"] = 7 * grid.ds.mask_u.copy()
    ds["mask_u"] = grid.ds.mask_u.copy()

    roms = make_roms_dataset(ds, grid)

    assert (roms.ds["u_field"] == 0).any()

    roms.apply_lateral_fill()

    assert (roms.ds["u_field"] == 7).all()


def test_apply_lateral_fill_u_missing_mask_raises():
    grid = Grid(
        nx=10, ny=10, size_x=2500, size_y=3000, center_lon=-30, center_lat=57, rot=-20
    )

    ds = xr.Dataset()
    ds["u_field"] = grid.ds.mask_u.copy()
    ds["mask_rho"] = grid.ds.mask_rho.copy()

    roms = make_roms_dataset(ds, grid)

    with pytest.raises(ValueError, match="mask_u"):
        roms.apply_lateral_fill()


def test_apply_lateral_fill_v_success():
    grid = Grid(
        nx=10, ny=10, size_x=2500, size_y=3000, center_lon=-30, center_lat=57, rot=-20
    )

    ds = xr.Dataset()
    ds["v_field"] = 9 * grid.ds.mask_v.copy()
    ds["mask_v"] = grid.ds.mask_v.copy()

    roms = make_roms_dataset(ds, grid)

    assert (roms.ds["v_field"] == 0).any()

    roms.apply_lateral_fill()

    assert (roms.ds["v_field"] == 9).all()


def test_apply_lateral_fill_v_missing_mask_raises():
    grid = Grid(
        nx=10, ny=10, size_x=2500, size_y=3000, center_lon=-30, center_lat=57, rot=-20
    )

    ds = xr.Dataset()
    ds["v_field"] = grid.ds.mask_v.copy()
    ds["mask_rho"] = grid.ds.mask_rho.copy()

    roms = make_roms_dataset(ds, grid)

    with pytest.raises(ValueError, match="mask_v"):
        roms.apply_lateral_fill()


def test_apply_lateral_fill_mixed_grids():
    grid = Grid(
        nx=10, ny=10, size_x=2500, size_y=3000, center_lon=-30, center_lat=57, rot=-20
    )

    ds = xr.Dataset()
    ds["rho"] = 1 * grid.ds.mask_rho.copy()
    ds["u"] = 2 * grid.ds.mask_u.copy()
    ds["v"] = 3 * grid.ds.mask_v.copy()

    ds["mask_rho"] = grid.ds.mask_rho.copy()
    ds["mask_u"] = grid.ds.mask_u.copy()
    ds["mask_v"] = grid.ds.mask_v.copy()

    roms = make_roms_dataset(ds, grid)

    roms.apply_lateral_fill()

    assert (roms.ds["rho"] == 1).all()
    assert (roms.ds["u"] == 2).all()
    assert (roms.ds["v"] == 3).all()


# Test choose_subdomain
grid_cases = [
    # --- SMALL INSIDE WESTERN HEMISPHERE -----------------------------------
    (
        # big grid west, does NOT cross Greenwich
        {
            "nx": 40,
            "ny": 40,
            "size_x": 4000,
            "size_y": 4000,
            "center_lon": -30,
            "center_lat": 50,
            "rot": 0,
        },
        # small grid fully west
        {
            "nx": 10,
            "ny": 10,
            "size_x": 500,
            "size_y": 500,
            "center_lon": -20,
            "center_lat": 50,
            "rot": 0,
        },
    ),
    # --- SMALL INSIDE EASTERN HEMISPHERE ----------------------------------
    (
        {
            "nx": 40,
            "ny": 40,
            "size_x": 4000,
            "size_y": 4000,
            "center_lon": 10,
            "center_lat": 45,
            "rot": 0,
        },
        {
            "nx": 10,
            "ny": 10,
            "size_x": 500,
            "size_y": 500,
            "center_lon": 20,
            "center_lat": 45,
            "rot": 0,
        },
    ),
    # --- SMALL GRID STRADDLING GREENWICH ----------------------------------
    (
        # big grid centered slightly west
        {
            "nx": 40,
            "ny": 40,
            "size_x": 5000,
            "size_y": 4000,
            "center_lon": -5,
            "center_lat": 45,
            "rot": 0,
        },
        # small grid crosses 0°
        {
            "nx": 10,
            "ny": 10,
            "size_x": 600,
            "size_y": 600,
            "center_lon": 0,
            "center_lat": 45,
            "rot": 0,
        },
    ),
    # --- LARGE GRID CROSSES GREENWICH, SMALL IS WEST -----------------------
    (
        {
            "nx": 50,
            "ny": 50,
            "size_x": 8000,
            "size_y": 5000,
            "center_lon": -2,
            "center_lat": 40,
            "rot": 0,
        },
        {
            "nx": 10,
            "ny": 10,
            "size_x": 400,
            "size_y": 400,
            "center_lon": -10,
            "center_lat": 40,
            "rot": 0,
        },
    ),
    # --- LARGE GRID CROSSES GREENWICH, SMALL IS EAST -----------------------
    (
        {
            "nx": 50,
            "ny": 50,
            "size_x": 8000,
            "size_y": 5000,
            "center_lon": -2,
            "center_lat": 40,
            "rot": 0,
        },
        {
            "nx": 10,
            "ny": 10,
            "size_x": 400,
            "size_y": 400,
            "center_lon": 8,
            "center_lat": 40,
            "rot": 0,
        },
    ),
    # --- BOTH GRIDS CROSS GREENWICH ---------------------------------------
    (
        {
            "nx": 60,
            "ny": 60,
            "size_x": 9000,
            "size_y": 6000,
            "center_lon": 1,
            "center_lat": 48,
            "rot": 0,
        },
        {
            "nx": 12,
            "ny": 12,
            "size_x": 900,
            "size_y": 900,
            "center_lon": -1,
            "center_lat": 48,
            "rot": 0,
        },
    ),
]


# ----------------------------------------------------------------------
# THE TEST
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "big_params, small_params",
    grid_cases,
)
def test_choose_subdomain_with(big_params, small_params):
    """Tests choose_subdomain() for rho, u, and v points over various grid configurations."""
    # --- build big grid ---
    big = Grid(**big_params)
    ds = xr.Dataset()
    ds = ds.assign_coords(
        {
            "lat_rho": big.ds.lat_rho,
            "lon_rho": big.ds.lon_rho,
            "lat_u": big.ds.lat_u,
            "lon_u": big.ds.lon_u,
            "lat_v": big.ds.lat_v,
            "lon_v": big.ds.lon_v,
        }
    )

    # simple fields for testing
    ds["field_rho"] = (
        ("eta_rho", "xi_rho"),
        (big.ds["eta_rho"] * big.ds["xi_rho"]).values,
    )
    ds["field_u"] = (("eta_rho", "xi_u"), np.random.rand(*big.ds["lat_u"].shape))
    ds["field_v"] = (("eta_v", "xi_rho"), np.random.rand(*big.ds["lat_v"].shape))

    # --- build small grid ---
    small = Grid(**small_params)
    target_coords = get_target_coords(small)

    # --- apply function ---
    sub = choose_subdomain(ds, big.ds, target_coords, buffer_points=1)

    # --- rho tests ---
    assert sub.lat_rho.shape[0] <= ds.lat_rho.shape[0]
    assert sub.lat_rho.shape[1] <= ds.lat_rho.shape[1]
    assert float(sub.lat_rho.min()) >= float(ds.lat_rho.min()) - 1e-6
    assert float(sub.lat_rho.max()) <= float(ds.lat_rho.max()) + 1e-6
    assert not sub.field_rho.isnull().any()

    # --- u tests ---
    if "lat_u" in sub.coords and "lon_u" in sub.coords:
        assert sub.lat_u.shape[0] <= ds.lat_u.shape[0]
        assert sub.lat_u.shape[1] <= ds.lat_u.shape[1]
        assert float(sub.lat_u.min()) >= float(ds.lat_u.min()) - 1e-6
        assert float(sub.lat_u.max()) <= float(ds.lat_u.max()) + 1e-6
        assert not sub.field_u.isnull().any()

    # --- v tests ---
    if "lat_v" in sub.coords and "lon_v" in sub.coords:
        assert sub.lat_v.shape[0] <= ds.lat_v.shape[0]
        assert sub.lat_v.shape[1] <= ds.lat_v.shape[1]
        assert float(sub.lat_v.min()) >= float(ds.lat_v.min()) - 1e-6
        assert float(sub.lat_v.max()) <= float(ds.lat_v.max()) + 1e-6
        assert not sub.field_v.isnull().any()
