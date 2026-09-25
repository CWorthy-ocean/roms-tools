import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from roms_tools import Grid
from roms_tools.analysis.eta_beta import calculate_eta_beta
from roms_tools.datasets.download import download_test_data


@pytest.fixture
def his_ds(grid):
    shape = (2, 3, grid.ds.sizes["eta_rho"], grid.ds.sizes["xi_rho"])
    dims = ("time", "s_rho", "eta_rho", "xi_rho")
    return xr.Dataset(
        {
            "temp": (dims, np.full(shape, 10.0)),
            "salt": (dims, np.full(shape, 35.0)),
            "ocean_time": (
                "time",
                [0.0, 86400.0],
                {"long_name": "Time since 2000/01/01"},
            ),
        }
    )


@pytest.fixture
def bgc_ds(grid):
    shape = (2, 3, grid.ds.sizes["eta_rho"], grid.ds.sizes["xi_rho"])
    dims = ("time", "s_rho", "eta_rho", "xi_rho")
    return xr.Dataset(
        {
            "ALK_ALT_CO2": (dims, np.full(shape, 2300.0)),
            "DIC_ALT_CO2": (dims, np.full(shape, 2050.0)),
            "PO4": (dims, np.full(shape, 1.0)),
            "SiO3": (dims, np.full(shape, 5.0)),
            "ocean_time": (
                "time",
                [0.0, 86400.0],
                {"long_name": "Time since 2000/01/01"},
            ),
        }
    )


@pytest.fixture
def eta_beta_iceland(use_dask):
    grid = Grid(filename=Path(download_test_data("Iceland_grid.nc")))

    return calculate_eta_beta(
        his_path=Path(download_test_data("Iceland_his.20240710120000.nc")),
        bgc_path=Path(download_test_data("Iceland_bgc.20240710120000.nc")),
        grid=grid,
        use_dask=use_dask,
    )


@pytest.mark.parametrize(
    "time_shift, context",
    [
        (0.0, nullcontext()),
        (3600.0, pytest.raises(ValueError, match="values do not match")),
    ],
)
def test_his_and_bgc_times_match(his_ds, bgc_ds, grid, tmp_path, time_shift, context):
    bgc_ds["ocean_time"] = bgc_ds.ocean_time + time_shift
    his_ds.to_netcdf(tmp_path / "his.nc")
    bgc_ds.to_netcdf(tmp_path / "bgc.nc")

    with context:
        calculate_eta_beta(
            his_path=tmp_path / "his.nc", bgc_path=tmp_path / "bgc.nc", grid=grid
        )


def test_his_and_bgc_number_of_times_match(his_ds, bgc_ds, grid, tmp_path):
    his_ds.to_netcdf(tmp_path / "his.nc")
    bgc_ds.isel(time=[0]).to_netcdf(tmp_path / "bgc.nc")

    with pytest.raises(ValueError, match="different number of time records"):
        calculate_eta_beta(
            his_path=tmp_path / "his.nc", bgc_path=tmp_path / "bgc.nc", grid=grid
        )


@pytest.mark.parametrize(
    "var, source",
    [
        ("temp", "his"),
        ("salt", "his"),
        ("ALK_ALT_CO2", "bgc"),
        ("DIC_ALT_CO2", "bgc"),
        ("PO4", "bgc"),
        ("SiO3", "bgc"),
    ],
)
def test_missing_variables(his_ds, bgc_ds, grid, tmp_path, var, source):
    if source == "his":
        his_ds = his_ds.drop_vars(var)
    else:
        bgc_ds = bgc_ds.drop_vars(var)
    his_ds.to_netcdf(tmp_path / "his.nc")
    bgc_ds.to_netcdf(tmp_path / "bgc.nc")

    with pytest.raises(KeyError, match=f"Missing required variables in {source}"):
        calculate_eta_beta(
            his_path=tmp_path / "his.nc", bgc_path=tmp_path / "bgc.nc", grid=grid
        )


def test_no_nans_on_water_points(eta_beta_iceland):
    water = eta_beta_iceland.grid.ds.mask_rho == 1
    ntimes = eta_beta_iceland.ds.sizes["time"]

    for var in ["eta", "beta"]:
        assert (
            eta_beta_iceland.ds[var].where(water).notnull().sum()
            == water.sum() * ntimes
        )


# this test will not be run by default
# to run it and overwrite the test data, invoke pytest as follows
# pytest --overwrite=eta_beta_iceland
def test_save_results(request):
    overwrite = request.config.getoption("--overwrite")
    if "all" not in overwrite and "eta_beta_iceland" not in overwrite:
        pytest.skip("Skipping overwrite for eta_beta_iceland")

    eta_beta = request.getfixturevalue("eta_beta_iceland")
    # get_test_data_path raises if the zarr doesn't exist yet, so build the path here
    fname = Path(__file__).parents[1] / "test_setup/test_data/eta_beta_iceland.zarr"
    if fname.exists():
        shutil.rmtree(fname)

    eta_beta.ds.to_zarr(fname)


def test_check_results(eta_beta_iceland, get_test_data_path):
    fname = get_test_data_path("eta_beta_iceland")
    expected_ds = xr.open_zarr(fname)

    xr.testing.assert_allclose(eta_beta_iceland.ds, expected_ds, rtol=1.0e-12)
