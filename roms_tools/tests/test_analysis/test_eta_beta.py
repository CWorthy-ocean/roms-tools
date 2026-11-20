import logging
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
def avg_his_ds(his_ds):
    """History output labelled as averaged, as ROMS writes it."""
    his_ds["temp"].attrs["long_name"] = "averaged potential temperature"
    return his_ds


@pytest.fixture
def ic_ds(his_ds, bgc_ds):
    """Initial conditions one day before the first his and bgc record."""
    ds = xr.merge([his_ds, bgc_ds]).isel(time=[0])
    ds["ocean_time"] = ds.ocean_time - 86400.0
    return ds


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


@pytest.mark.parametrize(
    "source, var, long_name",
    [
        ("his", "temp", "averaged potential temperature"),
        ("bgc", "ALK_ALT_CO2", "avg_Alkalinity, Alternative CO2"),
    ],
)
def test_averaged_output_requires_initial_conditions(
    his_ds, bgc_ds, ic_ds, grid, tmp_path, source, var, long_name
):
    ds = his_ds if source == "his" else bgc_ds
    ds[var].attrs["long_name"] = long_name
    his_ds.to_netcdf(tmp_path / "his.nc")
    bgc_ds.to_netcdf(tmp_path / "bgc.nc")
    ic_ds.to_netcdf(tmp_path / "ic.nc")

    with pytest.raises(ValueError, match="averaged output"):
        calculate_eta_beta(
            his_path=tmp_path / "his.nc", bgc_path=tmp_path / "bgc.nc", grid=grid
        )

    calculate_eta_beta(
        his_path=tmp_path / "his.nc",
        bgc_path=tmp_path / "bgc.nc",
        ic_path=tmp_path / "ic.nc",
        grid=grid,
    )


@pytest.mark.parametrize("roms_tools_layout", [True, False])
def test_initial_conditions_are_first_record(
    avg_his_ds, bgc_ds, ic_ds, grid, tmp_path, roms_tools_layout
):
    """The initial conditions record comes first, from a roms-tools initial
    conditions file (``ocean_time`` dimension) or a ROMS restart file (``time``
    dimension).
    """
    if roms_tools_layout:
        ic_ds = ic_ds.set_coords("ocean_time").swap_dims({"time": "ocean_time"})
    avg_his_ds.to_netcdf(tmp_path / "his.nc")
    bgc_ds.to_netcdf(tmp_path / "bgc.nc")
    ic_ds.to_netcdf(tmp_path / "ic.nc")

    eta_beta = calculate_eta_beta(
        his_path=tmp_path / "his.nc",
        bgc_path=tmp_path / "bgc.nc",
        ic_path=tmp_path / "ic.nc",
        grid=grid,
    )

    np.testing.assert_allclose(
        eta_beta.ds["ocean_time"].values, [-86400.0, 0.0, 86400.0]
    )
    np.testing.assert_allclose(eta_beta.ds["ddic_dco2_time"].values, [-1.0, 0.0, 1.0])


@pytest.mark.parametrize("ic_time, warns", [(-86400.0, False), (-43200.0, True)])
def test_initial_conditions_time_step_warning(
    avg_his_ds, bgc_ds, ic_ds, grid, tmp_path, caplog, ic_time, warns
):
    ic_ds["ocean_time"] = xr.full_like(ic_ds.ocean_time, ic_time)
    avg_his_ds.to_netcdf(tmp_path / "his.nc")
    bgc_ds.to_netcdf(tmp_path / "bgc.nc")
    ic_ds.to_netcdf(tmp_path / "ic.nc")

    with caplog.at_level(logging.WARNING):
        calculate_eta_beta(
            his_path=tmp_path / "his.nc",
            bgc_path=tmp_path / "bgc.nc",
            ic_path=tmp_path / "ic.nc",
            grid=grid,
        )

    assert ("differs from the time between records" in caplog.text) == warns


def test_exact_restart_uses_second_record(avg_his_ds, bgc_ds, ic_ds, grid, tmp_path):
    earlier = ic_ds.assign(ocean_time=ic_ds.ocean_time - 86400.0, temp=ic_ds.temp + 5)
    avg_his_ds.to_netcdf(tmp_path / "his.nc")
    bgc_ds.to_netcdf(tmp_path / "bgc.nc")
    xr.concat([earlier, ic_ds], dim="time").to_netcdf(tmp_path / "ic.nc")

    eta_beta = calculate_eta_beta(
        his_path=tmp_path / "his.nc",
        bgc_path=tmp_path / "bgc.nc",
        ic_path=tmp_path / "ic.nc",
        grid=grid,
    )

    np.testing.assert_allclose(
        eta_beta.ds["ocean_time"].values, [-86400.0, 0.0, 86400.0]
    )
    surface = eta_beta.ds_surface_control.isel(time=0)
    assert (surface.temp == ic_ds.temp.isel(time=0, s_rho=-1)).all()


def test_initial_conditions_more_than_two_records(
    avg_his_ds, bgc_ds, ic_ds, grid, tmp_path
):
    avg_his_ds.to_netcdf(tmp_path / "his.nc")
    bgc_ds.to_netcdf(tmp_path / "bgc.nc")
    xr.concat([ic_ds, ic_ds, ic_ds], dim="time").to_netcdf(tmp_path / "ic.nc")

    with pytest.raises(ValueError, match="exact restart file, but it has 3"):
        calculate_eta_beta(
            his_path=tmp_path / "his.nc",
            bgc_path=tmp_path / "bgc.nc",
            ic_path=tmp_path / "ic.nc",
            grid=grid,
        )


def test_initial_conditions_unused_for_instantaneous_output(
    his_ds, bgc_ds, ic_ds, grid, tmp_path, caplog
):
    his_ds.to_netcdf(tmp_path / "his.nc")
    bgc_ds.to_netcdf(tmp_path / "bgc.nc")
    ic_ds.to_netcdf(tmp_path / "ic.nc")

    with caplog.at_level(logging.WARNING):
        eta_beta = calculate_eta_beta(
            his_path=tmp_path / "his.nc",
            bgc_path=tmp_path / "bgc.nc",
            ic_path=tmp_path / "ic.nc",
            grid=grid,
        )

    assert "`ic_path` was not used" in caplog.text
    np.testing.assert_allclose(eta_beta.ds["ocean_time"].values, [0.0, 86400.0])


def test_zero_on_land_positive_on_water(eta_beta_iceland):
    """No NaN anywhere: 0 over land (as ROMS writes), positive on water points.

    A failed solve on a water point would also be 0, so positivity there checks
    that the Iceland case has none.
    """
    water = eta_beta_iceland.grid.ds.mask_rho == 1

    for var in ["ddic_dco2", "ddic_dalk"]:
        da = eta_beta_iceland.ds[var]
        assert da.notnull().all(), var
        assert (da.where(~water) == 0).sum() == (~water).sum() * da.sizes["time"], var
        assert (da.where(water) > 0).sum() == water.sum() * da.sizes["time"], var


def test_saved_file_matches_roms_forcing_names(his_ds, bgc_ds, grid, tmp_path):
    """Saved files carry the variables ROMS reads (surf_flux.F90).

    ROMS looks up ``ddic_dco2`` and ``ddic_dalk`` with their own time variables
    ``ddic_dco2_time`` and ``ddic_dalk_time``, in days since the model
    reference date.
    """
    his_ds.to_netcdf(tmp_path / "his.nc")
    bgc_ds.to_netcdf(tmp_path / "bgc.nc")
    eta_beta = calculate_eta_beta(
        his_path=tmp_path / "his.nc", bgc_path=tmp_path / "bgc.nc", grid=grid
    )

    (fname,) = eta_beta.save(tmp_path / "out" / "carbonate_sensitivity")
    with xr.open_dataset(fname, decode_times=False) as ds:
        for var in ("ddic_dco2", "ddic_dalk"):
            assert ds[var].dims == ("time", "eta_rho", "xi_rho")
            assert ds[f"{var}_time"].dims == ("time",)
            assert ds[f"{var}_time"].attrs["units"] == "days"
            np.testing.assert_allclose(ds[f"{var}_time"].values, [0.0, 1.0])


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
    # ddic_*_time carry units "days", which xarray would decode to timedelta64
    expected_ds = xr.open_zarr(fname, decode_timedelta=False)

    xr.testing.assert_allclose(eta_beta_iceland.ds, expected_ds, rtol=1.0e-12)
