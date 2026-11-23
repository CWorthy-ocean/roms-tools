import logging
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from roms_tools import Grid
from roms_tools.analysis import carbonate_sensitivity
from roms_tools.analysis.carbonate_sensitivity import CarbonateSensitivity
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


# Averaged fixtures: daily windows from the start of the run at 0, model step DT
DT = 3600.0
AVG_BEGIN = np.array([0.0, 86400.0])
AVG_END = AVG_BEGIN + 86400.0
AVG_HIS_TIME = 0.5 * (AVG_BEGIN + AVG_END + DT)  # mean of the window's step times


def _as_averaged(ds, var, long_name, ocean_time):
    """Label ``ds`` as averaged output, with window bounds as ucla-roms 0.16.0 writes."""
    ds = ds.copy()
    ds[var].attrs["long_name"] = long_name
    ds["ocean_time"] = ("time", ocean_time, ds.ocean_time.attrs)
    ds["avg_begin_time"] = ("time", AVG_BEGIN)
    ds["avg_end_time"] = ("time", AVG_END)
    return ds


@pytest.fixture
def avg_his_ds(his_ds):
    """History averages, stamped at the mean of the window's step times."""
    return _as_averaged(his_ds, "temp", "averaged potential temperature", AVG_HIS_TIME)


@pytest.fixture
def avg_bgc_ds(bgc_ds):
    """BGC averages, stamped at the window end."""
    return _as_averaged(
        bgc_ds, "ALK_ALT_CO2", "avg_Alkalinity, Alternative CO2", AVG_END
    )


@pytest.fixture
def ic_ds(his_ds, bgc_ds):
    """Initial conditions at the start of the run, where the first window begins."""
    ds = xr.merge([his_ds, bgc_ds]).isel(time=[0])
    ds["ocean_time"] = xr.full_like(ds.ocean_time, AVG_BEGIN[0])
    return ds


@pytest.fixture
def eta_beta_iceland(use_dask):
    grid = Grid(filename=Path(download_test_data("Iceland_grid.nc")))

    return CarbonateSensitivity(
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
        CarbonateSensitivity(
            his_path=tmp_path / "his.nc", bgc_path=tmp_path / "bgc.nc", grid=grid
        )


def test_his_and_bgc_number_of_times_match(his_ds, bgc_ds, grid, tmp_path):
    his_ds.to_netcdf(tmp_path / "his.nc")
    bgc_ds.isel(time=[0]).to_netcdf(tmp_path / "bgc.nc")

    with pytest.raises(ValueError, match="different number of time records"):
        CarbonateSensitivity(
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
        CarbonateSensitivity(
            his_path=tmp_path / "his.nc", bgc_path=tmp_path / "bgc.nc", grid=grid
        )


def test_averaged_output_requires_initial_conditions(
    avg_his_ds, avg_bgc_ds, ic_ds, grid, tmp_path
):
    avg_his_ds.to_netcdf(tmp_path / "his.nc")
    avg_bgc_ds.to_netcdf(tmp_path / "bgc.nc")
    ic_ds.to_netcdf(tmp_path / "ic.nc")

    with pytest.raises(ValueError, match="averaged output"):
        CarbonateSensitivity(
            his_path=tmp_path / "his.nc", bgc_path=tmp_path / "bgc.nc", grid=grid
        )

    CarbonateSensitivity(
        his_path=tmp_path / "his.nc",
        bgc_path=tmp_path / "bgc.nc",
        ic_path=tmp_path / "ic.nc",
        grid=grid,
    )


@pytest.mark.parametrize(
    "source, change, match",
    [
        # Both bounds shifted by a day: same window length, different windows
        ("bgc", {"avg_begin_time": 86400.0, "avg_end_time": 86400.0}, "windows"),
        # A two-day window ending where the his window ends
        ("bgc", {"avg_begin_time": -86400.0}, "windows"),
        # Stamps in the next window, or at the start of their own (excluded)
        ("his", {"ocean_time": 86400.0}, "his `ocean_time` at record 0"),
        ("bgc", {"ocean_time": 86400.0}, "bgc `ocean_time` at record 0"),
        ("bgc", {"ocean_time": -86400.0}, "bgc `ocean_time` at record 0"),
    ],
)
def test_averaged_records_must_match(
    avg_his_ds, avg_bgc_ds, ic_ds, grid, tmp_path, source, change, match
):
    """Averaged his and bgc records must cover the same windows (the same
    ``avg_begin_time`` and ``avg_end_time``), and each record's stamp must fall
    within its window.
    """
    ds = avg_his_ds if source == "his" else avg_bgc_ds
    for var, shift in change.items():
        ds[var] = ds[var] + shift
    avg_his_ds.to_netcdf(tmp_path / "his.nc")
    avg_bgc_ds.to_netcdf(tmp_path / "bgc.nc")
    ic_ds.to_netcdf(tmp_path / "ic.nc")

    with pytest.raises(ValueError, match=match):
        CarbonateSensitivity(
            his_path=tmp_path / "his.nc",
            bgc_path=tmp_path / "bgc.nc",
            ic_path=tmp_path / "ic.nc",
            grid=grid,
        )


def test_averaged_records_from_several_files(
    avg_his_ds, avg_bgc_ds, ic_ds, grid, tmp_path, use_dask
):
    """The window bounds survive loading one record per file by wildcard."""
    for k in range(avg_his_ds.sizes["time"]):
        avg_his_ds.isel(time=[k]).to_netcdf(tmp_path / f"run_avg.{k}.nc")
        avg_bgc_ds.isel(time=[k]).to_netcdf(tmp_path / f"run_bgc_avg.{k}.nc")
    ic_ds.to_netcdf(tmp_path / "ic.nc")

    eta_beta = CarbonateSensitivity(
        his_path=str(tmp_path / "run_avg.*.nc"),
        bgc_path=str(tmp_path / "run_bgc_avg.*.nc"),
        ic_path=tmp_path / "ic.nc",
        grid=grid,
        use_dask=use_dask,
    )

    np.testing.assert_allclose(
        eta_beta.ds["ocean_time"].values, [AVG_BEGIN[0], *AVG_END]
    )


@pytest.mark.parametrize("roms_tools_layout", [True, False])
def test_initial_conditions_are_first_record(
    avg_his_ds, avg_bgc_ds, ic_ds, grid, tmp_path, roms_tools_layout
):
    """The initial conditions record comes first, from a roms-tools initial
    conditions file (``ocean_time`` dimension) or a ROMS restart file (``time``
    dimension). The averaged records are stamped at the window end, like the
    bgc records, not at the his stamps near the window middle.
    """
    if roms_tools_layout:
        ic_ds = ic_ds.set_coords("ocean_time").swap_dims({"time": "ocean_time"})
    avg_his_ds.to_netcdf(tmp_path / "his.nc")
    avg_bgc_ds.to_netcdf(tmp_path / "bgc.nc")
    ic_ds.to_netcdf(tmp_path / "ic.nc")

    eta_beta = CarbonateSensitivity(
        his_path=tmp_path / "his.nc",
        bgc_path=tmp_path / "bgc.nc",
        ic_path=tmp_path / "ic.nc",
        grid=grid,
    )

    expected = np.concatenate([AVG_BEGIN[:1], AVG_END])
    np.testing.assert_allclose(eta_beta.ds["ocean_time"].values, expected)
    np.testing.assert_allclose(eta_beta.ds["ddic_dco2_time"].values, expected / 86400.0)


@pytest.mark.parametrize(
    "ic_time, warns",
    [(0.0, False), (43200.0, False), (-86400.0, True)],
)
def test_initial_conditions_time_step_warning(
    avg_his_ds, avg_bgc_ds, ic_ds, grid, tmp_path, caplog, ic_time, warns
):
    ic_ds["ocean_time"] = xr.full_like(ic_ds.ocean_time, ic_time)
    avg_his_ds.to_netcdf(tmp_path / "his.nc")
    avg_bgc_ds.to_netcdf(tmp_path / "bgc.nc")
    ic_ds.to_netcdf(tmp_path / "ic.nc")

    with caplog.at_level(logging.WARNING):
        CarbonateSensitivity(
            his_path=tmp_path / "his.nc",
            bgc_path=tmp_path / "bgc.nc",
            ic_path=tmp_path / "ic.nc",
            grid=grid,
        )

    assert ("is longer than the time between records" in caplog.text) == warns


def test_exact_restart_uses_second_record(
    avg_his_ds, avg_bgc_ds, ic_ds, grid, tmp_path
):
    earlier = ic_ds.assign(ocean_time=ic_ds.ocean_time - 86400.0, temp=ic_ds.temp + 5)
    avg_his_ds.to_netcdf(tmp_path / "his.nc")
    avg_bgc_ds.to_netcdf(tmp_path / "bgc.nc")
    xr.concat([earlier, ic_ds], dim="time").to_netcdf(tmp_path / "ic.nc")

    eta_beta = CarbonateSensitivity(
        his_path=tmp_path / "his.nc",
        bgc_path=tmp_path / "bgc.nc",
        ic_path=tmp_path / "ic.nc",
        grid=grid,
    )

    np.testing.assert_allclose(
        eta_beta.ds["ocean_time"].values, [AVG_BEGIN[0], *AVG_END]
    )
    surface = eta_beta.ds_surface_control.isel(time=0)
    assert (surface.temp == ic_ds.temp.isel(time=0, s_rho=-1)).all()


def test_initial_conditions_more_than_two_records(
    avg_his_ds, avg_bgc_ds, ic_ds, grid, tmp_path
):
    avg_his_ds.to_netcdf(tmp_path / "his.nc")
    avg_bgc_ds.to_netcdf(tmp_path / "bgc.nc")
    xr.concat([ic_ds, ic_ds, ic_ds], dim="time").to_netcdf(tmp_path / "ic.nc")

    with pytest.raises(ValueError, match="exact restart file, but it has 3"):
        CarbonateSensitivity(
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
        eta_beta = CarbonateSensitivity(
            his_path=tmp_path / "his.nc",
            bgc_path=tmp_path / "bgc.nc",
            ic_path=tmp_path / "ic.nc",
            grid=grid,
        )

    assert "`ic_path` was not used" in caplog.text
    np.testing.assert_allclose(eta_beta.ds["ocean_time"].values, [0.0, 86400.0])


@pytest.mark.parametrize(
    "source, var, value",
    [("his", "salt", 0.05), ("bgc", "ALK_ALT_CO2", 5.0), ("bgc", "DIC_ALT_CO2", 5.0)],
)
def test_below_floors_are_zero(his_ds, bgc_ds, grid, tmp_path, source, var, value):
    """Cells below MARBL's salinity, ALK, or DIC floor are skipped and left at 0."""
    ds = his_ds if source == "his" else bgc_ds
    ds[var] = xr.full_like(ds[var], value)
    his_ds.to_netcdf(tmp_path / "his.nc")
    bgc_ds.to_netcdf(tmp_path / "bgc.nc")

    cs = CarbonateSensitivity(
        his_path=tmp_path / "his.nc", bgc_path=tmp_path / "bgc.nc", grid=grid
    )

    for name in ("ddic_dco2", "ddic_dalk"):
        assert (cs.ds[name] == 0).all(), name


def test_dask_ds_is_lazy_and_saves_same_files(his_ds, bgc_ds, grid, tmp_path):
    """With use_dask=True, ds stays lazy, one record per chunk, until save."""
    his_ds.to_netcdf(tmp_path / "his.nc")
    bgc_ds.to_netcdf(tmp_path / "bgc.nc")
    paths = {"his_path": tmp_path / "his.nc", "bgc_path": tmp_path / "bgc.nc"}

    eager = CarbonateSensitivity(**paths, grid=grid)
    lazy = CarbonateSensitivity(**paths, grid=grid, use_dask=True)

    assert lazy.ds["ddic_dco2"].chunks[0] == (1, 1)

    (eager_file,) = eager.save(tmp_path / "eager" / "carbonate_sensitivity")
    (lazy_file,) = lazy.save(tmp_path / "lazy" / "carbonate_sensitivity")
    with (
        xr.open_dataset(eager_file, decode_timedelta=False) as expected,
        xr.open_dataset(lazy_file, decode_timedelta=False) as actual,
    ):
        xr.testing.assert_identical(actual, expected)


def test_cells_split_into_blocks(monkeypatch, get_test_data_path):
    """Splitting a record's cells over several PyCO2SYS calls gives the same result."""
    monkeypatch.setattr(carbonate_sensitivity, "MAX_CELLS_PER_CALL", 1000)

    cs = CarbonateSensitivity(
        his_path=Path(download_test_data("Iceland_his.20240710120000.nc")),
        bgc_path=Path(download_test_data("Iceland_bgc.20240710120000.nc")),
        grid=Grid(filename=Path(download_test_data("Iceland_grid.nc"))),
    )

    expected_ds = xr.open_zarr(
        get_test_data_path("eta_beta_iceland"), decode_timedelta=False
    )
    xr.testing.assert_allclose(cs.ds, expected_ds, rtol=1.0e-12)


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
    eta_beta = CarbonateSensitivity(
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
