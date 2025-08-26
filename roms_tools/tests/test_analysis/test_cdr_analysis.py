import logging

import numpy as np
import pytest
import xarray as xr

from roms_tools.analysis.cdr_analysis import (
    _validate_source,
    _validate_uptake_efficiency,
    compute_cdr_metrics,
)


@pytest.fixture
def minimal_grid_ds():
    """Minimal grid dataset with uniform spacing."""
    return xr.Dataset(
        {
            "pm": (("eta_rho", "xi_rho"), np.ones((2, 2))),
            "pn": (("eta_rho", "xi_rho"), np.ones((2, 2))),
        }
    )


@pytest.fixture
def minimal_ds():
    """Minimal ROMS dataset with required variables and dimensions."""
    time = np.arange(3)
    s_rho = np.arange(1)
    eta_rho = np.arange(2)
    xi_rho = np.arange(2)

    return xr.Dataset(
        {
            "avg_begin_time": ("time", time),
            "avg_end_time": ("time", time + 1),
            "ALK_source": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                np.ones((3, 1, 2, 2)),
            ),
            "DIC_source": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                -np.ones((3, 1, 2, 2)),
            ),
            "FG_CO2": (("time", "eta_rho", "xi_rho"), np.full((3, 2, 2), 2.0)),
            "FG_ALT_CO2": (("time", "eta_rho", "xi_rho"), np.full((3, 2, 2), 1.0)),
            "hDIC": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                np.full((3, 1, 2, 2), 10.0),
            ),
            "hDIC_ALT_CO2": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                np.full((3, 1, 2, 2), 9.0),
            ),
        },
        coords={"time": time, "s_rho": s_rho, "eta_rho": eta_rho, "xi_rho": xi_rho},
    )


def test_compute_cdr_metrics_outputs(
    minimal_ds: xr.Dataset, minimal_grid_ds: xr.Dataset
) -> None:
    ds_cdr = compute_cdr_metrics(minimal_ds, minimal_grid_ds)

    # Required outputs exist
    for var in [
        "area",
        "window_length",
        "FG_CO2",
        "FG_ALT_CO2",
        "hDIC",
        "hDIC_ALT_CO2",
        "cdr_efficiency",
        "cdr_efficiency_from_delta_diff",
    ]:
        assert var in ds_cdr

    # Area should be 1 (since pm=pn=1)
    assert np.allclose(ds_cdr["area"], 1.0)

    # Window length should be 1 everywhere
    assert np.all(ds_cdr["window_length"].values == 1)


def test_missing_variable_in_ds(
    minimal_ds: xr.Dataset, minimal_grid_ds: xr.Dataset
) -> None:
    bad_ds = minimal_ds.drop_vars("FG_CO2")
    with pytest.raises(KeyError, match="Missing required variables"):
        compute_cdr_metrics(bad_ds, minimal_grid_ds)


def test_missing_variable_in_grid(
    minimal_ds: xr.Dataset, minimal_grid_ds: xr.Dataset
) -> None:
    bad_grid_ds = minimal_grid_ds.drop_vars("pm")
    with pytest.raises(KeyError, match="Missing required variables"):
        compute_cdr_metrics(minimal_ds, bad_grid_ds)


def test_validate_source_passes(minimal_ds):
    # Should not raise
    _validate_source(minimal_ds)


def test_validate_source_alk_negative(minimal_ds):
    bad_ds = minimal_ds.copy()
    bad_ds["ALK_source"].loc[dict(time=0)] = -1
    with pytest.raises(ValueError, match="ALK_source"):
        _validate_source(bad_ds)


def test_validate_source_dic_positive(minimal_ds):
    bad_ds = minimal_ds.copy()
    bad_ds["DIC_source"].loc[dict(time=0)] = 1
    with pytest.raises(ValueError, match="DIC_source"):
        _validate_source(bad_ds)


def test_validate_uptake_efficiency_logs(caplog):
    arr1 = xr.DataArray([1.0, 2.0, 3.0], dims="time")
    arr2 = xr.DataArray([1.0, 2.5, 3.0], dims="time")

    with caplog.at_level(logging.INFO):
        diff = _validate_uptake_efficiency(arr1, arr2)

    assert np.isclose(diff, 0.5)
    assert "flux-based and DIC-based uptake efficiency" in caplog.text


def test_efficiency_nan_when_zero_source(minimal_ds, minimal_grid_ds):
    # Make ALK_source and DIC_source both zero at t=0
    ds = minimal_ds.copy()
    ds["ALK_source"].loc[dict(time=0)] = 0
    ds["DIC_source"].loc[dict(time=0)] = 0

    ds_cdr = compute_cdr_metrics(ds, minimal_grid_ds)

    eff_flux = ds_cdr["cdr_efficiency"].isel(time=0).item()
    eff_diff = ds_cdr["cdr_efficiency_from_delta_diff"].isel(time=0).item()

    # Should be NaN, not inf
    assert np.isnan(eff_flux)
    assert np.isnan(eff_diff)
