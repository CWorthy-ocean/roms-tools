import logging

import numpy as np
import pytest
import xarray as xr

from roms_tools.analysis.cdr_analysis import (
    _native_carbon_amount_to_tonnes_co2_scale,
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
        "cdr_efficiency_from_flux",
        "cdr_efficiency_from_DIC_difference",
        "cdr_carbon_uptake_from_flux",
        "cdr_carbon_uptake_from_DIC_difference",
    ]:
        assert var in ds_cdr

    # Area should be 1 (since pm=pn=1)
    assert np.allclose(ds_cdr["area"], 1.0)

    # Window length should be 1 everywhere
    assert np.all(ds_cdr["window_length"].values == 1)

    # Uptake variables are the efficiency numerators (efficiency * cumulative source)
    cumulative_source = (
        (
            (minimal_ds["ALK_source"] - minimal_ds["DIC_source"]).sum(
                dim=["s_rho", "eta_rho", "xi_rho"]
            )
            * ds_cdr["window_length"]
        )
        .cumsum(dim="time")
        .compute()
    )
    scale_flux = _native_carbon_amount_to_tonnes_co2_scale(
        minimal_ds["FG_CO2"].attrs.get("units")
    )
    scale_dic = _native_carbon_amount_to_tonnes_co2_scale(
        minimal_ds["hDIC"].attrs.get("units")
    )
    np.testing.assert_allclose(
        ds_cdr["cdr_carbon_uptake_from_flux"],
        ds_cdr["cdr_efficiency_from_flux"] * cumulative_source * scale_flux,
    )
    np.testing.assert_allclose(
        ds_cdr["cdr_carbon_uptake_from_DIC_difference"],
        ds_cdr["cdr_efficiency_from_DIC_difference"] * cumulative_source * scale_dic,
    )
    assert ds_cdr["cdr_carbon_uptake_from_flux"].attrs["units"] == "tonnes CO2"
    assert (
        ds_cdr["cdr_carbon_uptake_from_DIC_difference"].attrs["units"] == "tonnes CO2"
    )


def test_carbon_uptake_tonnes_co2_analytic() -> None:
    """Uptake (tonnes CO2) equals native mmol C times the mmol→tonnes CO2 scale."""
    grid = xr.Dataset(
        {
            "pm": (("eta_rho", "xi_rho"), np.ones((1, 1))),
            "pn": (("eta_rho", "xi_rho"), np.ones((1, 1))),
        }
    )
    # One cell, area 1 m^2; FG diff 3 mmol m^-2 s^-1 for 1 s -> 3 mmol C integrated
    ds = xr.Dataset(
        {
            "avg_begin_time": ("time", [0.0]),
            "avg_end_time": ("time", [1.0]),
            "ALK_source": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                np.ones((1, 1, 1, 1)),
            ),
            "DIC_source": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                -np.ones((1, 1, 1, 1)),
            ),
            "FG_CO2": (("time", "eta_rho", "xi_rho"), np.array([[5.0]])),
            "FG_ALT_CO2": (("time", "eta_rho", "xi_rho"), np.array([[2.0]])),
            "hDIC": (("time", "s_rho", "eta_rho", "xi_rho"), np.array([[[[10.0]]]])),
            "hDIC_ALT_CO2": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                np.array([[[[7.0]]]]),
            ),
        },
        coords={
            "time": [0],
            "s_rho": [0],
            "eta_rho": [0],
            "xi_rho": [0],
        },
    )
    ds["FG_CO2"].attrs["units"] = "mmol/m^2/s"
    ds["hDIC"].attrs["units"] = "mmol/m^2"

    ds_cdr = compute_cdr_metrics(ds, grid)

    native_mmol_flux = 3.0  # (5-2) * 1 m^2 * 1 s, one timestep cumsum
    native_mmol_dic = 3.0  # (10-7) * 1 m^2 * one column

    mmol_to_tonnes = _native_carbon_amount_to_tonnes_co2_scale("mmol/m^2/s")
    assert np.isclose(
        mmol_to_tonnes,
        _native_carbon_amount_to_tonnes_co2_scale("mmol/m^2"),
    )

    expected_flux_tonnes = native_mmol_flux * mmol_to_tonnes
    expected_dic_tonnes = native_mmol_dic * mmol_to_tonnes

    np.testing.assert_allclose(
        ds_cdr["cdr_carbon_uptake_from_flux"].values,
        expected_flux_tonnes,
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        ds_cdr["cdr_carbon_uptake_from_DIC_difference"].values,
        expected_dic_tonnes,
        rtol=0,
        atol=1e-15,
    )
    assert ds_cdr["cdr_carbon_uptake_from_flux"].attrs["units"] == "tonnes CO2"
    assert (
        ds_cdr["cdr_carbon_uptake_from_DIC_difference"].attrs["units"] == "tonnes CO2"
    )


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

    eff_flux = ds_cdr["cdr_efficiency_from_flux"].isel(time=0).item()
    eff_diff = ds_cdr["cdr_efficiency_from_DIC_difference"].isel(time=0).item()

    # Should be NaN, not inf
    assert np.isnan(eff_flux)
    assert np.isnan(eff_diff)
