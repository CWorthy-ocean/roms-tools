import logging

import numpy as np
import pytest
import xarray as xr

from roms_tools.analysis.cdr_analysis import (
    SOURCE_ROLES,
    _native_carbon_amount_to_tonnes_co2_scale,
    _native_dic_amount_to_mol_multiplier,
    _validate_source,
    _validate_uptake_efficiency,
    compute_cdr_metrics,
)


@pytest.fixture
def minimal_grid_ds():
    """Minimal grid dataset with uniform spacing."""
    return xr.Dataset(
        {
            "pm": (("eta_rho", "xi_rho"), np.ones((4, 4))),
            "pn": (("eta_rho", "xi_rho"), np.ones((4, 4))),
        }
    )


@pytest.fixture
def minimal_ds():
    """Minimal ROMS dataset with required variables and dimensions."""
    time = np.arange(3)
    s_rho = np.arange(1)
    eta_rho = np.arange(4)
    xi_rho = np.arange(4)

    return xr.Dataset(
        {
            "avg_begin_time": ("time", time),
            "avg_end_time": ("time", time + 1),
            "ALK_source": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                np.ones((3, 1, 4, 4)),
            ),
            "DIC_source": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                -np.ones((3, 1, 4, 4)),
            ),
            "FG_CO2": (("time", "eta_rho", "xi_rho"), np.full((3, 4, 4), 2.0)),
            "FG_ALT_CO2": (("time", "eta_rho", "xi_rho"), np.full((3, 4, 4), 1.0)),
            "hDIC": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                np.full((3, 1, 4, 4), 10.0),
            ),
            "hDIC_ALT_CO2": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                np.full((3, 1, 4, 4), 9.0),
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


def test_boundary_ring_is_not_counted(
    minimal_ds: xr.Dataset, minimal_grid_ds: xr.Dataset
) -> None:
    """A perturbation in the outermost rho rows and columns is left out of both uptakes.

    Those are boundary points: ucla-roms exports what reaches them to an upscaled
    parent, which counts it, so counting it in this domain too would count it twice.
    """
    ds = minimal_ds.copy(deep=True)
    for var in ["FG_CO2", "hDIC"]:
        values = ds[var].values
        values[..., [0, -1], :] = 100.0
        values[..., :, [0, -1]] = 100.0
    ds_cdr = compute_cdr_metrics(ds, minimal_grid_ds)

    # 2 x 2 interior cells of area 1, with FG and hDIC differences of 1, windows of 1 s
    native_flux = 4.0 * np.arange(1, 4)
    native_dic = np.full(3, 4.0)
    for var, native in [
        ("cdr_carbon_uptake_from_flux", native_flux),
        ("cdr_carbon_uptake_from_DIC_difference", native_dic),
    ]:
        np.testing.assert_allclose(
            ds_cdr[var] / ds_cdr[var].attrs["tonnes_co2_per_native_unit"], native
        )


def test_native_dic_amount_to_mol_multiplier_warns_without_units(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert _native_dic_amount_to_mol_multiplier(None) == 1e-3
    assert "missing" in caplog.text
    assert "assuming mmol" in caplog.text


def test_native_dic_amount_to_mol_multiplier_no_warn_for_mmol(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert _native_dic_amount_to_mol_multiplier("mmol/m^2/s") == 1e-3
    assert caplog.text == ""


def test_carbon_uptake_tonnes_co2_analytic() -> None:
    """Uptake (tonnes CO2) equals native mmol C times the mmol→tonnes CO2 scale."""
    grid = xr.Dataset(
        {
            "pm": (("eta_rho", "xi_rho"), np.ones((3, 3))),
            "pn": (("eta_rho", "xi_rho"), np.ones((3, 3))),
        }
    )
    # 3 x 3 cells of area 1 m^2, whose only interior cell is the centre; FG diff
    # 3 mmol m^-2 s^-1 for 1 s -> 3 mmol C integrated, hDIC diff 3 mmol m^-2 -> 3 mmol C
    dims4 = ("time", "s_rho", "eta_rho", "xi_rho")
    ds = xr.Dataset(
        {
            "avg_begin_time": ("time", [0.0]),
            "avg_end_time": ("time", [1.0]),
            "ALK_source": (dims4, np.ones((1, 1, 3, 3))),
            "DIC_source": (dims4, -np.ones((1, 1, 3, 3))),
            "FG_CO2": (("time", "eta_rho", "xi_rho"), np.full((1, 3, 3), 5.0)),
            "FG_ALT_CO2": (("time", "eta_rho", "xi_rho"), np.full((1, 3, 3), 2.0)),
            "hDIC": (dims4, np.full((1, 1, 3, 3), 10.0)),
            "hDIC_ALT_CO2": (dims4, np.full((1, 1, 3, 3), 7.0)),
        },
        coords={
            "time": [0],
            "s_rho": [0],
            "eta_rho": np.arange(3),
            "xi_rho": np.arange(3),
        },
    )
    ds["FG_CO2"].attrs["units"] = "mmol/m^2/s"
    ds["hDIC"].attrs["units"] = "mmol/m^2"

    ds_cdr = compute_cdr_metrics(ds, grid)

    native_mmol_flux = 3.0
    native_mmol_dic = 3.0

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
    _validate_source(minimal_ds, "release")


def test_validate_source_alk_negative(minimal_ds):
    bad_ds = minimal_ds.copy()
    bad_ds["ALK_source"].loc[dict(time=0)] = -1
    with pytest.raises(ValueError, match="ALK_source"):
        _validate_source(bad_ds, "release")


def test_validate_source_dic_positive(minimal_ds):
    bad_ds = minimal_ds.copy()
    bad_ds["DIC_source"].loc[dict(time=0)] = 1
    with pytest.raises(ValueError, match="DIC_source"):
        _validate_source(bad_ds, "release")


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


@pytest.fixture
def transfer_ds(minimal_ds):
    """Dataset whose source has the signs of a transfer (ALK < 0, DIC > 0)."""
    ds = minimal_ds.copy(deep=True)
    ds["ALK_source"] = -ds["ALK_source"]
    ds["DIC_source"] = -ds["DIC_source"]
    return ds


def _expected_cumulative_source(ds: xr.Dataset) -> xr.DataArray:
    window_length = ds["avg_end_time"] - ds["avg_begin_time"]
    return (
        (ds["ALK_source"] - ds["DIC_source"]).sum(dim=["s_rho", "eta_rho", "xi_rho"])
        * window_length
    ).cumsum("time")


def test_cumulative_source_stored(minimal_ds, minimal_grid_ds):
    ds_cdr = compute_cdr_metrics(minimal_ds, minimal_grid_ds)

    np.testing.assert_allclose(
        ds_cdr["cdr_cumulative_source"], _expected_cumulative_source(minimal_ds)
    )
    assert ds_cdr["cdr_cumulative_source"].attrs["units"] == "mmol"


def test_cumulative_source_units_strip_per_second(minimal_ds, minimal_grid_ds):
    minimal_ds["ALK_source"].attrs["units"] = "meq/s"
    ds_cdr = compute_cdr_metrics(minimal_ds, minimal_grid_ds)

    assert ds_cdr["cdr_cumulative_source"].attrs["units"] == "meq"


def test_release_role_metadata(minimal_ds, minimal_grid_ds):
    ds_cdr = compute_cdr_metrics(minimal_ds, minimal_grid_ds)

    assert ds_cdr.attrs["cdr_source_role"] == "release"
    for var, units_var in [
        ("cdr_carbon_uptake_from_flux", "FG_CO2"),
        ("cdr_carbon_uptake_from_DIC_difference", "hDIC"),
    ]:
        expected = _native_carbon_amount_to_tonnes_co2_scale(
            minimal_ds[units_var].attrs.get("units")
        )
        assert ds_cdr[var].attrs["tonnes_co2_per_native_unit"] == expected


def test_transfer_role(transfer_ds, minimal_ds, minimal_grid_ds):
    ds_cdr = compute_cdr_metrics(transfer_ds, minimal_grid_ds, source_role="transfer")
    release = compute_cdr_metrics(minimal_ds, minimal_grid_ds)

    assert ds_cdr.attrs["cdr_source_role"] == "transfer"
    for var in ["cdr_efficiency_from_flux", "cdr_efficiency_from_DIC_difference"]:
        assert ds_cdr[var].isnull().all()
        assert "time" in ds_cdr[var].coords
        assert (
            "not defined for source_role='transfer'" in ds_cdr[var].attrs["description"]
        )
    for var in [
        "cdr_carbon_uptake_from_flux",
        "cdr_carbon_uptake_from_DIC_difference",
    ]:
        assert np.isfinite(ds_cdr[var]).all()
        xr.testing.assert_allclose(ds_cdr[var], release[var])
    np.testing.assert_allclose(
        ds_cdr["cdr_cumulative_source"], _expected_cumulative_source(transfer_ds)
    )


def test_release_role_rejects_transfer_signs(transfer_ds, minimal_grid_ds):
    with pytest.raises(ValueError, match="release constraints"):
        compute_cdr_metrics(transfer_ds, minimal_grid_ds, source_role="release")


def test_transfer_role_still_requires_source_variables(minimal_ds, minimal_grid_ds):
    with pytest.raises(KeyError):
        compute_cdr_metrics(
            minimal_ds.drop_vars("ALK_source"), minimal_grid_ds, source_role="transfer"
        )


def test_invalid_source_role(minimal_ds, minimal_grid_ds):
    with pytest.raises(ValueError, match="source_role"):
        compute_cdr_metrics(minimal_ds, minimal_grid_ds, source_role="bogus")  # type: ignore[arg-type]
    assert SOURCE_ROLES == ("release", "transfer")
