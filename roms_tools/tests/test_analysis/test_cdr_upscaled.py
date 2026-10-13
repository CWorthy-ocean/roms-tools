import logging
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from roms_tools import Ensemble, UpscaledCDR
from roms_tools.analysis.cdr_analysis import (
    _native_carbon_amount_to_tonnes_co2_scale,
    compute_cdr_metrics,
)
from roms_tools.plot import plot_uptake_efficiency

N = 4  # cells per direction in every domain
NT = 3
TIMES = np.array(["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64[ns]")
SCALE = _native_carbon_amount_to_tonnes_co2_scale("mmol/m^2")


def _make_domain(
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    pm_pn: float,
    role: str,
    fg_diff: float,
    hdic_diff: np.ndarray,
    alk: float,
    dic: float,
) -> xr.Dataset:
    """Compute a member's ds_cdr from a synthetic domain (as ROMSOutput would)."""
    lat, lon = np.meshgrid(
        np.linspace(*lat_range, N), np.linspace(*lon_range, N), indexing="ij"
    )
    dims4 = ("time", "s_rho", "eta_rho", "xi_rho")
    time = np.arange(NT, dtype=float)
    ds = xr.Dataset(
        {
            "avg_begin_time": ("time", time),
            "avg_end_time": ("time", time + 1),
            "ALK_source": (dims4, np.full((NT, 1, N, N), alk)),
            "DIC_source": (dims4, np.full((NT, 1, N, N), dic)),
            "FG_CO2": (
                ("time", "eta_rho", "xi_rho"),
                np.full((NT, N, N), 1.0 + fg_diff),
                {"units": "mmol/m^2/s"},
            ),
            "FG_ALT_CO2": (("time", "eta_rho", "xi_rho"), np.ones((NT, N, N))),
            "hDIC": (dims4, 10.0 + hdic_diff, {"units": "mmol/m^2"}),
            "hDIC_ALT_CO2": (dims4, np.full((NT, 1, N, N), 10.0)),
        },
        coords={
            "time": TIMES,
            "s_rho": [0],
            "eta_rho": np.arange(N),
            "xi_rho": np.arange(N),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
            "lon_rho": (("eta_rho", "xi_rho"), lon),
        },
    )
    grid_ds = xr.Dataset(
        {
            "pm": (("eta_rho", "xi_rho"), np.full((N, N), pm_pn)),
            "pn": (("eta_rho", "xi_rho"), np.ones((N, N))),
        }
    )
    return compute_cdr_metrics(ds, grid_ds, source_role=role)  # type: ignore[arg-type]


def _hdic_mid() -> np.ndarray:
    """Mid-domain DIC difference: (eta + 1) * (time + 1)."""
    eta = np.arange(N)[None, None, :, None] + 1.0
    t = np.arange(NT)[:, None, None, None] + 1.0
    return np.broadcast_to(eta * t, (NT, 1, N, N)).copy()


@pytest.fixture
def members() -> dict[str, xr.Dataset]:
    """Child (release), mid and outer (transfer) members with nested footprints."""
    child = _make_domain(
        (10, 11), (20, 21), 1.0, "release", 2.0, np.full((NT, 1, N, N), 3.0), 1.0, -1.0
    )
    # pm * pn = 0.25 and 1/16 give cell areas of 4 and 16 (child: 1)
    mid = _make_domain((9, 12), (19, 22), 0.25, "transfer", 1.0, _hdic_mid(), -0.5, 0.5)
    outer = _make_domain(
        (0, 30),
        (0, 40),
        1 / 16,
        "transfer",
        0.5,
        np.full((NT, 1, N, N), 1.0),
        0.25,
        0.25,
    )
    return {"child": child, "mid": mid, "outer": outer}


def _native(ds: xr.Dataset, var: str) -> xr.DataArray:
    return ds[var] / ds[var].attrs["tonnes_co2_per_native_unit"]


def test_combined_source_and_uptake(members):
    up = UpscaledCDR(members)
    ds = up.ds

    np.testing.assert_allclose(
        ds["cdr_cumulative_source"], members["child"]["cdr_cumulative_source"]
    )
    for var in ["cdr_carbon_uptake_from_flux", "cdr_carbon_uptake_from_DIC_difference"]:
        expected = sum(m[var].values for m in members.values())
        np.testing.assert_allclose(ds[var], expected)
        assert ds[var].attrs["units"] == "tonnes CO2"
        assert ds[var].attrs["tonnes_co2_per_native_unit"] == SCALE
        for name, m in members.items():
            short = var.removeprefix("cdr_")
            np.testing.assert_allclose(ds[f"{name}_{short}"], m[var])

    assert ds.attrs["cdr_source_role"] == "release"
    assert ds.attrs["members"] == "child:release, mid:transfer, outer:transfer"
    assert ds["cdr_cumulative_source"].attrs["units"] == "mmol"


def test_efficiency_matches_analytic_ratio(members):
    ds = UpscaledCDR(members).ds

    # Flux: native uptake per step = diff * area * 16 cells * 1 s, cumulative.
    cum = np.arange(1, NT + 1)
    flux_native = (2.0 * 1 + 1.0 * 4 + 0.5 * 16) * 16 * cum
    source = 2.0 * 16 * cum  # (ALK - DIC) = 2 in each of 16 cells of the release
    np.testing.assert_allclose(ds["cdr_efficiency_from_flux"], flux_native / source)

    dic_native = sum(
        _native(m, "cdr_carbon_uptake_from_DIC_difference") for m in members.values()
    )
    np.testing.assert_allclose(
        ds["cdr_efficiency_from_DIC_difference"], dic_native / source
    )


def test_two_releases_sum_sources(members):
    members["mid"].attrs["cdr_source_role"] = "release"
    ds = UpscaledCDR(members).ds
    np.testing.assert_allclose(
        ds["cdr_cumulative_source"],
        members["child"]["cdr_cumulative_source"]
        + members["mid"]["cdr_cumulative_source"],
    )


def test_different_scales_omit_attr_and_use_member_scale(members):
    members["outer"] = members["outer"].copy(deep=True)
    members["outer"]["cdr_carbon_uptake_from_DIC_difference"].attrs[
        "tonnes_co2_per_native_unit"
    ] = 1000 * SCALE
    ds = UpscaledCDR(members).ds

    assert (
        "tonnes_co2_per_native_unit"
        not in ds["cdr_carbon_uptake_from_DIC_difference"].attrs
    )
    assert (
        ds["cdr_carbon_uptake_from_flux"].attrs["tonnes_co2_per_native_unit"] == SCALE
    )
    native = sum(
        _native(m, "cdr_carbon_uptake_from_DIC_difference") for m in members.values()
    )
    np.testing.assert_allclose(
        ds["cdr_efficiency_from_DIC_difference"],
        native / members["child"]["cdr_cumulative_source"],
    )


def test_shorter_member_gives_nan_and_warns(members, caplog):
    members["mid"] = members["mid"].isel(time=slice(0, NT - 1))
    with caplog.at_level(logging.WARNING):
        ds = UpscaledCDR(members).ds

    assert caplog.text.count("missing from at least one member") == 1
    assert "1 time record" in caplog.text
    assert "mid" in caplog.text
    assert ds.sizes["time"] == NT
    for var in [
        "cdr_carbon_uptake_from_flux",
        "cdr_carbon_uptake_from_DIC_difference",
        "cdr_efficiency_from_flux",
        "closure_residual",
    ]:
        assert np.isnan(ds[var].values[-1])
        assert np.isfinite(ds[var].values[:-1]).all()
    assert np.isnan(ds["mid_DIC_difference_inside_child"].values[-1])
    assert np.isfinite(ds["child_carbon_uptake_from_flux"]).all()


def test_no_warning_when_times_match(members, caplog):
    with caplog.at_level(logging.WARNING):
        UpscaledCDR(members)
    assert "missing from at least one member" not in caplog.text


def test_identical_grid_raises(members):
    members["mid"] = members["mid"].assign_coords(
        lon_rho=members["child"]["lon_rho"], lat_rho=members["child"]["lat_rho"]
    )
    with pytest.raises(ValueError, match="same grid"):
        UpscaledCDR(members)


def test_non_nested_order_raises(members):
    reordered = {
        "mid": members["mid"],
        "child": members["child"],
        "outer": members["outer"],
    }
    with pytest.raises(ValueError, match="strictly contain"):
        UpscaledCDR(reordered)


def test_parent_not_strictly_containing_raises(members):
    # Parent bbox touches the child's bbox on one side.
    mid = members["mid"].assign_coords(lon_rho=members["mid"]["lon_rho"].clip(min=10.0))
    with pytest.raises(ValueError, match="strictly contain"):
        UpscaledCDR({"child": members["child"], "mid": mid})


def test_missing_role_raises(members):
    del members["mid"].attrs["cdr_source_role"]
    with pytest.raises(ValueError, match="'mid'.*cdr_source_role"):
        UpscaledCDR(members)


def test_invalid_role_raises(members):
    members["mid"].attrs["cdr_source_role"] = "bogus"
    with pytest.raises(ValueError, match="'mid'.*bogus"):
        UpscaledCDR(members)


def test_missing_variable_raises(members):
    members["child"] = members["child"].drop_vars("cdr_cumulative_source")
    with pytest.raises(ValueError, match="'child'.*cdr_cumulative_source.*recompute"):
        UpscaledCDR(members)


def test_missing_coords_raises(members):
    members["outer"] = members["outer"].drop_vars(["lat_rho", "lon_rho"])
    with pytest.raises(ValueError, match="'outer'.*lat_rho"):
        UpscaledCDR(members)


def test_missing_scale_attr_raises(members):
    del members["mid"]["cdr_carbon_uptake_from_flux"].attrs[
        "tonnes_co2_per_native_unit"
    ]
    with pytest.raises(ValueError, match="'mid'.*tonnes_co2_per_native_unit"):
        UpscaledCDR(members)


def test_no_release_raises(members):
    members["child"].attrs["cdr_source_role"] = "transfer"
    with pytest.raises(ValueError, match="release"):
        UpscaledCDR(members)


def test_single_member_raises(members):
    with pytest.raises(ValueError, match="at least one parent"):
        UpscaledCDR({"child": members["child"]})


def test_mismatched_window_length_raises(members):
    members["mid"]["window_length"] = members["mid"]["window_length"] * 2
    with pytest.raises(ValueError, match="averaging window"):
        UpscaledCDR(members)


def test_non_unique_window_length_raises(members):
    members["mid"]["window_length"] = members["mid"]["window_length"] * np.array(
        [1.0, 2.0, 1.0]
    )
    with pytest.raises(ValueError, match="'mid'.*single unique window_length"):
        UpscaledCDR(members)


def test_different_release_source_units_raise(members):
    members["mid"].attrs["cdr_source_role"] = "release"
    members["mid"]["cdr_cumulative_source"].attrs["units"] = "meq"
    with pytest.raises(ValueError, match="units"):
        UpscaledCDR(members)


def test_footprint_diagnostic_hand_computed(members):
    ds = UpscaledCDR(members).ds

    # Mid cells inside the child's bbox (lon 10..11, lat 20..21): eta/xi idx 1, 2.
    # Per-cell DIC difference (eta + 1) * (t + 1), mid area 4: rows eta=1,2 give
    # (2 + 3) per column over 2 columns.
    expected = (2 + 3) * 2 * 4 * np.arange(1, NT + 1) * SCALE
    np.testing.assert_allclose(ds["mid_DIC_difference_inside_child"], expected)
    assert ds["mid_DIC_difference_inside_child"].attrs["units"] == "tonnes CO2"

    # No outer cell (lon 0/10/20/30, lat 0/13.3/26.7/40) lies in the mid bbox.
    np.testing.assert_allclose(ds["outer_DIC_difference_inside_mid"], 0.0)


def test_closure_residual(members):
    ds = UpscaledCDR(members).ds
    np.testing.assert_allclose(
        ds["closure_residual"],
        ds["cdr_carbon_uptake_from_flux"] - ds["cdr_carbon_uptake_from_DIC_difference"],
    )


def test_final_values_logged(members, caplog):
    with caplog.at_level(logging.INFO):
        UpscaledCDR(members)
    assert "closure_residual" in caplog.text
    assert "mid_DIC_difference_inside_child" in caplog.text
    assert "outer_DIC_difference_inside_mid" in caplog.text


def test_plot_uptake_efficiency_and_ensemble_accept_combined(members):
    ds = UpscaledCDR(members).ds
    with patch("matplotlib.pyplot.show"):
        plot_uptake_efficiency(ds)
    plt.close("all")

    ensemble = Ensemble({"a": ds})
    assert "ensemble_efficiency_mean" in ensemble.ds


def test_plot_saves_file(members, tmp_path: Path):
    up = UpscaledCDR(members)
    path = tmp_path / "x.png"
    up.plot(save_path=str(path))
    assert path.exists() and path.stat().st_size > 0

    with patch("matplotlib.pyplot.show") as show:
        up.plot()
    show.assert_called_once()
    plt.close("all")


def test_members_from_netcdf_paths(members, tmp_path: Path):
    paths: dict[str, str | Path | xr.Dataset] = {}
    for i, (name, ds) in enumerate(members.items()):
        path = tmp_path / f"{name}.nc"
        ds.to_netcdf(path)
        paths[name] = path if i % 2 else str(path)

    from_files = UpscaledCDR(paths).ds
    in_memory = UpscaledCDR(members).ds
    xr.testing.assert_allclose(from_files, in_memory)
    assert from_files.attrs["members"] == in_memory.attrs["members"]

    from_files.to_netcdf(tmp_path / "combined.nc")
    with xr.open_dataset(tmp_path / "combined.nc") as reloaded:
        xr.testing.assert_allclose(reloaded, in_memory)
