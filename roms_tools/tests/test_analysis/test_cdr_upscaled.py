import logging
from pathlib import Path
from unittest.mock import patch

import dask.array
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
    """Compute the ds_cdr of a synthetic domain (as ROMSOutput would)."""
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
def domains() -> dict[str, xr.Dataset]:
    """Child (release), mid and outer (transfer) domains with nested footprints."""
    child = _make_domain(
        (9.6, 11.4),
        (19.6, 21.4),
        1.0,
        "release",
        2.0,
        np.full((NT, 1, N, N), 3.0),
        1.0,
        -1.0,
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


def test_combined_source_and_uptake(domains):
    up = UpscaledCDR(domains)
    ds = up.ds

    np.testing.assert_allclose(
        ds["cdr_cumulative_source"], domains["child"]["cdr_cumulative_source"]
    )
    for var in ["cdr_carbon_uptake_from_flux", "cdr_carbon_uptake_from_DIC_difference"]:
        expected = sum(m[var].values for m in domains.values())
        np.testing.assert_allclose(ds[var], expected)
        assert ds[var].attrs["units"] == "tonnes CO2"
        assert ds[var].attrs["tonnes_co2_per_native_unit"] == SCALE
        for name, m in domains.items():
            short = var.removeprefix("cdr_")
            np.testing.assert_allclose(ds[f"{name}_{short}"], m[var])

    assert ds.attrs["cdr_source_role"] == "release"
    assert ds.attrs["domains"] == "child:release, mid:transfer, outer:transfer"
    assert ds["cdr_cumulative_source"].attrs["units"] == "mmol"


def test_efficiency_matches_analytic_ratio(domains):
    ds = UpscaledCDR(domains).ds

    # Flux: native uptake per step = diff * area * 4 interior cells * 1 s, cumulative.
    cum = np.arange(1, NT + 1)
    flux_native = (2.0 * 1 + 1.0 * 4 + 0.5 * 16) * 4 * cum
    source = 2.0 * 16 * cum  # (ALK - DIC) = 2 in each of 16 cells of the release
    np.testing.assert_allclose(ds["cdr_efficiency_from_flux"], flux_native / source)

    dic_native = sum(
        _native(m, "cdr_carbon_uptake_from_DIC_difference") for m in domains.values()
    )
    np.testing.assert_allclose(
        ds["cdr_efficiency_from_DIC_difference"], dic_native / source
    )


def test_two_releases_sum_sources(domains):
    domains["mid"].attrs["cdr_source_role"] = "release"
    ds = UpscaledCDR(domains).ds
    np.testing.assert_allclose(
        ds["cdr_cumulative_source"],
        domains["child"]["cdr_cumulative_source"]
        + domains["mid"]["cdr_cumulative_source"],
    )


def test_different_scales_raise(domains):
    domains["outer"] = domains["outer"].copy(deep=True)
    domains["outer"]["cdr_carbon_uptake_from_DIC_difference"].attrs[
        "tonnes_co2_per_native_unit"
    ] = 1000 * SCALE

    with pytest.raises(ValueError, match="native carbon units"):
        UpscaledCDR(domains)


def test_combined_uptake_keeps_shared_scale(domains):
    ds = UpscaledCDR(domains).ds

    for var in ("cdr_carbon_uptake_from_flux", "cdr_carbon_uptake_from_DIC_difference"):
        assert ds[var].attrs["tonnes_co2_per_native_unit"] == SCALE


def test_window_length_roundoff_tolerated(domains):
    domains["mid"] = domains["mid"].copy(deep=True)
    domains["mid"]["window_length"] = domains["mid"]["window_length"] + 1e-9

    UpscaledCDR(domains)


def test_shorter_domain_gives_nan_and_warns(domains, caplog):
    domains["mid"] = domains["mid"].isel(time=slice(0, NT - 1))
    with caplog.at_level(logging.WARNING):
        ds = UpscaledCDR(domains).ds

    assert caplog.text.count("missing from at least one domain") == 1
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


def test_no_warning_when_times_match(domains, caplog):
    with caplog.at_level(logging.WARNING):
        UpscaledCDR(domains)
    assert "missing from at least one domain" not in caplog.text


def test_identical_grid_raises(domains):
    domains["mid"] = domains["mid"].assign_coords(
        lon_rho=domains["child"]["lon_rho"], lat_rho=domains["child"]["lat_rho"]
    )
    with pytest.raises(ValueError, match="same grid"):
        UpscaledCDR(domains)


def test_non_nested_order_raises(domains):
    reordered = {
        "mid": domains["mid"],
        "child": domains["child"],
        "outer": domains["outer"],
    }
    with pytest.raises(ValueError, match="strictly contain"):
        UpscaledCDR(reordered)


def test_parent_not_strictly_containing_raises(domains):
    # Parent bbox touches the child's bbox on one side.
    mid = domains["mid"].assign_coords(lon_rho=domains["mid"]["lon_rho"].clip(min=9.6))
    with pytest.raises(ValueError, match="strictly contain"):
        UpscaledCDR({"child": domains["child"], "mid": mid})


@pytest.mark.parametrize("child_convention", ["0_360", "-180_180"])
def test_domains_crossing_greenwich(domains, child_convention):
    # Shift every domain so the child is centred near 0E. Grid stores lon in
    # [0, 360), so the child's longitudes wrap from ~359.1 to ~0.9.
    def shift(ds: xr.Dataset, convention: str) -> xr.Dataset:
        lon = ds["lon_rho"] - 10.5
        lon = lon % 360 if convention == "0_360" else (lon + 180) % 360 - 180
        return ds.assign_coords(lon_rho=lon)

    shifted = {
        name: shift(ds, child_convention if name == "child" else "0_360")
        for name, ds in domains.items()
    }
    assert float(shifted["outer"]["lon_rho"].max()) > 300

    xr.testing.assert_allclose(UpscaledCDR(shifted).ds, UpscaledCDR(domains).ds)


def test_parent_crossing_greenwich_still_checks_nesting(domains):
    # The parent wraps from 359.6 to 0.6, so its stored lon range spans almost
    # 360 degrees; a child 150 degrees away must still be rejected.
    mid = domains["mid"].assign_coords(lon_rho=(domains["mid"]["lon_rho"] - 9.4) % 360)
    child = domains["child"].assign_coords(lon_rho=domains["child"]["lon_rho"] - 9.3)
    UpscaledCDR({"child": child, "mid": mid})

    distant = child.assign_coords(lon_rho=child["lon_rho"] + 150)
    with pytest.raises(ValueError, match="strictly contain"):
        UpscaledCDR({"child": distant, "mid": mid})


def test_domain_name_clashing_with_combined_variable_raises(domains):
    # "cdr" + "_carbon_uptake_from_flux" is the combined total's name.
    renamed = {"cdr": domains["child"], "mid": domains["mid"]}
    with pytest.raises(ValueError, match="cdr_carbon_uptake_from_flux"):
        UpscaledCDR(renamed)


def test_missing_role_raises(domains):
    del domains["mid"].attrs["cdr_source_role"]
    with pytest.raises(ValueError, match="'mid'.*cdr_source_role"):
        UpscaledCDR(domains)


def test_invalid_role_raises(domains):
    domains["mid"].attrs["cdr_source_role"] = "bogus"
    with pytest.raises(ValueError, match="'mid'.*bogus"):
        UpscaledCDR(domains)


def test_missing_variable_raises(domains):
    domains["child"] = domains["child"].drop_vars("cdr_cumulative_source")
    with pytest.raises(ValueError, match="'child'.*cdr_cumulative_source.*recompute"):
        UpscaledCDR(domains)


def test_missing_coords_raises(domains):
    domains["outer"] = domains["outer"].drop_vars(["lat_rho", "lon_rho"])
    with pytest.raises(ValueError, match="'outer'.*lat_rho"):
        UpscaledCDR(domains)


def test_missing_scale_attr_raises(domains):
    del domains["mid"]["cdr_carbon_uptake_from_flux"].attrs[
        "tonnes_co2_per_native_unit"
    ]
    with pytest.raises(ValueError, match="'mid'.*tonnes_co2_per_native_unit"):
        UpscaledCDR(domains)


def test_no_release_raises(domains):
    domains["child"].attrs["cdr_source_role"] = "transfer"
    with pytest.raises(ValueError, match="release"):
        UpscaledCDR(domains)


def test_single_domain_raises(domains):
    with pytest.raises(ValueError, match="at least one parent"):
        UpscaledCDR({"child": domains["child"]})


def test_mismatched_window_length_raises(domains):
    domains["mid"]["window_length"] = domains["mid"]["window_length"] * 2
    with pytest.raises(ValueError, match="averaging window"):
        UpscaledCDR(domains)


def test_non_unique_window_length_raises(domains):
    domains["mid"]["window_length"] = domains["mid"]["window_length"] * np.array(
        [1.0, 2.0, 1.0]
    )
    with pytest.raises(ValueError, match="'mid'.*single unique window_length"):
        UpscaledCDR(domains)


def test_different_release_source_units_raise(domains):
    domains["mid"].attrs["cdr_source_role"] = "release"
    domains["mid"]["cdr_cumulative_source"].attrs["units"] = "meq"
    with pytest.raises(ValueError, match="units"):
        UpscaledCDR(domains)


def test_footprint_diagnostic_hand_computed(domains):
    ds = UpscaledCDR(domains).ds

    # Mid cells inside the child's interior (lon 9.9..11.1, lat 19.9..21.1): eta/xi
    # idx 1, 2.
    # Per-cell DIC difference (eta + 1) * (t + 1), mid area 4: rows eta=1,2 give
    # (2 + 3) per column over 2 columns.
    expected = (2 + 3) * 2 * 4 * np.arange(1, NT + 1) * SCALE
    np.testing.assert_allclose(ds["mid_DIC_difference_inside_child"], expected)
    assert ds["mid_DIC_difference_inside_child"].attrs["units"] == "tonnes CO2"

    # No outer cell (lon 0/10/20/30, lat 0/13.3/26.7/40) lies in the mid interior.
    np.testing.assert_allclose(ds["outer_DIC_difference_inside_mid"], 0.0)


def test_footprint_follows_rotated_child(domains):
    # Rotate the child by 45 degrees about its centre: the mid points at the
    # corners of its lon/lat bounding box now lie outside the child's area.
    child = domains["child"]
    theta = np.pi / 4
    lon, lat = child["lon_rho"] - 10.5, child["lat_rho"] - 20.5
    domains["child"] = child.assign_coords(
        lon_rho=10.5 + lon * np.cos(theta) - lat * np.sin(theta),
        lat_rho=20.5 + lon * np.sin(theta) + lat * np.cos(theta),
    )
    ds = UpscaledCDR(domains).ds

    np.testing.assert_allclose(ds["mid_DIC_difference_inside_child"], 0.0)


def test_footprint_reads_only_the_child_window(domains):
    # Give the mid hDIC fields one dask block per cell column, and fail any block
    # outside eta/xi 1..2, the mid cells inside the child (see the hand-computed
    # test). Cropping before the arithmetic means those blocks are never computed.
    def poisoned(da: xr.DataArray) -> xr.DataArray:
        def check(block, block_info=None):
            if block_info is not None:
                (eta, _), (xi, _) = block_info[0]["array-location"][2:]
                if not (1 <= eta <= 2 and 1 <= xi <= 2):
                    raise AssertionError(f"read {da.name} at eta={eta}, xi={xi}")
            return block

        data = dask.array.from_array(da.values, chunks=(-1, -1, 1, 1))
        return da.copy(data=data.map_blocks(check, meta=np.array((), dtype=float)))

    mid = domains["mid"]
    domains["mid"] = mid.assign(
        hDIC=poisoned(mid["hDIC"]), hDIC_ALT_CO2=poisoned(mid["hDIC_ALT_CO2"])
    )
    ds = UpscaledCDR(domains).ds

    expected = (2 + 3) * 2 * 4 * np.arange(1, NT + 1) * SCALE
    np.testing.assert_allclose(ds["mid_DIC_difference_inside_child"], expected)


def test_closure_residual(domains):
    ds = UpscaledCDR(domains).ds
    np.testing.assert_allclose(
        ds["closure_residual"],
        ds["cdr_carbon_uptake_from_flux"] - ds["cdr_carbon_uptake_from_DIC_difference"],
    )


def test_final_values_logged(domains, caplog):
    with caplog.at_level(logging.INFO):
        UpscaledCDR(domains)
    assert "closure_residual" in caplog.text
    assert "mid_DIC_difference_inside_child" in caplog.text
    assert "outer_DIC_difference_inside_mid" in caplog.text


def test_plot_uptake_efficiency_and_ensemble_accept_combined(domains):
    ds = UpscaledCDR(domains).ds
    with patch("matplotlib.pyplot.show"):
        plot_uptake_efficiency(ds)
    plt.close("all")

    ensemble = Ensemble({"a": ds})
    assert "ensemble_efficiency_mean" in ensemble.ds


def test_ensemble_rejects_transfer_domains(domains):
    with pytest.raises(ValueError, match=r"Members \['mid', 'outer'\].*UpscaledCDR"):
        Ensemble(domains)


def test_plot_saves_file(domains, tmp_path: Path):
    up = UpscaledCDR(domains)
    path = tmp_path / "x.png"
    up.plot(save_path=str(path))
    assert path.exists() and path.stat().st_size > 0

    with patch("matplotlib.pyplot.show") as show:
        up.plot()
    show.assert_called_once()
    plt.close("all")


@pytest.fixture
def domains_axes():
    """Return a function that plots the domains with ``show`` patched out."""

    def plot_domains(up: UpscaledCDR, **kwargs):
        with patch("matplotlib.pyplot.show") as show:
            up.plot_domains(**kwargs)
        show.assert_called_once()
        return plt.gcf().axes[0]

    yield plot_domains
    plt.close("all")


def test_plot_domains_saves_file(domains, tmp_path: Path):
    up = UpscaledCDR(domains)
    for margin in (10.0, 0.0):
        path = tmp_path / f"margin_{margin}.png"
        up.plot_domains(margin=margin, save_path=str(path))
        assert path.exists() and path.stat().st_size > 0
        plt.close("all")


def test_plot_domains_legend_lists_domains_and_roles(domains, domains_axes):
    ax = domains_axes(UpscaledCDR(domains))
    _, labels = ax.get_legend_handles_labels()
    assert labels == ["child (release)", "mid (transfer)", "outer (transfer)"]
    assert ax.get_title() == "Nested domains"


def test_plot_domains_margin_widens_the_map(domains, domains_axes):
    up = UpscaledCDR(domains)

    def width(**kwargs) -> float:
        x0, x1, _, _ = domains_axes(up, **kwargs).get_extent()
        plt.close("all")
        return x1 - x0

    assert width(margin=0) < width(margin=10) < width(margin=20)


def test_plot_domains_negative_margin_raises(domains):
    with pytest.raises(ValueError, match="margin must be non-negative"):
        UpscaledCDR(domains).plot_domains(margin=-1)


@pytest.mark.parametrize("convention", ["0_360", "-180_180", "mixed"])
@pytest.mark.parametrize("centre", [0.0, 180.0])
def test_plot_domains_across_greenwich_and_dateline(
    domains, domains_axes, tmp_path: Path, centre, convention
):
    # Centre the child on `centre` and store each domain's longitudes in either
    # convention ("mixed": child in [-180, 180), the others in [0, 360)).
    def shift(ds: xr.Dataset, name: str) -> xr.Dataset:
        lon = ds["lon_rho"] + centre - 10.5
        negative = convention == "-180_180" or (
            convention == "mixed" and name == "child"
        )
        lon = (lon + 180) % 360 - 180 if negative else lon % 360
        return ds.assign_coords(lon_rho=lon)

    shifted = {name: shift(ds, name) for name, ds in domains.items()}
    up = UpscaledCDR(shifted)
    ax = domains_axes(up)

    # Every outline is continuous (no 360 degree jumps) and lies in the outermost
    # domain's longitude range.
    outer_x = np.concatenate([line.get_xdata() for line in ax.lines[-4:]])
    for line in ax.lines:
        x = np.asarray(line.get_xdata())
        assert x.min() >= outer_x.min() - 1e-6 and x.max() <= outer_x.max() + 1e-6
    assert np.ptp(outer_x) == pytest.approx(30.0)

    up.plot_domains(save_path=str(tmp_path / "x.png"))
    assert (tmp_path / "x.png").stat().st_size > 0


def test_plot_domains_from_netcdf_paths(domains, domains_axes, tmp_path: Path):
    paths: dict[str, str | Path | xr.Dataset] = {}
    for i, (name, ds) in enumerate(domains.items()):
        path = tmp_path / f"{name}.nc"
        ds.to_netcdf(path)
        paths[name] = path if i % 2 else str(path)

    ax = domains_axes(UpscaledCDR(paths))
    _, labels = ax.get_legend_handles_labels()
    assert labels == ["child (release)", "mid (transfer)", "outer (transfer)"]
    plt.close("all")

    reference = domains_axes(UpscaledCDR(domains))
    for line, ref_line in zip(ax.lines, reference.lines, strict=True):
        np.testing.assert_allclose(line.get_xdata(), ref_line.get_xdata())
        np.testing.assert_allclose(line.get_ydata(), ref_line.get_ydata())


def test_domains_from_netcdf_paths(domains, tmp_path: Path):
    paths: dict[str, str | Path | xr.Dataset] = {}
    for i, (name, ds) in enumerate(domains.items()):
        path = tmp_path / f"{name}.nc"
        ds.to_netcdf(path)
        paths[name] = path if i % 2 else str(path)

    from_files = UpscaledCDR(paths).ds
    in_memory = UpscaledCDR(domains).ds
    xr.testing.assert_allclose(from_files, in_memory)
    assert from_files.attrs["domains"] == in_memory.attrs["domains"]

    from_files.to_netcdf(tmp_path / "combined.nc")
    with xr.open_dataset(tmp_path / "combined.nc") as reloaded:
        xr.testing.assert_allclose(reloaded, in_memory)
