from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from roms_tools import Ensemble

# ----------------------------
# Fixtures
# ----------------------------


@pytest.fixture
def create_member_ds() -> xr.Dataset:
    """Simple Dataset for testing."""
    times = np.array(["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"cdr_efficiency": ("time", [0.1, 0.2, 0.3]), "abs_time": ("time", times)},
        coords={"time": times},
    )
    return ds


@pytest.fixture
def identical_members(create_member_ds: xr.Dataset) -> dict[str, xr.Dataset]:
    """Two truly identical members for basic tests."""
    return {
        "member1": create_member_ds.copy(),
        "member2": create_member_ds.copy(),
    }


@pytest.fixture
def varied_members() -> dict[str, xr.Dataset]:
    """Ensemble members with different lengths, frequencies, start dates, and leading NaNs."""
    # Member 1: daily, 5 days, starts 2000-01-01, first value is NaN
    times1 = np.array(
        ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04", "2000-01-05"],
        dtype="datetime64[ns]",
    )
    ds1 = xr.Dataset(
        {
            "cdr_efficiency": ("time", [np.nan, 0.2, 0.3, 0.4, 0.5]),
            "abs_time": ("time", times1),
        },
        coords={"time": times1},
    )

    # Member 2: every 2 days, 4 entries, starts 2000-01-02, first two values NaN
    times2 = np.array(
        ["2000-01-02", "2000-01-04", "2000-01-06", "2000-01-08"], dtype="datetime64[ns]"
    )
    ds2 = xr.Dataset(
        {
            "cdr_efficiency": ("time", [np.nan, np.nan, 0.6, 0.8]),
            "abs_time": ("time", times2),
        },
        coords={"time": times2},
    )

    # Member 3: daily, 3 days, starts 1999-12-31, no NaNs
    times3 = np.array(
        ["1999-12-31", "2000-01-01", "2000-01-02"], dtype="datetime64[ns]"
    )
    ds3 = xr.Dataset(
        {"cdr_efficiency": ("time", [0.05, 0.15, 0.25]), "abs_time": ("time", times3)},
        coords={"time": times3},
    )

    return {"member1": ds1, "member2": ds2, "member3": ds3}


# ----------------------------
# Tests
# ----------------------------


def test_extract_efficiency(create_member_ds: xr.Dataset) -> None:
    ens = Ensemble.__new__(Ensemble)
    eff_rel = ens._extract_efficiency(create_member_ds)

    assert isinstance(eff_rel, xr.DataArray)
    assert np.issubdtype(eff_rel.time.dtype, np.timedelta64)
    assert "abs_time" not in eff_rel.coords
    assert eff_rel.time.attrs.get("long_name") == "time since release start"


def test_extract_efficiency_missing_abs_time() -> None:
    """Test that _extract_efficiency raises an error if 'abs_time' is missing."""
    times = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"cdr_efficiency": ("time", [0.1, 0.2])},
        coords={"time": times},  # Note: no 'abs_time' coordinate
    )

    ens = Ensemble.__new__(Ensemble)
    with pytest.raises(
        ValueError, match="Dataset must contain an 'abs_time' coordinate."
    ):
        ens._extract_efficiency(ds)


def test_align_times_identical(identical_members: dict[str, xr.Dataset]) -> None:
    ens = Ensemble.__new__(Ensemble)
    effs = {
        name: Ensemble._extract_efficiency(ens, ds)
        for name, ds in identical_members.items()
    }
    aligned = ens._align_times(effs)

    assert isinstance(aligned, xr.Dataset)
    for name in identical_members.keys():
        assert name in aligned.data_vars

    # Time dimension matches union of member times
    all_times = np.unique(np.concatenate([eff.time.values for eff in effs.values()]))
    assert len(aligned.time) == len(all_times)


def test_align_times_varied(varied_members: dict[str, xr.Dataset]) -> None:
    ens = Ensemble.__new__(Ensemble)
    effs = {
        name: Ensemble._extract_efficiency(ens, ds)
        for name, ds in varied_members.items()
    }
    aligned = ens._align_times(effs)

    # Check all members exist
    for name in varied_members.keys():
        assert name in aligned.data_vars

    # Time dimension is union of all times
    all_times = np.unique(np.concatenate([eff.time.values for eff in effs.values()]))
    assert len(aligned.time) == len(all_times)

    # Check that for each member, times before first valid value and after last valid value are NaN
    for name, eff in effs.items():
        # Find first and last valid relative times
        valid_mask = ~np.isnan(eff.values)
        first_valid_time = eff.time.values[valid_mask][0]
        last_valid_time = eff.time.values[valid_mask][-1]

        # Times before first valid → should be NaN
        missing_before = aligned.time.values < first_valid_time
        if missing_before.any():
            assert np.all(np.isnan(aligned[name].values[missing_before]))

        # Times after last valid → should be NaN
        missing_after = aligned.time.values > last_valid_time
        if missing_after.any():
            assert np.all(np.isnan(aligned[name].values[missing_after]))


def test_compute_statistics(identical_members: dict[str, xr.Dataset]) -> None:
    ens = Ensemble.__new__(Ensemble)
    effs = {
        name: Ensemble._extract_efficiency(ens, ds)
        for name, ds in identical_members.items()
    }
    aligned = ens._align_times(effs)
    ds_stats = ens._compute_statistics(aligned)

    assert "ensemble_mean" in ds_stats.data_vars
    assert "ensemble_std" in ds_stats.data_vars
    n_time = len(ds_stats.time)
    assert ds_stats.ensemble_mean.shape[0] == n_time
    assert ds_stats.ensemble_std.shape[0] == n_time

    # Ensemble mean should equal the member values
    first_member_name = next(iter(identical_members))
    xr.testing.assert_allclose(ds_stats.ensemble_mean, ds_stats[first_member_name])

    # For identical members, std should be 0
    np.testing.assert_allclose(ds_stats.ensemble_std.values, 0.0)


def test_ensemble_post_init(identical_members: dict[str, xr.Dataset]) -> None:
    ens = Ensemble(identical_members)
    assert isinstance(ens.ds, xr.Dataset)
    assert "ensemble_mean" in ens.ds.data_vars
    assert "ensemble_std" in ens.ds.data_vars
    np.testing.assert_allclose(ens.ds.ensemble_std.values, 0.0)


def test_plot(identical_members: dict[str, xr.Dataset], tmp_path: Path) -> None:
    ens = Ensemble(identical_members)
    save_path = tmp_path / "plot.png"
    ens.plot(save_path=str(save_path))
    assert save_path.exists()


def test_extract_efficiency_empty() -> None:
    # Dataset with all NaN
    times = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"cdr_efficiency": ("time", [np.nan, np.nan]), "abs_time": ("time", times)},
        coords={"time": times},
    )
    ens = Ensemble.__new__(Ensemble)
    with pytest.raises(ValueError):
        ens._extract_efficiency(ds)
