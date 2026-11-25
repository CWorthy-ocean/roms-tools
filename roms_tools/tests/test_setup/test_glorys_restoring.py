"""Tests for GLORYS as a source of surface-salinity restoring."""

import calendar
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from roms_tools import Grid, SurfaceForcing
from roms_tools.datasets.download import download_test_data
from roms_tools.datasets.lat_lon_datasets import (
    GLORYS_RESTORING_TIME_RESOLUTIONS,
    GLORYSRestoringSurfaceDataset,
)

MODEL_REF = datetime(2000, 1, 1)
SURFACE_DEPTH = 0.494025


@pytest.fixture(scope="module")
def grid() -> Grid:
    return Grid(
        nx=3, ny=3, size_x=400, size_y=400, center_lon=-8, center_lat=58, rot=0, N=3
    )


def _surface_so(day: datetime) -> float:
    return 30.0 + day.timetuple().tm_yday * 0.01


def _daily_dataset(days: list[datetime]) -> xr.Dataset:
    lat = np.arange(52.0, 64.01, 1.0)
    lon = np.arange(-16.0, 0.01, 1.0)
    depth = np.array([SURFACE_DEPTH, 10.0])
    shape = (len(days), 2, lat.size, lon.size)
    so = np.empty(shape)
    for i, day in enumerate(days):
        so[i, 0] = _surface_so(day)
        so[i, 1] = 99.0
    so[:, :, 0, 0] = np.nan  # a land point far from the target grid
    return xr.Dataset(
        {
            "so": (("time", "depth", "latitude", "longitude"), so),
            "zos": (
                ("time", "latitude", "longitude"),
                np.zeros((len(days), *shape[2:])),
            ),
        },
        coords={
            "time": pd.DatetimeIndex(days),
            "depth": depth,
            "latitude": lat,
            "longitude": lon,
        },
    )


def _write(tmp_path, days: list[datetime], use_dask: bool):
    """Write daily records; one file per record with dask, else a single file."""
    ds = _daily_dataset(days)
    if not use_dask:
        path = tmp_path / "glorys.nc"
        ds.to_netcdf(path)
        return path
    paths = []
    for i in range(len(days)):
        path = tmp_path / f"glorys_{i:04d}.nc"
        ds.isel(time=[i]).to_netcdf(path)
        paths.append(path)
    return paths


def _days(start: datetime, end: datetime) -> list[datetime]:
    n = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(n)]


def _month_mean(year: int, month: int) -> float:
    n = calendar.monthrange(year, month)[1]
    return float(np.mean([_surface_so(datetime(year, month, d + 1)) for d in range(n)]))


def _midpoint_days(year: int, month: int) -> float:
    start = datetime(year, month, 1)
    nxt = datetime(year + month // 12, month % 12 + 1, 1)
    return ((start + (nxt - start) / 2) - MODEL_REF) / timedelta(days=1)


def _surface_forcing(grid, path, start, end, use_dask, **source):
    return SurfaceForcing(
        grid=grid,
        type="restoring",
        restoring_forces=["sss"],
        source={"name": "GLORYS", "path": path, **source},
        start_time=start,
        end_time=end,
        coarse_grid_mode="never",
        model_reference_date=MODEL_REF,
        use_dask=use_dask,
    )


def test_monthly_restoring_matches_monthly_means(grid, tmp_path, use_dask):
    path = _write(
        tmp_path, _days(datetime(2015, 12, 1), datetime(2016, 3, 31)), use_dask
    )
    sf = _surface_forcing(
        grid, path, datetime(2016, 1, 20), datetime(2016, 2, 10), use_dask
    )

    assert sf.ds["sss_time"].dims == ("time",)
    np.testing.assert_allclose(
        sf.ds["sss_time"].values, [_midpoint_days(2016, 1), _midpoint_days(2016, 2)]
    )
    vals = sf.ds["sss"].values
    np.testing.assert_allclose(vals[0], _month_mean(2016, 1), atol=1e-5)
    np.testing.assert_allclose(vals[1], _month_mean(2016, 2), atol=1e-5)
    assert "cycle_length" not in sf.ds["sss_time"].attrs
    assert "climatology" not in sf.ds.attrs


@pytest.mark.parametrize(
    "start, end, start_pad, end_pad, expected",
    [
        # Jan 10 is before January's midpoint, so padding needs December.
        (datetime(2016, 1, 10), datetime(2016, 2, 10), True, True, (12, 2)),
        (datetime(2016, 1, 20), datetime(2016, 2, 10), True, True, (1, 2)),
        (datetime(2016, 1, 10), datetime(2016, 2, 20), False, False, (1, 2)),
        (datetime(2016, 1, 20), datetime(2016, 2, 20), False, False, (2, 2)),
        (datetime(2016, 1, 10), datetime(2016, 2, 20), False, True, (1, 3)),
        (datetime(2016, 1, 10), datetime(2016, 3, 20), True, False, (12, 3)),
    ],
)
def test_monthly_window_rule(
    tmp_path, use_dask, start, end, start_pad, end_pad, expected
):
    path = _write(
        tmp_path, _days(datetime(2015, 11, 1), datetime(2016, 4, 30)), use_dask
    )
    data = GLORYSRestoringSurfaceDataset(
        filename=path,
        start_time=start,
        end_time=end,
        start_time_pad=start_pad,
        end_time_pad=end_pad,
        use_dask=use_dask,
    )
    months = pd.DatetimeIndex(data.ds["time"].values).month
    assert (months[0], months[-1]) == expected
    assert "depth" not in data.dim_names
    assert "depth" not in data.ds.coords


def _days_since_ref(t: datetime) -> float:
    return (t - MODEL_REF) / timedelta(days=1)


def test_monthly_gap_raises(grid, tmp_path, use_dask):
    days = [
        d
        for d in _days(datetime(2015, 12, 1), datetime(2016, 3, 31))
        if d != datetime(2016, 2, 12)
    ]
    path = _write(tmp_path, days, use_dask)
    with pytest.raises(ValueError, match=r"missing between: 2016-02-11 to 2016-02-13"):
        _surface_forcing(
            grid, path, datetime(2016, 1, 20), datetime(2016, 2, 20), use_dask
        )


def test_monthly_duplicate_day_raises(grid, tmp_path, use_dask):
    days = _days(datetime(2015, 12, 1), datetime(2016, 3, 31))
    # Same date, different stamp (an identical stamp is dropped by the loader).
    days.append(datetime(2016, 1, 5, 12))
    days.sort()
    path = _write(tmp_path, days, use_dask)
    with pytest.raises(ValueError, match=r"more than once.*2016-01-05"):
        _surface_forcing(
            grid, path, datetime(2016, 1, 20), datetime(2016, 2, 10), use_dask
        )


def test_monthly_partial_edge_months_stay_daily(grid, tmp_path, use_dask):
    """Files for the run window plus a day (as Forge stages them)."""
    path = _write(
        tmp_path, _days(datetime(2015, 12, 28), datetime(2016, 2, 1)), use_dask
    )
    sf = _surface_forcing(
        grid, path, datetime(2015, 12, 29), datetime(2016, 1, 31), use_dask
    )
    daily = [
        *_days(datetime(2015, 12, 28), datetime(2015, 12, 31)),
        datetime(2016, 2, 1),
    ]
    expected_times = sorted(
        [_days_since_ref(d) for d in daily] + [_midpoint_days(2016, 1)]
    )
    np.testing.assert_allclose(sf.ds["sss_time"].values, expected_times)
    vals = sf.ds["sss"].values[:, 1, 1]
    expected_vals = [_surface_so(d) for d in daily[:4]]
    expected_vals += [_month_mean(2016, 1), _surface_so(daily[4])]
    np.testing.assert_allclose(vals, expected_vals, atol=1e-5)


def test_monthly_complete_edge_month_stays_daily_when_needed(grid, tmp_path, use_dask):
    # The padded window wants December, which is absent; January is complete but
    # its mid-month record would come after start_time, so its days stay daily.
    path = _write(
        tmp_path, _days(datetime(2016, 1, 1), datetime(2016, 3, 31)), use_dask
    )
    sf = _surface_forcing(
        grid, path, datetime(2016, 1, 10), datetime(2016, 2, 10), use_dask
    )
    times = sf.ds["sss_time"].values
    assert len(times) == 32
    np.testing.assert_allclose(times[0], _days_since_ref(datetime(2016, 1, 1)))
    np.testing.assert_allclose(times[-1], _midpoint_days(2016, 2))


def test_monthly_files_not_reaching_start_raise(grid, tmp_path, use_dask):
    path = _write(
        tmp_path, _days(datetime(2016, 1, 15), datetime(2016, 3, 31)), use_dask
    )
    with pytest.raises(ValueError, match=r"after `start_time`"):
        _surface_forcing(
            grid, path, datetime(2016, 1, 10), datetime(2016, 2, 10), use_dask
        )


def test_monthly_forge_staged_year(tmp_path, use_dask):
    """A 1993-01-02 to 1994-01-01 run with files for the run window plus a day."""
    path = _write(tmp_path, _days(datetime(1993, 1, 1), datetime(1994, 1, 2)), use_dask)
    data = GLORYSRestoringSurfaceDataset(
        filename=path,
        start_time=datetime(1993, 1, 2),
        end_time=datetime(1994, 1, 1),
        use_dask=use_dask,
    )
    times = pd.DatetimeIndex(data.ds["time"].values)
    # January stays daily: its mid-month record would come after start_time.
    expected = [
        *_days(datetime(1993, 1, 1), datetime(1993, 1, 31)),
        *[
            datetime(1993, m, 1)
            + (datetime(1993 + m // 12, m % 12 + 1, 1) - datetime(1993, m, 1)) / 2
            for m in range(2, 13)
        ],
        datetime(1994, 1, 1),
        datetime(1994, 1, 2),
    ]
    assert list(times) == [pd.Timestamp(t) for t in expected]


def test_monthly_unpadded_window_without_a_midpoint_raises(tmp_path, use_dask):
    path = _write(
        tmp_path, _days(datetime(2016, 1, 1), datetime(2016, 3, 31)), use_dask
    )
    with pytest.raises(ValueError, match="No month midpoint"):
        GLORYSRestoringSurfaceDataset(
            filename=path,
            start_time=datetime(2016, 1, 20),
            end_time=datetime(2016, 2, 10),
            start_time_pad=False,
            end_time_pad=False,
            use_dask=use_dask,
        )


def test_monthly_without_time_window_checks_gaps(tmp_path):
    days = [
        d
        for d in _days(datetime(2016, 1, 1), datetime(2016, 3, 31))
        if d != datetime(2016, 2, 12)
    ]
    path = _write(tmp_path, days, False)
    with pytest.raises(ValueError, match="missing between"):
        GLORYSRestoringSurfaceDataset(filename=path)


def test_daily_restoring_uses_top_level(grid, tmp_path, use_dask):
    path = _write(
        tmp_path, _days(datetime(2016, 1, 1), datetime(2016, 1, 31)), use_dask
    )
    sf = _surface_forcing(
        grid,
        path,
        datetime(2016, 1, 10),
        datetime(2016, 1, 12),
        use_dask,
        time_resolution="daily",
    )
    assert sf.source["time_resolution"] == "daily"
    assert sf.ds.sizes["time"] == 5  # padded by one record each side
    expected = [_surface_so(datetime(2016, 1, d)) for d in range(9, 14)]
    np.testing.assert_allclose(
        sf.ds["sss"].values.mean(axis=(1, 2)), expected, atol=1e-5
    )


def test_daily_restoring_with_test_data(grid, use_dask):
    path = [
        download_test_data("GLORYS_NA_20120101.nc"),
        download_test_data("GLORYS_NA_20121231.nc"),
    ]
    if not use_dask:
        pytest.skip("a list of files requires use_dask=True")
    sf = _surface_forcing(
        grid,
        path,
        datetime(2012, 1, 1),
        datetime(2012, 12, 31),
        use_dask,
        time_resolution="daily",
    )
    assert sf.ds.sizes["time"] == 2
    assert not np.isnan(sf.ds["sss"].values).any()


def test_monthly_stays_lazy(tmp_path):
    path = _write(tmp_path, _days(datetime(2015, 12, 1), datetime(2016, 3, 31)), True)
    data = GLORYSRestoringSurfaceDataset(
        filename=path,
        start_time=datetime(2016, 1, 20),
        end_time=datetime(2016, 2, 10),
        use_dask=True,
    )
    assert data.ds["so"].chunks is not None
    assert data.ds["mask"].chunks is not None
    # Default chunks hold one depth level, so the surface read skips the rest.
    assert data.chunks is not None
    assert data.chunks["depth"] == 1


def test_rejections(grid, tmp_path):
    path = _write(tmp_path, _days(datetime(2016, 1, 1), datetime(2016, 1, 31)), False)
    start, end = datetime(2016, 1, 10), datetime(2016, 1, 12)

    with pytest.raises(ValueError, match="climatology"):
        _surface_forcing(grid, path, start, end, False, climatology=True)
    with pytest.raises(ValueError, match="time_resolution"):
        _surface_forcing(grid, path, start, end, False, time_resolution="weekly")
    with pytest.raises(ValueError, match="time_resolution"):
        SurfaceForcing(
            grid=grid,
            type="restoring",
            restoring_forces=["sss"],
            source={"name": "WOA", "path": path, "time_resolution": "daily"},
            start_time=start,
            end_time=end,
            use_dask=False,
        )
    with pytest.raises(ValueError, match="restoring_forces"):
        SurfaceForcing(
            grid=grid,
            type="restoring",
            restoring_forces=["sDIC", "sALK"],
            source={"name": "GLORYS", "path": path},
            start_time=start,
            end_time=end,
            use_dask=False,
        )
    with pytest.raises(ValueError, match="must include a 'path'"):
        SurfaceForcing(
            grid=grid,
            type="restoring",
            restoring_forces=["sss"],
            source={"name": "GLORYS"},
            start_time=start,
            end_time=end,
            use_dask=False,
        )


def test_dataset_validates_options(tmp_path):
    path = _write(tmp_path, _days(datetime(2016, 1, 1), datetime(2016, 1, 31)), False)
    kwargs = {"filename": path, "start_time": datetime(2016, 1, 10)}
    with pytest.raises(ValueError, match="not supported"):
        GLORYSRestoringSurfaceDataset(climatology=True, **kwargs)
    with pytest.raises(ValueError, match="monthly"):
        GLORYSRestoringSurfaceDataset(time_resolution="weekly", **kwargs)
    assert GLORYS_RESTORING_TIME_RESOLUTIONS[0] == "monthly"


@pytest.mark.parametrize("resolution", [None, "daily"])
def test_yaml_roundtrip_keeps_time_resolution(grid, tmp_path, use_dask, resolution):
    path = _write(
        tmp_path, _days(datetime(2015, 12, 1), datetime(2016, 3, 31)), use_dask
    )
    extra = {} if resolution is None else {"time_resolution": resolution}
    sf = _surface_forcing(
        grid, path, datetime(2016, 1, 20), datetime(2016, 2, 10), use_dask, **extra
    )
    assert ("time_resolution" in sf.source) == (resolution is not None)

    yaml_path = tmp_path / "sf.yaml"
    sf.to_yaml(yaml_path)
    rebuilt = SurfaceForcing.from_yaml(yaml_path, use_dask=use_dask)
    assert rebuilt.source.get("time_resolution") == resolution
    assert rebuilt == sf
