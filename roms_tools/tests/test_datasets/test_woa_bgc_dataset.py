"""Tests for :class:`WOABGCDataset`, the WOA23 1-degree gridded BGC source.

The fixtures synthesise the real WOA23 file layout (one netCDF per variable per
month, plus a full-depth annual file) rather than downloading the ~3 GB product.
The values are deterministic profiles, which is what lets the deep-fill and
month-alignment assertions below be exact rather than approximate.
"""

import logging

import numpy as np
import pytest
import xarray as xr

from roms_tools.datasets.download import WOA23_BGC_VARIABLES, woa23_filename
from roms_tools.datasets.lat_lon_datasets import WOABGCDataset
from roms_tools.setup.utils import compute_potential_density

# The WOA standard depth levels. The 43- and 57-level monthly axes are exact leading
# slices of the 102-level annual axis, which is what makes the splice interpolation-free.
STANDARD_DEPTHS = (
    list(range(0, 105, 5))
    + list(range(125, 525, 25))
    + list(range(550, 2050, 50))
    + list(range(2100, 5600, 100))
)

#: Deepest monthly level per variable code: 800 m for the nutrients, 1500 m for
#: oxygen and T/S. Only the annual field reaches 5500 m.
MONTHLY_LEVELS = {"n": 43, "p": 43, "i": 43, "o": 57, "t": 57, "s": 57}

#: Rough umol/kg magnitudes, so the unit conversion is visible in the output.
BASE_VALUE = {"n": 20.0, "p": 1.5, "i": 40.0, "o": 200.0, "t": 12.0, "s": 34.5}

#: Raw "months since 1965-01-01" offsets. The nutrient and T/S families genuinely
#: disagree in the real files (January is 336.5 for nitrate but 396.5 for
#: temperature), so the fixtures reproduce that and the merge must not align on it.
TIME_ORIGIN = {"n": 336.5, "p": 336.5, "i": 336.5, "o": 336.5, "t": 396.5, "s": 396.5}

#: The deep offset the annual field carries and the monthly field does not. Large
#: enough that a monthly/annual mix-up cannot hide inside the seasonal term.
ANNUAL_OFFSET = 5.0

LAT = np.arange(-89.5, 90, 1.0, dtype="f4")
LON = np.arange(-179.5, 180, 1.0, dtype="f4")


def expected_profile(code: str, period: int, nlev: int) -> np.ndarray:
    """The profile the fixture writes for ``code`` at ``period`` (0 = annual)."""
    z = np.array(STANDARD_DEPTHS[:nlev], dtype="f8")
    seasonal = ANNUAL_OFFSET if period == 0 else np.cos(2 * np.pi * (period - 1) / 12.0)
    return BASE_VALUE[code] + 0.004 * z + seasonal


def _write_file(directory, code, decade, period):
    nlev = 102 if period == 0 else MONTHLY_LEVELS[code]
    profile = expected_profile(code, period, nlev).astype("f4")
    values = np.broadcast_to(
        profile[None, :, None, None], (1, nlev, LAT.size, LON.size)
    ).copy()
    values[:, :, :, :2] = np.nan  # a sliver of land, so the mask is exercised

    ds = xr.Dataset(
        {
            f"{code}_an": (("time", "depth", "lat", "lon"), values),
            # A decoy statistical field, as in the real files. Reading it would be a
            # large waste of I/O, so the class must leave it alone.
            f"{code}_sd": (
                ("time", "depth", "lat", "lon"),
                np.zeros((1, nlev, LAT.size, LON.size), "f4"),
            ),
        },
        coords={
            "time": ("time", np.array([TIME_ORIGIN[code] + period], "f4")),
            "depth": ("depth", np.array(STANDARD_DEPTHS[:nlev], "f4")),
            "lat": ("lat", LAT),
            "lon": ("lon", LON),
        },
    )
    ds["time"].attrs["units"] = "months since 1965-01-01 00:00:00"
    ds[f"{code}_an"].attrs["units"] = "micromoles_per_kilogram"
    ds.to_netcdf(directory / woa23_filename(code, period, decade))


def build_woa_directory(directory, include_annual: bool = True):
    """Write a synthetic WOA23 tree with the real filenames and dimensions."""
    directory.mkdir(parents=True, exist_ok=True)
    for _subdir, decade, code in WOA23_BGC_VARIABLES.values():
        for period in list(range(1, 13)) + ([0] if include_annual else []):
            _write_file(directory, code, decade, period)
    return directory


@pytest.fixture(scope="module")
def woa_dir(tmp_path_factory):
    return build_woa_directory(tmp_path_factory.mktemp("woa_full"))


@pytest.fixture(scope="module")
def woa_dir_monthly_only(tmp_path_factory):
    return build_woa_directory(
        tmp_path_factory.mktemp("woa_monthly"), include_annual=False
    )


@pytest.fixture(scope="module")
def woa_data(woa_dir):
    return WOABGCDataset(filename=woa_dir, use_dask=True)


OCEAN_POINT = {"latitude": 100, "longitude": 200}


def test_merges_into_one_twelve_month_climatology(woa_data):
    """All six variables land on a single 12-step axis, not a 24-step union.

    The nutrient and T/S files carry different raw time values, so merging on them
    would silently produce a union axis padded with NaN.
    """
    ds = woa_data.ds
    assert ds.sizes["time"] == 12
    assert ds.sizes["depth"] == 102
    assert {"n_an", "p_an", "i_an", "o_an", "t_an", "s_an"} <= set(ds.data_vars)
    for name in ("n_an", "t_an"):
        assert ds[name].sizes["time"] == 12
        assert not ds[name].isnull().all(dim=("latitude", "longitude")).any()


def test_decoy_statistical_fields_are_not_read(woa_data):
    """Only the objectively-analyzed ``*_an`` field is loaded from each file."""
    assert not [v for v in woa_data.ds.data_vars if v.endswith("_sd")]


def test_time_axis_is_the_shared_mid_month_climatology(woa_data):
    """The raw per-variable offsets are replaced by one shared mid-month axis."""
    days = (woa_data.ds.time.values / np.timedelta64(1, "D")).astype(int)
    np.testing.assert_array_equal(
        days, [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    )


def test_monthly_tracers_are_paired_with_monthly_temperature_and_salinity(woa_data):
    """Every variable's seasonal cycle must be in phase after the merge.

    The fixture gives all six variables the same ``cos(2*pi*(m-1)/12)`` seasonal term,
    so they must peak in the same month. Aligning on the raw ``time`` values instead
    would offset the tracers from T/S by the 60 months between their start years,
    which is exactly the failure this guards.
    """
    peaks = {
        name: int(woa_data.ds[name].isel(depth=0, **OCEAN_POINT).values.argmax())
        for name in ("n_an", "p_an", "i_an", "o_an", "t_an", "s_an")
    }
    assert len(set(peaks.values())) == 1, f"months are misaligned: {peaks}"
    assert set(peaks.values()) == {0}  # January, per the fixture's cosine


def test_tracers_are_converted_to_mmol_per_m3(woa_data):
    """umol/kg -> mmol/m3 via TEOS-10 sigma-0 from the dataset's own T/S."""
    ds = woa_data.ds
    sigma0 = compute_potential_density(ds["t_an"], ds["s_an"])
    factor = float(((sigma0 + 1000.0) / 1000.0).isel(depth=0, time=0, **OCEAN_POINT))
    assert factor > 1.0

    raw = expected_profile("n", 1, 1)[0]
    got = float(ds["n_an"].isel(depth=0, time=0, **OCEAN_POINT))
    assert got == pytest.approx(raw * factor, rel=1e-5)


def test_temperature_and_salinity_are_left_unconverted(woa_data):
    """T/S are ancillary, not tracers: they must not pick up the density factor."""
    got = float(woa_data.ds["t_an"].isel(depth=0, time=0, **OCEAN_POINT))
    assert got == pytest.approx(expected_profile("t", 1, 1)[0], rel=1e-5)


@pytest.mark.parametrize(
    ("code", "seam"),
    [("t_an", 1500.0), ("n_an", 800.0)],
    ids=["ts_1500m", "nutrient_800m"],
)
def test_annual_blend_is_pure_monthly_above_and_pure_annual_below_the_band(
    woa_dir, code, seam
):
    """The taper band is centred on each variable's own deepest monthly level.

    With the default 100 m half-width that is 700-900 m for the nutrients and
    1400-1600 m for oxygen and T/S -- the band is not a fixed depth range.
    """
    # T/S are never unit-converted, so t_an can be compared against the raw fixture
    # profile directly. For the nutrient case, skip post-processing so the umol/kg ->
    # mmol/m3 factor (which varies with depth) does not muddy the comparison.
    data = WOABGCDataset(
        filename=woa_dir,
        use_dask=True,
        deep_fill="annual_blend",
        apply_post_processing=(code == "t_an"),
    )
    z = data.ds.depth.values
    profile = data.ds[code].isel(time=0, **OCEAN_POINT).values
    letter = code[0]

    nlev = MONTHLY_LEVELS[letter]
    monthly = expected_profile(letter, 1, nlev)
    monthly_ffilled = np.concatenate([monthly, np.full(102 - nlev, monthly[-1])])
    annual = expected_profile(letter, 0, 102)

    above = int(np.argmin(abs(z - (seam - 100))))
    below = int(np.argmin(abs(z - (seam + 100))))
    assert profile[above] == pytest.approx(monthly_ffilled[above], rel=1e-5)
    assert profile[below] == pytest.approx(annual[below], rel=1e-5)
    # And no step discontinuity through the band.
    assert np.all(np.diff(profile[above : below + 1]) > 0)


def test_ffill_holds_the_deepest_monthly_value_to_the_seafloor(woa_dir):
    data = WOABGCDataset(filename=woa_dir, use_dask=True, deep_fill="ffill")
    z = data.ds.depth.values
    profile = data.ds["t_an"].isel(time=0, **OCEAN_POINT).values
    assert data.ds.sizes["depth"] == 102
    seam_index = int(np.argmin(abs(z - 1500)))
    assert np.allclose(profile[seam_index:], profile[seam_index])


def test_annual_blend_degrades_to_ffill_when_the_annual_files_are_absent(
    woa_dir_monthly_only, caplog
):
    """Missing annual files must warn and fall back, not fail."""
    with caplog.at_level(logging.WARNING):
        WOABGCDataset(
            filename=woa_dir_monthly_only, use_dask=True, deep_fill="annual_blend"
        )
    assert any("falling back to 'ffill'" in r.message for r in caplog.records)


def test_missing_required_tracer_files_raise_and_name_the_paths(tmp_path):
    directory = tmp_path / "ts_only"
    directory.mkdir()
    for _subdir, decade, code in WOA23_BGC_VARIABLES.values():
        if code in ("t", "s"):
            for period in list(range(1, 13)) + [0]:
                _write_file(directory, code, decade, period)

    with pytest.raises(FileNotFoundError, match="Required WOA23 files are missing"):
        WOABGCDataset(filename=directory, use_dask=True)


def test_missing_optional_ts_warns_and_leaves_the_tracers_usable(tmp_path, caplog):
    """Without T/S the source still loads; density interpolation falls back later."""
    directory = tmp_path / "tracers_only"
    directory.mkdir()
    for _subdir, decade, code in WOA23_BGC_VARIABLES.values():
        if code not in ("t", "s"):
            for period in list(range(1, 13)) + [0]:
                _write_file(directory, code, decade, period)

    with caplog.at_level(logging.WARNING):
        data = WOABGCDataset(filename=directory, use_dask=True)
    assert "t_an" not in data.ds.data_vars
    assert {"n_an", "p_an", "i_an", "o_an"} <= set(data.ds.data_vars)
    assert any("uniform 1025" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"deep_fill": "bogus"}, "must be 'ffill' or 'annual_blend'"),
        ({"deep_blend_halfwidth": 0}, "must be positive"),
    ],
)
def test_invalid_options_are_rejected(woa_dir, kwargs, match):
    with pytest.raises(ValueError, match=match):
        WOABGCDataset(filename=woa_dir, **kwargs)


def test_mask_marks_the_land_sliver(woa_data):
    """apply_lateral_fill() needs a mask; land must be 0 and open ocean 1."""
    mask = woa_data.ds["mask"]
    assert set(np.unique(mask.values)) <= {0, 1}
    assert int(mask.isel(longitude=0, latitude=100)) == 0
    assert int(mask.isel(**OCEAN_POINT)) == 1


# ---------------------------------------------------------------------------
# Pipeline integration: WOA as a registered BGC source
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _roms_grid():
    from roms_tools import Grid

    return Grid(
        nx=2,
        ny=2,
        size_x=500,
        size_y=1000,
        center_lon=0,
        center_lat=55,
        rot=10,
        N=3,
        theta_s=5.0,
        theta_b=2.0,
        hc=250.0,
    )


@pytest.fixture(scope="module")
def _physics_path():
    from roms_tools.datasets.download import download_test_data

    return download_test_data("GLORYS_coarse_test_data.nc")


def test_woa_is_registered_for_both_initial_and_boundary_bgc():
    from roms_tools.setup.boundary_forcing import (
        _BGC_SOURCE_NAMES as bf_names,
    )
    from roms_tools.setup.boundary_forcing import (
        _DATASET_MAP as bf_map,
    )
    from roms_tools.setup.initial_conditions import (
        _BGC_SOURCE_NAMES as ic_names,
    )
    from roms_tools.setup.initial_conditions import (
        _DATASET_MAP as ic_map,
    )

    assert "WOA" in bf_map["bgc"] and "WOA" in bf_names
    assert "WOA" in ic_map["bgc"] and "WOA" in ic_names


@pytest.mark.parametrize("method", ["depth", "density", "density_mld"])
def test_initial_conditions_from_woa(
    _roms_grid, _physics_path, woa_dir, method, caplog
):
    """WOA drives initial conditions under every BGC interpolation method.

    The density methods need T/S from both sides. If WOA's ancillary T/S were not
    wired up, roms-tools would log a fallback and silently interpolate in depth
    space instead, so the absence of that message is the real assertion here.
    """
    from datetime import datetime

    from roms_tools.setup.bgc_model import BGCMarbl
    from roms_tools.setup.initial_conditions import InitialConditions

    with caplog.at_level(logging.INFO):
        ic = InitialConditions(
            grid=_roms_grid,
            ini_time=datetime(2021, 6, 29),
            source={"path": _physics_path, "name": "GLORYS"},
            bgc_sources=[{"source": {"name": "WOA", "path": woa_dir}}],
            bgc_model=BGCMarbl,
            bgc_interpolation_method=method,
            prefill="2d_lateral_fill",
            regrid_method="scipy",
            use_dask=True,
        )

    for tracer in ("NO3", "PO4", "SiO3", "O2"):
        assert tracer in ic.ds, f"{tracer} missing from the initial conditions"
        assert not np.isnan(ic.ds[tracer].values).any()

    # The ancillary T/S are an implementation detail and must not be written out.
    assert not [
        v for v in ic.ds.data_vars if v in ("t_an", "s_an", "temp_bgc", "salt_bgc")
    ]

    if method != "depth":
        assert not [
            r for r in caplog.records if "falling back to depth-space" in r.message
        ], "density interpolation silently degraded; the WOA T/S are not reaching it"


def test_boundary_forcing_from_woa(_roms_grid, _physics_path, woa_dir):
    from datetime import datetime

    from roms_tools.setup.bgc_model import BGCMarbl
    from roms_tools.setup.boundary_forcing import BoundaryForcing

    bf = BoundaryForcing(
        grid=_roms_grid,
        start_time=datetime(2021, 6, 29),
        end_time=datetime(2021, 6, 30),
        source={"path": _physics_path, "name": "GLORYS"},
        bgc_sources=[{"source": {"name": "WOA", "path": woa_dir}}],
        bgc_model=BGCMarbl,
        bgc_interpolation_method="density_mld",
        prefill="2d_lateral_fill",
        regrid_method="scipy",
        use_dask=True,
    )

    ds = bf.bgc[0].ds
    for direction in ("north", "south", "east"):
        name = f"NO3_{direction}"
        assert name in ds
        # A climatology source keeps all twelve months on the boundary.
        assert ds[name].sizes["bry_time"] == 12
        assert not np.isnan(ds[name].values).any()


def test_climatology_defaults_to_true_for_woa(_roms_grid, woa_dir):
    """WOA only exists as a 12-month climatology, so omitting the flag must work.

    Defaulting it to False (as for the other sources) fails later with a confusing
    message about integer time values.
    """
    from datetime import datetime

    from roms_tools.setup.boundary_forcing import BoundaryForcingSource

    source = {"name": "WOA", "path": woa_dir}
    BoundaryForcingSource(
        grid=_roms_grid,
        start_time=datetime(2021, 6, 29),
        end_time=datetime(2021, 6, 30),
        type="bgc",
        source=source,
        use_dask=True,
        regrid_method="scipy",
    )
    assert source["climatology"] is True
