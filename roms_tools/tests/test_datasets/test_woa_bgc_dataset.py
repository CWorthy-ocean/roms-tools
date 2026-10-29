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
from roms_tools.fill import BOTTOM_FILL_BLEND_M
from roms_tools.setup.utils import compute_in_situ_density

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


def _write_file(directory, code, decade, period, lat=LAT, lon=LON, field_fn=None):
    """Write one WOA23 file.

    ``field_fn(code, period, values, depth)``, when given, edits ``values`` in place
    instead of the default sliver of land, to shape the columns.
    """
    nlev = 102 if period == 0 else MONTHLY_LEVELS[code]
    profile = expected_profile(code, period, nlev).astype("f4")
    values = np.broadcast_to(
        profile[None, :, None, None], (1, nlev, lat.size, lon.size)
    ).copy()
    depth = np.array(STANDARD_DEPTHS[:nlev], "f4")
    if field_fn is None:
        values[:, :, :, :2] = np.nan  # a sliver of land, so the mask is exercised
    else:
        field_fn(code, period, values, depth)

    ds = xr.Dataset(
        {
            f"{code}_an": (("time", "depth", "lat", "lon"), values),
            # A decoy statistical field, as in the real files. Reading it would be a
            # large waste of I/O, so the class must leave it alone.
            f"{code}_sd": (
                ("time", "depth", "lat", "lon"),
                np.zeros((1, nlev, lat.size, lon.size), "f4"),
            ),
        },
        coords={
            "time": ("time", np.array([TIME_ORIGIN[code] + period], "f4")),
            "depth": ("depth", depth),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
    )
    ds["time"].attrs["units"] = "months since 1965-01-01 00:00:00"
    ds[f"{code}_an"].attrs["units"] = "micromoles_per_kilogram"
    ds.to_netcdf(directory / woa23_filename(code, period, decade))


def build_woa_directory(
    directory, include_annual: bool = True, lat=LAT, lon=LON, field_fn=None
):
    """Write a synthetic WOA23 tree with the real filenames and dimensions."""
    directory.mkdir(parents=True, exist_ok=True)
    for _subdir, decade, code in WOA23_BGC_VARIABLES.values():
        for period in list(range(1, 13)) + ([0] if include_annual else []):
            _write_file(directory, code, decade, period, lat, lon, field_fn)
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


@pytest.fixture(scope="module")
def woa_dir_annual_nan_at_600m(tmp_path_factory):
    """A WOA tree where the annual temperature file is deliberately NaN at 600 m.

    600 m is deep within the pure-monthly region for T/S (monthly T/S extends to
    1500 m; the annual_blend taper only starts at 1400 m), so ``weight`` there is
    exactly 1. Before the ``annual.fillna(extended)`` fix, ``NaN * 0.0`` still
    poisons the blend with NaN even at weight 1 -- exercising that fix requires a
    NaN strictly outside the taper band, not one inside it.
    """
    directory = build_woa_directory(tmp_path_factory.mktemp("woa_annual_nan"))
    path = directory / woa23_filename("t", 0, "decav")
    ds = xr.open_dataset(path, decode_times=False).load()
    ds.close()
    depth_index = int(np.argmin(np.abs(ds["depth"].values - 600.0)))
    ds["t_an"].values[:, depth_index, :, :] = np.nan
    ds.to_netcdf(path, mode="w")
    return directory


def test_annual_nan_does_not_poison_a_valid_monthly_value(woa_dir_annual_nan_at_600m):
    """A NaN in the annual field must not overwrite a valid monthly value.

    Without ``annual.fillna(extended)`` in ``_extend_to_full_depth``, ``NaN *
    0.0`` is NaN (not 0), so a NaN annual value poisons the blended result even
    at weight 1 (pure monthly). Post-processing is skipped so the umol/kg ->
    mmol/m3 factor does not muddy the direct value comparison (T/S themselves
    are never unit-converted regardless).
    """
    data = WOABGCDataset(
        filename=woa_dir_annual_nan_at_600m,
        use_dask=True,
        deep_fill="annual_blend",
        apply_post_processing=False,
    )
    depth_index = int(np.argmin(np.abs(data.ds.depth.values - 600.0)))
    got = float(data.ds["t_an"].isel(depth=depth_index, time=0, **OCEAN_POINT))
    assert np.isfinite(got)
    assert got == pytest.approx(expected_profile("t", 1, 57)[38], rel=1e-5)


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


def test_use_dask_false_produces_numpy_backed_arrays(woa_dir):
    """`use_dask=False` must behave eagerly, like every other dataset class here.

    `_open_monthly`'s `xr.open_mfdataset` is always dask-backed regardless of
    `use_dask`, so `load_data` must `.load()` the merged result when
    `use_dask=False` or this class would silently stay lazy against the caller's
    request.
    """
    data = WOABGCDataset(filename=woa_dir, use_dask=False)
    for var_name in ("n_an", "p_an", "i_an", "o_an", "t_an", "s_an"):
        assert not hasattr(data.ds[var_name].data, "chunks"), (
            f"{var_name} is still dask-backed with use_dask=False"
        )


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
    """umol/kg -> mmol/m3 via in-situ density (TEOS-10 gsw.rho) from the dataset's T/S.

    Uses a non-zero depth index deliberately: at depth=0 in-situ density and
    sigma-0 density coincide (pressure is zero at the surface), so asserting
    there would not distinguish the two formulas.
    """
    ds = woa_data.ds
    depth_index = 10
    density = compute_in_situ_density(
        ds["t_an"], ds["s_an"], ds["depth"], ds["latitude"]
    )
    factor = float((density / 1000.0).isel(depth=depth_index, time=0, **OCEAN_POINT))
    assert factor > 1.0

    raw = expected_profile("n", 1, depth_index + 1)[-1]
    got = float(ds["n_an"].isel(depth=depth_index, time=0, **OCEAN_POINT))
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
            for period in [*range(1, 13), 0]:
                _write_file(directory, code, decade, period)

    with pytest.raises(FileNotFoundError, match="Required WOA23 files are missing"):
        WOABGCDataset(filename=directory, use_dask=True)


def test_missing_optional_ts_warns_and_leaves_the_tracers_usable(tmp_path, caplog):
    """Without T/S the source still loads; density interpolation falls back later."""
    directory = tmp_path / "tracers_only"
    directory.mkdir()
    for _subdir, decade, code in WOA23_BGC_VARIABLES.values():
        if code not in ("t", "s"):
            for period in [*range(1, 13), 0]:
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
# Columns that end at different depths: the splice, the fill below the seafloor
# and the unit conversion
# ---------------------------------------------------------------------------

#: A small regional grid, 2 degrees apart in latitude and 1 in longitude, so that along
#: a row the nearest column is unambiguous.
COLUMN_LAT = np.array([0.5, 2.5, 4.5], dtype="f4")
COLUMN_LON = np.arange(0.5, 6.0, 1.0, dtype="f4")

#: Columns that are not simply water at every level, (latitude index, longitude index).
SHALLOW = (1, 0)  # no data below 50 m, in the annual and in every monthly file
SHORT_MONTHLY = (0, 3)  # monthly data only down to 100 m, annual data at every level
SHORT_ANNUAL = (2, 4)  # annual data only down to 300 m, monthly data to its own bottom
SURFACE_GAP = (0, 5)  # no data above 10 m
LAND = (2, 5)
T_HOLE = (1, 4)  # no temperature at 500 m, though every tracer has data there
HOLE_DEPTH = 500.0
#: The columns next to SHALLOW and SHORT_ANNUAL, which have water at every level.
NEAR_SHALLOW, NEAR_SHORT_ANNUAL = (1, 1), (2, 3)
#: Deepest monthly level of each tracer, which SHORT_ANNUAL's data ends at.
MONTHLY_BOTTOM = {"n_an": 800.0, "o_an": 1500.0}


def _shape_columns(code, period, values, z):
    """Give the small grid the columns above; ``values`` is edited in place."""
    if code in "npio":
        # Make the columns differ, so a filled cell shows which column it came from.
        lat_index, lon_index = np.indices(values.shape[2:])
        values += (10.0 * lon_index + lat_index).astype("f4")
    values[:, z > 50.0, SHALLOW[0], SHALLOW[1]] = np.nan
    if period > 0:
        values[:, z > 100.0, SHORT_MONTHLY[0], SHORT_MONTHLY[1]] = np.nan
    else:
        values[:, z > 300.0, SHORT_ANNUAL[0], SHORT_ANNUAL[1]] = np.nan
    values[:, z < 10.0, SURFACE_GAP[0], SURFACE_GAP[1]] = np.nan
    values[:, :, LAND[0], LAND[1]] = np.nan
    if code == "t":
        values[:, z == HOLE_DEPTH, T_HOLE[0], T_HOLE[1]] = np.nan


@pytest.fixture(scope="module")
def woa_dir_columns(tmp_path_factory):
    return build_woa_directory(
        tmp_path_factory.mktemp("woa_columns"),
        lat=COLUMN_LAT,
        lon=COLUMN_LON,
        field_fn=_shape_columns,
    )


def _load_columns(directory, use_dask, **kwargs):
    """The spliced fields, not unit-converted, as ``extrapolate_deepest_to_bottom``
    receives them from the pipeline unless ``apply_post_processing`` is given.
    """
    kwargs.setdefault("apply_post_processing", False)
    return WOABGCDataset(filename=directory, use_dask=use_dask, **kwargs)


def _in_order(da):
    """The values of a field as (time, depth, latitude, longitude)."""
    lat, lon = (d for d in da.dims if d in ("lat", "latitude", "lon", "longitude"))
    return da.transpose("time", "depth", lat, lon).values


def _read_raw(directory, code):
    """One variable's monthly fields (time, depth, lat, lon) and annual field."""
    decade = {c: d for _subdir, d, c in WOA23_BGC_VARIABLES.values()}[code]

    def read(period):
        path = directory / woa23_filename(code, period, decade)
        with xr.open_dataset(path, decode_times=False) as ds:
            return ds[f"{code}_an"].isel(time=0, drop=True).load()

    return xr.concat([read(m) for m in range(1, 13)], dim="time"), read(0)


def _old_annual_blend(directory, code, halfwidth=100.0):
    """The ``annual_blend`` field as it was before cells below both seafloors were
    masked, rebuilt from the raw files, and the cells where either file has data.
    """
    monthly, annual = _read_raw(directory, code)
    on_annual_axis = monthly.reindex(depth=annual["depth"])
    seam = float(monthly["depth"].max())
    weight = ((seam + halfwidth - annual["depth"]) / (2 * halfwidth)).clip(0.0, 1.0)
    extended = on_annual_axis.ffill("depth").fillna(annual)
    old = extended * weight + annual.fillna(extended) * (1.0 - weight)
    has_data = annual.notnull() | on_annual_axis.notnull()
    return _in_order(old), _in_order(has_data)


@pytest.mark.parametrize(
    ("code", "monthly_bottom"), [("n", 800.0), ("t", 1500.0)], ids=["n", "t"]
)
def test_annual_blend_leaves_nan_below_the_seafloor_of_both_fields(
    woa_dir_columns, use_dask, code, monthly_bottom
):
    """Where the annual and the monthly field both lack data the spliced field is NaN
    (it used to hold a copy of the monthly deepest value); every cell where either has
    data keeps the value the old formula gave it.
    """
    data = _load_columns(woa_dir_columns, use_dask)
    got = _in_order(data.ds[f"{code}_an"])
    z = data.ds["depth"].values
    old, has_data = _old_annual_blend(woa_dir_columns, code)

    np.testing.assert_array_equal(got[has_data], old[has_data])
    assert np.isnan(got[~has_data]).all()
    # The old formula did put values in those cells, so this checks the masking.
    assert np.isfinite(old[~has_data]).any()

    def present(column):
        return np.isfinite(got[:, :, column[0], column[1]]).all(axis=0)

    np.testing.assert_array_equal(present(SHALLOW), z <= 50.0)
    np.testing.assert_array_equal(present(SHORT_ANNUAL), z <= monthly_bottom)
    assert present(SHORT_MONTHLY).all()  # the annual field reaches the seafloor
    assert not np.isfinite(got[:, :, LAND[0], LAND[1]]).any()
    if code == "t":
        np.testing.assert_array_equal(present(T_HOLE), z != HOLE_DEPTH)


@pytest.mark.parametrize("blend_m", [0.0, BOTTOM_FILL_BLEND_M])
@pytest.mark.parametrize(
    ("column", "donor"),
    [(SHALLOW, NEAR_SHALLOW), (SHORT_ANNUAL, NEAR_SHORT_ANNUAL)],
    ids=["shallow", "short_annual"],
)
@pytest.mark.parametrize("name", ["n_an", "o_an"])
def test_cells_below_the_seafloor_take_the_nearest_columns_value_at_that_depth(
    woa_dir_columns, use_dask, name, column, donor, blend_m
):
    """A cell below a column's seafloor takes the value, at its own depth, of the
    nearest column that has water there, blended in from the column's deepest value
    over ``blend_m`` (``0`` takes the neighbour's value at once). The old fill copied
    the deepest value down to the bottom of the grid.
    """
    data = _load_columns(woa_dir_columns, use_dask)
    before = _in_order(data.ds[name].copy(deep=True))
    data.extrapolate_deepest_to_bottom(blend_m=blend_m)
    after = _in_order(data.ds[name])

    z = data.ds["depth"].values
    bottom_depth = 50.0 if column == SHALLOW else MONTHLY_BOTTOM[name]
    bottom = int(np.flatnonzero(z <= bottom_depth)[-1])
    below = slice(bottom + 1, None)

    own = before[:, bottom, column[0], column[1]][:, None]
    neighbour = before[:, below, donor[0], donor[1]]
    dz = z[below] - z[bottom]
    weight = np.zeros_like(dz) if blend_m == 0 else np.clip(1.0 - dz / blend_m, 0, 1)
    expected = weight * own + (1.0 - weight) * neighbour
    np.testing.assert_allclose(
        after[:, below, column[0], column[1]], expected, rtol=1e-6
    )
    # Not the deepest value copied down, which is what the blend starts from.
    assert not np.allclose(after[:, below, column[0], column[1]], own)
    # The cells above the seafloor, and every other column, are untouched.
    unchanged = np.isfinite(before)
    np.testing.assert_array_equal(after[unchanged], before[unchanged])


def test_no_nan_is_left_in_a_column_with_water(woa_dir_columns, use_dask):
    """After the fill every variable is finite in every column with water, at every
    depth and month, including the one without a temperature at 500 m and the one
    without data above 10 m; land stays NaN.
    """
    data = _load_columns(woa_dir_columns, use_dask, apply_post_processing=True)
    data.extrapolate_deepest_to_bottom()

    wet = np.ones((COLUMN_LAT.size, COLUMN_LON.size), dtype=bool)
    wet[LAND] = False
    for name in data.ds.data_vars:
        if name.startswith("mask"):
            continue
        values = _in_order(data.ds[name])
        assert np.isfinite(values[:, :, wet]).all(), name
        assert np.isnan(values[:, :, ~wet]).all(), name


def test_infinite_blend_reproduces_the_forward_then_backward_fill(
    woa_dir_columns, use_dask
):
    """``blend_m=inf`` is exactly a forward then a backward fill along depth, the
    fill before the same-depth fill, and the mask is built from the filled first
    tracer.
    """
    data = _load_columns(woa_dir_columns, use_dask)
    before = data.ds.copy(deep=True)
    data.extrapolate_deepest_to_bottom(blend_m=np.inf)

    expected = before.ffill("depth").bfill("depth")
    for name in expected.data_vars:
        np.testing.assert_array_equal(
            _in_order(data.ds[name]), _in_order(expected[name])
        )
    first = next(iter(expected.data_vars))
    surface = expected[first].isel(depth=0, time=0, drop=True)
    np.testing.assert_array_equal(
        data.ds["mask"].values, xr.where(surface.isnull(), 0, 1).values
    )


def test_ffill_mode_masks_cells_below_the_seafloor_of_both_fields(
    woa_dir_columns, use_dask
):
    """With the annual files present, ``deep_fill="ffill"`` holds the deepest monthly
    value down only where the annual or the monthly field has data. Below the seafloor
    of both the cell is NaN, and the fill below the seafloor then gives it the
    same-depth value of the nearest column, as in ``annual_blend``.
    """
    data = _load_columns(woa_dir_columns, use_dask, deep_fill="ffill")
    z = data.ds["depth"].values
    for code, monthly_bottom in (("n", 800.0), ("t", 1500.0)):
        monthly, annual = _read_raw(woa_dir_columns, code)
        on_annual_axis = monthly.reindex(depth=annual["depth"])
        expected = _in_order(on_annual_axis.ffill("depth"))
        has_data = _in_order(annual.notnull() | on_annual_axis.notnull())
        got = _in_order(data.ds[f"{code}_an"])

        # Where either field has data: the forward-filled monthly field, exactly.
        np.testing.assert_array_equal(got[has_data], expected[has_data])
        # Below the seafloor of both: NaN, where a plain forward fill held a value.
        assert np.isnan(got[~has_data]).all()
        assert np.isfinite(expected[~has_data]).any()
        np.testing.assert_array_equal(
            np.isfinite(got[:, :, SHALLOW[0], SHALLOW[1]]).all(axis=0), z <= 50.0
        )
        # A column with water to the bottom still holds the deepest monthly value
        # below the monthly depth limit.
        limit = int(np.flatnonzero(z == monthly_bottom)[0])
        deep = got[:, :, NEAR_SHALLOW[0], NEAR_SHALLOW[1]]
        assert (deep[:, limit + 1 :] == deep[:, [limit]]).all()

    before = _in_order(data.ds["n_an"].copy(deep=True))
    data.extrapolate_deepest_to_bottom(blend_m=BOTTOM_FILL_BLEND_M)
    after = _in_order(data.ds["n_an"])

    bottom = int(np.flatnonzero(z == 50.0)[0])
    below = slice(bottom + 1, None)
    column, donor = SHALLOW, NEAR_SHALLOW
    own = before[:, bottom, column[0], column[1]][:, None]
    neighbour = before[:, below, donor[0], donor[1]]
    weight = np.clip(1.0 - (z[below] - z[bottom]) / BOTTOM_FILL_BLEND_M, 0.0, 1.0)
    np.testing.assert_allclose(
        after[:, below, column[0], column[1]],
        weight * own + (1.0 - weight) * neighbour,
        rtol=1e-6,
    )
    assert not np.allclose(after[:, below, column[0], column[1]], own)
    wet = np.ones((COLUMN_LAT.size, COLUMN_LON.size), dtype=bool)
    wet[LAND] = False
    assert np.isfinite(after[:, :, wet]).all()


def test_mask_is_rebuilt_after_the_fill(woa_dir_columns, use_dask):
    """The mask built by ``post_process`` marks the column without data above 10 m as
    land, because the first tracer is NaN at the surface there. After the fill that
    column has data at the surface, so the rebuilt mask marks it as ocean.
    """
    data = _load_columns(woa_dir_columns, use_dask, apply_post_processing=True)
    assert (
        int(data.ds["mask"].isel(latitude=SURFACE_GAP[0], longitude=SURFACE_GAP[1]))
        == 0
    )
    data.extrapolate_deepest_to_bottom()

    expected = np.ones((COLUMN_LAT.size, COLUMN_LON.size), dtype=int)
    expected[LAND] = 0
    np.testing.assert_array_equal(data.ds["mask"].values, expected)
    for name in data.var_names.values():
        surface = _in_order(data.ds[name])[:, 0]
        assert np.isfinite(surface[:, expected == 1]).all(), name


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


@pytest.mark.parametrize("method", ["density_mld", "depth"])
def test_boundary_forcing_from_woa(_roms_grid, _physics_path, woa_dir, method):
    """A density method places the climatology on every physics record (the output
    follows the physics time axis); ``depth`` keeps the 12-month cycled climatology.
    """
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
        bgc_interpolation_method=method,
        prefill="2d_lateral_fill",
        regrid_method="scipy",
        use_dask=True,
    )

    ds = bf.bgc[0].ds
    for direction in ("north", "south", "east"):
        name = f"NO3_{direction}"
        assert name in ds
        if method == "depth":
            assert ds[name].sizes["bry_time"] == 12
            assert ds["bry_time"].attrs["cycle_length"] == 365.25
        else:
            assert ds[name].sizes["bry_time"] == bf.physics.ds.sizes["bry_time"]
            assert "cycle_length" not in ds["bry_time"].attrs
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
