"""Tests for :class:`GLODAPv2BGCDataset`: the vertical fill and the unit conversion.

The fixtures write a tiny GLODAP-shaped tree (see ``glodap_test_utils``) with a few
columns that end at different depths next to land, so the value a filled cell must take
is known exactly rather than approximately.
"""

import numpy as np
import pytest

from roms_tools.datasets.lat_lon_datasets import GLODAPv2BGCDataset
from roms_tools.processing_methods import BOTTOM_FILL_BLEND_M, _xesmf_available
from roms_tools.regrid import per_variable_mask_name
from roms_tools.setup.utils import compute_in_situ_density
from roms_tools.tests.glodap_test_utils import write_synthetic_glodap

DEPTH = np.array([0.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0])
LAT = np.array([10.0, 11.0, 12.0])
LON = np.array([20.0, 21.0, 22.0, 23.0, 24.0])

#: The columns with water, (lat index, lon index) -> (deepest level with data, value
#: offset). Every other column is land, so the nearest column with water at a level
#: is unambiguous: the shallow column's neighbour is ~110 km away, the far one ~440 km.
COLUMNS = {
    (1, 0): (2, 0.0),  # shallow: water down to 20 m
    (1, 1): (5, 100.0),  # its neighbour: water down to 100 m
    (1, 4): (6, 200.0),  # the far column: water down to 200 m
}
SHALLOW, NEAR, FAR = (1, 0), (1, 1), (1, 4)
#: Level index of the deepest data of the shallow and the near column.
SHALLOW_BOTTOM, NEAR_BOTTOM = COLUMNS[SHALLOW][0], COLUMNS[NEAR][0]


def _write_columns(directory, holes=()):
    """Write the three columns; ``holes`` are ``(variable or "*", column, levels)``.

    A value is ``base + offset + 0.01 * depth``, so the columns, and the levels of a
    column, all differ and a filled cell shows which column it came from.
    """

    def field_fn(name, values):
        out = np.full_like(values, np.nan)
        for (i, j), (bottom, offset) in COLUMNS.items():
            n = bottom + 1
            out[:n, i, j] = values[:n, i, j] + offset + 0.01 * DEPTH[:n]
        for variable, (i, j), levels in holes:
            if variable in ("*", name):
                out[levels, i, j] = np.nan
        return out

    return write_synthetic_glodap(
        directory, include_ts=False, lat=LAT, lon=LON, depth=DEPTH, field_fn=field_fn
    )


def _load(directory, use_dask, **kwargs):
    return GLODAPv2BGCDataset(filename=str(directory), use_dask=use_dask, **kwargs)


def _at(ds, name, column, level):
    return float(ds[name].isel(depth=level, latitude=column[0], longitude=column[1]))


def _blend(own, donor, level, bottom, blend_m):
    """``w * own + (1 - w) * donor``, with w falling from 1 at the bottom to 0."""
    dz = DEPTH[level] - DEPTH[bottom]
    weight = 0.0 if blend_m == 0 else max(0.0, 1.0 - dz / blend_m)
    return weight * own + (1.0 - weight) * donor


@pytest.fixture
def glodap_columns(tmp_path):
    return _write_columns(tmp_path / "glodap")


@pytest.mark.parametrize("blend_m", [0.0, BOTTOM_FILL_BLEND_M, 100.0])
def test_cells_below_a_columns_bottom_take_the_nearest_column_with_water_there(
    glodap_columns, use_dask, blend_m
):
    """A cell below a column's bottom takes the same-depth value of the nearest column
    that has water at that level, blended in from the column's own deepest value.

    The shallow column's neighbour has water down to 100 m, so below that the donor is
    the far column; the neighbour in turn takes the far column's value at 200 m. The
    old fill copied each column's deepest value down instead.
    """
    data = _load(glodap_columns, use_dask, apply_post_processing=False)
    before = data.ds.copy(deep=True)
    data.extrapolate_deepest_to_bottom(blend_m=blend_m)

    for name in data.var_names.values():
        own = _at(before, name, SHALLOW, SHALLOW_BOTTOM)
        for level in range(SHALLOW_BOTTOM + 1, DEPTH.size):
            donor = FAR if level > NEAR_BOTTOM else NEAR
            expected = _blend(
                own,
                _at(before, name, donor, level),
                level,
                SHALLOW_BOTTOM,
                blend_m,
            )
            got = _at(data.ds, name, SHALLOW, level)
            assert got == pytest.approx(expected, rel=1e-6), (name, level)
            # Not the column's own deepest value copied down.
            assert got != own

        own = _at(before, name, NEAR, NEAR_BOTTOM)
        expected = _blend(own, _at(before, name, FAR, 6), 6, NEAR_BOTTOM, blend_m)
        assert _at(data.ds, name, NEAR, 6) == pytest.approx(expected, rel=1e-6)

        # Cells that had data are untouched, and land stays land.
        np.testing.assert_array_equal(
            data.ds[name].where(before[name].notnull()).values,
            before[name].values,
        )
        land = np.ones(data.ds[name].shape[1:], dtype=bool)
        for i, j in COLUMNS:
            land[i, j] = False
        assert data.ds[name].isnull().transpose("depth", ...).values[:, land].all()


@pytest.mark.parametrize("blend_m", [0.0, BOTTOM_FILL_BLEND_M])
def test_an_interior_gap_takes_the_donor_value_not_an_interpolation(
    tmp_path, use_dask, blend_m
):
    """A NaN with data above and below takes the same-depth value of the nearest column
    with water there, however large the blend length: it is not interpolated between
    the values above and below, nor filled with the value above.

    Only NO3 has the gap, so only its wet mask loses the near column at 30 m and the
    far column becomes the donor of the gap and of the shallow column's 30 m cell.
    """
    gap = 3  # the 30 m level
    directory = _write_columns(tmp_path / "gap", holes=[("NO3", NEAR, [gap])])
    data = _load(directory, use_dask, apply_post_processing=False)
    before = data.ds.copy(deep=True)
    assert np.isnan(_at(before, "NO3", NEAR, gap))
    data.extrapolate_deepest_to_bottom(blend_m=blend_m)

    above, below = (_at(before, "NO3", NEAR, k) for k in (gap - 1, gap + 1))
    got = _at(data.ds, "NO3", NEAR, gap)
    assert got == pytest.approx(_at(before, "NO3", FAR, gap), rel=1e-6)
    assert got != pytest.approx(0.5 * (above + below), rel=1e-3)
    assert got != above

    # The other tracers have no gap here, so their near column keeps its own data.
    for name in ("PO4", "oxygen"):
        np.testing.assert_array_equal(
            data.ds[name].isel(latitude=NEAR[0], longitude=NEAR[1]).values[: gap + 2],
            before[name].isel(latitude=NEAR[0], longitude=NEAR[1]).values[: gap + 2],
        )


def test_nan_above_the_data_is_filled_backward(tmp_path, use_dask):
    """NaN above a column's data takes the shallowest value below it, not the value of
    a neighbouring column at that depth, and the fill below the bottom still works.
    """
    directory = _write_columns(tmp_path / "above", holes=[("PO4", NEAR, [0, 1])])
    data = _load(directory, use_dask, apply_post_processing=False)
    before = data.ds.copy(deep=True)
    assert np.isnan([_at(before, "PO4", NEAR, k) for k in (0, 1)]).all()
    data.extrapolate_deepest_to_bottom()

    shallowest = _at(before, "PO4", NEAR, 2)
    for level in (0, 1):
        assert _at(data.ds, "PO4", NEAR, level) == pytest.approx(shallowest)
    # The far column's value at the surface is not what the near column took.
    assert _at(data.ds, "PO4", NEAR, 0) != pytest.approx(_at(before, "PO4", FAR, 0))

    # The shallow column's 30 m cell still comes from the near column's 30 m value.
    expected = _blend(
        _at(before, "PO4", SHALLOW, SHALLOW_BOTTOM),
        _at(before, "PO4", NEAR, 3),
        3,
        SHALLOW_BOTTOM,
        BOTTOM_FILL_BLEND_M,
    )
    assert _at(data.ds, "PO4", SHALLOW, 3) == pytest.approx(expected, rel=1e-6)


def test_mask_is_rebuilt_after_the_fill(tmp_path, use_dask):
    """A column whose shallowest data is below the surface is ocean once it is filled.

    ``clean_up`` builds the mask from the first variable's surface level, which marks
    the near column as land because every variable is NaN at the surface there. The
    far column has a second variable (oxygen) with a surface gap that the first does
    not have. After the fill every variable is finite at the surface of every ocean
    column, which is what ``apply_lateral_fill`` needs.
    """
    directory = _write_columns(
        tmp_path / "mask", holes=[("*", NEAR, [0]), ("oxygen", FAR, [0])]
    )
    data = _load(directory, use_dask, apply_post_processing=False)
    assert int(data.ds["mask"].isel(latitude=NEAR[0], longitude=NEAR[1])) == 0
    data.extrapolate_deepest_to_bottom()

    mask = data.ds["mask"].values
    expected = np.zeros_like(mask)
    for i, j in COLUMNS:
        expected[i, j] = 1
    np.testing.assert_array_equal(mask, expected)
    for name in data.var_names.values():
        surface = data.ds[name].isel(depth=0).values
        assert np.isfinite(surface[mask == 1]).all(), name


def test_a_column_with_tracers_but_no_temperature_or_salinity_gets_its_own_ts_mask(
    tmp_path, use_dask
):
    """A column with a finite tracer but T and S NaN at every depth is land for T and S.

    The mask used to come from the first variable (a tracer) alone, so the column was
    ocean for T and S too while they stayed NaN there, and ``apply_lateral_fill``
    raised. Now T and S carry their own masks, with the column as land, and after the
    lateral fill no variable is NaN where its own mask is 1.
    """
    lat, lon = np.arange(10.0, 14.0), np.arange(20.0, 25.0)
    no_ts = (1, 2)

    def field_fn(name, values):
        if name in ("temperature", "salinity"):
            values[:, no_ts[0], no_ts[1]] = np.nan
        return values

    directory = write_synthetic_glodap(
        tmp_path / "no_ts", lat=lat, lon=lon, field_fn=field_fn
    )
    data = _load(directory, use_dask)
    assert np.isfinite(data.ds["NO3"].isel(latitude=no_ts[0], longitude=no_ts[1])).all()
    assert (
        data.ds["temperature"]
        .isel(latitude=no_ts[0], longitude=no_ts[1])
        .isnull()
        .all()
    )
    data.extrapolate_deepest_to_bottom()

    np.testing.assert_array_equal(
        data.ds["mask"].values, np.ones((lat.size, lon.size), dtype=int)
    )
    expected = np.ones((lat.size, lon.size), dtype=int)
    expected[no_ts] = 0
    for name in ("temperature", "salinity"):
        np.testing.assert_array_equal(
            data.ds[per_variable_mask_name(name)].values, expected
        )

    data.apply_lateral_fill()
    for name in data.ds.data_vars:
        if name.startswith("mask"):
            continue
        own = data.ds.get(per_variable_mask_name(name), data.ds["mask"]).values == 1
        values = data.ds[name].transpose("depth", "latitude", "longitude").values
        assert np.isfinite(values[:, own]).all(), name
    # T and S are filled in the column from their neighbours.
    column = data.ds["temperature"].isel(latitude=no_ts[0], longitude=no_ts[1])
    np.testing.assert_allclose(column.values, 2.0, rtol=1e-3)


#: Columns of an all-wet 5 x 5 source for the per-variable mask tests, (lat, lon).
GAP_LAT = np.arange(10.0, 15.0)
GAP_LON = np.arange(20.0, 25.0)
LAND, NO_SILICATE, NO_TCO2 = (0, 0), (2, 1), (2, 3)
#: Every variable of that source is uniform, at this value.
UNIFORM = 7.0


def _write_gap_columns(directory):
    """A uniform source where whole columns are missing, per variable.

    ``NO_SILICATE`` has no silicate at any depth and ``NO_TCO2`` has no TCO2, though
    every other variable (NO3, the one the shared mask comes from, included) has data
    there; ``LAND`` has none of any variable.
    """

    def field_fn(name, values):
        values = np.full_like(values, UNIFORM)
        values[:, LAND[0], LAND[1]] = np.nan
        for variable, column in (("silicate", NO_SILICATE), ("TCO2", NO_TCO2)):
            if name == variable:
                values[:, column[0], column[1]] = np.nan
        return values

    return write_synthetic_glodap(
        directory,
        include_ts=False,
        lat=GAP_LAT,
        lon=GAP_LON,
        depth=DEPTH,
        field_fn=field_fn,
    )


def _gap_dataset(tmp_path, use_dask):
    data = _load(
        _write_gap_columns(tmp_path / "gaps"), use_dask, apply_post_processing=False
    )
    data.extrapolate_deepest_to_bottom()
    return data


def test_a_variable_missing_a_column_the_shared_mask_calls_ocean_gets_its_own_mask(
    tmp_path, use_dask
):
    """Only a variable whose wet cells differ from ``mask`` gets a per-variable mask."""
    data = _gap_dataset(tmp_path, use_dask)

    masks = {v for v in data.ds.data_vars if v.startswith("mask")}
    assert masks == {"mask", "mask_silicate", "mask_TCO2"}

    expected = np.ones((GAP_LAT.size, GAP_LON.size), dtype=int)
    expected[LAND] = 0
    np.testing.assert_array_equal(data.ds["mask"].values, expected)
    expected[NO_SILICATE] = 0
    np.testing.assert_array_equal(data.ds["mask_silicate"].values, expected)
    expected[NO_SILICATE] = 1
    expected[NO_TCO2] = 0
    np.testing.assert_array_equal(data.ds["mask_TCO2"].values, expected)


def test_extrapolating_twice_leaves_no_stale_per_variable_mask(tmp_path, use_dask):
    """A second call rebuilds the masks: a variable that no longer differs loses its own."""
    data = _gap_dataset(tmp_path, use_dask)
    data.ds["silicate"] = (
        data.ds["silicate"].fillna(UNIFORM).where(data.ds["mask"] == 1)
    )
    data.extrapolate_deepest_to_bottom()
    assert "mask_silicate" not in data.ds
    assert "mask_TCO2" in data.ds


@pytest.mark.parametrize(
    "fill",
    [
        "apply_lateral_fill",
        "apply_nearest_neighbor_fill",
        pytest.param(
            "apply_xesmf_source_fill",
            marks=pytest.mark.skipif(
                not _xesmf_available(), reason="xesmf required for the xESMF fill"
            ),
        ),
    ],
)
def test_each_variable_is_filled_against_its_own_mask(tmp_path, use_dask, fill):
    """No NaN is left at the cells a variable lacks, and the cells it has keep their value.

    With one shared mask, silicate's gap column is marked as known and stays NaN, and
    LateralFill takes it as a value.
    """
    data = _gap_dataset(tmp_path, use_dask)
    before = data.ds.copy(deep=True).compute()
    getattr(data, fill)()

    for name in data.var_names.values():
        filled = data.ds[name].compute()
        assert not filled.isnull().any(), name
        own = before.get(per_variable_mask_name(name), before["mask"]) == 1
        np.testing.assert_allclose(
            filled.where(own).values, before[name].where(own).values
        )
    # Where silicate had nothing, it takes its neighbours' value (the field is uniform).
    np.testing.assert_allclose(
        data.ds["silicate"].isel(latitude=NO_SILICATE[0], longitude=NO_SILICATE[1]),
        UNIFORM,
    )


@pytest.mark.parametrize(
    "holes",
    [
        [],
        [
            ("NO3", NEAR, [3]),
            ("PO4", NEAR, [0, 1]),
            ("*", FAR, [0]),
            ("TAlk", FAR, [6]),
        ],
    ],
    ids=["plain", "gaps"],
)
def test_infinite_blend_reproduces_the_forward_then_backward_fill(
    tmp_path, use_dask, holes
):
    """``blend_m=inf`` is exactly ``ffill`` then ``bfill`` along depth, as before the
    same-depth fill, interior gaps and NaN above the data included, with the mask
    rebuilt from every filled variable.
    """
    directory = _write_columns(tmp_path / "inf", holes=holes)
    data = _load(directory, use_dask, apply_post_processing=False)
    before = data.ds.copy(deep=True)
    data.extrapolate_deepest_to_bottom(blend_m=np.inf)

    expected = before.drop_vars("mask").ffill("depth").bfill("depth")
    for name in expected.data_vars:
        np.testing.assert_array_equal(data.ds[name].values, expected[name].values)
    surface_finite = np.stack(
        [expected[name].isel(depth=0).values for name in expected.data_vars]
    )
    expected_mask = np.isfinite(surface_finite).all(axis=0).astype(int)
    np.testing.assert_array_equal(data.ds["mask"].values, expected_mask)


# ---------------------------------------------------------------------------
# umol/kg -> mmol/m3: a tracer must survive a missing temperature or salinity
# ---------------------------------------------------------------------------

#: (level, lat index, lon index) of the cells without a temperature / salinity.
NO_TEMPERATURE = [(2, 1, 3), (1, 0, 4)]
NO_SALINITY = [(2, 3, 0)]


def _write_conversion_source(directory):
    """A fully wet 4 x 5 grid whose T and S vary, with a few cells lacking one of them.

    Every tracer is finite everywhere, so a NaN in the converted tracer can only come
    from the NaN T/S.
    """
    lat = np.arange(10.0, 14.0)
    lon = np.arange(20.0, 25.0)

    def field_fn(name, values):
        if name == "temperature":
            values = values + np.arange(lon.size, dtype="f4")[None, None, :]
            for k, i, j in NO_TEMPERATURE:
                values[k, i, j] = np.nan
        elif name == "salinity":
            values = values + 0.2 * np.arange(lat.size, dtype="f4")[None, :, None]
            for k, i, j in NO_SALINITY:
                values[k, i, j] = np.nan
        return values

    return write_synthetic_glodap(directory, lat=lat, lon=lon, field_fn=field_fn)


def test_a_tracer_with_nan_temperature_or_salinity_gets_the_mean_density_of_its_level(
    tmp_path, use_dask
):
    """The converted tracer stays finite where T or S is NaN and equals the tracer times
    the mean in-situ density of that depth level over the cells that have both, divided
    by 1000; every other cell is converted with its own density, as before.
    """
    directory = _write_conversion_source(tmp_path / "conversion")
    raw = _load(directory, use_dask, apply_post_processing=False).ds
    converted = _load(directory, use_dask).ds

    density = compute_in_situ_density(
        raw["temperature"], raw["salinity"], raw["depth"], raw["latitude"]
    ).transpose("depth", "latitude", "longitude")
    rho = density.values
    no_ts = [*NO_TEMPERATURE, *NO_SALINITY]
    assert np.isnan(rho).sum() == len(no_ts)

    for name in ("TAlk", "NO3", "oxygen"):
        tracer = raw[name].transpose("depth", "latitude", "longitude").values
        got = converted[name].transpose("depth", "latitude", "longitude").values
        assert np.isfinite(got).all()

        # Where T and S exist the plain formula is unchanged ...
        have_ts = np.isfinite(rho)
        plain = tracer * rho / 1000.0
        np.testing.assert_allclose(got[have_ts], plain[have_ts], rtol=1e-6)
        # ... and elsewhere the level's mean density over the cells that have both.
        for k, i, j in no_ts:
            expected = tracer[k, i, j] * np.nanmean(rho[k]) / 1000.0
            assert got[k, i, j] == pytest.approx(expected, rel=1e-6), (name, k, i, j)

    # T and S are not tracers and keep their NaNs.
    for name in ("temperature", "salinity"):
        np.testing.assert_array_equal(converted[name].values, raw[name].values)


def test_a_level_without_any_temperature_uses_a_density_of_1025(tmp_path, use_dask):
    """At a level where T is NaN everywhere the level mean density is NaN too, and the
    tracer is converted with 1025 kg m-3 rather than lost; other levels are unchanged.
    """
    level = 1
    lat, lon = np.arange(10.0, 13.0), np.arange(20.0, 23.0)

    def field_fn(name, values):
        if name == "temperature":
            values[level] = np.nan
        return values

    directory = write_synthetic_glodap(
        tmp_path / "no_t_level", lat=lat, lon=lon, field_fn=field_fn
    )
    raw = _load(directory, use_dask, apply_post_processing=False).ds
    converted = _load(directory, use_dask).ds

    rho = compute_in_situ_density(
        raw["temperature"], raw["salinity"], raw["depth"], raw["latitude"]
    ).transpose("depth", "latitude", "longitude")
    rho = rho.values
    assert np.isnan(rho[level]).all()
    others = [k for k in range(rho.shape[0]) if k != level]
    assert np.isfinite(rho[others]).all()

    for name in ("TAlk", "NO3", "oxygen"):
        tracer = raw[name].transpose("depth", "latitude", "longitude").values
        got = converted[name].transpose("depth", "latitude", "longitude").values
        assert np.isfinite(got).all()
        np.testing.assert_allclose(
            got[level], tracer[level] * 1025.0 / 1000.0, rtol=1e-6
        )
        np.testing.assert_allclose(
            got[others], tracer[others] * rho[others] / 1000.0, rtol=1e-6
        )
