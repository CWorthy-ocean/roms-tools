"""Tests for :class:`GLODAPv2BGCDataset`: the vertical fill and the unit conversion.

The fixtures write a tiny GLODAP-shaped tree (see ``glodap_test_utils``) with a few
columns that end at different depths next to land, so the value a filled cell must take
is known exactly rather than approximately.
"""

import numpy as np
import pytest
import xarray as xr

from roms_tools.datasets.lat_lon_datasets import GLODAPv2BGCDataset
from roms_tools.fill import BOTTOM_FILL_BLEND_M
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
    rebuilt from the filled first variable.
    """
    directory = _write_columns(tmp_path / "inf", holes=holes)
    data = _load(directory, use_dask, apply_post_processing=False)
    before = data.ds.copy(deep=True)
    data.extrapolate_deepest_to_bottom(blend_m=np.inf)

    expected = before.drop_vars("mask").ffill("depth").bfill("depth")
    for name in expected.data_vars:
        np.testing.assert_array_equal(data.ds[name].values, expected[name].values)
    first = next(iter(expected.data_vars))
    expected_mask = xr.where(expected[first].isel(depth=0, drop=True).isnull(), 0, 1)
    np.testing.assert_array_equal(data.ds["mask"].values, expected_mask.values)
