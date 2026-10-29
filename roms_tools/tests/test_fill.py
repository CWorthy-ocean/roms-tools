import logging

import dask
import dask.array
import numpy as np
import pytest
import xarray as xr

from roms_tools.constants import R_EARTH
from roms_tools.fill import (
    BelowBottomFill,
    LateralFill,
    _find_deep_donors,
    fill_below_source_bottom,
    nearest_neighbor_fill,
)
from roms_tools.processing_methods import BOTTOM_FILL_BLEND_M
from roms_tools.setup.utils import latlon_to_xyz
from roms_tools.utils import unchunk_dask


@pytest.mark.parametrize(
    "data_fixture",
    [
        "era5_data",
        "glorys_data",
        "tpxo_data",
        "coarsened_cesm_bgc_data",
        "cesm_surface_bgc_data",
    ],
)
def test_lateral_fill_no_nans(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.ds = unchunk_dask(data.ds, data.dim_names)
    lateral_fill = LateralFill(
        data.ds["mask"],
        (data.dim_names["latitude"], data.dim_names["longitude"]),
    )
    if "mask_vel" in data.ds.data_vars:
        lateral_fill_vel = LateralFill(
            data.ds["mask_vel"],
            (data.dim_names["latitude"], data.dim_names["longitude"]),
        )

    for var in data.var_names:
        if var in ["u", "v"]:
            filled = lateral_fill_vel.apply(
                data.ds[data.var_names[var]].astype(np.float64)
            )
        else:
            filled = lateral_fill.apply(data.ds[data.var_names[var]].astype(np.float64))
        assert not filled.isnull().any()


def test_lateral_fill_correct_order_of_magnitude(coarsened_cesm_bgc_data):
    coarsened_cesm_bgc_data.ds = unchunk_dask(
        coarsened_cesm_bgc_data.ds, coarsened_cesm_bgc_data.dim_names
    )
    lateral_fill = LateralFill(
        coarsened_cesm_bgc_data.ds["mask"],
        (
            coarsened_cesm_bgc_data.dim_names["latitude"],
            coarsened_cesm_bgc_data.dim_names["longitude"],
        ),
    )

    ALK = coarsened_cesm_bgc_data.ds["ALK"]

    # zero out alkalinity field in all depth levels but the uppermost
    ALK = xr.where(
        coarsened_cesm_bgc_data.ds.ALK.depth > 25, 0, coarsened_cesm_bgc_data.ds.ALK
    )
    ALK = ALK.where(coarsened_cesm_bgc_data.ds.mask)

    filled = lateral_fill.apply(ALK.astype(np.float64))

    # check that alkalinity values in the uppermost values are of the correct order of magnitude
    # and that no new minima and maxima are introduced
    assert filled.isel(depth=0).min() == ALK.isel(depth=0).min()
    assert filled.isel(depth=0).max() == ALK.isel(depth=0).max()

    # check that the filled alkalinity values are zero in all deeper layers
    assert filled.isel(depth=slice(1, None)).equals(
        xr.zeros_like(filled.isel(depth=slice(1, None)))
    )


@pytest.mark.parametrize(
    "data_fixture",
    [
        "era5_data",
        "glorys_data",
    ],
)
def test_lateral_fill_reproducibility(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.ds = unchunk_dask(data.ds, data.dim_names)
    lateral_fill0 = LateralFill(
        data.ds["mask"],
        (data.dim_names["latitude"], data.dim_names["longitude"]),
    )
    if "mask_vel" in data.ds.data_vars:
        lateral_fill_vel0 = LateralFill(
            data.ds["mask_vel"],
            (data.dim_names["latitude"], data.dim_names["longitude"]),
        )
    lateral_fill1 = LateralFill(
        data.ds["mask"],
        (data.dim_names["latitude"], data.dim_names["longitude"]),
    )
    if "mask_vel" in data.ds.data_vars:
        lateral_fill_vel1 = LateralFill(
            data.ds["mask_vel"],
            (data.dim_names["latitude"], data.dim_names["longitude"]),
        )

    ds0 = data.ds.copy()
    ds1 = data.ds.copy()

    for var in data.var_names:
        if var in ["u", "v"]:
            ds0[data.var_names[var]] = lateral_fill_vel0.apply(
                ds0[data.var_names[var]].astype(np.float64)
            )
            ds1[data.var_names[var]] = lateral_fill_vel1.apply(
                ds1[data.var_names[var]].astype(np.float64)
            )

        else:
            ds0[data.var_names[var]] = lateral_fill0.apply(
                ds0[data.var_names[var]].astype(np.float64)
            )
            ds1[data.var_names[var]] = lateral_fill1.apply(
                ds1[data.var_names[var]].astype(np.float64)
            )

    assert ds0.equals(ds1)


def test_lateralfill_raises_notimplemented_for_non2d_mask():
    mask_3d = xr.DataArray(np.ones((3, 3, 3)), dims=("eta_rho", "xi_rho", "s_rho"))
    with pytest.raises(
        NotImplementedError, match="LateralFill currently supports only 2D masks"
    ):
        LateralFill(mask_3d, dims=("eta_rho", "xi_rho"))


def test_lateralfill_raises_valueerror_for_list_of_dims():
    mask = xr.DataArray(np.ones((3, 3)), dims=("eta_rho", "xi_rho"))
    with pytest.raises(TypeError, match="must be a tuple"):
        LateralFill(mask, dims=["eta_rho", "xi_rho"])


def test_lateralfill_raises_valueerror_for_wrong_number_of_dims():
    mask = xr.DataArray(np.ones((3, 3)), dims=("eta_rho", "xi_rho"))

    with pytest.raises(ValueError, match="must contain exactly two"):
        LateralFill(mask, dims=("eta_rho", "xi_rho", "extra_dim"))  # 3 dims


def test_lateralfill_raises_valueerror_for_mask_dim_order_mismatch():
    mask = xr.DataArray(np.ones((3, 3)), dims=("xi_rho", "eta_rho"))  # dims reversed
    with pytest.raises(ValueError, match="dimension order is incorrect"):
        LateralFill(mask, dims=("eta_rho", "xi_rho"))


def test_apply_raises_if_var_missing_dims():
    mask = xr.DataArray(np.ones((3, 3)), dims=("eta_rho", "xi_rho"))
    lf = LateralFill(mask, dims=("eta_rho", "xi_rho"))
    var = xr.DataArray(np.ones((3, 3)), dims=("eta_rho", "eta_u"))  # missing xi_rho
    with pytest.raises(
        ValueError, match="does not contain the required horizontal dimensions"
    ):
        lf.apply(var)


def test_apply_raises_if_var_dim_order_mismatch():
    mask = xr.DataArray(np.ones((3, 3)), dims=("eta_rho", "xi_rho"))
    lf = LateralFill(mask, dims=("eta_rho", "xi_rho"))
    var = xr.DataArray(np.ones((3, 3)), dims=("xi_rho", "eta_rho"))  # wrong order
    with pytest.raises(ValueError, match="dimension order is incorrect"):
        lf.apply(var)


def test_nearest_neighbor_fill_leaves_no_nans_and_preserves_valid():
    # Single ocean cell in the top-right corner; everything else is land/NaN.
    mask = xr.DataArray(
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=bool),
        dims=("eta_rho", "xi_rho"),
    )
    data = np.full((3, 3), np.nan)
    data[2, 2] = 7.0
    var = xr.DataArray(data, dims=("eta_rho", "xi_rho"))

    filled = nearest_neighbor_fill(var, mask, dims=("eta_rho", "xi_rho"))

    # No NaNs remain, the valid cell is unchanged, and every cell takes the only
    # valid value available.
    assert not filled.isnull().any()
    assert float(filled.sel(eta_rho=2, xi_rho=2)) == 7.0
    assert np.allclose(filled.values, 7.0)


def test_nearest_neighbor_fill_broadcasts_over_extra_dims():
    # Two valid columns; nearest fill should copy column-wise across a depth dim.
    mask = xr.DataArray(
        np.array([[1, 0, 0], [1, 0, 0]], dtype=bool), dims=("eta_rho", "xi_rho")
    )
    # depth=2, eta_rho=2, xi_rho=3
    arr = np.full((2, 2, 3), np.nan)
    arr[:, :, 0] = np.array([[1.0, 2.0], [10.0, 20.0]]).T  # valid column values
    var = xr.DataArray(arr, dims=("depth", "eta_rho", "xi_rho"))

    filled = nearest_neighbor_fill(var, mask, dims=("eta_rho", "xi_rho"))

    assert not filled.isnull().any()
    assert filled.dims == ("depth", "eta_rho", "xi_rho")
    # Masked columns (xi_rho 1, 2) take the value of the nearest valid column (0).
    for j in (1, 2):
        assert np.array_equal(
            filled.isel(xi_rho=j).values, filled.isel(xi_rho=0).values
        )


def test_apply_raises_if_var_has_nans_on_valid_mask():
    # 2D mask
    mask = xr.DataArray(np.ones((3, 3), dtype=bool), dims=("eta_rho", "xi_rho"))

    lf = LateralFill(mask, dims=("eta_rho", "xi_rho"))

    # Variable with a NaN at a location where mask is True
    data = np.ones((3, 3))
    data[1, 1] = np.nan
    var = xr.DataArray(data, dims=("eta_rho", "xi_rho"))

    # Expect ValueError
    with pytest.raises(
        ValueError, match="contains NaNs at grid points marked as valid"
    ):
        lf.apply(var)


DIMS = ("depth", "lat", "lon")

# The "Vanuatu column": an island column (lon 0) with data down to 300 m only, next to
# two deep columns (lon 0.1, which is nearer, and 0.2, 5 degC warmer so that taking the
# wrong donor shows). The levels 310 m and 325 m lie 10 m and 25 m below the island's
# bottom, i.e. inside and at the end of the default 25 m blend.
DEPTH = np.array([0.0, 100.0, 200.0, 300.0, 310.0, 325.0, 400.0, 1000.0, 2000.0])
DEEP_TEMP = np.array([26.0, 18.0, 14.0, 10.5, 10.3, 10.0, 8.0, 4.0, 2.24])
DEEP_SALT = np.array([35.5, 35.2, 35.0, 34.9, 34.9, 34.88, 34.8, 34.7, 34.633])
ISLAND_TEMP = np.array([27.0, 20.0, 15.0, 12.71])
ISLAND_SALT = np.array([35.6, 35.3, 35.15, 35.067])


def _vanuatu_source():
    """Island column at lon 0 (lat 0 and 0.1) and deep columns at lon 0.1 and 0.2."""
    lat, lon = np.array([0.0, 0.1]), np.array([0.0, 0.1, 0.2])
    out = {}
    for name, island, deep in (
        ("temp", ISLAND_TEMP, DEEP_TEMP),
        ("salt", ISLAND_SALT, DEEP_SALT),
    ):
        data = np.full((DEPTH.size, lat.size, lon.size), np.nan)
        data[: island.size, :, 0] = island[:, None]
        data[:, :, 1] = deep[:, None]
        data[:, :, 2] = deep[:, None] + 5.0
        out[name] = xr.DataArray(
            data, dims=DIMS, coords={"depth": DEPTH, "lat": lat, "lon": lon}
        )
    return out


def _fill_for(da, blend_m, dims=DIMS):
    """The fill built from the NaN pattern of ``da`` (its first record, if any)."""
    first = da.isel({d: 0 for d in da.dims if d not in dims})
    return BelowBottomFill(
        first.notnull().transpose(*dims).values,
        da["depth"].values,
        da["lat"].values,
        da["lon"].values,
        dims=dims,
        blend_m=blend_m,
    )


def test_below_bottom_fill_replaces_an_islands_deep_water_by_its_neighbours():
    """The island's 12.71 degC / 35.067 water is not copied down to 2000 m."""
    source = _vanuatu_source()
    temp, salt = source["temp"], source["salt"]

    # A forward fill copies the island's bottom water down to 2000 m: thermocline
    # water at the depth of the abyss, 10 degC warmer and 0.4 saltier than the
    # 2.24 degC / 34.633 of the deep columns around it.
    ffill = temp.ffill("depth")
    assert ffill.isel(depth=-1, lat=0, lon=0) == 12.71
    assert ffill.isel(depth=-1, lat=0, lon=1) == 2.24

    assert BOTTOM_FILL_BLEND_M == 25.0
    filled = _fill_for(temp, BOTTOM_FILL_BLEND_M).apply(temp)
    # Own values down to the island's bottom, the blend (60 % own at 10 m below, a
    # blend length of 25 m) at 310 m, then the nearest deep column's water at the same
    # depth from 25 m below the bottom on, down to its 2.24 degC at the bottom.
    np.testing.assert_allclose(
        filled.isel(lat=0, lon=0),
        [27.0, 20.0, 15.0, 12.71, 0.6 * 12.71 + 0.4 * 10.3, 10.0, 8.0, 4.0, 2.24],
        rtol=1e-12,
    )
    salt_filled = _fill_for(salt, BOTTOM_FILL_BLEND_M).apply(salt)
    np.testing.assert_allclose(
        salt_filled.isel(lat=0, lon=0),
        [
            35.6,
            35.3,
            35.15,
            35.067,
            0.6 * 35.067 + 0.4 * 34.9,
            34.88,
            34.8,
            34.7,
            34.633,
        ],
        rtol=1e-12,
    )
    # The deep columns are unchanged, and nothing is left NaN.
    np.testing.assert_array_equal(filled.isel(lon=[1, 2]), temp.isel(lon=[1, 2]))
    assert filled.notnull().all()
    assert filled.dims == temp.dims and filled.dtype == temp.dtype


def test_below_bottom_fill_with_zero_blend_takes_the_donor_right_below_the_bottom():
    temp = _vanuatu_source()["temp"]
    filled = _fill_for(temp, 0.0).apply(temp)
    np.testing.assert_allclose(
        filled.isel(lat=0, lon=0),
        [27.0, 20.0, 15.0, 12.71, 10.3, 10.0, 8.0, 4.0, 2.24],
    )
    # The island's own bottom value is kept at its bottom level.
    assert filled.isel(depth=3, lat=0, lon=0) == 12.71


def test_below_bottom_fill_with_infinite_blend_is_the_forward_fill():
    for da in _vanuatu_source().values():
        filled = _fill_for(da, np.inf).apply(da)
        np.testing.assert_array_equal(filled, da.ffill("depth"))


@pytest.mark.parametrize("blend_m", [10.0, 100.0, 1000.0])
def test_below_bottom_fill_blends_linearly_over_the_blend_length(blend_m):
    temp = _vanuatu_source()["temp"]
    filled = _fill_for(temp, blend_m).apply(temp)
    dz = DEPTH[4:] - 300.0
    w = np.clip(1.0 - dz / blend_m, 0.0, 1.0)
    expected = w * 12.71 + (1.0 - w) * DEEP_TEMP[4:]
    np.testing.assert_allclose(
        filled.isel(depth=slice(4, None), lat=0, lon=0), expected, rtol=1e-12
    )


def test_below_bottom_fill_repeats_the_filled_level_above_where_no_column_has_water():
    # Column A has water down to 200 m, column B down to 100 m; nothing at 300 and
    # 400 m, so those levels repeat the level above (B's 200 m cell is filled from A).
    depth = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    data = np.full((5, 1, 2), np.nan)
    data[:3, 0, 0] = [10.0, 8.0, 6.0]
    data[:2, 0, 1] = [11.0, 9.0]
    da = xr.DataArray(
        data, dims=DIMS, coords={"depth": depth, "lat": [0.0], "lon": [0.0, 0.1]}
    )
    for blend_m in (BOTTOM_FILL_BLEND_M, 0.0):
        filled = _fill_for(da, blend_m).apply(da)
        np.testing.assert_array_equal(
            filled.isel(lat=0), [[10, 11], [8, 9], [6, 6], [6, 6], [6, 6]]
        )


def test_below_bottom_fill_takes_the_donor_outright_for_a_gap_above_the_bottom():
    # Column 0 has no value at 100 m but has water below: it is a gap, not a bottom,
    # and takes column 1's value there however large the blend length is.
    depth = np.array([0.0, 100.0, 200.0])
    data = np.array([[[10.0, 11.0]], [[np.nan, 9.0]], [[6.0, 7.0]]])
    da = xr.DataArray(
        data, dims=DIMS, coords={"depth": depth, "lat": [0.0], "lon": [0.0, 0.1]}
    )
    filled = _fill_for(da, 1000.0).apply(da)
    np.testing.assert_array_equal(filled.isel(lat=0), [[10, 11], [9, 9], [6, 7]])


def test_below_bottom_fill_leaves_land_alone_and_never_takes_donors_from_it():
    # lon 0: island (water to 100 m); lon 1: land, nearest to the island; lon 2: deep.
    depth = np.array([0.0, 100.0, 200.0, 300.0])
    data = np.full((4, 1, 3), np.nan)
    data[:2, 0, 0] = [10.0, 8.0]
    data[:, 0, 2] = [20.0, 16.0, 12.0, 9.0]
    da = xr.DataArray(
        data, dims=DIMS, coords={"depth": depth, "lat": [0.0], "lon": [0.0, 0.1, 0.2]}
    )
    filled = _fill_for(da, BOTTOM_FILL_BLEND_M).apply(da)

    # The island's cells below its bottom come from the deep column, past the land one.
    np.testing.assert_array_equal(filled.isel(lat=0, lon=0), [10.0, 8.0, 12.0, 9.0])
    assert filled.isel(lon=1).isnull().all()
    # Cells with data are unchanged.
    wet = da.notnull()
    np.testing.assert_array_equal(filled.where(wet), da)


@pytest.mark.parametrize(
    "order",
    [
        ("time", "depth", "lat", "lon"),
        ("depth", "time", "lat", "lon"),
        ("lon", "lat", "depth", "time"),
    ],
)
def test_below_bottom_fill_uses_the_same_donors_for_every_time_record(order):
    temp = _vanuatu_source()["temp"]
    # The fill is a blend of the column's own value and the donor's, so a record
    # 2 * T + 1 gives 2 * fill(T) + 1 only if both records use the same donors.
    two = xr.concat([temp, 2.0 * temp + 1.0], dim="time").transpose(*order)
    filled = _fill_for(two, BOTTOM_FILL_BLEND_M).apply(two)

    assert filled.dims == order
    first = filled.isel(time=0).transpose(*DIMS)
    np.testing.assert_allclose(filled.isel(time=1).transpose(*DIMS), 2 * first + 1)
    np.testing.assert_array_equal(
        first, _fill_for(temp, BOTTOM_FILL_BLEND_M).apply(temp)
    )
    assert first.isel(depth=-1, lat=0, lon=0) == 2.24
    last = filled.isel(time=1).transpose(*DIMS).isel(depth=-1, lat=0, lon=0)
    assert last == pytest.approx(2 * 2.24 + 1)


def test_below_bottom_fill_keeps_the_data_of_a_later_record_in_a_filled_cell():
    # The donors come from the first record's NaN pattern. A cell that is NaN there
    # (the island's 400 m cell) but has data in the second record keeps it; the cells
    # that are still NaN there are filled as usual.
    temp = _vanuatu_source()["temp"]
    later = temp.copy()
    later[{"depth": 6, "lat": 0, "lon": 0}] = 7.0
    two = xr.concat([temp, later], dim="time")
    filled = _fill_for(two, BOTTOM_FILL_BLEND_M).apply(two)

    np.testing.assert_array_equal(
        filled.isel(time=0, lat=0, lon=0),
        _fill_for(temp, BOTTOM_FILL_BLEND_M).apply(temp).isel(lat=0, lon=0),
    )
    np.testing.assert_allclose(
        filled.isel(time=1, lat=0, lon=0),
        [27.0, 20.0, 15.0, 12.71, 0.6 * 12.71 + 0.4 * 10.3, 10.0, 7.0, 4.0, 2.24],
    )
    # The cells with data in the second record are all unchanged.
    has_data = two.notnull()
    np.testing.assert_array_equal(filled.where(has_data), two)


def test_below_bottom_fill_keeps_the_data_of_a_later_record_at_a_level_without_water():
    # Nothing is wet at 300 m in the first record, so the cells of that level repeat
    # the level above; column B has data there in the second record and keeps it, and
    # the level below it repeats that value.
    depth = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    data = np.full((5, 1, 2), np.nan)
    data[:3, 0, 0] = [10.0, 8.0, 6.0]
    data[:2, 0, 1] = [11.0, 9.0]
    first = xr.DataArray(
        data, dims=DIMS, coords={"depth": depth, "lat": [0.0], "lon": [0.0, 0.1]}
    )
    later = first.copy()
    later[{"depth": 3, "lat": 0, "lon": 1}] = 5.0
    two = xr.concat([first, later], dim="time")
    for blend_m in (BOTTOM_FILL_BLEND_M, 0.0):
        filled = _fill_for(two, blend_m).apply(two)
        np.testing.assert_array_equal(
            filled.isel(time=0, lat=0), [[10, 11], [8, 9], [6, 6], [6, 6], [6, 6]]
        )
        np.testing.assert_array_equal(
            filled.isel(time=1, lat=0), [[10, 11], [8, 9], [6, 6], [6, 5], [6, 5]]
        )


def test_below_bottom_fill_stays_lazy_on_dask_input_and_matches_the_eager_result():
    temp = _vanuatu_source()["temp"]
    two = xr.concat([temp, 2.0 * temp + 1.0], dim="time")
    fill = _fill_for(two, BOTTOM_FILL_BLEND_M)
    eager = fill.apply(two)

    def forbid_compute(*args, **kwargs):
        raise AssertionError("apply() computed the dask graph")

    with dask.config.set(scheduler=forbid_compute):
        lazy = fill.apply(two.chunk({"time": 1}))
    assert isinstance(lazy.data, dask.array.Array)
    assert lazy.chunks is not None
    np.testing.assert_array_equal(lazy.compute(), eager)


def test_below_bottom_fill_needs_whole_depth_lat_lon_blocks_on_dask_input():
    temp = _vanuatu_source()["temp"]
    fill = _fill_for(temp, BOTTOM_FILL_BLEND_M)
    with pytest.raises(ValueError, match="chunk"):
        fill.apply(temp.chunk({"depth": 3})).compute()


def test_below_bottom_fill_accepts_2d_lat_and_lon():
    temp = _vanuatu_source()["temp"]
    lat2d, lon2d = np.meshgrid(temp["lat"].values, temp["lon"].values, indexing="ij")
    fill2d = BelowBottomFill(
        temp.notnull().values, DEPTH, lat2d, lon2d, dims=DIMS, blend_m=25.0
    )
    np.testing.assert_array_equal(
        fill2d.apply(temp), _fill_for(temp, BOTTOM_FILL_BLEND_M).apply(temp)
    )


def test_below_bottom_fill_raises_for_inconsistent_inputs():
    wet = np.ones((3, 2, 2), dtype=bool)
    depth, lat, lon = np.array([0.0, 10.0, 20.0]), [0.0, 1.0], [0.0, 1.0]
    kwargs = {"dims": DIMS, "blend_m": 25.0}

    with pytest.raises(ValueError, match="must increase"):
        BelowBottomFill(wet, np.array([0.0, 10.0, 10.0]), lat, lon, **kwargs)
    with pytest.raises(ValueError, match="must increase"):
        BelowBottomFill(wet, depth[::-1], lat, lon, **kwargs)
    with pytest.raises(ValueError, match="must cover the horizontal shape"):
        BelowBottomFill(wet, depth, [0.0, 1.0, 2.0], lon, **kwargs)
    with pytest.raises(ValueError, match="must cover the horizontal shape"):
        BelowBottomFill(wet, depth, np.zeros((2, 3)), np.zeros((2, 3)), **kwargs)
    with pytest.raises(ValueError, match="must be 3D"):
        BelowBottomFill(wet[0], depth, lat, lon, **kwargs)
    with pytest.raises(ValueError, match="one depth per level"):
        BelowBottomFill(wet, depth[:2], lat, lon, **kwargs)


def test_below_bottom_fill_apply_raises_on_the_wrong_dimensions():
    temp = _vanuatu_source()["temp"]
    fill = _fill_for(temp, BOTTOM_FILL_BLEND_M)

    with pytest.raises(ValueError, match="must have dimension 'depth'"):
        fill.apply(temp.rename(depth="z"))
    with pytest.raises(ValueError, match="must have dimension 'lon'"):
        fill.apply(temp.isel(lon=0, drop=True))
    with pytest.raises(ValueError, match="of size 2"):
        fill.apply(temp.isel(lat=[0]))


S_RHO = ("s_rho", "eta_rho", "xi_rho")
HORIZONTAL = ("eta_rho", "xi_rho")


def _nested_case(
    parent_h=(100.0, 1000.0, 2000.0),
    levels=(950.0, 400.0, 250.0, 110.0, 100.0, 50.0),
    child_mask=None,
):
    """A two-column child grid over the west column of a three-column parent grid.

    The parent has ``eta_rho`` 2 and ``xi_rho`` 3 (lon 0, 1 and 2), bottoms ``parent_h``
    and 4 layers at 90, 60, 30 and 5 % of the local depth (deepest first). Its
    temperature is ``offset - depth / 100`` in the east columns (offsets 20 and 30) and
    10 in the west one. The child has ``levels`` (m) at lon 0 and its bottom, the
    parent's regridded ``h``, is that of the west parent column; the regridded
    temperature is the parent's deepest value there (10) repeated down, as vertical
    regridding leaves it. Returns the arguments of ``fill_below_source_bottom``
    except ``blend_m``.
    """
    bottom = np.broadcast_to(np.array(parent_h), (2, 3))
    pdepth = np.array([0.9, 0.6, 0.3, 0.05])[:, None, None] * bottom
    ptemp = np.array([10.0, 20.0, 30.0]) - pdepth / 100.0
    ptemp[:, :, 0] = 10.0
    tdepth = np.broadcast_to(np.array(levels)[:, None, None], (len(levels), 2, 1))
    child_lat = np.array([[0.0], [0.5]])
    return {
        "regridded": {
            "temp": xr.DataArray(
                np.full((1, len(levels), 2, 1), 10.0),
                dims=("time", *S_RHO),
                attrs={"units": "degC"},
            )
        },
        "target_depth": xr.DataArray(tdepth[None], dims=("time", *S_RHO)),
        "source_bottom": xr.DataArray(np.full((2, 1), parent_h[0]), dims=HORIZONTAL),
        "lon": xr.DataArray(np.zeros((2, 1)), dims=HORIZONTAL),
        "lat": xr.DataArray(child_lat, dims=HORIZONTAL),
        "mask": xr.DataArray(
            np.ones((2, 1)) if child_mask is None else child_mask, dims=HORIZONTAL
        ),
        "parent_fields": {"temp": xr.DataArray(ptemp[None], dims=("time", *S_RHO))},
        "parent_depth": xr.DataArray(pdepth, dims=S_RHO),
        "parent_h": xr.DataArray(bottom.copy(), dims=HORIZONTAL),
        "parent_mask": xr.DataArray(np.ones((2, 3)), dims=HORIZONTAL),
        "parent_lon": xr.DataArray(
            np.broadcast_to([0.0, 1.0, 2.0], (2, 3)).copy(), dims=HORIZONTAL
        ),
        "parent_lat": xr.DataArray(
            np.broadcast_to([[0.0], [0.5]], (2, 3)).copy(), dims=HORIZONTAL
        ),
    }


def _child_column(out):
    """Temperature of the first child column, by level."""
    return out["temp"].values[0, :, 0, 0]


def test_fill_below_source_bottom_takes_the_parent_columns_value_at_that_depth(caplog):
    kwargs = _nested_case()
    with caplog.at_level(logging.WARNING):
        out = fill_below_source_bottom(**kwargs, blend_m=BOTTOM_FILL_BLEND_M)

    # Below the 100 m bottom, the nearest parent column that is deeper than the cell is
    # the 1000 m one, temperature 20 - depth / 100 (11.0 at 950 m: the value of its
    # deepest layer, 900 m, clamped). The 110 m cell is 10 m below the bottom, 40 % of
    # the way (blend length 25 m) from the clamped 10 to the donor's 18.9; the cell at
    # the bottom (100 m) and the one above it keep the clamp.
    np.testing.assert_allclose(
        _child_column(out), [11.0, 16.0, 17.5, 10.0 + 0.4 * 8.9, 10.0, 10.0]
    )
    assert BOTTOM_FILL_BLEND_M == 25.0
    # Both child columns alike, and no warning when every cell has a donor.
    np.testing.assert_array_equal(
        out["temp"].values[0, :, 0, 0], out["temp"].values[0, :, 1, 0]
    )
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize(
    ("blend_m", "expected"),
    [
        # Donor values at 950, 400, 250 and 110 m: 11, 16, 17.5 and 18.9; the clamp is 10.
        (0.0, [11.0, 16.0, 17.5, 18.9, 10.0, 10.0]),
        (25.0, [11.0, 16.0, 17.5, 13.56, 10.0, 10.0]),
        (200.0, [11.0, 16.0, 15.625, 10.445, 10.0, 10.0]),
        (np.inf, [10.0] * 6),
    ],
)
def test_fill_below_source_bottom_blends_over_the_blend_length(blend_m, expected):
    kwargs = _nested_case()
    out = fill_below_source_bottom(**kwargs, blend_m=blend_m)
    np.testing.assert_allclose(_child_column(out), expected)
    if np.isinf(blend_m):
        # Nothing is filled: the input comes back as it was.
        assert out["temp"].identical(kwargs["regridded"]["temp"])


def test_fill_below_source_bottom_skips_nearer_columns_that_are_not_deep_enough():
    # The 300 m column lies between the child and the 2000 m one: it donates to the
    # cells shallower than 300 m only (the 300 m cell is not above its bottom).
    kwargs = _nested_case(parent_h=(100.0, 300.0, 2000.0), levels=(400, 300, 250, 150))
    out = fill_below_source_bottom(**kwargs, blend_m=0.0)
    # 30 - 4.0 and 30 - 3.0 from the 2000 m column, then 20 - 2.5 and 20 - 1.5.
    np.testing.assert_allclose(_child_column(out), [26.0, 27.0, 17.5, 18.5])


def test_fill_below_source_bottom_takes_donors_from_outside_the_child_grid():
    # The only deep column is the east one, 2 degrees from the child's own columns.
    kwargs = _nested_case(parent_h=(100.0, 100.0, 2000.0))
    out = fill_below_source_bottom(**kwargs, blend_m=0.0)
    np.testing.assert_allclose(_child_column(out), [20.5, 26.0, 27.5, 28.9, 10.0, 10.0])


def test_fill_below_source_bottom_keeps_the_clamp_where_no_column_is_deep_enough(
    caplog,
):
    # No parent column is deeper than 950 m, so the 950 m cells (2) have no donor.
    kwargs = _nested_case(parent_h=(100.0, 800.0, 900.0))
    with caplog.at_level(logging.WARNING):
        out = fill_below_source_bottom(**kwargs, blend_m=0.0)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "2 target cell(s) lie deeper than every parent column" in warnings[0]
    assert "within the loaded parent subset" in warnings[0]
    # The 400, 250 and 110 m cells are filled from the 800 m column (20 - depth / 100).
    np.testing.assert_allclose(_child_column(out), [10.0, 16.0, 17.5, 18.9, 10.0, 10.0])
    assert np.isfinite(out["temp"].values).all()


@pytest.mark.parametrize("case", ["all_parent_columns_shallow", "all_parent_land"])
def test_fill_below_source_bottom_returns_the_clamp_when_no_column_can_donate(
    case, caplog
):
    kwargs = _nested_case(parent_h=(100.0, 100.0, 100.0))
    if case == "all_parent_land":
        kwargs = _nested_case()
        kwargs["parent_mask"][:] = 0.0
    with caplog.at_level(logging.WARNING):
        out = fill_below_source_bottom(**kwargs, blend_m=0.0)
    assert out["temp"].identical(kwargs["regridded"]["temp"])
    assert (
        sum("lie deeper than every parent column" in r.message for r in caplog.records)
        == 1
    )


def test_fill_below_source_bottom_changes_nothing_when_the_child_is_not_deeper(caplog):
    kwargs = _nested_case(levels=(100.0, 60.0, 20.0))
    with caplog.at_level(logging.WARNING):
        out = fill_below_source_bottom(**kwargs, blend_m=0.0)
    assert out["temp"].identical(kwargs["regridded"]["temp"])
    assert not caplog.records


@pytest.mark.parametrize("unusable", ["land", "nan_layer_depths"])
def test_fill_below_source_bottom_never_donates_from_unusable_parent_columns(unusable):
    kwargs = _nested_case()
    # The 1000 m column would donate to every cell if it could.
    if unusable == "land":
        kwargs["parent_mask"][:, 1] = 0.0
        kwargs["parent_fields"]["temp"][..., 1] = 999.0
    else:
        kwargs["parent_depth"][..., 1] = np.nan
    out = fill_below_source_bottom(**kwargs, blend_m=0.0)
    # The 2000 m column donates instead: 30 - depth / 100.
    np.testing.assert_allclose(_child_column(out), [20.5, 26.0, 27.5, 28.9, 10.0, 10.0])


def test_fill_below_source_bottom_leaves_child_land_cells_alone():
    kwargs = _nested_case(child_mask=np.array([[1.0], [0.0]]))
    out = fill_below_source_bottom(**kwargs, blend_m=0.0)["temp"]
    np.testing.assert_array_equal(out.values[0, :, 1, 0], 10.0)
    assert out.values[0, 0, 0, 0] == 11.0


@pytest.mark.parametrize("layout", ["float32 in Fortran order", "transposed"])
def test_fill_below_source_bottom_preserves_dims_attrs_and_dtype(layout):
    kwargs = _nested_case()
    reference = fill_below_source_bottom(**kwargs, blend_m=BOTTOM_FILL_BLEND_M)["temp"]
    field = kwargs["regridded"]["temp"]
    if layout == "transposed":
        field = field.transpose("xi_rho", "s_rho", "time", "eta_rho")
    else:
        field = field.astype("float32").copy(
            data=np.asfortranarray(field.values.astype("float32"))
        )
    kwargs["regridded"] = {"temp": field}

    out = fill_below_source_bottom(**kwargs, blend_m=BOTTOM_FILL_BLEND_M)["temp"]

    assert out.dims == field.dims and out.attrs == field.attrs
    assert out.dtype == field.dtype
    np.testing.assert_allclose(
        out.transpose(*reference.dims).values, reference.values, rtol=1e-6
    )


def test_fill_below_source_bottom_raises_for_parent_arrays_of_different_shapes():
    kwargs = _nested_case()
    kwargs["parent_fields"] = {"temp": kwargs["parent_fields"]["temp"][:, 1:]}
    with pytest.raises(ValueError, match="differ in shape"):
        fill_below_source_bottom(**kwargs, blend_m=BOTTOM_FILL_BLEND_M)

    kwargs = _nested_case()
    kwargs["parent_h"] = kwargs["parent_h"].isel(xi_rho=slice(0, 2))
    with pytest.raises(ValueError, match="differ in shape"):
        fill_below_source_bottom(**kwargs, blend_m=BOTTOM_FILL_BLEND_M)


def test_fill_below_source_bottom_needs_a_single_parent_record():
    kwargs = _nested_case()
    temp = kwargs["parent_fields"]["temp"]
    kwargs["parent_fields"] = {"temp": xr.concat([temp, temp], dim="time")}
    with pytest.raises(ValueError, match="single record"):
        fill_below_source_bottom(**kwargs, blend_m=BOTTOM_FILL_BLEND_M)


def _random_columns(rng, n_pool, n_cells):
    """A donor pool of ``n_pool`` columns and ``n_cells`` cells, each in its own column,
    on a 10 x 10 degree patch. All but one pool column are 100 to 1000 m deep, one is
    3000 m deep.
    """
    return {
        "column": np.arange(n_cells),
        "column_lat": rng.uniform(0, 10, n_cells),
        "column_lon": rng.uniform(0, 10, n_cells),
        "pool_lat": rng.uniform(0, 10, n_pool),
        "pool_lon": rng.uniform(0, 10, n_pool),
        "pool_h": np.concatenate([rng.uniform(100.0, 1000.0, n_pool - 1), [3000.0]]),
    }


@pytest.fixture
def xyz_queries(monkeypatch):
    """The sizes of the coordinate arrays ``_find_deep_donors`` converts to xyz."""
    queried = []

    def spy(lat, lon):
        queried.append(len(lat))
        return latlon_to_xyz(lat, lon)

    monkeypatch.setattr("roms_tools.fill.latlon_to_xyz", spy)
    return queried


def test_find_deep_donors_batches_its_queries_without_changing_the_donors(xyz_queries):
    # 20000 columns of one cell each, with 512 neighbours queried at once: the cells
    # are queried in chunks (three here). Cells deeper than 1000 m have a deep enough
    # column only in the deepest of the 1000 pool columns, which is not among the 512
    # nearest of about half of them: those are retried with 600 neighbours, then go on
    # to the brute-force search (which needs the coordinates of the whole pool). The
    # donor must be the nearest deep-enough column, as an exhaustive search finds.
    rng = np.random.default_rng(7)
    n_cells = 20000
    args = _random_columns(rng, 1000, n_cells)
    depth = rng.uniform(100.0, 3500.0, n_cells)

    donor, distance = _find_deep_donors(depth, **args, neighbors=(512, 600))
    # The tree is queried before the brute-force search converts the pool: in chunks.
    assert 1000 in xyz_queries
    tree_queries = xyz_queries[: xyz_queries.index(1000)]
    assert len(tree_queries) > 1 and max(tree_queries) <= 4_000_000 // 512

    pool_xyz = latlon_to_xyz(args["pool_lat"], args["pool_lon"])
    for start in range(0, n_cells, 2000):
        sel = slice(start, start + 2000)
        column_xyz = latlon_to_xyz(args["column_lat"][sel], args["column_lon"][sel])
        chord = np.linalg.norm(column_xyz[:, None] - pool_xyz, axis=-1)
        chord = np.where(args["pool_h"] > depth[sel, None], chord, np.inf)
        found = np.isfinite(chord.min(axis=1))
        np.testing.assert_array_equal(donor[sel] >= 0, found)
        np.testing.assert_allclose(
            distance[sel][found], chord.min(axis=1)[found] * R_EARTH / 1000.0
        )
        # The nearest deep-enough column (equidistant columns excepted).
        np.testing.assert_array_equal(donor[sel][found], chord.argmin(axis=1)[found])
    # Cells deeper than 3000 m have no donor.
    assert (donor[depth >= 3000.0] == -1).all() and (donor[depth < 3000.0] >= 0).all()


def test_find_deep_donors_does_not_search_for_cells_deeper_than_every_pool_column(
    xyz_queries,
):
    # 20000 cells in 20000 columns lie deeper than every one of the 5000 pool columns,
    # 10 do not: only those reach a query (the brute-force search would need the
    # coordinates of the whole pool and of every unresolved cell).
    rng = np.random.default_rng(3)
    n_cells = 20010
    args = _random_columns(rng, 5000, n_cells)
    depth = np.concatenate([np.full(10, 50.0), np.full(n_cells - 10, 6000.0)])

    donor, distance = _find_deep_donors(depth, **args)

    assert (donor[:10] >= 0).all()
    assert (donor[10:] == -1).all() and np.isnan(distance[10:]).all()
    assert max(xyz_queries) <= 10


def test_find_deep_donors_builds_no_tree_when_no_cell_can_find_a_donor(monkeypatch):
    rng = np.random.default_rng(5)
    args = _random_columns(rng, 5000, 20000)

    def forbidden(*args, **kwargs):
        raise AssertionError("a tree was built")

    monkeypatch.setattr("roms_tools.fill.build_kdtree_from_latlon", forbidden)
    donor, distance = _find_deep_donors(np.full(20000, 5000.0), **args)
    assert (donor == -1).all() and np.isnan(distance).all()
