import logging

import numpy as np
import pytest
import xarray as xr

from roms_tools.fill import (
    LateralFill,
    fill_below_source_bottom,
    nearest_neighbor_fill,
)
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


def _below_bottom_case(
    bottoms=(100.0, 1000.0, 2000.0),
    mask=None,
    nan_column=None,
    levels=(950.0, 400.0, 250.0, 150.0, 50.0),
):
    """Three columns west to east with source bottoms ``bottoms`` (eta_rho is 2).

    The source has 4 levels at 90, 60, 30 and 5 % of the local bottom (deepest first);
    the target has levels at ``levels`` m everywhere. Source temperature
    is ``offset - depth / 100`` in the deep columns (offsets 20 and 30) and constant
    10 in the shallow one. Returns the arguments of ``fill_below_source_bottom``.
    """
    dims, horizontal = ("s_rho", "eta_rho", "xi_rho"), ("eta_rho", "xi_rho")
    bottom = np.broadcast_to(np.array(bottoms), (2, 3))
    sdepth = np.array([0.9, 0.6, 0.3, 0.05])[:, None, None] * bottom
    stemp = np.array([10.0, 20.0, 30.0]) - sdepth / 100.0
    stemp[:, :, 0] = 10.0
    tdepth = np.broadcast_to(np.array(levels)[:, None, None], (len(levels), 2, 3))
    if nan_column is not None:
        sdepth = sdepth.copy()
        sdepth[:, :, nan_column] = np.nan
    # What vertical regridding gives: the source's deepest value repeated downwards.
    regridded = np.broadcast_to(stemp[0], (len(levels), 2, 3)).copy()
    time = {"time": [0]}
    return {
        "regridded": {
            "temp": xr.DataArray(
                regridded[None],
                dims=("time", *dims),
                coords=time,
                attrs={"units": "degC"},
            )
        },
        "source": {"temp": xr.DataArray(stemp[None], dims=("time", *dims))},
        "target_depth": xr.DataArray(tdepth[None], dims=("time", *dims)),
        "source_depth": xr.DataArray(sdepth, dims=dims),
        "source_bottom": xr.DataArray(bottom.copy(), dims=horizontal),
        "lon": xr.DataArray(
            np.broadcast_to([0.0, 1.0, 2.0], (2, 3)).copy(), dims=horizontal
        ),
        "lat": xr.DataArray(
            np.broadcast_to([[0.0], [0.5]], (2, 3)).copy(), dims=horizontal
        ),
        "mask": xr.DataArray(
            np.ones((2, 3)) if mask is None else mask, dims=horizontal
        ),
    }


def test_fill_below_source_bottom_takes_the_deeper_columns_value_at_that_depth():
    kwargs = _below_bottom_case()
    before = kwargs["regridded"]["temp"]
    out = fill_below_source_bottom(**kwargs)["temp"]

    # In the 100 m column (clamp 10) the 950, 400, 250 and 150 m cells lie below the
    # bottom; the nearest column deep enough is the 1000 m one, temp = 20 - depth / 100
    # (clamped to its deepest layer, 900 m, for the 950 m cell). The first three are
    # at least 100 m below the bottom and take that value; the 150 m cell is 50 m
    # below it and is halfway between the clamp and the donor's 18.5. The 50 m cell
    # is above the bottom.
    np.testing.assert_allclose(
        out.values[0, :4, :, 0],
        [[11.0] * 2, [16.0] * 2, [17.5] * 2, [10.0 + 0.5 * (18.5 - 10.0)] * 2],
    )
    # Everything else is unchanged, bitwise (including the 950 m cell of the 1000 m
    # column, which lies between its deepest layer and its bottom).
    unchanged = np.ones(out.shape, bool)
    unchanged[0, :4, :, 0] = False
    np.testing.assert_array_equal(out.values[unchanged], before.values[unchanged])
    # Dims, coords, attrs and dtype are preserved, also for a field whose memory
    # layout is not C order and that is float32.
    assert out.dims == before.dims and out.attrs == before.attrs
    assert out.dtype == before.dtype
    odd = {
        "regridded": {
            "temp": before.astype("float32").copy(
                data=np.asfortranarray(before.values.astype("float32"))
            )
        },
        "source": {
            "temp": kwargs["source"]["temp"].copy(
                data=np.asfortranarray(kwargs["source"]["temp"].values)
            )
        },
    }
    odd_out = fill_below_source_bottom(**{**kwargs, **odd})["temp"]
    assert odd_out.dtype == np.float32
    np.testing.assert_allclose(odd_out.values, out.values, rtol=1e-6)


def test_fill_below_source_bottom_stays_close_to_the_clamp_just_below_the_bottom():
    # The 103 m cell lies 3 m below the 100 m column's bottom: its value is the clamp
    # (10) plus 3 % of the way to the donor's (20 - 103 / 100 interpolated).
    kwargs = _below_bottom_case(levels=(950.0, 400.0, 250.0, 103.0, 50.0))
    clamp = kwargs["regridded"]["temp"].values[0, 3, :, 0]
    out = fill_below_source_bottom(**kwargs)["temp"].values[0, 3, :, 0]
    donor = 20.0 - 1.03
    assert (np.abs(out - clamp) <= 0.03 * abs(donor - clamp) + 1e-12).all()
    np.testing.assert_allclose(out, clamp + 0.03 * (donor - clamp))


def test_fill_below_source_bottom_keeps_the_clamp_where_no_column_is_deep_enough(
    caplog,
):
    # No column is deeper than 950 m, so the 950 m cells below the bottoms of the 100 m
    # and 800 m columns (and the 900 m one) have no donor.
    kwargs = _below_bottom_case(bottoms=(100.0, 800.0, 900.0))
    with caplog.at_level(logging.WARNING):
        out = fill_below_source_bottom(**kwargs)["temp"]
    assert (
        sum("lie deeper than every source column" in r.message for r in caplog.records)
        == 1
    )
    np.testing.assert_array_equal(
        out.values[0, 0], kwargs["regridded"]["temp"].values[0, 0]
    )
    assert np.isfinite(out.values).all()


def test_fill_below_source_bottom_never_donates_from_land_or_nan_columns():
    land = np.ones((2, 3))
    land[:, 1] = 0.0
    for kwargs in (_below_bottom_case(mask=land), _below_bottom_case(nan_column=1)):
        out = fill_below_source_bottom(**kwargs)["temp"]
        assert np.isfinite(out.values).all()
        # The 100 m column is filled from the 2000 m column: temp = 30 - depth / 100.
        np.testing.assert_allclose(out.values[0, :3, 0, 0], [20.5, 26.0, 27.5])
        # Column 1 (land, or with an unusable source profile) is left as it was.
        np.testing.assert_array_equal(
            out.values[0, :, :, 1], kwargs["regridded"]["temp"].values[0, :, :, 1]
        )
