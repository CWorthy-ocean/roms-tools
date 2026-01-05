import numpy as np
import pytest
import xarray as xr

from roms_tools.fill import LateralFill


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
