import logging
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from roms_tools.datasets.utils import (
    check_dataset,
    convert_to_float64,
    extrapolate_deepest_to_bottom,
    validate_start_end_time,
)


def test_convert_to_float64():
    # Create dataset with mixed dtypes
    ds = xr.Dataset(
        {
            "temp": (("x",), np.array([1, 2, 3], dtype=np.float32)),
            "mask": (("x",), np.array([0, 1, 0], dtype=np.int16)),
        }
    )

    out = convert_to_float64(ds)

    # All variables should now be float64
    assert all(out[var].dtype == np.float64 for var in out.data_vars)

    # Values should be preserved
    xr.testing.assert_equal(out["temp"], ds["temp"].astype("float64"))
    xr.testing.assert_equal(out["mask"], ds["mask"].astype("float64"))


def test_extrapolate_deepest_to_bottom():
    data = np.array(
        [
            [1, 2],
            [3, 4],
            [5, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ]
    )
    ds = xr.Dataset({"var": (("s_rho", "x"), data)})

    ds_filled = extrapolate_deepest_to_bottom(ds, "s_rho")

    expected = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 4],
            [5, 4],
            [5, 4],
        ]
    )
    np.testing.assert_array_equal(ds_filled["var"].values, expected)


# tests for  validate_start_end_time


def test_valid_times():
    start = datetime(2024, 1, 1, 12, 0)
    end = start + timedelta(hours=3)

    # Should not raise
    validate_start_end_time(start, end)


def test_none_times():
    # None for both is allowed
    validate_start_end_time(None, None)

    # Only start_time provided
    validate_start_end_time(datetime(2024, 1, 1), None)

    # Only end_time provided
    validate_start_end_time(None, datetime(2024, 1, 2))


def test_equal_times():
    t = datetime(2024, 1, 1, 12, 0)

    # end_time == start_time is allowed
    validate_start_end_time(t, t)


def test_invalid_start_type():
    with pytest.raises(TypeError):
        validate_start_end_time("2024-01-01", None)


def test_invalid_end_type():
    with pytest.raises(TypeError):
        validate_start_end_time(None, 123)


def test_end_time_before_start_time():
    start = datetime(2024, 1, 2)
    end = datetime(2024, 1, 1)

    with pytest.raises(ValueError):
        validate_start_end_time(start, end)


# tests for check_dataset


def test_valid_dataset():
    ds = xr.Dataset(
        data_vars={
            "temp": (("x", "y"), [[1, 2], [3, 4]]),
            "salt": (("x", "y"), [[5, 6], [7, 8]]),
        },
        coords={
            "x": [0, 1],
            "y": [0, 1],
        },
    )

    dim_names = {"xdim": "x", "ydim": "y"}
    var_names = {"temperature": "temp", "salinity": "salt"}

    # Should NOT raise
    check_dataset(ds, dim_names, var_names)


def test_missing_required_dimension():
    ds = xr.Dataset(data_vars={"temp": (("x",), [1])}, coords={"x": [0]})

    dim_names = {"xdim": "x", "ydim": "y"}

    with pytest.raises(ValueError, match="missing"):
        check_dataset(ds, dim_names=dim_names)


def test_missing_required_variable():
    ds = xr.Dataset(data_vars={"temp": (("x",), [1])}, coords={"x": [0]})

    var_names = {"temperature": "temp", "salinity": "salt"}

    with pytest.raises(ValueError, match="missing"):
        check_dataset(ds, var_names=var_names)


def test_optional_variables_warning(caplog):
    ds = xr.Dataset(data_vars={"temp": (("x",), [1])}, coords={"x": [0]})

    opt_var_names = {"some_opt": "opt_var"}

    with caplog.at_level(logging.WARNING):
        # Should not raise, only warn
        check_dataset(ds, opt_var_names=opt_var_names)

    assert "Optional variables missing" in caplog.text
    assert "opt_var" in caplog.text


def test_no_checks_pass():
    ds = xr.Dataset(data_vars={"temp": (("x",), [1])}, coords={"x": [0]})

    # Should not fail because no dim_names/var_names provided
    check_dataset(ds)
