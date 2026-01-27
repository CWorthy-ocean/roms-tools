import logging
from copy import deepcopy
from datetime import datetime, timedelta

import cftime
import numpy as np
import pytest
import xarray as xr

from roms_tools.datasets.utils import (
    _select_initial_time,
    check_dataset,
    convert_cftime_to_datetime,
    convert_to_float64,
    extrapolate_deepest_to_bottom,
    get_time_type,
    select_relevant_fields,
    select_relevant_times,
    validate_start_end_time,
)


def test_convert_to_float64():
    # Create dataset with mixed dtypes
    ds = xr.Dataset(
        {
            "temp": (("x",), np.array([1, 2, 3], dtype=np.float32)),
            "mask_rho": (("x",), np.array([0, 1, 0], dtype=np.int16)),
        }
    )

    out = convert_to_float64(ds)

    # Non-mask variables should be converted to float64
    assert out["temp"].dtype == np.float64

    # Mask variables should keep their original dtype
    assert out["mask_rho"].dtype == np.int16

    # Values should be preserved
    xr.testing.assert_equal(out["temp"], ds["temp"].astype("float64"))
    xr.testing.assert_equal(out["mask_rho"], ds["mask_rho"])


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


# tests for select_relevant_fields


def test_select_relevant_fields_basic():
    """Keep only required + optional variables and drop others."""
    ds = xr.Dataset(
        {
            "temp": (("x",), np.arange(5)),
            "salt": (("x",), np.arange(5) * 2),
            "u": (("x",), np.arange(5) * 3),
            "mask": (("x",), np.ones(5)),
            "extra": (("x",), np.zeros(5)),
        }
    )

    keep_vars = ["temp", "u"]  # required + optional

    out = select_relevant_fields(ds, keep_vars)

    assert set(out.data_vars) == {"temp", "u", "mask"}
    assert "salt" not in out
    assert "extra" not in out


def test_select_relevant_fields_does_not_modify_input():
    """Function must not mutate the original dataset."""
    ds = xr.Dataset(
        {
            "temp": (("x",), np.arange(5)),
            "salt": (("x",), np.arange(5)),
            "mask": (("x",), np.ones(5)),
        }
    )

    ds_copy = deepcopy(ds)

    _ = select_relevant_fields(ds, ["temp"])

    # ensure original dataset is unchanged
    assert set(ds.data_vars) == set(ds_copy.data_vars)
    assert "salt" in ds
    assert "mask" in ds


def test_select_relevant_fields_duplicate_keep_names():
    """Duplicate keep names should not cause issues."""
    ds = xr.Dataset(
        {
            "temp": (("x",), np.arange(5)),
            "mask": (("x",), np.ones(5)),
        }
    )

    keep_vars = ["temp", "temp"]  # duplicates

    out = select_relevant_fields(ds, keep_vars)

    assert set(out.data_vars) == {"temp", "mask"}


# tests for select_relevant_times


def make_time_dataset(times):
    return xr.Dataset(
        data_vars={"var": ("time", np.arange(len(times)))},
        coords={"time": times},
    )


def test_missing_time_dimension(caplog):
    ds = xr.Dataset({"x": ("x", [1, 2])})

    with caplog.at_level(logging.WARNING):
        out = select_relevant_times(
            ds, "time", datetime(2024, 1, 1), datetime(2024, 1, 2)
        )

    assert "does not contain time dimension" in caplog.text
    assert out is ds  # unchanged


def test_climatology_must_have_12_steps():
    times = np.arange(10).astype("datetime64[D]")
    ds = make_time_dataset(times)

    with pytest.raises(ValueError):
        select_relevant_times(
            ds,
            "time",
            "time",
            datetime(2024, 1, 1),
            climatology=True,
            end_time=datetime(2024, 1, 2),
        )


def test_climatology_pass_through():
    times = np.arange(12)
    ds = make_time_dataset(times)

    out = select_relevant_times(
        ds,
        "time",
        "time",
        datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 10),
        climatology=True,
    )
    assert out.equals(ds)


def test_int_time_rejected():
    ds = xr.Dataset(
        data_vars={"var": ("time", [1, 2, 3])},
        coords={"time": [1, 2, 3]},
    )

    with pytest.raises(ValueError):
        select_relevant_times(
            ds, "time", "time", datetime(2024, 1, 1), datetime(2024, 1, 2)
        )


def test_cftime_conversion(monkeypatch):
    times = xr.DataArray(
        [cftime.DatetimeGregorian(2024, 1, i + 1) for i in range(3)], dims="time"
    )
    ds = xr.Dataset({"var": ("time", [1, 2, 3])}, coords={"time": times})

    # Mock conversion function to ensure it is called
    def mock_convert(t):
        return np.array(
            ["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]"
        )

    monkeypatch.setattr(
        "roms_tools.datasets.utils.convert_cftime_to_datetime", mock_convert
    )

    out = select_relevant_times(
        ds, "time", "time", datetime(2024, 1, 1), datetime(2024, 1, 3)
    )

    assert np.issubdtype(out["time"].dtype, np.datetime64)


def test_time_range_selection():
    times = np.array(
        ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-10"], dtype="datetime64[ns]"
    )
    ds = make_time_dataset(times)

    out = select_relevant_times(
        ds,
        "time",
        "time",
        datetime(2024, 1, 2),
        datetime(2024, 1, 7),
    )

    # Should include:
    # - closest before start: 2024-01-02
    # - records inside strict range: 2024-01-05
    # - closest after end: 2024-01-10
    expected_times = np.array(
        ["2024-01-02", "2024-01-05", "2024-01-10"],
        dtype="datetime64[ns]",
    )

    assert np.array_equal(out["time"].values, expected_times)


def test_range_selection_missing_before(caplog):
    times = np.array(["2024-01-05", "2024-01-10"], dtype="datetime64[ns]")
    ds = make_time_dataset(times)

    with caplog.at_level(logging.WARNING):
        out = select_relevant_times(
            ds,
            "time",
            "time",
            datetime(2024, 1, 1),
            datetime(2024, 1, 12),
        )

    assert "No records found at or before the start_time" in caplog.text
    # Should fallback to first record
    assert out["time"].values[0] == np.datetime64("2024-01-05")


def test_range_selection_missing_after(caplog):
    times = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    ds = make_time_dataset(times)

    with caplog.at_level(logging.WARNING):
        out = select_relevant_times(
            ds,
            "time",
            "time",
            datetime(2024, 1, 1),
            datetime(2024, 1, 10),
        )

    assert "No records found at or after the end_time" in caplog.text
    assert out["time"].values[-1] == np.datetime64("2024-01-02")


# Tests for _select_initial_time


def test_initial_time_exact_match():
    times = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    ds = make_time_dataset(times)

    out = _select_initial_time(
        ds,
        "time",
        "time",
        datetime(2024, 1, 2),
        climatology=False,
        allow_flex_time=False,
    )

    assert out["time"].values == np.datetime64("2024-01-02")


def test_initial_time_no_exact_match():
    times = np.array(["2024-01-01", "2024-01-03"], dtype="datetime64[ns]")
    ds = make_time_dataset(times)

    with pytest.raises(ValueError):
        _select_initial_time(
            ds,
            "time",
            "time",
            datetime(2024, 1, 2),
            climatology=False,
            allow_flex_time=False,
        )


def test_initial_flexible_time():
    times = np.array(["2024-01-01", "2024-01-02", "2024-01-15"], dtype="datetime64[ns]")
    ds = make_time_dataset(times)

    out = _select_initial_time(
        ds,
        "time",
        "time",
        datetime(2024, 1, 1),
        climatology=False,
        allow_flex_time=True,
    )

    assert out["time"].values == np.datetime64("2024-01-01")


def test_initial_flexible_time_out_of_range():
    times = np.array(["2024-01-10", "2024-01-11"], dtype="datetime64[ns]")
    ds = make_time_dataset(times)

    with pytest.raises(ValueError):
        _select_initial_time(
            ds,
            "time",
            "time",
            datetime(2024, 1, 1),
            climatology=False,
            allow_flex_time=True,
        )


# tests for get_time_type


def test_get_time_type_datetime():
    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    da = xr.DataArray(times)
    assert get_time_type(da) == "datetime"


def test_get_time_type_cftime():
    times = np.array(
        [
            cftime.DatetimeNoLeap(2000, 1, 1),
            cftime.DatetimeNoLeap(2000, 1, 2),
        ],
        dtype=object,
    )
    da = xr.DataArray(times)
    assert get_time_type(da) == "cftime"


def test_get_time_type_integer():
    da = xr.DataArray(np.array([1, 2, 3], dtype=int))
    assert get_time_type(da) == "int"


def test_get_time_type_unsupported_type():
    da = xr.DataArray(np.array(["a", "b"], dtype=object))
    with pytest.raises(ValueError, match="Unsupported data type"):
        get_time_type(da)


def test_get_time_type_invalid_input_type():
    da = xr.DataArray("not-an-array")
    with pytest.raises(ValueError):
        get_time_type(da)


# Tests for convert_cftime_to_datetime


def test_convert_cftime_to_datetime_basic():
    arr = np.array(
        [
            cftime.DatetimeNoLeap(2000, 1, 1),
            cftime.DatetimeNoLeap(2000, 1, 2),
        ]
    )

    converted = convert_cftime_to_datetime(arr)

    assert converted.dtype == "datetime64[ns]"
    assert converted[0] == np.datetime64("2000-01-01")
    assert converted[1] == np.datetime64("2000-01-02")


def test_convert_cftime_to_datetime_mixed_inputs():
    arr = np.array(
        [
            cftime.DatetimeNoLeap(2000, 1, 1),
            "2001-01-01",
            np.datetime64("2002-01-01"),
        ],
        dtype=object,
    )

    converted = convert_cftime_to_datetime(arr)

    assert converted[0] == np.datetime64("2000-01-01")
    assert converted[1] == np.datetime64("2001-01-01")
    assert converted[2] == np.datetime64("2002-01-01")


def test_convert_cftime_to_datetime_returns_numpy_array():
    arr = np.array([cftime.DatetimeNoLeap(1999, 12, 31)])
    converted = convert_cftime_to_datetime(arr)
    assert isinstance(converted, np.ndarray)
