import numpy as np
import xarray as xr

from roms_tools.datasets.utils import convert_to_float64


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
