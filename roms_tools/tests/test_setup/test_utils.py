from roms_tools.setup.utils import interpolate_from_climatology
from roms_tools.setup.datasets import ERA5Correction
from roms_tools.setup.download import download_test_data
import xarray as xr


def test_interpolate_from_climatology():

    fname = download_test_data("ERA5_regional_test_data.nc")
    era5_times = xr.open_dataset(fname).time

    climatology = ERA5Correction()
    field = climatology.ds["ssr_corr"]

    interpolated_field = interpolate_from_climatology(field, "time", era5_times)
    assert len(interpolated_field.time) == len(era5_times)


