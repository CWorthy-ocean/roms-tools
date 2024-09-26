import pytest
from roms_tools.setup.datasets import GLORYSDataset, ERA5Dataset, CESMBGCDataset, CESMBGCSurfaceForcingDataset, TPXODataset
from roms_tools.setup.download import download_test_data
from roms_tools.setup.fill import LateralFill
from roms_tools.setup.utils import extrapolate_deepest_to_bottom
from datetime import datetime
import numpy as np


@pytest.fixture()
def era5_data(request, use_dask):
    fname = download_test_data("ERA5_regional_test_data.nc")
    data = ERA5Dataset(
        filename=fname,
        start_time=datetime(2020, 1, 31),
        end_time=datetime(2020, 2, 2),
        use_dask=use_dask,
    )
    data.post_process()

    return data


@pytest.fixture()
def glorys_data(request, use_dask):
    fname = download_test_data("GLORYS_NA_2012.nc")

    data = GLORYSDataset(
        filename=fname,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2013, 1, 1),
        use_dask=use_dask,
    )
    data.post_process()

    print(data.var_names)
    # extrapolate deepest value to bottom so all levels can use the same surface mask
    for var in data.var_names:
        if var != "zeta":
            data.ds[data.var_names[var]] = extrapolate_deepest_to_bottom(
                data.ds[data.var_names[var]], data.dim_names["depth"]
                )

    return data

@pytest.fixture()
def tpxo_data(request, use_dask):
    fname = download_test_data("TPXO_regional_test_data.nc")

    data = TPXODataset(
        filename=fname,
        use_dask=use_dask,
    )
    data.post_process()

    return data

@pytest.fixture()
def cesm_bgc_data(request, use_dask):
    fname = download_test_data("CESM_BGC_2012.nc")

    data = CESMBGCDataset(
        filename=fname,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2013, 1, 1),
        climatology=False,
        use_dask=use_dask,
    )
    data.post_process()
    
    # extrapolate deepest value to bottom so all levels can use the same surface mask
    for var in data.var_names:
        data.ds[data.var_names[var]] = extrapolate_deepest_to_bottom(
            data.ds[data.var_names[var]], data.dim_names["depth"]
            )

    return data

@pytest.fixture()
def cesm_surface_bgc_data(request, use_dask):
    fname = download_test_data("CESM_BGC_SURFACE_2012.nc")

    data = CESMBGCSurfaceForcingDataset(
        filename=fname,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2013, 1, 1),
        climatology=True,
        use_dask=use_dask,
    )
    data.post_process()

    return data

@pytest.mark.parametrize(
    "data_fixture",
    [
        "era5_data",
        "glorys_data",
        "tpxo_data",
        "cesm_bgc_data",
        "cesm_surface_bgc_data"
    ],
)
def test_lateral_fill_no_nans(data_fixture, request, use_dask):
    data = request.getfixturevalue(data_fixture)
    lateral_fill = LateralFill(
        data.ds["mask"],
        [data.dim_names["latitude"], data.dim_names["longitude"]],
    )

    for var in data.var_names:
        filled = lateral_fill.apply(data.ds[data.var_names[var]].astype(np.float64))
        assert not filled.isnull().any()

@pytest.mark.parametrize(
    "data_fixture",
    [
        "era5_data",
        "glorys_data",
    ],
)
def test_lateral_fill_reproducibility(data_fixture, request, use_dask):

    data = request.getfixturevalue(data_fixture)
    lateral_fill0 = LateralFill(
        data.ds["mask"],
        [data.dim_names["latitude"], data.dim_names["longitude"]],
    )
    lateral_fill1 = LateralFill(
        data.ds["mask"],
        [data.dim_names["latitude"], data.dim_names["longitude"]],
    )

    ds0 = data.ds.copy()
    ds1 = data.ds.copy()

    for var in data.var_names:

        ds0[data.var_names[var]] = lateral_fill0.apply(
            ds0[data.var_names[var]].astype(np.float64)
        )
        ds1[data.var_names[var]] = lateral_fill1.apply(
            ds1[data.var_names[var]].astype(np.float64)
        )

    assert ds0.equals(ds1)
