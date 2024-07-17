import pytest
from datetime import datetime
from roms_tools import Grid, AtmosphericForcing, SWRCorrection
from roms_tools.setup.datasets import download_test_data
import xarray as xr
import tempfile
import os
import pooch
import numpy as np


@pytest.fixture
def grid_that_straddles_dateline():
    """
    Fixture for creating a domain that straddles the dateline and lies within the bounds of the regional ERA5 data.
    """
    grid = Grid(
        nx=5,
        ny=5,
        size_x=1800,
        size_y=2400,
        center_lon=-10,
        center_lat=61,
        rot=20,
    )

    return grid


@pytest.fixture
def grid_that_straddles_dateline_but_is_too_big_for_regional_test_data():
    """
    Fixture for creating a domain that straddles the dateline but exceeds the bounds of the regional ERA5 data.
    Centered east of dateline.
    """
    grid = Grid(
        nx=5,
        ny=5,
        size_x=2000,
        size_y=2400,
        center_lon=10,
        center_lat=61,
        rot=20,
    )

    return grid


@pytest.fixture
def another_grid_that_straddles_dateline_but_is_too_big_for_regional_test_data():
    """
    Fixture for creating a domain that straddles the dateline but exceeds the bounds of the regional ERA5 data.
    Centered west of dateline. This one was hard to catch for the nan_check for a long time, but should work now.
    """
    grid = Grid(
        nx=5,
        ny=5,
        size_x=1950,
        size_y=2400,
        center_lon=-30,
        center_lat=61,
        rot=25,
    )

    return grid


@pytest.fixture
def grid_that_lies_east_of_dateline_less_than_five_degrees_away():
    """
    Fixture for creating a domain that lies east of Greenwich meridian, but less than 5 degrees away.
    We care about the 5 degree mark because it decides whether the code handles the longitudes as straddling the dateline or not.
    """

    grid = Grid(
        nx=5,
        ny=5,
        size_x=500,
        size_y=2000,
        center_lon=10,
        center_lat=61,
        rot=0,
    )

    return grid


@pytest.fixture
def grid_that_lies_east_of_dateline_more_than_five_degrees_away():
    """
    Fixture for creating a domain that lies east of Greenwich meridian, more than 5 degrees away.
    We care about the 5 degree mark because it decides whether the code handles the longitudes as straddling the dateline or not.
    """
    grid = Grid(
        nx=5,
        ny=5,
        size_x=500,
        size_y=2400,
        center_lon=15,
        center_lat=61,
        rot=0,
    )

    return grid


@pytest.fixture
def grid_that_lies_west_of_dateline_less_than_five_degrees_away():
    """
    Fixture for creating a domain that lies west of Greenwich meridian, less than 5 degrees away.
    We care about the 5 degree mark because it decides whether the code handles the longitudes as straddling the dateline or not.
    """

    grid = Grid(
        nx=5,
        ny=5,
        size_x=700,
        size_y=2400,
        center_lon=-15,
        center_lat=61,
        rot=0,
    )

    return grid


@pytest.fixture
def grid_that_lies_west_of_dateline_more_than_five_degrees_away():
    """
    Fixture for creating a domain that lies west of Greenwich meridian, more than 5 degrees away.
    We care about the 5 degree mark because it decides whether the code handles the longitudes as straddling the dateline or not.
    """

    grid = Grid(
        nx=5,
        ny=5,
        size_x=1000,
        size_y=2400,
        center_lon=-25,
        center_lat=61,
        rot=0,
    )

    return grid


@pytest.fixture
def grid_that_straddles_180_degree_meridian():
    """
    Fixture for creating a domain that straddles 180 degree meridian. This is a good test grid for the global ERA5 data, which comes on an [-180, 180] longitude grid.
    """

    grid = Grid(
        nx=5,
        ny=5,
        size_x=1800,
        size_y=2400,
        center_lon=180,
        center_lat=61,
        rot=20,
    )

    return grid


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid_that_straddles_dateline",
        "grid_that_lies_east_of_dateline_less_than_five_degrees_away",
        "grid_that_lies_east_of_dateline_more_than_five_degrees_away",
        "grid_that_lies_west_of_dateline_less_than_five_degrees_away",
        "grid_that_lies_west_of_dateline_more_than_five_degrees_away",
    ],
)
def test_successful_initialization_with_regional_data(grid_fixture, request):
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    atm_forcing = AtmosphericForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        source="era5",
        filename=fname,
    )

    assert atm_forcing.ds is not None

    grid.coarsen()

    atm_forcing = AtmosphericForcing(
        grid=grid,
        use_coarse_grid=True,
        start_time=start_time,
        end_time=end_time,
        source="era5",
        filename=fname,
    )

    assert isinstance(atm_forcing.ds, xr.Dataset)
    assert "uwnd" in atm_forcing.ds
    assert "vwnd" in atm_forcing.ds
    assert "swrad" in atm_forcing.ds
    assert "lwrad" in atm_forcing.ds
    assert "Tair" in atm_forcing.ds
    assert "qair" in atm_forcing.ds
    assert "rain" in atm_forcing.ds

    assert atm_forcing.start_time == start_time
    assert atm_forcing.end_time == end_time
    assert atm_forcing.filename == fname
    assert atm_forcing.source == "era5"


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
        "another_grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
    ],
)
def test_nan_detection_initialization_with_regional_data(grid_fixture, request):
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    with pytest.raises(ValueError, match="NaN values found"):

        AtmosphericForcing(
            grid=grid,
            start_time=start_time,
            end_time=end_time,
            source="era5",
            filename=fname,
        )

    grid.coarsen()

    with pytest.raises(ValueError, match="NaN values found"):
        AtmosphericForcing(
            grid=grid,
            use_coarse_grid=True,
            start_time=start_time,
            end_time=end_time,
            source="era5",
            filename=fname,
        )


def test_no_longitude_intersection_initialization_with_regional_data(
    grid_that_straddles_180_degree_meridian,
):
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")

    with pytest.raises(
        ValueError, match="Selected longitude range does not intersect with dataset"
    ):

        AtmosphericForcing(
            grid=grid_that_straddles_180_degree_meridian,
            start_time=start_time,
            end_time=end_time,
            source="era5",
            filename=fname,
        )

    grid_that_straddles_180_degree_meridian.coarsen()

    with pytest.raises(
        ValueError, match="Selected longitude range does not intersect with dataset"
    ):
        AtmosphericForcing(
            grid=grid_that_straddles_180_degree_meridian,
            use_coarse_grid=True,
            start_time=start_time,
            end_time=end_time,
            source="era5",
            filename=fname,
        )


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid_that_straddles_dateline",
        "grid_that_lies_east_of_dateline_less_than_five_degrees_away",
        "grid_that_lies_east_of_dateline_more_than_five_degrees_away",
        "grid_that_lies_west_of_dateline_less_than_five_degrees_away",
        "grid_that_lies_west_of_dateline_more_than_five_degrees_away",
        "grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
        "another_grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
        "grid_that_straddles_180_degree_meridian",
    ],
)
def test_successful_initialization_with_global_data(grid_fixture, request):
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    atm_forcing = AtmosphericForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        source="era5",
        filename=fname,
    )

    assert isinstance(atm_forcing.ds, xr.Dataset)
    assert "uwnd" in atm_forcing.ds
    assert "vwnd" in atm_forcing.ds
    assert "swrad" in atm_forcing.ds
    assert "lwrad" in atm_forcing.ds
    assert "Tair" in atm_forcing.ds
    assert "qair" in atm_forcing.ds
    assert "rain" in atm_forcing.ds

    grid.coarsen()

    atm_forcing = AtmosphericForcing(
        grid=grid,
        use_coarse_grid=True,
        start_time=start_time,
        end_time=end_time,
        source="era5",
        filename=fname,
    )

    assert isinstance(atm_forcing.ds, xr.Dataset)
    assert "uwnd" in atm_forcing.ds
    assert "vwnd" in atm_forcing.ds
    assert "swrad" in atm_forcing.ds
    assert "lwrad" in atm_forcing.ds
    assert "Tair" in atm_forcing.ds
    assert "qair" in atm_forcing.ds
    assert "rain" in atm_forcing.ds

    assert atm_forcing.start_time == start_time
    assert atm_forcing.end_time == end_time
    assert atm_forcing.filename == fname
    assert atm_forcing.source == "era5"


@pytest.fixture
def atmospheric_forcing(grid_that_straddles_dateline):
    """
    Fixture for creating a AtmosphericForcing object.
    """

    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")

    return AtmosphericForcing(
        grid=grid_that_straddles_dateline,
        start_time=start_time,
        end_time=end_time,
        source="era5",
        filename=fname,
    )


def test_atmospheric_forcing_data_consistency_plot_save(atmospheric_forcing, tmp_path):
    """
    Test that the data within the AtmosphericForcing object remains consistent.
    Also test plot and save methods in the same test since we dask arrays are already computed.
    """
    atmospheric_forcing.ds.load()

    # Define the expected data
    expected_uwnd = np.array(
        [
            [
                [9.682143, 9.6873045, 7.658616, 5.34984, 5.8008876, 5.651831, 3.964332],
                [
                    7.2196193,
                    13.155616,
                    13.873741,
                    11.506337,
                    9.11589,
                    11.8453,
                    7.4513154,
                ],
                [
                    8.810431,
                    10.912895,
                    12.71125,
                    11.848574,
                    7.0017467,
                    11.066868,
                    4.995178,
                ],
                [
                    4.5096593,
                    -0.7138924,
                    -4.92656,
                    -6.637835,
                    -3.2835376,
                    -0.41570085,
                    -0.6512133,
                ],
                [
                    7.821298,
                    0.5172747,
                    -2.695407,
                    -0.2485534,
                    -0.8816123,
                    -2.2297409,
                    5.1966243,
                ],
                [
                    4.7915897,
                    -12.2942,
                    -9.683341,
                    -7.89024,
                    -4.4064064,
                    -5.620487,
                    -3.8923697,
                ],
                [
                    -1.8733679,
                    -5.0046277,
                    -2.4120338,
                    -0.05066272,
                    -4.6232734,
                    -5.434345,
                    -5.786118,
                ],
            ],
            [
                [
                    9.934675,
                    9.0043335,
                    8.245487,
                    5.4943433,
                    6.186509,
                    5.6049147,
                    3.923312,
                ],
                [
                    6.519705,
                    12.86845,
                    13.569553,
                    11.566428,
                    9.4294405,
                    11.291251,
                    7.618743,
                ],
                [
                    8.19874,
                    8.630614,
                    11.540438,
                    12.164708,
                    6.582674,
                    11.751556,
                    6.5439706,
                ],
                [
                    4.421919,
                    -0.12123968,
                    -4.60852,
                    -6.07126,
                    -2.3052542,
                    -1.2082452,
                    -0.98714674,
                ],
                [
                    7.874192,
                    1.8622701,
                    -2.7311177,
                    -1.2987597,
                    -0.5676282,
                    -1.577043,
                    4.33589,
                ],
                [
                    4.973361,
                    -12.578562,
                    -9.460623,
                    -7.4661274,
                    -4.6160173,
                    -5.7835197,
                    -4.8997827,
                ],
                [
                    -1.7371889,
                    -5.1384463,
                    -2.1262898,
                    0.12077145,
                    -4.3513064,
                    -5.11245,
                    -5.351688,
                ],
            ],
        ],
        dtype=np.float32,
    )

    expected_vwnd = np.array(
        [
            [
                [
                    3.5414524,
                    1.3840214,
                    -0.27970433,
                    2.5684514,
                    5.6336703,
                    4.5941186,
                    1.8153937,
                ],
                [
                    -3.7881892,
                    -4.406904,
                    -0.6925203,
                    5.257118,
                    5.949448,
                    6.6912074,
                    4.811325,
                ],
                [
                    -9.63415,
                    -10.073754,
                    -5.496115,
                    7.593431,
                    6.148733,
                    5.8233504,
                    8.856961,
                ],
                [
                    -9.65799,
                    -8.555591,
                    -7.7859373,
                    -5.8133893,
                    -6.165076,
                    -4.4117365,
                    5.0667415,
                ],
                [
                    -2.6980543,
                    -3.2350047,
                    -3.2634366,
                    -3.1665637,
                    -11.98655,
                    -12.107817,
                    17.101234,
                ],
                [
                    6.97365,
                    5.451807,
                    -1.5713935,
                    -8.296681,
                    -13.055259,
                    -2.9508927,
                    2.0257008,
                ],
                [
                    -2.0955405,
                    -0.13483004,
                    -0.3349161,
                    -0.19222538,
                    -8.381893,
                    -8.231602,
                    -5.982184,
                ],
            ],
            [
                [
                    2.8066342,
                    1.7998191,
                    -0.23953092,
                    2.3238611,
                    5.058705,
                    4.522139,
                    1.9135665,
                ],
                [
                    -3.4791694,
                    -4.4220605,
                    -0.89243746,
                    5.6455116,
                    5.857368,
                    6.829339,
                    5.0575895,
                ],
                [
                    -8.4660425,
                    -10.995333,
                    -4.230904,
                    6.6700015,
                    5.9140325,
                    6.410501,
                    8.075157,
                ],
                [
                    -8.727736,
                    -8.250191,
                    -7.804627,
                    -6.4764595,
                    -6.964692,
                    -4.318679,
                    4.895726,
                ],
                [
                    -2.5031612,
                    -4.610895,
                    -3.2356691,
                    -3.8183403,
                    -11.733936,
                    -13.136506,
                    17.504322,
                ],
                [
                    6.6591873,
                    5.404177,
                    -1.2032545,
                    -8.633326,
                    -13.144289,
                    -3.8026283,
                    1.4954354,
                ],
                [
                    -2.5334792,
                    -0.11332972,
                    -0.42115375,
                    -0.33742002,
                    -8.273639,
                    -8.526825,
                    -6.8844285,
                ],
            ],
        ],
        dtype=np.float32,
    )

    expected_swrad = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=np.float32,
    )

    expected_lwrad = np.array(
        [
            [
                [
                    351.22202,
                    307.02017,
                    314.76047,
                    361.5371,
                    355.78796,
                    344.0307,
                    338.98972,
                ],
                [
                    274.61188,
                    273.76956,
                    294.55707,
                    288.39404,
                    330.12332,
                    353.84125,
                    350.8963,
                ],
                [
                    258.14426,
                    312.65857,
                    320.44858,
                    325.92133,
                    275.2125,
                    308.33493,
                    343.80405,
                ],
                [263.9574, 262.3807, 272.43158, 258.7095, 254.5646, 317.14734, 306.476],
                [
                    252.87622,
                    255.07326,
                    227.7193,
                    280.09802,
                    277.74237,
                    302.9409,
                    265.9497,
                ],
                [
                    216.22571,
                    293.07065,
                    249.55763,
                    279.32294,
                    302.5996,
                    250.66896,
                    246.8607,
                ],
                [
                    203.67563,
                    226.4481,
                    189.01756,
                    157.1616,
                    233.10999,
                    273.19785,
                    292.6843,
                ],
            ],
            [
                [340.6211, 325.79004, 306.85, 347.59702, 356.6355, 343.4543, 336.4094],
                [
                    275.32065,
                    277.87665,
                    279.90936,
                    279.5159,
                    299.0584,
                    355.40555,
                    351.1922,
                ],
                [
                    241.89168,
                    319.70547,
                    303.0773,
                    326.4687,
                    272.3111,
                    307.7095,
                    345.46054,
                ],
                [
                    261.66803,
                    258.12085,
                    277.25986,
                    255.81079,
                    250.32898,
                    316.02298,
                    310.90982,
                ],
                [
                    244.40097,
                    266.35535,
                    227.59578,
                    279.96176,
                    285.27383,
                    302.9465,
                    279.0034,
                ],
                [
                    214.79074,
                    295.196,
                    243.09726,
                    280.26495,
                    297.957,
                    244.42346,
                    238.17075,
                ],
                [
                    202.25783,
                    224.2656,
                    185.28705,
                    156.95543,
                    219.1866,
                    275.84387,
                    290.03543,
                ],
            ],
        ],
        dtype=np.float32,
    )

    expected_Tair = np.array(
        [
            [
                [
                    13.983097,
                    12.288186,
                    12.024256,
                    11.237248,
                    11.24234,
                    10.024218,
                    8.906293,
                ],
                [
                    9.439986,
                    9.884249,
                    9.821308,
                    10.267262,
                    9.899462,
                    9.2958765,
                    8.312866,
                ],
                [6.102492, 8.619774, 9.774704, 9.046288, 9.237789, 8.893497, 6.9524736],
                [
                    3.8028858,
                    6.557079,
                    6.1228037,
                    5.3670797,
                    4.5178456,
                    4.998852,
                    3.734068,
                ],
                [
                    1.9243673,
                    4.0067954,
                    0.44351077,
                    -0.04124513,
                    2.4300609,
                    3.9442632,
                    5.5764585,
                ],
                [
                    -0.606846,
                    1.748711,
                    -4.0239983,
                    -3.980118,
                    -0.4307354,
                    2.98153,
                    2.7953317,
                ],
                [
                    -8.650124,
                    -8.689847,
                    -14.453805,
                    -20.8693,
                    -14.023803,
                    -4.911254,
                    1.2565224,
                ],
            ],
            [
                [
                    14.030856,
                    12.502566,
                    11.992314,
                    11.134161,
                    11.112591,
                    9.92276,
                    8.849695,
                ],
                [9.363431, 9.737764, 9.71652, 10.064842, 9.670996, 9.355096, 8.369347],
                [5.962798, 8.298665, 9.665368, 9.0844145, 8.935353, 8.914775, 7.104829],
                [
                    3.851593,
                    6.324237,
                    6.063094,
                    5.198216,
                    4.311827,
                    4.7639103,
                    3.6383307,
                ],
                [
                    1.7128028,
                    3.7774146,
                    0.3994808,
                    -0.05175301,
                    2.3984277,
                    3.958246,
                    5.514753,
                ],
                [
                    -0.5678973,
                    1.7564851,
                    -4.0164347,
                    -4.0414805,
                    -0.32657683,
                    2.8409872,
                    2.7383697,
                ],
                [
                    -8.6348295,
                    -8.697271,
                    -14.417632,
                    -20.815548,
                    -14.016626,
                    -4.727944,
                    1.2773117,
                ],
            ],
        ],
        dtype=np.float32,
    )

    expected_qair = np.array(
        [
            [
                [
                    0.00894345,
                    0.00785896,
                    0.00766562,
                    0.00795309,
                    0.00754406,
                    0.0068888,
                    0.00648944,
                ],
                [
                    0.00525541,
                    0.00561289,
                    0.00593816,
                    0.00636572,
                    0.00642818,
                    0.00671865,
                    0.006304,
                ],
                [
                    0.0040144,
                    0.00533809,
                    0.00635292,
                    0.00625088,
                    0.00543018,
                    0.00617801,
                    0.0059004,
                ],
                [
                    0.00313334,
                    0.00394297,
                    0.00393208,
                    0.00371642,
                    0.0035812,
                    0.00410653,
                    0.00402102,
                ],
                [
                    0.00279723,
                    0.00332423,
                    0.00271788,
                    0.00315154,
                    0.00386289,
                    0.00457509,
                    0.00486981,
                ],
                [
                    0.00233743,
                    0.00337603,
                    0.00205563,
                    0.00208385,
                    0.0032772,
                    0.0039004,
                    0.00332571,
                ],
                [
                    0.00140172,
                    0.00130828,
                    0.00079319,
                    0.00048005,
                    0.0010781,
                    0.00190267,
                    0.0038644,
                ],
            ],
            [
                [
                    0.00892763,
                    0.00798396,
                    0.00761596,
                    0.0077239,
                    0.00759585,
                    0.00690188,
                    0.0064849,
                ],
                [
                    0.00519145,
                    0.0055212,
                    0.0058503,
                    0.00601724,
                    0.00637109,
                    0.00673753,
                    0.00632534,
                ],
                [
                    0.00375761,
                    0.00536452,
                    0.0064391,
                    0.00629855,
                    0.00541504,
                    0.00611869,
                    0.00598616,
                ],
                [
                    0.00296714,
                    0.00396571,
                    0.00395476,
                    0.00371533,
                    0.00355474,
                    0.00391858,
                    0.00401376,
                ],
                [
                    0.00286595,
                    0.00343385,
                    0.00271064,
                    0.00318937,
                    0.00388673,
                    0.0045439,
                    0.0048316,
                ],
                [
                    0.0024749,
                    0.00356596,
                    0.00205163,
                    0.00206788,
                    0.00322582,
                    0.0039785,
                    0.00332241,
                ],
                [
                    0.00140471,
                    0.00130097,
                    0.00078171,
                    0.00047431,
                    0.00108264,
                    0.00190718,
                    0.00384446,
                ],
            ],
        ],
        dtype=np.float32,
    )

    expected_rain = np.array(
        [
            [
                [
                    2.99456865e-01,
                    1.05213928e00,
                    0.00000000e00,
                    2.92596889e00,
                    2.16856766e00,
                    9.86347020e-01,
                    3.54191065e-01,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    1.16903797e-01,
                    2.34461692e-03,
                    1.01448540e-02,
                    3.54999751e-01,
                    9.01469827e-01,
                ],
                [
                    3.39331217e-02,
                    4.83708292e-01,
                    3.10700893e-01,
                    5.93945026e-01,
                    5.65721747e-03,
                    4.32890141e-04,
                    2.40207815e00,
                ],
                [
                    9.64358076e-02,
                    5.15322611e-02,
                    3.37415561e-02,
                    3.20277736e-02,
                    6.11294934e-04,
                    1.10518873e00,
                    6.56543672e-01,
                ],
                [
                    1.37753397e-01,
                    6.00837469e-02,
                    2.02192962e-02,
                    7.50673488e-02,
                    3.29248726e-01,
                    2.23563838e00,
                    1.65411806e00,
                ],
                [
                    6.04586899e-02,
                    6.80949390e-01,
                    9.37972683e-03,
                    3.57965007e-02,
                    1.48106456e-01,
                    2.34549433e-01,
                    2.97144018e-02,
                ],
                [
                    3.37868631e-01,
                    7.55462587e-01,
                    1.19116351e-01,
                    1.64777692e-02,
                    4.59388383e-02,
                    4.48680371e-02,
                    5.13717473e-01,
                ],
            ],
            [
                [
                    9.54854414e-02,
                    4.83684591e-04,
                    0.00000000e00,
                    8.36170137e-01,
                    3.24128366e00,
                    1.19893610e00,
                    4.24858660e-01,
                ],
                [
                    0.00000000e00,
                    6.45908201e-03,
                    1.02696314e-01,
                    6.28535869e-04,
                    8.28882027e-03,
                    1.78970858e-01,
                    1.18088460e00,
                ],
                [
                    0.00000000e00,
                    7.09786594e-01,
                    3.39415789e-01,
                    4.49054897e-01,
                    1.98132977e-01,
                    1.27551390e-03,
                    1.18759966e00,
                ],
                [
                    8.88594091e-02,
                    1.43849060e-01,
                    3.37909535e-02,
                    3.57030928e-02,
                    0.00000000e00,
                    1.27599168e00,
                    5.78730941e-01,
                ],
                [
                    2.08939224e-01,
                    2.21915424e-01,
                    1.95211619e-02,
                    3.41181546e-01,
                    3.55485380e-01,
                    2.10859799e00,
                    1.84965956e00,
                ],
                [
                    1.05249137e-01,
                    8.96957159e-01,
                    8.17435328e-03,
                    3.73747423e-02,
                    2.65661597e-01,
                    2.67820716e-01,
                    6.45710912e-04,
                ],
                [
                    3.47904205e-01,
                    8.09429526e-01,
                    1.34275183e-01,
                    1.47897974e-02,
                    4.69088703e-02,
                    3.35826539e-02,
                    5.29768586e-01,
                ],
            ],
        ],
        dtype=np.float32,
    )

    # Check the values in the dataset
    assert np.allclose(atmospheric_forcing.ds["uwnd"].values, expected_uwnd)
    assert np.allclose(atmospheric_forcing.ds["vwnd"].values, expected_vwnd)
    assert np.allclose(atmospheric_forcing.ds["swrad"].values, expected_swrad)
    assert np.allclose(atmospheric_forcing.ds["lwrad"].values, expected_lwrad)
    assert np.allclose(atmospheric_forcing.ds["Tair"].values, expected_Tair)
    assert np.allclose(atmospheric_forcing.ds["qair"].values, expected_qair)
    assert np.allclose(atmospheric_forcing.ds["rain"].values, expected_rain)

    atmospheric_forcing.plot(varname="uwnd", time=0)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    atmospheric_forcing.save(filepath)
    extended_filepath = filepath + ".20200201-01.nc"

    try:
        assert os.path.exists(extended_filepath)
    finally:
        os.remove(extended_filepath)


# SWRCorrection checks


@pytest.fixture
def swr_correction():

    correction_filename = pooch.retrieve(
        url="https://github.com/CWorthy-ocean/roms-tools-data/raw/main/SSR_correction.nc",
        known_hash="a170c1698e6cc2765b3f0bb51a18c6a979bc796ac3a4c014585aeede1f1f8ea0",
    )
    correction_filename

    return SWRCorrection(
        filename=correction_filename,
        varname="ssr_corr",
        dim_names={"time": "time", "latitude": "latitude", "longitude": "longitude"},
        temporal_resolution="climatology",
    )


def test_check_dataset(swr_correction):

    ds = swr_correction.ds.copy()
    ds = ds.drop_vars("ssr_corr")
    with pytest.raises(ValueError):
        swr_correction._check_dataset(ds)

    ds = swr_correction.ds.copy()
    ds = ds.rename({"latitude": "lat", "longitude": "long"})
    with pytest.raises(ValueError):
        swr_correction._check_dataset(ds)


def test_ensure_latitude_ascending(swr_correction):

    ds = swr_correction.ds.copy()

    ds["latitude"] = ds["latitude"][::-1]
    ds = swr_correction._ensure_latitude_ascending(ds)
    assert np.all(np.diff(ds["latitude"]) > 0)


def test_handle_longitudes(swr_correction):
    swr_correction.ds["longitude"] = (
        (swr_correction.ds["longitude"] + 180) % 360
    ) - 180  # Convert to [-180, 180]
    swr_correction._handle_longitudes(straddle=False)
    assert np.all(
        (swr_correction.ds["longitude"] >= 0) & (swr_correction.ds["longitude"] <= 360)
    )


def test_choose_subdomain(swr_correction):
    lats = swr_correction.ds.latitude[10:20]
    lons = swr_correction.ds.longitude[10:20]
    coords = {"latitude": lats, "longitude": lons}
    subdomain = swr_correction._choose_subdomain(coords)
    assert (subdomain["latitude"] == lats).all()
    assert (subdomain["longitude"] == lons).all()


def test_interpolate_temporally(swr_correction):
    field = swr_correction.ds["ssr_corr"]

    fname = download_test_data("ERA5_regional_test_data.nc")
    era5_times = xr.open_dataset(fname).time
    interpolated_field = swr_correction._interpolate_temporally(field, era5_times)
    assert len(interpolated_field.time) == len(era5_times)
