import pytest
from datetime import datetime
from roms_tools import Grid, AtmosphericForcing, SWRCorrection
from roms_tools.setup.download import download_test_data
import xarray as xr
import tempfile
import os
import pooch
import numpy as np
import textwrap


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
        source="ERA5",
        filename=fname,
    )

    assert atm_forcing.ds is not None

    grid.coarsen()

    atm_forcing = AtmosphericForcing(
        grid=grid,
        use_coarse_grid=True,
        start_time=start_time,
        end_time=end_time,
        source="ERA5",
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
    assert atm_forcing.source == "ERA5"


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
            source="ERA5",
            filename=fname,
        )

    grid.coarsen()

    with pytest.raises(ValueError, match="NaN values found"):
        AtmosphericForcing(
            grid=grid,
            use_coarse_grid=True,
            start_time=start_time,
            end_time=end_time,
            source="ERA5",
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
            source="ERA5",
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
            source="ERA5",
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
        source="ERA5",
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
        source="ERA5",
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
    assert atm_forcing.source == "ERA5"


@pytest.fixture
def atmospheric_forcing(grid_that_straddles_180_degree_meridian):
    """
    Fixture for creating a AtmosphericForcing object.
    """

    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")

    return AtmosphericForcing(
        grid=grid_that_straddles_180_degree_meridian,
        start_time=start_time,
        end_time=end_time,
        source="ERA5",
        filename=fname,
    )


@pytest.fixture
def corrected_atmospheric_forcing(grid_that_straddles_180_degree_meridian):
    """
    Fixture for creating a AtmosphericForcing object with shortwave radiation correction.
    """
    correction_filename = pooch.retrieve(
        url="https://github.com/CWorthy-ocean/roms-tools-data/raw/main/SSR_correction.nc",
        known_hash="a170c1698e6cc2765b3f0bb51a18c6a979bc796ac3a4c014585aeede1f1f8ea0",
    )
    correction = SWRCorrection(
        filename=correction_filename,
        varname="ssr_corr",
        dim_names={
            "longitude": "longitude",
            "latitude": "latitude",
            "time": "time",
        },
        temporal_resolution="climatology",
    )

    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")

    return AtmosphericForcing(
        grid=grid_that_straddles_180_degree_meridian,
        start_time=start_time,
        end_time=end_time,
        source="ERA5",
        filename=fname,
        swr_correction=correction,
    )


@pytest.mark.parametrize(
    "atm_forcing_fixture, expected_swrad",
    [
        (
            "atmospheric_forcing",
            [
                np.array(
                    [
                        [
                            [
                                2.5448965e01,
                                8.8304184e01,
                                1.5497435e02,
                                2.1024680e02,
                                2.8283499e02,
                                2.6214133e02,
                                2.4970605e02,
                            ],
                            [
                                2.5577042e01,
                                2.7288372e01,
                                1.4094499e02,
                                1.7884666e02,
                                2.6748840e02,
                                1.2747585e02,
                                2.2650435e02,
                            ],
                            [
                                1.3132906e02,
                                1.3878809e02,
                                1.5093468e02,
                                2.1958447e02,
                                1.0976684e02,
                                8.2733292e01,
                                1.0512259e02,
                            ],
                            [
                                7.4981323e01,
                                1.2092771e02,
                                7.1995018e01,
                                6.9545914e01,
                                4.7777473e01,
                                3.1707375e01,
                                5.3525272e01,
                            ],
                            [
                                3.9274761e01,
                                3.0639046e01,
                                4.3424637e01,
                                2.4530054e01,
                                1.6565577e01,
                                1.3347603e01,
                                2.5200823e01,
                            ],
                            [
                                1.7271753e01,
                                1.4315107e01,
                                1.3658159e01,
                                8.3364201e00,
                                3.3492086e00,
                                2.3186626e00,
                                2.1239614e00,
                            ],
                            [
                                1.0622320e01,
                                6.2983389e00,
                                1.1909541e00,
                                0.0000000e00,
                                0.0000000e00,
                                0.0000000e00,
                                0.0000000e00,
                            ],
                        ],
                        [
                            [
                                4.1088528e01,
                                9.1928207e01,
                                2.3474031e02,
                                2.4281172e02,
                                2.6647174e02,
                                2.4823715e02,
                                2.1143315e02,
                            ],
                            [
                                3.6657497e01,
                                2.7476831e01,
                                9.0126961e01,
                                1.7404645e02,
                                2.3448058e02,
                                1.1212284e02,
                                2.0697829e02,
                            ],
                            [
                                1.4556892e02,
                                1.6401329e02,
                                1.5902567e02,
                                2.2931277e02,
                                8.7306885e01,
                                7.2889664e01,
                                8.1363327e01,
                            ],
                            [
                                8.6130676e01,
                                1.4423856e02,
                                7.9196968e01,
                                6.6386658e01,
                                4.0398537e01,
                                2.5448185e01,
                                4.4373856e01,
                            ],
                            [
                                6.2564697e01,
                                4.2063602e01,
                                5.5343834e01,
                                2.2452137e01,
                                1.8183729e01,
                                1.0362802e01,
                                1.8347992e01,
                            ],
                            [
                                3.0191124e01,
                                2.2519470e01,
                                2.0529692e01,
                                1.1819255e01,
                                3.7765646e00,
                                1.6394116e00,
                                2.3138286e-01,
                            ],
                            [
                                2.4330065e01,
                                1.2172291e01,
                                2.9702284e00,
                                4.5030624e-02,
                                0.0000000e00,
                                0.0000000e00,
                                0.0000000e00,
                            ],
                        ],
                    ],
                    dtype=np.float32,
                )
            ],
        ),
        (
            "corrected_atmospheric_forcing",
            [
                np.array(
                    [
                        [
                            [2.0345396e01, 3.2848579e01],
                            [7.1687141e01, 7.4629196e01],
                            [1.2855386e02, 1.9472108e02],
                            [1.7835985e02, 2.0598582e02],
                            [2.4759137e02, 2.3326712e02],
                            [2.3055316e02, 2.1832445e02],
                            [2.1866821e02, 1.8515254e02],
                        ],
                        [
                            [1.8974031e01, 2.7193939e01],
                            [2.0692104e01, 2.0835009e01],
                            [1.1139817e02, 7.1233307e01],
                            [1.4908310e02, 1.4508174e02],
                            [2.3330269e02, 2.0451334e02],
                            [1.1800186e02, 1.0378988e02],
                            [2.1442001e02, 1.9593568e02],
                        ],
                        [
                            [9.5367859e01, 1.0570849e02],
                            [1.1730759e02, 1.3862865e02],
                            [1.3710544e02, 1.4445511e02],
                            [2.1596344e02, 2.2553131e02],
                            [1.1043810e02, 8.7840797e01],
                            [7.6220886e01, 6.7152107e01],
                            [9.0121040e01, 6.9752350e01],
                        ],
                        [
                            [7.7221687e01, 8.8704170e01],
                            [1.2257743e02, 1.4620630e02],
                            [7.8461441e01, 8.6310257e01],
                            [7.3550613e01, 7.0209435e01],
                            [4.7777473e01, 4.0398537e01],
                            [3.1707375e01, 2.5448185e01],
                            [5.4698215e01, 4.5346252e01],
                        ],
                        [
                            [3.9274761e01, 6.2564697e01],
                            [3.0638720e01, 4.2063152e01],
                            [4.3393879e01, 5.5304630e01],
                            [2.4530497e01, 2.2452543e01],
                            [1.6565577e01, 1.8183731e01],
                            [1.3347603e01, 1.0362802e01],
                            [2.5458445e01, 1.8535559e01],
                        ],
                        [
                            [1.7271362e01, 3.0190443e01],
                            [1.4313513e01, 2.2516962e01],
                            [1.3647677e01, 2.0513937e01],
                            [8.3360624e00, 1.1818749e01],
                            [3.3492086e00, 3.7765646e00],
                            [2.3186626e00, 1.6394116e00],
                            [2.1239614e00, 2.3138286e-01],
                        ],
                        [
                            [1.0622130e01, 2.4329630e01],
                            [6.2982388e00, 1.2172097e01],
                            [1.1909422e00, 2.9701986e00],
                            [0.0000000e00, 4.5030624e-02],
                            [0.0000000e00, 0.0000000e00],
                            [0.0000000e00, 0.0000000e00],
                            [0.0000000e00, 0.0000000e00],
                        ],
                    ],
                    dtype=np.float32,
                )
            ],
        ),
    ],
)
def test_atmospheric_forcing_data_consistency_plot_save(
    atm_forcing_fixture, expected_swrad, request, tmp_path
):
    """
    Test that the data within the AtmosphericForcing object remains consistent.
    Also test plot and save methods in the same test since we dask arrays are already computed.
    """
    atm_forcing = request.getfixturevalue(atm_forcing_fixture)

    atm_forcing.ds.load()

    expected_uwnd = np.array(
        [
            [
                [
                    -6.007625,
                    -3.42977,
                    0.21806262,
                    5.360945,
                    11.831764,
                    12.798729,
                    11.391347,
                ],
                [
                    -6.8175993,
                    -6.792098,
                    -0.18560751,
                    3.0563045,
                    8.710696,
                    -9.596989,
                    8.090187,
                ],
                [
                    -13.518117,
                    -2.7482898,
                    5.891369,
                    5.1779666,
                    -7.379195,
                    -0.9924428,
                    2.5351613,
                ],
                [
                    -4.3629227,
                    -6.437724,
                    -13.663748,
                    -12.0565195,
                    -1.2215672,
                    1.772012,
                    1.8781031,
                ],
                [
                    0.47687545,
                    -1.5980082,
                    -1.5401305,
                    0.6282165,
                    -3.6321266,
                    1.5924206,
                    1.7971114,
                ],
                [
                    0.31979027,
                    0.78913784,
                    0.7617095,
                    2.0062523,
                    3.6265984,
                    1.6264036,
                    3.1069543,
                ],
                [
                    -0.7392475,
                    0.82420295,
                    2.6924515,
                    4.395664,
                    5.582694,
                    5.9560614,
                    5.92299,
                ],
            ],
            [
                [
                    -6.615435,
                    -4.2648177,
                    -0.41911495,
                    4.795286,
                    10.781866,
                    13.12497,
                    11.968585,
                ],
                [
                    -6.980567,
                    -6.5854936,
                    -1.6141279,
                    3.2338471,
                    8.506243,
                    -9.9049225,
                    7.0150843,
                ],
                [
                    -13.734081,
                    -4.05169,
                    5.714791,
                    4.696765,
                    -7.2288737,
                    -2.1190686,
                    2.7367542,
                ],
                [
                    -4.908295,
                    -6.761066,
                    -13.2362175,
                    -11.991249,
                    -0.5408727,
                    2.354013,
                    2.0929167,
                ],
                [
                    0.41402856,
                    -1.3379232,
                    -1.4920205,
                    0.5738855,
                    -3.6704328,
                    1.5384829,
                    2.1192918,
                ],
                [
                    0.30093297,
                    0.971923,
                    0.93931144,
                    2.0213463,
                    3.6705747,
                    1.5487708,
                    2.9228706,
                ],
                [
                    -0.54992414,
                    0.91518635,
                    2.853546,
                    4.5043445,
                    5.6799273,
                    6.0500693,
                    6.0166245,
                ],
            ],
        ],
        dtype=np.float32,
    )

    expected_vwnd = np.array(
        [
            [
                [
                    15.366679,
                    16.297422,
                    11.619374,
                    7.6118455,
                    -0.17023174,
                    -0.70611537,
                    0.918054,
                ],
                [
                    5.8549976,
                    12.26627,
                    8.788221,
                    0.95048386,
                    -3.6778722,
                    2.183071,
                    -8.522625,
                ],
                [
                    -4.818965,
                    -2.2781372,
                    0.78555244,
                    -2.8322685,
                    8.696728,
                    -1.6022366,
                    -6.2568765,
                ],
                [
                    -8.705794,
                    -6.312195,
                    -8.364019,
                    -8.063112,
                    11.151511,
                    -0.7562826,
                    -3.3305223,
                ],
                [
                    0.5503583,
                    -0.5612107,
                    -3.0128725,
                    -3.231436,
                    -4.0373354,
                    -4.1747046,
                    -1.7400578,
                ],
                [
                    -2.0023136,
                    -1.1348844,
                    -1.0493268,
                    -1.3957449,
                    -5.500842,
                    -2.139104,
                    0.51574695,
                ],
                [
                    -3.2449172,
                    -3.0666091,
                    -2.4833071,
                    -1.8068211,
                    -0.78668225,
                    0.16977428,
                    0.3107974,
                ],
            ],
            [
                [
                    15.095973,
                    16.516485,
                    12.527047,
                    7.781989,
                    0.30071807,
                    -1.2423985,
                    1.2902508,
                ],
                [
                    4.8637633,
                    12.339098,
                    9.587799,
                    2.498924,
                    -3.3265023,
                    1.8309238,
                    -8.718385,
                ],
                [
                    -4.9639883,
                    -2.5414722,
                    1.0753635,
                    -1.1787742,
                    9.358983,
                    -1.7478623,
                    -5.962756,
                ],
                [
                    -8.519542,
                    -5.2647786,
                    -8.270556,
                    -8.179985,
                    11.100576,
                    -0.04957591,
                    -3.1815748,
                ],
                [
                    0.52395123,
                    -0.5520083,
                    -2.9008386,
                    -3.410807,
                    -4.2334967,
                    -3.9974525,
                    -1.6043161,
                ],
                [
                    -2.1034713,
                    -1.2443935,
                    -1.030919,
                    -1.4211166,
                    -5.52376,
                    -2.4278605,
                    0.50510705,
                ],
                [
                    -3.3849669,
                    -3.3259943,
                    -2.9783287,
                    -1.9397556,
                    -1.0758145,
                    -0.05373563,
                    0.31708276,
                ],
            ],
        ],
        dtype=np.float32,
    )

    expected_lwrad = np.array(
        [
            [
                [
                    349.6652,
                    329.73444,
                    309.8111,
                    277.24893,
                    269.887,
                    259.9545,
                    252.45737,
                ],
                [
                    334.3932,
                    324.30423,
                    303.3417,
                    279.31738,
                    239.14821,
                    268.40637,
                    218.19089,
                ],
                [
                    272.54703,
                    261.52417,
                    250.08763,
                    211.38737,
                    274.8412,
                    180.4204,
                    157.67314,
                ],
                [
                    248.60202,
                    238.70256,
                    265.10126,
                    272.1406,
                    229.21912,
                    204.43091,
                    163.18147,
                ],
                [
                    182.60025,
                    159.89,
                    168.08347,
                    145.50589,
                    175.71254,
                    171.32625,
                    145.81366,
                ],
                [
                    167.89578,
                    161.10167,
                    144.19185,
                    147.91838,
                    174.96294,
                    173.1659,
                    133.38031,
                ],
                [
                    168.20734,
                    154.56114,
                    139.15524,
                    131.65768,
                    129.26778,
                    138.174,
                    167.19113,
                ],
            ],
            [
                [
                    349.78278,
                    328.78848,
                    276.55316,
                    273.44193,
                    266.28256,
                    254.63882,
                    252.37021,
                ],
                [
                    335.17593,
                    324.55026,
                    305.78543,
                    283.29886,
                    234.72049,
                    270.92914,
                    210.62497,
                ],
                [
                    276.69925,
                    260.20963,
                    248.46922,
                    209.37762,
                    281.7514,
                    173.16145,
                    161.8793,
                ],
                [
                    254.5296,
                    238.58257,
                    261.37173,
                    277.44806,
                    217.54973,
                    211.76616,
                    162.33136,
                ],
                [
                    176.24583,
                    155.69785,
                    164.8903,
                    158.71767,
                    170.58754,
                    174.22809,
                    148.65094,
                ],
                [
                    168.88553,
                    162.3888,
                    143.45665,
                    144.12737,
                    174.56282,
                    173.04388,
                    137.46349,
                ],
                [
                    167.90765,
                    154.07794,
                    145.49677,
                    132.35924,
                    131.83968,
                    147.98386,
                    153.403,
                ],
            ],
        ],
        dtype=np.float32,
    )

    expected_Tair = np.array(
        [
            [
                [
                    8.382342,
                    6.532331,
                    4.761866,
                    3.7316587,
                    3.4166677,
                    3.1780987,
                    2.1949596,
                ],
                [
                    4.619903,
                    3.0633764,
                    3.041514,
                    2.3080995,
                    1.5948807,
                    -1.6109427,
                    -4.4116983,
                ],
                [
                    -1.5634246,
                    -2.5504875,
                    -1.4600551,
                    -0.5690303,
                    -2.3495994,
                    -15.370553,
                    -19.28607,
                ],
                [
                    -9.177512,
                    -3.4015727,
                    -4.4484963,
                    -3.7056556,
                    -10.697647,
                    -18.948631,
                    -23.336071,
                ],
                [
                    -15.661556,
                    -23.100325,
                    -21.965107,
                    -29.660362,
                    -26.471872,
                    -26.348959,
                    -28.95834,
                ],
                [
                    -24.605135,
                    -26.762844,
                    -29.470911,
                    -30.107069,
                    -25.887157,
                    -26.812098,
                    -32.926876,
                ],
                [
                    -26.673147,
                    -29.413385,
                    -30.100338,
                    -29.684294,
                    -30.642694,
                    -29.153072,
                    -27.390156,
                ],
            ],
            [
                [
                    8.394094,
                    6.5803084,
                    4.8808966,
                    3.784057,
                    3.468052,
                    3.2407324,
                    2.3333037,
                ],
                [
                    4.6263127,
                    3.4498365,
                    3.032674,
                    2.4419715,
                    1.5665886,
                    -1.6296549,
                    -4.232036,
                ],
                [
                    -1.5784973,
                    -2.401672,
                    -1.3959641,
                    -0.6793834,
                    -1.9613675,
                    -15.220611,
                    -19.080027,
                ],
                [
                    -8.883304,
                    -3.3834128,
                    -4.6765513,
                    -3.9015381,
                    -10.273033,
                    -18.5212,
                    -23.289701,
                ],
                [
                    -14.871544,
                    -22.748655,
                    -21.94059,
                    -29.776194,
                    -26.546505,
                    -26.173407,
                    -29.121338,
                ],
                [
                    -24.285597,
                    -26.681915,
                    -29.44489,
                    -30.269762,
                    -26.018312,
                    -26.81835,
                    -32.84492,
                ],
                [
                    -26.240913,
                    -29.185444,
                    -30.05311,
                    -29.673147,
                    -30.659044,
                    -29.123781,
                    -27.695673,
                ],
            ],
        ],
        dtype=np.float32,
    )

    expected_qair = np.array(
        [
            [
                [
                    0.00608214,
                    0.00485943,
                    0.00434451,
                    0.00392352,
                    0.00350673,
                    0.00393602,
                    0.00386828,
                ],
                [
                    0.00496389,
                    0.0044483,
                    0.00362612,
                    0.00270351,
                    0.00263139,
                    0.00267393,
                    0.00188468,
                ],
                [
                    0.00218881,
                    0.00164652,
                    0.00180753,
                    0.00201019,
                    0.00222358,
                    0.00087732,
                    0.00062476,
                ],
                [
                    0.00138125,
                    0.00205246,
                    0.00235401,
                    0.00265417,
                    0.00145432,
                    0.00066223,
                    0.00043854,
                ],
                [
                    0.00082216,
                    0.00046076,
                    0.00046506,
                    0.00023305,
                    0.00031997,
                    0.00033908,
                    0.00025723,
                ],
                [
                    0.0003852,
                    0.00031066,
                    0.00024204,
                    0.00022732,
                    0.00032728,
                    0.00030786,
                    0.00018567,
                ],
                [
                    0.00032216,
                    0.00024468,
                    0.00023315,
                    0.00026326,
                    0.00022695,
                    0.00025577,
                    0.00029945,
                ],
            ],
            [
                [
                    0.00611161,
                    0.00487512,
                    0.00432123,
                    0.00410325,
                    0.00363046,
                    0.00388684,
                    0.00398353,
                ],
                [
                    0.004997,
                    0.00455964,
                    0.00391408,
                    0.00286727,
                    0.00263738,
                    0.00266042,
                    0.00188382,
                ],
                [
                    0.00221575,
                    0.00166183,
                    0.00184349,
                    0.00198757,
                    0.00228314,
                    0.00087687,
                    0.00063481,
                ],
                [
                    0.00141166,
                    0.0020337,
                    0.00229338,
                    0.00262552,
                    0.00149892,
                    0.00068138,
                    0.00043698,
                ],
                [
                    0.00085138,
                    0.00047381,
                    0.00046279,
                    0.00023133,
                    0.00031678,
                    0.00034515,
                    0.00025435,
                ],
                [
                    0.0003951,
                    0.0003132,
                    0.0002411,
                    0.00022282,
                    0.00031977,
                    0.00031098,
                    0.00018751,
                ],
                [
                    0.00033148,
                    0.00024932,
                    0.00023372,
                    0.00026297,
                    0.00022758,
                    0.00025663,
                    0.00029231,
                ],
            ],
        ],
        dtype=np.float32,
    )

    expected_rain = np.array(
        [
            [
                [
                    3.61238503e00,
                    3.40189010e-01,
                    5.07690720e-02,
                    4.43821624e-02,
                    2.10043773e-01,
                    4.68921065e-01,
                    1.17921913e00,
                ],
                [
                    3.48412228e00,
                    3.86755776e00,
                    2.48581603e-01,
                    3.64060067e-02,
                    5.49296811e-02,
                    2.96591282e-01,
                    5.34151215e-03,
                ],
                [
                    9.73676443e-02,
                    1.11962268e-02,
                    9.25773978e-02,
                    1.36675648e-04,
                    1.21756345e-01,
                    0.00000000e00,
                    1.35925822e-02,
                ],
                [
                    7.06618875e-02,
                    3.13506752e-01,
                    4.56249267e-01,
                    1.30473804e00,
                    2.46778969e-02,
                    1.31649813e-02,
                    3.05349231e-02,
                ],
                [
                    0.00000000e00,
                    7.10712187e-03,
                    1.27204210e-02,
                    1.14610912e-02,
                    5.81587963e-02,
                    1.63536705e-02,
                    1.79725345e-02,
                ],
                [
                    2.68480852e-02,
                    2.18332373e-02,
                    1.34839285e-02,
                    8.68453179e-03,
                    2.92103793e-02,
                    2.44176220e-02,
                    4.91440296e-03,
                ],
                [
                    2.63865143e-02,
                    2.48033050e-02,
                    1.22478902e-02,
                    3.26886214e-03,
                    0.00000000e00,
                    7.88422953e-03,
                    3.26447487e-02,
                ],
            ],
            [
                [
                    5.15011311e00,
                    1.07888615e00,
                    4.82074311e-03,
                    6.01990409e-02,
                    1.50670428e-02,
                    4.35768366e-01,
                    1.39202833e00,
                ],
                [
                    3.28930092e00,
                    3.81418300e00,
                    7.94297993e-01,
                    1.15379691e-02,
                    7.38679767e-02,
                    3.54040235e-01,
                    2.69045797e-03,
                ],
                [
                    1.16046831e-01,
                    5.51193906e-03,
                    9.27464366e-02,
                    2.15600841e-02,
                    1.63438022e-01,
                    0.00000000e00,
                    1.25724524e-02,
                ],
                [
                    6.99779242e-02,
                    3.13537568e-01,
                    3.66352916e-01,
                    1.00021172e00,
                    2.27534100e-02,
                    2.78335381e-02,
                    3.07151340e-02,
                ],
                [
                    0.00000000e00,
                    6.13165740e-03,
                    1.10241473e-02,
                    1.25272367e-02,
                    4.68330160e-02,
                    2.40270998e-02,
                    1.89573225e-02,
                ],
                [
                    2.87261512e-02,
                    2.31106617e-02,
                    1.35146212e-02,
                    7.67777674e-03,
                    3.07483859e-02,
                    1.98638346e-02,
                    6.78182067e-03,
                ],
                [
                    2.84809899e-02,
                    2.73145214e-02,
                    1.30094122e-02,
                    3.26886214e-03,
                    0.00000000e00,
                    8.95222276e-03,
                    2.09490024e-02,
                ],
            ],
        ],
        dtype=np.float32,
    )

    # Check the values in the dataset
    assert np.allclose(atm_forcing.ds["uwnd"].values, expected_uwnd)
    assert np.allclose(atm_forcing.ds["vwnd"].values, expected_vwnd)
    assert np.allclose(atm_forcing.ds["swrad"].values, expected_swrad)
    assert np.allclose(atm_forcing.ds["lwrad"].values, expected_lwrad)
    assert np.allclose(atm_forcing.ds["Tair"].values, expected_Tair)
    assert np.allclose(atm_forcing.ds["qair"].values, expected_qair)
    assert np.allclose(atm_forcing.ds["rain"].values, expected_rain)

    atm_forcing.plot(varname="uwnd", time=0)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    atm_forcing.save(filepath)
    extended_filepath = filepath + ".20200201-01.nc"

    try:
        assert os.path.exists(extended_filepath)
    finally:
        os.remove(extended_filepath)


@pytest.mark.parametrize(
    "atm_forcing_fixture",
    [
        "atmospheric_forcing",
        "corrected_atmospheric_forcing",
    ],
)
def test_roundtrip_yaml(atm_forcing_fixture, request):
    """Test that creating an AtmosphericForcing object, saving its parameters to yaml file, and re-opening yaml file creates the same object."""

    atm_forcing = request.getfixturevalue(atm_forcing_fixture)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    try:
        atm_forcing.to_yaml(filepath)

        atm_forcing_from_file = AtmosphericForcing.from_yaml(filepath)

        assert atm_forcing == atm_forcing_from_file

    finally:
        os.remove(filepath)


def test_from_yaml_missing_atmospheric_forcing():
    yaml_content = textwrap.dedent(
        """\
    ---
    roms_tools_version: 0.0.0
    ---
    Grid:
      nx: 100
      ny: 100
      size_x: 1800
      size_y: 2400
      center_lon: -10
      center_lat: 61
      rot: -20
      topography_source: ETOPO5
      smooth_factor: 8
      hmin: 5.0
      rmax: 0.2
    """
    )

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        yaml_filepath = tmp_file.name
        tmp_file.write(yaml_content.encode())

    try:
        with pytest.raises(
            ValueError,
            match="No AtmosphericForcing configuration found in the YAML file.",
        ):
            AtmosphericForcing.from_yaml(yaml_filepath)
    finally:
        os.remove(yaml_filepath)


# SWRCorrection unit checks


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


def test_from_yaml_missing_swr_correction():
    yaml_content = textwrap.dedent(
        """\
    ---
    roms_tools_version: 0.0.0
    ---
    Grid:
      nx: 100
      ny: 100
      size_x: 1800
      size_y: 2400
      center_lon: -10
      center_lat: 61
      rot: -20
      topography_source: ETOPO5
      smooth_factor: 8
      hmin: 5.0
      rmax: 0.2
    """
    )

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        yaml_filepath = tmp_file.name
        tmp_file.write(yaml_content.encode())

    try:
        with pytest.raises(
            ValueError, match="No SWRCorrection configuration found in the YAML file."
        ):
            SWRCorrection.from_yaml(yaml_filepath)
    finally:
        os.remove(yaml_filepath)
