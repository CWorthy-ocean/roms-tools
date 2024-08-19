import pytest
from datetime import datetime
from roms_tools import Grid, SurfaceForcing
from roms_tools.setup.download import download_test_data
import tempfile
import os
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
    """
    Test successful initialization of SurfaceForcing with regional data.

    Verifies that attributes are correctly set and data contains expected variables.
    """
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    sfc_forcing = SurfaceForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        physics_source={"name": "ERA5", "path": fname},
    )

    assert sfc_forcing.ds is not None

    grid.coarsen()

    sfc_forcing = SurfaceForcing(
        grid=grid,
        use_coarse_grid=True,
        start_time=start_time,
        end_time=end_time,
        physics_source={"name": "ERA5", "path": fname},
    )

    assert "uwnd" in sfc_forcing.ds["physics"]
    assert "vwnd" in sfc_forcing.ds["physics"]
    assert "swrad" in sfc_forcing.ds["physics"]
    assert "lwrad" in sfc_forcing.ds["physics"]
    assert "Tair" in sfc_forcing.ds["physics"]
    assert "qair" in sfc_forcing.ds["physics"]
    assert "rain" in sfc_forcing.ds["physics"]

    assert sfc_forcing.start_time == start_time
    assert sfc_forcing.end_time == end_time
    assert sfc_forcing.physics_source == {
        "name": "ERA5",
        "path": fname,
        "climatology": False,
    }


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
        "another_grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
    ],
)
def test_nan_detection_initialization_with_regional_data(grid_fixture, request):
    """
    Test handling of NaN values during initialization with regional data.

    Ensures ValueError is raised if NaN values are detected in the dataset.
    """
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    with pytest.raises(ValueError, match="NaN values found"):

        SurfaceForcing(
            grid=grid,
            start_time=start_time,
            end_time=end_time,
            physics_source={"name": "ERA5", "path": fname},
        )

    grid.coarsen()

    with pytest.raises(ValueError, match="NaN values found"):
        SurfaceForcing(
            grid=grid,
            use_coarse_grid=True,
            start_time=start_time,
            end_time=end_time,
            physics_source={"name": "ERA5", "path": fname},
        )


def test_no_longitude_intersection_initialization_with_regional_data(
    grid_that_straddles_180_degree_meridian,
):
    """
    Test initialization of SurfaceForcing with a grid that straddles the 180° meridian.

    Ensures ValueError is raised when the longitude range does not intersect with the dataset.
    """
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")

    with pytest.raises(
        ValueError, match="Selected longitude range does not intersect with dataset"
    ):

        SurfaceForcing(
            grid=grid_that_straddles_180_degree_meridian,
            start_time=start_time,
            end_time=end_time,
            physics_source={"name": "ERA5", "path": fname},
        )

    grid_that_straddles_180_degree_meridian.coarsen()

    with pytest.raises(
        ValueError, match="Selected longitude range does not intersect with dataset"
    ):
        SurfaceForcing(
            grid=grid_that_straddles_180_degree_meridian,
            use_coarse_grid=True,
            start_time=start_time,
            end_time=end_time,
            physics_source={"name": "ERA5", "path": fname},
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
    """
    Test initialization of SurfaceForcing with global data.

    Verifies that the SurfaceForcing object is correctly initialized with global data,
    including the correct handling of the grid and physics data. Checks both coarse and fine grid initialization.
    """
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    sfc_forcing = SurfaceForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        physics_source={"name": "ERA5", "path": fname},
    )
    assert sfc_forcing.start_time == start_time
    assert sfc_forcing.end_time == end_time
    assert sfc_forcing.physics_source == {
        "name": "ERA5",
        "path": fname,
        "climatology": False,
    }

    assert "uwnd" in sfc_forcing.ds["physics"]
    assert "vwnd" in sfc_forcing.ds["physics"]
    assert "swrad" in sfc_forcing.ds["physics"]
    assert "lwrad" in sfc_forcing.ds["physics"]
    assert "Tair" in sfc_forcing.ds["physics"]
    assert "qair" in sfc_forcing.ds["physics"]
    assert "rain" in sfc_forcing.ds["physics"]
    assert sfc_forcing.ds["physics"].attrs["physics_source"] == "ERA5"

    grid.coarsen()

    sfc_forcing = SurfaceForcing(
        grid=grid,
        use_coarse_grid=True,
        start_time=start_time,
        end_time=end_time,
        physics_source={"name": "ERA5", "path": fname},
    )
    assert sfc_forcing.start_time == start_time
    assert sfc_forcing.end_time == end_time
    assert sfc_forcing.physics_source == {
        "name": "ERA5",
        "path": fname,
        "climatology": False,
    }

    assert "uwnd" in sfc_forcing.ds["physics"]
    assert "vwnd" in sfc_forcing.ds["physics"]
    assert "swrad" in sfc_forcing.ds["physics"]
    assert "lwrad" in sfc_forcing.ds["physics"]
    assert "Tair" in sfc_forcing.ds["physics"]
    assert "qair" in sfc_forcing.ds["physics"]
    assert "rain" in sfc_forcing.ds["physics"]
    assert sfc_forcing.ds["physics"].attrs["physics_source"] == "ERA5"


@pytest.fixture
def surface_forcing(grid_that_straddles_180_degree_meridian):
    """
    Fixture for creating a SurfaceForcing object.
    """

    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")

    return SurfaceForcing(
        grid=grid_that_straddles_180_degree_meridian,
        start_time=start_time,
        end_time=end_time,
        physics_source={"name": "ERA5", "path": fname},
    )


@pytest.fixture
def corrected_surface_forcing(grid_that_straddles_180_degree_meridian):
    """
    Fixture for creating a SurfaceForcing object with shortwave radiation correction.
    """

    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")

    return SurfaceForcing(
        grid=grid_that_straddles_180_degree_meridian,
        start_time=start_time,
        end_time=end_time,
        physics_source={"name": "ERA5", "path": fname},
        correct_radiation=True,
    )


@pytest.fixture
def corrected_surface_forcing_with_bgc(grid_that_straddles_180_degree_meridian):
    """
    Fixture for creating a SurfaceForcing object with shortwave radiation correction and BGC.
    """

    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")
    fname_bgc = download_test_data("CESM_surface_global_test_data.nc")

    return SurfaceForcing(
        grid=grid_that_straddles_180_degree_meridian,
        start_time=start_time,
        end_time=end_time,
        physics_source={"name": "ERA5", "path": fname},
        bgc_source={"name": "CESM_REGRIDDED", "path": fname_bgc},
        correct_radiation=True,
    )


@pytest.fixture
def corrected_surface_forcing_with_bgc_from_climatology(
    grid_that_straddles_180_degree_meridian,
):
    """
    Fixture for creating a SurfaceForcing object with shortwave radiation correction and BGC from climatology.
    """

    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")
    fname_bgc = download_test_data("CESM_surface_global_test_data_climatology.nc")

    return SurfaceForcing(
        grid=grid_that_straddles_180_degree_meridian,
        start_time=start_time,
        end_time=end_time,
        physics_source={"name": "ERA5", "path": fname},
        bgc_source={"name": "CESM_REGRIDDED", "path": fname_bgc, "climatology": True},
        correct_radiation=True,
    )


def test_time_attr_climatology(corrected_surface_forcing_with_bgc_from_climatology):
    """
    Test that the 'cycle_length' attribute is present in the time coordinate of the BGC dataset
    when using climatology data.
    """
    assert hasattr(
        corrected_surface_forcing_with_bgc_from_climatology.ds["bgc"].time,
        "cycle_length",
    )


def test_time_attr(corrected_surface_forcing_with_bgc):
    """
    Test that the 'cycle_length' attribute is not present in the time coordinate of the BGC dataset
    when not using climatology data.
    """
    assert not hasattr(
        corrected_surface_forcing_with_bgc.ds["bgc"].time, "cycle_length"
    )


@pytest.mark.parametrize(
    "sfc_forcing_fixture",
    [
        "corrected_surface_forcing_with_bgc",
        "corrected_surface_forcing_with_bgc_from_climatology",
    ],
)
def test_surface_forcing_creation(sfc_forcing_fixture, request):
    """
    Test the creation and initialization of the SurfaceForcing object.

    Verifies that the SurfaceForcing object is properly created with correct attributes,
    including physics and BGC sources. Ensures that expected variables are present in the dataset
    and that attributes match the given configurations.
    """

    fname = download_test_data("ERA5_global_test_data.nc")

    sfc_forcing = request.getfixturevalue(sfc_forcing_fixture)

    assert sfc_forcing.start_time == datetime(2020, 1, 31)
    assert sfc_forcing.end_time == datetime(2020, 2, 2)
    assert sfc_forcing.physics_source == {
        "name": "ERA5",
        "path": fname,
        "climatology": False,
    }
    assert sfc_forcing.bgc_source["name"] == "CESM_REGRIDDED"

    assert "uwnd" in sfc_forcing.ds["physics"]
    assert "vwnd" in sfc_forcing.ds["physics"]
    assert "swrad" in sfc_forcing.ds["physics"]
    assert "lwrad" in sfc_forcing.ds["physics"]
    assert "Tair" in sfc_forcing.ds["physics"]
    assert "qair" in sfc_forcing.ds["physics"]
    assert "rain" in sfc_forcing.ds["physics"]
    assert sfc_forcing.ds["physics"].attrs["physics_source"] == "ERA5"

    assert "pco2_air" in sfc_forcing.ds["bgc"]
    assert "pco2_air_alt" in sfc_forcing.ds["bgc"]
    assert "iron" in sfc_forcing.ds["bgc"]
    assert "dust" in sfc_forcing.ds["bgc"]
    assert "nox" in sfc_forcing.ds["bgc"]
    assert "nhy" in sfc_forcing.ds["bgc"]
    assert sfc_forcing.ds["bgc"].attrs["bgc_source"] == "CESM_REGRIDDED"


@pytest.mark.parametrize(
    "sfc_forcing_fixture, expected_swrad",
    [
        (
            "surface_forcing",
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
            "corrected_surface_forcing",
            [
                np.array(
                    [
                        [
                            [
                                2.0345396e01,
                                7.1687141e01,
                                1.2855386e02,
                                1.7835985e02,
                                2.4759137e02,
                                2.3055316e02,
                                2.1866821e02,
                            ],
                            [
                                1.8974031e01,
                                2.0692104e01,
                                1.1139817e02,
                                1.4908310e02,
                                2.3330269e02,
                                1.1800186e02,
                                2.1442001e02,
                            ],
                            [
                                9.5367859e01,
                                1.1730759e02,
                                1.3710544e02,
                                2.1596344e02,
                                1.1043810e02,
                                7.6220886e01,
                                9.0121040e01,
                            ],
                            [
                                7.7221687e01,
                                1.2257743e02,
                                7.8461441e01,
                                7.3550613e01,
                                4.7777473e01,
                                3.1707375e01,
                                5.4698215e01,
                            ],
                            [
                                3.9274761e01,
                                3.0638720e01,
                                4.3393879e01,
                                2.4530497e01,
                                1.6565577e01,
                                1.3347603e01,
                                2.5458445e01,
                            ],
                            [
                                1.7271362e01,
                                1.4313513e01,
                                1.3647677e01,
                                8.3360624e00,
                                3.3492086e00,
                                2.3186626e00,
                                2.1239614e00,
                            ],
                            [
                                1.0622130e01,
                                6.2982388e00,
                                1.1909422e00,
                                0.0000000e00,
                                0.0000000e00,
                                0.0000000e00,
                                0.0000000e00,
                            ],
                        ],
                        [
                            [
                                3.2848579e01,
                                7.4629196e01,
                                1.9472108e02,
                                2.0598582e02,
                                2.3326712e02,
                                2.1832445e02,
                                1.8515254e02,
                            ],
                            [
                                2.7193939e01,
                                2.0835009e01,
                                7.1233307e01,
                                1.4508174e02,
                                2.0451334e02,
                                1.0378988e02,
                                1.9593568e02,
                            ],
                            [
                                1.0570849e02,
                                1.3862865e02,
                                1.4445511e02,
                                2.2553131e02,
                                8.7840797e01,
                                6.7152107e01,
                                6.9752350e01,
                            ],
                            [
                                8.8704170e01,
                                1.4620630e02,
                                8.6310257e01,
                                7.0209435e01,
                                4.0398537e01,
                                2.5448185e01,
                                4.5346252e01,
                            ],
                            [
                                6.2564697e01,
                                4.2063152e01,
                                5.5304630e01,
                                2.2452543e01,
                                1.8183731e01,
                                1.0362802e01,
                                1.8535559e01,
                            ],
                            [
                                3.0190443e01,
                                2.2516962e01,
                                2.0513937e01,
                                1.1818749e01,
                                3.7765646e00,
                                1.6394116e00,
                                2.3138286e-01,
                            ],
                            [
                                2.4329630e01,
                                1.2172097e01,
                                2.9701986e00,
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
    ],
)
def test_surface_forcing_data_consistency_plot_save(
    sfc_forcing_fixture, expected_swrad, request, tmp_path
):
    """
    Test that the data within the SurfaceForcing object remains consistent.
    Also test plot and save methods in the same test since we dask arrays are already computed.
    """
    sfc_forcing = request.getfixturevalue(sfc_forcing_fixture)

    sfc_forcing.ds.load()

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
                    -19.289997,
                ],
                [
                    -9.177512,
                    -3.4015727,
                    -4.4484963,
                    -3.7056556,
                    -10.697647,
                    -18.948631,
                    -23.376123,
                ],
                [
                    -15.661556,
                    -23.100355,
                    -21.994339,
                    -29.661201,
                    -26.471874,
                    -26.348959,
                    -28.97584,
                ],
                [
                    -24.909988,
                    -26.92668,
                    -29.5114,
                    -30.11024,
                    -25.887157,
                    -26.812098,
                    -32.926876,
                ],
                [
                    -26.899937,
                    -29.645899,
                    -30.121904,
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
                    -19.084148,
                ],
                [
                    -8.883304,
                    -3.3834128,
                    -4.6765513,
                    -3.9015381,
                    -10.273033,
                    -18.5212,
                    -23.331766,
                ],
                [
                    -14.871544,
                    -22.748688,
                    -21.973629,
                    -29.777143,
                    -26.546505,
                    -26.173407,
                    -29.139715,
                ],
                [
                    -24.59206,
                    -26.851955,
                    -29.489199,
                    -30.27334,
                    -26.018312,
                    -26.81835,
                    -32.84492,
                ],
                [
                    -26.451527,
                    -29.415953,
                    -30.07525,
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
                    0.0060822,
                    0.0048595,
                    0.00434452,
                    0.00392386,
                    0.00350679,
                    0.00393684,
                    0.00386832,
                ],
                [
                    0.00496396,
                    0.00444844,
                    0.00362669,
                    0.0027039,
                    0.00263158,
                    0.00268065,
                    0.00188471,
                ],
                [
                    0.00218931,
                    0.00164752,
                    0.00180771,
                    0.00201025,
                    0.00222373,
                    0.00087739,
                    0.0006346,
                ],
                [
                    0.0014008,
                    0.00205272,
                    0.00235442,
                    0.00265417,
                    0.00145444,
                    0.00066224,
                    0.00048542,
                ],
                [
                    0.00082356,
                    0.0004623,
                    0.00057463,
                    0.00025071,
                    0.00032594,
                    0.00033909,
                    0.00028633,
                ],
                [
                    0.00041308,
                    0.00034891,
                    0.00027696,
                    0.00024237,
                    0.00032729,
                    0.00030786,
                    0.00018568,
                ],
                [
                    0.00034559,
                    0.00027157,
                    0.00024169,
                    0.00026326,
                    0.00022696,
                    0.00025577,
                    0.00029947,
                ],
            ],
            [
                [
                    0.0061117,
                    0.00487523,
                    0.00432124,
                    0.00410334,
                    0.00363048,
                    0.00388888,
                    0.00398394,
                ],
                [
                    0.00499707,
                    0.0045597,
                    0.00391427,
                    0.00286811,
                    0.00263745,
                    0.00266608,
                    0.00188403,
                ],
                [
                    0.00221635,
                    0.00166319,
                    0.00184379,
                    0.00198759,
                    0.00228325,
                    0.00087703,
                    0.00064521,
                ],
                [
                    0.00143134,
                    0.00203374,
                    0.00229373,
                    0.00262552,
                    0.00149901,
                    0.00068139,
                    0.0004829,
                ],
                [
                    0.00085248,
                    0.00047552,
                    0.00056961,
                    0.00024881,
                    0.00032247,
                    0.00034516,
                    0.00028347,
                ],
                [
                    0.00042441,
                    0.00035251,
                    0.00027714,
                    0.00023793,
                    0.00031979,
                    0.00031098,
                    0.00018754,
                ],
                [
                    0.00035735,
                    0.00027766,
                    0.00024232,
                    0.00026297,
                    0.00022758,
                    0.00025663,
                    0.00029233,
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
    ds = sfc_forcing.ds["physics"]

    assert np.allclose(ds["uwnd"].values, expected_uwnd)
    assert np.allclose(ds["vwnd"].values, expected_vwnd)
    assert np.allclose(ds["swrad"].values, expected_swrad)
    assert np.allclose(ds["lwrad"].values, expected_lwrad)
    assert np.allclose(ds["Tair"].values, expected_Tair)
    assert np.allclose(ds["qair"].values, expected_qair)
    assert np.allclose(ds["rain"].values, expected_rain)

    sfc_forcing.plot(varname="uwnd", time=0)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        filepath = tmpfile.name

    sfc_forcing.save(filepath)
    extended_filepath = filepath + "_physics_20200201-01.nc"

    try:
        assert os.path.exists(extended_filepath)
    finally:
        os.remove(extended_filepath)


def test_surface_forcing_bgc_data_consistency_plot_save(
    corrected_surface_forcing_with_bgc,
):
    """
    Test that the BGC data within the SurfaceForcing object remains consistent.
    Also test plot and save methods in the same test since we dask arrays are already computed.
    """

    # Check the values in the dataset
    corrected_surface_forcing_with_bgc.plot(varname="pco2_air", time=0)

    expected_pco2_air = np.array(
        [
            [
                [
                    404.22748,
                    413.57806,
                    415.53137,
                    412.00464,
                    408.90378,
                    403.009,
                    399.49335,
                ],
                [
                    418.73532,
                    426.9579,
                    437.74713,
                    441.56055,
                    442.04376,
                    388.9692,
                    388.6991,
                ],
                [
                    428.60126,
                    426.8612,
                    432.61078,
                    436.4323,
                    403.53485,
                    331.7332,
                    343.11868,
                ],
                [
                    401.68954,
                    425.14883,
                    436.7216,
                    455.18954,
                    357.83847,
                    316.44016,
                    354.0953,
                ],
                [
                    430.29868,
                    398.86063,
                    400.73868,
                    399.2477,
                    357.3982,
                    340.97977,
                    354.399,
                ],
                [
                    425.6459,
                    403.8653,
                    375.61847,
                    368.8612,
                    353.72507,
                    340.38684,
                    352.58127,
                ],
                [
                    417.63498,
                    414.74066,
                    393.7536,
                    361.35803,
                    357.5395,
                    353.18665,
                    361.26233,
                ],
            ]
        ],
        dtype=np.float32,
    )

    assert np.allclose(
        corrected_surface_forcing_with_bgc.ds["bgc"]["pco2_air"].values,
        expected_pco2_air,
    )

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        filepath = tmpfile.name

    corrected_surface_forcing_with_bgc.save(filepath)
    physics_filepath = filepath + "_physics_20200201-01.nc"
    bgc_filepath = filepath + "_bgc_20200201-01.nc"

    try:
        assert os.path.exists(physics_filepath)
        assert os.path.exists(bgc_filepath)
    finally:
        os.remove(physics_filepath)
        os.remove(bgc_filepath)


def test_surface_forcing_bgc_data_from_clim_consistency_plot_save(
    corrected_surface_forcing_with_bgc_from_climatology,
):
    """
    Test that the BGC data within the SurfaceForcing object remains consistent.
    Also test plot and save methods in the same test since we dask arrays are already computed.
    """

    # Check the values in the dataset
    corrected_surface_forcing_with_bgc_from_climatology.plot(varname="pco2_air", time=0)

    expected_pco2_air = np.array(
        [
            [
                [
                    398.14603,
                    404.46167,
                    407.15097,
                    405.2776,
                    404.13507,
                    400.7547,
                    398.85083,
                ],
                [
                    409.1616,
                    415.79483,
                    425.67477,
                    427.0116,
                    430.06903,
                    395.88733,
                    390.09415,
                ],
                [
                    417.66122,
                    414.91043,
                    417.62463,
                    422.71118,
                    412.35883,
                    368.0207,
                    366.4949,
                ],
                [
                    387.8851,
                    412.7159,
                    418.03122,
                    449.04837,
                    382.74442,
                    324.06628,
                    360.40964,
                ],
                [
                    416.6366,
                    390.45486,
                    390.24994,
                    392.66037,
                    374.50775,
                    342.20087,
                    352.27393,
                ],
                [
                    411.73032,
                    394.3381,
                    370.42654,
                    366.05936,
                    353.22403,
                    343.92136,
                    341.2689,
                ],
                [
                    407.44583,
                    415.57016,
                    384.86288,
                    360.26337,
                    356.596,
                    352.26584,
                    357.4343,
                ],
            ],
            [
                [
                    404.22748,
                    413.57806,
                    415.53137,
                    412.00464,
                    408.90378,
                    403.009,
                    399.49335,
                ],
                [
                    418.73532,
                    426.9579,
                    437.74713,
                    441.56055,
                    442.04376,
                    388.9692,
                    388.6991,
                ],
                [
                    428.60126,
                    426.8612,
                    432.61078,
                    436.4323,
                    403.53485,
                    331.7332,
                    343.11868,
                ],
                [
                    401.68954,
                    425.14883,
                    436.7216,
                    455.18954,
                    357.83847,
                    316.44016,
                    354.0953,
                ],
                [
                    430.29868,
                    398.86063,
                    400.73868,
                    399.2477,
                    357.3982,
                    340.97977,
                    354.399,
                ],
                [
                    425.6459,
                    403.8653,
                    375.61847,
                    368.8612,
                    353.72507,
                    340.38684,
                    352.58127,
                ],
                [
                    417.63498,
                    414.74066,
                    393.7536,
                    361.35803,
                    357.5395,
                    353.18665,
                    361.26233,
                ],
            ],
            [
                [
                    410.8616,
                    423.02103,
                    423.09604,
                    419.10156,
                    414.9103,
                    407.07053,
                    401.62665,
                ],
                [
                    429.6854,
                    434.3063,
                    439.06766,
                    443.2926,
                    444.38937,
                    392.52994,
                    387.95245,
                ],
                [
                    438.13297,
                    440.14212,
                    446.70282,
                    445.9411,
                    395.38516,
                    326.51703,
                    338.68753,
                ],
                [
                    415.84042,
                    440.29053,
                    457.30914,
                    440.55865,
                    349.61758,
                    325.13593,
                    353.53876,
                ],
                [430.2406, 410.839, 411.3754, 402.27014, 358.84747, 346.353, 354.80054],
                [
                    435.1648,
                    413.51578,
                    385.1607,
                    372.467,
                    354.8004,
                    343.80344,
                    351.89355,
                ],
                [
                    429.8856,
                    423.3144,
                    405.6345,
                    363.35962,
                    357.49466,
                    354.8506,
                    361.87515,
                ],
            ],
            [
                [
                    404.8583,
                    415.5544,
                    416.2556,
                    410.7852,
                    405.22998,
                    398.49936,
                    395.41647,
                ],
                [
                    430.01413,
                    432.6222,
                    436.35675,
                    440.0094,
                    442.04233,
                    393.30008,
                    387.82687,
                ],
                [
                    439.19922,
                    439.70526,
                    444.8726,
                    444.12234,
                    384.99542,
                    326.7951,
                    337.15298,
                ],
                [
                    423.3211,
                    442.34293,
                    452.6943,
                    422.3202,
                    350.5441,
                    333.0299,
                    353.19067,
                ],
                [
                    435.60132,
                    416.37122,
                    412.31454,
                    396.5343,
                    360.8705,
                    351.0615,
                    354.82278,
                ],
                [
                    442.0921,
                    421.49625,
                    389.07895,
                    372.72894,
                    356.47137,
                    347.64758,
                    349.50967,
                ],
                [
                    423.49146,
                    437.96732,
                    416.40515,
                    364.74283,
                    357.54456,
                    357.6978,
                    363.92026,
                ],
            ],
            [
                [
                    364.66605,
                    348.29202,
                    327.21948,
                    312.5929,
                    310.1843,
                    311.27948,
                    319.08026,
                ],
                [
                    405.23355,
                    383.24057,
                    369.13156,
                    379.3948,
                    376.67355,
                    333.73822,
                    331.91266,
                ],
                [
                    405.19348,
                    396.86462,
                    388.22614,
                    390.81296,
                    335.50076,
                    305.39764,
                    324.62347,
                ],
                [
                    394.32132,
                    420.4284,
                    434.33023,
                    402.13177,
                    322.8607,
                    324.56433,
                    335.1503,
                ],
                [438.04025, 402.429, 398.4195, 385.3214, 360.3192, 352.68503, 345.7393],
                [
                    434.2024,
                    417.87787,
                    384.5583,
                    369.64655,
                    356.73856,
                    350.8263,
                    351.05643,
                ],
                [
                    400.25546,
                    432.73685,
                    409.8863,
                    364.9216,
                    359.2582,
                    361.94647,
                    366.58585,
                ],
            ],
            [
                [
                    320.0029,
                    330.2735,
                    303.61395,
                    307.58713,
                    320.75143,
                    331.16663,
                    335.8144,
                ],
                [353.0393, 332.46973, 303.4242, 296.3255, 298.1235, 326.996, 332.63177],
                [333.67642, 326.4859, 311.252, 282.5213, 294.0293, 341.41116, 356.1976],
                [
                    279.0954,
                    313.43964,
                    318.8591,
                    284.96204,
                    266.36942,
                    287.91168,
                    343.8091,
                ],
                [
                    336.50403,
                    312.51163,
                    317.2202,
                    300.65918,
                    293.41602,
                    290.27216,
                    345.08566,
                ],
                [
                    332.25598,
                    352.03445,
                    350.67816,
                    345.28265,
                    347.45816,
                    351.7342,
                    352.62457,
                ],
                [
                    337.58936,
                    362.37796,
                    386.74255,
                    366.58105,
                    361.49588,
                    365.00146,
                    368.1141,
                ],
            ],
            [
                [
                    305.92316,
                    306.93582,
                    320.42093,
                    334.3491,
                    350.7195,
                    373.15265,
                    368.141,
                ],
                [
                    293.1966,
                    303.76968,
                    312.18463,
                    316.862,
                    324.14975,
                    362.1812,
                    373.89975,
                ],
                [
                    287.50125,
                    288.732,
                    292.04642,
                    304.70465,
                    342.6752,
                    399.62433,
                    398.39957,
                ],
                [
                    285.4672,
                    281.2273,
                    281.9908,
                    287.84683,
                    327.07794,
                    345.76187,
                    364.13577,
                ],
                [
                    317.54077,
                    318.2395,
                    314.44913,
                    296.35068,
                    285.5732,
                    311.2991,
                    344.32465,
                ],
                [
                    340.03323,
                    356.31216,
                    360.23056,
                    346.04034,
                    301.41037,
                    286.5328,
                    332.2325,
                ],
                [
                    367.48395,
                    388.24136,
                    413.18228,
                    357.0047,
                    337.4069,
                    344.10382,
                    354.71527,
                ],
            ],
            [
                [
                    350.15866,
                    350.08246,
                    359.35062,
                    374.62405,
                    385.71454,
                    393.22922,
                    405.19373,
                ],
                [
                    346.96497,
                    341.0631,
                    348.13092,
                    356.104,
                    362.08224,
                    402.34842,
                    413.2263,
                ],
                [
                    335.66232,
                    335.02692,
                    333.9118,
                    345.72632,
                    377.76294,
                    438.64017,
                    440.5227,
                ],
                [
                    336.00873,
                    331.4235,
                    330.77084,
                    324.82394,
                    364.6463,
                    371.63107,
                    393.25018,
                ],
                [
                    376.79837,
                    364.86725,
                    352.64667,
                    332.04468,
                    330.4255,
                    342.30908,
                    369.07553,
                ],
                [
                    385.23654,
                    392.9487,
                    391.2472,
                    365.89276,
                    319.08,
                    340.74893,
                    358.08716,
                ],
                [
                    402.53976,
                    408.4651,
                    418.34943,
                    391.3722,
                    364.6873,
                    334.38284,
                    338.2357,
                ],
            ],
            [
                [
                    384.82806,
                    385.0281,
                    387.55463,
                    393.90097,
                    400.93903,
                    410.05496,
                    414.2652,
                ],
                [
                    373.90726,
                    375.20236,
                    376.56927,
                    380.03513,
                    377.76764,
                    408.7514,
                    421.60242,
                ],
                [
                    368.0287,
                    365.7465,
                    365.68936,
                    370.90225,
                    395.10336,
                    452.66885,
                    456.04254,
                ],
                [
                    359.98935,
                    355.42517,
                    356.04587,
                    353.28638,
                    387.7216,
                    389.9909,
                    412.54553,
                ],
                [
                    401.3988,
                    385.66016,
                    374.76376,
                    361.84274,
                    364.27496,
                    369.83444,
                    391.7403,
                ],
                [
                    406.04547,
                    410.4884,
                    404.08243,
                    390.9909,
                    364.52176,
                    372.848,
                    380.68518,
                ],
                [
                    421.35165,
                    423.7671,
                    430.52847,
                    414.28665,
                    377.83176,
                    361.36493,
                    363.71622,
                ],
            ],
            [
                [
                    384.3036,
                    386.03488,
                    390.42053,
                    393.7354,
                    398.90906,
                    407.59833,
                    414.67212,
                ],
                [
                    383.4824,
                    376.4454,
                    365.8679,
                    370.3737,
                    384.7454,
                    396.86765,
                    401.57266,
                ],
                [
                    384.68747,
                    384.93744,
                    376.9039,
                    374.37,
                    387.89017,
                    425.60678,
                    423.83215,
                ],
                [
                    349.58817,
                    353.14197,
                    354.39044,
                    362.20624,
                    376.7501,
                    388.17532,
                    397.94363,
                ],
                [
                    375.7726,
                    367.9115,
                    364.30243,
                    357.57803,
                    359.09372,
                    378.4864,
                    384.53183,
                ],
                [
                    392.0134,
                    393.79785,
                    389.5198,
                    381.37845,
                    367.65408,
                    371.40897,
                    376.8739,
                ],
                [
                    411.46152,
                    413.26355,
                    417.53354,
                    400.22797,
                    378.11246,
                    364.16946,
                    366.87015,
                ],
            ],
            [
                [
                    388.30124,
                    388.92444,
                    384.93433,
                    385.27805,
                    388.80548,
                    399.42645,
                    405.78516,
                ],
                [
                    379.37393,
                    383.77905,
                    381.04263,
                    387.48773,
                    383.45462,
                    388.63864,
                    392.0524,
                ],
                [
                    382.59354,
                    389.78592,
                    383.2384,
                    384.32312,
                    393.3613,
                    401.43216,
                    400.07477,
                ],
                [
                    347.17233,
                    368.4824,
                    367.16025,
                    390.05325,
                    381.87103,
                    375.94528,
                    384.04514,
                ],
                [
                    363.02823,
                    362.139,
                    365.81894,
                    370.0153,
                    376.21783,
                    376.71158,
                    375.15192,
                ],
                [
                    373.53513,
                    377.82776,
                    376.80414,
                    377.8196,
                    376.17963,
                    375.08887,
                    370.21902,
                ],
                [
                    384.6436,
                    392.61368,
                    397.12335,
                    392.19543,
                    374.7854,
                    366.11572,
                    364.30585,
                ],
            ],
            [
                [
                    397.632,
                    401.96716,
                    398.11554,
                    400.73257,
                    400.37378,
                    401.6573,
                    404.91113,
                ],
                [
                    406.49796,
                    414.25653,
                    415.96265,
                    422.45657,
                    425.28955,
                    399.92108,
                    391.84106,
                ],
                [
                    400.9218,
                    412.2611,
                    414.38895,
                    411.9613,
                    410.46597,
                    382.8714,
                    379.89563,
                ],
                [
                    367.06268,
                    389.9282,
                    390.5468,
                    416.58414,
                    404.75714,
                    356.0946,
                    370.88605,
                ],
                [
                    379.26697,
                    372.37427,
                    374.7852,
                    381.90277,
                    392.6096,
                    369.72604,
                    364.15964,
                ],
                [
                    385.33365,
                    380.70264,
                    369.23062,
                    368.49274,
                    371.57126,
                    372.64706,
                    358.38586,
                ],
                [
                    391.62207,
                    403.18228,
                    385.95084,
                    370.7342,
                    364.23724,
                    361.9076,
                    362.69427,
                ],
            ],
        ],
        dtype=np.float32,
    )

    assert np.allclose(
        corrected_surface_forcing_with_bgc_from_climatology.ds["bgc"][
            "pco2_air"
        ].values,
        expected_pco2_air,
    )

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        filepath = tmpfile.name

    corrected_surface_forcing_with_bgc_from_climatology.save(filepath)
    physics_filepath = filepath + "_physics_20200201-01.nc"
    bgc_filepath = filepath + "_bgc_clim.nc"

    try:
        assert os.path.exists(physics_filepath)
        assert os.path.exists(bgc_filepath)
    finally:
        os.remove(physics_filepath)
        os.remove(bgc_filepath)


@pytest.mark.parametrize(
    "sfc_forcing_fixture",
    [
        "surface_forcing",
        "corrected_surface_forcing",
        "corrected_surface_forcing_with_bgc",
        "corrected_surface_forcing_with_bgc_from_climatology",
    ],
)
def test_roundtrip_yaml(sfc_forcing_fixture, request):
    """Test that creating an SurfaceForcing object, saving its parameters to yaml file, and re-opening yaml file creates the same object."""

    sfc_forcing = request.getfixturevalue(sfc_forcing_fixture)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    try:
        sfc_forcing.to_yaml(filepath)

        sfc_forcing_from_file = SurfaceForcing.from_yaml(filepath)

        assert sfc_forcing == sfc_forcing_from_file

    finally:
        os.remove(filepath)


def test_from_yaml_missing_surface_forcing():
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
            match="No SurfaceForcing configuration found in the YAML file.",
        ):
            SurfaceForcing.from_yaml(yaml_filepath)
    finally:
        os.remove(yaml_filepath)
