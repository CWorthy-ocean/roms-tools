import pytest
from datetime import datetime
from roms_tools import BoundaryForcing, Grid
import numpy as np
import tempfile
import os
import textwrap
from roms_tools.setup.download import download_test_data


@pytest.fixture
def example_grid():
    """
    Fixture for creating a Grid object.
    """
    grid = Grid(
        nx=2,
        ny=2,
        size_x=500,
        size_y=1000,
        center_lon=0,
        center_lat=55,
        rot=10,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    return grid


@pytest.fixture
def boundary_forcing(example_grid):
    """
    Fixture for creating a BoundaryForcing object.
    """

    fname = download_test_data("GLORYS_test_data.nc")

    return BoundaryForcing(
        grid=example_grid,
        start_time=datetime(2021, 6, 29),
        end_time=datetime(2021, 6, 30),
        source={"name": "GLORYS", "path": fname},
    )


@pytest.fixture
def bgc_boundary_forcing_from_climatology(example_grid):
    """
    Fixture for creating a BoundaryForcing object.
    """

    fname_bgc = download_test_data("CESM_regional_test_data_climatology.nc")

    return BoundaryForcing(
        grid=example_grid,
        start_time=datetime(2021, 6, 29),
        end_time=datetime(2021, 6, 30),
        source={"path": fname_bgc, "name": "CESM_REGRIDDED", "climatology": True},
        type="bgc",
    )


def test_boundary_forcing_creation(boundary_forcing):
    """
    Test the creation of the BoundaryForcing object.
    """

    fname = download_test_data("GLORYS_test_data.nc")

    assert boundary_forcing.start_time == datetime(2021, 6, 29)
    assert boundary_forcing.end_time == datetime(2021, 6, 30)
    assert boundary_forcing.source == {
        "name": "GLORYS",
        "path": fname,
        "climatology": False,
    }
    assert boundary_forcing.model_reference_date == datetime(2000, 1, 1)
    assert boundary_forcing.boundaries == {
        "south": True,
        "east": True,
        "north": True,
        "west": True,
    }

    assert boundary_forcing.ds.source == "GLORYS"
    for direction in ["south", "east", "north", "west"]:
        assert f"temp_{direction}" in boundary_forcing.ds
        assert f"salt_{direction}" in boundary_forcing.ds
        assert f"u_{direction}" in boundary_forcing.ds
        assert f"v_{direction}" in boundary_forcing.ds
        assert f"zeta_{direction}" in boundary_forcing.ds

    assert len(boundary_forcing.ds.bry_time) == 1


def test_boundary_forcing_creation_with_bgc(bgc_boundary_forcing_from_climatology):
    """
    Test the creation of the BoundaryForcing object.
    """

    fname_bgc = download_test_data("CESM_regional_test_data_climatology.nc")

    assert bgc_boundary_forcing_from_climatology.start_time == datetime(2021, 6, 29)
    assert bgc_boundary_forcing_from_climatology.end_time == datetime(2021, 6, 30)
    assert bgc_boundary_forcing_from_climatology.source == {
        "path": fname_bgc,
        "name": "CESM_REGRIDDED",
        "climatology": True,
    }
    assert bgc_boundary_forcing_from_climatology.model_reference_date == datetime(
        2000, 1, 1
    )
    assert bgc_boundary_forcing_from_climatology.boundaries == {
        "south": True,
        "east": True,
        "north": True,
        "west": True,
    }

    assert bgc_boundary_forcing_from_climatology.ds.source == "CESM_REGRIDDED"
    for direction in ["south", "east", "north", "west"]:
        for var in ["ALK", "PO4"]:
            assert f"{var}_{direction}" in bgc_boundary_forcing_from_climatology.ds

    assert len(bgc_boundary_forcing_from_climatology.ds.bry_time) == 12


@pytest.mark.parametrize(
    "bdry_forcing_fixture, expected_coords",
    [
        (
            "boundary_forcing",
            {
                "abs_time": np.array(["2021-06-29T00:00:00"], dtype="datetime64[ns]"),
                "bry_time": np.array([678240000000000000], dtype="timedelta64[ns]"),
            },
        ),
        (
            "bgc_boundary_forcing_from_climatology",
            {
                "abs_time": np.array(
                    [
                        1296000000000000,
                        3888000000000000,
                        6393600000000000,
                        9072000000000000,
                        11664000000000000,
                        14342400000000000,
                        16934400000000000,
                        19612800000000000,
                        22291200000000000,
                        24883200000000000,
                        27561600000000000,
                        30153600000000000,
                    ],
                    dtype="timedelta64[ns]",
                ),
                "bry_time": np.array(
                    [
                        1296000000000000,
                        3888000000000000,
                        6393600000000000,
                        9072000000000000,
                        11664000000000000,
                        14342400000000000,
                        16934400000000000,
                        19612800000000000,
                        22291200000000000,
                        24883200000000000,
                        27561600000000000,
                        30153600000000000,
                    ],
                    dtype="timedelta64[ns]",
                ),
            },
        ),
    ],
)
def test_coordinates_existence_and_values(
    bdry_forcing_fixture, expected_coords, request
):
    """
    Test that the dataset contains the expected coordinates with the correct values.
    """

    bdry_forcing = request.getfixturevalue(bdry_forcing_fixture)

    # Check that the dataset contains exactly the expected coordinates and no others
    actual_coords = set(bdry_forcing.ds.coords.keys())
    expected_coords_set = set(expected_coords.keys())

    assert actual_coords == expected_coords_set, (
        f"Unexpected coordinates found. Expected only {expected_coords_set}, "
        f"but found {actual_coords}."
    )

    # Check that the coordinate values match the expected values
    np.testing.assert_array_equal(
        bdry_forcing.ds.coords["abs_time"].values,
        expected_coords["abs_time"],
    )
    np.testing.assert_allclose(
        bdry_forcing.ds.coords["bry_time"].values,
        expected_coords["bry_time"],
        rtol=1e-9,
        atol=0,
    )


def test_boundary_forcing_data_consistency_plot_save(
    boundary_forcing,
):
    """
    Test that the data within the BoundaryForcing object remains consistent.
    Also test plot and save methods in the same test since we dask arrays are already computed.
    """
    # Define the expected data
    expected_zeta_south = np.array(
        [[-0.30468762, -0.29416865, -0.30391693, -0.32985148]], dtype=np.float32
    )
    expected_zeta_east = np.array(
        [[-0.32985148, -0.36176518, -0.40663475, -0.40699923]], dtype=np.float32
    )
    expected_zeta_north = np.array(
        [[-0.5534979, -0.5270749, -0.45107934, -0.40699923]], dtype=np.float32
    )
    expected_zeta_west = np.array(
        [[-0.30468762, -0.34336275, -0.3699948, -0.5534979]], dtype=np.float32
    )

    expected_temp_south = np.array(
        [
            [
                [16.888039, 18.113976, 18.491693, 19.302889],
                [16.96757, 18.292757, 18.528805, 19.310738],
                [16.968897, 18.347046, 18.774857, 19.45075],
            ]
        ],
        dtype=np.float32,
    )

    expected_temp_east = np.array(
        [
            [
                [19.302889, 18.63616, 10.860589, 10.501563],
                [19.310738, 18.656109, 12.576546, 11.2624445],
                [19.45075, 18.65017, 13.1259985, 11.879851],
            ]
        ],
        dtype=np.float32,
    )

    expected_temp_north = np.array(
        [
            [
                [10.516198, 10.465848, 11.167091, 10.501563],
                [10.625185, 10.623089, 11.301617, 11.2624445],
                [10.70972, 10.72327, 11.414437, 11.879851],
            ]
        ],
        dtype=np.float32,
    )

    expected_temp_west = np.array(
        [
            [
                [16.888039, 13.35342, 11.584284, 10.516198],
                [16.96757, 13.915829, 11.922902, 10.625185],
                [16.968897, 14.607185, 12.252905, 10.70972],
            ]
        ],
        dtype=np.float32,
    )

    expected_u_south = np.array(
        [
            [
                [-0.02990346, -0.00527473, -0.00209207],
                [-0.03373717, -0.0064072, -0.00393128],
                [-0.0401831, -0.01946738, -0.02024308],
            ]
        ],
        dtype=np.float32,
    )

    expected_u_east = np.array(
        [
            [
                [-0.00209207, -0.08197243, 0.06334759, 0.03555659],
                [-0.00393128, -0.09550258, 0.0236859, 0.03713445],
                [-0.02024308, -0.11909306, -0.02354126, 0.02392998],
            ]
        ],
        dtype=np.float32,
    )

    expected_u_north = np.array(
        [
            [
                [0.04009043, 0.05006053, 0.03555659],
                [0.0349412, 0.04345757, 0.03713445],
                [0.03305725, 0.03291277, 0.02392998],
            ]
        ],
        dtype=np.float32,
    )

    expected_u_west = np.array(
        [
            [
                [-0.02990346, -0.00619788, 0.00090594, 0.04009043],
                [-0.03373717, -0.03140489, -0.00865677, 0.0349412],
                [-0.0401831, -0.08688492, -0.03151456, 0.03305725],
            ]
        ],
        dtype=np.float32,
    )

    expected_v_south = np.array(
        [
            [
                [0.04143963, 0.00235313, -0.04121511, -0.00271586],
                [0.05191367, 0.00191267, -0.03233896, -0.00375686],
                [0.01049447, -0.00655571, -0.040242, -0.00815171],
            ]
        ],
        dtype=np.float32,
    )

    expected_v_east = np.array(
        [
            [
                [-0.00271586, -0.03477801, -0.08123003],
                [-0.00375686, -0.02148061, -0.08685598],
                [-0.00815171, -0.06388983, -0.15538278],
            ]
        ],
        dtype=np.float32,
    )

    expected_v_north = np.array(
        [
            [
                [-0.00779556, -0.01769326, 0.0237356, -0.08123003],
                [-0.00770537, -0.0187724, 0.03198036, -0.08685598],
                [-0.02848312, -0.01897671, 0.00935264, -0.15538278],
            ]
        ],
        dtype=np.float32,
    )
    expected_v_west = np.array(
        [
            [
                [0.04143963, -0.01356939, -0.00779556],
                [0.05191367, -0.00451732, -0.00770537],
                [0.01049447, -0.04645749, -0.02848312],
            ]
        ],
        dtype=np.float32,
    )

    expected_ubar_south = np.array(
        [[-0.03445563, -0.01013886, -0.00840288]], dtype=np.float32
    )
    expected_ubar_east = np.array(
        [[-0.00840288, -0.09810777, 0.02369577, 0.03251844]], dtype=np.float32
    )
    expected_ubar_north = np.array(
        [[0.036187, 0.04258458, 0.03251844]], dtype=np.float32
    )
    expected_ubar_west = np.array(
        [[-0.03445563, -0.04031063, -0.01240468, 0.036187]], dtype=np.float32
    )

    expected_vbar_south = np.array(
        [[0.03502939, -0.00063722, -0.03797533, -0.00476785]], dtype=np.float32
    )
    expected_vbar_east = np.array(
        [[-0.00476785, -0.03939667, -0.10577434]], dtype=np.float32
    )
    expected_vbar_north = np.array(
        [[-0.01424322, -0.01845166, 0.02204982, -0.10577434]], dtype=np.float32
    )
    expected_vbar_west = np.array(
        [[0.03502939, -0.02095497, -0.01424322]], dtype=np.float32
    )

    ds = boundary_forcing.ds.compute()

    # Check the values in the dataset
    assert np.allclose(ds["zeta_south"].values, expected_zeta_south)
    assert np.allclose(ds["zeta_east"].values, expected_zeta_east)
    assert np.allclose(ds["zeta_north"].values, expected_zeta_north)
    assert np.allclose(ds["zeta_west"].values, expected_zeta_west)
    assert np.allclose(ds["temp_south"].values, expected_temp_south)
    assert np.allclose(ds["temp_east"].values, expected_temp_east)
    assert np.allclose(ds["temp_north"].values, expected_temp_north)
    assert np.allclose(ds["temp_west"].values, expected_temp_west)
    assert np.allclose(ds["u_south"].values, expected_u_south)
    assert np.allclose(ds["u_east"].values, expected_u_east)
    assert np.allclose(ds["u_north"].values, expected_u_north)
    assert np.allclose(ds["u_west"].values, expected_u_west)
    assert np.allclose(ds["v_south"].values, expected_v_south)
    assert np.allclose(ds["v_east"].values, expected_v_east)
    assert np.allclose(ds["v_north"].values, expected_v_north)
    assert np.allclose(ds["v_west"].values, expected_v_west)
    assert np.allclose(ds["ubar_south"].values, expected_ubar_south)
    assert np.allclose(ds["ubar_east"].values, expected_ubar_east)
    assert np.allclose(ds["ubar_north"].values, expected_ubar_north)
    assert np.allclose(ds["ubar_west"].values, expected_ubar_west)
    assert np.allclose(ds["vbar_south"].values, expected_vbar_south)
    assert np.allclose(ds["vbar_east"].values, expected_vbar_east)
    assert np.allclose(ds["vbar_north"].values, expected_vbar_north)
    assert np.allclose(ds["vbar_west"].values, expected_vbar_west)

    boundary_forcing.plot(varname="temp_south", layer_contours=True)
    boundary_forcing.plot(varname="temp_east", layer_contours=True)
    boundary_forcing.plot(varname="temp_north", layer_contours=True)
    boundary_forcing.plot(varname="temp_west", layer_contours=True)
    boundary_forcing.plot(varname="zeta_south")
    boundary_forcing.plot(varname="zeta_east")
    boundary_forcing.plot(varname="zeta_north")
    boundary_forcing.plot(varname="zeta_west")
    boundary_forcing.plot(varname="vbar_north")
    boundary_forcing.plot(varname="ubar_west")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        filepath = tmpfile.name

    boundary_forcing.save(filepath)
    extended_filepath = filepath + "_20210629-29.nc"

    try:
        assert os.path.exists(extended_filepath)
    finally:
        os.remove(extended_filepath)


def test_bgc_boundary_forcing_data_consistency_plot_save(
    bgc_boundary_forcing_from_climatology,
):
    """
    Test that the data within the BoundaryForcing object remains consistent.
    Also test plot and save methods in the same test since we dask arrays are already computed.
    """
    # Define the expected data
    expected_alk_south = np.array(
        [
            [
                [2352.1333, 2333.0105, 2308.4944, 2286.0708],
                [2352.1084, 2332.9297, 2308.3806, 2285.9463],
                [2352.0847, 2332.8533, 2308.2747, 2285.831],
            ],
            [
                [2354.6926, 2337.279, 2313.8926, 2293.9246],
                [2354.6624, 2337.1863, 2313.7551, 2293.7686],
                [2354.6338, 2337.0989, 2313.627, 2293.6233],
            ],
            [
                [2354.9302, 2336.1455, 2312.5203, 2291.9067],
                [2354.8633, 2336.013, 2312.3555, 2291.765],
                [2354.7998, 2335.888, 2312.2031, 2291.6353],
            ],
            [
                [2355.0718, 2333.9512, 2308.3606, 2285.6335],
                [2354.9797, 2333.7737, 2308.0688, 2285.2783],
                [2354.893, 2333.606, 2307.8062, 2284.9624],
            ],
            [
                [2353.7942, 2329.9983, 2302.2275, 2276.459],
                [2353.6982, 2329.7644, 2301.733, 2275.739],
                [2353.6072, 2329.5432, 2301.2952, 2275.1042],
            ],
            [
                [2345.8477, 2320.4146, 2291.9263, 2265.4736],
                [2345.623, 2319.9868, 2290.925, 2263.9817],
                [2345.4106, 2319.5825, 2290.079, 2262.7253],
            ],
            [
                [2335.3225, 2310.841, 2279.7, 2251.497],
                [2335.0918, 2310.3765, 2278.388, 2249.3713],
                [2334.8738, 2309.9373, 2277.298, 2247.5896],
            ],
            [
                [2333.2937, 2309.6064, 2276.4534, 2246.6724],
                [2333.1597, 2309.268, 2275.4666, 2245.062],
                [2333.0327, 2308.9482, 2274.6692, 2243.7563],
            ],
            [
                [2336.5217, 2313.1685, 2280.2307, 2250.1519],
                [2336.4736, 2313.01, 2279.786, 2249.4578],
                [2336.428, 2312.86, 2279.457, 2248.9458],
            ],
            [
                [2341.0742, 2317.4734, 2286.2961, 2257.571],
                [2341.0452, 2317.3901, 2286.0627, 2257.19],
                [2341.0176, 2317.3113, 2285.8757, 2256.8833],
            ],
            [
                [2344.8657, 2323.547, 2296.8374, 2273.2778],
                [2344.8398, 2323.4849, 2296.7454, 2273.1663],
                [2344.8152, 2323.4263, 2296.6619, 2273.067],
            ],
            [
                [2349.1382, 2329.301, 2304.035, 2281.427],
                [2349.1177, 2329.2473, 2303.9546, 2281.3325],
                [2349.0981, 2329.1968, 2303.8806, 2281.2454],
            ],
        ],
        dtype=np.float32,
    )

    expected_alk_east = np.array(
        [
            [
                [2286.0708, 2268.1797, 2296.637, 2349.1807],
                [2285.9463, 2268.0945, 2296.39, 2349.1306],
                [2285.831, 2268.015, 2296.2256, 2349.0803],
            ],
            [
                [2293.9246, 2272.553, 2300.6946, 2352.7664],
                [2293.7686, 2272.4255, 2300.517, 2352.7217],
                [2293.6233, 2272.3062, 2300.392, 2352.6794],
            ],
            [
                [2291.9067, 2272.7405, 2303.8984, 2352.8047],
                [2291.765, 2272.6216, 2303.715, 2352.7651],
                [2291.6353, 2272.5115, 2303.6008, 2352.7368],
            ],
            [
                [2285.6335, 2269.2698, 2307.5952, 2350.4307],
                [2285.2783, 2269.0054, 2306.387, 2350.1292],
                [2284.9624, 2268.763, 2305.969, 2349.9863],
            ],
            [
                [2276.459, 2264.8486, 2298.3813, 2348.738],
                [2275.739, 2264.2546, 2294.3838, 2348.0044],
                [2275.1042, 2263.713, 2293.4097, 2347.6458],
            ],
            [
                [2265.4736, 2255.1155, 2291.8098, 2345.7793],
                [2263.9817, 2253.9202, 2285.9136, 2343.21],
                [2262.7253, 2252.8772, 2284.6167, 2342.228],
            ],
            [
                [2251.497, 2240.0898, 2287.3826, 2340.8647],
                [2249.3713, 2238.7424, 2278.939, 2336.5393],
                [2247.5896, 2237.644, 2277.2878, 2335.111],
            ],
            [
                [2246.6724, 2234.5867, 2278.6394, 2331.7383],
                [2245.062, 2233.416, 2267.7524, 2326.326],
                [2243.7563, 2232.4644, 2265.7693, 2325.0994],
            ],
            [
                [2250.1519, 2236.288, 2271.4678, 2323.1704],
                [2249.4578, 2235.7515, 2264.6418, 2318.1099],
                [2248.9458, 2235.3486, 2263.6143, 2316.5557],
            ],
            [
                [2257.571, 2241.8618, 2275.5603, 2323.0852],
                [2257.19, 2241.689, 2272.7703, 2321.232],
                [2256.8833, 2241.552, 2272.0093, 2320.7656],
            ],
            [
                [2273.2778, 2254.4583, 2287.1035, 2332.8403],
                [2273.1663, 2254.3816, 2286.312, 2332.5403],
                [2273.067, 2254.3118, 2286.006, 2332.3428],
            ],
            [
                [2281.427, 2262.3428, 2293.7424, 2340.6316],
                [2281.3325, 2262.2893, 2293.2742, 2340.537],
                [2281.2454, 2262.2397, 2293.0657, 2340.4834],
            ],
        ],
        dtype=np.float32,
    )

    expected_alk_north = np.array(
        [
            [
                [2376.7966, 2375.0652, 2372.221, 2349.1807],
                [2376.7944, 2375.0618, 2372.2104, 2349.1306],
                [2376.7927, 2375.0586, 2372.1975, 2349.0803],
            ],
            [
                [2376.7542, 2374.829, 2371.8271, 2352.7664],
                [2376.7505, 2374.8267, 2371.82, 2352.7217],
                [2376.7468, 2374.8242, 2371.8115, 2352.6794],
            ],
            [
                [2377.0146, 2374.9712, 2371.7087, 2352.8047],
                [2377.0103, 2374.9685, 2371.7075, 2352.7651],
                [2377.0063, 2374.9658, 2371.7068, 2352.7368],
            ],
            [
                [2377.384, 2375.5723, 2371.8706, 2350.4307],
                [2377.381, 2375.5708, 2371.8716, 2350.1292],
                [2377.3784, 2375.5698, 2371.8752, 2349.9863],
            ],
            [
                [2378.4973, 2377.2202, 2373.884, 2348.738],
                [2378.501, 2377.2493, 2373.9553, 2348.0044],
                [2378.502, 2377.273, 2374.0251, 2347.6458],
            ],
            [
                [2383.1602, 2382.406, 2379.65, 2345.7793],
                [2383.698, 2383.0017, 2380.1445, 2343.21],
                [2384.0688, 2383.4612, 2380.5059, 2342.228],
            ],
            [
                [2389.3777, 2387.968, 2383.417, 2340.8647],
                [2389.569, 2388.2068, 2383.5066, 2336.5393],
                [2389.715, 2388.3784, 2383.5264, 2335.111],
            ],
            [
                [2385.992, 2383.6086, 2377.531, 2331.7383],
                [2385.9045, 2383.4995, 2376.7078, 2326.326],
                [2385.848, 2383.4219, 2376.5684, 2325.0994],
            ],
            [
                [2373.8606, 2368.452, 2365.9084, 2323.1704],
                [2373.4746, 2367.9546, 2365.2036, 2318.1099],
                [2373.194, 2367.663, 2365.093, 2316.5557],
            ],
            [
                [2371.8428, 2364.6536, 2362.7688, 2323.0852],
                [2371.8066, 2364.526, 2362.5703, 2321.232],
                [2371.7783, 2364.4412, 2362.5378, 2320.7656],
            ],
            [
                [2373.204, 2369.9758, 2367.2737, 2332.8403],
                [2373.1848, 2369.9604, 2367.24, 2332.5403],
                [2373.1663, 2369.9473, 2367.215, 2332.3428],
            ],
            [
                [2374.983, 2373.8923, 2370.9468, 2340.6316],
                [2374.9763, 2373.8818, 2370.925, 2340.537],
                [2374.97, 2373.8723, 2370.9004, 2340.4834],
            ],
        ],
        dtype=np.float32,
    )

    expected_alk_west = np.array(
        [
            [
                [2352.1333, 2335.144, 2362.5112, 2376.7966],
                [2352.1084, 2335.0142, 2362.486, 2376.7944],
                [2352.0847, 2334.891, 2362.459, 2376.7927],
            ],
            [
                [2354.6926, 2334.997, 2363.0098, 2376.7542],
                [2354.6624, 2334.869, 2362.9844, 2376.7505],
                [2354.6338, 2334.7476, 2362.9578, 2376.7468],
            ],
            [
                [2354.9302, 2334.565, 2365.804, 2377.0146],
                [2354.8633, 2334.3918, 2365.7708, 2377.0103],
                [2354.7998, 2334.228, 2365.7332, 2377.0063],
            ],
            [
                [2355.0718, 2335.3025, 2370.7595, 2377.384],
                [2354.9797, 2335.1372, 2370.7163, 2377.381],
                [2354.893, 2334.981, 2370.6716, 2377.3784],
            ],
            [
                [2353.7942, 2335.1846, 2375.3472, 2378.4973],
                [2353.6982, 2334.9646, 2375.2195, 2378.501],
                [2353.6072, 2334.7568, 2375.109, 2378.502],
            ],
            [
                [2345.8477, 2331.5723, 2376.6313, 2383.1602],
                [2345.623, 2331.3728, 2376.1418, 2383.698],
                [2345.4106, 2331.184, 2375.8367, 2384.0688],
            ],
            [
                [2335.3225, 2327.9155, 2370.5518, 2389.3777],
                [2335.0918, 2327.7148, 2369.7534, 2389.569],
                [2334.8738, 2327.5251, 2369.3943, 2389.715],
            ],
            [
                [2333.2937, 2326.0374, 2352.5532, 2385.992],
                [2333.1597, 2325.88, 2351.7349, 2385.9045],
                [2333.0327, 2325.7312, 2351.3489, 2385.848],
            ],
            [
                [2336.5217, 2325.5437, 2345.1482, 2373.8606],
                [2336.4736, 2325.3962, 2344.8074, 2373.4746],
                [2336.428, 2325.2568, 2344.613, 2373.194],
            ],
            [
                [2341.0742, 2328.004, 2351.6812, 2371.8428],
                [2341.0452, 2327.9, 2351.5198, 2371.8066],
                [2341.0176, 2327.8018, 2351.3987, 2371.7783],
            ],
            [
                [2344.8657, 2330.9324, 2355.953, 2373.204],
                [2344.8398, 2330.8267, 2355.8948, 2373.1848],
                [2344.8152, 2330.7268, 2355.8384, 2373.1663],
            ],
            [
                [2349.1382, 2334.0857, 2359.336, 2374.983],
                [2349.1177, 2333.8867, 2359.2932, 2374.9763],
                [2349.0981, 2333.6985, 2359.2524, 2374.97],
            ],
        ],
        dtype=np.float32,
    )

    ds_bgc = bgc_boundary_forcing_from_climatology.ds

    # Check the values in the dataset
    assert np.allclose(ds_bgc["ALK_south"].values, expected_alk_south)
    assert np.allclose(ds_bgc["ALK_east"].values, expected_alk_east)
    assert np.allclose(ds_bgc["ALK_north"].values, expected_alk_north)
    assert np.allclose(ds_bgc["ALK_west"].values, expected_alk_west)

    bgc_boundary_forcing_from_climatology.plot(varname="ALK_south")
    bgc_boundary_forcing_from_climatology.plot(varname="ALK_east")
    bgc_boundary_forcing_from_climatology.plot(varname="ALK_north")
    bgc_boundary_forcing_from_climatology.plot(varname="ALK_west")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        filepath = tmpfile.name

    bgc_boundary_forcing_from_climatology.save(filepath)
    extended_filepath = filepath + "_clim.nc"

    try:
        assert os.path.exists(extended_filepath)
    finally:
        os.remove(extended_filepath)


@pytest.mark.parametrize(
    "bdry_forcing_fixture",
    [
        "boundary_forcing",
        "bgc_boundary_forcing_from_climatology",
    ],
)
def test_roundtrip_yaml(bdry_forcing_fixture, request):
    """Test that creating a BoundaryForcing object, saving its parameters to yaml file, and re-opening yaml file creates the same object."""

    bdry_forcing = request.getfixturevalue(bdry_forcing_fixture)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    try:
        bdry_forcing.to_yaml(filepath)

        boundary_forcing_from_file = BoundaryForcing.from_yaml(filepath)

        assert bdry_forcing == boundary_forcing_from_file

    finally:
        os.remove(filepath)


def test_from_yaml_missing_boundary_forcing():
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
            ValueError, match="No BoundaryForcing configuration found in the YAML file."
        ):
            BoundaryForcing.from_yaml(yaml_filepath)
    finally:
        os.remove(yaml_filepath)
