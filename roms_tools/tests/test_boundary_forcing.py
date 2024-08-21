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
        physics_source={"name": "GLORYS", "path": fname},
    )


@pytest.fixture
def boundary_forcing_with_bgc_from_climatology(example_grid):
    """
    Fixture for creating a BoundaryForcing object.
    """

    fname = download_test_data("GLORYS_test_data.nc")
    fname_bgc = download_test_data("CESM_regional_test_data_climatology.nc")

    return BoundaryForcing(
        grid=example_grid,
        start_time=datetime(2021, 6, 29),
        end_time=datetime(2021, 6, 30),
        physics_source={"name": "GLORYS", "path": fname},
        bgc_source={"path": fname_bgc, "name": "CESM_REGRIDDED", "climatology": True},
    )


@pytest.mark.parametrize(
    "bdry_forcing_fixture",
    [
        "boundary_forcing",
        "boundary_forcing_with_bgc_from_climatology",
    ],
)
def test_boundary_forcing_creation(bdry_forcing_fixture, request):
    """
    Test the creation of the BoundaryForcing object.
    """

    bdry_forcing = request.getfixturevalue(bdry_forcing_fixture)

    assert bdry_forcing.start_time == datetime(2021, 6, 29)
    assert bdry_forcing.end_time == datetime(2021, 6, 30)

    assert bdry_forcing.ds["physics"].physics_source == "GLORYS"
    for direction in ["south", "east", "north", "west"]:
        assert f"temp_{direction}" in bdry_forcing.ds["physics"]
        assert f"salt_{direction}" in bdry_forcing.ds["physics"]
        assert f"u_{direction}" in bdry_forcing.ds["physics"]
        assert f"v_{direction}" in bdry_forcing.ds["physics"]
        assert f"zeta_{direction}" in bdry_forcing.ds["physics"]


def test_boundary_forcing_creation_with_bgc(boundary_forcing_with_bgc_from_climatology):
    """
    Test the creation of the BoundaryForcing object.
    """

    assert (
        boundary_forcing_with_bgc_from_climatology.ds["bgc"].bgc_source
        == "CESM_REGRIDDED"
    )
    for direction in ["south", "east", "north", "west"]:
        for var in ["ALK", "PO4"]:
            assert (
                f"{var}_{direction}"
                in boundary_forcing_with_bgc_from_climatology.ds["bgc"]
            )


def test_boundary_forcing_data_consistency_plot_save(
    boundary_forcing_with_bgc_from_climatology,
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
                [16.84414, 16.905312, 16.967817],
                [18.088203, 18.121834, 18.315424],
                [18.431192, 18.496748, 18.718002],
                [19.294329, 19.30358, 19.439777],
            ]
        ],
        dtype=np.float32,
    )
    expected_temp_east = np.array(
        [
            [
                [19.294329, 19.30358, 19.439777],
                [18.633307, 18.637077, 18.667465],
                [8.710737, 11.25943, 13.111585],
                [9.20282, 10.667074, 11.752404],
            ]
        ],
        dtype=np.float32,
    )
    expected_temp_north = np.array(
        [
            [
                [10.233599, 10.546486, 10.671082],
                [10.147332, 10.502733, 10.68275],
                [10.458557, 11.209945, 11.377164],
                [9.20282, 10.667074, 11.752404],
            ]
        ],
        dtype=np.float32,
    )
    expected_temp_west = np.array(
        [
            [
                [16.84414, 16.905312, 16.967817],
                [12.639833, 13.479691, 14.426711],
                [11.027701, 11.650267, 12.200586],
                [10.233599, 10.546486, 10.671082],
            ]
        ],
        dtype=np.float32,
    )

    expected_u_south = np.array(
        [[[-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0], [0.0, -0.0, -0.0]]], dtype=np.float32
    )
    expected_u_east = np.array(
        [
            [
                [0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0],
                [0.06979556, 0.06167743, -0.02247071],
                [0.0211786, 0.03679834, 0.0274788],
            ]
        ],
        dtype=np.float32,
    )
    expected_u_north = np.array(
        [
            [
                [0.04268532, 0.03889201, 0.03351666],
                [0.04645353, 0.04914769, 0.03673013],
                [0.0211786, 0.03679834, 0.0274788],
            ]
        ],
        dtype=np.float32,
    )
    expected_u_west = np.array(
        [
            [
                [-0.0, -0.0, -0.0],
                [0.0, -0.0, -0.0],
                [0.0, 0.0, -0.0],
                [0.04268532, 0.03889201, 0.03351666],
            ]
        ],
        dtype=np.float32,
    )

    expected_v_south = np.array(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, -0.0], [-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0]]],
        dtype=np.float32,
    )
    expected_v_east = np.array(
        [
            [
                [-0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0],
                [-0.06720348, -0.08354441, -0.13835917],
            ]
        ],
        dtype=np.float32,
    )
    expected_v_north = np.array(
        [
            [
                [-0.00951457, -0.00576979, -0.02147919],
                [-0.0, -0.0, -0.0],
                [0.01915873, 0.02625698, 0.01757628],
                [-0.06720348, -0.08354441, -0.13835917],
            ]
        ],
        dtype=np.float32,
    )
    expected_v_west = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [-0.0, -0.0, -0.0],
                [-0.00951457, -0.00576979, -0.02147919],
            ]
        ],
        dtype=np.float32,
    )

    expected_ubar_south = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    expected_ubar_east = np.array(
        [[0.0, 0.0, 0.04028399, 0.02812303]], dtype=np.float32
    )
    expected_ubar_north = np.array(
        [[0.03866891, 0.04446249, 0.02812303]], dtype=np.float32
    )
    expected_ubar_west = np.array([[0.0, 0.0, 0.0, 0.03866891]], dtype=np.float32)

    expected_vbar_south = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    expected_vbar_east = np.array([[0.0, 0.0, -0.09326097]], dtype=np.float32)
    expected_vbar_north = np.array(
        [[-0.01189703, 0.0, 0.02102064, -0.09326097]], dtype=np.float32
    )
    expected_vbar_west = np.array([[0.0, 0.0, -0.01189703]], dtype=np.float32)

    expected_alk_south = np.array(
        [
            [
                [2352.1636, 2352.128, 2352.091],
                [2333.1094, 2332.9932, 2332.8738],
                [2308.618, 2308.4705, 2308.3032],
                [2286.2327, 2286.0442, 2285.8623],
            ],
            [
                [2354.7297, 2354.6863, 2354.6414],
                [2337.391, 2337.259, 2337.1223],
                [2314.0232, 2313.8638, 2313.6616],
                [2294.0876, 2293.8918, 2293.6626],
            ],
            [
                [2355.02, 2354.9158, 2354.817],
                [2336.321, 2336.117, 2335.9214],
                [2312.7905, 2312.4849, 2312.2444],
                [2292.2637, 2291.8752, 2291.6702],
            ],
            [
                [2355.2583, 2355.052, 2354.9163],
                [2334.3098, 2333.913, 2333.651],
                [2309.5273, 2308.292, 2307.8772],
                [2287.4226, 2285.5466, 2285.0479],
            ],
            [
                [2354.1216, 2353.7737, 2353.6316],
                [2330.6604, 2329.948, 2329.6023],
                [2304.88, 2302.105, 2301.4136],
                [2280.6, 2276.278, 2275.2761],
            ],
            [
                [2346.881, 2345.7993, 2345.4675],
                [2322.218, 2320.3225, 2319.691],
                [2301.2046, 2291.6438, 2290.308],
                [2279.729, 2265.0488, 2263.0654],
            ],
            [
                [2336.2632, 2335.273, 2334.9324],
                [2312.938, 2310.7412, 2310.055],
                [2293.6711, 2279.314, 2277.593],
                [2272.6418, 2250.8848, 2248.0718],
            ],
            [
                [2333.8801, 2333.265, 2333.0667],
                [2311.3232, 2309.5337, 2309.034],
                [2289.1548, 2276.1438, 2274.885],
                [2266.9172, 2246.1714, 2244.1096],
            ],
            [
                [2336.8193, 2336.5115, 2336.4402],
                [2314.2544, 2313.1343, 2312.9004],
                [2288.887, 2280.0654, 2279.546],
                [2263.7944, 2249.8926, 2249.0842],
            ],
            [
                [2341.18, 2341.0679, 2341.025],
                [2317.8801, 2317.4556, 2317.3325],
                [2289.4714, 2286.2214, 2285.9263],
                [2262.6213, 2257.4502, 2256.9663],
            ],
            [
                [2344.9026, 2344.86, 2344.8218],
                [2323.6404, 2323.5337, 2323.442],
                [2297.1284, 2296.8164, 2296.6846],
                [2273.7996, 2273.2507, 2273.0938],
            ],
            [
                [2349.1636, 2349.1338, 2349.1035],
                [2329.3665, 2329.2893, 2329.2102],
                [2304.1536, 2304.0178, 2303.9006],
                [2281.6301, 2281.4062, 2281.269],
            ],
        ],
        dtype=np.float32,
    )

    expected_alk_east = np.array(
        [
            [
                [2286.2327, 2286.0442, 2285.8623],
                [2268.2373, 2268.162, 2268.0366],
                [2297.2825, 2296.5747, 2296.2695],
                [2349.247, 2349.1697, 2349.0938],
            ],
            [
                [2294.0876, 2293.8918, 2293.6626],
                [2272.6245, 2272.5269, 2272.3384],
                [2301.072, 2300.6504, 2300.4255],
                [2352.814, 2352.756, 2352.6907],
            ],
            [
                [2292.2637, 2291.8752, 2291.6702],
                [2272.901, 2272.7153, 2272.5413],
                [2304.492, 2303.8516, 2303.6313],
                [2352.8616, 2352.795, 2352.7444],
            ],
            [
                [2287.4226, 2285.5466, 2285.0479],
                [2269.8926, 2269.2112, 2268.8284],
                [2310.2466, 2307.2576, 2306.0806],
                [2350.9084, 2350.3499, 2350.0247],
            ],
            [
                [2280.6, 2276.278, 2275.2761],
                [2266.4937, 2264.7148, 2263.8596],
                [2309.5537, 2297.2288, 2293.6697],
                [2350.1707, 2348.5422, 2347.7415],
            ],
            [
                [2279.729, 2265.0488, 2263.0654],
                [2263.0278, 2254.806, 2253.1594],
                [2311.0305, 2290.098, 2284.9631],
                [2350.3872, 2345.0693, 2342.4902],
            ],
            [
                [2272.6418, 2250.8848, 2248.0718],
                [2256.4866, 2239.6753, 2237.9412],
                [2309.7383, 2284.9133, 2277.729],
                [2349.7207, 2339.6501, 2335.4924],
            ],
            [
                [2266.9172, 2246.1714, 2244.1096],
                [2249.1013, 2234.224, 2232.722],
                [2307.639, 2275.4429, 2266.299],
                [2347.7605, 2330.1702, 2325.427],
            ],
            [
                [2263.7944, 2249.8926, 2249.0842],
                [2246.2002, 2236.0933, 2235.4578],
                [2301.0461, 2269.4448, 2263.889],
                [2338.5127, 2321.7393, 2316.971],
            ],
            [
                [2262.6213, 2257.4502, 2256.9663],
                [2244.3662, 2241.8052, 2241.589],
                [2292.1316, 2274.763, 2272.2124],
                [2331.162, 2322.5522, 2320.8901],
            ],
            [
                [2273.7996, 2273.2507, 2273.0938],
                [2254.6848, 2254.4407, 2254.3308],
                [2290.5637, 2286.885, 2286.088],
                [2333.583, 2332.7646, 2332.3955],
            ],
            [
                [2281.6301, 2281.4062, 2281.269],
                [2262.4211, 2262.3313, 2262.2532],
                [2296.095, 2293.6157, 2293.1213],
                [2340.852, 2340.6072, 2340.4978],
            ],
        ],
        dtype=np.float32,
    )

    expected_alk_north = np.array(
        [
            [
                [2376.7993, 2376.7961, 2376.7932],
                [2375.0688, 2375.0647, 2375.0596],
                [2372.2307, 2372.2188, 2372.201],
                [2349.247, 2349.1697, 2349.0938],
            ],
            [
                [2376.757, 2376.7537, 2376.7478],
                [2374.8325, 2374.8286, 2374.825],
                [2371.8342, 2371.8257, 2371.814],
                [2352.814, 2352.756, 2352.6907],
            ],
            [
                [2377.0188, 2377.0137, 2377.0073],
                [2374.9753, 2374.9707, 2374.9666],
                [2371.7104, 2371.7085, 2371.707],
                [2352.8616, 2352.795, 2352.7444],
            ],
            [
                [2377.3914, 2377.383, 2377.3792],
                [2375.5757, 2375.5718, 2375.57],
                [2371.869, 2371.8706, 2371.8743],
                [2350.9084, 2350.3499, 2350.0247],
            ],
            [
                [2378.4722, 2378.5, 2378.5017],
                [2377.151, 2377.2292, 2377.2666],
                [2373.7805, 2373.9, 2374.0066],
                [2350.1707, 2348.5422, 2347.7415],
            ],
            [
                [2381.3555, 2383.3838, 2383.9692],
                [2380.8108, 2382.6125, 2383.3374],
                [2378.7131, 2379.7717, 2380.4094],
                [2350.3872, 2345.0693, 2342.4902],
            ],
            [
                [2388.7502, 2389.4453, 2389.6758],
                [2387.0815, 2388.0613, 2388.3323],
                [2381.9631, 2383.4429, 2383.521],
                [2349.7207, 2339.6501, 2335.4924],
            ],
            [
                [2386.29, 2385.9524, 2385.863],
                [2383.9194, 2383.5654, 2383.4429],
                [2380.237, 2377.2883, 2376.6057],
                [2347.7605, 2330.1702, 2325.427],
            ],
            [
                [2375.1409, 2373.7124, 2373.2695],
                [2370.7205, 2368.2014, 2367.7415],
                [2371.8071, 2365.7, 2365.1226],
                [2338.5127, 2321.7393, 2316.971],
            ],
            [
                [2371.951, 2371.8308, 2371.786],
                [2365.2358, 2364.598, 2364.4639],
                [2365.8464, 2362.7102, 2362.5464],
                [2331.162, 2322.5522, 2320.8901],
            ],
            [
                [2373.2239, 2373.2004, 2373.1711],
                [2370.007, 2369.9714, 2369.9507],
                [2367.3286, 2367.2654, 2367.2217],
                [2333.583, 2332.7646, 2332.3955],
            ],
            [
                [2374.99, 2374.9817, 2374.9717],
                [2373.9058, 2373.8901, 2373.8748],
                [2370.9688, 2370.9421, 2370.907],
                [2340.852, 2340.6072, 2340.4978],
            ],
        ],
        dtype=np.float32,
    )

    expected_alk_west = np.array(
        [
            [
                [2352.1636, 2352.128, 2352.091],
                [2335.311, 2335.1162, 2334.924],
                [2362.5132, 2362.5088, 2362.4663],
                [2376.7993, 2376.7961, 2376.7932],
            ],
            [
                [2354.7297, 2354.6863, 2354.6414],
                [2335.165, 2334.9695, 2334.78],
                [2363.018, 2363.0068, 2362.965],
                [2376.757, 2376.7537, 2376.7478],
            ],
            [
                [2355.02, 2354.9158, 2354.817],
                [2334.8015, 2334.5276, 2334.272],
                [2365.7913, 2365.8025, 2365.7432],
                [2377.0188, 2377.0137, 2377.0073],
            ],
            [
                [2355.2583, 2355.052, 2354.9163],
                [2335.5789, 2335.267, 2335.0227],
                [2370.7808, 2370.754, 2370.6836],
                [2377.3914, 2377.383, 2377.3792],
            ],
            [
                [2354.1216, 2353.7737, 2353.6316],
                [2335.7122, 2335.1372, 2334.8125],
                [2375.5793, 2375.3132, 2375.1387],
                [2378.4722, 2378.5, 2378.5017],
            ],
            [
                [2346.881, 2345.7993, 2345.4675],
                [2332.2158, 2331.5295, 2331.2346],
                [2378.4954, 2376.4001, 2375.9187],
                [2381.3555, 2383.3838, 2383.9692],
            ],
            [
                [2336.2632, 2335.273, 2334.9324],
                [2328.7188, 2327.8726, 2327.576],
                [2374.763, 2370.0576, 2369.491],
                [2388.7502, 2389.4453, 2389.6758],
            ],
            [
                [2333.8801, 2333.265, 2333.0667],
                [2326.699, 2326.0037, 2325.771],
                [2356.8801, 2352.0618, 2351.4526],
                [2386.29, 2385.9524, 2385.863],
            ],
            [
                [2336.8193, 2336.5115, 2336.4402],
                [2325.9893, 2325.512, 2325.2942],
                [2346.735, 2344.9722, 2344.6653],
                [2375.1409, 2373.7124, 2373.2695],
            ],
            [
                [2341.18, 2341.0679, 2341.025],
                [2328.2231, 2327.9817, 2327.8281],
                [2352.159, 2351.6226, 2351.4314],
                [2371.951, 2371.8308, 2371.786],
            ],
            [
                [2344.9026, 2344.86, 2344.8218],
                [2331.0703, 2330.9097, 2330.7537],
                [2356.0095, 2355.9424, 2355.8538],
                [2373.2239, 2373.2004, 2373.1711],
            ],
            [
                [2349.1636, 2349.1338, 2349.1035],
                [2334.3328, 2334.043, 2333.749],
                [2359.3823, 2359.3276, 2359.2634],
                [2374.99, 2374.9817, 2374.9717],
            ],
        ],
        dtype=np.float32,
    )

    ds = boundary_forcing_with_bgc_from_climatology.ds["physics"]
    ds_bgc = boundary_forcing_with_bgc_from_climatology.ds["bgc"]

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
    assert np.allclose(ds_bgc["ALK_south"].values, expected_alk_south)
    assert np.allclose(ds_bgc["ALK_east"].values, expected_alk_east)
    assert np.allclose(ds_bgc["ALK_north"].values, expected_alk_north)
    assert np.allclose(ds_bgc["ALK_west"].values, expected_alk_west)

    boundary_forcing_with_bgc_from_climatology.plot(varname="temp_south")
    boundary_forcing_with_bgc_from_climatology.plot(varname="temp_east")
    boundary_forcing_with_bgc_from_climatology.plot(varname="temp_north")
    boundary_forcing_with_bgc_from_climatology.plot(varname="temp_west")
    boundary_forcing_with_bgc_from_climatology.plot(varname="zeta_south")
    boundary_forcing_with_bgc_from_climatology.plot(varname="zeta_east")
    boundary_forcing_with_bgc_from_climatology.plot(varname="zeta_north")
    boundary_forcing_with_bgc_from_climatology.plot(varname="zeta_west")
    boundary_forcing_with_bgc_from_climatology.plot(varname="ALK_south")
    boundary_forcing_with_bgc_from_climatology.plot(varname="ALK_east")
    boundary_forcing_with_bgc_from_climatology.plot(varname="ALK_north")
    boundary_forcing_with_bgc_from_climatology.plot(varname="ALK_west")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        filepath = tmpfile.name

    boundary_forcing_with_bgc_from_climatology.save(filepath)
    physics_filepath = filepath + "_physics_20210629-29.nc"
    bgc_filepath = filepath + "_bgc_clim.nc"

    try:
        assert os.path.exists(physics_filepath)
        assert os.path.exists(bgc_filepath)
    finally:
        os.remove(physics_filepath)
        os.remove(bgc_filepath)


@pytest.mark.parametrize(
    "bdry_forcing_fixture",
    [
        "boundary_forcing",
        "boundary_forcing_with_bgc_from_climatology",
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
