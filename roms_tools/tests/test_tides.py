import pytest
import tempfile
import os
from roms_tools import Grid, TidalForcing
import xarray as xr
import numpy as np
from roms_tools.setup.download import download_test_data
import textwrap


@pytest.fixture
def grid_that_lies_within_bounds_of_regional_tpxo_data():
    grid = Grid(
        nx=3, ny=3, size_x=1500, size_y=1500, center_lon=235, center_lat=25, rot=-20
    )
    return grid


@pytest.fixture
def grid_that_is_out_of_bounds_of_regional_tpxo_data():
    grid = Grid(
        nx=3, ny=3, size_x=1800, size_y=1500, center_lon=235, center_lat=25, rot=-20
    )
    return grid


@pytest.fixture
def grid_that_straddles_dateline():
    """
    Fixture for creating a domain that straddles the dateline.
    """
    grid = Grid(
        nx=5,
        ny=5,
        size_x=1800,
        size_y=2400,
        center_lon=-10,
        center_lat=30,
        rot=20,
    )

    return grid


@pytest.fixture
def grid_that_straddles_180_degree_meridian():
    """
    Fixture for creating a domain that straddles 180 degree meridian.
    """

    grid = Grid(
        nx=5,
        ny=5,
        size_x=1800,
        size_y=2400,
        center_lon=180,
        center_lat=30,
        rot=20,
    )

    return grid


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid_that_lies_within_bounds_of_regional_tpxo_data",
        "grid_that_is_out_of_bounds_of_regional_tpxo_data",
        "grid_that_straddles_dateline",
        "grid_that_straddles_180_degree_meridian",
    ],
)
def test_successful_initialization_with_global_data(grid_fixture, request):

    fname = download_test_data("TPXO_global_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    tidal_forcing = TidalForcing(
        grid=grid, source={"name": "TPXO", "path": fname}, ntides=2
    )

    assert isinstance(tidal_forcing.ds, xr.Dataset)
    assert "omega" in tidal_forcing.ds
    assert "ssh_Re" in tidal_forcing.ds
    assert "ssh_Im" in tidal_forcing.ds
    assert "pot_Re" in tidal_forcing.ds
    assert "pot_Im" in tidal_forcing.ds
    assert "u_Re" in tidal_forcing.ds
    assert "u_Im" in tidal_forcing.ds
    assert "v_Re" in tidal_forcing.ds
    assert "v_Im" in tidal_forcing.ds

    assert tidal_forcing.source == {"name": "TPXO", "path": fname}
    assert tidal_forcing.ntides == 2


def test_successful_initialization_with_regional_data(
    grid_that_lies_within_bounds_of_regional_tpxo_data,
):

    fname = download_test_data("TPXO_regional_test_data.nc")

    tidal_forcing = TidalForcing(
        grid=grid_that_lies_within_bounds_of_regional_tpxo_data,
        source={"name": "TPXO", "path": fname},
        ntides=10,
    )

    assert isinstance(tidal_forcing.ds, xr.Dataset)
    assert "omega" in tidal_forcing.ds
    assert "ssh_Re" in tidal_forcing.ds
    assert "ssh_Im" in tidal_forcing.ds
    assert "pot_Re" in tidal_forcing.ds
    assert "pot_Im" in tidal_forcing.ds
    assert "u_Re" in tidal_forcing.ds
    assert "u_Im" in tidal_forcing.ds
    assert "v_Re" in tidal_forcing.ds
    assert "v_Im" in tidal_forcing.ds

    assert tidal_forcing.source == {"name": "TPXO", "path": fname}
    assert tidal_forcing.ntides == 10


def test_unsuccessful_initialization_with_regional_data_due_to_nans(
    grid_that_is_out_of_bounds_of_regional_tpxo_data,
):

    fname = download_test_data("TPXO_regional_test_data.nc")

    with pytest.raises(ValueError, match="NaN values found"):
        TidalForcing(
            grid=grid_that_is_out_of_bounds_of_regional_tpxo_data,
            source={"name": "TPXO", "path": fname},
            ntides=10,
        )


@pytest.mark.parametrize(
    "grid_fixture",
    ["grid_that_straddles_dateline", "grid_that_straddles_180_degree_meridian"],
)
def test_unsuccessful_initialization_with_regional_data_due_to_no_overlap(
    grid_fixture, request
):

    fname = download_test_data("TPXO_regional_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    with pytest.raises(
        ValueError, match="Selected longitude range does not intersect with dataset"
    ):
        TidalForcing(grid=grid, source={"name": "TPXO", "path": fname}, ntides=10)


def test_insufficient_number_of_consituents(grid_that_straddles_dateline):

    fname = download_test_data("TPXO_global_test_data.nc")

    with pytest.raises(ValueError, match="The dataset contains fewer"):
        TidalForcing(
            grid=grid_that_straddles_dateline,
            source={"name": "TPXO", "path": fname},
            ntides=10,
        )


@pytest.fixture
def tidal_forcing(
    grid_that_lies_within_bounds_of_regional_tpxo_data,
):

    fname = download_test_data("TPXO_regional_test_data.nc")

    return TidalForcing(
        grid=grid_that_lies_within_bounds_of_regional_tpxo_data,
        source={"name": "TPXO", "path": fname},
        ntides=1,
    )


def test_tidal_forcing_data_consistency_plot_save(tidal_forcing, tmp_path):
    """
    Test that the data within the TidalForcing object remains consistent.
    Also test plot and save methods in the same test since we dask arrays are already computed.
    """
    tidal_forcing.ds.load()

    expected_ssh_Re = np.array(
        [
            [
                [0.03362583, 0.0972546, 0.15625167, 0.20162642, 0.22505085],
                [0.04829295, 0.13148762, 0.2091077, 0.26777256, 0.28947946],
                [0.0574473, 0.16427538, 0.26692376, 0.335315, 0.35217384],
                [0.0555277, 0.1952368, 0.32960117, 0.41684473, 0.43021917],
                [0.04893931, 0.22524744, 0.39933527, 0.39793402, -0.18146336],
            ]
        ],
        dtype=np.float32,
    )

    expected_ssh_Im = np.array(
        [
            [
                [0.28492996, 0.33401084, 0.3791059, 0.40458283, 0.39344734],
                [0.14864475, 0.19812492, 0.25232342, 0.29423112, 0.30793712],
                [-0.01214434, 0.04206207, 0.11318521, 0.18785079, 0.24001373],
                [-0.18849652, -0.13063835, -0.02998546, 0.0921034, 0.20685565],
                [-0.36839223, -0.31615746, -0.18911538, -0.08607443, -0.51923835],
            ]
        ],
        dtype=np.float32,
    )

    expected_pot_Re = np.array(
        [
            [
                [-0.11110803, -0.08998635, -0.06672653, -0.04285957, -0.01980283],
                [-0.10053363, -0.07692371, -0.05161811, -0.02654761, -0.00358691],
                [-0.08996155, -0.06400539, -0.03679418, -0.01115401, 0.01084424],
                [-0.08017255, -0.05190206, -0.0231975, 0.00163647, 0.01880641],
                [-0.07144432, -0.04169955, -0.0143679, -0.00313035, 0.00145161],
            ]
        ],
        dtype=np.float32,
    )

    expected_pot_Im = np.array(
        [
            [
                [-0.05019786, -0.06314129, -0.07475527, -0.08616351, -0.09869237],
                [-0.06716369, -0.07930522, -0.08920974, -0.09815053, -0.10770469],
                [-0.08508184, -0.09582505, -0.10310414, -0.10833713, -0.1137427],
                [-0.10244609, -0.11144008, -0.1151103, -0.11618311, -0.11833992],
                [-0.11764989, -0.12432244, -0.12302232, -0.12279626, -0.13328244],
            ]
        ],
        dtype=np.float32,
    )

    expected_u_Re = np.array(
        [
            [
                [-0.01076973, -0.00799587, -0.00392205, 0.00187405],
                [-0.01080387, -0.00868488, -0.00567306, -0.00039643],
                [-0.01225182, -0.01278006, -0.01299872, -0.00752568],
                [-0.01588197, -0.02170864, -0.02925579, -0.02560116],
                [-0.01759518, -0.02857426, -0.01967089, -0.00347487],
            ]
        ],
        dtype=np.float32,
    )

    expected_u_Im = np.array(
        [
            [
                [0.00070284, 0.00042831, -0.00106211, -0.00338581],
                [0.00107029, 0.00076322, -0.00107583, -0.00441891],
                [0.00183897, 0.00170466, -0.001616, -0.00802724],
                [0.00507402, 0.00500577, 0.00047888, -0.00948272],
                [0.0102898, 0.01108951, 0.00556095, 0.00050655],
            ]
        ],
        dtype=np.float32,
    )

    expected_v_Re = np.array(
        [
            [
                [0.01928766, 0.01808383, 0.0171951, 0.01475528, 0.01227283],
                [0.02116097, 0.02027666, 0.01961637, 0.01808529, 0.0124496],
                [0.02363378, 0.02281034, 0.02485597, 0.02410039, 0.01272225],
                [0.02517346, 0.02420838, 0.02752665, 0.01528184, 0.00812746],
            ]
        ],
        dtype=np.float32,
    )

    expected_v_Im = np.array(
        [
            [
                [-0.00314247, 0.00071553, 0.00405133, 0.00673812, 0.00800767],
                [-0.00496862, -0.00116939, 0.00257849, 0.00691007, 0.00872064],
                [-0.00762639, -0.00393035, 0.00073738, 0.00796987, 0.01237934],
                [-0.01055667, -0.00689958, -0.0019012, 0.00480565, 0.00649413],
            ]
        ],
        dtype=np.float32,
    )

    # Check the values in the dataset
    assert np.allclose(tidal_forcing.ds["ssh_Re"].values, expected_ssh_Re)
    assert np.allclose(tidal_forcing.ds["ssh_Im"].values, expected_ssh_Im)
    assert np.allclose(tidal_forcing.ds["pot_Re"].values, expected_pot_Re)
    assert np.allclose(tidal_forcing.ds["pot_Im"].values, expected_pot_Im)
    assert np.allclose(tidal_forcing.ds["u_Re"].values, expected_u_Re)
    assert np.allclose(tidal_forcing.ds["u_Im"].values, expected_u_Im)
    assert np.allclose(tidal_forcing.ds["v_Re"].values, expected_v_Re)
    assert np.allclose(tidal_forcing.ds["v_Im"].values, expected_v_Im)

    tidal_forcing.plot(varname="ssh_Re", ntides=0)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    tidal_forcing.save(filepath)

    try:
        assert os.path.exists(filepath)
    finally:
        os.remove(filepath)


def test_roundtrip_yaml(tidal_forcing):
    """Test that creating a TidalForcing object, saving its parameters to yaml file, and re-opening yaml file creates the same object."""

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    try:
        tidal_forcing.to_yaml(filepath)

        tidal_forcing_from_file = TidalForcing.from_yaml(filepath)

        assert tidal_forcing == tidal_forcing_from_file

    finally:
        os.remove(filepath)


def test_from_yaml_missing_tidal_forcing():
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
            ValueError, match="No TidalForcing configuration found in the YAML file."
        ):
            TidalForcing.from_yaml(yaml_filepath)
    finally:
        os.remove(yaml_filepath)
