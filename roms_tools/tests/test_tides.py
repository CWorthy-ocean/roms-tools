import pytest
import tempfile
import os
from roms_tools import Grid, TidalForcing
import xarray as xr
import numpy as np
from roms_tools.setup.datasets import download_test_data
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

    tidal_forcing = TidalForcing(grid=grid, filename=fname, source="TPXO", ntides=2)

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

    assert tidal_forcing.filename == fname
    assert tidal_forcing.source == "TPXO"
    assert tidal_forcing.ntides == 2


def test_successful_initialization_with_regional_data(
    grid_that_lies_within_bounds_of_regional_tpxo_data,
):

    fname = download_test_data("TPXO_regional_test_data.nc")

    tidal_forcing = TidalForcing(
        grid=grid_that_lies_within_bounds_of_regional_tpxo_data,
        filename=fname,
        source="TPXO",
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

    assert tidal_forcing.filename == fname
    assert tidal_forcing.source == "TPXO"
    assert tidal_forcing.ntides == 10


def test_unsuccessful_initialization_with_regional_data_due_to_nans(
    grid_that_is_out_of_bounds_of_regional_tpxo_data,
):

    fname = download_test_data("TPXO_regional_test_data.nc")

    with pytest.raises(ValueError, match="NaN values found"):
        TidalForcing(
            grid=grid_that_is_out_of_bounds_of_regional_tpxo_data,
            filename=fname,
            source="TPXO",
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
        TidalForcing(grid=grid, filename=fname, source="TPXO", ntides=10)


def test_insufficient_number_of_consituents(grid_that_straddles_dateline):

    fname = download_test_data("TPXO_global_test_data.nc")

    with pytest.raises(ValueError, match="The dataset contains fewer"):
        TidalForcing(
            grid=grid_that_straddles_dateline, filename=fname, source="TPXO", ntides=10
        )


@pytest.fixture
def tidal_forcing(
    grid_that_lies_within_bounds_of_regional_tpxo_data,
):

    fname = download_test_data("TPXO_regional_test_data.nc")

    return TidalForcing(
        grid=grid_that_lies_within_bounds_of_regional_tpxo_data,
        filename=fname,
        source="TPXO",
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
                [-0.01043007, -0.00768077, -0.00370782, 0.00174401],
                [-0.01046313, -0.00833564, -0.00534876, -0.00036892],
                [-0.01149787, -0.0117521, -0.01165313, -0.00668873],
                [-0.01435909, -0.01959155, -0.02610414, -0.02264688],
                [-0.01590802, -0.02578601, -0.01770638, -0.00307389],
            ]
        ],
        dtype=np.float32,
    )

    expected_u_Im = np.array(
        [
            [
                [0.00068068, 0.00041515, -0.00098873, -0.00315086],
                [0.00103654, 0.0007357, -0.000998, -0.00411228],
                [0.0017258, 0.00158265, -0.0014292, -0.00713451],
                [0.00458748, 0.00451903, 0.00046625, -0.00838845],
                [0.00930313, 0.01001076, 0.00501656, 0.0004481],
            ]
        ],
        dtype=np.float32,
    )

    expected_v_Re = np.array(
        [
            [
                [0.01867937, 0.0175135, 0.0163139, 0.01373139, 0.0114212],
                [0.02016588, 0.01930715, 0.01812451, 0.01638661, 0.01130948],
                [0.02174281, 0.02100098, 0.02242658, 0.02136369, 0.01128179],
                [0.02275964, 0.0218871, 0.02481382, 0.01351837, 0.00718958],
            ]
        ],
        dtype=np.float32,
    )

    expected_v_Im = np.array(
        [
            [
                [-0.00304336, 0.00069296, 0.00384371, 0.00627055, 0.00745201],
                [-0.00472402, -0.00109876, 0.0024061, 0.00627166, 0.00790893],
                [-0.00699575, -0.00359212, 0.00066638, 0.00706607, 0.01097147],
                [-0.00954442, -0.00623799, -0.00171383, 0.00425109, 0.00574474],
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
