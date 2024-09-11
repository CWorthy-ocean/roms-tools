import pytest
from roms_tools import Grid, TidalForcing
import xarray as xr
from roms_tools.setup.download import download_test_data
import textwrap
from roms_tools.tests.test_setup.conftest import calculate_file_hash
from pathlib import Path


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
def test_successful_initialization_with_global_data(grid_fixture, request, use_dask):

    fname = download_test_data("TPXO_global_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    tidal_forcing = TidalForcing(
        grid=grid, source={"name": "TPXO", "path": fname}, ntides=2, use_dask=use_dask
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
    grid_that_lies_within_bounds_of_regional_tpxo_data, use_dask
):

    fname = download_test_data("TPXO_regional_test_data.nc")

    tidal_forcing = TidalForcing(
        grid=grid_that_lies_within_bounds_of_regional_tpxo_data,
        source={"name": "TPXO", "path": fname},
        ntides=10,
        use_dask=use_dask,
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
    grid_that_is_out_of_bounds_of_regional_tpxo_data, use_dask
):

    fname = download_test_data("TPXO_regional_test_data.nc")

    with pytest.raises(ValueError, match="NaN values found"):
        TidalForcing(
            grid=grid_that_is_out_of_bounds_of_regional_tpxo_data,
            source={"name": "TPXO", "path": fname},
            ntides=10,
            use_dask=use_dask,
        )


@pytest.mark.parametrize(
    "grid_fixture",
    ["grid_that_straddles_dateline", "grid_that_straddles_180_degree_meridian"],
)
def test_unsuccessful_initialization_with_regional_data_due_to_no_overlap(
    grid_fixture, request, use_dask
):

    fname = download_test_data("TPXO_regional_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    with pytest.raises(
        ValueError, match="Selected longitude range does not intersect with dataset"
    ):
        TidalForcing(
            grid=grid,
            source={"name": "TPXO", "path": fname},
            ntides=10,
            use_dask=use_dask,
        )


def test_insufficient_number_of_consituents(grid_that_straddles_dateline, use_dask):

    fname = download_test_data("TPXO_global_test_data.nc")

    with pytest.raises(ValueError, match="The dataset contains fewer"):
        TidalForcing(
            grid=grid_that_straddles_dateline,
            source={"name": "TPXO", "path": fname},
            ntides=10,
            use_dask=use_dask,
        )


def test_tidal_forcing_plot_save(tidal_forcing, tmp_path):
    """
    Test plot and save methods in the same test since we dask arrays are already computed.
    """
    tidal_forcing.ds.load()

    tidal_forcing.plot(varname="ssh_Re", ntides=0)

    for file_str in ["test_tides", "test_tides.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            # Test saving without partitioning
            tidal_forcing.save(filepath)
            # Test saving with partitioning
            tidal_forcing.save(filepath, np_eta=3, np_xi=3)

            # Check if the .nc file was created
            filepath = Path(filepath)
            assert (filepath.with_suffix(".nc")).exists()
            # Clean up the .nc file
            (filepath.with_suffix(".nc")).unlink()

            filepath_str = str(filepath.with_suffix(""))
            expected_filepath_list = [
                (filepath_str + f".{index}.nc") for index in range(9)
            ]
            for expected_filepath in expected_filepath_list:
                assert Path(expected_filepath).exists()
                Path(expected_filepath).unlink()


def test_roundtrip_yaml(tidal_forcing, tmp_path, use_dask):
    """Test that creating a TidalForcing object, saving its parameters to yaml file, and re-opening yaml file creates the same object."""

    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str

        tidal_forcing.to_yaml(filepath)

        tidal_forcing_from_file = TidalForcing.from_yaml(filepath, use_dask=use_dask)

        assert tidal_forcing == tidal_forcing_from_file

        filepath = Path(filepath)
        filepath.unlink()


def test_files_have_same_hash(tidal_forcing, tmp_path, use_dask):

    yaml_filepath = tmp_path / "test_yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"

    tidal_forcing.to_yaml(yaml_filepath)
    tidal_forcing.save(filepath1)
    tidal_forcing_from_file = TidalForcing.from_yaml(yaml_filepath, use_dask=use_dask)
    tidal_forcing_from_file.save(filepath2)

    hash1 = calculate_file_hash(filepath1)
    hash2 = calculate_file_hash(filepath2)

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    filepath1.unlink()
    filepath2.unlink()


def test_from_yaml_missing_tidal_forcing(tmp_path, use_dask):
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

    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for yaml_filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str

        # Write YAML content to file
        if isinstance(yaml_filepath, Path):
            yaml_filepath.write_text(yaml_content)
        else:
            with open(yaml_filepath, "w") as f:
                f.write(yaml_content)

        with pytest.raises(
            ValueError, match="No TidalForcing configuration found in the YAML file."
        ):
            TidalForcing.from_yaml(yaml_filepath, use_dask=use_dask)

        yaml_filepath = Path(yaml_filepath)
        yaml_filepath.unlink()
