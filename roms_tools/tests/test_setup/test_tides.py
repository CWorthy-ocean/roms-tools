import pytest
from roms_tools import Grid, TidalForcing
import xarray as xr
from roms_tools.download import download_test_data
import textwrap
from pathlib import Path
from conftest import calculate_data_hash


@pytest.fixture(scope="session")
def tidal_forcing_9v5a(use_dask):

    grid = Grid(
        nx=3, ny=3, size_x=1500, size_y=1500, center_lon=235, center_lat=25, rot=-20
    )
    fname_grid = Path(download_test_data("regional_grid_tpxo9v5a.nc"))
    fname_h = Path(download_test_data("regional_h_tpxo9v5a.nc"))
    fname_u = Path(download_test_data("regional_u_tpxo9v5a.nc"))
    fname_dict = {"grid": fname_grid, "h": fname_h, "u": fname_u}

    return TidalForcing(
        grid=grid,
        source={"name": "TPXO", "path": fname_dict},
        ntides=1,
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def tidal_forcing_10v2a(use_dask):

    grid = Grid(
        nx=3, ny=3, size_x=1500, size_y=1500, center_lon=235, center_lat=25, rot=-20
    )
    fname_grid = Path(download_test_data("regional_grid_tpxo10v2a.nc"))
    fname_h = Path(download_test_data("regional_h_tpxo10v2a.nc"))
    fname_u = Path(download_test_data("regional_u_tpxo10v2a.nc"))
    fname_dict = {"grid": fname_grid, "h": fname_h, "u": fname_u}

    return TidalForcing(
        grid=grid,
        source={"name": "TPXO", "path": fname_dict},
        ntides=1,
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def tidal_forcing_from_global_data(use_dask):

    grid = Grid(
        nx=3, ny=3, size_x=1800, size_y=1500, center_lon=235, center_lat=25, rot=-20
    )

    fname_grid = Path(download_test_data("global_grid_tpxo10.v2.nc"))
    fname_h = Path(download_test_data("global_h_tpxo10.v2.nc"))
    fname_u = Path(download_test_data("global_u_tpxo10.v2.nc"))
    fname_dict = {"grid": fname_grid, "h": fname_h, "u": fname_u}

    return TidalForcing(
        grid=grid,
        source={"name": "TPXO", "path": fname_dict},
        ntides=1,
        use_dask=use_dask,
    )


@pytest.mark.parametrize(
    "tidal_forcing_fixture",
    ["tidal_forcing", "tidal_forcing_9v5a", "tidal_forcing_10v2a"],  # 10v2
)
def test_successful_initialization(tidal_forcing_fixture, request, use_dask):
    tidal_forcing = request.getfixturevalue(tidal_forcing_fixture)

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

    assert tidal_forcing.ntides == 1


@pytest.fixture
def grid_that_is_out_of_bounds_of_regional_tpxo_data():
    grid = Grid(
        nx=3, ny=3, size_x=1800, size_y=1500, center_lon=235, center_lat=25, rot=-20
    )
    return grid


@pytest.fixture
def grid_that_straddles_dateline():
    """Fixture for creating a domain that straddles the dateline."""
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
    """Fixture for creating a domain that straddles 180 degree meridian."""

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
        "grid_that_is_out_of_bounds_of_regional_tpxo_data",
        "grid_that_straddles_dateline",
        "grid_that_straddles_180_degree_meridian",
    ],
)
def test_successful_initialization_with_global_data(grid_fixture, request, use_dask):

    fname_grid = Path(download_test_data("global_grid_tpxo10.v2.nc"))
    fname_h = Path(download_test_data("global_h_tpxo10.v2.nc"))
    fname_u = Path(download_test_data("global_u_tpxo10.v2.nc"))
    fname_dict = {"grid": fname_grid, "h": fname_h, "u": fname_u}

    grid = request.getfixturevalue(grid_fixture)

    tidal_forcing = TidalForcing(
        grid=grid,
        source={"name": "TPXO", "path": fname_dict},
        ntides=1,
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

    assert tidal_forcing.ntides == 1


def test_unsuccessful_initialization_with_regional_data_due_to_nans(
    grid_that_is_out_of_bounds_of_regional_tpxo_data, use_dask
):

    fname_grid = Path(download_test_data("regional_grid_tpxo10v2.nc"))
    fname_h = Path(download_test_data("regional_h_tpxo10v2.nc"))
    fname_u = Path(download_test_data("regional_u_tpxo10v2.nc"))
    fname_dict = {"grid": fname_grid, "h": fname_h, "u": fname_u}

    with pytest.raises(ValueError, match="NaN values found"):
        TidalForcing(
            grid=grid_that_is_out_of_bounds_of_regional_tpxo_data,
            source={"name": "TPXO", "path": fname_dict},
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

    fname_grid = Path(download_test_data("regional_grid_tpxo10v2.nc"))
    fname_h = Path(download_test_data("regional_h_tpxo10v2.nc"))
    fname_u = Path(download_test_data("regional_u_tpxo10v2.nc"))
    fname_dict = {"grid": fname_grid, "h": fname_h, "u": fname_u}

    grid = request.getfixturevalue(grid_fixture)

    with pytest.raises(
        ValueError, match="Selected longitude range does not intersect with dataset"
    ):
        TidalForcing(
            grid=grid,
            source={"name": "TPXO", "path": fname_dict},
            ntides=10,
            use_dask=use_dask,
        )


def test_insufficient_number_of_consituents(grid_that_straddles_dateline, use_dask):

    fname_grid = Path(download_test_data("global_grid_tpxo10.v2.nc"))
    fname_h = Path(download_test_data("global_h_tpxo10.v2.nc"))
    fname_u = Path(download_test_data("global_u_tpxo10.v2.nc"))
    fname_dict = {"grid": fname_grid, "h": fname_h, "u": fname_u}

    with pytest.raises(ValueError, match="The dataset contains tidal"):
        TidalForcing(
            grid=grid_that_straddles_dateline,
            source={"name": "TPXO", "path": fname_dict},
            ntides=10,
            use_dask=use_dask,
        )


def test_tidal_forcing_plot(tidal_forcing):
    """Test plot method."""

    tidal_forcing.plot(var_name="ssh_Re", ntides=0)


def test_tidal_forcing_save(tidal_forcing, tmp_path):
    """Test save method."""

    for file_str in ["test_tides", "test_tides.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            saved_filenames = tidal_forcing.save(filepath)
            # Check if the .nc file was created
            filepath = Path(filepath).with_suffix(".nc")
            assert saved_filenames == [filepath]
            assert filepath.exists()
            # Clean up the .nc file
            filepath.unlink()


@pytest.mark.parametrize(
    "tidal_forcing_fixture",
    ["tidal_forcing", "tidal_forcing_from_global_data"],
)
def test_roundtrip_yaml(tidal_forcing_fixture, tmp_path, use_dask, request):
    """Test that creating a TidalForcing object, saving its parameters to yaml file, and
    re-opening yaml file creates the same object."""

    tidal_forcing = request.getfixturevalue(tidal_forcing_fixture)

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


@pytest.mark.parametrize(
    "tidal_forcing_fixture",
    ["tidal_forcing", "tidal_forcing_from_global_data"],
)
def test_files_have_same_hash(tidal_forcing_fixture, tmp_path, use_dask, request):

    tidal_forcing = request.getfixturevalue(tidal_forcing_fixture)

    yaml_filepath = tmp_path / "test_yaml.yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"

    tidal_forcing.to_yaml(yaml_filepath)
    tidal_forcing.save(filepath1)
    tidal_forcing_from_file = TidalForcing.from_yaml(yaml_filepath, use_dask=use_dask)
    tidal_forcing_from_file.save(filepath2)

    # Only compare hash of datasets because metadata is non-deterministic with dask
    hash1 = calculate_data_hash(filepath1)
    hash2 = calculate_data_hash(filepath2)

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
      topography_source:
        name: ETOPO5
      hmin: 5.0
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
