from roms_tools import RiverForcing, Grid
import xarray as xr
import numpy as np
from datetime import datetime
import textwrap
from pathlib import Path
import pytest
from conftest import calculate_file_hash


@pytest.fixture
def river_forcing_climatology():
    """Fixture for creating a RiverForcing object from the global Dai river dataset."""
    grid = Grid(
        nx=18, ny=18, size_x=800, size_y=800, center_lon=-18, center_lat=65, rot=20, N=3
    )

    start_time = datetime(1998, 1, 1)
    end_time = datetime(1998, 3, 1)

    return RiverForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        convert_to_climatology="always",
    )


def compare_dictionaries(dict1, dict2):
    assert dict1.keys() == dict2.keys()

    for key in dict1.keys():
        assert np.array_equal(dict1[key], dict2[key])


def test_successful_initialization_with_climatological_dai_data(river_forcing):

    assert isinstance(river_forcing.ds, xr.Dataset)
    assert river_forcing.climatology
    assert "river_volume" in river_forcing.ds
    assert "river_tracer" in river_forcing.ds


def test_successful_initialization_with_monthly_dai_data(river_forcing_no_climatology):

    assert isinstance(river_forcing_no_climatology.ds, xr.Dataset)
    assert not river_forcing_no_climatology.climatology
    assert "river_volume" in river_forcing_no_climatology.ds
    assert "river_tracer" in river_forcing_no_climatology.ds


def test_reproducibility(river_forcing, river_forcing_climatology):

    xr.testing.assert_allclose(river_forcing.ds, river_forcing_climatology.ds)

    compare_dictionaries(
        river_forcing.original_indices, river_forcing_climatology.original_indices
    )
    compare_dictionaries(
        river_forcing.updated_indices, river_forcing_climatology.updated_indices
    )


def test_reproducibility_indices(river_forcing, river_forcing_no_climatology):

    compare_dictionaries(
        river_forcing.original_indices, river_forcing_no_climatology.original_indices
    )
    compare_dictionaries(
        river_forcing.updated_indices, river_forcing_no_climatology.updated_indices
    )


@pytest.mark.parametrize(
    "river_forcing_fixture",
    ["river_forcing_climatology", "river_forcing_no_climatology"],
)
def test_constant_tracers(river_forcing_fixture, request):
    river_forcing = request.getfixturevalue(river_forcing_fixture)

    np.testing.assert_allclose(
        river_forcing.ds.river_tracer.isel(ntracers=0).values, 17.0, atol=0
    )
    np.testing.assert_allclose(
        river_forcing.ds.river_tracer.isel(ntracers=1).values, 1.0, atol=0
    )


@pytest.mark.parametrize(
    "river_forcing_fixture",
    ["river_forcing_climatology", "river_forcing_no_climatology"],
)
def test_river_locations_are_along_coast(river_forcing_fixture, request):
    river_forcing = request.getfixturevalue(river_forcing_fixture)

    mask = river_forcing.grid.ds.mask_rho
    faces = (
        mask.shift(eta_rho=1)
        + mask.shift(eta_rho=-1)
        + mask.shift(xi_rho=1)
        + mask.shift(xi_rho=-1)
    )
    coast = (1 - mask) * (faces > 0)

    indices = river_forcing.updated_indices
    for i in range(len(indices["station"])):
        assert coast[indices["eta_rho"][i], indices["xi_rho"][i]]


# @pytest.mark.parametrize(
#    "river_forcing_fixture",
#    [
#        "river_forcing_climatology",
#        "river_forcing_no_climatology"
#    ],
# )
# def test_river_flux_variable_grid_file(river_forcing_fixture, request):


def test_river_forcing_plot(river_forcing):
    """Test plot method."""

    river_forcing.plot_locations()
    river_forcing.plot("river_volume")
    river_forcing.plot("river_temperature")
    river_forcing.plot("river_salinity")


def test_river_forcing_save(river_forcing, tmp_path):
    """Test save method."""

    for file_str, grid_file_str in zip(
        ["test_rivers", "test_rivers.nc"], ["test_grid", "test_grid.nc"]
    ):
        # Create a temporary filepath using the tmp_path fixture
        for filepath, grid_filepath in zip(
            [tmp_path / file_str, str(tmp_path / file_str)],
            [tmp_path / grid_file_str, str(tmp_path / grid_file_str)],
        ):  # test for Path object and str

            # Test saving without partitioning
            saved_filenames = river_forcing.save(filepath, grid_filepath)
            # Check if the .nc file was created
            filepath = Path(filepath).with_suffix(".nc")
            grid_filepath = Path(grid_filepath).with_suffix(".nc")
            assert saved_filenames == [filepath, grid_filepath]
            assert filepath.exists()
            assert grid_filepath.exists()
            # Clean up the .nc file
            filepath.unlink()
            grid_filepath.unlink()

            # Test saving with partitioning
            saved_filenames = river_forcing.save(
                filepath, grid_filepath, np_eta=3, np_xi=3
            )

            filepath_str = str(filepath.with_suffix(""))
            grid_filepath_str = str(grid_filepath.with_suffix(""))
            expected_filepath_list = [
                Path(filepath_str + f".{index}.nc") for index in range(9)
            ] + [Path(grid_filepath_str + f".{index}.nc") for index in range(9)]
            assert saved_filenames == expected_filepath_list
            for expected_filepath in expected_filepath_list:
                assert expected_filepath.exists()
                expected_filepath.unlink()


def test_roundtrip_yaml(river_forcing, tmp_path):
    """Test that creating an RiverForcing object, saving its parameters to yaml file,
    and re-opening yaml file creates the same object."""

    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str

        river_forcing.to_yaml(filepath)

        river_forcing_from_file = RiverForcing.from_yaml(filepath)

        assert river_forcing == river_forcing_from_file

        filepath = Path(filepath)
        filepath.unlink()


def test_files_have_same_hash(river_forcing, tmp_path):

    yaml_filepath = tmp_path / "test_yaml.yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"
    grid_filepath1 = tmp_path / "grid_test1.nc"
    grid_filepath2 = tmp_path / "grid_test2.nc"

    river_forcing.to_yaml(yaml_filepath)
    river_forcing.save(filepath1, grid_filepath1)
    rf_from_file = RiverForcing.from_yaml(yaml_filepath)
    rf_from_file.save(filepath2, grid_filepath2)

    hash1 = calculate_file_hash(filepath1)
    hash2 = calculate_file_hash(filepath2)

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    filepath1.unlink()
    filepath2.unlink()
    grid_filepath1.unlink()
    grid_filepath2.unlink()


def test_from_yaml_missing_river_forcing(tmp_path):
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
            ValueError,
            match="No RiverForcing configuration found in the YAML file.",
        ):
            RiverForcing.from_yaml(yaml_filepath)

        yaml_filepath = Path(yaml_filepath)
        yaml_filepath.unlink()
