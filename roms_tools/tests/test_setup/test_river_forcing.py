from roms_tools import RiverForcing
import xarray as xr
import textwrap
from pathlib import Path
import pytest
from conftest import calculate_file_hash


def test_successful_initialization_with_dai_data(river_forcing):

    assert isinstance(river_forcing.ds, xr.Dataset)
    assert "river_volume" in river_forcing.ds
    assert "river_tracer" in river_forcing.ds


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


def test_from_yaml_missing_initial_conditions(tmp_path):
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
