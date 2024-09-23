import pytest
from datetime import datetime
from roms_tools import BoundaryForcing
import textwrap
from roms_tools.setup.download import download_test_data
from conftest import calculate_file_hash
from pathlib import Path
import uuid


def test_boundary_forcing_creation(boundary_forcing):
    """
    Test the creation of the BoundaryForcing object.
    """

    fname = download_test_data("GLORYS_coarse_test_data.nc")

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
    assert boundary_forcing.ds.coords["bry_time"].attrs["units"] == "days"
    assert not hasattr(boundary_forcing.ds, "climatology")


def test_boundary_forcing_creation_with_bgc(bgc_boundary_forcing_from_climatology):
    """
    Test the creation of the BoundaryForcing object.
    """

    fname_bgc = download_test_data("CESM_regional_coarse_test_data_climatology.nc")

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
    assert (
        bgc_boundary_forcing_from_climatology.ds.coords["bry_time"].attrs["units"]
        == "days"
    )
    assert hasattr(bgc_boundary_forcing_from_climatology.ds, "climatology")


def test_boundary_forcing_plot_save(boundary_forcing, tmp_path):
    """
    Test plot and save methods.
    """

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

    for file_str in ["test_bf", "test_bf.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            # Test saving without partitioning
            saved_filenames = boundary_forcing.save(filepath)

            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}_202106.nc")

            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()

            # Test saving with partitioning
            saved_filenames = boundary_forcing.save(filepath, np_eta=2)
            expected_filepath_list = [
                Path(filepath_str + f"_202106.{index}.nc") for index in range(2)
            ]

            assert saved_filenames == expected_filepath_list

            for expected_filepath in expected_filepath_list:
                assert expected_filepath.exists()
                expected_filepath.unlink()


def test_bgc_boundary_forcing_plot_save(
    bgc_boundary_forcing_from_climatology, tmp_path
):
    """
    Test plot and save methods.
    """

    bgc_boundary_forcing_from_climatology.plot(varname="ALK_south")
    bgc_boundary_forcing_from_climatology.plot(varname="ALK_east")
    bgc_boundary_forcing_from_climatology.plot(varname="ALK_north")
    bgc_boundary_forcing_from_climatology.plot(varname="ALK_west")

    for file_str in ["test_bf", "test_bf.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            # Test saving without partitioning
            saved_filenames = bgc_boundary_forcing_from_climatology.save(filepath)

            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}_clim.nc")
            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()

            # Test saving with partitioning
            saved_filenames = bgc_boundary_forcing_from_climatology.save(
                filepath, np_xi=2
            )

            expected_filepath_list = [
                Path(filepath_str + f"_clim.{index}.nc") for index in range(2)
            ]
            assert saved_filenames == expected_filepath_list

            for expected_filepath in expected_filepath_list:
                assert expected_filepath.exists()
                expected_filepath.unlink()


@pytest.mark.parametrize(
    "bdry_forcing_fixture",
    [
        "boundary_forcing",
        "bgc_boundary_forcing_from_climatology",
    ],
)
def test_roundtrip_yaml(bdry_forcing_fixture, request, tmp_path, use_dask):
    """Test that creating a BoundaryForcing object, saving its parameters to yaml file, and re-opening yaml file creates the same object."""

    bdry_forcing = request.getfixturevalue(bdry_forcing_fixture)

    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str

        bdry_forcing.to_yaml(filepath)

        bdry_forcing_from_file = BoundaryForcing.from_yaml(filepath, use_dask=use_dask)

        assert bdry_forcing == bdry_forcing_from_file

        filepath = Path(filepath)
        filepath.unlink()


def test_files_have_same_hash(boundary_forcing, tmp_path, use_dask):

    unique_id = uuid.uuid4()  # Generate a unique identifier
    yaml_filepath = tmp_path / f"test_yaml_{unique_id}.yaml"
    filepath1 = tmp_path / f"test1_{unique_id}.nc"
    filepath2 = tmp_path / f"test2_{unique_id}.nc"

    boundary_forcing.to_yaml(yaml_filepath)
    boundary_forcing.save(filepath1)
    bdry_forcing_from_file = BoundaryForcing.from_yaml(yaml_filepath, use_dask=use_dask)
    bdry_forcing_from_file.save(filepath2)

    filepath_str1 = str(Path(filepath1).with_suffix(""))
    filepath_str2 = str(Path(filepath2).with_suffix(""))
    expected_filepath1 = f"{filepath_str1}_202106.nc"
    expected_filepath2 = f"{filepath_str2}_202106.nc"

    hash1 = calculate_file_hash(expected_filepath1)
    hash2 = calculate_file_hash(expected_filepath2)

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    Path(expected_filepath1).unlink()
    Path(expected_filepath2).unlink()


def test_files_have_same_hash_clim(
    bgc_boundary_forcing_from_climatology, tmp_path, use_dask
):

    yaml_filepath = tmp_path / "test_yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"

    bgc_boundary_forcing_from_climatology.to_yaml(yaml_filepath)
    bgc_boundary_forcing_from_climatology.save(filepath1)
    bdry_forcing_from_file = BoundaryForcing.from_yaml(yaml_filepath, use_dask=use_dask)
    bdry_forcing_from_file.save(filepath2)

    filepath_str1 = str(Path(filepath1).with_suffix(""))
    filepath_str2 = str(Path(filepath2).with_suffix(""))
    expected_filepath1 = f"{filepath_str1}_clim.nc"
    expected_filepath2 = f"{filepath_str2}_clim.nc"

    hash1 = calculate_file_hash(expected_filepath1)
    hash2 = calculate_file_hash(expected_filepath2)

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    Path(expected_filepath1).unlink()
    Path(expected_filepath2).unlink()


def test_from_yaml_missing_boundary_forcing(tmp_path, request, use_dask):
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
            ValueError, match="No BoundaryForcing configuration found in the YAML file."
        ):
            BoundaryForcing.from_yaml(yaml_filepath, use_dask=use_dask)

        yaml_filepath = Path(yaml_filepath)
        yaml_filepath.unlink()
