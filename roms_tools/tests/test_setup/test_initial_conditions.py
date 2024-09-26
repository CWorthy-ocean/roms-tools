import pytest
from datetime import datetime
from roms_tools import InitialConditions, Grid
import xarray as xr
import numpy as np
import textwrap
from roms_tools.setup.download import download_test_data
from roms_tools.setup.datasets import CESMBGCDataset
from pathlib import Path
from conftest import calculate_file_hash


@pytest.mark.parametrize(
    "ic_fixture",
    [
        "initial_conditions",
        "initial_conditions_with_bgc",
        "initial_conditions_with_bgc_from_climatology",
    ],
)
def test_initial_conditions_creation(ic_fixture, request):
    """
    Test the creation of the InitialConditions object.
    """

    ic = request.getfixturevalue(ic_fixture)

    assert ic.ini_time == datetime(2021, 6, 29)
    assert ic.source == {
        "name": "GLORYS",
        "path": download_test_data("GLORYS_coarse_test_data.nc"),
        "climatology": False,
    }
    assert isinstance(ic.ds, xr.Dataset)
    assert "temp" in ic.ds
    assert "salt" in ic.ds
    assert "u" in ic.ds
    assert "v" in ic.ds
    assert "zeta" in ic.ds
    assert ic.ds.coords["ocean_time"].attrs["units"] == "seconds"


@pytest.fixture
def example_grid():

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


# Test initialization with missing 'name' in source
def test_initial_conditions_missing_physics_name(example_grid, use_dask):
    with pytest.raises(ValueError, match="`source` must include a 'name'."):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"path": "physics_data.nc"},
            use_dask=use_dask,
        )


# Test initialization with missing 'path' in source
def test_initial_conditions_missing_physics_path(example_grid, use_dask):
    with pytest.raises(ValueError, match="`source` must include a 'path'."):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS"},
            use_dask=use_dask,
        )


# Test initialization with missing 'name' in bgc_source
def test_initial_conditions_missing_bgc_name(example_grid, use_dask):

    fname = download_test_data("GLORYS_coarse_test_data.nc")
    with pytest.raises(
        ValueError, match="`bgc_source` must include a 'name' if it is provided."
    ):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            bgc_source={"path": "bgc_data.nc"},
            use_dask=use_dask,
        )


# Test initialization with missing 'path' in bgc_source
def test_initial_conditions_missing_bgc_path(example_grid, use_dask):

    fname = download_test_data("GLORYS_coarse_test_data.nc")
    with pytest.raises(
        ValueError, match="`bgc_source` must include a 'path' if it is provided."
    ):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            bgc_source={"name": "CESM_REGRIDDED"},
            use_dask=use_dask,
        )


# Test default climatology value
def test_initial_conditions_default_climatology(example_grid, use_dask):

    fname = download_test_data("GLORYS_coarse_test_data.nc")

    initial_conditions = InitialConditions(
        grid=example_grid,
        ini_time=datetime(2021, 6, 29),
        source={"name": "GLORYS", "path": fname},
        use_dask=use_dask,
    )

    assert initial_conditions.source["climatology"] is False
    assert initial_conditions.bgc_source is None


def test_initial_conditions_default_bgc_climatology(example_grid, use_dask):

    fname = download_test_data("GLORYS_coarse_test_data.nc")
    fname_bgc = download_test_data("CESM_regional_test_data_one_time_slice.nc")

    initial_conditions = InitialConditions(
        grid=example_grid,
        ini_time=datetime(2021, 6, 29),
        source={"name": "GLORYS", "path": fname},
        bgc_source={"name": "CESM_REGRIDDED", "path": fname_bgc},
        use_dask=use_dask,
    )

    assert initial_conditions.bgc_source["climatology"] is True


def test_interpolation_from_climatology(
    initial_conditions_with_bgc_from_climatology, use_dask
):

    fname_bgc = download_test_data("CESM_regional_coarse_test_data_climatology.nc")
    ds = xr.open_dataset(fname_bgc)

    # check if interpolated value for Jan 15 is indeed January value from climatology
    bgc_data = CESMBGCDataset(
        filename=fname_bgc,
        start_time=datetime(2012, 1, 15),
        climatology=True,
        use_dask=use_dask,
        apply_post_processing=False,
    )
    assert np.allclose(ds["ALK"].sel(month=1), bgc_data.ds["ALK"], equal_nan=True)

    # check if interpolated value for Jan 30 is indeed average of January and February value from climatology
    bgc_data = CESMBGCDataset(
        filename=fname_bgc,
        start_time=datetime(2012, 1, 30),
        climatology=True,
        use_dask=use_dask,
        apply_post_processing=False,
    )
    assert np.allclose(
        0.5 * (ds["ALK"].sel(month=1) + ds["ALK"].sel(month=2)),
        bgc_data.ds["ALK"],
        equal_nan=True,
    )


def test_initial_conditions_plot_save(
    initial_conditions_with_bgc_from_climatology, tmp_path
):
    """
    Test plot and save methods.
    """

    initial_conditions_with_bgc_from_climatology.plot(varname="temp", s=0)
    initial_conditions_with_bgc_from_climatology.plot(
        varname="temp", s=0, depth_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(
        varname="temp", eta=0, layer_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(
        varname="temp", xi=0, layer_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", eta=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", s=0, xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", eta=0, xi=0)
    initial_conditions_with_bgc_from_climatology.plot(
        varname="u", s=0, layer_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(varname="u", s=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="u", eta=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="u", xi=0)
    initial_conditions_with_bgc_from_climatology.plot(
        varname="v", s=0, layer_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(varname="v", s=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="v", eta=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="v", xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="zeta")
    initial_conditions_with_bgc_from_climatology.plot(varname="ubar")
    initial_conditions_with_bgc_from_climatology.plot(varname="vbar")
    initial_conditions_with_bgc_from_climatology.plot(varname="ALK", s=0, xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="ALK", eta=0, xi=0)

    for file_str in ["test_ic", "test_ic.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            # Test saving without partitioning
            saved_filenames = initial_conditions_with_bgc_from_climatology.save(
                filepath
            )
            # Check if the .nc file was created
            filepath = Path(filepath).with_suffix(".nc")
            assert saved_filenames == [filepath]
            assert filepath.exists()
            # Clean up the .nc file
            filepath.unlink()

            # Test saving with partitioning
            saved_filenames = initial_conditions_with_bgc_from_climatology.save(
                filepath, np_eta=2
            )

            filepath_str = str(filepath.with_suffix(""))
            expected_filepath_list = [
                Path(filepath_str + f".{index}.nc") for index in range(2)
            ]
            assert saved_filenames == expected_filepath_list
            for expected_filepath in expected_filepath_list:
                assert expected_filepath.exists()
                expected_filepath.unlink()


def test_roundtrip_yaml(initial_conditions, tmp_path, use_dask):
    """Test that creating an InitialConditions object, saving its parameters to yaml file, and re-opening yaml file creates the same object."""

    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str

        initial_conditions.to_yaml(filepath)

        initial_conditions_from_file = InitialConditions.from_yaml(
            filepath, use_dask=use_dask
        )

        assert initial_conditions == initial_conditions_from_file

        filepath = Path(filepath)
        filepath.unlink()


def test_files_have_same_hash(initial_conditions, tmp_path, use_dask):

    yaml_filepath = tmp_path / "test_yaml.yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"

    initial_conditions.to_yaml(yaml_filepath)
    initial_conditions.save(filepath1)
    ic_from_file = InitialConditions.from_yaml(yaml_filepath, use_dask)
    ic_from_file.save(filepath2)

    hash1 = calculate_file_hash(filepath1)
    hash2 = calculate_file_hash(filepath2)

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    filepath1.unlink()
    filepath2.unlink()


def test_from_yaml_missing_initial_conditions(tmp_path, use_dask):
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
            ValueError,
            match="No InitialConditions configuration found in the YAML file.",
        ):
            InitialConditions.from_yaml(yaml_filepath, use_dask)

        yaml_filepath = Path(yaml_filepath)
        yaml_filepath.unlink()
