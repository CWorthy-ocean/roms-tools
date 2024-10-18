import pytest
from datetime import datetime
from roms_tools import Grid, SurfaceForcing
from roms_tools.setup.download import download_test_data
import textwrap
from pathlib import Path
from conftest import calculate_file_hash


@pytest.fixture
def grid_that_straddles_dateline():
    """Fixture for creating a domain that straddles the dateline and lies
    within the bounds of the regional ERA5 data."""
    grid = Grid(
        nx=20,
        ny=20,
        size_x=1800,
        size_y=2400,
        center_lon=-10,
        center_lat=61,
        rot=20,
    )

    return grid


@pytest.fixture
def grid_that_straddles_dateline_but_is_too_big_for_regional_test_data():
    """Fixture for creating a domain that straddles the dateline but exceeds
    the bounds of the regional ERA5 data.

    Centered east of dateline.
    """
    grid = Grid(
        nx=5,
        ny=5,
        size_x=2000,
        size_y=2400,
        center_lon=10,
        center_lat=61,
        rot=20,
    )

    return grid


@pytest.fixture
def another_grid_that_straddles_dateline_but_is_too_big_for_regional_test_data():
    """Fixture for creating a domain that straddles the dateline but exceeds
    the bounds of the regional ERA5 data.

    Centered west of dateline. This one was hard to catch for the
    nan_check for a long time, but should work now.
    """
    grid = Grid(
        nx=5,
        ny=5,
        size_x=1950,
        size_y=2400,
        center_lon=-30,
        center_lat=61,
        rot=25,
    )

    return grid


@pytest.fixture
def grid_that_lies_east_of_dateline_less_than_five_degrees_away():
    """Fixture for creating a domain that lies east of Greenwich meridian, but
    less than 5 degrees away.

    We care about the 5 degree mark because it decides whether the code
    handles the longitudes as straddling the dateline or not.
    """

    grid = Grid(
        nx=5,
        ny=5,
        size_x=500,
        size_y=2000,
        center_lon=10,
        center_lat=61,
        rot=0,
    )

    return grid


@pytest.fixture
def grid_that_lies_east_of_dateline_more_than_five_degrees_away():
    """Fixture for creating a domain that lies east of Greenwich meridian, more
    than 5 degrees away.

    We care about the 5 degree mark because it decides whether the code
    handles the longitudes as straddling the dateline or not.
    """
    grid = Grid(
        nx=5,
        ny=5,
        size_x=500,
        size_y=2400,
        center_lon=15,
        center_lat=61,
        rot=0,
    )

    return grid


@pytest.fixture
def grid_that_lies_west_of_dateline_less_than_five_degrees_away():
    """Fixture for creating a domain that lies west of Greenwich meridian, less
    than 5 degrees away.

    We care about the 5 degree mark because it decides whether the code
    handles the longitudes as straddling the dateline or not.
    """

    grid = Grid(
        nx=5,
        ny=5,
        size_x=700,
        size_y=2400,
        center_lon=-15,
        center_lat=61,
        rot=0,
    )

    return grid


@pytest.fixture
def grid_that_lies_west_of_dateline_more_than_five_degrees_away():
    """Fixture for creating a domain that lies west of Greenwich meridian, more
    than 5 degrees away.

    We care about the 5 degree mark because it decides whether the code
    handles the longitudes as straddling the dateline or not.
    """

    grid = Grid(
        nx=5,
        ny=5,
        size_x=1000,
        size_y=2400,
        center_lon=-25,
        center_lat=61,
        rot=0,
    )

    return grid


@pytest.fixture
def grid_that_straddles_180_degree_meridian():
    """Fixture for creating a domain that straddles 180 degree meridian.

    This is a good test grid for the global ERA5 data, which comes on an
    [-180, 180] longitude grid.
    """

    grid = Grid(
        nx=5,
        ny=5,
        size_x=1800,
        size_y=2400,
        center_lon=180,
        center_lat=61,
        rot=20,
    )

    return grid


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid_that_straddles_dateline",
        "grid_that_lies_east_of_dateline_less_than_five_degrees_away",
        "grid_that_lies_east_of_dateline_more_than_five_degrees_away",
        "grid_that_lies_west_of_dateline_less_than_five_degrees_away",
        "grid_that_lies_west_of_dateline_more_than_five_degrees_away",
    ],
)
def test_successful_initialization_with_regional_data(grid_fixture, request, use_dask):
    """Test the initialization of SurfaceForcing with regional ERA5 data.

    This test checks the following:
    1. SurfaceForcing object initializes successfully with provided regional data.
    2. Attributes such as `start_time`, `end_time`, and `source` are set correctly.
    3. The dataset contains expected variables, including "uwnd", "vwnd", "swrad", "lwrad", "Tair", "qair", and "rain".
    4. Surface forcing plots for "uwnd", "vwnd", and "rain" are generated without errors.

    The test is performed twice:
    - First with the default fine grid.
    - Then with the coarse grid enabled.
    """

    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    for use_coarse_grid in [False, True]:
        sfc_forcing = SurfaceForcing(
            grid=grid,
            use_coarse_grid=use_coarse_grid,
            start_time=start_time,
            end_time=end_time,
            source={"name": "ERA5", "path": fname},
            use_dask=use_dask,
        )

        assert sfc_forcing.ds is not None
        assert "uwnd" in sfc_forcing.ds
        assert "vwnd" in sfc_forcing.ds
        assert "swrad" in sfc_forcing.ds
        assert "lwrad" in sfc_forcing.ds
        assert "Tair" in sfc_forcing.ds
        assert "qair" in sfc_forcing.ds
        assert "rain" in sfc_forcing.ds

        assert sfc_forcing.start_time == start_time
        assert sfc_forcing.end_time == end_time
        assert sfc_forcing.type == "physics"
        assert sfc_forcing.source == {
            "name": "ERA5",
            "path": fname,
            "climatology": False,
        }
        assert sfc_forcing.ds.coords["time"].attrs["units"] == "days"

        if use_coarse_grid:
            assert sfc_forcing.use_coarse_grid
        else:
            assert not sfc_forcing.use_coarse_grid

        sfc_forcing.plot("uwnd", time=0)
        sfc_forcing.plot("vwnd", time=0)
        sfc_forcing.plot("rain", time=0)


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
        "another_grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
    ],
)
def test_nan_detection_initialization_with_regional_data(
    grid_fixture, request, use_dask
):
    """Test handling of NaN values during initialization with regional data.

    Ensures ValueError is raised if NaN values are detected in the
    dataset.
    """
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    for use_coarse_grid in [True, False]:
        with pytest.raises(ValueError, match="NaN values found"):

            SurfaceForcing(
                grid=grid,
                use_coarse_grid=use_coarse_grid,
                start_time=start_time,
                end_time=end_time,
                source={"name": "ERA5", "path": fname},
                use_dask=use_dask,
            )


def test_no_longitude_intersection_initialization_with_regional_data(
    grid_that_straddles_180_degree_meridian, use_dask
):
    """Test initialization of SurfaceForcing with a grid that straddles the
    180° meridian.

    Ensures ValueError is raised when the longitude range does not
    intersect with the dataset.
    """
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")

    for use_coarse_grid in [True, False]:
        with pytest.raises(
            ValueError, match="Selected longitude range does not intersect with dataset"
        ):

            SurfaceForcing(
                grid=grid_that_straddles_180_degree_meridian,
                use_coarse_grid=use_coarse_grid,
                start_time=start_time,
                end_time=end_time,
                source={"name": "ERA5", "path": fname},
                use_dask=use_dask,
            )


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "grid_that_straddles_dateline",
        "grid_that_lies_east_of_dateline_less_than_five_degrees_away",
        "grid_that_lies_east_of_dateline_more_than_five_degrees_away",
        "grid_that_lies_west_of_dateline_less_than_five_degrees_away",
        "grid_that_lies_west_of_dateline_more_than_five_degrees_away",
        "grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
        "another_grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
        "grid_that_straddles_180_degree_meridian",
    ],
)
def test_successful_initialization_with_global_data(grid_fixture, request, use_dask):
    """Test initialization of SurfaceForcing with global data.

    Verifies that the SurfaceForcing object is correctly initialized
    with global data, including the correct handling of the grid and
    physics data. Checks both coarse and fine grid initialization.
    """
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_global_test_data.nc")

    grid = request.getfixturevalue(grid_fixture)

    for use_coarse_grid in [True, False]:
        sfc_forcing = SurfaceForcing(
            grid=grid,
            use_coarse_grid=use_coarse_grid,
            start_time=start_time,
            end_time=end_time,
            source={"name": "ERA5", "path": fname},
            use_dask=use_dask,
        )
        assert sfc_forcing.start_time == start_time
        assert sfc_forcing.end_time == end_time
        assert sfc_forcing.type == "physics"
        assert sfc_forcing.source == {
            "name": "ERA5",
            "path": fname,
            "climatology": False,
        }

        assert "uwnd" in sfc_forcing.ds
        assert "vwnd" in sfc_forcing.ds
        assert "swrad" in sfc_forcing.ds
        assert "lwrad" in sfc_forcing.ds
        assert "Tair" in sfc_forcing.ds
        assert "qair" in sfc_forcing.ds
        assert "rain" in sfc_forcing.ds
        assert sfc_forcing.ds.attrs["source"] == "ERA5"
        assert sfc_forcing.ds.coords["time"].attrs["units"] == "days"

        if use_coarse_grid:
            assert sfc_forcing.use_coarse_grid
        else:
            assert not sfc_forcing.use_coarse_grid


def test_nans_filled_in(grid_that_straddles_dateline, use_dask):
    """Test that the surface forcing fields contain no NaNs.

    The test is performed twice:
    - First with the default fine grid.
    - Then with the coarse grid enabled.
    """

    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")
    fname_bgc = download_test_data("CESM_surface_global_test_data_climatology.nc")

    for use_coarse_grid in [True, False]:
        sfc_forcing = SurfaceForcing(
            grid=grid_that_straddles_dateline,
            use_coarse_grid=use_coarse_grid,
            start_time=start_time,
            end_time=end_time,
            source={"name": "ERA5", "path": fname},
            use_dask=use_dask,
        )

        # Check that no NaNs are in surface forcing fields (they could make ROMS blow up)
        # Note that ROMS-Tools should replace NaNs with a fill value after the nan_check has successfully
        # completed; the nan_check passes if there are NaNs only over land
        assert not sfc_forcing.ds["uwnd"].isnull().any().values.item()
        assert not sfc_forcing.ds["vwnd"].isnull().any().values.item()
        assert not sfc_forcing.ds["rain"].isnull().any().values.item()

        sfc_forcing = SurfaceForcing(
            grid=grid_that_straddles_dateline,
            use_coarse_grid=use_coarse_grid,
            start_time=start_time,
            end_time=end_time,
            source={"name": "CESM_REGRIDDED", "path": fname_bgc, "climatology": True},
            type="bgc",
            use_dask=use_dask,
        )

        # Check that no NaNs are in surface forcing fields (they could make ROMS blow up)
        # Note that ROMS-Tools should replace NaNs with a fill value after the nan_check has successfully
        # completed; the nan_check passes if there are NaNs only over land
        assert not sfc_forcing.ds["pco2_air"].isnull().any().values.item()


def test_time_attr_climatology(bgc_surface_forcing_from_climatology):
    """Test that the 'cycle_length' attribute is present in the time coordinate
    of the BGC dataset when using climatology data."""
    for time_coord in ["pco2_time", "iron_time", "dust_time", "nox_time", "nhy_time"]:
        assert hasattr(
            bgc_surface_forcing_from_climatology.ds[time_coord],
            "cycle_length",
        )
    assert hasattr(bgc_surface_forcing_from_climatology.ds, "climatology")


def test_time_attr(bgc_surface_forcing):
    """Test that the 'cycle_length' attribute is not present in the time
    coordinate of the BGC dataset when not using climatology data."""
    for time_coord in ["pco2_time", "iron_time", "dust_time", "nox_time", "nhy_time"]:
        assert not hasattr(
            bgc_surface_forcing.ds[time_coord],
            "cycle_length",
        )
    assert not hasattr(bgc_surface_forcing.ds, "climatology")


@pytest.mark.parametrize(
    "sfc_forcing_fixture, expected_climatology, expected_fname",
    [
        (
            "bgc_surface_forcing",
            False,
            download_test_data("CESM_surface_global_test_data.nc"),
        ),
        (
            "bgc_surface_forcing_from_climatology",
            True,
            download_test_data("CESM_surface_global_test_data_climatology.nc"),
        ),
    ],
)
def test_surface_forcing_creation(
    sfc_forcing_fixture, expected_climatology, expected_fname, request
):
    """Test the creation and initialization of the SurfaceForcing object with
    BGC.

    Verifies that the SurfaceForcing object is properly created with
    correct attributes. Ensures that expected variables are present in
    the dataset and that attributes match the given configurations.
    """

    sfc_forcing = request.getfixturevalue(sfc_forcing_fixture)

    assert sfc_forcing.ds is not None
    assert "pco2_air" in sfc_forcing.ds
    assert "pco2_air_alt" in sfc_forcing.ds
    assert "iron" in sfc_forcing.ds
    assert "dust" in sfc_forcing.ds
    assert "nox" in sfc_forcing.ds
    assert "nhy" in sfc_forcing.ds

    assert sfc_forcing.start_time == datetime(2020, 2, 1)
    assert sfc_forcing.end_time == datetime(2020, 2, 1)
    assert sfc_forcing.type == "bgc"
    assert sfc_forcing.source == {
        "name": "CESM_REGRIDDED",
        "path": expected_fname,
        "climatology": expected_climatology,
    }
    assert not sfc_forcing.use_coarse_grid
    assert sfc_forcing.ds.attrs["source"] == "CESM_REGRIDDED"
    for time_coord in ["pco2_time", "iron_time", "dust_time", "nox_time", "nhy_time"]:
        assert sfc_forcing.ds.coords[time_coord].attrs["units"] == "days"

    sfc_forcing.plot("pco2_air", time=0)


@pytest.mark.parametrize(
    "sfc_forcing_fixture",
    [
        "surface_forcing",
        "corrected_surface_forcing",
        "coarse_surface_forcing",
    ],
)
def test_surface_forcing_plot_save(sfc_forcing_fixture, request, tmp_path):
    """Test plot and save methods."""
    sfc_forcing = request.getfixturevalue(sfc_forcing_fixture)
    sfc_forcing.plot(varname="uwnd", time=0)

    for file_str in ["test_sf", "test_sf.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            # Test saving without partitioning
            saved_filenames = sfc_forcing.save(filepath)
            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}_202002.nc")
            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()

            # Test saving with partitioning
            saved_filenames = sfc_forcing.save(filepath, np_eta=1)

            expected_filepath_list = [
                Path(filepath_str + f"_202002.{index}.nc") for index in range(1)
            ]
            assert saved_filenames == expected_filepath_list
            for expected_filepath in expected_filepath_list:
                assert expected_filepath.exists()
                expected_filepath.unlink()


def test_surface_forcing_bgc_plot_save(bgc_surface_forcing, tmp_path):
    """Test plot and save methods."""

    # Check the values in the dataset
    bgc_surface_forcing.plot(varname="pco2_air", time=0)

    for file_str in ["test_sf", "test_sf.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            # Test saving without partitioning
            saved_filenames = bgc_surface_forcing.save(filepath)
            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}_202002.nc")
            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()

            # Test saving with partitioning
            saved_filenames = bgc_surface_forcing.save(filepath, np_xi=5)

            expected_filepath_list = [
                Path(filepath_str + f"_202002.{index}.nc") for index in range(5)
            ]
            assert saved_filenames == expected_filepath_list
            for expected_filepath in expected_filepath_list:
                assert expected_filepath.exists()
                expected_filepath.unlink()


def test_surface_forcing_bgc_from_clim_plot_save(
    bgc_surface_forcing_from_climatology, tmp_path
):
    """Test plot and save methods."""

    # Check the values in the dataset
    bgc_surface_forcing_from_climatology.plot(varname="pco2_air", time=0)

    for file_str in ["test_sf", "test_sf.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            # Test saving without partitioning
            saved_filenames = bgc_surface_forcing_from_climatology.save(filepath)
            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}_clim.nc")
            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()

            # Test saving with partitioning
            saved_filenames = bgc_surface_forcing_from_climatology.save(
                filepath, np_eta=5
            )

            expected_filepath_list = [
                Path(filepath_str + f"_clim.{index}.nc") for index in range(5)
            ]
            assert saved_filenames == expected_filepath_list
            for expected_filepath in expected_filepath_list:
                assert expected_filepath.exists()
                expected_filepath.unlink()


@pytest.mark.parametrize(
    "sfc_forcing_fixture",
    [
        "surface_forcing",
        "coarse_surface_forcing",
        "corrected_surface_forcing",
        "bgc_surface_forcing",
        "bgc_surface_forcing_from_climatology",
    ],
)
def test_roundtrip_yaml(sfc_forcing_fixture, request, tmp_path, use_dask):
    """Test that creating an SurfaceForcing object, saving its parameters to
    yaml file, and re-opening yaml file creates the same object."""

    sfc_forcing = request.getfixturevalue(sfc_forcing_fixture)

    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str

        sfc_forcing.to_yaml(filepath)

        sfc_forcing_from_file = SurfaceForcing.from_yaml(filepath, use_dask)

        assert sfc_forcing == sfc_forcing_from_file

        filepath = Path(filepath)
        filepath.unlink()


@pytest.mark.parametrize(
    "sfc_forcing_fixture",
    [
        "surface_forcing",
        "corrected_surface_forcing",
        "coarse_surface_forcing",
        "bgc_surface_forcing",
    ],
)
def test_files_have_same_hash(sfc_forcing_fixture, request, tmp_path, use_dask):

    sfc_forcing = request.getfixturevalue(sfc_forcing_fixture)

    yaml_filepath = tmp_path / "test_yaml.yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"

    sfc_forcing.to_yaml(yaml_filepath)
    sfc_forcing.save(filepath1)
    sfc_forcing_from_file = SurfaceForcing.from_yaml(yaml_filepath, use_dask=use_dask)
    sfc_forcing_from_file.save(filepath2)

    filepath_str1 = str(Path(filepath1).with_suffix(""))
    filepath_str2 = str(Path(filepath2).with_suffix(""))
    expected_filepath1 = f"{filepath_str1}_202002.nc"
    expected_filepath2 = f"{filepath_str2}_202002.nc"

    hash1 = calculate_file_hash(expected_filepath1)
    hash2 = calculate_file_hash(expected_filepath2)

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    Path(expected_filepath1).unlink()
    Path(expected_filepath2).unlink()


def test_files_have_same_hash_clim(
    bgc_surface_forcing_from_climatology, tmp_path, use_dask
):

    yaml_filepath = tmp_path / "test_yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"

    bgc_surface_forcing_from_climatology.to_yaml(yaml_filepath)
    bgc_surface_forcing_from_climatology.save(filepath1)
    sfc_forcing_from_file = SurfaceForcing.from_yaml(yaml_filepath, use_dask=use_dask)
    sfc_forcing_from_file.save(filepath2)

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


def test_from_yaml_missing_surface_forcing(tmp_path, use_dask):
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
            match="No SurfaceForcing configuration found in the YAML file.",
        ):
            SurfaceForcing.from_yaml(yaml_filepath, use_dask=use_dask)
        yaml_filepath = Path(yaml_filepath)
        yaml_filepath.unlink()
