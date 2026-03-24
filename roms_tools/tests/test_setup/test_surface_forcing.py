import logging
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from conftest import calculate_data_hash
from roms_tools import Grid, SurfaceForcing
from roms_tools.datasets.download import download_test_data
from roms_tools.setup.utils import RawDataSource


@pytest.fixture
def grid_that_straddles_dateline():
    """Fixture for creating a domain that straddles the dateline and lies within the
    bounds of the regional ERA5 data.
    """
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
    """Fixture for creating a domain that straddles the dateline but exceeds the bounds
    of the regional ERA5 data.

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
    """Fixture for creating a domain that straddles the dateline but exceeds the bounds
    of the regional ERA5 data.

    Centered west of dateline. This one was hard to catch for the nan_check for a long
    time, but should work now.
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
    """Fixture for creating a domain that lies east of Greenwich meridian, but less than
    5 degrees away.

    We care about the 5 degree mark because it decides whether the code handles the
    longitudes as straddling the dateline or not.
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
    """Fixture for creating a domain that lies east of Greenwich meridian, more than 5
    degrees away.

    We care about the 5 degree mark because it decides whether the code handles the
    longitudes as straddling the dateline or not.
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
    """Fixture for creating a domain that lies west of Greenwich meridian, less than 5
    degrees away.

    We care about the 5 degree mark because it decides whether the code handles the
    longitudes as straddling the dateline or not.
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
    """Fixture for creating a domain that lies west of Greenwich meridian, more than 5
    degrees away.

    We care about the 5 degree mark because it decides whether the code handles the
    longitudes as straddling the dateline or not.
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


def _test_successful_initialization(
    grid: Grid,
    start_time: datetime,
    end_time: datetime,
    source: RawDataSource,
    coarse_grid_mode: str,
    use_dask: bool,
    caplog,
):
    with caplog.at_level(logging.INFO):
        sfc_forcing = SurfaceForcing(
            grid=grid,
            start_time=start_time,
            end_time=end_time,
            source=source,
            correct_radiation=True,
            wind_dropoff=True,
            coarse_grid_mode=coarse_grid_mode,
            use_dask=use_dask,
        )

        assert sfc_forcing.grid == grid
        assert sfc_forcing.start_time == start_time
        assert sfc_forcing.end_time == end_time
        assert sfc_forcing.type == "physics"
        assert sfc_forcing.source == source
        assert sfc_forcing.correct_radiation
        assert sfc_forcing.wind_dropoff

        if coarse_grid_mode == "always":
            assert sfc_forcing.use_coarse_grid
            assert (
                "Data will be interpolated onto the grid coarsened by factor 2."
                in caplog.text
            )
        elif coarse_grid_mode == "never":
            assert not sfc_forcing.use_coarse_grid
            assert "Data will be interpolated onto the fine grid." in caplog.text

        assert isinstance(sfc_forcing.ds, xr.Dataset)
        assert "uwnd" in sfc_forcing.ds
        assert "vwnd" in sfc_forcing.ds
        assert "swrad" in sfc_forcing.ds
        assert "lwrad" in sfc_forcing.ds
        assert "Tair" in sfc_forcing.ds
        assert "qair" in sfc_forcing.ds
        assert "rain" in sfc_forcing.ds
        assert sfc_forcing.ds.coords["time"].attrs["units"] == "days"

        assert sfc_forcing.ds.attrs["source"] == source["name"]
        assert sfc_forcing.ds.attrs["correct_radiation"] == "True"
        assert sfc_forcing.ds.attrs["wind_dropoff"] == "True"


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
def test_successful_initialization_with_regional_data(
    grid_fixture, request, use_dask, caplog
):
    """Test the initialization of SurfaceForcing with regional ERA5 data.

    The test is performed twice:
    - First with the default fine grid.
    - Then with the coarse grid enabled.
    """
    fname = Path(download_test_data("ERA5_regional_test_data.nc"))
    grid = request.getfixturevalue(grid_fixture)

    for coarse_grid_mode in ["always", "never"]:
        _test_successful_initialization(
            grid=grid,
            start_time=datetime(2020, 1, 31),
            end_time=datetime(2020, 2, 2),
            source={"name": "ERA5", "path": fname, "climatology": False},
            coarse_grid_mode=coarse_grid_mode,
            use_dask=use_dask,
            caplog=caplog,
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
def test_successful_initialization_with_global_data(
    grid_fixture, request, use_dask, caplog
):
    """Test initialization of SurfaceForcing with global data.

    Verifies that the SurfaceForcing object is correctly initialized with global data,
    including the correct handling of the grid and physics data. Checks both coarse and
    fine grid initialization.
    """
    fname = Path(download_test_data("ERA5_global_test_data.nc"))
    grid = request.getfixturevalue(grid_fixture)

    for coarse_grid_mode in ["always", "never"]:
        _test_successful_initialization(
            grid=grid,
            start_time=datetime(2020, 1, 31),
            end_time=datetime(2020, 2, 2),
            source={"name": "ERA5", "path": fname, "climatology": False},
            coarse_grid_mode=coarse_grid_mode,
            use_dask=use_dask,
            caplog=caplog,
        )


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

    Ensures ValueError is raised if NaN values are detected in the dataset.
    """
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = Path(download_test_data("ERA5_regional_test_data.nc"))

    grid = request.getfixturevalue(grid_fixture)

    for coarse_grid_mode in ["always", "never"]:
        with pytest.raises(ValueError, match="NaN values found"):
            SurfaceForcing(
                grid=grid,
                coarse_grid_mode=coarse_grid_mode,
                start_time=start_time,
                end_time=end_time,
                source={"name": "ERA5", "path": fname},
                use_dask=use_dask,
            )


def test_no_longitude_intersection_initialization_with_regional_data(
    grid_that_straddles_180_degree_meridian, use_dask
):
    """Test initialization of SurfaceForcing with a grid that straddles the 180°
    meridian.

    Ensures ValueError is raised when the longitude range does not intersect with the
    dataset.
    """
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = Path(download_test_data("ERA5_regional_test_data.nc"))

    for coarse_grid_mode in ["always", "never"]:
        with pytest.raises(
            ValueError, match="Selected longitude range does not intersect with dataset"
        ):
            SurfaceForcing(
                grid=grid_that_straddles_180_degree_meridian,
                coarse_grid_mode=coarse_grid_mode,
                start_time=start_time,
                end_time=end_time,
                source={"name": "ERA5", "path": fname},
                use_dask=use_dask,
            )


def test_start_time_end_time_error(use_dask):
    """Test error when start_time and end_time are not both provided or both None."""
    # Case 1: Only start_time provided
    with pytest.raises(
        ValueError, match="Both `start_time` and `end_time` must be provided together"
    ):
        SurfaceForcing(
            grid=None,
            start_time=datetime(2022, 1, 1),
            end_time=None,  # end_time is None, should raise an error
            source={"name": "ERA5", "path": "era5_data.nc"},
            use_dask=use_dask,
        )

    # Case 2: Only end_time provided
    with pytest.raises(
        ValueError, match="Both `start_time` and `end_time` must be provided together"
    ):
        SurfaceForcing(
            grid=None,
            start_time=None,  # start_time is None, should raise an error
            end_time=datetime(2022, 1, 2),
            source={"name": "ERA5", "path": "era5_data.nc"},
            use_dask=use_dask,
        )


def test_start_time_end_time_warning(grid_that_straddles_dateline, use_dask, caplog):
    """Test that a warning is triggered when both start_time and end_time are None."""
    fname = Path(download_test_data("ERA5_regional_test_data.nc"))

    with caplog.at_level(logging.INFO):
        SurfaceForcing(
            grid=grid_that_straddles_dateline,
            start_time=None,
            end_time=None,
            source={"name": "ERA5", "path": fname},
            use_dask=use_dask,
        )

    # Verify the warning message in the log
    assert (
        "Both `start_time` and `end_time` are None. No time filtering will be applied to the source data."
        in caplog.text
    )


@pytest.mark.parametrize(
    "name, fname, type, climatology",
    [
        ("ERA5", "ERA5_regional_test_data.nc", "physics", False),
        ("CESM_REGRIDDED", "CESM_surface_global_test_data_climatology.nc", "bgc", True),
        ("UNIFIED", "coarsened_UNIFIED_bgc_dataset.nc", "bgc", True),
    ],
)
def test_nans_filled_in(
    grid_that_straddles_dateline, name, fname, type, climatology, use_dask
):
    """Test that the surface forcing fields contain no NaNs.

    The test is performed twice:
    - First with the default fine grid.
    - Then with the coarse grid enabled.
    """
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = Path(download_test_data(fname))
    print(fname)
    print(type)
    print(name)

    for coarse_grid_mode in ["always", "never"]:
        sfc_forcing = SurfaceForcing(
            grid=grid_that_straddles_dateline,
            coarse_grid_mode=coarse_grid_mode,
            start_time=start_time,
            end_time=end_time,
            source={"name": name, "path": fname, "climatology": climatology},
            type=type,
            use_dask=use_dask,
        )

        # Check that no NaNs are in surface forcing fields (they could make ROMS blow up)
        # Note that ROMS-Tools should replace NaNs with a fill value after the nan_check has successfully
        # completed; the nan_check passes if there are NaNs only over land
        for var in sfc_forcing.ds.data_vars:
            assert not sfc_forcing.ds[var].isnull().any().values.item()


@pytest.mark.parametrize(
    "bgc_surface_forcing_fixture",
    [
        "bgc_surface_forcing_from_climatology",
        "bgc_surface_forcing_from_unified_climatology",
    ],
)
def test_time_attr_climatology(bgc_surface_forcing_fixture, request):
    """Test that the 'cycle_length' attribute is present in the time coordinate of the
    BGC dataset when using climatology data.
    """
    bgc_surface_forcing = request.getfixturevalue(bgc_surface_forcing_fixture)
    for time_coord in ["pco2_time", "iron_time", "dust_time", "nox_time", "nhy_time"]:
        assert hasattr(
            bgc_surface_forcing.ds[time_coord],
            "cycle_length",
        )
    assert hasattr(bgc_surface_forcing.ds, "climatology")


def test_time_attr(bgc_surface_forcing):
    """Test that the 'cycle_length' attribute is not present in the time coordinate of
    the BGC dataset when not using climatology data.
    """
    for time_coord in ["pco2_time", "iron_time", "dust_time", "nox_time", "nhy_time"]:
        assert not hasattr(
            bgc_surface_forcing.ds[time_coord],
            "cycle_length",
        )
    assert not hasattr(bgc_surface_forcing.ds, "climatology")


@pytest.mark.parametrize(
    "sfc_forcing_fixture, expected_name, expected_climatology, expected_fname",
    [
        (
            "bgc_surface_forcing",
            "CESM_REGRIDDED",
            False,
            Path(download_test_data("CESM_surface_global_test_data.nc")),
        ),
        (
            "bgc_surface_forcing_from_climatology",
            "CESM_REGRIDDED",
            True,
            Path(download_test_data("CESM_surface_global_test_data_climatology.nc")),
        ),
        (
            "bgc_surface_forcing_from_unified_climatology",
            "UNIFIED",
            True,
            Path(download_test_data("coarsened_UNIFIED_bgc_dataset.nc")),
        ),
    ],
)
def test_surface_forcing_creation(
    sfc_forcing_fixture, expected_name, expected_climatology, expected_fname, request
):
    """Test the creation and initialization of the SurfaceForcing object with BGC.

    Verifies that the SurfaceForcing object is properly created with correct attributes.
    Ensures that expected variables are present in the dataset and that attributes match
    the given configurations.
    """
    sfc_forcing = request.getfixturevalue(sfc_forcing_fixture)

    assert sfc_forcing.ds is not None
    for var_name in ["pco2_air", "pco2_air_alt", "iron", "dust", "nox", "nhy"]:
        assert var_name in sfc_forcing.ds

    assert sfc_forcing.start_time == datetime(2020, 2, 1)
    assert sfc_forcing.end_time == datetime(2020, 2, 1)
    assert sfc_forcing.type == "bgc"
    assert sfc_forcing.source == {
        "name": expected_name,
        "path": expected_fname,
        "climatology": expected_climatology,
    }
    assert not sfc_forcing.use_coarse_grid
    assert sfc_forcing.ds.attrs["source"] == expected_name
    for time_coord in ["pco2_time", "iron_time", "dust_time", "nox_time", "nhy_time"]:
        assert sfc_forcing.ds.coords[time_coord].attrs["units"] == "days"


@pytest.mark.parametrize(
    "sfc_forcing_fixture",
    [
        "bgc_surface_forcing",
        "bgc_surface_forcing_from_climatology",
        "bgc_surface_forcing_from_unified_climatology",
    ],
)
def test_surface_forcing_pco2_replication(sfc_forcing_fixture, request):
    """Test whether pco2_air and pco2_air_alt is the same after processing."""
    sfc_forcing = request.getfixturevalue(sfc_forcing_fixture)

    xr.testing.assert_allclose(
        sfc_forcing.ds.pco2_air, sfc_forcing.ds.pco2_air_alt, rtol=1.0e-5
    )


def test_computed_missing_optional_fields(bgc_surface_forcing_from_unified_climatology):
    ds = bgc_surface_forcing_from_unified_climatology.ds

    # Use tight tolerances because 'nox' and 'nhy' can have values order 1e-12

    # 'nhy' was missing in the source data and should have been filled with a constant default value
    assert np.allclose(ds.nhy.std(), 0.0, rtol=1e-13, atol=1e-13), (
        "NHy should be constant across space and time"
    )
    # 'nox' was present in the source data and should show spatial or temporal variability
    assert ds.nox.std() > 1e-13, "NOx should vary across space and time"


def test_determine_usage_coarse_grid():
    # ERA5 data with 1/4 degree resolution spanning [-50E, 30E] and [40N, 80N]
    fname = download_test_data("ERA5_regional_test_data.nc")

    # at 50N, 1/4 degree of longitude is about 17.87 km; to automatically use the coarse grid, the ROMS grid needs to be of resolution < 17.87km / 2 = 8.9km
    grid_10km = Grid(
        nx=3, ny=3, size_x=30, size_y=30, center_lon=-10, center_lat=50, rot=0
    )
    surface_forcing = SurfaceForcing(
        grid=grid_10km,
        start_time=datetime(2020, 2, 1),
        end_time=datetime(2020, 2, 2),
        source={"name": "ERA5", "path": fname},
    )
    assert not surface_forcing.use_coarse_grid

    grid_7km = Grid(
        nx=3, ny=3, size_x=21, size_y=21, center_lon=-10, center_lat=50, rot=0
    )
    surface_forcing = SurfaceForcing(
        grid=grid_7km,
        start_time=datetime(2020, 2, 1),
        end_time=datetime(2020, 2, 2),
        source={"name": "ERA5", "path": fname},
    )
    assert surface_forcing.use_coarse_grid

    # at 70N, 1/4 degree of longitude is about 9.5 km; to automatically use the coarse grid, the ROMS grid needs to be of resolution < 9.5km / 2 = 4.75km
    grid_7km = Grid(
        nx=3, ny=3, size_x=21, size_y=21, center_lon=-10, center_lat=70, rot=0
    )
    surface_forcing = SurfaceForcing(
        grid=grid_7km,
        start_time=datetime(2020, 2, 1),
        end_time=datetime(2020, 2, 2),
        source={"name": "ERA5", "path": fname},
    )
    assert not surface_forcing.use_coarse_grid

    grid_4km = Grid(
        nx=3, ny=3, size_x=12, size_y=12, center_lon=-10, center_lat=70, rot=0
    )
    surface_forcing = SurfaceForcing(
        grid=grid_4km,
        start_time=datetime(2020, 2, 1),
        end_time=datetime(2020, 2, 2),
        source={"name": "ERA5", "path": fname},
    )
    assert surface_forcing.use_coarse_grid


def test_apply_wind_correction(surface_forcing):
    uwnd = surface_forcing.ds["uwnd"]
    vwnd = surface_forcing.ds["vwnd"]

    prev_coords = surface_forcing.target_coords.copy()
    uwnd_corr, vwnd_corr = surface_forcing._apply_wind_correction(uwnd, vwnd)

    assert isinstance(uwnd_corr, xr.DataArray)
    assert isinstance(vwnd_corr, xr.DataArray)

    # sanity check that the degrees conversion doesn't change the coords on our xarray
    assert prev_coords == surface_forcing.target_coords

    # Wind correction should not increase magnitude
    assert (abs(uwnd_corr) <= abs(uwnd)).all()
    assert (abs(vwnd_corr) <= abs(vwnd)).all()

    # Direction (sign) should be preserved
    assert (np.sign(uwnd_corr) == np.sign(uwnd)).all()
    assert (np.sign(vwnd_corr) == np.sign(vwnd)).all()

    # the ratio should be 1 far away, and 0.6 over/near land
    assert np.isclose((uwnd_corr / uwnd).max(), 1.0)
    assert np.isclose((vwnd_corr / vwnd).max(), 1.0)

    assert np.isclose((uwnd_corr / uwnd).min(), 0.6)
    assert np.isclose((vwnd_corr / vwnd).min(), 0.6)


@pytest.mark.parametrize(
    "sfc_forcing_fixture",
    [
        "surface_forcing",
        "corrected_surface_forcing",
        "surface_forcing_with_wind_dropoff",
        "coarse_surface_forcing",
    ],
)
def test_surface_forcing_plot(sfc_forcing_fixture, request, tmp_path):
    """Test save method."""
    sfc_forcing = request.getfixturevalue(sfc_forcing_fixture)
    sfc_forcing.plot(var_name="uwnd", time=0)


@pytest.mark.parametrize(
    "sfc_forcing_fixture",
    [
        "surface_forcing",
        "corrected_surface_forcing",
        "surface_forcing_with_wind_dropoff",
        "coarse_surface_forcing",
    ],
)
def test_surface_forcing_save(sfc_forcing_fixture, request, tmp_path):
    """Test save method."""
    sfc_forcing = request.getfixturevalue(sfc_forcing_fixture)

    for file_str in ["test_sf", "test_sf.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str
            # Test saving without grouping
            saved_filenames = sfc_forcing.save(filepath, group=False)
            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}.nc")
            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()

            # Test saving with grouping
            saved_filenames = sfc_forcing.save(filepath, group=True)
            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}_202002.nc")
            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()


@pytest.mark.parametrize(
    "sfc_forcing_fixture",
    [
        "bgc_surface_forcing",
        "bgc_surface_forcing_from_climatology",
        "bgc_surface_forcing_from_unified_climatology",
    ],
)
def test_surface_forcing_bgc_plot(sfc_forcing_fixture, request):
    """Test plot method."""
    sfc_forcing = request.getfixturevalue(sfc_forcing_fixture)
    sfc_forcing.plot(var_name="pco2_air", time=0)


def test_surface_forcing_bgc_save(bgc_surface_forcing, tmp_path):
    """Test save method."""
    for file_str in ["test_sf", "test_sf.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str
            # Test saving without grouping
            saved_filenames = bgc_surface_forcing.save(filepath, group=False)
            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}.nc")
            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()

            # Test saving with grouping
            saved_filenames = bgc_surface_forcing.save(filepath, group=True)
            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}_2020.nc")
            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()


@pytest.mark.parametrize(
    "sfc_forcing_fixture",
    [
        "bgc_surface_forcing_from_climatology",
        "bgc_surface_forcing_from_unified_climatology",
    ],
)
def test_surface_forcing_bgc_from_clim_save(sfc_forcing_fixture, tmp_path, request):
    """Test save method."""
    bgc_surface_forcing_from_climatology = request.getfixturevalue(sfc_forcing_fixture)

    for file_str in ["test_sf", "test_sf.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str
            # Test saving without grouping
            saved_filenames = bgc_surface_forcing_from_climatology.save(
                filepath, group=False
            )
            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}.nc")
            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()

            # Test saving with grouping
            saved_filenames = bgc_surface_forcing_from_climatology.save(
                filepath, group=True
            )
            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}_clim.nc")
            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()


@pytest.mark.parametrize(
    "sfc_forcing_fixture",
    [
        "surface_forcing",
        "coarse_surface_forcing",
        "corrected_surface_forcing",
        "surface_forcing_with_wind_dropoff",
        "bgc_surface_forcing",
        "bgc_surface_forcing_from_climatology",
        "bgc_surface_forcing_from_unified_climatology",
    ],
)
def test_roundtrip_yaml(sfc_forcing_fixture, request, tmp_path, use_dask):
    """Test that creating an SurfaceForcing object, saving its parameters to yaml file,
    and re-opening yaml file creates the same object.
    """
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


def assert_roundtrip_hash_equal(
    sfc_forcing,
    tmp_path,
    use_dask,
    expected_suffix,
):
    yaml_filepath = tmp_path / "test_yaml.yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"

    sfc_forcing.to_yaml(yaml_filepath)
    sfc_forcing.save(filepath1, group=True)

    sfc_forcing_from_file = SurfaceForcing.from_yaml(yaml_filepath, use_dask=use_dask)
    sfc_forcing_from_file.save(filepath2, group=True)

    base1 = str(Path(filepath1).with_suffix(""))
    base2 = str(Path(filepath2).with_suffix(""))
    expected_filepath1 = f"{base1}_{expected_suffix}.nc"
    expected_filepath2 = f"{base2}_{expected_suffix}.nc"

    hash1 = calculate_data_hash(expected_filepath1)
    hash2 = calculate_data_hash(expected_filepath2)

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    Path(expected_filepath1).unlink()
    Path(expected_filepath2).unlink()


@pytest.mark.parametrize(
    "sfc_forcing_fixture, expected_suffix",
    [
        (
            "surface_forcing",
            "202002",
        ),
        (
            "corrected_surface_forcing",
            "202002",
        ),
        (
            "surface_forcing_with_wind_dropoff",
            "202002",
        ),
        (
            "coarse_surface_forcing",
            "202002",
        ),
        (
            "bgc_surface_forcing",
            "2020",
        ),
        (
            "bgc_surface_forcing_from_climatology",
            "clim",
        ),
        (
            "bgc_surface_forcing_from_unified_climatology",
            "clim",
        ),
    ],
)
def test_files_have_same_hash(
    sfc_forcing_fixture,
    expected_suffix,
    request,
    tmp_path,
    use_dask,
):
    sfc_forcing = request.getfixturevalue(sfc_forcing_fixture)

    assert_roundtrip_hash_equal(
        sfc_forcing=sfc_forcing,
        tmp_path=tmp_path,
        use_dask=use_dask,
        expected_suffix=expected_suffix,
    )


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
            match="No SurfaceForcing configuration found in the YAML file.",
        ):
            SurfaceForcing.from_yaml(yaml_filepath, use_dask=use_dask)
        yaml_filepath = Path(yaml_filepath)
        yaml_filepath.unlink()


@pytest.mark.skip("Temporary skip until memory consumption issue is addressed. # TODO")
@pytest.mark.stream
@pytest.mark.use_dask
def test_surface_forcing_arco(surface_forcing_arco, tmp_path):
    """One big integration test for cloud-based ERA5 data because the streaming takes a
    long time.
    """
    # Test plotting
    surface_forcing_arco.plot(var_name="uwnd", time=0)

    # Roundtrip yaml
    yaml_filepath = tmp_path / "test_yaml.yaml"
    surface_forcing_arco.to_yaml(yaml_filepath)
    sfc_forcing_from_yaml = SurfaceForcing.from_yaml(yaml_filepath, use_dask=True)
    assert surface_forcing_arco == sfc_forcing_from_yaml

    # Compare hashes
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"
    surface_forcing_arco.save(filepath1, group=True)
    sfc_forcing_from_yaml.save(filepath2, group=True)
    filepath_str1 = str(Path(filepath1).with_suffix(""))
    filepath_str2 = str(Path(filepath2).with_suffix(""))
    expected_filepath1 = f"{filepath_str1}_202002.nc"
    expected_filepath2 = f"{filepath_str2}_202002.nc"
    hash1 = calculate_data_hash(expected_filepath1)
    hash2 = calculate_data_hash(expected_filepath2)
    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    Path(expected_filepath1).unlink()
    Path(expected_filepath2).unlink()


@pytest.mark.skip("Temporary skip until memory consumption issue is addressed. # TODO")
@pytest.mark.stream
@pytest.mark.use_dask
@pytest.mark.use_gcsfs
def test_default_era5_dataset_loading(small_grid: Grid) -> None:
    """Verify the default ERA5 dataset is loaded when a path is not provided."""
    start_time = datetime(2020, 2, 1)
    end_time = datetime(2020, 2, 2)

    sf = SurfaceForcing(
        grid=small_grid,
        source={"name": "ERA5"},
        type="physics",
        start_time=start_time,
        end_time=end_time,
        use_dask=True,
    )

    expected_vars = {"uwnd", "vwnd", "swrad", "lwrad", "Tair", "rain"}
    assert set(sf.ds.data_vars).issuperset(expected_vars)
