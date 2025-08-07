import logging
import os
import textwrap
from datetime import datetime
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from conftest import calculate_data_hash
from roms_tools import BoundaryForcing, Grid
from roms_tools.download import download_test_data
from roms_tools.utils import _has_dask


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "boundary_forcing",
        "boundary_forcing_adjusted_for_zeta",
        "boundary_forcing_with_2d_fill",
        "boundary_forcing_with_2d_fill_adjusted_for_zeta",
    ],
)
def test_boundary_forcing_creation(boundary_forcing_fixture, request):
    """Test the creation of the BoundaryForcing object."""
    boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))
    assert boundary_forcing.start_time == datetime(2012, 1, 1)
    assert boundary_forcing.end_time == datetime(2012, 12, 31)
    assert boundary_forcing.source == {
        "name": "GLORYS",
        "path": [fname1, fname2],
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

    assert len(boundary_forcing.ds.bry_time) == 2
    assert boundary_forcing.ds.coords["bry_time"].attrs["units"] == "days"
    assert not hasattr(boundary_forcing.ds, "climatology")
    assert hasattr(boundary_forcing.ds, "adjust_depth_for_sea_surface_height")
    assert hasattr(boundary_forcing.ds, "apply_2d_horizontal_fill")


def test_boundary_forcing_creation_with_duplicates(
    boundary_forcing: BoundaryForcing, use_dask: bool
) -> None:
    """Test the creation of the BoundaryForcing object with duplicates in source data
    works as expected.
    """
    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))

    boundary_forcing_with_duplicates_in_source_data = BoundaryForcing(
        grid=boundary_forcing.grid,
        start_time=boundary_forcing.start_time,
        end_time=boundary_forcing.end_time,
        source={"name": "GLORYS", "path": [fname1, fname1, fname2]},
        apply_2d_horizontal_fill=boundary_forcing.apply_2d_horizontal_fill,
        use_dask=use_dask,
    )

    assert boundary_forcing.ds.identical(
        boundary_forcing_with_duplicates_in_source_data.ds
    )


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "bgc_boundary_forcing_from_climatology",
        "bgc_boundary_forcing_from_unified_climatology",
    ],
)
def test_bgc_boundary_forcing_creation(boundary_forcing_fixture, request):
    """Test the creation of the BoundaryForcing object."""
    boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    assert boundary_forcing.start_time == datetime(2021, 6, 29)
    assert boundary_forcing.end_time == datetime(2021, 6, 30)
    assert boundary_forcing.source["climatology"]
    assert boundary_forcing.model_reference_date == datetime(2000, 1, 1)
    assert boundary_forcing.boundaries == {
        "south": True,
        "east": True,
        "north": True,
        "west": True,
    }

    expected_bgc_variables = [
        "PO4",
        "NO3",
        "SiO3",
        "NH4",
        "Fe",
        "Lig",
        "O2",
        "DIC",
        "DIC_ALT_CO2",
        "ALK",
        "ALK_ALT_CO2",
        "DOC",
        "DON",
        "DOP",
        "DOCr",
        "DONr",
        "DOPr",
        "zooC",
        "spChl",
        "spC",
        "spP",
        "spFe",
        "spCaCO3",
        "diatChl",
        "diatC",
        "diatP",
        "diatFe",
        "diatSi",
        "diazChl",
        "diazC",
        "diazP",
        "diazFe",
    ]

    for direction in ["south", "east", "north", "west"]:
        for var in expected_bgc_variables:
            assert f"{var}_{direction}" in boundary_forcing.ds

    assert len(boundary_forcing.ds.bry_time) == 12
    assert boundary_forcing.ds.coords["bry_time"].attrs["units"] == "days"
    assert hasattr(boundary_forcing.ds, "climatology")


def test_unsuccessful_boundary_forcing_creation_with_1d_fill(use_dask):
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

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))

    with pytest.raises(ValueError, match="consists entirely of NaNs"):
        BoundaryForcing(
            grid=grid,
            start_time=datetime(2021, 6, 29),
            end_time=datetime(2021, 6, 30),
            source={"name": "GLORYS", "path": fname},
            apply_2d_horizontal_fill=False,
            use_dask=use_dask,
        )

    fname_bgc = download_test_data("CESM_regional_coarse_test_data_climatology.nc")

    with pytest.raises(ValueError, match="consists entirely of NaNs"):
        BoundaryForcing(
            grid=grid,
            start_time=datetime(2021, 6, 29),
            end_time=datetime(2021, 6, 30),
            source={"path": fname_bgc, "name": "CESM_REGRIDDED", "climatology": True},
            type="bgc",
            apply_2d_horizontal_fill=False,
            use_dask=use_dask,
        )


def test_start_time_end_time_error(use_dask):
    """Test error when start_time and end_time are not both provided or both None."""
    # Case 1: Only start_time provided
    with pytest.raises(
        ValueError, match="Both `start_time` and `end_time` must be provided together"
    ):
        BoundaryForcing(
            grid=None,
            start_time=datetime(2022, 1, 1),
            end_time=None,  # end_time is None, should raise an error
            source={"name": "GLORYS", "path": "glorys_data.nc"},
            use_dask=use_dask,
        )

    # Case 2: Only end_time provided
    with pytest.raises(
        ValueError, match="Both `start_time` and `end_time` must be provided together"
    ):
        BoundaryForcing(
            grid=None,
            start_time=None,  # start_time is None, should raise an error
            end_time=datetime(2022, 1, 2),
            source={"name": "GLORYS", "path": "glorys_data.nc"},
            use_dask=use_dask,
        )


def test_start_time_end_time_warning(use_dask, caplog):
    """Test that a warning is triggered when both start_time and end_time are None."""
    # Catching the warning during test
    grid = Grid(
        nx=3,
        ny=3,
        size_x=400,
        size_y=400,
        center_lon=-8,
        center_lat=58,
        rot=0,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))

    with caplog.at_level(logging.INFO):
        BoundaryForcing(
            grid=grid,
            start_time=None,
            end_time=None,
            source={"name": "GLORYS", "path": [fname1, fname2]},
            use_dask=use_dask,
        )

    # Verify the warning message in the log
    assert (
        "Both `start_time` and `end_time` are None. No time filtering will be applied to the source data."
        in caplog.text
    )


def test_boundary_divided_by_land_warning(caplog, use_dask):
    # Iceland intersects the western boundary of the following grid
    grid = Grid(
        nx=5, ny=5, size_x=500, size_y=500, center_lon=-10, center_lat=65, rot=0
    )

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))

    with caplog.at_level(logging.WARNING):
        BoundaryForcing(
            grid=grid,
            start_time=datetime(2021, 6, 29),
            end_time=datetime(2021, 6, 30),
            source={"path": fname, "name": "GLORYS", "climatology": False},
            apply_2d_horizontal_fill=False,
            use_dask=use_dask,
        )
    # Verify the warning message in the log
    assert "the western boundary is divided by land" in caplog.text


def test_info_depth(caplog, use_dask):
    grid = Grid(
        nx=3,
        ny=3,
        size_x=400,
        size_y=400,
        center_lon=-8,
        center_lat=58,
        rot=0,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))

    with caplog.at_level(logging.INFO):
        BoundaryForcing(
            grid=grid,
            start_time=datetime(2012, 1, 1),
            end_time=datetime(2012, 12, 31),
            source={"name": "GLORYS", "path": [fname1, fname2]},
            adjust_depth_for_sea_surface_height=True,
            use_dask=use_dask,
        )

    # Verify the warning message in the log
    assert "Sea surface height will be used to adjust depth coordinates." in caplog.text

    # Clear the log before the next test
    caplog.clear()

    with caplog.at_level(logging.INFO):
        BoundaryForcing(
            grid=grid,
            start_time=datetime(2012, 1, 1),
            end_time=datetime(2012, 12, 31),
            source={"name": "GLORYS", "path": [fname1, fname2]},
            adjust_depth_for_sea_surface_height=False,
            use_dask=use_dask,
        )
    # Verify the warning message in the log
    assert (
        "Sea surface height will NOT be used to adjust depth coordinates."
        in caplog.text
    )


def test_info_fill(caplog, use_dask):
    grid = Grid(
        nx=3,
        ny=3,
        size_x=400,
        size_y=400,
        center_lon=-8,
        center_lat=58,
        rot=0,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))

    with caplog.at_level(logging.INFO):
        BoundaryForcing(
            grid=grid,
            start_time=datetime(2012, 1, 1),
            end_time=datetime(2012, 12, 31),
            source={"name": "GLORYS", "path": [fname1, fname2]},
            apply_2d_horizontal_fill=True,
            use_dask=use_dask,
        )

    # Verify the warning message in the log
    assert (
        "Applying 2D horizontal fill to the source data before regridding."
        in caplog.text
    )

    # Clear the log before the next test
    caplog.clear()

    with caplog.at_level(logging.INFO):
        BoundaryForcing(
            grid=grid,
            start_time=datetime(2012, 1, 1),
            end_time=datetime(2012, 12, 31),
            source={"name": "GLORYS", "path": [fname1, fname2]},
            apply_2d_horizontal_fill=False,
            use_dask=use_dask,
        )
    # Verify the warning message in the log
    for direction in ["south", "east", "north", "west"]:
        assert f"Applying 1D horizontal fill to {direction}ern boundary." in caplog.text


def test_1d_and_2d_fill_coincide_if_no_fill(use_dask):
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

    # this climatology has already filled land values and horizontal fill is skipped
    fname_bgc = Path(download_test_data("coarsened_UNIFIED_bgc_dataset.nc"))

    kwargs = {
        "grid": grid,
        "start_time": datetime(2021, 6, 29),
        "end_time": datetime(2021, 6, 29),
        "source": {"path": fname_bgc, "name": "UNIFIED", "climatology": True},
        "type": "bgc",
        "use_dask": use_dask,
    }

    bf_1d_fill = BoundaryForcing(
        **kwargs,
        apply_2d_horizontal_fill=False,
    )
    bf_2d_fill = BoundaryForcing(
        **kwargs,
        apply_2d_horizontal_fill=True,
    )

    xr.testing.assert_allclose(bf_1d_fill.ds, bf_2d_fill.ds, rtol=1.0e-4)


def test_1d_and_2d_fill_coincide_if_no_land(use_dask):
    # this grid lies entirely over open ocean
    grid = Grid(nx=5, ny=5, size_x=300, size_y=300, center_lon=-5, center_lat=65, rot=0)

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))

    kwargs = {
        "grid": grid,
        "start_time": datetime(2021, 6, 29),
        "end_time": datetime(2021, 6, 29),
        "source": {"path": fname, "name": "GLORYS", "climatology": False},
        "use_dask": use_dask,
    }

    bf_1d_fill = BoundaryForcing(
        **kwargs,
        apply_2d_horizontal_fill=False,
    )
    bf_2d_fill = BoundaryForcing(
        **kwargs,
        apply_2d_horizontal_fill=True,
    )

    xr.testing.assert_allclose(bf_1d_fill.ds, bf_2d_fill.ds, rtol=1.0e-4)


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "boundary_forcing_adjusted_for_zeta",
        "boundary_forcing_with_2d_fill_adjusted_for_zeta",
    ],
)
def test_correct_depth_coords_adjusted_for_zeta(
    boundary_forcing_fixture, request, use_dask
):
    boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    for direction in ["south", "east", "north", "west"]:
        # Test that uppermost interface coincides with sea surface height
        assert np.allclose(
            boundary_forcing.ds_depth_coords[f"interface_depth_rho_{direction}"]
            .isel(s_w=-1)
            .values,
            -boundary_forcing.ds[f"zeta_{direction}"].values,
            atol=1e-6,
        )


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "boundary_forcing",
        "boundary_forcing_with_2d_fill",
    ],
)
def test_correct_depth_coords_zero_zeta(boundary_forcing_fixture, request, use_dask):
    boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    for direction in ["south", "east", "north", "west"]:
        # Test that uppermost interface coincides with sea surface height
        assert np.allclose(
            boundary_forcing.ds_depth_coords[f"interface_depth_rho_{direction}"]
            .isel(s_w=-1)
            .values,
            0 * boundary_forcing.ds[f"zeta_{direction}"].values,
            atol=1e-6,
        )


def test_computed_missing_optional_fields(
    bgc_boundary_forcing_from_unified_climatology,
):
    ds = bgc_boundary_forcing_from_unified_climatology.ds

    # Use tight tolerances because 'DOC' and 'DOCr' can have values order 1e-6

    # 'DOCr' was missing in the source data and should have been filled with a constant default value
    for direction in ["south", "east", "north", "west"]:
        assert np.allclose(
            ds[f"DOCr_{direction}"].std(), 0.0, rtol=1e-10, atol=1e-10
        ), "DOCr should be constant across space and time"
    # 'DOC' was present in the source data and should show spatial or temporal variability
    for direction in ["south", "east", "north", "west"]:
        assert ds[f"DOC_{direction}"].std() > 1e-10, (
            "DOC should vary across space and time"
        )


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "boundary_forcing",
        "boundary_forcing_with_2d_fill",
        "boundary_forcing_adjusted_for_zeta",
        "boundary_forcing_with_2d_fill_adjusted_for_zeta",
    ],
)
def test_boundary_forcing_plot(boundary_forcing_fixture, request):
    """Test plot."""
    boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    for direction in ["south", "east", "north", "west"]:
        for layer_contours in [False, True]:
            boundary_forcing.plot(
                var_name=f"temp_{direction}", layer_contours=layer_contours
            )
            boundary_forcing.plot(
                var_name=f"u_{direction}", layer_contours=layer_contours
            )
            boundary_forcing.plot(
                var_name=f"v_{direction}", layer_contours=layer_contours
            )
        boundary_forcing.plot(var_name=f"zeta_{direction}")
        boundary_forcing.plot(var_name=f"vbar_{direction}")
        boundary_forcing.plot(var_name=f"ubar_{direction}")

    # Test that passing a matplotlib.axes.Axes works
    fig, ax = plt.subplots(1, 1)
    boundary_forcing.plot(var_name=f"temp_{direction}", ax=ax)
    boundary_forcing.plot(var_name=f"zeta_{direction}", ax=ax)


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "boundary_forcing",
        "boundary_forcing_with_2d_fill",
        "boundary_forcing_adjusted_for_zeta",
        "boundary_forcing_with_2d_fill_adjusted_for_zeta",
    ],
)
def test_boundary_forcing_save(boundary_forcing_fixture, request, tmp_path):
    """Test save method."""
    boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    for file_str in ["test_bf", "test_bf.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str
            # Test saving without grouping
            saved_filenames = boundary_forcing.save(filepath, group=False)

            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}.nc")

            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()

            # Test saving with grouping
            saved_filenames = boundary_forcing.save(filepath, group=True)

            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}_2012.nc")

            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "bgc_boundary_forcing_from_climatology",
        "bgc_boundary_forcing_from_unified_climatology",
    ],
)
def test_bgc_boundary_forcing_plot(boundary_forcing_fixture, request):
    """Test plot method."""
    bgc_boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    bgc_boundary_forcing.plot(var_name="ALK_south", layer_contours=True)
    bgc_boundary_forcing.plot(var_name="ALK_east", layer_contours=True)
    bgc_boundary_forcing.plot(var_name="ALK_north", layer_contours=True)
    bgc_boundary_forcing.plot(var_name="ALK_west", layer_contours=True)


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "bgc_boundary_forcing_from_climatology",
        "bgc_boundary_forcing_from_unified_climatology",
    ],
)
def test_bgc_boundary_forcing_save(boundary_forcing_fixture, tmp_path, request):
    """Test save method."""
    bgc_boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    for file_str in ["test_bf", "test_bf.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str
            # Test saving without partitioning and grouping
            saved_filenames = bgc_boundary_forcing.save(filepath, group=False)

            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}.nc")
            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()

            # Test saving without partitioning but with grouping
            saved_filenames = bgc_boundary_forcing.save(filepath, group=True)

            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}_clim.nc")
            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()


@pytest.mark.parametrize(
    "bdry_forcing_fixture",
    [
        "boundary_forcing",
        "bgc_boundary_forcing_from_climatology",
        "bgc_boundary_forcing_from_unified_climatology",
    ],
)
def test_roundtrip_yaml(bdry_forcing_fixture, request, tmp_path, use_dask):
    """Test that creating a BoundaryForcing object, saving its parameters to yaml file,
    and re-opening yaml file creates the same object.
    """
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
    yaml_filepath = tmp_path / "test_yaml_.yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"

    boundary_forcing.to_yaml(yaml_filepath)
    boundary_forcing.save(filepath1, group=True)
    bdry_forcing_from_file = BoundaryForcing.from_yaml(yaml_filepath, use_dask=use_dask)
    bdry_forcing_from_file.save(filepath2, group=True)

    filepath_str1 = str(Path(filepath1).with_suffix(""))
    filepath_str2 = str(Path(filepath2).with_suffix(""))
    expected_filepath1 = f"{filepath_str1}_2012.nc"
    expected_filepath2 = f"{filepath_str2}_2012.nc"

    # Only compare hash of datasets because metadata is non-deterministic with dask
    hash1 = calculate_data_hash(expected_filepath1)
    hash2 = calculate_data_hash(expected_filepath2)

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    Path(expected_filepath1).unlink()
    Path(expected_filepath2).unlink()


@pytest.mark.parametrize(
    "bdry_forcing_fixture",
    [
        "bgc_boundary_forcing_from_climatology",
        "bgc_boundary_forcing_from_unified_climatology",
    ],
)
def test_files_have_same_hash_clim(bdry_forcing_fixture, tmp_path, use_dask, request):
    bgc_boundary_forcing = request.getfixturevalue(bdry_forcing_fixture)

    yaml_filepath = tmp_path / "test_yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"

    bgc_boundary_forcing.to_yaml(yaml_filepath)
    bgc_boundary_forcing.save(filepath1, group=True)
    bdry_forcing_from_file = BoundaryForcing.from_yaml(yaml_filepath, use_dask=use_dask)
    bdry_forcing_from_file.save(filepath2, group=True)

    filepath_str1 = str(Path(filepath1).with_suffix(""))
    filepath_str2 = str(Path(filepath2).with_suffix(""))
    expected_filepath1 = f"{filepath_str1}_clim.nc"
    expected_filepath2 = f"{filepath_str2}_clim.nc"

    # Only compare hash of datasets because metadata is non-deterministic with dask
    hash1 = calculate_data_hash(expected_filepath1)
    hash2 = calculate_data_hash(expected_filepath2)

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    Path(expected_filepath1).unlink()
    Path(expected_filepath2).unlink()


def test_from_yaml_missing_boundary_forcing(tmp_path, use_dask):
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
            ValueError, match="No BoundaryForcing configuration found in the YAML file."
        ):
            BoundaryForcing.from_yaml(yaml_filepath, use_dask=use_dask)

        yaml_filepath = Path(yaml_filepath)
        yaml_filepath.unlink()


@pytest.mark.stream
@pytest.mark.skipif(
    not _has_dask(),
    reason="Executed only if Dask package is installed",
)
def test_default_glorys_dataset_loading(tiny_grid: Grid) -> None:
    """Verify the default ERA5 dataset is loaded when a path is not provided."""
    start_time = datetime(2010, 2, 1)
    end_time = datetime(2010, 2, 2)

    with mock.patch.dict(
        os.environ,
        {
            "COPERNICUSMARINE_SERVICE_USERNAME": "cmcbride",
            "PYDEVD_WARN_EVALUATION_TIMEOUT": "90",
        },
        clear=True,
    ):
        sf = BoundaryForcing(
            grid=tiny_grid,
            source={"name": "GLORYS"},
            type="physics",
            start_time=start_time,
            end_time=end_time,
            use_dask=True,
        )

        expected_vars = {"u_south", "v_south", "temp_south", "salt_south"}
        assert set(sf.ds.data_vars).issuperset(expected_vars)


@pytest.mark.skipif(
    not _has_dask(),
    reason="Executed only if Dask package is installed",
)
def test_nondefault_glorys_dataset_loading(small_grid: Grid) -> None:
    """Verify the default ERA5 dataset is loaded when a path is not provided."""
    start_time = datetime(2012, 1, 1)
    end_time = datetime(2012, 12, 31)

    local_path = Path(download_test_data("GLORYS_NA_20120101.nc"))

    with mock.patch.dict(
        os.environ,
        {
            "COPERNICUSMARINE_SERVICE_USERNAME": "cmcbride",
            "PYDEVD_WARN_EVALUATION_TIMEOUT": "90",
        },
        clear=True,
    ):
        bf = BoundaryForcing(
            grid=small_grid,
            source={
                "name": "GLORYS",
                "path": local_path,
            },
            type="physics",
            start_time=start_time,
            end_time=end_time,
            use_dask=True,
        )

        expected_vars = {"u_south", "v_south", "temp_south", "salt_south"}
        assert set(bf.ds.data_vars).issuperset(expected_vars)
