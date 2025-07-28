import logging
from datetime import datetime
from pathlib import Path

import pytest
import xarray as xr

from roms_tools import BoundaryForcing, Grid
from roms_tools.download import download_test_data
from roms_tools.setup.datasets import ERA5Correction
from roms_tools.setup.utils import interpolate_from_climatology, validate_names


def test_interpolate_from_climatology(use_dask):
    fname = download_test_data("ERA5_regional_test_data.nc")
    era5_times = xr.open_dataset(fname).time

    climatology = ERA5Correction(use_dask=use_dask)
    field = climatology.ds["ssr_corr"]
    field["time"] = field["time"].dt.days

    interpolated_field = interpolate_from_climatology(field, "time", era5_times)
    assert len(interpolated_field.time) == len(era5_times)


# Test yaml roundtrip with multiple source files
@pytest.fixture()
def boundary_forcing_from_multiple_source_files(request, use_dask):
    """Fixture for creating a BoundaryForcing object."""
    grid = Grid(
        nx=5,
        ny=5,
        size_x=100,
        size_y=100,
        center_lon=-8,
        center_lat=60,
        rot=10,
        N=3,  # number of vertical levels
    )

    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))

    return BoundaryForcing(
        grid=grid,
        start_time=datetime(2011, 1, 1),
        end_time=datetime(2013, 1, 1),
        source={"name": "GLORYS", "path": [fname1, fname2]},
        use_dask=use_dask,
    )


def test_roundtrip_yaml(
    boundary_forcing_from_multiple_source_files, request, tmp_path, use_dask
):
    """Test that creating a BoundaryForcing object, saving its parameters to yaml file,
    and re-opening yaml file creates the same object.
    """
    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str
        boundary_forcing_from_multiple_source_files.to_yaml(filepath)

        bdry_forcing_from_file = BoundaryForcing.from_yaml(filepath, use_dask=use_dask)

        assert boundary_forcing_from_multiple_source_files == bdry_forcing_from_file

        filepath = Path(filepath)
        filepath.unlink()


# test validate_names function

VALID_NAMES = ["a", "b", "c", "d"]
SENTINEL = "ALL"
MAX_TO_PLOT = 3


def test_valid_names_no_truncation():
    names = ["a", "b"]
    result = validate_names(names, VALID_NAMES, SENTINEL, MAX_TO_PLOT, label="test")
    assert result == names


def test_valid_names_with_truncation(caplog):
    names = ["a", "b", "c", "d"]
    with caplog.at_level(logging.WARNING):
        result = validate_names(
            names, VALID_NAMES, SENTINEL, max_to_plot=2, label="test"
        )
        assert result == ["a", "b"]
        assert "Only the first 2 tests will be plotted" in caplog.text


def test_include_all_sentinel():
    result = validate_names(SENTINEL, VALID_NAMES, SENTINEL, MAX_TO_PLOT, label="test")
    assert result == VALID_NAMES[:MAX_TO_PLOT]


def test_invalid_name_raises():
    with pytest.raises(ValueError, match="Invalid tests: z"):
        validate_names(["a", "z"], VALID_NAMES, SENTINEL, MAX_TO_PLOT, label="test")


def test_non_list_input_raises():
    with pytest.raises(ValueError, match="`test_names` should be a list of strings."):
        validate_names("a", VALID_NAMES, SENTINEL, MAX_TO_PLOT, label="test")


def test_non_string_elements_in_list_raises():
    with pytest.raises(
        ValueError, match="All elements in `test_names` must be strings."
    ):
        validate_names(["a", 2], VALID_NAMES, SENTINEL, MAX_TO_PLOT, label="test")


def test_custom_label_in_errors():
    with pytest.raises(ValueError, match="Invalid foozs: z"):
        validate_names(["z"], VALID_NAMES, SENTINEL, MAX_TO_PLOT, label="fooz")
