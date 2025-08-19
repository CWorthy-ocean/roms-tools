import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import xarray as xr


@pytest.mark.parametrize(
    "forcing_fixture",
    [
        "grid",
        "grid_that_straddles_dateline",
        "tidal_forcing",
        "initial_conditions_with_bgc_from_climatology",
        "initial_conditions_with_unified_bgc_from_climatology",
        "surface_forcing",
        "coarse_surface_forcing",
        "corrected_surface_forcing",
        "bgc_surface_forcing",
        "bgc_surface_forcing_from_climatology",
        "bgc_surface_forcing_from_unified_climatology",
        "boundary_forcing",
        "bgc_boundary_forcing_from_climatology",
        "bgc_boundary_forcing_from_unified_climatology",
        "river_forcing_with_bgc",
        "river_forcing_no_climatology",
    ],
)
# this test will not be run by default
# to run it and overwrite the test data, invoke pytest as follows
# pytest --overwrite=tidal_forcing --overwrite=boundary_forcing
def test_save_results(
    forcing_fixture,
    request: pytest.FixtureRequest,
    get_test_data_path: Callable[[str], Path],
) -> None:
    overwrite = request.config.getoption("--overwrite")

    # Skip the test if the fixture isn't marked for overwriting, unless 'all' is specified
    if "all" not in overwrite and forcing_fixture not in overwrite:
        pytest.skip(f"Skipping overwrite for {forcing_fixture}")

    forcing = request.getfixturevalue(forcing_fixture)
    fname = get_test_data_path(forcing_fixture)

    # Check if the Zarr directory exists and delete it if it does
    if fname.exists():
        shutil.rmtree(fname)

    forcing.ds.to_zarr(fname)


@pytest.mark.parametrize(
    "forcing_fixture",
    [
        "grid",
        "grid_that_straddles_dateline",
        "tidal_forcing",
        "initial_conditions_with_bgc_from_climatology",
        "initial_conditions_with_unified_bgc_from_climatology",
        "surface_forcing",
        "coarse_surface_forcing",
        "corrected_surface_forcing",
        "bgc_surface_forcing",
        "bgc_surface_forcing_from_climatology",
        "bgc_surface_forcing_from_unified_climatology",
        "boundary_forcing",
        "bgc_boundary_forcing_from_climatology",
        "bgc_boundary_forcing_from_unified_climatology",
        "river_forcing_with_bgc",
        "river_forcing_no_climatology",
    ],
)
def test_check_results(
    forcing_fixture,
    request: pytest.FixtureRequest,
    get_test_data_path: Callable[[str], Path],
) -> None:
    fname = get_test_data_path(forcing_fixture)
    expected_forcing_ds = xr.open_zarr(fname, decode_timedelta=False)
    forcing = request.getfixturevalue(forcing_fixture)

    # Set small tolerance because some fields like NOx, NHy have values order 1e-12
    xr.testing.assert_allclose(
        forcing.ds, expected_forcing_ds, rtol=1.0e-12, atol=1e-13
    )


@pytest.mark.use_dask
@pytest.mark.parametrize(
    "forcing_fixture",
    [
        "tidal_forcing",
        "initial_conditions_with_bgc_from_climatology",
        "surface_forcing",
        "coarse_surface_forcing",
        "corrected_surface_forcing",
        "bgc_surface_forcing",
        "bgc_surface_forcing_from_climatology",
        "boundary_forcing",
        "bgc_boundary_forcing_from_climatology",
    ],
)
def test_dask_vs_no_dask(
    forcing_fixture: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    """Test comparing the forcing created with and without Dask on same platform."""
    # Get the forcing with Dask
    forcing_with_dask = request.getfixturevalue(forcing_fixture)

    filepath = tmp_path / "test.yaml"
    forcing_with_dask.to_yaml(filepath)

    # Get the forcing without Dask
    forcing_without_dask = type(forcing_with_dask).from_yaml(filepath, use_dask=False)

    # Compare the two datasets using assert_equal (only within the same platform)
    xr.testing.assert_equal(forcing_with_dask.ds, forcing_without_dask.ds)
