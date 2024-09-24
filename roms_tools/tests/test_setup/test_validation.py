import pytest
import os
import shutil
import xarray as xr


def _get_fname(name):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, "test_data", f"{name}.zarr")


@pytest.mark.parametrize(
    "forcing_fixture",
    [
        "grid",
        "grid_that_straddles_dateline",
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
# this test will not be run by default
# to run it and overwrite the test data, invoke pytest as follows
# pytest --overwrite=tidal_forcing --overwrite=boundary_forcing
def test_save_results(forcing_fixture, request):

    overwrite = request.config.getoption("--overwrite")

    # Skip the test if the fixture isn't marked for overwriting, unless 'all' is specified
    if "all" not in overwrite and forcing_fixture not in overwrite:
        pytest.skip(f"Skipping overwrite for {forcing_fixture}")

    forcing = request.getfixturevalue(forcing_fixture)
    fname = _get_fname(forcing_fixture)

    # Check if the Zarr directory exists and delete it if it does
    if os.path.exists(fname):
        shutil.rmtree(fname)

    forcing.ds.to_zarr(fname)


@pytest.mark.parametrize(
    "forcing_fixture",
    [
        "grid",
        "grid_that_straddles_dateline",
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
def test_check_results(forcing_fixture, request):

    fname = _get_fname(forcing_fixture)
    expected_forcing_ds = xr.open_zarr(fname, decode_timedelta=False)

    forcing = request.getfixturevalue(forcing_fixture)
    # TODO: Solve PyAMG reproducibility issue and lower rtol to 1.0e-5 again
    xr.testing.assert_allclose(forcing.ds, expected_forcing_ds, rtol=1.0e-2)
