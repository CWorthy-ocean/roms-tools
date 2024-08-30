import pytest
import os
import shutil
import xarray as xr


# https://stackoverflow.com/questions/66970626/pytest-skip-test-condition-depending-on-environment
def requires_env(varname, value):
    env_value = os.environ.get(varname)
    return pytest.mark.skipif(
        not env_value == value,
        reason=f"Test skipped unless environment variable {varname}=={value}",
    )


def _get_fname(name):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, "test_data", f"{name}.zarr")


# this test will not be run by default
# to run it and overwrite the test data, invoke pytest with an environment variable as follows
# $ ROMS_TOOLS_OVERWRITE_TEST_DATA=1 pytest test_validation.py
@requires_env("ROMS_TOOLS_OVERWRITE_TEST_DATA", "1")
@pytest.mark.parametrize(
    "forcing_fixture, name",
    [
        ("simple_grid", "grid"),
        ("simple_grid_that_straddles_dateline", "grid_dateline"),
        ("tidal_forcing", "tides"),
        ("initial_conditions_with_bgc_from_climatology", "initial_conditions"),
        ("surface_forcing", "surface_forcing"),
        ("coarse_surface_forcing", "coarse_surface_forcing"),
        ("corrected_surface_forcing", "corrected_surface_forcing"),
        ("bgc_surface_forcing", "bgc_surface_forcing"),
        ("bgc_surface_forcing_from_climatology", "bgc_surface_forcing_from_clim"),
        ("boundary_forcing", "boundary_forcing"),
        ("bgc_boundary_forcing_from_climatology", "bgc_boundary_forcing_from_clim"),
    ],
)
def test_save_results(forcing_fixture, name, request):

    forcing = request.getfixturevalue(forcing_fixture)
    fname = _get_fname(name)

    # Check if the Zarr directory exists and delete it if it does
    if os.path.exists(fname):
        shutil.rmtree(fname)

    forcing.ds.to_zarr(fname)


@pytest.mark.parametrize(
    "forcing_fixture, name",
    [
        ("simple_grid", "grid"),
        ("simple_grid_that_straddles_dateline", "grid_dateline"),
        ("tidal_forcing", "tides"),
        ("initial_conditions_with_bgc_from_climatology", "initial_conditions"),
        ("surface_forcing", "surface_forcing"),
        ("coarse_surface_forcing", "coarse_surface_forcing"),
        ("corrected_surface_forcing", "corrected_surface_forcing"),
        ("bgc_surface_forcing", "bgc_surface_forcing"),
        ("bgc_surface_forcing_from_climatology", "bgc_surface_forcing_from_clim"),
        ("boundary_forcing", "boundary_forcing"),
        ("bgc_boundary_forcing_from_climatology", "bgc_boundary_forcing_from_clim"),
    ],
)
def test_check_results(forcing_fixture, name, request):

    forcing = request.getfixturevalue(forcing_fixture)

    fname = _get_fname(name)
    expected_forcing_ds = xr.open_zarr(fname, decode_timedelta=False)

    xr.testing.assert_allclose(forcing.ds, expected_forcing_ds)