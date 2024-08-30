import pytest
import os
import xarray as xr
import zarr

from roms_tools.tests.test_setup.test_grid import simple_grid, simple_grid_that_straddles_dateline
from roms_tools.tests.test_setup.test_tides import tidal_forcing
from roms_tools.tests.test_setup.test_initial_conditions import initial_conditions_with_bgc_from_climatology
from roms_tools.tests.test_setup.test_boundary_forcing import boundary_forcing, bgc_boundary_forcing_from_climatology
from roms_tools.tests.test_setup.test_surface_forcing import surface_forcing, coarse_surface_forcing, corrected_surface_forcing, bgc_surface_forcing, bgc_surface_forcing_from_climatology


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
        ("bgc_boundary_forcing_from_climatology", "bgc_boundary_forcing_from_clim")
    ]
)
def test_save_results(forcing_fixture, name, request):

    forcing = request.getfixturevalue(forcing_fixture)
    fname = _get_fname(name)
    forcing.ds.to_zarr(fname, mode="w")

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
        ("bgc_boundary_forcing_from_climatology", "bgc_boundary_forcing_from_clim")
    ]
)
def test_check_results(forcing_fixture, name, request):
    
    forcing = request.getfixturevalue(forcing_fixture)
    forcing.ds.load()

    fname = _get_fname(name)
    expected_forcing_ds = xr.open_zarr(fname, decode_timedelta=False)

    xr.testing.assert_allclose(forcing.ds, expected_forcing_ds)

    

