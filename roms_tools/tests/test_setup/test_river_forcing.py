import xarray as xr

def test_successful_initialization_with_dai_data(river_forcing_from_dai):

    assert isinstance(river_forcing_from_dai.ds, xr.Dataset)
    assert "river_volume" in river_forcing_from_dai.ds
    assert "river_tracer" in river_forcing_from_dai.ds
