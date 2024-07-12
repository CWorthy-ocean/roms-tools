import pytest
from datetime import datetime
from roms_tools import Grid, AtmosphericForcing
from roms_tools.setup.datasets import download_test_data

@pytest.fixture
def grid_that_straddles_dateline():
    """
    Fixture for creating a domain that straddles the dateline and lies within the bounds of the regional ERA5 data.
    """
    grid = Grid(
        nx=5,          
        ny=5,          
        size_x=1800,   
        size_y=2400,   
        center_lon=-10,
        center_lat=61, 
        rot=20,        
    )

    return grid

@pytest.fixture
def grid_that_straddles_dateline_but_is_too_big_for_regional_test_data():
    """
    Fixture for creating a domain that straddles the dateline but exceeds the bounds of the regional ERA5 data.
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
    """
    Fixture for creating a domain that straddles the dateline but exceeds the bounds of the regional ERA5 data.
    Centered west of dateline. This one was hard to catch for the nan_check for a long time, but should work now.
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
    """
    Fixture for creating a domain that lies east of Greenwich meridian, but less than 5 degrees away.
    We care about the 5 degree mark because it decides whether the code handles the longitudes as straddling the dateline or not.
    """

    grid = Grid(
        nx=5,          
        ny=5,          
        size_x=500,    
        size_y=2000,   
        center_lon=20, 
        center_lat=61, 
        rot=0,         
    )

    return grid

@pytest.fixture
def grid_that_lies_east_of_dateline_more_than_five_degrees_away():
    """
    Fixture for creating a domain that lies east of Greenwich meridian, more than 5 degrees away.
    We care about the 5 degree mark because it decides whether the code handles the longitudes as straddling the dateline or not.
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
    """
    Fixture for creating a domain that lies west of Greenwich meridian, less than 5 degrees away.
    We care about the 5 degree mark because it decides whether the code handles the longitudes as straddling the dateline or not.
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
    """
    Fixture for creating a domain that lies west of Greenwich meridian, more than 5 degrees away.
    We care about the 5 degree mark because it decides whether the code handles the longitudes as straddling the dateline or not.
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

@pytest.mark.parametrize("grid_fixture", [
    "grid_that_straddles_dateline",
    "grid_that_lies_east_of_dateline_less_than_five_degrees_away",
    "grid_that_lies_east_of_dateline_more_than_five_degrees_away",
    "grid_that_lies_west_of_dateline_less_than_five_degrees_away",
    "grid_that_lies_west_of_dateline_more_than_five_degrees_away"
    ])

def test_successful_initialization_with_regional_data(grid_fixture, request):
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")
    
    grid = request.getfixturevalue(grid_fixture)

    # Instantiate AtmosphericForcing object
    atm_forcing = AtmosphericForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        source="era5",
        filename=fname
    )

    assert atm_forcing.ds is not None

@pytest.mark.parametrize("grid_fixture", [
    "grid_that_straddles_dateline_but_is_too_big_for_regional_test_data",
    "another_grid_that_straddles_dateline_but_is_too_big_for_regional_test_data"
    ])

def test_unsuccessful_initialization_with_regional_data(grid_fixture, request):
    start_time = datetime(2020, 1, 31)
    end_time = datetime(2020, 2, 2)

    fname = download_test_data("ERA5_regional_test_data.nc")
    
    grid = request.getfixturevalue(grid_fixture)

    # Instantiate AtmosphericForcing object
    with pytest.raises(ValueError, match="NaN values found"):

        atm_forcing = AtmosphericForcing(
            grid=grid,
            start_time=start_time,
            end_time=end_time,
            source="era5",
            filename=fname
        )

