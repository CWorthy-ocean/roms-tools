import pytest
from datetime import datetime
from roms_tools import InitialConditions, Grid
import xarray as xr
import numpy as np
import tempfile
import os
import textwrap
from roms_tools.setup.download import download_test_data
from roms_tools.setup.datasets import CESMBGCDataset


@pytest.fixture
def initial_conditions():
    """
    Fixture for creating an InitialConditions object.
    """

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

    fname = download_test_data("GLORYS_coarse_test_data.nc")

    return InitialConditions(
        grid=grid,
        ini_time=datetime(2021, 6, 29),
        source={"path": fname, "name": "GLORYS"},
    )


@pytest.fixture
def initial_conditions_with_bgc():
    """
    Fixture for creating an InitialConditions object.
    """

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

    fname = download_test_data("GLORYS_coarse_test_data.nc")
    fname_bgc = download_test_data("CESM_regional_test_data_one_time_slice.nc")

    return InitialConditions(
        grid=grid,
        ini_time=datetime(2021, 6, 29),
        source={"path": fname, "name": "GLORYS"},
        bgc_source={"path": fname_bgc, "name": "CESM_REGRIDDED"},
    )


@pytest.fixture
def initial_conditions_with_bgc_from_climatology():
    """
    Fixture for creating an InitialConditions object.
    """

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

    fname = download_test_data("GLORYS_coarse_test_data.nc")
    fname_bgc = download_test_data("CESM_regional_test_data_climatology.nc")

    return InitialConditions(
        grid=grid,
        ini_time=datetime(2021, 6, 29),
        source={"path": fname, "name": "GLORYS"},
        bgc_source={
            "path": fname_bgc,
            "name": "CESM_REGRIDDED",
            "climatology": True,
        },
    )


@pytest.mark.parametrize(
    "ic_fixture",
    [
        "initial_conditions",
        "initial_conditions_with_bgc",
        "initial_conditions_with_bgc_from_climatology",
    ],
)
def test_initial_conditions_creation(ic_fixture, request):
    """
    Test the creation of the InitialConditions object.
    """

    ic = request.getfixturevalue(ic_fixture)

    assert ic.ini_time == datetime(2021, 6, 29)
    assert ic.source == {
        "name": "GLORYS",
        "path": download_test_data("GLORYS_coarse_test_data.nc"),
        "climatology": False,
    }
    assert isinstance(ic.ds, xr.Dataset)
    assert "temp" in ic.ds
    assert "salt" in ic.ds
    assert "u" in ic.ds
    assert "v" in ic.ds
    assert "zeta" in ic.ds
    assert ic.ds.coords["ocean_time"].attrs["units"] == "seconds"


@pytest.fixture
def example_grid():

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

    return grid


# Test initialization with missing 'name' in source
def test_initial_conditions_missing_physics_name(example_grid):
    with pytest.raises(ValueError, match="`source` must include a 'name'."):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"path": "physics_data.nc"},
        )


# Test initialization with missing 'path' in source
def test_initial_conditions_missing_physics_path(example_grid):
    with pytest.raises(ValueError, match="`source` must include a 'path'."):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS"},
        )


# Test initialization with missing 'name' in bgc_source
def test_initial_conditions_missing_bgc_name(example_grid):

    fname = download_test_data("GLORYS_coarse_test_data.nc")
    with pytest.raises(
        ValueError, match="`bgc_source` must include a 'name' if it is provided."
    ):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            bgc_source={"path": "bgc_data.nc"},
        )


# Test initialization with missing 'path' in bgc_source
def test_initial_conditions_missing_bgc_path(example_grid):

    fname = download_test_data("GLORYS_coarse_test_data.nc")
    with pytest.raises(
        ValueError, match="`bgc_source` must include a 'path' if it is provided."
    ):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            bgc_source={"name": "CESM_REGRIDDED"},
        )


# Test default climatology value
def test_initial_conditions_default_climatology(example_grid):

    fname = download_test_data("GLORYS_coarse_test_data.nc")

    initial_conditions = InitialConditions(
        grid=example_grid,
        ini_time=datetime(2021, 6, 29),
        source={"name": "GLORYS", "path": fname},
    )

    assert initial_conditions.source["climatology"] is False
    assert initial_conditions.bgc_source is None


def test_initial_conditions_default_bgc_climatology(example_grid):

    fname = download_test_data("GLORYS_coarse_test_data.nc")
    fname_bgc = download_test_data("CESM_regional_test_data_one_time_slice.nc")

    initial_conditions = InitialConditions(
        grid=example_grid,
        ini_time=datetime(2021, 6, 29),
        source={"name": "GLORYS", "path": fname},
        bgc_source={"name": "CESM_REGRIDDED", "path": fname_bgc},
    )

    assert initial_conditions.bgc_source["climatology"] is True


def test_interpolation_from_climatology(initial_conditions_with_bgc_from_climatology):

    fname_bgc = download_test_data("CESM_regional_coarse_test_data_climatology.nc")
    ds = xr.open_dataset(fname_bgc)

    # check if interpolated value for Jan 15 is indeed January value from climatology
    bgc_data = CESMBGCDataset(
        filename=fname_bgc, start_time=datetime(2012, 1, 15), climatology=True
    )
    assert np.allclose(ds["ALK"].sel(month=1), bgc_data.ds["ALK"], equal_nan=True)

    # check if interpolated value for Jan 30 is indeed average of January and February value from climatology
    bgc_data = CESMBGCDataset(
        filename=fname_bgc, start_time=datetime(2012, 1, 30), climatology=True
    )
    assert np.allclose(
        0.5 * (ds["ALK"].sel(month=1) + ds["ALK"].sel(month=2)),
        bgc_data.ds["ALK"],
        equal_nan=True,
    )


def test_initial_conditions_plot_save(
    initial_conditions_with_bgc_from_climatology, tmp_path
):
    """
    Test plot and save methods.
    """

    initial_conditions_with_bgc_from_climatology.plot(varname="temp", s=0)
    initial_conditions_with_bgc_from_climatology.plot(
        varname="temp", s=0, depth_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(
        varname="temp", eta=0, layer_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(
        varname="temp", xi=0, layer_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", eta=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", s=0, xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="temp", eta=0, xi=0)
    initial_conditions_with_bgc_from_climatology.plot(
        varname="u", s=0, layer_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(varname="u", s=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="u", eta=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="u", xi=0)
    initial_conditions_with_bgc_from_climatology.plot(
        varname="v", s=0, layer_contours=True
    )
    initial_conditions_with_bgc_from_climatology.plot(varname="v", s=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="v", eta=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="v", xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="zeta")
    initial_conditions_with_bgc_from_climatology.plot(varname="ubar")
    initial_conditions_with_bgc_from_climatology.plot(varname="vbar")
    initial_conditions_with_bgc_from_climatology.plot(varname="ALK", s=0, xi=0)
    initial_conditions_with_bgc_from_climatology.plot(varname="ALK", eta=0, xi=0)

    filepath = tmp_path / "initial_conditions.nc"
    initial_conditions_with_bgc_from_climatology.save(filepath)
    assert filepath.exists()


def test_roundtrip_yaml(initial_conditions):
    """Test that creating an InitialConditions object, saving its parameters to yaml file, and re-opening yaml file creates the same object."""

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    try:
        initial_conditions.to_yaml(filepath)

        initial_conditions_from_file = InitialConditions.from_yaml(filepath)

        assert initial_conditions == initial_conditions_from_file

    finally:
        os.remove(filepath)


def test_from_yaml_missing_initial_conditions():
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

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        yaml_filepath = tmp_file.name
        tmp_file.write(yaml_content.encode())

    try:
        with pytest.raises(
            ValueError,
            match="No InitialConditions configuration found in the YAML file.",
        ):
            InitialConditions.from_yaml(yaml_filepath)
    finally:
        os.remove(yaml_filepath)
