import pytest
from datetime import datetime
from roms_tools import BoundaryForcing
import tempfile
import os
import textwrap
from roms_tools.setup.download import download_test_data


def test_boundary_forcing_creation(boundary_forcing):
    """
    Test the creation of the BoundaryForcing object.
    """

    fname = download_test_data("GLORYS_coarse_test_data.nc")

    assert boundary_forcing.start_time == datetime(2021, 6, 29)
    assert boundary_forcing.end_time == datetime(2021, 6, 30)
    assert boundary_forcing.source == {
        "name": "GLORYS",
        "path": fname,
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

    assert len(boundary_forcing.ds.bry_time) == 1
    assert boundary_forcing.ds.coords["bry_time"].attrs["units"] == "days"
    assert not hasattr(boundary_forcing.ds, "climatology")


def test_boundary_forcing_creation_with_bgc(bgc_boundary_forcing_from_climatology):
    """
    Test the creation of the BoundaryForcing object.
    """

    fname_bgc = download_test_data("CESM_regional_coarse_test_data_climatology.nc")

    assert bgc_boundary_forcing_from_climatology.start_time == datetime(2021, 6, 29)
    assert bgc_boundary_forcing_from_climatology.end_time == datetime(2021, 6, 30)
    assert bgc_boundary_forcing_from_climatology.source == {
        "path": fname_bgc,
        "name": "CESM_REGRIDDED",
        "climatology": True,
    }
    assert bgc_boundary_forcing_from_climatology.model_reference_date == datetime(
        2000, 1, 1
    )
    assert bgc_boundary_forcing_from_climatology.boundaries == {
        "south": True,
        "east": True,
        "north": True,
        "west": True,
    }

    assert bgc_boundary_forcing_from_climatology.ds.source == "CESM_REGRIDDED"
    for direction in ["south", "east", "north", "west"]:
        for var in ["ALK", "PO4"]:
            assert f"{var}_{direction}" in bgc_boundary_forcing_from_climatology.ds

    assert len(bgc_boundary_forcing_from_climatology.ds.bry_time) == 12
    assert (
        bgc_boundary_forcing_from_climatology.ds.coords["bry_time"].attrs["units"]
        == "days"
    )
    assert hasattr(bgc_boundary_forcing_from_climatology.ds, "climatology")


def test_boundary_forcing_plot_save(
    boundary_forcing,
):
    """
    Test plot and save methods.
    """

    boundary_forcing.plot(varname="temp_south", layer_contours=True)
    boundary_forcing.plot(varname="temp_east", layer_contours=True)
    boundary_forcing.plot(varname="temp_north", layer_contours=True)
    boundary_forcing.plot(varname="temp_west", layer_contours=True)
    boundary_forcing.plot(varname="zeta_south")
    boundary_forcing.plot(varname="zeta_east")
    boundary_forcing.plot(varname="zeta_north")
    boundary_forcing.plot(varname="zeta_west")
    boundary_forcing.plot(varname="vbar_north")
    boundary_forcing.plot(varname="ubar_west")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        filepath = tmpfile.name

    boundary_forcing.save(filepath)
    extended_filepath = filepath + "_20210629-29.nc"

    try:
        assert os.path.exists(extended_filepath)
    finally:
        os.remove(extended_filepath)


def test_bgc_boundary_forcing_data_consistency_plot_save(
    bgc_boundary_forcing_from_climatology,
):
    """
    Test plot and save methods.
    """

    bgc_boundary_forcing_from_climatology.plot(varname="ALK_south")
    bgc_boundary_forcing_from_climatology.plot(varname="ALK_east")
    bgc_boundary_forcing_from_climatology.plot(varname="ALK_north")
    bgc_boundary_forcing_from_climatology.plot(varname="ALK_west")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        filepath = tmpfile.name

    bgc_boundary_forcing_from_climatology.save(filepath)
    extended_filepath = filepath + "_clim.nc"

    try:
        assert os.path.exists(extended_filepath)
    finally:
        os.remove(extended_filepath)


@pytest.mark.parametrize(
    "bdry_forcing_fixture",
    [
        "boundary_forcing",
        "bgc_boundary_forcing_from_climatology",
    ],
)
def test_roundtrip_yaml(bdry_forcing_fixture, request):
    """Test that creating a BoundaryForcing object, saving its parameters to yaml file, and re-opening yaml file creates the same object."""

    bdry_forcing = request.getfixturevalue(bdry_forcing_fixture)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filepath = tmpfile.name

    try:
        bdry_forcing.to_yaml(filepath)

        boundary_forcing_from_file = BoundaryForcing.from_yaml(filepath)

        assert bdry_forcing == boundary_forcing_from_file

    finally:
        os.remove(filepath)


def test_from_yaml_missing_boundary_forcing():
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
            ValueError, match="No BoundaryForcing configuration found in the YAML file."
        ):
            BoundaryForcing.from_yaml(yaml_filepath)
    finally:
        os.remove(yaml_filepath)