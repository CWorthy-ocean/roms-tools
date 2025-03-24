import pytest
from pathlib import Path
import xarray as xr
import os
import logging
from datetime import datetime
from roms_tools import Grid, ROMSOutput
from roms_tools.download import download_test_data

try:
    import xesmf  # type: ignore
except ImportError:
    xesmf = None


@pytest.fixture
def roms_output_from_restart_file(use_dask):

    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # Single file
    return ROMSOutput(
        grid=grid,
        path=Path(download_test_data("eastpac25km_rst.19980106000000.nc")),
        use_dask=use_dask,
    )


def test_load_model_output_file(roms_output_from_restart_file, use_dask):

    assert isinstance(roms_output_from_restart_file.ds, xr.Dataset)


def test_load_model_output_file_list(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # List of files
    file1 = Path(download_test_data("eastpac25km_rst.19980106000000.nc"))
    file2 = Path(download_test_data("eastpac25km_rst.19980126000000.nc"))
    output = ROMSOutput(grid=grid, path=[file1, file2], use_dask=use_dask)
    assert isinstance(output.ds, xr.Dataset)


def test_load_model_output_with_wildcard(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # Download at least two files, so these will be found within the pooch directory
    Path(download_test_data("eastpac25km_rst.19980106000000.nc"))
    Path(download_test_data("eastpac25km_rst.19980126000000.nc"))
    directory = Path(
        os.path.dirname(download_test_data("eastpac25km_rst.19980106000000.nc"))
    )

    output = ROMSOutput(grid=grid, path=directory / "*rst*.nc", use_dask=use_dask)
    assert isinstance(output.ds, xr.Dataset)


def test_invalid_path(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        ROMSOutput(
            grid=grid,
            path=Path("/path/to/nonexistent/file.nc"),
            use_dask=use_dask,
        )


def test_set_correct_model_reference_date(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    output = ROMSOutput(
        grid=grid,
        path=Path(download_test_data("eastpac25km_rst.19980106000000.nc")),
        use_dask=use_dask,
    )
    assert output.model_reference_date == datetime(1995, 1, 1)


def test_model_reference_date_mismatch(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # Create a ROMSOutput with a specified model_reference_date
    model_ref_date = datetime(2020, 1, 1)
    with pytest.raises(
        ValueError, match="Mismatch between `self.model_reference_date`"
    ):
        ROMSOutput(
            grid=grid,
            path=Path(download_test_data("eastpac25km_rst.19980106000000.nc")),
            model_reference_date=model_ref_date,
            use_dask=use_dask,
        )


def test_model_reference_date_no_metadata(use_dask, tmp_path, caplog):
    # Helper function to handle the test logic for cases where metadata is missing or invalid
    def test_no_metadata(faulty_ocean_time_attr, expected_exception, log_message=None):
        ds = xr.open_dataset(fname)
        ds["ocean_time"].attrs = faulty_ocean_time_attr

        # Write modified dataset to a new file
        fname_mod = tmp_path / "eastpac25km_rst.19980106000000_without_metadata.nc"
        ds.to_netcdf(fname_mod)

        # Test case 1: Expecting a ValueError when metadata is missing or invalid
        with pytest.raises(
            expected_exception,
            match="Model reference date could not be inferred from the metadata",
        ):
            ROMSOutput(grid=grid, path=fname_mod, use_dask=use_dask)

        # Test case 2: When a model reference date is explicitly set, verify the warning
        with caplog.at_level(logging.WARNING):
            ROMSOutput(
                grid=grid,
                path=fname_mod,
                model_reference_date=datetime(1995, 1, 1),
                use_dask=use_dask,
            )

            if log_message:
                # Verify the warning message in the log
                assert log_message in caplog.text

        fname_mod.unlink()

    # Load grid and test data
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)
    fname = download_test_data("eastpac25km_rst.19980106000000.nc")

    # Test 1: Ocean time attribute 'long_name' is missing
    test_no_metadata({}, ValueError)

    # Test 2: Ocean time attribute 'long_name' contains invalid information
    test_no_metadata(
        {"long_name": "some random text"},
        ValueError,
        "Could not infer the model reference date from the metadata.",
    )


def test_compute_depth_coordinates(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)
    fname_restart1 = Path(download_test_data("eastpac25km_rst.19980106000000.nc"))

    for adjust_depth_for_sea_surface_height in [True, False]:
        output = ROMSOutput(
            grid=grid,
            path=fname_restart1,
            use_dask=use_dask,
            adjust_depth_for_sea_surface_height=adjust_depth_for_sea_surface_height,
        )

        # Before calling get_vertical_coordinates, check if the dataset doesn't already have depth coordinates
        assert "layer_depth_rho" not in output.ds_depth_coords.data_vars

        # Call the method to get vertical coordinates
        output._get_depth_coordinates(depth_type="layer")

        # Check if the depth coordinates were added
        assert "layer_depth_rho" in output.ds_depth_coords.data_vars


def test_missing_zeta_gets_raised(use_dask):
    """Test that a ValueError is raised when `zeta` is missing from the dataset and
    `adjust_depth_for_sea_surface_height` is enabled."""
    # Load the grid
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # Load the ROMS output
    fname_restart1 = Path(download_test_data("eastpac25km_rst.19980106000000.nc"))
    roms_output = ROMSOutput(
        grid=grid,
        path=fname_restart1,
        use_dask=use_dask,
        adjust_depth_for_sea_surface_height=True,
    )

    # Remove `zeta` from the dataset
    object.__setattr__(
        roms_output, "ds", roms_output.ds.drop_vars("zeta", errors="ignore")
    )

    # Expect ValueError when calling `_get_depth_coordinates`
    with pytest.raises(
        ValueError,
        match="`zeta` is required in provided ROMS output when `adjust_depth_for_sea_surface_height` is enabled.",
    ):
        roms_output._get_depth_coordinates()


def test_check_vertical_coordinate_mismatch(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    fname_restart1 = Path(download_test_data("eastpac25km_rst.19980106000000.nc"))
    output = ROMSOutput(grid=grid, path=fname_restart1, use_dask=use_dask)

    # create a mock dataset with inconsistent vertical coordinate parameters
    ds_mock = output.ds.copy()

    # Modify one of the vertical coordinate attributes to cause a mismatch
    ds_mock.attrs["theta_s"] = 999

    # Check if ValueError is raised due to mismatch
    with pytest.raises(ValueError, match="theta_s from grid"):
        output._check_vertical_coordinate(ds_mock)

    # create a mock dataset with inconsistent vertical coordinate parameters
    ds_mock = output.ds.copy()

    # Modify one of the vertical coordinate attributes to cause a mismatch
    ds_mock.attrs["Cs_w"] = ds_mock.attrs["Cs_w"] + 0.01

    # Check if ValueError is raised due to mismatch
    with pytest.raises(ValueError, match="Cs_w from grid"):
        output._check_vertical_coordinate(ds_mock)


def test_that_coordinates_are_added(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    fname_restart1 = Path(download_test_data("eastpac25km_rst.19980106000000.nc"))
    output = ROMSOutput(grid=grid, path=fname_restart1, use_dask=use_dask)

    assert "abs_time" in output.ds.coords
    assert "lat_rho" in output.ds.coords
    assert "lon_rho" in output.ds.coords


def test_plot_on_native_model_grid(roms_output_from_restart_file, use_dask):

    for include_boundary in [False, True]:
        for depth_contours in [False, True]:

            # 3D fields
            for var_name in ["temp", "u", "v"]:
                kwargs = {
                    "include_boundary": include_boundary,
                    "depth_contours": depth_contours,
                }

                roms_output_from_restart_file.plot(var_name, time=1, s=-1, **kwargs)
                roms_output_from_restart_file.plot(var_name, time=1, eta=1, **kwargs)
                roms_output_from_restart_file.plot(var_name, time=1, xi=1, **kwargs)
                roms_output_from_restart_file.plot(
                    var_name,
                    time=1,
                    eta=1,
                    xi=1,
                    **kwargs,
                )
                roms_output_from_restart_file.plot(
                    var_name,
                    time=1,
                    s=-1,
                    eta=1,
                    **kwargs,
                )
                roms_output_from_restart_file.plot(
                    var_name,
                    time=1,
                    s=-1,
                    xi=1,
                    **kwargs,
                )

            # 2D fields
            roms_output_from_restart_file.plot("zeta", time=1, **kwargs)
            roms_output_from_restart_file.plot("zeta", time=1, eta=1, **kwargs)
            roms_output_from_restart_file.plot("zeta", time=1, xi=1, **kwargs)


@pytest.mark.skipif(xesmf is None, reason="xesmf required")
def test_plot_on_lat_lon(roms_output_from_restart_file, use_dask):

    for include_boundary in [False, True]:
        for depth_contours in [False, True]:

            # 3D fields
            for var_name in ["temp", "u", "v"]:
                kwargs = {
                    "include_boundary": include_boundary,
                    "depth_contours": depth_contours,
                }
                roms_output_from_restart_file.plot(
                    var_name,
                    time=1,
                    lat=9,
                    lon=-128,
                    **kwargs,
                )
                roms_output_from_restart_file.plot(
                    var_name,
                    time=1,
                    lat=9,
                    **kwargs,
                )
                roms_output_from_restart_file.plot(
                    var_name,
                    time=1,
                    lat=9,
                    s=-1,
                    **kwargs,
                )
                roms_output_from_restart_file.plot(
                    var_name,
                    time=1,
                    lon=-128,
                    **kwargs,
                )
                roms_output_from_restart_file.plot(
                    var_name,
                    time=1,
                    lon=-128,
                    s=-1,
                    **kwargs,
                )


def test_plot_errors(roms_output_from_restart_file, use_dask):
    """Test error conditions for the ROMSOutput.plot() method."""

    # Invalid time index
    with pytest.raises(ValueError, match="Invalid time index"):
        roms_output_from_restart_file.plot("temp", time=10, s=-1)

    # Conflicting inputs: Both 's' and 'depth' specified
    # TODO: Uncomment the following test once plotting and 'depth' is implemented
    # with pytest.raises(ValueError, match="Conflicting input: You cannot specify both 's' and 'depth' at the same time."):
    #     roms_output_from_restart_file.plot("temp", time=0, s=-1, depth=10)

    # Ambiguous input: Too many dimensions specified for 3D fields
    with pytest.raises(ValueError, match="Ambiguous input"):
        roms_output_from_restart_file.plot("temp", time=1, s=-1, eta=0, xi=0)

    # Vertical dimension specified for 2D fields
    with pytest.raises(
        ValueError, match="Vertical dimension 's' should be None for 2D fields"
    ):
        roms_output_from_restart_file.plot("zeta", time=1, s=-1)

    # Conflicting input: Both eta and xi specified for 2D fields
    with pytest.raises(
        ValueError,
        match="Conflicting input: For 2D fields, specify only one dimension, either 'eta' or 'xi', not both.",
    ):
        roms_output_from_restart_file.plot("zeta", time=1, eta=0, xi=0)

    # Conflicting input: lat or lon provided with eta or xi
    with pytest.raises(
        ValueError,
        match="Conflicting input: You cannot specify 'lat' or 'lon' simultaneously with 'eta' or 'xi'.",
    ):
        roms_output_from_restart_file.plot("temp", time=1, lat=10, lon=20, eta=5)

    # NotImplementedError: depth specified
    with pytest.raises(
        NotImplementedError,
        match="Plotting at a specific depth is not implemented yet.",
    ):
        roms_output_from_restart_file.plot("temp", time=1, depth=5)

    # Invalid eta index out of bounds
    with pytest.raises(ValueError, match="Invalid eta index"):
        roms_output_from_restart_file.plot("temp", time=1, eta=999)

    # Invalid xi index out of bounds
    with pytest.raises(ValueError, match="Invalid eta index"):
        roms_output_from_restart_file.plot("temp", time=1, xi=999)

    # Boundary exclusion error for eta
    with pytest.raises(ValueError, match="Invalid eta index.*boundary.*excluded"):
        roms_output_from_restart_file.plot(
            "temp", time=1, eta=0, include_boundary=False
        )

    # Boundary exclusion error for xi
    with pytest.raises(ValueError, match="Invalid xi index.*boundary.*excluded"):
        roms_output_from_restart_file.plot("temp", time=1, xi=0, include_boundary=False)

    # No dimension specified for 3D field
    with pytest.raises(
        ValueError,
        match="Invalid input: For 3D fields, you must specify at least one of the dimensions",
    ):
        roms_output_from_restart_file.plot("temp", time=1)
