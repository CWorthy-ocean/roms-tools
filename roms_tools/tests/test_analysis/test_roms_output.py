from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

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


@pytest.fixture
def roms_output_from_restart_file_adjusted_for_zeta(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # Single file
    return ROMSOutput(
        grid=grid,
        path=Path(download_test_data("eastpac25km_rst.19980106000000.nc")),
        adjust_depth_for_sea_surface_height=True,
        use_dask=use_dask,
    )


@pytest.fixture
def roms_output_from_restart_file_with_straddling_grid(use_dask):
    # Make fake grid that straddles the dateline and that has consistent sizes with test data below
    grid = Grid(
        nx=8, ny=13, center_lon=0, center_lat=60, rot=32, size_x=244, size_y=365
    )

    return ROMSOutput(
        grid=grid,
        path=Path(download_test_data("eastpac25km_rst.19980106000000.nc")),
        use_dask=use_dask,
    )


@pytest.fixture
def roms_output_from_two_restart_files(use_dask):
    fname_grid = Path(download_test_data("epac25km_grd.nc"))
    grid = Grid.from_file(fname_grid)

    # List of files
    file1 = Path(download_test_data("eastpac25km_rst.19980106000000.nc"))
    file2 = Path(download_test_data("eastpac25km_rst.19980126000000.nc"))
    return ROMSOutput(grid=grid, path=[file1, file2], use_dask=use_dask)


@pytest.mark.parametrize(
    "roms_output_fixture",
    [
        "roms_output_from_restart_file",
        "roms_output_from_restart_file_adjusted_for_zeta",
        "roms_output_from_restart_file_with_straddling_grid",
    ],
)
def test_plot_on_native_model_grid(roms_output_fixture, request):
    roms_output = request.getfixturevalue(roms_output_fixture)

    for include_boundary in [False, True]:
        for depth_contours in [False, True]:
            # 3D fields
            for var_name in ["temp", "u", "v"]:
                kwargs = {
                    "include_boundary": include_boundary,
                    "depth_contours": depth_contours,
                }

                roms_output.plot(var_name, time=1, s=-1, **kwargs)
                roms_output.plot(var_name, time=1, depth=1000, **kwargs)

                roms_output.plot(var_name, time=1, eta=1, **kwargs)
                roms_output.plot(var_name, time=1, xi=1, **kwargs)

                roms_output.plot(
                    var_name,
                    time=1,
                    eta=1,
                    xi=1,
                    **kwargs,
                )

                roms_output.plot(
                    var_name,
                    time=1,
                    s=-1,
                    eta=1,
                    **kwargs,
                )
                roms_output.plot(
                    var_name,
                    time=1,
                    depth=1000,
                    eta=1,
                    **kwargs,
                )

                roms_output.plot(
                    var_name,
                    time=1,
                    s=-1,
                    xi=1,
                    **kwargs,
                )
                roms_output.plot(
                    var_name,
                    time=1,
                    depth=1000,
                    xi=1,
                    **kwargs,
                )

            # 2D fields
            roms_output.plot("zeta", time=1, **kwargs)
            roms_output.plot("zeta", time=1, eta=1, **kwargs)
            roms_output.plot("zeta", time=1, xi=1, **kwargs)

    # Test that passing a matplotlib.axes.Axes works
    fig, ax = plt.subplots(1, 1)
    roms_output.plot(var_name="temp", time=1, s=-1, ax=ax)
    roms_output.plot(var_name="temp", time=1, depth=1000, ax=ax)
    roms_output.plot(var_name="temp", time=1, eta=1, ax=ax)
    roms_output.plot(var_name="temp", time=1, eta=1, xi=1, ax=ax)
    roms_output.plot(var_name="temp", time=1, s=-1, eta=1, ax=ax)
    roms_output.plot(var_name="temp", time=1, depth=1000, eta=1, ax=ax)
    roms_output.plot(var_name="zeta", time=1, ax=ax)
    roms_output.plot(var_name="zeta", time=1, eta=1, ax=ax)


@pytest.mark.parametrize(
    "roms_output_fixture, lat, lon",
    [
        ("roms_output_from_restart_file", 9, -128),
        ("roms_output_from_restart_file_adjusted_for_zeta", 9, -128),
        ("roms_output_from_restart_file_with_straddling_grid", 60, 0),
    ],
)
@pytest.mark.skipif(xesmf is None, reason="xesmf required")
def test_plot_on_lat_lon(roms_output_fixture, lat, lon, request):
    roms_output = request.getfixturevalue(roms_output_fixture)

    for include_boundary in [False, True]:
        for depth_contours in [False, True]:
            # 3D fields
            for var_name in ["temp", "u", "v"]:
                kwargs = {
                    "include_boundary": include_boundary,
                    "depth_contours": depth_contours,
                }
                roms_output.plot(
                    var_name,
                    time=1,
                    lat=lat,
                    lon=lon,
                    **kwargs,
                )
                roms_output.plot(
                    var_name,
                    time=1,
                    lat=lat,
                    **kwargs,
                )
                roms_output.plot(
                    var_name,
                    time=1,
                    lat=lat,
                    s=-1,
                    **kwargs,
                )
                roms_output.plot(
                    var_name,
                    time=1,
                    lat=lat,
                    depth=1000,
                    **kwargs,
                )
                roms_output.plot(
                    var_name,
                    time=1,
                    lon=lon,
                    **kwargs,
                )
                roms_output.plot(
                    var_name,
                    time=1,
                    lon=lon,
                    s=-1,
                    **kwargs,
                )
                roms_output.plot(
                    var_name,
                    time=1,
                    lon=lon,
                    depth=1000,
                    **kwargs,
                )

            # 2D fields
            roms_output.plot("zeta", time=1, lat=lat, **kwargs)
            roms_output.plot("zeta", time=1, lon=lon, **kwargs)

    # Test that passing a matplotlib.axes.Axes works
    fig, ax = plt.subplots(1, 1)
    roms_output.plot(var_name="temp", time=1, lat=lat, lon=lon, ax=ax)
    roms_output.plot(var_name="temp", time=1, lat=lat, ax=ax)
    roms_output.plot(var_name="temp", time=1, lat=lat, s=-1, ax=ax)
    roms_output.plot(var_name="temp", time=1, lat=lat, depth=1000, ax=ax)
    roms_output.plot(var_name="zeta", time=1, lat=lat, ax=ax)


def test_plot_errors(roms_output_from_restart_file):
    """Test error conditions for the ROMSOutput.plot() method."""
    # Invalid time index
    with pytest.raises(ValueError, match="Invalid time index"):
        roms_output_from_restart_file.plot("temp", time=10, s=-1)

    with pytest.raises(
        ValueError,
        match="Conflicting input: You cannot specify both 's' and 'depth' at the same time.",
    ):
        roms_output_from_restart_file.plot("temp", time=0, s=-1, depth=10)

    # Ambiguous input: Too many dimensions specified for 3D fields
    with pytest.raises(ValueError, match="Ambiguous input"):
        roms_output_from_restart_file.plot("temp", time=1, s=-1, eta=0, xi=0)

    # Vertical dimension specified for 2D fields
    with pytest.raises(
        ValueError, match="Vertical dimension 's' should be None for 2D fields"
    ):
        roms_output_from_restart_file.plot("zeta", time=1, s=-1)
    with pytest.raises(
        ValueError, match="Vertical dimension 'depth' should be None for 2D fields"
    ):
        roms_output_from_restart_file.plot("zeta", time=1, depth=100)

    # Conflicting input: Both eta and xi specified for 2D fields
    with pytest.raises(
        ValueError,
        match="Conflicting input: For 2D fields, specify only one dimension, either 'eta' or 'xi', not both.",
    ):
        roms_output_from_restart_file.plot("zeta", time=1, eta=0, xi=0)
    # Conflicting input: Both lat and lon specified for 2D fields
    with pytest.raises(
        ValueError,
        match="Conflicting input: For 2D fields, specify only one dimension, either 'lat' or 'lon', not both.",
    ):
        roms_output_from_restart_file.plot("zeta", time=1, lat=0, lon=0)

    # Conflicting input: lat or lon provided with eta or xi
    with pytest.raises(
        ValueError,
        match="Conflicting input: You cannot specify 'lat' or 'lon' simultaneously with 'eta' or 'xi'.",
    ):
        roms_output_from_restart_file.plot("temp", time=1, lat=10, lon=20, eta=5)

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


def test_figure_gets_saved(roms_output_from_restart_file, tmp_path):
    filename = tmp_path / "figure.png"
    roms_output_from_restart_file.plot("temp", time=0, depth=1000, save_path=filename)

    assert filename.exists()
    filename.unlink()


@pytest.mark.parametrize(
    "roms_output_fixture",
    [
        "roms_output_from_restart_file",
        "roms_output_from_restart_file_adjusted_for_zeta",
        "roms_output_from_restart_file_with_straddling_grid",
    ],
)
@pytest.mark.skipif(xesmf is None, reason="xesmf required")
def test_regrid_all_variables(roms_output_fixture, request):
    roms_output = request.getfixturevalue(roms_output_fixture)
    ds_regridded = roms_output.regrid()
    assert isinstance(ds_regridded, xr.Dataset)
    assert set(ds_regridded.data_vars).issubset(set(roms_output.ds.data_vars))
    assert "lon" in ds_regridded.coords
    assert "lat" in ds_regridded.coords
    assert "depth" in ds_regridded.coords
    assert "time" in ds_regridded.coords


@pytest.mark.parametrize(
    "roms_output_fixture",
    [
        "roms_output_from_restart_file",
        "roms_output_from_restart_file_adjusted_for_zeta",
        "roms_output_from_restart_file_with_straddling_grid",
    ],
)
@pytest.mark.skipif(xesmf is None, reason="xesmf required")
def test_regrid_specific_variables(roms_output_fixture, request):
    roms_output = request.getfixturevalue(roms_output_fixture)
    var_names = ["temp", "salt"]
    ds_regridded = roms_output.regrid(var_names=var_names)
    assert isinstance(ds_regridded, xr.Dataset)
    assert set(ds_regridded.data_vars) == set(var_names)

    ds = roms_output.regrid(var_names=[])
    assert ds is None


@pytest.mark.parametrize(
    "roms_output_fixture",
    [
        "roms_output_from_restart_file",
        "roms_output_from_restart_file_adjusted_for_zeta",
        "roms_output_from_restart_file_with_straddling_grid",
    ],
)
@pytest.mark.skipif(xesmf is None, reason="xesmf required")
def test_regrid_missing_variable_raises_error(roms_output_fixture, request):
    roms_output = request.getfixturevalue(roms_output_fixture)
    with pytest.raises(
        ValueError, match="The following variables are not found in the dataset"
    ):
        roms_output.regrid(var_names=["fake_variable"])


@pytest.mark.parametrize(
    "roms_output_fixture",
    [
        "roms_output_from_restart_file",
        "roms_output_from_restart_file_adjusted_for_zeta",
        "roms_output_from_restart_file_with_straddling_grid",
    ],
)
@pytest.mark.skipif(xesmf is None, reason="xesmf required")
def test_regrid_with_custom_horizontal_resolution(roms_output_fixture, request):
    roms_output = request.getfixturevalue(roms_output_fixture)
    ds_regridded = roms_output.regrid(horizontal_resolution=0.1)
    assert isinstance(ds_regridded, xr.Dataset)
    assert "lon" in ds_regridded.coords
    assert "lat" in ds_regridded.coords

    assert np.allclose(ds_regridded.lon.diff(dim="lon"), 0.1, atol=1e-4)
    assert np.allclose(ds_regridded.lat.diff(dim="lat"), 0.1, atol=1e-4)


@pytest.mark.parametrize(
    "roms_output_fixture",
    [
        "roms_output_from_restart_file",
        "roms_output_from_restart_file_adjusted_for_zeta",
        "roms_output_from_restart_file_with_straddling_grid",
    ],
)
@pytest.mark.skipif(xesmf is None, reason="xesmf required")
def test_regrid_with_custom_depth_levels(roms_output_fixture, request):
    roms_output = request.getfixturevalue(roms_output_fixture)
    depth_levels = xr.DataArray(
        np.linspace(0, 500, 51), dims=["depth"], attrs={"units": "m"}
    )
    ds_regridded = roms_output.regrid(depth_levels=depth_levels)
    assert isinstance(ds_regridded, xr.Dataset)
    assert "depth" in ds_regridded.coords
    np.allclose(ds_regridded.depth, depth_levels, atol=0.0)


@pytest.fixture
def roms_output_with_cdr_vars(roms_output_from_two_restart_files):
    """Adds minimal CDR variables to the ROMSOutput dataset."""
    ds = roms_output_from_two_restart_files.ds.copy()

    # Dimensions
    time = ds.sizes["time"]
    eta_rho = ds.sizes["eta_rho"]
    xi_rho = ds.sizes["xi_rho"]
    s_rho = ds.sizes["s_rho"]

    # Add required variables for CDR metrics
    ds["ALK_source"] = xr.DataArray(
        np.abs(np.random.randn(time, s_rho, eta_rho, xi_rho)),
        dims=("time", "s_rho", "eta_rho", "xi_rho"),
    )
    ds["DIC_source"] = xr.DataArray(
        -np.abs(np.random.randn(time, s_rho, eta_rho, xi_rho)),
        dims=("time", "s_rho", "eta_rho", "xi_rho"),
    )
    ds["FG_CO2"] = xr.DataArray(
        np.random.randn(time, eta_rho, xi_rho), dims=("time", "eta_rho", "xi_rho")
    )
    ds["FG_ALT_CO2"] = xr.DataArray(
        np.random.randn(time, eta_rho, xi_rho), dims=("time", "eta_rho", "xi_rho")
    )
    ds["hDIC"] = xr.DataArray(
        np.random.randn(time, s_rho, eta_rho, xi_rho),
        dims=("time", "s_rho", "eta_rho", "xi_rho"),
    )
    ds["hDIC_ALT_CO2"] = xr.DataArray(
        np.random.randn(time, s_rho, eta_rho, xi_rho),
        dims=("time", "s_rho", "eta_rho", "xi_rho"),
    )

    # Add average begin/end times (simulate seconds)
    ds["avg_begin_time"] = xr.DataArray(np.arange(time) * 3600, dims=("time",))
    ds["avg_end_time"] = xr.DataArray((np.arange(time) + 1) * 3600, dims=("time",))

    roms_output_from_two_restart_files.ds = ds
    return roms_output_from_two_restart_files


def test_cdr_metrics_computes_and_plots(roms_output_with_cdr_vars):
    roms_output_with_cdr_vars.cdr_metrics()
    assert hasattr(roms_output_with_cdr_vars, "ds_cdr")

    ds_cdr = roms_output_with_cdr_vars.ds_cdr

    # Check presence of both efficiency variables
    assert "cdr_efficiency" in ds_cdr
    assert "cdr_efficiency_from_delta_diff" in ds_cdr
