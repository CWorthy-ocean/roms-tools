import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from conftest import calculate_data_hash
from roms_tools import Grid, InitialConditions
from roms_tools.datasets.download import download_test_data
from roms_tools.datasets.lat_lon_datasets import (
    CESMBGCDataset,
    UnifiedBGCDataset,
)
from roms_tools.setup.initial_conditions import _set_required_vars
from roms_tools.tests.test_setup.utils import download_regional_and_bigger

try:
    import copernicusmarine  # type: ignore
except ImportError:
    copernicusmarine = None
try:
    import xesmf  # type: ignore
except ImportError:
    xesmf = None

skip_xesmf = pytest.mark.skipif(
    xesmf is None, reason="xesmf required for ROMS regridding"
)


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


@pytest.mark.parametrize(
    "ic_fixture",
    [
        "initial_conditions",
        "initial_conditions_with_bgc",
        "initial_conditions_with_bgc_from_climatology",
        "initial_conditions_with_unified_bgc_from_climatology",
        pytest.param("initial_conditions_from_roms_without_bgc", marks=skip_xesmf),
    ],
)
def test_initial_conditions_creation_with_nondefault_glorys_dataset(
    ic_fixture, request
):
    """Test the creation of the InitialConditions object."""
    ic = request.getfixturevalue(ic_fixture)
    assert hasattr(ic.ds, "adjust_depth_for_sea_surface_height")
    assert isinstance(ic.ds, xr.Dataset)
    assert ic.ds.coords["ocean_time"].attrs["units"] == "seconds"
    expected_vars = {"temp", "salt", "u", "v", "zeta", "ubar", "vbar"}
    assert set(ic.ds.data_vars).issuperset(expected_vars)


@pytest.mark.stream
@pytest.mark.use_copernicus
@pytest.mark.use_dask
def test_initial_conditions_creation_with_default_glorys_dataset(example_grid: Grid):
    """Verify the default GLORYS dataset is loaded when a path is not provided."""
    ic = InitialConditions(
        grid=example_grid,
        ini_time=datetime(2021, 6, 29),
        source={"name": "GLORYS"},
        use_dask=True,
        bypass_validation=True,
    )
    expected_vars = {"temp", "salt", "u", "v", "zeta", "ubar", "vbar"}
    assert set(ic.ds.data_vars).issuperset(expected_vars)


@pytest.mark.use_copernicus
@pytest.mark.skipif(copernicusmarine is None, reason="copernicusmarine required")
@pytest.mark.parametrize(
    "grid_fixture",
    [
        "tiny_grid_that_straddles_dateline",
        "tiny_grid_that_straddles_180_degree_meridian",
        "tiny_rotated_grid",
    ],
)
def test_invariance_to_get_glorys_bounds(tmp_path, grid_fixture, use_dask, request):
    ini_time = datetime(2012, 1, 1)
    grid = request.getfixturevalue(grid_fixture)
    regional_file, bigger_regional_file = download_regional_and_bigger(
        tmp_path, grid, ini_time
    )

    ic_from_regional = InitialConditions(
        grid=grid,
        source={"name": "GLORYS", "path": str(regional_file)},
        ini_time=ini_time,
        use_dask=use_dask,
    )
    ic_from_bigger_regional = InitialConditions(
        grid=grid,
        source={"name": "GLORYS", "path": str(bigger_regional_file)},
        ini_time=ini_time,
        use_dask=use_dask,
    )

    # Use assert_allclose instead of equals: necessary for grids that straddle the 180° meridian.
    # Copernicus returns data on [-180, 180] by default, but if you request a range
    # like [170, 190], it remaps longitudes. That remapping introduces tiny floating
    # point differences in the longitude coordinate, which will then propagate into further differences once you do regridding.
    # Need to adjust the tolerances for these grids that straddle the 180° meridian.
    xr.testing.assert_allclose(
        ic_from_bigger_regional.ds, ic_from_regional.ds, rtol=1e-4, atol=1e-5
    )


def test_initial_conditions_creation_with_duplicates(use_dask: bool) -> None:
    """Test the creation of the InitialConditions object with duplicates in source data
    works as expected.
    """
    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))

    grid = Grid(
        nx=3,
        ny=3,
        size_x=400,
        size_y=400,
        center_lon=-8,
        center_lat=58,
        rot=0,
        N=3,
    )

    initial_conditions = InitialConditions(
        grid=grid,
        ini_time=datetime(2012, 1, 1),
        source={"path": [fname1, fname2], "name": "GLORYS"},
        allow_flex_time=True,
        use_dask=use_dask,
    )

    initial_conditions_with_duplicates_in_source_data = InitialConditions(
        grid=grid,
        ini_time=datetime(2012, 1, 1),
        source={"path": [fname1, fname1, fname2], "name": "GLORYS"},
        allow_flex_time=True,
        use_dask=use_dask,
    )

    assert initial_conditions.ds.identical(
        initial_conditions_with_duplicates_in_source_data.ds
    )


@pytest.mark.parametrize(
    "ic_fixture",
    [
        "initial_conditions_with_bgc",
        "initial_conditions_with_bgc_from_climatology",
        "initial_conditions_with_unified_bgc_from_climatology",
        pytest.param("initial_conditions_from_roms", marks=skip_xesmf),
    ],
)
def test_initial_condition_creation_with_bgc(ic_fixture, request):
    """Test the creation of the BoundaryForcing object."""
    ic = request.getfixturevalue(ic_fixture)
    expected_bgc_variables = [
        "PO4",
        "NO3",
        "SiO3",
        "NH4",
        "Fe",
        "Lig",
        "O2",
        "DIC",
        "DIC_ALT_CO2",
        "ALK",
        "ALK_ALT_CO2",
        "DOC",
        "DON",
        "DOP",
        "DOCr",
        "DONr",
        "DOPr",
        "zooC",
        "spChl",
        "spC",
        "spP",
        "spFe",
        "spCaCO3",
        "diatChl",
        "diatC",
        "diatP",
        "diatFe",
        "diatSi",
        "diazChl",
        "diazC",
        "diazP",
        "diazFe",
    ]

    for var in expected_bgc_variables:
        assert var in ic.ds


@pytest.mark.skipif(xesmf is None, reason="xesmf required")
def test_initial_conditions_raises_on_regridded_nans():
    """Raise ValueError if regridded ROMS fields contain NaNs due to grid mismatch."""
    parent_grid = Grid(
        center_lon=-120, center_lat=30, nx=8, ny=13, size_x=3000, size_y=4000, rot=32
    )
    restart_file = Path(download_test_data("eastpac25km_rst.19980106000000.nc"))
    # create grid that is not entirely contained in the parent grid
    grid_params = {
        "nx": 5,
        "ny": 5,
        "center_lon": -128,
        "center_lat": 9,
        "size_x": 100,
        "size_y": 100,
    }
    grid = Grid(**grid_params)

    with pytest.raises(ValueError, match="NaN values found in regridded field."):
        InitialConditions(
            grid=grid,
            ini_time=datetime(1998, 1, 6),
            source={"name": "ROMS", "grid": parent_grid, "path": restart_file},
            use_dask=True,
            bgc_source={
                "name": "ROMS",
                "grid": parent_grid,
                "path": restart_file,
            },
        )


@pytest.mark.skipif(xesmf is None, reason="xesmf required")
def test_initial_conditions_unchanged_when_parent_and_child_grids_match():
    grid_params = {
        "nx": 8,
        "ny": 13,
        "center_lon": -120,
        "center_lat": 30,
        "size_x": 3000,
        "size_y": 4000,
        "rot": 32,
    }
    parent_grid = Grid(**grid_params)
    grid = Grid(**grid_params)

    restart_file = Path(download_test_data("eastpac25km_rst.19980106000000.nc"))
    ds = xr.open_dataset(restart_file)

    ic = InitialConditions(
        grid=grid,
        ini_time=datetime(1998, 1, 6),
        source={"name": "ROMS", "grid": parent_grid, "path": restart_file},
        use_dask=True,
    )

    mask_map = {
        "temp": grid.ds.mask_rho,
        "salt": grid.ds.mask_rho,
        "zeta": grid.ds.mask_rho,
        "u": grid.ds.mask_u,
        "v": grid.ds.mask_v,
    }

    for var_name in ["temp", "salt", "zeta", "u", "v"]:
        mask = mask_map[var_name]

        restart_values = ds[var_name].isel(time=1).where(mask).values
        ic_values = ic.ds[var_name].squeeze().where(mask).values

        assert np.allclose(
            ic_values,
            restart_values,
            equal_nan=True,
        ), f"{var_name} values changed during initialization"


# Test initialization with missing 'name' in source
def test_initial_conditions_missing_physics_name(example_grid, use_dask):
    with pytest.raises(ValueError, match="`source` must include a 'name'"):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"path": "physics_data.nc"},
            use_dask=use_dask,
        )


# Test initialization with missing 'name' in bgc_source
def test_initial_conditions_missing_bgc_name(example_grid, use_dask):
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with pytest.raises(ValueError, match="`bgc_source` must include a 'name'"):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            bgc_source={"path": "bgc_data.nc"},
            use_dask=use_dask,
        )


# Test initialization with missing 'path' in bgc_source
def test_initial_conditions_missing_bgc_path(example_grid, use_dask):
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with pytest.raises(
        ValueError, match="`bgc_source` must include a 'path' if it is provided."
    ):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS", "path": fname},
            bgc_source={"name": "CESM_REGRIDDED"},
            use_dask=use_dask,
        )


# Test initialization with missing ini_time
def test_initial_conditions_missing_ini_time(example_grid, use_dask):
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with pytest.raises(
        TypeError,
        match="`ini_time` must be a datetime object",
    ):
        InitialConditions(
            grid=example_grid,
            ini_time=None,
            source={"name": "GLORYS", "path": fname},
            use_dask=use_dask,
        )


# Test default climatology value
def test_initial_conditions_default_climatology(example_grid, use_dask):
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))

    initial_conditions = InitialConditions(
        grid=example_grid,
        ini_time=datetime(2021, 6, 29),
        source={"name": "GLORYS", "path": fname},
        use_dask=use_dask,
    )

    assert initial_conditions.source["climatology"] is False
    assert initial_conditions.bgc_source is None


def test_initial_conditions_default_bgc_climatology(example_grid, use_dask):
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    fname_bgc = Path(download_test_data("CESM_regional_test_data_one_time_slice.nc"))

    initial_conditions = InitialConditions(
        grid=example_grid,
        ini_time=datetime(2021, 6, 29),
        source={"name": "GLORYS", "path": fname},
        bgc_source={"name": "CESM_REGRIDDED", "path": fname_bgc},
        use_dask=use_dask,
    )

    assert initial_conditions.bgc_source["climatology"] is False


@pytest.mark.parametrize(
    "initial_conditions_fixture",
    [
        "initial_conditions",
        "initial_conditions_with_bgc",
        "initial_conditions_with_unified_bgc_from_climatology",
    ],
)
def test_correct_depth_coords_zero_zeta(initial_conditions_fixture, request, use_dask):
    initial_conditions = request.getfixturevalue(initial_conditions_fixture)

    # compute interface depth at rho-points and write it into .ds_depth_coords
    initial_conditions._get_depth_coordinates(0, location="rho", depth_type="interface")
    # Test that lowermost interface coincides with topography
    assert np.allclose(
        initial_conditions.ds_depth_coords["interface_depth_rho"]
        .isel(s_w=0)
        .squeeze()
        .values,  # Extract raw NumPy array
        initial_conditions.grid.ds.h.values,
        atol=1e-6,  # Adjust tolerance as needed
    )

    # Test that uppermost interface coincides with sea surface height
    assert np.allclose(
        initial_conditions.ds_depth_coords["interface_depth_rho"]
        .isel(s_w=-1)
        .squeeze()
        .values,
        0 * initial_conditions.grid.ds.h.values,
        atol=1e-6,
    )


def test_interpolation_from_climatology(use_dask):
    # CESM climatology
    fname_bgc = download_test_data("CESM_regional_coarse_test_data_climatology.nc")
    ds = xr.open_dataset(fname_bgc)

    # check if interpolated value for Jan 15 is indeed January value from climatology
    bgc_data = CESMBGCDataset(
        filename=fname_bgc,
        start_time=datetime(2012, 1, 15),
        climatology=True,
        use_dask=use_dask,
        apply_post_processing=False,
    )
    assert np.allclose(ds["ALK"].sel(month=1), bgc_data.ds["ALK"], equal_nan=True)

    # check if interpolated value for Jan 30 is indeed average of January and February value from climatology
    bgc_data = CESMBGCDataset(
        filename=fname_bgc,
        start_time=datetime(2012, 1, 30),
        climatology=True,
        use_dask=use_dask,
        apply_post_processing=False,
    )
    assert np.allclose(
        0.5 * (ds["ALK"].sel(month=1) + ds["ALK"].sel(month=2)),
        bgc_data.ds["ALK"],
        equal_nan=True,
    )

    # Unified BGC climatology
    fname_bgc = download_test_data("coarsened_UNIFIED_bgc_dataset.nc")
    ds = xr.open_dataset(fname_bgc)

    # check if interpolated value for Feb 14 is indeed February value from climatology
    bgc_data = UnifiedBGCDataset(
        filename=fname_bgc,
        start_time=datetime(2012, 2, 14),
        climatology=True,
        use_dask=use_dask,
        apply_post_processing=False,
    )
    assert np.allclose(ds["Alk"].isel(month=1), bgc_data.ds["Alk"], equal_nan=True)

    # check if interpolated value for Jan 30.25 is indeed average of January and February value from climatology
    bgc_data = UnifiedBGCDataset(
        filename=fname_bgc,
        start_time=datetime(2012, 1, 30, 6),  # time: 6 am, Jan 30
        climatology=True,
        use_dask=use_dask,
        apply_post_processing=False,
    )

    assert np.allclose(
        0.5 * (ds["Alk"].isel(month=0) + ds["Alk"].isel(month=1)),
        bgc_data.ds["Alk"],
        equal_nan=True,
    )


def test_computed_missing_optional_fields(
    initial_conditions_with_unified_bgc_from_climatology,
):
    ds = initial_conditions_with_unified_bgc_from_climatology.ds

    # Use tight tolerances because 'DOC' and 'DOCr' can have values order 1e-6

    # 'DOCr' was missing in the source data and should have been filled with a constant default value
    assert np.allclose(ds.DOCr.std(), 0.0, rtol=1e-10, atol=1e-10), (
        "DOCr should be constant across space and time"
    )
    # 'DOC' was present in the source data and should show spatial or temporal variability
    assert ds.DOC.std() > 1e-10, "DOC should vary across space and time"


@pytest.mark.parametrize(
    "initial_conditions_fixture",
    [
        "initial_conditions_with_bgc_from_climatology",
        "initial_conditions_with_unified_bgc_from_climatology",
        pytest.param("initial_conditions_from_roms", marks=skip_xesmf),
    ],
)
def test_initial_conditions_plot(initial_conditions_fixture, request):
    """Test plot method."""
    initial_conditions = request.getfixturevalue(initial_conditions_fixture)

    # horizontal slices plots
    for depth_contours in [True, False]:
        for var_name in ["temp", "u", "v", "ALK", "DOC"]:
            initial_conditions.plot(
                var_name=var_name, s=0, depth_contours=depth_contours
            )

    # section plots
    for layer_contours in [True, False]:
        for var_name in ["temp", "u", "v", "ALK", "DOC"]:
            initial_conditions.plot(
                var_name=var_name, eta=0, layer_contours=layer_contours
            )
            initial_conditions.plot(
                var_name=var_name, xi=0, layer_contours=layer_contours
            )

    for var_name in ["temp", "ALK", "DOC"]:
        # 1D plot in horizontal
        initial_conditions.plot(var_name=var_name, s=0, xi=0)
        # 1D plot in vertical
        initial_conditions.plot(var_name=var_name, eta=0, xi=0)

    initial_conditions.plot(var_name="zeta")
    initial_conditions.plot(var_name="ubar")
    initial_conditions.plot(var_name="vbar")

    # Test that passing a matplotlib.axes.Axes works
    fig, ax = plt.subplots(1, 1)
    initial_conditions.plot(var_name="temp", s=0, ax=ax)
    initial_conditions.plot(var_name="temp", eta=0, ax=ax)
    initial_conditions.plot(var_name="temp", s=0, xi=0, ax=ax)
    initial_conditions.plot(var_name="zeta", ax=ax)


@pytest.mark.parametrize(
    "initial_conditions_fixture",
    [
        "initial_conditions",
        "initial_conditions_with_bgc_from_climatology",
        "initial_conditions_with_unified_bgc_from_climatology",
        pytest.param("initial_conditions_from_roms", marks=skip_xesmf),
    ],
)
def test_initial_conditions_save(initial_conditions_fixture, request, tmp_path):
    """Test save method."""
    initial_conditions = request.getfixturevalue(initial_conditions_fixture)

    for file_str in ["test_ic", "test_ic.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str
            saved_filenames = initial_conditions.save(filepath)
            # Check if the .nc file was created
            filepath = Path(filepath).with_suffix(".nc")
            assert saved_filenames == [filepath]
            assert filepath.exists()
            # Clean up the .nc file
            filepath.unlink()


@pytest.mark.parametrize(
    "initial_conditions_fixture",
    [
        "initial_conditions",
        "initial_conditions_with_bgc_from_climatology",
        "initial_conditions_with_unified_bgc_from_climatology",
        pytest.param("initial_conditions_from_roms", marks=skip_xesmf),
    ],
)
def test_roundtrip_yaml(initial_conditions_fixture, request, tmp_path, use_dask):
    """Test that creating an InitialConditions object, saving its parameters to yaml
    file, and re-opening yaml file creates the same object.
    """
    initial_conditions = request.getfixturevalue(initial_conditions_fixture)

    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str
        initial_conditions.to_yaml(filepath)

        initial_conditions_from_file = InitialConditions.from_yaml(
            filepath, use_dask=use_dask
        )

        assert initial_conditions == initial_conditions_from_file

        filepath = Path(filepath)
        filepath.unlink()


@pytest.mark.parametrize(
    "initial_conditions_fixture",
    [
        "initial_conditions",
        "initial_conditions_with_bgc_from_climatology",
        "initial_conditions_with_unified_bgc_from_climatology",
        pytest.param("initial_conditions_from_roms_without_bgc", marks=skip_xesmf),
        pytest.param("initial_conditions_from_roms", marks=skip_xesmf),
    ],
)
def test_files_have_same_hash(initial_conditions_fixture, request, tmp_path, use_dask):
    initial_conditions = request.getfixturevalue(initial_conditions_fixture)

    yaml_filepath = tmp_path / "test_yaml.yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"

    initial_conditions.to_yaml(yaml_filepath)
    initial_conditions.save(filepath1)
    ic_from_file = InitialConditions.from_yaml(yaml_filepath, use_dask)
    ic_from_file.save(filepath2)

    # Only compare hash of datasets because metadata is non-deterministic with dask
    hash1 = calculate_data_hash(filepath1)
    hash2 = calculate_data_hash(filepath2)

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    filepath1.unlink()
    filepath2.unlink()


def test_from_yaml_missing_initial_conditions(tmp_path, use_dask):
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
      topography_source:
        name: ETOPO5
      hmin: 5.0
    """
    )

    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for yaml_filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str
        # Write YAML content to file
        if isinstance(yaml_filepath, Path):
            yaml_filepath.write_text(yaml_content)
        else:
            with open(yaml_filepath, "w") as f:
                f.write(yaml_content)

        with pytest.raises(
            ValueError,
            match="No InitialConditions configuration found in the YAML file.",
        ):
            InitialConditions.from_yaml(yaml_filepath, use_dask)

        yaml_filepath = Path(yaml_filepath)
        yaml_filepath.unlink()


# Test _set_required_vars


def test_default_var_type():
    vars_map = _set_required_vars()
    # Default is "physics"
    expected_keys = {"zeta", "temp", "salt", "u", "v"}
    assert set(vars_map.keys()) == expected_keys
    # Values should match keys
    for key, val in vars_map.items():
        assert key == val


def test_bgc_var_type():
    vars_map = _set_required_vars("bgc")
    # Check a few expected keys exist
    expected_keys = {"PO4", "NO3", "SiO3", "NH4", "Fe", "DIC", "spChl", "zooC"}
    assert expected_keys.issubset(vars_map.keys())
    # Values should match keys
    for key, val in vars_map.items():
        assert key == val


def test_invalid_var_type():
    with pytest.raises(ValueError, match="Unsupported var_type"):
        _set_required_vars("invalid_type")
