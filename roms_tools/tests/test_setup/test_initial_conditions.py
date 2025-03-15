import pytest
from datetime import datetime
from roms_tools import InitialConditions, Grid
import xarray as xr
import numpy as np
import textwrap
import logging
from roms_tools.download import download_test_data
from roms_tools.setup.datasets import CESMBGCDataset, UnifiedBGCDataset
from pathlib import Path
from conftest import calculate_data_hash


@pytest.fixture
def initial_conditions_with_unified_bgc_from_climatology(use_dask):
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

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    fname_bgc = Path(download_test_data("coarsened_UNIFIED_bgc_dataset.nc"))

    return InitialConditions(
        grid=grid,
        ini_time=datetime(2021, 6, 29),
        source={"path": fname, "name": "GLORYS"},
        bgc_source={"path": fname_bgc, "name": "UNIFIED", "climatology": True},
        use_dask=use_dask,
    )


@pytest.mark.parametrize(
    "ic_fixture",
    [
        "initial_conditions",
        "initial_conditions_adjusted_for_zeta",
        "initial_conditions_with_bgc",
        "initial_conditions_with_bgc_adjusted_for_zeta",
        "initial_conditions_with_bgc_from_climatology",
        "initial_conditions_with_unified_bgc_from_climatology",
    ],
)
def test_initial_conditions_creation(ic_fixture, request):
    """Test the creation of the InitialConditions object."""

    ic = request.getfixturevalue(ic_fixture)

    assert ic.ini_time == datetime(2021, 6, 29)
    assert ic.source == {
        "name": "GLORYS",
        "path": Path(download_test_data("GLORYS_coarse_test_data.nc")),
        "climatology": False,
    }
    assert hasattr(ic.ds, "adjust_depth_for_sea_surface_height")
    assert isinstance(ic.ds, xr.Dataset)
    assert "temp" in ic.ds
    assert "salt" in ic.ds
    assert "u" in ic.ds
    assert "v" in ic.ds
    assert "zeta" in ic.ds
    assert ic.ds.coords["ocean_time"].attrs["units"] == "seconds"


@pytest.mark.parametrize(
    "ic_fixture",
    [
        "initial_conditions_with_bgc",
        "initial_conditions_with_bgc_adjusted_for_zeta",
        "initial_conditions_with_bgc_from_climatology",
        "initial_conditions_with_unified_bgc_from_climatology",
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
def test_initial_conditions_missing_physics_name(example_grid, use_dask):
    with pytest.raises(ValueError, match="`source` must include a 'name'."):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"path": "physics_data.nc"},
            use_dask=use_dask,
        )


# Test initialization with missing 'path' in source
def test_initial_conditions_missing_physics_path(example_grid, use_dask):
    with pytest.raises(ValueError, match="`source` must include a 'path'."):
        InitialConditions(
            grid=example_grid,
            ini_time=datetime(2021, 6, 29),
            source={"name": "GLORYS"},
            use_dask=use_dask,
        )


# Test initialization with missing 'name' in bgc_source
def test_initial_conditions_missing_bgc_name(example_grid, use_dask):

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    with pytest.raises(
        ValueError, match="`bgc_source` must include a 'name' if it is provided."
    ):
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
        ValueError,
        match="`ini_time` must be a valid datetime object and cannot be None.",
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


def test_info_depth(caplog, use_dask):

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

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))

    with caplog.at_level(logging.INFO):

        InitialConditions(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"path": fname, "name": "GLORYS"},
            adjust_depth_for_sea_surface_height=True,
            use_dask=use_dask,
        )
    # Verify the warning message in the log
    assert "Sea surface height will be used to adjust depth coordinates." in caplog.text

    # Clear the log before the next test
    caplog.clear()

    with caplog.at_level(logging.INFO):

        InitialConditions(
            grid=grid,
            ini_time=datetime(2021, 6, 29),
            source={"path": fname, "name": "GLORYS"},
            adjust_depth_for_sea_surface_height=False,
            use_dask=use_dask,
        )
    # Verify the warning message in the log
    assert (
        "Sea surface height will NOT be used to adjust depth coordinates."
        in caplog.text
    )


@pytest.mark.parametrize(
    "initial_conditions_fixture",
    [
        "initial_conditions_adjusted_for_zeta",
        "initial_conditions_with_bgc_adjusted_for_zeta",
    ],
)
def test_correct_depth_coords_adjusted_for_zeta(
    initial_conditions_fixture, request, use_dask
):

    initial_conditions = request.getfixturevalue(initial_conditions_fixture)

    # compute interface depth at rho-points and write it into .ds_depth_coords
    zeta = initial_conditions.ds.zeta
    initial_conditions._get_depth_coordinates(
        zeta, location="rho", depth_type="interface"
    )
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
        -zeta.values,
        atol=1e-6,
    )


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


@pytest.mark.parametrize(
    "initial_conditions_fixture",
    [
        "initial_conditions_with_bgc_adjusted_for_zeta",
        "initial_conditions_with_bgc_from_climatology",
        "initial_conditions_with_unified_bgc_from_climatology",
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


@pytest.mark.parametrize(
    "initial_conditions_fixture",
    [
        "initial_conditions",
        "initial_conditions_adjusted_for_zeta",
        "initial_conditions_with_bgc_from_climatology",
        "initial_conditions_with_unified_bgc_from_climatology",
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
        "initial_conditions_adjusted_for_zeta",
        "initial_conditions_with_bgc_from_climatology",
        "initial_conditions_with_unified_bgc_from_climatology",
    ],
)
def test_roundtrip_yaml(initial_conditions_fixture, request, tmp_path, use_dask):
    """Test that creating an InitialConditions object, saving its parameters to yaml
    file, and re-opening yaml file creates the same object."""

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
        "initial_conditions_adjusted_for_zeta",
        "initial_conditions_with_bgc_from_climatology",
        "initial_conditions_with_unified_bgc_from_climatology",
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
