from roms_tools import RiverForcing, Grid
import xarray as xr
import numpy as np
from datetime import datetime
import textwrap
from pathlib import Path
import pytest
import logging
from conftest import calculate_file_hash


@pytest.fixture
def iceland_test_grid():
    return Grid(
        nx=18, ny=18, size_x=800, size_y=800, center_lon=-18, center_lat=65, rot=20, N=3
    )


@pytest.fixture
def river_forcing_climatology(iceland_test_grid):
    """Fixture for creating a RiverForcing object from the global Dai river dataset."""

    start_time = datetime(1998, 1, 1)
    end_time = datetime(1998, 3, 1)

    return RiverForcing(
        grid=iceland_test_grid,
        start_time=start_time,
        end_time=end_time,
        convert_to_climatology="always",
    )


@pytest.fixture
def river_forcing_for_grid_that_straddles_dateline():
    """Fixture for creating a RiverForcing object from the global Dai river dataset for
    a grid that straddles the dateline."""

    grid = Grid(
        nx=18,
        ny=18,
        size_x=1500,
        size_y=1500,
        center_lon=-0,
        center_lat=65,
        rot=20,
        N=3,
    )
    start_time = datetime(1998, 1, 1)
    end_time = datetime(1998, 3, 1)

    return RiverForcing(
        grid=grid,
        start_time=start_time,
        end_time=end_time,
    )


@pytest.fixture
def single_cell_indices():
    # These are the indices that the `river_forcing` fixture generates automatically.
    return {
        "Hvita(Olfusa)": [(8, 6)],
        "Thjorsa": [(8, 6)],
        "JkulsFjll": [(11, 12)],
        "Lagarfljot": [(9, 13)],
        "Bruara": [(8, 6)],
        "Svarta": [(12, 9)],
    }


@pytest.fixture
def multi_cell_indices():
    # These are the indices that the `river_forcing` fixture generates automatically.
    return {
        "Hvita(Olfusa)": [(8, 6)],
        "Thjorsa": [(8, 6)],
        "JkulsFjll": [(11, 12)],
        "Lagarfljot": [(9, 13), (10, 13)],
        "Bruara": [(8, 6)],
        "Svarta": [(12, 8), (12, 9), (12, 10)],
    }


@pytest.fixture
def river_forcing_with_prescribed_single_cell_indices(
    single_cell_indices, iceland_test_grid
):
    """Fixture for creating a RiverForcing object based on the global Dai river dataset,
    using manually specified single-cell river indices instead of relying on automatic
    detection."""

    start_time = datetime(1998, 1, 1)
    end_time = datetime(1998, 3, 1)

    return RiverForcing(
        grid=iceland_test_grid,
        start_time=start_time,
        end_time=end_time,
        indices=single_cell_indices,
    )


@pytest.fixture
def river_forcing_with_prescribed_multi_cell_indices(
    multi_cell_indices, iceland_test_grid
):
    """Fixture for creating a RiverForcing object based on the global Dai river dataset,
    using manually specified multi-cell river indices instead of relying on automatic
    detection."""

    start_time = datetime(1998, 1, 1)
    end_time = datetime(1998, 3, 1)

    return RiverForcing(
        grid=iceland_test_grid,
        start_time=start_time,
        end_time=end_time,
        indices=multi_cell_indices,
    )


def compare_dictionaries(dict1, dict2):
    assert dict1.keys() == dict2.keys()

    for key in dict1.keys():
        assert np.array_equal(dict1[key], dict2[key])


class TestRiverForcingGeneral:
    @pytest.mark.parametrize(
        "river_forcing_fixture",
        [
            "river_forcing",
            "river_forcing_climatology",
            "river_forcing_with_bgc",
            "river_forcing_for_grid_that_straddles_dateline",
            "river_forcing_with_prescribed_single_cell_indices",
            "river_forcing_with_prescribed_multi_cell_indices",
        ],
    )
    def test_successful_initialization(self, river_forcing_fixture, request):
        river_forcing = request.getfixturevalue(river_forcing_fixture)
        assert isinstance(river_forcing.ds, xr.Dataset)
        assert len(river_forcing.ds.nriver) > 0
        assert len(river_forcing.original_indices) > 0
        assert len(river_forcing.indices) > 0
        assert "river_volume" in river_forcing.ds
        assert "river_tracer" in river_forcing.ds
        assert "river_time" in river_forcing.ds

    @pytest.mark.parametrize(
        "river_forcing_fixture",
        [
            "river_forcing",
            "river_forcing_climatology",
            "river_forcing_with_bgc",
            "river_forcing_with_prescribed_single_cell_indices",
            "river_forcing_with_prescribed_multi_cell_indices",
        ],
    )
    def test_climatology_attributes(self, river_forcing_fixture, request):
        river_forcing = request.getfixturevalue(river_forcing_fixture)
        assert river_forcing.climatology
        assert hasattr(
            river_forcing.ds.river_time,
            "cycle_length",
        )
        assert hasattr(river_forcing.ds, "climatology")

    def test_no_climatology_attributes(self, river_forcing_no_climatology, request):
        assert not river_forcing_no_climatology.climatology
        assert not hasattr(
            river_forcing_no_climatology.ds.river_time,
            "cycle_length",
        )
        assert not hasattr(river_forcing_no_climatology.ds, "climatology")

    @pytest.mark.parametrize(
        "river_forcing_fixture",
        [
            "river_forcing_climatology",
            "river_forcing_no_climatology",
            "river_forcing_with_bgc",
            "river_forcing_with_prescribed_single_cell_indices",
            "river_forcing_with_prescribed_multi_cell_indices",
        ],
    )
    def test_constant_tracers(self, river_forcing_fixture, request):
        river_forcing = request.getfixturevalue(river_forcing_fixture)
        np.testing.assert_allclose(
            river_forcing.ds.river_tracer.isel(ntracers=0).values, 17.0, atol=0
        )
        np.testing.assert_allclose(
            river_forcing.ds.river_tracer.isel(ntracers=1).values, 1.0, atol=0
        )
        np.testing.assert_allclose(
            river_forcing.ds.river_tracer.isel(ntracers=slice(2, None)).values,
            0.0,
            atol=0,
        )

    def test_reproducibility_same_grid(self, river_forcing):

        the_same_river_forcing = RiverForcing(
            grid=river_forcing.grid,
            start_time=datetime(1998, 1, 1),
            end_time=datetime(1998, 3, 1),
        )

        assert river_forcing == the_same_river_forcing

    @pytest.mark.parametrize(
        "river_forcing_fixture",
        [
            "river_forcing_climatology",
            "river_forcing_no_climatology",
            "river_forcing_with_bgc",
            "river_forcing_with_prescribed_single_cell_indices",
            "river_forcing_with_prescribed_multi_cell_indices",
        ],
    )
    def test_river_locations_are_along_coast(self, river_forcing_fixture, request):
        river_forcing = request.getfixturevalue(river_forcing_fixture)

        mask = river_forcing.grid.ds.mask_rho
        faces = (
            mask.shift(eta_rho=1)
            + mask.shift(eta_rho=-1)
            + mask.shift(xi_rho=1)
            + mask.shift(xi_rho=-1)
        )
        coast = (1 - mask) * (faces > 0)

        indices = river_forcing.indices
        for name in indices.keys():
            for (eta_rho, xi_rho) in indices[name]:
                assert coast[eta_rho, xi_rho]
                assert river_forcing.ds["river_location"][eta_rho, xi_rho] > 0

    def test_missing_source_name(self, iceland_test_grid):
        with pytest.raises(ValueError, match="`source` must include a 'name'."):
            RiverForcing(
                grid=iceland_test_grid,
                start_time=datetime(1998, 1, 1),
                end_time=datetime(1998, 3, 1),
                source={"path": "river_data.nc"},
            )

    def test_river_forcing_plot(self, river_forcing_with_bgc):
        """Test plot method."""

        river_forcing_with_bgc.plot_locations()
        river_forcing_with_bgc.plot("river_volume")
        river_forcing_with_bgc.plot("river_temp")
        river_forcing_with_bgc.plot("river_salt")
        river_forcing_with_bgc.plot("river_ALK")
        river_forcing_with_bgc.plot("river_PO4")

    @pytest.mark.parametrize(
        "river_forcing_fixture",
        [
            "river_forcing_with_bgc",
            "river_forcing_with_prescribed_multi_cell_indices",
        ],
    )
    def test_river_forcing_save(self, river_forcing_fixture, tmp_path, request):
        """Test save method."""

        river_forcing = request.getfixturevalue(river_forcing_fixture)
        for file_str in ["test_rivers", "test_rivers.nc"]:
            # Create a temporary filepath using the tmp_path fixture
            for filepath in [tmp_path / file_str, str(tmp_path / file_str)]:

                saved_filenames = river_forcing.save(filepath)
                # Check if the .nc file was created
                filepath = Path(filepath).with_suffix(".nc")
                assert saved_filenames == [filepath]
                assert filepath.exists()
                # Clean up the .nc file
                filepath.unlink()

    @pytest.mark.parametrize(
        "river_forcing_fixture",
        [
            "river_forcing_climatology",
            "river_forcing_no_climatology",
            "river_forcing_with_bgc",
            "river_forcing_with_prescribed_single_cell_indices",
            "river_forcing_with_prescribed_multi_cell_indices",
        ],
    )
    def test_roundtrip_yaml(self, river_forcing_fixture, request, tmp_path, caplog):
        """Test that creating an RiverForcing object, saving its parameters to yaml
        file, and re-opening yaml file creates the same object."""

        river_forcing = request.getfixturevalue(river_forcing_fixture)

        # Create a temporary filepath using the tmp_path fixture
        file_str = "test_yaml"
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str

            river_forcing.to_yaml(filepath)

            # Clear caplog before running the test
            caplog.clear()

            with caplog.at_level(logging.INFO):
                river_forcing_from_file = RiverForcing.from_yaml(filepath)

            assert "Use provided river indices." in caplog.text
            assert river_forcing == river_forcing_from_file

            filepath = Path(filepath)
            filepath.unlink()

    @pytest.mark.parametrize(
        "river_forcing_fixture",
        [
            "river_forcing_climatology",
            "river_forcing_no_climatology",
            "river_forcing_with_bgc",
            "river_forcing_with_prescribed_single_cell_indices",
            "river_forcing_with_prescribed_multi_cell_indices",
        ],
    )
    def test_files_have_same_hash(self, river_forcing_fixture, request, tmp_path):

        river_forcing = request.getfixturevalue(river_forcing_fixture)

        yaml_filepath = tmp_path / "test_yaml.yaml"
        filepath1 = tmp_path / "test1.nc"
        filepath2 = tmp_path / "test2.nc"

        river_forcing.to_yaml(yaml_filepath)
        river_forcing.save(filepath1)
        rf_from_file = RiverForcing.from_yaml(yaml_filepath)
        rf_from_file.save(filepath2)

        hash1 = calculate_file_hash(filepath1)
        hash2 = calculate_file_hash(filepath2)

        assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

        yaml_filepath.unlink()
        filepath1.unlink()
        filepath2.unlink()

    def test_from_yaml_missing_river_forcing(self, tmp_path):
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
                match="No RiverForcing configuration found in the YAML file.",
            ):
                RiverForcing.from_yaml(yaml_filepath)

            yaml_filepath = Path(yaml_filepath)
            yaml_filepath.unlink()


class TestRiverForcingWithoutPrescribedIndices:
    def test_logging_message(self, iceland_test_grid, caplog):

        with caplog.at_level(logging.INFO):
            RiverForcing(
                grid=iceland_test_grid,
                start_time=datetime(1998, 1, 1),
                end_time=datetime(1998, 3, 1),
            )
        # Verify the info message in the log
        assert "No river indices provided." in caplog.text

    def test_reproducibility(self, river_forcing, river_forcing_climatology):
        """Verify that `river_forcing` and `river_forcing_climatology` produce identical
        outputs.

        `river_forcing` is initialized with `convert_to_climatology="if_any_missing"`, meaning
        it fell back to climatology. This test ensures that the resulting datasets
        and river index mappings are the same between the two cases.
        """
        xr.testing.assert_allclose(river_forcing.ds, river_forcing_climatology.ds)
        compare_dictionaries(
            river_forcing.original_indices, river_forcing_climatology.original_indices
        )
        compare_dictionaries(river_forcing.indices, river_forcing_climatology.indices)

    def test_no_rivers_found(self):

        # Create a grid over open ocean
        grid = Grid(
            nx=2, ny=2, size_x=50, size_y=50, center_lon=0, center_lat=55, rot=10, N=3
        )

        with pytest.raises(ValueError, match="No relevant rivers found."):
            RiverForcing(
                grid=grid,
                start_time=datetime(1998, 1, 1),
                end_time=datetime(1998, 3, 1),
            )


class TestRiverForcingWithPrescribedIndices:
    def test_logging_message(self, iceland_test_grid, single_cell_indices, caplog):

        with caplog.at_level(logging.INFO):
            RiverForcing(
                grid=iceland_test_grid,
                start_time=datetime(1998, 1, 1),
                end_time=datetime(1998, 3, 1),
                indices=single_cell_indices,
            )
        # Verify the info message in the log
        assert "Use provided river indices." in caplog.text

    @pytest.mark.parametrize(
        "indices_fixture", ["single_cell_indices", "multi_cell_indices"]
    )
    def test_indices_stay_untouched(self, iceland_test_grid, indices_fixture, request):
        indices = request.getfixturevalue(indices_fixture)

        start_time = datetime(1998, 1, 1)
        end_time = datetime(1998, 3, 1)

        river_forcing = RiverForcing(
            grid=iceland_test_grid,
            start_time=start_time,
            end_time=end_time,
            indices=indices,
        )
        river_forcing.original_indices == indices
        river_forcing.indices == indices

    def test_fraction(
        self,
        river_forcing_with_prescribed_single_cell_indices,
        river_forcing_with_prescribed_multi_cell_indices,
    ):
        def list_non_zero_values(data_array):
            non_zero_values = data_array.values
            return non_zero_values[non_zero_values != 0].tolist()

        # check that all values are integers for single cell rivers
        non_zero_values = river_forcing_with_prescribed_single_cell_indices.ds[
            "river_location"
        ]
        is_integer = non_zero_values == np.floor(non_zero_values)
        assert (is_integer).all()

        # check that not all values are integers for multi cell rivers
        non_zero_values = river_forcing_with_prescribed_multi_cell_indices.ds[
            "river_location"
        ]
        is_integer = non_zero_values == np.floor(non_zero_values)
        assert not (is_integer).all()

    def test_reproducibility(
        self, river_forcing, river_forcing_with_prescribed_single_cell_indices
    ):
        """river_forcing_with_prescribed_single_cell_indices was created with the
        indices that were automatically inferred for river_forcing.

        Test that these two are identical.
        """
        assert (
            river_forcing.indices
            == river_forcing_with_prescribed_single_cell_indices.indices
        )
        assert river_forcing.ds.identical(
            river_forcing_with_prescribed_single_cell_indices.ds
        )
        assert river_forcing == river_forcing_with_prescribed_single_cell_indices

    def test_invalid_indices(self, iceland_test_grid):
        invalid_single_cell_indices = {"Hvita(Olfusa)": [(0, 6)]}
        invalid_multi_cell_indices = {"Hvita(Olfusa)": [(8, 6), (0, 6)]}

        start_time = datetime(1998, 1, 1)
        end_time = datetime(1998, 3, 1)

        for indices in [invalid_single_cell_indices, invalid_multi_cell_indices]:
            with pytest.raises(
                ValueError, match="is not located on the coast at grid cell"
            ):
                RiverForcing(
                    grid=iceland_test_grid,
                    start_time=start_time,
                    end_time=end_time,
                    indices=indices,
                )
