import logging
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from conftest import calculate_file_hash
from roms_tools import Grid, RiverForcing
from roms_tools.constants import MAX_DISTINCT_COLORS
from roms_tools.setup.utils import get_tracer_defaults
from roms_tools.tests.rivr2o_test_utils import write_rivr2o_file

STANDARD_RIVER_FIXTURES = [
    "river_forcing",
    "river_forcing_with_bgc",
    "river_forcing_with_prescribed_multi_cell_indices",
]

INITIALIZATION_FIXTURES = [
    *STANDARD_RIVER_FIXTURES,
    "river_forcing_for_grid_that_straddles_dateline",
]

CLIMATOLOGY_FIXTURES = [
    "river_forcing",
    "river_forcing_climatology",
]

TRACER_FIXTURES = [
    "river_forcing_no_climatology",
    "river_forcing_with_bgc",
]

COAST_FIXTURES = [
    "river_forcing",
    "river_forcing_with_prescribed_multi_cell_indices",
]

ROUNDTRIP_FIXTURES = [
    "river_forcing_with_bgc",
    "river_forcing_with_prescribed_multi_cell_indices",
    "river_forcing_with_glofas",
]


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
    a grid that straddles the dateline.
    """
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
def river_forcing_for_gulf_of_mexico():
    """Fixture for creating a RiverForcing object for the Gulf of Mexico with 45 rivers."""
    grid = Grid(
        nx=20,
        ny=15,
        size_x=2000,
        size_y=1500,
        center_lon=-89,
        center_lat=24,
        rot=0,
        N=3,
    )
    start_time = datetime(2012, 1, 1)
    end_time = datetime(2012, 1, 31)

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
    detection.
    """
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
    detection.
    """
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
    @pytest.mark.parametrize("river_forcing_fixture", INITIALIZATION_FIXTURES)
    def test_successful_initialization(self, river_forcing_fixture, request):
        river_forcing = request.getfixturevalue(river_forcing_fixture)
        assert isinstance(river_forcing.ds, xr.Dataset)
        assert len(river_forcing.ds.nriver) > 0
        assert len(river_forcing.original_indices) > 0
        assert len(river_forcing.indices) > 0
        assert "river_volume" in river_forcing.ds
        assert "river_tracer" in river_forcing.ds
        assert "river_time" in river_forcing.ds

    @pytest.mark.parametrize("river_forcing_fixture", CLIMATOLOGY_FIXTURES)
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

    @pytest.mark.parametrize("river_forcing_fixture", TRACER_FIXTURES)
    def test_tracers_are_filled(self, river_forcing_fixture, request):
        river_forcing = request.getfixturevalue(river_forcing_fixture)
        assert (river_forcing.ds.river_tracer >= 0.0).all()
        # Test that tracers are constant across rivers and time.
        # Cast to float64 before std to avoid catastrophic cancellation in float32
        # arithmetic when summing O(100) identical large values (e.g. DIC ~1640).
        assert np.allclose(
            river_forcing.ds.river_tracer.astype(float).std(
                dim=["river_time", "nriver"]
            ),
            0.0,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_reproducibility_same_grid(self, river_forcing):
        the_same_river_forcing = RiverForcing(
            grid=river_forcing.grid,
            start_time=datetime(1998, 1, 1),
            end_time=datetime(1998, 3, 1),
        )

        assert river_forcing == the_same_river_forcing

    @pytest.mark.parametrize("river_forcing_fixture", COAST_FIXTURES)
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
            for eta_rho, xi_rho in indices[name]:
                assert coast[eta_rho, xi_rho]
                assert river_forcing.ds["river_index"][eta_rho, xi_rho] > 0
                assert river_forcing.ds["river_fraction"][eta_rho, xi_rho] > 0

    def test_missing_source_name(self, iceland_test_grid):
        with pytest.raises(ValueError, match="`source` must include a 'name'."):
            RiverForcing(
                grid=iceland_test_grid,
                start_time=datetime(1998, 1, 1),
                end_time=datetime(1998, 3, 1),
                source={"path": "river_data.nc"},
            )

    def test_invalid_convert_to_climatology(self, iceland_test_grid):
        with pytest.raises(ValueError, match="Invalid convert_to_climatology"):
            RiverForcing(
                grid=iceland_test_grid,
                start_time=datetime(1998, 1, 1),
                end_time=datetime(1998, 3, 1),
                convert_to_climatology="sometimes",
            )

    def test_river_forcing_plot(self, river_forcing_with_bgc):
        """Test plot methods with and without specifying river_names."""
        river_names = list(river_forcing_with_bgc.indices.keys())[0:2]

        # Test plot_locations
        river_forcing_with_bgc.plot_locations()
        river_forcing_with_bgc.plot_locations(river_names=river_names)

        # Fields to test
        variables = [
            "river_volume",
            "river_temp",
            "river_salt",
            "river_ALK",
            "river_PO4",
        ]

        for var in variables:
            river_forcing_with_bgc.plot(var)
            river_forcing_with_bgc.plot(var, river_names=river_names)

    def test_plot_max_releases(self, caplog, river_forcing_for_gulf_of_mexico):
        river_names = list(river_forcing_for_gulf_of_mexico.indices.keys())

        caplog.clear()
        with caplog.at_level("WARNING"):
            river_forcing_for_gulf_of_mexico.plot_locations()
        assert any(
            f"Only the first {MAX_DISTINCT_COLORS} rivers will be plotted" in message
            for message in caplog.messages
        )

        with caplog.at_level("WARNING"):
            river_forcing_for_gulf_of_mexico.plot(
                "river_volume", river_names=river_names
            )

        assert any(
            f"Only the first {MAX_DISTINCT_COLORS} rivers will be plotted" in message
            for message in caplog.messages
        )

    @pytest.mark.parametrize("river_forcing_fixture", ROUNDTRIP_FIXTURES)
    def test_river_forcing_save(self, river_forcing_fixture, tmp_path, request):
        """Test save method."""
        river_forcing = request.getfixturevalue(river_forcing_fixture)
        filepath = tmp_path / "test_rivers.nc"
        saved_filenames = river_forcing.save(filepath)
        assert saved_filenames == [filepath]
        assert filepath.exists()
        filepath.unlink()

    @pytest.mark.parametrize("river_forcing_fixture", ROUNDTRIP_FIXTURES)
    def test_roundtrip_yaml(self, river_forcing_fixture, request, tmp_path, caplog):
        """Test that creating an RiverForcing object, saving its parameters to yaml
        file, and re-opening yaml file creates the same object.
        """
        river_forcing = request.getfixturevalue(river_forcing_fixture)

        filepath = tmp_path / "test_yaml"
        river_forcing.to_yaml(filepath)

        caplog.clear()
        with caplog.at_level(logging.INFO):
            river_forcing_from_file = RiverForcing.from_yaml(filepath)

        assert "Use provided river indices." in caplog.text
        assert river_forcing == river_forcing_from_file
        filepath.unlink()

    @pytest.mark.parametrize("river_forcing_fixture", ROUNDTRIP_FIXTURES)
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
    start_time = datetime(1998, 1, 1)
    end_time = datetime(1998, 3, 1)

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
            RiverForcing(grid=grid, start_time=self.start_time, end_time=self.end_time)


class TestRiverForcingWithPrescribedIndices:
    start_time = datetime(1998, 1, 1)
    end_time = datetime(1998, 3, 1)

    @pytest.mark.parametrize(
        "indices_fixture", ["single_cell_indices", "multi_cell_indices"]
    )
    def test_indices_stay_untouched(self, indices_fixture, request, iceland_test_grid):
        indices = request.getfixturevalue(indices_fixture)

        river_forcing = RiverForcing(
            grid=iceland_test_grid,
            start_time=self.start_time,
            end_time=self.end_time,
            indices=indices,
        )
        assert river_forcing.original_indices == indices
        # indices may be reordered (volume sort) and extended (overlap synthetics);
        # verify all user-specified rivers are still present
        assert set(indices.keys()).issubset(set(river_forcing.indices.keys()))

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
            "river_fraction"
        ]
        is_integer = non_zero_values == np.floor(non_zero_values)
        assert (is_integer).all()

        # check that not all values are integers for multi cell rivers
        non_zero_values = river_forcing_with_prescribed_multi_cell_indices.ds[
            "river_fraction"
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

    @pytest.mark.parametrize(
        "indices,match",
        [
            ({"Hvita(Olfusa)": [(0, 6)]}, "is not located on the coast at grid cell"),
            (
                {"Hvita(Olfusa)": [(8, 6), (0, 6)]},
                "is not located on the coast at grid cell",
            ),
            (
                {"Hvita(Olfusa)": [(8, 6)], "fake": [(11, 12)]},
                "The following rivers were not found in the dataset",
            ),
            ("invalid", "`indices` must be a dictionary."),
            ({}, "The provided 'indices' dictionary must contain at least one river."),
            ({123: [(8, 6)]}, "River name `123` must be a string."),
            ({"Hvita(Olfusa)": "8, 6"}, "must be a list of tuples."),
            ({"Hvita(Olfusa)": [(8, 6, 7)]}, "must be a tuple of length 2"),
            ({"Hvita(Olfusa)": [("a", 6)]}, "First element of tuple for river"),
            ({"Hvita(Olfusa)": [(8, "b")]}, "Second element of tuple for river"),
            ({"Hvita(Olfusa)": [(20, 6)]}, "Value of eta_rho for river"),
            ({"Hvita(Olfusa)": [(8, 20)]}, "Value of xi_rho for river"),
            ({"Hvita(Olfusa)": [(8, 6), (8, 6)]}, "Duplicate location"),
        ],
    )
    def test_invalid_indices(self, iceland_test_grid, indices, match):
        with pytest.raises(ValueError, match=match):
            RiverForcing(
                grid=iceland_test_grid,
                start_time=self.start_time,
                end_time=self.end_time,
                indices=indices,
            )


class TestRiverForcingWithOverlappingIndices:
    def test_get_overlapping_rivers(self):
        indices = {
            "RiverA": [(1, 2), (3, 4)],
            "RiverB": [(1, 2), (5, 6)],
            "RiverC": [(7, 8)],
            "RiverD": [(3, 4)],
        }

        # Create a dummy RiverForcing instance with these indices
        rf = RiverForcing.__new__(RiverForcing)
        rf.indices = indices

        overlapping = rf._get_overlapping_rivers()

        expected = {
            (1, 2): ["RiverA", "RiverB"],
            (3, 4): ["RiverA", "RiverD"],
        }

        # Check keys
        assert set(overlapping.keys()) == set(expected.keys())

        # Check that all expected rivers are present for each overlapping cell
        for idx, rivers in expected.items():
            assert set(overlapping[idx]) == set(rivers)

        # Confirm non-overlapping indices are not present
        assert (5, 6) not in overlapping
        assert (7, 8) not in overlapping

    @pytest.fixture
    def mock_river_dataset_with_simple_overlap(self):
        """Creates a small mock dataset with overlapping rivers."""
        ntracers = 1
        river_names = ["RiverA", "RiverB", "RiverC"]
        tracer_names = ["temp"]
        nriver = len(river_names)

        ds = xr.Dataset(
            {
                "river_volume": (("time", "nriver"), [[2.0, 3.0, 4.0]]),
                "river_tracer": (
                    ("time", "ntracers", "nriver"),
                    [[[10.0, 20.0, 30.0]]],
                ),
            },
            coords={
                "time": [0],
                "ntracers": range(ntracers),
                "nriver": range(nriver),
                "river_name": ("nriver", river_names),
                "tracer_name": ("ntracers", tracer_names),
            },
        )

        # Assign overlapping indices to two rivers
        indices = {
            "RiverA": [(10, 20), (10, 21)],
            "RiverB": [(10, 20)],
            "RiverC": [(15, 25)],
        }

        return ds, indices

    @pytest.fixture
    def mock_river_dataset_with_complex_overlap(self):
        """Creates a small mock dataset with overlapping rivers."""
        ntracers = 1
        river_names = ["RiverA", "RiverB", "RiverC"]
        tracer_names = ["temp"]
        nriver = len(river_names)

        ds = xr.Dataset(
            {
                "river_volume": (("time", "nriver"), [[1.0, 2.0, 3.0]]),
                "river_tracer": (
                    ("time", "ntracers", "nriver"),
                    [[[10.0, 20.0, 30.0]]],
                ),
            },
            coords={
                "time": [0],
                "ntracers": range(ntracers),
                "nriver": range(nriver),
                "river_name": ("nriver", river_names),
                "tracer_name": ("ntracers", tracer_names),
            },
        )

        # Assign overlapping indices to two rivers
        indices = {
            "RiverA": [(10, 20), (10, 22), (11, 20)],
            "RiverB": [(10, 20), (10, 21), (10, 22)],
            "RiverC": [(15, 25), (10, 21), (11, 23)],
        }

        return ds, indices

    @pytest.mark.parametrize(
        "fixture_name, river_list, idx_pair, expected_volume, expected_tracer",
        [
            (
                "mock_river_dataset_with_simple_overlap",
                ["RiverA", "RiverB"],
                (10, 20),
                2.0 / 2 + 3.0 / 1,
                (10 * (2.0 / 2) + 20 * (3.0 / 1)) / (2.0 / 2 + 3.0 / 1),
            ),
        ],
    )
    def test_create_combined_river(
        self,
        request,
        fixture_name,
        river_list,
        idx_pair,
        expected_volume,
        expected_tracer,
    ):
        ds, indices = request.getfixturevalue(fixture_name)

        rf = RiverForcing.__new__(RiverForcing)
        rf.indices = indices

        name_to_idx = {n: i for i, n in enumerate(ds.river_name.values)}
        name = "overlap_" + sorted(river_list)[0].replace("GloFAS_", "")
        new_nriver = ds.sizes["nriver"] + 1  # first overlap group (i=0)

        combined_volume, combined_tracer = rf._create_combined_river(
            ds=ds,
            name=name,
            new_nriver=new_nriver,
            idx_pair=idx_pair,
            river_list=river_list,
            name_to_idx=name_to_idx,
        )

        assert combined_volume.sizes["nriver"] == 1
        assert combined_tracer.sizes["nriver"] == 1
        assert combined_volume.coords["river_name"].item() == name
        assert combined_tracer.coords["river_name"].item() == name

        np.testing.assert_allclose(combined_volume.values, expected_volume, rtol=1e-6)
        np.testing.assert_allclose(
            combined_tracer.values, [[[expected_tracer]]], rtol=1e-6
        )

    @pytest.mark.parametrize(
        "fixture_name, expected_volumes",
        [
            # RiverA has 2 cells, 1 overlap -> scale = 0.5
            # RiverB has 1 cell, 1 overlap -> scale = 0.0
            # RiverC has 1 cell, 0 overlap -> stays the same
            ("mock_river_dataset_with_simple_overlap", [1.0, 0.0, 4.0]),
            # RiverA has 3 cells, 2 overlaps -> scale = 1/3
            # RiverB has 3 cells, 3 overlaps -> scale = 0.0
            # RiverC has 3 cells, 1 overlap -> scale = 2/3
            (
                "mock_river_dataset_with_complex_overlap",
                [1.0 * (1 / 3), 0.0, 3.0 * (2 / 3)],
            ),
        ],
    )
    def test_reduce_river_volumes(self, request, fixture_name, expected_volumes):
        ds, indices = request.getfixturevalue(fixture_name)

        # Manually create RiverForcing instance with mock indices
        rf = RiverForcing.__new__(RiverForcing)
        rf.indices = indices

        # Compute overlapping rivers
        overlapping = rf._get_overlapping_rivers()

        # Apply volume reduction
        ds_out = rf._reduce_river_volumes(ds.copy(), overlapping)

        # Validate reduced volumes
        actual = ds_out["river_volume"].sel(time=0).values
        np.testing.assert_allclose(actual, expected_volumes, rtol=1e-6)

    @pytest.mark.parametrize(
        "mock_river_dataset_fixture, expected_synthetic_count",
        [
            ("mock_river_dataset_with_simple_overlap", 1),
            ("mock_river_dataset_with_complex_overlap", 3),
        ],
    )
    def test_handle_overlapping_rivers(
        self, mock_river_dataset_fixture, expected_synthetic_count, request
    ):
        ds, indices = request.getfixturevalue(mock_river_dataset_fixture)

        rf = RiverForcing.__new__(RiverForcing)
        rf.indices = indices
        rf.original_indices = indices.copy()
        rf.grid = None  # Not needed for this test

        ds_out = rf._handle_overlapping_rivers(ds)

        # Assert number of synthetic rivers added
        expected_nriver = ds.dims["nriver"] + expected_synthetic_count
        assert ds_out.dims["nriver"] == expected_nriver

        # Check no river's volume increased
        for name in ds["river_name"].values:
            river_idx = np.where(ds["river_name"].values == name)[0].item()
            assert float(ds_out["river_volume"].isel(nriver=river_idx)) <= float(
                ds["river_volume"].isel(nriver=river_idx)
            )

        # Check rivers that contribute to overlapping indices have strictly decreased volume
        overlapping = rf._get_overlapping_rivers()
        for name, idx_list in indices.items():
            if name in ds["river_name"].values and any(
                idx in overlapping for idx in idx_list
            ):
                river_idx = np.where(ds["river_name"].values == name)[0].item()
                assert float(ds_out["river_volume"].isel(nriver=river_idx)) < float(
                    ds["river_volume"].isel(nriver=river_idx)
                )

        # Check that total volume is preserved (within tolerance)
        np.testing.assert_allclose(
            ds_out["river_volume"].sum(dim="nriver").values,
            ds["river_volume"].sum(dim="nriver").values,
            rtol=1e-6,
        )

        # Check that the synthetic river volume is positive and tracer is not NaN
        synthetic_idx = ds_out.dims["nriver"] - 1
        assert float(ds_out["river_volume"].isel(nriver=synthetic_idx)) > 0
        assert not np.isnan(ds_out["river_tracer"].isel(nriver=synthetic_idx).item())

    def test_volume_sort_after_overlap(self, mock_river_dataset_with_simple_overlap):
        ds, indices = mock_river_dataset_with_simple_overlap

        rf = RiverForcing.__new__(RiverForcing)
        rf.indices = indices
        rf.original_indices = indices.copy()
        rf.grid = None

        ds_out = rf._handle_overlapping_rivers(ds)

        # Apply the same re-sort logic __post_init__ uses after overlap handling
        volume_means = ds_out["river_volume"].mean(dim="time")
        sorted_nriver = np.argsort(volume_means.values)[::-1]
        ds_sorted = ds_out.isel(nriver=sorted_nriver)

        means = ds_sorted["river_volume"].mean(dim="time").values
        assert all(means[i] >= means[i + 1] for i in range(len(means) - 1))

    def test_overlap_strip_from_provided_indices(
        self, iceland_test_grid, single_cell_indices
    ):
        # Simulate indices saved from a prior run that includes a synthetic river.
        # Constructing RiverForcing with these indices must not raise ValueError
        # about overlap_ not found in the source dataset.
        indices_with_overlap = {
            **single_cell_indices,
            "overlap_Bruara": [(8, 6)],
        }
        rf = RiverForcing(
            grid=iceland_test_grid,
            start_time=datetime(1998, 1, 1),
            end_time=datetime(1998, 3, 1),
            indices=indices_with_overlap,
        )
        assert len(rf.indices) > 0
        assert "overlap_Bruara" not in rf.original_indices


class TestRiverForcingBGCSource:
    def test_bgc_constants_by_default(self, iceland_test_grid, single_cell_indices):
        defaults = get_tracer_defaults()
        river_forcing = RiverForcing(
            grid=iceland_test_grid,
            start_time=datetime(1998, 1, 1),
            end_time=datetime(1998, 3, 1),
            indices=single_cell_indices,
            include_bgc=True,
        )

        assert river_forcing.bgc_source.model_dump() == {
            "name": "CONSTANTS",
            "fill": {"name": "CONSTANTS"},
        }
        tracer_names = list(river_forcing.ds.tracer_name.values)
        po4 = river_forcing.ds.river_tracer.isel(ntracers=tracer_names.index("PO4"))
        alk = river_forcing.ds.river_tracer.isel(ntracers=tracer_names.index("ALK"))
        np.testing.assert_allclose(float(po4.min()), defaults["PO4"], rtol=1e-6)
        np.testing.assert_allclose(float(alk.min()), defaults["ALK"], rtol=1e-6)

    def test_bgc_rivr2o_requires_path(self, iceland_test_grid, single_cell_indices):
        with pytest.raises(ValueError, match="path"):
            RiverForcing(
                grid=iceland_test_grid,
                start_time=datetime(1998, 1, 1),
                end_time=datetime(1998, 3, 1),
                indices=single_cell_indices,
                include_bgc=True,
                bgc_source={"name": "RIVR2O"},
            )

    def test_bgc_rivr2o_default_fill_source(
        self, tmp_path, iceland_test_grid, single_cell_indices
    ):
        lat = np.arange(55.0, 75.5, 0.5)
        lon = np.arange(325.0, 355.5, 0.5)
        path = tmp_path / "rivr2o_riverinputs_1998.nc"
        write_rivr2o_file(
            path,
            lat,
            lon,
            {
                "DIC": np.full((len(lat), len(lon)), 100.0),
                "DIN": np.full((len(lat), len(lon)), 100.0),
                "DOC_l": np.full((len(lat), len(lon)), 50.0),
                "DOC_sl": np.full((len(lat), len(lon)), 50.0),
                "POC": np.full((len(lat), len(lon)), 25.0),
                "DIP": np.full((len(lat), len(lon)), 100.0),
            },
        )
        river_forcing = RiverForcing(
            grid=iceland_test_grid,
            start_time=datetime(1998, 1, 1),
            end_time=datetime(1998, 3, 1),
            indices=single_cell_indices,
            include_bgc=True,
            bgc_source={"name": "RIVR2O", "path": str(path)},
        )
        assert river_forcing.bgc_source.fill.name == "CONSTANTS"

    def test_rivr2o_climatology_discharge_repeats_bgc_varies_by_year(
        self, tmp_path, iceland_test_grid, single_cell_indices
    ):
        """Dai climatology + RIVR2O: repeating Q, year-varying alkalinity (like plot)."""
        lat = np.arange(55.0, 75.5, 0.5)
        lon = np.arange(325.0, 355.5, 0.5)
        paths = []
        for year in (1998, 1999, 2000):
            export = float(year)
            tracer_values = {
                "DIC": np.full((len(lat), len(lon)), export),
                "DIN": np.full((len(lat), len(lon)), export),
                "DOC_l": np.full((len(lat), len(lon)), export / 2),
                "DOC_sl": np.full((len(lat), len(lon)), export / 2),
                "POC": np.full((len(lat), len(lon)), export / 4),
                "DIP": np.full((len(lat), len(lon)), export),
            }
            paths.append(
                tmp_path / f"rivr2o_riverinputs_{year}.nc",
            )
            write_rivr2o_file(paths[-1], lat, lon, tracer_values)

        river_forcing = RiverForcing(
            grid=iceland_test_grid,
            start_time=datetime(1998, 1, 1),
            end_time=datetime(2000, 12, 31),
            indices=single_cell_indices,
            include_bgc=True,
            bgc_source={"name": "RIVR2O", "path": [str(p) for p in paths]},
            convert_to_climatology="always",
        )

        assert river_forcing.climatology
        assert (
            str(river_forcing.ds.attrs.get("discharge_climatology")).lower() == "true"
        )
        assert river_forcing.ds.sizes["river_time"] == 36
        assert not hasattr(river_forcing.ds.river_time, "cycle_length")

        abs_time = river_forcing.ds["abs_time"]
        jan_steps = np.where(abs_time.dt.month.values == 1)[0]

        volume = river_forcing.ds["river_volume"].isel(nriver=0)
        np.testing.assert_allclose(
            float(volume.isel(river_time=jan_steps[0]).values),
            float(volume.isel(river_time=jan_steps[-1]).values),
            rtol=1e-6,
        )

        alk = river_forcing.ds.river_tracer.sel(
            ntracers=river_forcing.ds.tracer_name == "ALK"
        ).isel(nriver=0)
        alk_jan_first = float(alk.isel(river_time=jan_steps[0]).squeeze().values)
        alk_jan_last = float(alk.isel(river_time=jan_steps[-1]).squeeze().values)
        assert alk_jan_first != alk_jan_last


class TestRiverForcingRivr2oFromTestData:
    def test_dynamic_tracers_differ_from_defaults(self, river_forcing_with_rivr2o_bgc):
        defaults = get_tracer_defaults()
        river_forcing = river_forcing_with_rivr2o_bgc

        def tracer(name):
            return river_forcing.ds.river_tracer.sel(
                ntracers=river_forcing.ds.tracer_name == name
            ).squeeze("ntracers", drop=True)

        assert not np.allclose(tracer("DIC").values, defaults["DIC"])
        assert not np.allclose(tracer("ALK").values, defaults["ALK"])
        np.testing.assert_allclose(
            float(tracer("SiO3").min()), defaults["SiO3"], rtol=1e-6
        )

    def test_roundtrip_yaml(self, river_forcing_with_rivr2o_bgc, tmp_path):
        filepath = tmp_path / "river_forcing_rivr2o.yaml"
        river_forcing_with_rivr2o_bgc.to_yaml(filepath)
        restored = RiverForcing.from_yaml(filepath)
        assert river_forcing_with_rivr2o_bgc == restored


class TestRiverForcingWithGloFAS:
    def test_successful_initialization(self, river_forcing_with_glofas):
        rf = river_forcing_with_glofas
        assert isinstance(rf.ds, xr.Dataset)
        assert len(rf.ds.nriver) > 0
        assert len(rf.original_indices) > 0
        assert len(rf.indices) > 0
        assert "river_volume" in rf.ds
        assert "river_tracer" in rf.ds
        assert "river_time" in rf.ds

    def test_river_names_have_coordinate_format(self, river_forcing_with_glofas):
        rf = river_forcing_with_glofas
        for name in rf.indices:
            if name.startswith("overlap_"):
                continue
            assert name.startswith("GloFAS_"), f"{name!r} missing GloFAS_ prefix"
            parts = name.split("_")
            assert len(parts) >= 3, f"{name!r} does not match GloFAS_<lat>_<lon> format"
            assert parts[1][-1] in ("N", "S"), f"lat part {parts[1]!r} missing N/S"
            assert parts[2][-1] in ("E", "W"), f"lon part {parts[2]!r} missing E/W"

    def test_overlap_rivers_created(self, river_forcing_with_glofas):
        rf = river_forcing_with_glofas
        overlap_keys = [k for k in rf.indices if k.startswith("overlap_")]
        assert len(overlap_keys) >= 1, (
            "Expected at least one overlap_ river from two stations at the same coordinate"
        )

    def test_round_trip_yaml(self, river_forcing_with_glofas, tmp_path, caplog):
        filepath = tmp_path / "test_yaml_glofas"
        river_forcing_with_glofas.to_yaml(filepath)

        caplog.clear()
        with caplog.at_level(logging.INFO):
            river_forcing_from_file = RiverForcing.from_yaml(filepath)

        assert "Use provided river indices." in caplog.text
        assert river_forcing_with_glofas == river_forcing_from_file
        filepath.unlink()
