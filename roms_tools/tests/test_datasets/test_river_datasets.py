import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from roms_tools.datasets.river_datasets import (
    RIVR2O_FILL_VALUE,
    SECONDS_PER_YEAR,
    RiverDataset,
    Rivr2oRiverBGCDataset,
    fill_river_bgc_concentrations,
    rivr2o_coerce_time,
)
from roms_tools.setup.river_forcing import _mask_invalid_dynamic_bgc_concentrations
from roms_tools.setup.utils import interpolate_dynamic_bgc_by_calendar_year
from roms_tools.tests.rivr2o_test_utils import write_rivr2o_file


class TestRiverDataset:
    def test_deduplicate_river_names(self, tmp_path):
        sample_dim_and_var_names = {
            "dim_names": {"station": "station", "time": "time"},
            "var_names": {
                "latitude": "lat",
                "longitude": "lon",
                "flux": "flux",
                "ratio": "ratio",
                "name": "name",
            },
        }

        data = {
            "lat": (["station"], [10.0, 20.0, 30.0]),
            "lon": (["station"], [100.0, 110.0, 120.0]),
            "flux": (["time", "station"], np.random.rand(1, 3)),
            "ratio": (["time", "station"], np.random.rand(1, 3)),
            "name": (["station"], ["Amazon", "Nile", "Amazon"]),  # duplicate
        }
        coords = {"station": [0, 1, 2], "time": [0]}
        ds = xr.Dataset(data, coords=coords)

        # Write to temporary NetCDF file
        file_path = tmp_path / "rivers.nc"
        ds.to_netcdf(file_path)

        river_dataset = RiverDataset(
            filename=file_path,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2000, 1, 2),
            dim_names=sample_dim_and_var_names["dim_names"],
            var_names=sample_dim_and_var_names["var_names"],
        )

        names = river_dataset.ds["name"].values
        assert "Amazon_1" in names
        assert "Amazon_2" in names
        assert "Nile" in names
        assert len(set(names)) == len(names)  # all names must be unique


class TestFillRiverBGCConcentrations:
    @pytest.fixture
    def template(self):
        return xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=["river_time", "nriver"],
            coords={"river_time": [0, 1], "nriver": [0, 1]},
        )

    def test_missing_tracer_uses_fill(self, template):
        fill = {"DIC": 10.0, "SiO3": 5.0}
        merged = fill_river_bgc_concentrations({}, fill, ["DIC", "SiO3"], template)
        np.testing.assert_allclose(merged["DIC"].values, 10.0)
        np.testing.assert_allclose(merged["SiO3"].values, 5.0)

    def test_dynamic_overrides_fill(self, template):
        fill = {"DIC": 10.0, "SiO3": 5.0}
        dynamic = {"DIC": template * 2.0}
        merged = fill_river_bgc_concentrations(dynamic, fill, ["DIC", "SiO3"], template)
        np.testing.assert_allclose(merged["DIC"].values, template.values * 2.0)
        np.testing.assert_allclose(merged["SiO3"].values, 5.0)

    def test_nan_filled_from_defaults(self, template):
        fill = {"DIC": 10.0}
        dic = template.copy()
        dic.values[0, 1] = np.nan
        dynamic = {"DIC": dic}
        merged = fill_river_bgc_concentrations(
            dynamic, fill, ["DIC"], template, fill_at_nan=True
        )
        assert merged["DIC"].values[0, 0] == 1.0
        assert merged["DIC"].values[0, 1] == 10.0

    def test_fill_at_nan_false_preserves_nan(self, template):
        fill = {"DIC": 10.0}
        dic = template.copy()
        dic.values[0, 1] = np.nan
        dynamic = {"DIC": dic}
        merged = fill_river_bgc_concentrations(
            dynamic, fill, ["DIC"], template, fill_at_nan=False
        )
        assert np.isnan(merged["DIC"].values[0, 1])


class TestRivr2oRiverBGCDataset:
    @staticmethod
    def _make_files(tmp_path, years=(2000, 2001)):
        lat = np.array([-10.0, 10.0])
        lon = np.array([100.0, 110.0])
        files = []
        for i, year in enumerate(years):
            tracer_values = {
                "DIC": np.full((2, 2), 1.0 + i),
                "DIN": np.full((2, 2), 2.0 + i),
                "DOC_l": np.full((2, 2), 0.5 + i),
                "DOC_sl": np.full((2, 2), 0.25 + i),
                "POC": np.full((2, 2), 0.1 + i),
                "DIP": np.full((2, 2), 3.0 + i),
            }
            path = tmp_path / f"rivr2o_riverinputs_{year}.nc"
            write_rivr2o_file(path, lat, lon, tracer_values)
            files.append(path)
        return files, lat, lon

    def test_load_and_map_tracers(self, tmp_path):
        files, lat, lon = self._make_files(tmp_path)

        dataset = Rivr2oRiverBGCDataset(
            filename=files,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2001, 12, 31),
        )

        assert set(dataset.ds.data_vars) == {
            "DIC",
            "DOC_l",
            "DOC_sl",
            "POC",
            "NO3",
            "PO4",
        }
        assert dataset.ds.sizes["time"] == 2
        assert np.datetime64("2000-07-01") in dataset.ds.time.values
        assert np.datetime64("2001-07-01") in dataset.ds.time.values

        doc_l = dataset.ds["DOC_l"].isel(time=0, lat=0, lon=0).item()
        assert doc_l == 0.5

        no3 = dataset.ds["NO3"].isel(time=1, lat=0, lon=0).item()
        assert no3 == 3.0

    def test_wildcard_and_time_subset(self, tmp_path):
        self._make_files(tmp_path, years=(1999, 2000, 2001))

        dataset = Rivr2oRiverBGCDataset(
            filename=str(tmp_path / "rivr2o_riverinputs_*.nc"),
            start_time=datetime(2000, 6, 1),
            end_time=datetime(2000, 8, 1),
        )

        assert dataset.ds.sizes["time"] == 1
        assert dataset.ds.time.dt.year.item() == 2000

    def test_window_outside_loaded_files_raises_clear_error(self, tmp_path):
        files, _, _ = self._make_files(tmp_path, years=(2000, 2001))

        with pytest.raises(
            ValueError, match="No RIVR2O files cover the requested years 2010-2011"
        ):
            Rivr2oRiverBGCDataset(
                filename=files,
                start_time=datetime(2010, 1, 1),
                end_time=datetime(2011, 12, 31),
            )

    def test_fill_value_exposed_for_protocol(self, tmp_path):
        files, _, _ = self._make_files(tmp_path)
        dataset = Rivr2oRiverBGCDataset(
            filename=files,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2001, 12, 31),
        )
        assert dataset.fill_value == RIVR2O_FILL_VALUE

    def test_sample_at_points(self, tmp_path):
        files, lat, lon = self._make_files(tmp_path)

        dataset = Rivr2oRiverBGCDataset(
            filename=files,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2001, 12, 31),
        )

        sampled = dataset.sample_at_points(
            lon=lon[0],
            lat=lat[0],
        )

        assert sampled.sizes == {"time": 2}
        assert sampled["DIC"].isel(time=0).item() == 1.0
        assert sampled["DOC_sl"].isel(time=1).item() == 1.25

    def test_sample_at_points_nearest_nonzero_export(self, tmp_path):
        lat = np.array([0.0, 2.0])
        lon = np.array([0.0, 2.0])
        tracer_values = {
            "DIC": np.array([[0.0, 0.0], [0.0, 7.0]]),
            "DIN": np.array([[0.0, 0.0], [0.0, 7.0]]),
            "DOC_l": np.array([[0.0, 0.0], [0.0, 1.0]]),
            "DOC_sl": np.array([[0.0, 0.0], [0.0, 1.0]]),
            "POC": np.array([[0.0, 0.0], [0.0, 1.0]]),
            "DIP": np.array([[0.0, 0.0], [0.0, 7.0]]),
        }
        path = tmp_path / "rivr2o_riverinputs_2000.nc"
        write_rivr2o_file(path, lat, lon, tracer_values)

        dataset = Rivr2oRiverBGCDataset(
            filename=path,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2000, 12, 31),
        )

        # Closer to (0, 0) which is zero; nearest non-zero cell is (2, 2).
        sampled = dataset.sample_at_points(lon=0.1, lat=0.1)

        assert sampled["DIC"].isel(time=0).item() == 7.0

    def test_sample_at_points_same_dic_cell_all_years(self, tmp_path):
        """Nearest DIC cell is fixed in space; values vary along ``time``."""
        lat = np.array([0.0, 2.0])
        lon = np.array([0.0, 2.0])
        values_1999 = {
            "DIC": np.array([[5.0, 0.0], [0.0, 0.0]]),
            "DIN": np.array([[5.0, 0.0], [0.0, 0.0]]),
            "DOC_l": np.array([[1.0, 0.0], [0.0, 0.0]]),
            "DOC_sl": np.array([[1.0, 0.0], [0.0, 0.0]]),
            "POC": np.array([[1.0, 0.0], [0.0, 0.0]]),
            "DIP": np.array([[5.0, 0.0], [0.0, 0.0]]),
        }
        values_2000 = {
            "DIC": np.array([[0.0, 0.0], [0.0, 8.0]]),
            "DIN": np.array([[0.0, 0.0], [0.0, 8.0]]),
            "DOC_l": np.array([[0.0, 0.0], [0.0, 1.0]]),
            "DOC_sl": np.array([[0.0, 0.0], [0.0, 1.0]]),
            "POC": np.array([[0.0, 0.0], [0.0, 1.0]]),
            "DIP": np.array([[0.0, 0.0], [0.0, 8.0]]),
        }
        path_1999 = tmp_path / "rivr2o_riverinputs_1999.nc"
        path_2000 = tmp_path / "rivr2o_riverinputs_2000.nc"
        write_rivr2o_file(path_1999, lat, lon, values_1999)
        write_rivr2o_file(path_2000, lat, lon, values_2000)

        dataset = Rivr2oRiverBGCDataset(
            filename=[path_1999, path_2000],
            start_time=datetime(1999, 1, 1),
            end_time=datetime(2000, 12, 31),
        )

        sampled = dataset.sample_at_points(lon=0.1, lat=0.1)

        assert sampled["DIC"].isel(time=0).item() == 5.0
        assert sampled["DIC"].isel(time=1).item() == 0.0
        assert sampled["NO3"].isel(time=1).item() == 0.0

    def test_sample_at_points_returns_full_cell_export_on_shared_cell(self, tmp_path):
        lat = np.array([0.0, 2.0])
        lon = np.array([0.0, 2.0])
        tracer_values = {
            "DIC": np.array([[0.0, 0.0], [0.0, 10.0]]),
            "DIN": np.array([[0.0, 0.0], [0.0, 10.0]]),
            "DOC_l": np.array([[0.0, 0.0], [0.0, 2.0]]),
            "DOC_sl": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "POC": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "DIP": np.array([[0.0, 0.0], [0.0, 10.0]]),
        }
        path = tmp_path / "rivr2o_riverinputs_2000.nc"
        write_rivr2o_file(path, lat, lon, tracer_values)

        dataset = Rivr2oRiverBGCDataset(
            filename=path,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2000, 12, 31),
        )

        sampled = dataset.sample_at_points(
            lon=[0.1, 0.15],
            lat=[0.1, 0.12],
        )

        assert sampled["DIC"].isel(time=0, points=0).item() == 10.0
        assert sampled["DIC"].isel(time=0, points=1).item() == 10.0

    def test_nearest_dic_cell_warns_when_mouth_far_from_cell(self, tmp_path, caplog):
        lat = np.array([0.0, 2.0])
        lon = np.array([0.0, 2.0])
        tracer_values = {
            "DIC": np.array([[0.0, 0.0], [0.0, 10.0]]),
            "DIN": np.array([[0.0, 0.0], [0.0, 10.0]]),
            "DOC_l": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "DOC_sl": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "POC": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "DIP": np.array([[0.0, 0.0], [0.0, 10.0]]),
        }
        path = tmp_path / "rivr2o_riverinputs_2000.nc"
        write_rivr2o_file(path, lat, lon, tracer_values)

        dataset = Rivr2oRiverBGCDataset(
            filename=path,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2000, 12, 31),
        )

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            dataset.nearest_dic_cell_indices_for_points(
                lon=[-50.0],
                lat=[-50.0],
                river_names=["FarRiver"],
            )
        assert "RIVR2O DIC export cell for FarRiver" in caplog.text
        assert "km from the river mouth" in caplog.text

        near_path = tmp_path / "rivr2o_riverinputs_2001.nc"
        near_tracer_values = {
            **tracer_values,
            "DIC": np.array([[10.0, 0.0], [0.0, 0.0]]),
        }
        write_rivr2o_file(near_path, lat, lon, near_tracer_values)
        near_dataset = Rivr2oRiverBGCDataset(
            filename=near_path,
            start_time=datetime(2001, 1, 1),
            end_time=datetime(2001, 12, 31),
        )

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            near_dataset.nearest_dic_cell_indices_for_points(
                lon=[0.05],
                lat=[0.05],
                river_names=["NearRiver"],
            )
        assert "RIVR2O DIC export cell" not in caplog.text

    def test_discharge_partition_weights_with_fortran_nriver_index(self, tmp_path):
        """``nriver`` uses 1-based IDs like ``RiverForcing`` output."""
        lat = np.array([0.0, 2.0])
        lon = np.array([0.0, 2.0])
        tracer_values = {
            "DIC": np.array([[0.0, 0.0], [0.0, 10.0]]),
            "DIN": np.array([[0.0, 0.0], [0.0, 10.0]]),
            "DOC_l": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "DOC_sl": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "POC": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "DIP": np.array([[0.0, 0.0], [0.0, 10.0]]),
        }
        path = tmp_path / "rivr2o_riverinputs_2000.nc"
        write_rivr2o_file(path, lat, lon, tracer_values)

        dataset = Rivr2oRiverBGCDataset(
            filename=path,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2000, 12, 31),
        )

        n_time = 180
        river_volume = xr.DataArray(
            np.ones((n_time, 2)),
            dims=["river_time", "nriver"],
            coords={
                "river_time": np.arange(n_time),
                "nriver": [1, 2],
            },
        )
        nearest_lat, nearest_lon = dataset.nearest_dic_cell_indices_for_points(
            lon=[0.1, 0.15],
            lat=[0.1, 0.12],
        )
        weights = dataset.discharge_partition_weights(
            river_volume, nearest_lat, nearest_lon
        )

        assert list(weights.nriver.values) == [1, 2]
        assert weights.shape == (n_time, 2)
        assert weights.values.ndim == 2

    def test_discharge_partition_weights_match_concentrations(self, tmp_path):
        lat = np.array([0.0, 2.0])
        lon = np.array([0.0, 2.0])
        tracer_values = {
            "DIC": np.array([[0.0, 0.0], [0.0, 10.0]]),
            "DIN": np.array([[0.0, 0.0], [0.0, 10.0]]),
            "DOC_l": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "DOC_sl": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "POC": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "DIP": np.array([[0.0, 0.0], [0.0, 10.0]]),
        }
        path = tmp_path / "rivr2o_riverinputs_2000.nc"
        write_rivr2o_file(path, lat, lon, tracer_values)

        dataset = Rivr2oRiverBGCDataset(
            filename=path,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2000, 12, 31),
        )

        river_volume = xr.DataArray(
            [[100.0, 300.0]],
            dims=["river_time", "nriver"],
            coords={"river_time": [0], "nriver": [0, 1]},
        )
        nearest_lat, nearest_lon = dataset.nearest_dic_cell_indices_for_points(
            lon=[0.1, 0.15],
            lat=[0.1, 0.12],
        )
        weights = dataset.discharge_partition_weights(
            river_volume, nearest_lat, nearest_lon
        )
        sampled = dataset.sample_at_points(
            lon=[0.1, 0.15],
            lat=[0.1, 0.12],
        )

        weights_t0 = weights.isel(river_time=0).rename(nriver="points")
        export = sampled["DIC"].isel(time=0) * weights_t0
        mass_flux = export * 1e6 / SECONDS_PER_YEAR
        mmol_flux = mass_flux / 12.011 * 1000.0
        q_t0 = river_volume.isel(river_time=0).rename(nriver="points")
        conc = mmol_flux / q_t0
        np.testing.assert_allclose(
            conc.isel(points=0).item(), conc.isel(points=1).item(), rtol=1e-6
        )
        assert export.isel(points=0).item() == 2.5
        assert export.isel(points=1).item() == 7.5

    def test_discharge_partition_constant_concentration_each_month(self, tmp_path):
        lat = np.array([0.0, 2.0])
        lon = np.array([0.0, 2.0])
        tracer_values = {
            "DIC": np.array([[0.0, 0.0], [0.0, 10.0]]),
            "DIN": np.array([[0.0, 0.0], [0.0, 10.0]]),
            "DOC_l": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "DOC_sl": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "POC": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "DIP": np.array([[0.0, 0.0], [0.0, 10.0]]),
        }
        path = tmp_path / "rivr2o_riverinputs_2000.nc"
        write_rivr2o_file(path, lat, lon, tracer_values)

        dataset = Rivr2oRiverBGCDataset(
            filename=path,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2000, 12, 31),
        )

        river_volume = xr.DataArray(
            [[100.0, 300.0], [200.0, 600.0]],
            dims=["river_time", "nriver"],
            coords={"river_time": [0, 1], "nriver": [0, 1]},
        )
        nearest_lat, nearest_lon = dataset.nearest_dic_cell_indices_for_points(
            lon=[0.1, 0.15],
            lat=[0.1, 0.12],
        )
        weights = dataset.discharge_partition_weights(
            river_volume, nearest_lat, nearest_lon
        )
        sampled = dataset.sample_at_points(
            lon=[0.1, 0.15],
            lat=[0.1, 0.12],
        )
        cell_export = 10.0

        def mmol_m3(export, q):
            mass_flux = export * 1e6 / SECONDS_PER_YEAR
            return mass_flux / 12.011 * 1000.0 / q

        for t in (0, 1):
            weights_t = weights.isel(river_time=t).rename(nriver="points")
            export = sampled["DIC"].isel(time=0) * weights_t
            q = river_volume.isel(river_time=t).rename(nriver="points")
            conc = mmol_m3(export, q)
            np.testing.assert_allclose(
                conc.isel(points=0).item(), conc.isel(points=1).item(), rtol=1e-6
            )

        export_low = (cell_export * weights.isel(river_time=0, nriver=0)).item()
        export_high = (cell_export * weights.isel(river_time=1, nriver=0)).item()
        assert export_high > export_low

    def test_sample_at_points_ignores_cells_with_only_other_tracers(self, tmp_path):
        """Cell mask uses DIC only; DIP/NO3-only cells are not selected."""
        lat = np.array([0.0, 2.0])
        lon = np.array([0.0, 2.0])
        tracer_values = {
            "DIC": np.array([[0.0, 5.0], [0.0, RIVR2O_FILL_VALUE]]),
            "DIN": np.array([[0.0, 0.0], [0.0, 8.0]]),
            "DOC_l": np.array([[0.0, 1.0], [0.0, RIVR2O_FILL_VALUE]]),
            "DOC_sl": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "POC": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "DIP": np.array([[0.0, 0.0], [0.0, 8.0]]),
        }
        path = tmp_path / "rivr2o_riverinputs_2000.nc"
        write_rivr2o_file(path, lat, lon, tracer_values)

        dataset = Rivr2oRiverBGCDataset(
            filename=path,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2000, 12, 31),
        )

        sampled = dataset.sample_at_points(lon=0.1, lat=0.1)

        assert sampled["DIC"].isel(time=0).item() == 5.0
        assert sampled["NO3"].isel(time=0).item() == 0.0

    def test_rivr2o_coerce_time_accepts_int_nanoseconds(self):
        expected = np.datetime64("2000-01-15", "ns")
        as_int = int(expected.view("int64"))
        assert rivr2o_coerce_time(as_int) == expected
        assert rivr2o_coerce_time(datetime(2000, 1, 15)) == expected

    def test_time_clamped_before_min_year(self, tmp_path):
        self._make_files(tmp_path, years=(1903, 1904, 1905))

        dataset = Rivr2oRiverBGCDataset(
            filename=str(tmp_path / "rivr2o_riverinputs_*.nc"),
            start_time=datetime(1890, 1, 1),
            end_time=datetime(1895, 12, 31),
        )

        assert dataset.ds.sizes["time"] == 1
        assert dataset.ds.time.dt.year.item() == 1903

    def test_time_clamped_after_max_year(self, tmp_path):
        self._make_files(tmp_path, years=(2022, 2023, 2024))

        dataset = Rivr2oRiverBGCDataset(
            filename=str(tmp_path / "rivr2o_riverinputs_*.nc"),
            start_time=datetime(2025, 1, 1),
            end_time=datetime(2030, 12, 31),
        )

        assert dataset.ds.sizes["time"] == 1
        assert dataset.ds.time.dt.year.item() == 2024

    def test_forcing_concentrations_dims_and_marbl_mapping(self, tmp_path):
        lat = np.array([0.0, 2.0])
        lon = np.array([0.0, 2.0])
        export_value = 100.0
        tracer_values = {
            "DIC": np.array([[0.0, 0.0], [0.0, export_value]]),
            "DIN": np.array([[0.0, 0.0], [0.0, export_value]]),
            "DOC_l": np.array([[0.0, 0.0], [0.0, export_value / 2]]),
            "DOC_sl": np.array([[0.0, 0.0], [0.0, export_value / 2]]),
            "POC": np.array([[0.0, 0.0], [0.0, export_value / 4]]),
            "DIP": np.array([[0.0, 0.0], [0.0, export_value]]),
        }
        path = tmp_path / "rivr2o_riverinputs_1998.nc"
        write_rivr2o_file(path, lat, lon, tracer_values)

        dataset = Rivr2oRiverBGCDataset(
            filename=path,
            start_time=datetime(1998, 1, 1),
            end_time=datetime(1998, 3, 1),
        )
        assert dataset.requires_calendar_discharge_time is True

        river_volume = xr.DataArray(
            [[100.0], [200.0], [150.0]],
            dims=["river_time", "nriver"],
            coords={
                "river_time": [0, 1, 2],
                "nriver": [1],
            },
        )
        abs_time = xr.DataArray(
            np.array(
                [
                    np.datetime64("1998-01-15"),
                    np.datetime64("1998-02-15"),
                    np.datetime64("1998-03-15"),
                ]
            ),
            dims=["river_time"],
        )
        concentrations = dataset.forcing_concentrations(
            river_volume,
            abs_time,
            lons=np.array([0.1]),
            lats=np.array([0.1]),
            straddle=False,
            river_names=["test_river"],
        )

        assert set(concentrations) >= {
            "DIC",
            "DOC",
            "DON",
            "DOP",
            "ALK",
            "DIC_ALT_CO2",
            "ALK_ALT_CO2",
            "NO3",
            "PO4",
        }
        for values in concentrations.values():
            assert values.dims == ("river_time", "nriver")
            assert values.shape == (3, 1)

        weights = dataset.discharge_partition_weights(
            river_volume,
            *dataset.nearest_dic_cell_indices_for_points(
                np.array([0.1]), np.array([0.1])
            ),
        )

        def expected_conc(export):
            mass_flux = export * weights * 1e6 / SECONDS_PER_YEAR
            mmol_flux = mass_flux / 12.011 * 1000.0
            return (mmol_flux / river_volume).astype(np.float32)

        expected_dic_file = expected_conc(export_value)
        expected_doc_l = expected_conc(export_value / 2)
        expected_doc_sl = expected_conc(export_value / 2)
        expected_poc = expected_conc(export_value / 4)

        np.testing.assert_allclose(
            concentrations["DIC"].values,
            (expected_dic_file + expected_doc_l).values,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            concentrations["DOC"].values,
            (expected_doc_sl + expected_poc).values,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            concentrations["DON"].values,
            (expected_doc_sl * (103 / 2583) + expected_poc * (25 / 276)).values,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            concentrations["ALK"].values,
            expected_dic_file.values,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            concentrations["DIC_ALT_CO2"].values,
            concentrations["DIC"].values,
            rtol=1e-5,
        )

    def test_temporal_interpolation_is_calendar_year(self, tmp_path):
        path = tmp_path / "rivr2o_riverinputs_2000.nc"
        lat = np.array([0.0, 2.0])
        lon = np.array([0.0, 2.0])
        export = 1.0
        tracer_values = {
            "DIC": np.full((2, 2), export),
            "DIN": np.full((2, 2), export),
            "DOC_l": np.full((2, 2), export / 2),
            "DOC_sl": np.full((2, 2), export / 2),
            "POC": np.full((2, 2), export / 4),
            "DIP": np.full((2, 2), export),
        }
        write_rivr2o_file(path, lat, lon, tracer_values)
        dataset = Rivr2oRiverBGCDataset(
            filename=path,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2000, 12, 31),
        )
        assert dataset.temporal_interpolation == "calendar_year"

    @staticmethod
    def _uniform_tracer_values(export: float, shape: tuple[int, int]) -> dict:
        return {
            "DIC": np.full(shape, export),
            "DIN": np.full(shape, export),
            "DOC_l": np.full(shape, export / 2),
            "DOC_sl": np.full(shape, export / 2),
            "POC": np.full(shape, export / 4),
            "DIP": np.full(shape, export),
        }

    def test_missing_year_file_is_temporally_interpolated(self, tmp_path):
        lat = np.array([0.0, 2.0])
        lon = np.array([0.0, 2.0])
        path_2000 = tmp_path / "rivr2o_riverinputs_2000.nc"
        path_2002 = tmp_path / "rivr2o_riverinputs_2002.nc"
        write_rivr2o_file(
            path_2000, lat, lon, self._uniform_tracer_values(100.0, (2, 2))
        )
        write_rivr2o_file(
            path_2002, lat, lon, self._uniform_tracer_values(200.0, (2, 2))
        )

        dataset = Rivr2oRiverBGCDataset(
            filename=[path_2000, path_2002],
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2002, 12, 31),
        )
        abs_time = xr.DataArray(
            np.array(
                [datetime(y, m, 15) for y in (2000, 2001, 2002) for m in range(1, 13)],
                dtype="datetime64[ns]",
            ),
            dims=["river_time"],
        )
        river_volume = xr.DataArray(
            np.full((abs_time.size, 1), 100.0),
            dims=["river_time", "nriver"],
            coords={"river_time": np.arange(abs_time.size), "nriver": [0]},
        )
        dynamic = dataset.forcing_concentrations(
            river_volume,
            abs_time,
            lons=np.array([0.1]),
            lats=np.array([0.1]),
            straddle=False,
            river_names=["test_river"],
        )
        dynamic = _mask_invalid_dynamic_bgc_concentrations(
            dynamic, fill_value=RIVR2O_FILL_VALUE
        )
        dynamic = interpolate_dynamic_bgc_by_calendar_year(dynamic, abs_time)

        alk_2000 = float(
            dynamic["ALK"].isel(river_time=abs_time.dt.year == 2000).mean()
        )
        alk_2001 = float(
            dynamic["ALK"].isel(river_time=abs_time.dt.year == 2001).mean()
        )
        alk_2002 = float(
            dynamic["ALK"].isel(river_time=abs_time.dt.year == 2002).mean()
        )
        assert alk_2000 < alk_2001 < alk_2002
        np.testing.assert_allclose(alk_2001, (alk_2000 + alk_2002) / 2, rtol=1e-5)

    def test_fill_value_gap_year_is_temporally_interpolated(self, tmp_path):
        lat = np.array([0.0, 2.0])
        lon = np.array([0.0, 2.0])
        shape = (2, 2)
        paths = []
        for year, export in ((2000, 100.0), (2001, RIVR2O_FILL_VALUE), (2002, 200.0)):
            path = tmp_path / f"rivr2o_riverinputs_{year}.nc"
            write_rivr2o_file(
                path, lat, lon, self._uniform_tracer_values(export, shape)
            )
            paths.append(path)

        dataset = Rivr2oRiverBGCDataset(
            filename=paths,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2002, 12, 31),
        )
        abs_time = xr.DataArray(
            np.array(
                [datetime(y, m, 15) for y in (2000, 2001, 2002) for m in range(1, 13)],
                dtype="datetime64[ns]",
            ),
            dims=["river_time"],
        )
        river_volume = xr.DataArray(
            np.full((abs_time.size, 1), 100.0),
            dims=["river_time", "nriver"],
            coords={"river_time": np.arange(abs_time.size), "nriver": [0]},
        )
        dynamic = dataset.forcing_concentrations(
            river_volume,
            abs_time,
            lons=np.array([0.1]),
            lats=np.array([0.1]),
            straddle=False,
            river_names=["test_river"],
        )
        dynamic = _mask_invalid_dynamic_bgc_concentrations(
            dynamic, fill_value=RIVR2O_FILL_VALUE
        )
        dynamic = interpolate_dynamic_bgc_by_calendar_year(dynamic, abs_time)

        alk_2001 = float(
            dynamic["ALK"].isel(river_time=abs_time.dt.year == 2001).mean()
        )
        alk_2000 = float(
            dynamic["ALK"].isel(river_time=abs_time.dt.year == 2000).mean()
        )
        alk_2002 = float(
            dynamic["ALK"].isel(river_time=abs_time.dt.year == 2002).mean()
        )
        np.testing.assert_allclose(alk_2001, (alk_2000 + alk_2002) / 2, rtol=1e-5)


class TestRivr2oRiverBGCDatasetFromTestData:
    def test_load_list_and_wildcard(self, rivr2o_test_data_paths, tmp_path):
        dataset = Rivr2oRiverBGCDataset(
            filename=rivr2o_test_data_paths,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2002, 12, 31),
        )
        assert dataset.ds.sizes["time"] == 3
        assert set(dataset.ds.data_vars) >= {
            "DIC",
            "NO3",
            "PO4",
            "DOC_l",
            "DOC_sl",
            "POC",
        }
        assert float(dataset.ds.lat.min()) == 47.0
        assert float(dataset.ds.lon.max()) == 6.0

        for src in rivr2o_test_data_paths:
            (tmp_path / Path(src).name).symlink_to(src)
        wildcard_dataset = Rivr2oRiverBGCDataset(
            filename=str(tmp_path / "rivr2o_riverinputs_*.nc"),
            start_time=datetime(2000, 6, 1),
            end_time=datetime(2001, 8, 1),
        )
        assert wildcard_dataset.ds.sizes["time"] == 2

    def test_forcing_concentrations_finite_at_iceland(self, rivr2o_test_data_paths):
        dataset = Rivr2oRiverBGCDataset(
            filename=rivr2o_test_data_paths,
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2000, 12, 31),
        )
        abs_time = xr.DataArray(
            [datetime(2000, 1, 15), datetime(2000, 2, 15)],
            dims=["river_time"],
        )
        river_volume = xr.DataArray(
            [[100.0], [100.0]],
            dims=["river_time", "nriver"],
            coords={"river_time": [0, 1], "nriver": [0]},
        )
        concentrations = dataset.forcing_concentrations(
            river_volume,
            abs_time,
            lons=np.array([-10.0]),
            lats=np.array([65.0]),
            straddle=False,
            river_names=["test"],
        )
        assert np.isfinite(concentrations["DIC"].values).all()
        assert (concentrations["DIC"].values > 0).any()
