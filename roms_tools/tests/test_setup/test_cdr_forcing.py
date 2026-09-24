import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from conftest import calculate_file_hash
from roms_tools import CDRForcing, Grid, TracerPerturbation, VolumeRelease
from roms_tools.constants import MAX_DISTINCT_COLORS, NUM_TRACERS
from roms_tools.setup.cdr_forcing import (
    CDRForcingDatasetBuilder,
    ReleaseCollector,
    ReleaseSimulationManager,
)
from roms_tools.setup.cdr_release import ReleaseType
from roms_tools.setup.utils import get_tracer_metadata_dict

try:
    import xesmf  # type: ignore
except ImportError:
    xesmf = None


class TestReleaseSimulationManager:
    def setup_method(self):
        self.grid = Grid(
            nx=18,
            ny=18,
            size_x=800,
            size_y=800,
            center_lon=-18,
            center_lat=65,
            rot=0,
            N=3,
        )  # grid surrounding Iceland
        self.grid_that_straddles = Grid(
            nx=18,
            ny=18,
            size_x=800,
            size_y=800,
            center_lon=0,
            center_lat=65,
            rot=0,
            N=3,
        )  # grid that straddles dateline
        self.valid_iceland_release_location = {
            "lat": 66.0,
            "lon": -25.0,
            "depth": 50.0,
        }  # release location consistent with Iceland grid
        self.volume_release_without_times = VolumeRelease(
            name="vol_iceland_without_times",
            **self.valid_iceland_release_location,
            volume_fluxes=100.0,
        )
        self.volume_release_with_times = VolumeRelease(
            name="vol_iceland_with_times",
            **self.valid_iceland_release_location,
            times=[datetime(2022, 1, 1), datetime(2022, 1, 3), datetime(2022, 1, 5)],
            volume_fluxes=[1.0, 2.0, 3.0],
            tracer_concentrations={
                "DIC": [10.0, 20.0, 30.0],
                "temp": 10.0,
                "salt": 35.0,
            },
        )
        self.tracer_perturbation_without_times = TracerPerturbation(
            name="pert_iceland_without_times",
            **self.valid_iceland_release_location,
            tracer_fluxes={"ALK": 100.0},
        )
        self.tracer_perturbation_with_times = TracerPerturbation(
            name="pert_iceland_with_times",
            **self.valid_iceland_release_location,
            times=[datetime(2022, 1, 1), datetime(2022, 1, 3), datetime(2022, 1, 5)],
            tracer_fluxes={
                "DIC": [10.0, 20.0, 30.0],
            },
        )
        self.start_time = datetime(2022, 1, 1)
        self.end_time = datetime(2022, 12, 31)

    def test_volume_release_correctly_extended(self):
        # Save copies of mutable fields before they are modified by ReleaseSimulationManager
        times = self.volume_release_with_times.times.copy()  # list
        volume_fluxes = (
            self.volume_release_with_times.volume_fluxes.values.copy()
        )  # list
        tracer_concentrations_dic = (
            self.volume_release_with_times.tracer_concentrations["DIC"].values.copy()
        )  # list
        tracer_concentrations_temp = (
            self.volume_release_with_times.tracer_concentrations["temp"].values
        )  # float, no copy needed

        ReleaseSimulationManager(
            release=self.volume_release_with_times,
            grid=None,
            start_time=self.start_time,
            end_time=self.end_time,
        )

        # check that release was properly extended to end points
        assert self.volume_release_with_times.times == [*times, self.end_time]
        assert self.volume_release_with_times.volume_fluxes.values == [
            *volume_fluxes,
            0.0,
        ]
        assert self.volume_release_with_times.tracer_concentrations["DIC"].values == [
            *tracer_concentrations_dic,
            tracer_concentrations_dic[-1],
        ]
        assert self.volume_release_with_times.tracer_concentrations[
            "temp"
        ].values == 4 * [tracer_concentrations_temp]

    def test_tracer_perturbation_correctly_extended(self):
        # Save copies of mutable fields before they are modified by ReleaseSimulationManager
        times = self.tracer_perturbation_with_times.times.copy()  # list
        tracer_fluxes_dic = self.tracer_perturbation_with_times.tracer_fluxes[
            "DIC"
        ].values.copy()  # list
        tracer_fluxes_alk = self.tracer_perturbation_with_times.tracer_fluxes[
            "ALK"
        ].values  # float, no copy needed

        ReleaseSimulationManager(
            release=self.tracer_perturbation_with_times,
            grid=None,
            start_time=self.start_time,
            end_time=self.end_time,
        )

        # check that release was properly extended to end points
        assert self.tracer_perturbation_with_times.times == [*times, self.end_time]
        assert self.tracer_perturbation_with_times.tracer_fluxes["DIC"].values == [
            *tracer_fluxes_dic,
            0.0,
        ]
        assert self.tracer_perturbation_with_times.tracer_fluxes["ALK"].values == 4 * [
            tracer_fluxes_alk
        ]

    def test_release_starts_too_early(self):
        for release in [
            self.volume_release_with_times,
            self.tracer_perturbation_with_times,
        ]:
            times = release.times
            start_time = times[0] + timedelta(days=1)
            end_time = times[-1] + timedelta(days=1)

            with pytest.raises(ValueError, match="before start_time"):
                ReleaseSimulationManager(
                    release=release,
                    grid=None,
                    start_time=start_time,
                    end_time=end_time,
                )

    def test_release_ends_too_late(self):
        for release in [
            self.volume_release_with_times,
            self.tracer_perturbation_with_times,
        ]:
            times = release.times
            start_time = times[0] - timedelta(days=1)
            end_time = times[-1] - timedelta(days=1)

            with pytest.raises(ValueError, match="after end_time"):
                ReleaseSimulationManager(
                    release=release,
                    grid=None,
                    start_time=start_time,
                    end_time=end_time,
                )

    def test_warning_no_grid(self, caplog):
        for release in [
            self.volume_release_with_times,
            self.tracer_perturbation_with_times,
        ]:
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                ReleaseSimulationManager(
                    release=release,
                    grid=None,
                    start_time=self.start_time,
                    end_time=self.end_time,
                )

            assert "Grid not provided" in caplog.text

    def test_invalid_release_longitude(self):
        """Test that error is raised if release location is outside grid."""
        # Define release location both outside of Iceland grid and grid that straddles dateline
        lon0 = -30
        lat0 = 60
        depth0 = 0

        for lon in [lon0, lon0 - 360, lon0 + 360]:
            params = {"lon": lon, "lat": lat0, "depth": depth0}

            for grid in [self.grid, self.grid_that_straddles]:
                for release in [
                    VolumeRelease(name="vol", **params),
                    TracerPerturbation(name="vol", **params),
                ]:
                    with pytest.raises(ValueError, match="outside of the grid domain"):
                        ReleaseSimulationManager(
                            release=release,
                            grid=grid,
                            start_time=self.start_time,
                            end_time=self.end_time,
                        )

    def test_invalid_release_location(self):
        """Test that error is raised if release location is outside grid or on land."""
        # Release location too close to boundary of Iceland domain; lat_rho[0, 0] = 60.97, lon_rho[0, 0] = 334.17
        params = {"lon": 334.17, "lat": 60.97, "depth": 0.0}
        for release in [
            VolumeRelease(name="vol", **params),
            TracerPerturbation(name="pert", **params),
        ]:
            with pytest.raises(ValueError, match="too close to the grid boundary"):
                ReleaseSimulationManager(
                    release=release,
                    grid=self.grid,
                    start_time=self.start_time,
                    end_time=self.end_time,
                )

        # Release location lies on land
        params = {"lon": -20, "lat": 64.5, "depth": 0.0}
        for release in [
            VolumeRelease(name="vol", **params),
            TracerPerturbation(name="vol", **params),
        ]:
            with pytest.raises(ValueError, match="on land"):
                ReleaseSimulationManager(
                    release=release,
                    grid=self.grid,
                    start_time=self.start_time,
                    end_time=self.end_time,
                )

        # Release location lies below seafloor
        invalid_depth = 4000

        for valid_release in [
            self.volume_release_without_times,
            self.tracer_perturbation_without_times,
        ]:
            params = {
                "lon": valid_release.lon,
                "lat": valid_release.lat,
                "depth": invalid_depth,
            }

            if isinstance(valid_release, VolumeRelease):
                release = VolumeRelease(name="vol", **params)
            elif isinstance(valid_release, TracerPerturbation):
                release = VolumeRelease(name="pert", **params)
                with pytest.raises(ValueError, match="below the seafloor"):
                    ReleaseSimulationManager(
                        release=release,
                        grid=self.grid,
                        start_time=self.start_time,
                        end_time=self.end_time,
                    )


class TestReleaseCollector:
    def setup_method(self):
        self.volume_release = VolumeRelease(
            name="vol", lat=66, lon=-25, depth=50, volume_fluxes=100
        )
        self.another_volume_release = VolumeRelease(
            name="vol2", lat=66, lon=-25, depth=50, volume_fluxes=100
        )
        self.tracer_perturbation = TracerPerturbation(
            name="pert", lat=66, lon=-25, depth=50, tracer_fluxes={"ALK": 100}
        )
        self.another_tracer_perturbation = TracerPerturbation(
            name="pert2", lat=66, lon=-25, depth=50, tracer_fluxes={"ALK": 100}
        )

    def test_check_unique_name(self):
        with pytest.raises(ValidationError):
            ReleaseCollector(releases=[self.volume_release, self.volume_release])
        with pytest.raises(ValidationError):
            ReleaseCollector(
                releases=[self.tracer_perturbation, self.tracer_perturbation]
            )

    def test_raises_inconsistent_release_type(self):
        with pytest.raises(
            ValidationError, match="Not all releases have the same type"
        ):
            ReleaseCollector(releases=[self.volume_release, self.tracer_perturbation])

    def test_determine_release_type(self):
        """Test that release type is correctly inferred."""
        collector = ReleaseCollector(releases=[self.volume_release])
        assert collector.release_type == ReleaseType.volume

        collector = ReleaseCollector(
            releases=[self.volume_release, self.another_volume_release]
        )
        assert collector.release_type == ReleaseType.volume

        collector = ReleaseCollector(releases=[self.tracer_perturbation])
        assert collector.release_type == ReleaseType.tracer_perturbation

        collector = ReleaseCollector(
            releases=[self.tracer_perturbation, self.another_tracer_perturbation]
        )
        assert collector.release_type == ReleaseType.tracer_perturbation

    def test_determine_tracer_set(self):
        collector = ReleaseCollector(releases=[self.volume_release])
        assert collector.tracer_set == "marbl"
        assert collector.tracer_sets == {"marbl"}

        cdr = TracerPerturbation(
            name="cdr",
            lat=66,
            lon=-25,
            depth=50,
            tracer_set="cdr_lite",
            tracer_fluxes={"ALK": 100.0},
        )
        collector = ReleaseCollector(releases=[cdr])
        assert collector.tracer_set == "cdr_lite"

    def test_mixed_tracer_sets_allowed(self):
        marbl = TracerPerturbation(
            name="marbl", lat=66, lon=-25, depth=50, tracer_fluxes={"ALK": 100.0}
        )
        cdr = TracerPerturbation(
            name="cdr",
            lat=66,
            lon=-25,
            depth=50,
            tracer_set="cdr_lite",
            tracer_fluxes={"ALK": 100.0},
        )
        collector = ReleaseCollector(releases=[marbl, cdr])
        assert collector.tracer_set == "mixed"
        assert collector.tracer_sets == {"marbl", "cdr_lite"}


class TestCDRForcingDatasetBuilder:
    def setup_method(self):
        self.start_time = datetime(2022, 1, 1)
        self.end_time = datetime(2022, 12, 31)

        first_volume_release = VolumeRelease(
            name="first_release",
            lat=66.0,
            lon=-25.0,
            depth=50.0,
            times=[datetime(2022, 1, 1), datetime(2022, 1, 3), datetime(2022, 1, 5)],
            volume_fluxes=[1.0, 2.0, 3.0],
            tracer_concentrations={
                "DIC": [10.0, 20.0, 30.0],
                "temp": 10.0,
                "salt": 35.0,
            },
        )

        second_volume_release = VolumeRelease(
            name="second_release",
            lon=first_volume_release.lon - 1,
            lat=first_volume_release.lat - 1,
            depth=first_volume_release.depth - 1,
            times=[
                datetime(2022, 1, 2),
                datetime(2022, 1, 4),
                datetime(2022, 1, 5),
            ],
            volume_fluxes=[2.0, 4.0, 10.0],
            tracer_concentrations={"DIC": [20.0, 40.0, 100.0]},
        )

        first_tracer_perturbation = TracerPerturbation(
            name="first_release",
            lat=66.0,
            lon=-25.0,
            depth=50.0,
            times=[datetime(2022, 1, 1), datetime(2022, 1, 3), datetime(2022, 1, 5)],
            tracer_fluxes={
                "DIC": [10.0, 20.0, 30.0],
            },
        )

        second_tracer_perturbation = TracerPerturbation(
            name="second_release",
            lon=first_tracer_perturbation.lon - 1,
            lat=first_tracer_perturbation.lat - 1,
            depth=first_tracer_perturbation.depth - 1,
            times=[
                datetime(2022, 1, 2),
                datetime(2022, 1, 4),
                datetime(2022, 1, 5),
            ],
            tracer_fluxes={"DIC": [20.0, 40.0, 100.0]},
        )

        # Modify all releases including extending it to the endpoints
        for release in [
            first_volume_release,
            second_volume_release,
            first_tracer_perturbation,
            second_tracer_perturbation,
        ]:
            ReleaseSimulationManager(
                release=release,
                start_time=self.start_time,
                end_time=self.end_time,
            )

        self.first_volume_release = first_volume_release
        self.second_volume_release = second_volume_release
        self.first_tracer_perturbation = first_tracer_perturbation
        self.second_tracer_perturbation = second_tracer_perturbation

    def check_ds_dims_and_coords(
        self, ds, num_times, num_releases, release_type=VolumeRelease
    ):
        """Assert expected dimensions and coordinates for a CDR dataset."""
        # Dimensions
        assert ds.time.size == num_times
        assert ds.ncdr.size == num_releases
        assert ds.ntracers.size == NUM_TRACERS

        # Coordinates and metadata
        assert ds.release_name.size == num_releases
        assert ds.tracer_name.size == NUM_TRACERS
        assert ds.tracer_unit.size == NUM_TRACERS
        assert ds.tracer_long_name.size == NUM_TRACERS
        assert ds.cdr_time.size == num_times
        assert ds.cdr_lon.size == num_releases
        assert ds.cdr_lat.size == num_releases
        assert ds.cdr_dep.size == num_releases
        assert ds.cdr_hsc.size == num_releases
        assert ds.cdr_vsc.size == num_releases

        if release_type == VolumeRelease:
            assert ds.cdr_volume.shape == (num_times, num_releases)
            assert ds.cdr_tracer.shape == (num_times, NUM_TRACERS, num_releases)
        elif release_type == TracerPerturbation:
            assert ds.cdr_trcflx.shape == (num_times, NUM_TRACERS, num_releases)

    def check_ds_name_and_location(self, ds, release, ncdr_index):
        """Assert expected release name and location for a CDR dataset."""
        # Name
        assert release.name in ds["release_name"].values

        # Location
        assert ds["cdr_lon"].isel(ncdr=ncdr_index).values == release.lon
        assert ds["cdr_lat"].isel(ncdr=ncdr_index).values == release.lat
        assert ds["cdr_dep"].isel(ncdr=ncdr_index).values == release.depth
        assert ds["cdr_hsc"].isel(ncdr=ncdr_index).values == release.hsc
        assert ds["cdr_vsc"].isel(ncdr=ncdr_index).values == release.vsc

        # TODO: Check for tracer metadata

    def test_build_with_single_volume_release(self):
        builder = CDRForcingDatasetBuilder(
            releases=[self.first_volume_release],
            model_reference_date=datetime(2000, 1, 1),
            release_type=ReleaseType.volume,
        )
        ds = builder.build()

        num_times = len(self.first_volume_release.times)
        num_releases = 1
        self.check_ds_dims_and_coords(
            ds, num_times, num_releases, release_type=builder.release_type
        )

        ncdr_index = 0
        self.check_ds_name_and_location(ds, self.first_volume_release, ncdr_index)

        # Time values
        assert np.array_equal(
            ds["time"].values,
            np.array(self.first_volume_release.times, dtype="datetime64[ns]"),
        )

        # Volume flux values
        np.testing.assert_allclose(
            ds.cdr_volume.isel(ncdr=ncdr_index).values,
            self.first_volume_release.volume_fluxes.values,
        )

        # Tracer concentration values
        tracer_index = {name: i for i, name in enumerate(ds.tracer_name.values)}
        for tracer, expected in self.first_volume_release.tracer_concentrations.items():
            i = tracer_index[tracer]
            np.testing.assert_allclose(
                ds.cdr_tracer.isel(ncdr=ncdr_index, ntracers=i), expected.values
            )

    def test_build_with_single_tracer_perturbation(self):
        builder = CDRForcingDatasetBuilder(
            releases=[self.first_tracer_perturbation],
            model_reference_date=datetime(2000, 1, 1),
            release_type=ReleaseType.tracer_perturbation,
        )
        ds = builder.build()

        num_times = len(self.first_tracer_perturbation.times)
        num_releases = 1
        self.check_ds_dims_and_coords(
            ds, num_times, num_releases, release_type=builder.release_type
        )

        ncdr_index = 0
        self.check_ds_name_and_location(ds, self.first_tracer_perturbation, ncdr_index)

        # Time values
        assert np.array_equal(
            ds["time"].values,
            np.array(self.first_tracer_perturbation.times, dtype="datetime64[ns]"),
        )

        # Tracer flux values
        tracer_index = {name: i for i, name in enumerate(ds.tracer_name.values)}
        for tracer, expected in self.first_tracer_perturbation.tracer_fluxes.items():
            i = tracer_index[tracer]
            np.testing.assert_allclose(
                ds.cdr_trcflx.isel(ncdr=ncdr_index, ntracers=i), expected.values
            )

    def test_build_with_multiple_volume_releases(self):
        builder = CDRForcingDatasetBuilder(
            releases=[self.first_volume_release, self.second_volume_release],
            model_reference_date=datetime(2000, 1, 1),
            release_type=ReleaseType.volume,
        )
        ds = builder.build()

        # expected times is the union of the times of the first and second release without duplication
        expected_times = [
            datetime(2022, 1, 1),
            datetime(2022, 1, 2),
            datetime(2022, 1, 3),
            datetime(2022, 1, 4),
            datetime(2022, 1, 5),
            datetime(2022, 12, 31),
        ]
        num_times = len(expected_times)
        num_releases = 2
        self.check_ds_dims_and_coords(
            ds, num_times, num_releases, release_type=builder.release_type
        )

        self.check_ds_name_and_location(ds, self.first_volume_release, 0)
        self.check_ds_name_and_location(ds, self.second_volume_release, 1)

        # Time values
        assert np.array_equal(
            ds["time"].values, np.array(expected_times, dtype="datetime64[ns]")
        )

        # Volume flux values first release
        ncdr_index = 0
        expected_volume_fluxes = [1.0, 1.5, 2.0, 2.5, 3.0, 0.0]
        assert np.allclose(
            ds["cdr_volume"].isel(ncdr=ncdr_index).values,
            np.array(expected_volume_fluxes),
        )

        # Volume flux values second release
        ncdr_index = 1
        expected_volume_fluxes = [0.0, 2.0, 3.0, 4.0, 10.0, 0.0]
        assert np.allclose(
            ds["cdr_volume"].isel(ncdr=ncdr_index).values,
            np.array(expected_volume_fluxes),
        )

        # Tracer concentration values first release
        ncdr_index = 0
        dic_index = 9

        expected_dics = [10.0, 15.0, 20.0, 25.0, 30.0, 30.0]
        assert np.allclose(
            ds["cdr_tracer"].isel(ncdr=ncdr_index, ntracers=dic_index).values,
            np.array(expected_dics),
        )

        # Tracer concentration values second release
        ncdr_index = 1

        expected_dics = [20.0, 20.0, 30.0, 40.0, 100.0, 100.0]
        assert np.allclose(
            ds["cdr_tracer"].isel(ncdr=ncdr_index, ntracers=dic_index).values,
            np.array(expected_dics),
        )

    def test_build_with_multiple_tracer_perturbations(self):
        builder = CDRForcingDatasetBuilder(
            releases=[self.first_tracer_perturbation, self.second_tracer_perturbation],
            model_reference_date=datetime(2000, 1, 1),
            release_type=ReleaseType.tracer_perturbation,
        )
        ds = builder.build()

        # expected times is the union of the times of the first and second release without duplication
        expected_times = [
            datetime(2022, 1, 1),
            datetime(2022, 1, 2),
            datetime(2022, 1, 3),
            datetime(2022, 1, 4),
            datetime(2022, 1, 5),
            datetime(2022, 12, 31),
        ]
        num_times = len(expected_times)
        num_releases = 2
        self.check_ds_dims_and_coords(
            ds, num_times, num_releases, release_type=builder.release_type
        )

        self.check_ds_name_and_location(ds, self.first_tracer_perturbation, 0)
        self.check_ds_name_and_location(ds, self.second_tracer_perturbation, 1)

        # Time values
        assert np.array_equal(
            ds["time"].values, np.array(expected_times, dtype="datetime64[ns]")
        )

        # Tracer flux values first release
        ncdr_index = 0
        dic_index = 9

        expected_dics = [10.0, 15.0, 20.0, 25.0, 30.0, 0.0]
        assert np.allclose(
            ds["cdr_trcflx"].isel(ncdr=ncdr_index, ntracers=dic_index).values,
            np.array(expected_dics),
        )

        # Tracer flux values second release
        ncdr_index = 1

        expected_dics = [0.0, 20.0, 30.0, 40.0, 100.0, 0.0]
        assert np.allclose(
            ds["cdr_trcflx"].isel(ncdr=ncdr_index, ntracers=dic_index).values,
            np.array(expected_dics),
        )


class TestCDRForcing:
    def setup_method(self):
        self.start_time = datetime(2022, 1, 1)
        self.end_time = datetime(2022, 12, 31)

        first_volume_release = VolumeRelease(
            name="first_release",
            lat=66.0,
            lon=-25.0,
            depth=50.0,
            hsc=0.0,
            vsc=0.0,
            times=[datetime(2022, 1, 1), datetime(2022, 1, 3), datetime(2022, 1, 5)],
            volume_fluxes=[1.0, 2.0, 3.0],
            tracer_concentrations={
                "DIC": [10.0, 20.0, 30.0],
                "temp": 10.0,
                "salt": 35.0,
            },
        )

        second_volume_release = VolumeRelease(
            name="second_release",
            lon=first_volume_release.lon + 360,
            lat=first_volume_release.lat,
            depth=first_volume_release.depth,
            hsc=40000.0,
            vsc=0.0,
            times=[
                datetime(2022, 1, 2),
                datetime(2022, 1, 4),
                datetime(2022, 1, 5),
            ],
            volume_fluxes=[2.0, 4.0, 10.0],
            tracer_concentrations={"DIC": [20.0, 40.0, 100.0]},
        )

        first_tracer_perturbation = TracerPerturbation(
            name="first_release",
            lat=66.0,
            lon=-25.0,
            depth=50.0,
            hsc=40000.0,
            vsc=100.0,
            times=[datetime(2022, 1, 1), datetime(2022, 1, 3), datetime(2022, 1, 5)],
            tracer_fluxes={
                "DIC": [10.0, 20.0, 30.0],
            },
        )

        second_tracer_perturbation = TracerPerturbation(
            name="second_release",
            lon=first_tracer_perturbation.lon + 360,
            lat=first_tracer_perturbation.lat,
            depth=first_tracer_perturbation.depth,
            hsc=0.0,
            vsc=10.0,
            times=[
                datetime(2022, 1, 2),
                datetime(2022, 1, 4),
                datetime(2022, 1, 5),
            ],
            tracer_fluxes={"DIC": [20.0, 40.0, 100.0]},
        )

        # Modify all releases including extending it to the endpoints
        for release in [
            first_volume_release,
            second_volume_release,
            first_tracer_perturbation,
            second_tracer_perturbation,
        ]:
            ReleaseSimulationManager(
                release=release,
                start_time=self.start_time,
                end_time=self.end_time,
            )

        self.first_volume_release = first_volume_release
        self.second_volume_release = second_volume_release
        self.first_tracer_perturbation = first_tracer_perturbation
        self.second_tracer_perturbation = second_tracer_perturbation

        self.volume_release_cdr_forcing_without_grid = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=[self.first_volume_release, self.second_volume_release],
        )
        self.tracer_perturbation_cdr_forcing_without_grid = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=[self.first_tracer_perturbation, self.second_tracer_perturbation],
        )

        grid = Grid(
            nx=18,
            ny=18,
            size_x=800,
            size_y=800,
            center_lon=-18,
            center_lat=65,
            rot=0,
            N=3,
        )
        self.grid = grid

        grid_that_straddles = Grid(
            nx=18,
            ny=18,
            size_x=2500,
            size_y=2500,
            center_lon=0,
            center_lat=65,
            rot=0,
            N=3,
        )  # grid that straddles dateline

        self.volume_release_cdr_forcing = CDRForcing(
            grid=grid,
            start_time=self.start_time,
            end_time=self.end_time,
            releases=[self.first_volume_release, self.second_volume_release],
        )
        self.volume_release_cdr_forcing_with_straddling_grid = CDRForcing(
            grid=grid_that_straddles,
            start_time=self.start_time,
            end_time=self.end_time,
            releases=[self.first_volume_release, self.second_volume_release],
        )
        self.tracer_perturbation_cdr_forcing = CDRForcing(
            grid=grid,
            start_time=self.start_time,
            end_time=self.end_time,
            releases=[self.first_tracer_perturbation, self.second_tracer_perturbation],
        )
        self.tracer_perturbation_cdr_forcing_with_straddling_grid = CDRForcing(
            grid=grid_that_straddles,
            start_time=self.start_time,
            end_time=self.end_time,
            releases=[self.first_tracer_perturbation, self.second_tracer_perturbation],
        )

    def test_inconsistent_start_end_time(self):
        start_time = datetime(2022, 5, 1)
        end_time = datetime(2022, 5, 1)
        with pytest.raises(ValueError, match="must be earlier"):
            CDRForcing(
                start_time=start_time,
                end_time=end_time,
                releases=[self.first_volume_release],
            )
        with pytest.raises(ValueError, match="must be earlier"):
            CDRForcing(
                start_time=start_time,
                end_time=end_time,
                releases=[self.first_tracer_perturbation],
            )

    def test_empty_release_list(self):
        with pytest.raises(ValidationError):
            CDRForcing(start_time=self.start_time, end_time=self.end_time)

    def test_ds_attribute(self):
        assert isinstance(self.volume_release_cdr_forcing_without_grid.ds, xr.Dataset)
        assert isinstance(
            self.tracer_perturbation_cdr_forcing_without_grid.ds, xr.Dataset
        )
        assert isinstance(self.volume_release_cdr_forcing.ds, xr.Dataset)
        assert isinstance(self.tracer_perturbation_cdr_forcing.ds, xr.Dataset)
        assert isinstance(
            self.volume_release_cdr_forcing_with_straddling_grid.ds, xr.Dataset
        )
        assert isinstance(
            self.tracer_perturbation_cdr_forcing_with_straddling_grid.ds, xr.Dataset
        )

    def test_plot_error_when_no_grid(self):
        for cdr in [
            self.volume_release_cdr_forcing_without_grid,
            self.tracer_perturbation_cdr_forcing_without_grid,
        ]:
            with pytest.raises(
                ValueError, match="A grid must be provided for plotting"
            ):
                cdr.plot_locations("all")

            with pytest.raises(
                ValueError, match="A grid must be provided for plotting"
            ):
                cdr.plot_distribution("first_release")

    def test_plot_volume_release(self):
        for cdr in [
            self.volume_release_cdr_forcing_without_grid,
            self.volume_release_cdr_forcing,
            self.volume_release_cdr_forcing_with_straddling_grid,
        ]:
            cdr.plot_volume_flux()
            cdr.plot_volume_flux(release_names=["first_release"])

            cdr.plot_tracer_concentration("ALK")
            cdr.plot_tracer_concentration("ALK", release_names=["first_release"])

            cdr.plot_tracer_concentration("DIC")
            cdr.plot_tracer_concentration("DIC", release_names=["first_release"])

        self.volume_release_cdr_forcing.plot_locations()
        self.volume_release_cdr_forcing.plot_locations(release_names=["first_release"])

    def test_plot_tracer_perturbation(self):
        for cdr in [
            self.tracer_perturbation_cdr_forcing_without_grid,
            self.tracer_perturbation_cdr_forcing,
            self.tracer_perturbation_cdr_forcing_with_straddling_grid,
        ]:
            cdr.plot_tracer_flux("ALK")
            cdr.plot_tracer_flux("ALK", release_names=["first_release"])

            cdr.plot_tracer_flux("DIC")
            cdr.plot_tracer_flux("DIC", release_names=["first_release"])

        self.tracer_perturbation_cdr_forcing.plot_locations()
        self.tracer_perturbation_cdr_forcing.plot_locations(
            release_names=["first_release"]
        )

    def test_plot_max_releases(self, caplog):
        # Prepare releases with more than MAX_DISTINCT_COLORS unique names
        releases = []
        for i in range(MAX_DISTINCT_COLORS + 1):
            release = self.first_volume_release.__replace__(name=f"release_{i}")
            releases.append(release)

        # Construct a CDRForcing object with too many releases to plot
        cdr_forcing = CDRForcing(
            grid=self.grid,
            start_time=self.start_time,
            end_time=self.end_time,
            releases=releases,
        )

        release_names = [r.name for r in releases]

        plot_methods_with_release_names = [
            cdr_forcing.plot_locations,
            cdr_forcing.plot_volume_flux,
        ]

        for plot_func in plot_methods_with_release_names:
            caplog.clear()
            with caplog.at_level("WARNING"):
                plot_func(release_names=release_names)
            assert any(
                f"Only the first {MAX_DISTINCT_COLORS} releases will be plotted"
                in message
                for message in caplog.messages
            ), f"Warning not raised by {plot_func.__name__}"

        with caplog.at_level("WARNING"):
            cdr_forcing.plot_locations(release_names=release_names)

        assert any(
            f"Only the first {MAX_DISTINCT_COLORS} releases will be plotted" in message
            for message in caplog.messages
        )

    @pytest.mark.skipif(xesmf is None, reason="xesmf required")
    def test_plot_distribution(self):
        self.volume_release_cdr_forcing.plot_distribution("first_release")
        self.volume_release_cdr_forcing_with_straddling_grid.plot_distribution(
            "first_release"
        )
        self.tracer_perturbation_cdr_forcing.plot_distribution("first_release")
        self.tracer_perturbation_cdr_forcing_with_straddling_grid.plot_distribution(
            "first_release"
        )

    def test_plot_more_errors(self):
        """Test that error is raised on bad plot args or ambiguous release."""
        with pytest.raises(ValueError, match="Invalid release"):
            self.volume_release_cdr_forcing.plot_distribution(release_name="fake")

        with pytest.raises(ValueError, match="Invalid releases"):
            self.volume_release_cdr_forcing.plot_locations(release_names=["fake"])

        with pytest.raises(ValueError, match="should be a list"):
            self.volume_release_cdr_forcing.plot_locations(release_names=4)

        with pytest.raises(ValueError, match="must be strings"):
            self.volume_release_cdr_forcing.plot_locations(release_names=[4])

    def test_cdr_forcing_save(self, tmp_path):
        """Test save method."""
        for cdr_forcing in [
            self.volume_release_cdr_forcing,
            self.tracer_perturbation_cdr_forcing,
        ]:
            for file_str in ["test_cdr_forcing", "test_cdr_forcing.nc"]:
                # Create a temporary filepath using the tmp_path fixture
                for filepath in [tmp_path / file_str, str(tmp_path / file_str)]:
                    saved_filenames = cdr_forcing.save(filepath)
                    # Check if the .nc file was created
                    filepath = Path(filepath).with_suffix(".nc")
                    assert saved_filenames == [filepath]
                    assert filepath.exists()
                    # Clean up the .nc file
                    filepath.unlink()

    def test_roundtrip_yaml(self, tmp_path):
        """Test that creating a CDRVolumePointSource object, saving its parameters to
        yaml file, and re-opening yaml file creates the same object.
        """
        for cdr_forcing in [
            self.volume_release_cdr_forcing,
            self.tracer_perturbation_cdr_forcing,
        ]:
            # Create a temporary filepath using the tmp_path fixture
            file_str = "test_yaml"
            for filepath in [
                tmp_path / file_str,
                str(tmp_path / file_str),
            ]:  # test for Path object and str
                cdr_forcing.to_yaml(filepath)

                cdr_forcing_from_file = CDRForcing.from_yaml(filepath)

                assert cdr_forcing == cdr_forcing_from_file

                filepath = Path(filepath)
                filepath.unlink()

    def test_files_have_same_hash(self, tmp_path):
        """Test that saving the same CDR forcing configuration to NetCDF twice results
        in reproducible file hashes.
        """
        for cdr_forcing in [
            self.volume_release_cdr_forcing,
            self.tracer_perturbation_cdr_forcing,
        ]:
            yaml_filepath = tmp_path / "test_yaml.yaml"
            filepath1 = tmp_path / "test1.nc"
            filepath2 = tmp_path / "test2.nc"

            cdr_forcing.to_yaml(yaml_filepath)
            cdr_forcing.save(filepath1)
            cdr_from_file = CDRForcing.from_yaml(yaml_filepath)
            cdr_from_file.save(filepath2)

            hash1 = calculate_file_hash(filepath1)
            hash2 = calculate_file_hash(filepath2)

            assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

            yaml_filepath.unlink()
            filepath1.unlink()
            filepath2.unlink()

    @pytest.mark.parametrize(
        "cdr_forcing, tracer_attr",
        [
            ("volume_release_cdr_forcing_without_grid", "tracer_concentrations"),
            ("tracer_perturbation_cdr_forcing_without_grid", "tracer_fluxes"),
        ],
    )
    def test_compute_total_cdr_source(self, cdr_forcing, tracer_attr, request):
        dt = 30.0
        cdr_instance = getattr(self, cdr_forcing)

        df = cdr_instance.compute_total_cdr_source(dt)

        # Check type
        assert isinstance(df, pd.DataFrame)

        # Check rows = number of releases + 1 for the units row
        assert df.shape[0] == len(cdr_instance.releases) + 1

        # Columns = tracer names
        all_tracers = set()
        for r in cdr_instance.releases:
            all_tracers.update(getattr(r, tracer_attr).keys())

        # Remove temp and salt since they are excluded from integrated totals
        all_tracers.discard("temp")
        all_tracers.discard("salt")

        # Columns are now just tracer names (units row removed)
        col_tracers = set(df.columns)
        assert col_tracers == all_tracers

        # Check that units are included in the units row
        tracer_meta = get_tracer_metadata_dict(include_bgc=True, unit_type="integrated")
        for tracer in df.columns:
            unit = tracer_meta.get(tracer, {}).get("units", None)
            if unit:
                assert df.loc["units", tracer] == unit, (
                    f"Units row for '{tracer}' is incorrect"
                )

        # Exclude units row
        data_only = df.drop("units")

        # Convert all columns to numeric, coerce errors to NaN
        data_numeric = data_only.apply(pd.to_numeric, errors="coerce")

        # Now check all finite (ignoring any NaNs that were non-numeric)
        assert np.all(np.isfinite(data_numeric.values)), (
            "Some values are not finite numbers"
        )


class TestCdrLiteForcing:
    """End-to-end CDRForcing tests for tracer_set='cdr_lite'.

    Targeting is automatic: ALK-bearing releases get OAE pairs, DIC-only
    releases get DOR tracers, numbered in release order. The tracer schema is
    derived; include_marbl_bgc appends the MARBL BGC block (auto-enabled when
    marbl releases are mixed in).
    """

    def setup_method(self):
        self.start_time = datetime(2022, 1, 1)
        self.end_time = datetime(2022, 1, 31)

    def _perturbation(self, name, fluxes, tracer_set="cdr_lite", lon=-25.0):
        return TracerPerturbation(
            name=name,
            lat=66.0,
            lon=lon,
            depth=50.0,
            tracer_set=tracer_set,
            tracer_fluxes=fluxes,
        )

    def test_volume_release_rejected(self):
        with pytest.raises(ValidationError, match="not supported on VolumeRelease"):
            VolumeRelease(
                name="oae",
                lat=66.0,
                lon=-25.0,
                depth=50.0,
                tracer_set="cdr_lite",
                volume_fluxes=10.0,
            )

    def test_auto_assignment_order_and_routing(self):
        releases = [
            self._perturbation("oae_a", {"ALK": 1.0e6}),
            self._perturbation("dor_a", {"DIC": -5.0e5}, lon=-24.0),
            self._perturbation("combo", {"ALK": 2.0e6, "DIC": -1.0e5}, lon=-23.0),
        ]
        cdr = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=releases,
        )
        assert cdr.tracer_set == "cdr_lite"
        assert list(cdr.ds.tracer_name.values) == [
            "temp",
            "salt",
            "CDR_OAE_ALK1",
            "CDR_OAE_DIC1",
            "CDR_OAE_ALK2",
            "CDR_OAE_DIC2",
            "CDR_DOR_DIC1",
        ]
        assert cdr.release_tracers == {
            "oae_a": ("CDR_OAE_ALK1", "CDR_OAE_DIC1"),
            "dor_a": ("CDR_DOR_DIC1",),
            "combo": ("CDR_OAE_ALK2", "CDR_OAE_DIC2"),
        }
        # the release <-> tracer linkage is stored on the tracer axis
        assert list(cdr.ds.tracer_release.values) == [
            "",
            "",
            "oae_a",
            "oae_a",
            "combo",
            "combo",
            "dor_a",
        ]
        names = list(cdr.ds.tracer_name.values)
        flx = cdr.ds.cdr_trcflx

        def rows(tracer, ncdr):
            return flx.isel(ntracers=names.index(tracer), ncdr=ncdr).values

        # physics rows stay zero; each release feeds only its own tracers
        assert np.allclose(flx.isel(ntracers=[0, 1]).values, 0.0)
        assert np.allclose(rows("CDR_OAE_ALK1", 0), 1.0e6)
        assert np.allclose(rows("CDR_OAE_DIC1", 0), 0.0)
        assert np.allclose(rows("CDR_DOR_DIC1", 1), -5.0e5)
        assert np.allclose(rows("CDR_OAE_ALK2", 2), 2.0e6)
        assert np.allclose(rows("CDR_OAE_DIC2", 2), -1.0e5)
        assert np.allclose(rows("CDR_OAE_ALK2", 0), 0.0)

    def test_derived_tracer_schema(self):
        cdr = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=[self._perturbation("oae", {"ALK": 1.0e6})],
        )
        assert cdr.tracer_schema is not None
        assert cdr.tracer_schema.n_oae_pairs == 1
        assert cdr.tracer_schema.n_dor == 0
        assert cdr.tracer_schema.include_marbl_bgc is False
        assert cdr.ds.sizes["ntracers"] == 4

    def test_pure_marbl_has_no_schema(self):
        cdr = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=[self._perturbation("m", {"ALK": 1.0e6}, tracer_set="marbl")],
        )
        assert cdr.tracer_schema is None
        assert cdr.release_tracers == {}
        assert "tracer_release" not in cdr.ds.coords
        assert cdr.ds.sizes["ntracers"] == NUM_TRACERS

    def test_include_marbl_bgc_appends_bgc_block(self):
        from roms_tools.setup.utils import MARBL_TRACER_NAMES

        cdr = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=[self._perturbation("oae", {"ALK": 1.0e6})],
            include_marbl_bgc=True,
        )
        names = list(cdr.ds.tracer_name.values)
        expected_tail = [n for n in MARBL_TRACER_NAMES if n not in ("temp", "salt")]
        assert names == ["temp", "salt", "CDR_OAE_ALK1", "CDR_OAE_DIC1", *expected_tail]

    def test_include_marbl_bgc_rejected_for_pure_marbl(self):
        with pytest.raises(ValidationError, match="only valid when cdr_lite"):
            CDRForcing(
                start_time=self.start_time,
                end_time=self.end_time,
                releases=[self._perturbation("m", {"ALK": 1.0e6}, tracer_set="marbl")],
                include_marbl_bgc=True,
            )

    def test_mixed_marbl_and_cdr_lite(self, caplog):
        releases = [
            self._perturbation("oae", {"ALK": 1.0e6}),
            self._perturbation("marbl_alk", {"ALK": 3.0e6}, tracer_set="marbl"),
        ]
        with caplog.at_level(logging.INFO):
            cdr = CDRForcing(
                start_time=self.start_time,
                end_time=self.end_time,
                releases=releases,
            )
        assert "appending the MARBL BGC tracers" in caplog.text
        assert cdr.tracer_set == "mixed"
        assert cdr.tracer_schema.include_marbl_bgc is True
        names = list(cdr.ds.tracer_name.values)
        flx = cdr.ds.cdr_trcflx
        # cdr_lite release feeds its pair; marbl release feeds the MARBL ALK row
        assert np.allclose(
            flx.isel(ntracers=names.index("CDR_OAE_ALK1"), ncdr=0).values, 1.0e6
        )
        assert np.allclose(flx.isel(ntracers=names.index("ALK"), ncdr=1).values, 3.0e6)
        assert np.allclose(
            flx.isel(ntracers=names.index("CDR_OAE_ALK1"), ncdr=1).values, 0.0
        )
        assert np.allclose(flx.isel(ntracers=names.index("ALK"), ncdr=0).values, 0.0)
        tracer_release = list(cdr.ds.tracer_release.values)
        assert tracer_release[names.index("CDR_OAE_ALK1")] == "oae"
        assert tracer_release[names.index("ALK")] == ""

    def test_fifty_release_auto_assignment(self):
        releases = [
            self._perturbation(
                f"release_{k}", {"ALK": 1.0e6, "DIC": 1.0e3 * k}, lon=-25.0 + 0.1 * k
            )
            for k in range(1, 51)
        ]
        cdr = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=releases,
        )
        assert cdr.ds.sizes["ntracers"] == 102
        assert cdr.ds.sizes["ncdr"] == 50
        assert cdr.tracer_schema.n_oae_pairs == 50
        names = list(cdr.ds.tracer_name.values)
        flx = cdr.ds.cdr_trcflx
        k = 37
        assert np.allclose(
            flx.isel(ntracers=names.index(f"CDR_OAE_DIC{k}"), ncdr=k - 1).values,
            1.0e3 * k,
        )
        assert np.allclose(
            flx.isel(ntracers=names.index(f"CDR_OAE_DIC{k}"), ncdr=k).values, 0.0
        )
        # each ncdr column has exactly two nonzero tracer rows (its ALK + DIC)
        nonzero_rows = (np.abs(flx.values) > 0).any(axis=0).sum(axis=0)
        assert np.all(nonzero_rows == 2)

    def test_roundtrip_yaml(self, tmp_path):
        grid = Grid(
            nx=10,
            ny=10,
            size_x=500,
            size_y=500,
            center_lon=-25,
            center_lat=66,
            rot=0,
            N=3,
        )
        releases = [
            self._perturbation("oae", {"ALK": 1.0e6}),
            self._perturbation("dor", {"DIC": -5.0e5}, lon=-24.0),
        ]
        cdr = CDRForcing(
            grid=grid,
            start_time=self.start_time,
            end_time=self.end_time,
            releases=releases,
            include_marbl_bgc=True,
        )
        filepath = tmp_path / "cdr_lite.yaml"
        cdr.to_yaml(filepath)
        restored = CDRForcing.from_yaml(filepath)
        assert restored.tracer_set == "cdr_lite"
        assert restored.include_marbl_bgc is True
        assert restored.tracer_schema == cdr.tracer_schema
        assert restored.release_tracers == cdr.release_tracers
        assert restored.ds.identical(cdr.ds)

    def test_save(self, tmp_path):
        cdr = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=[self._perturbation("oae", {"ALK": 2.0e6})],
        )
        saved_paths = cdr.save(tmp_path / "cdr_lite_frc.nc")
        ds = xr.open_dataset(saved_paths[0])
        assert ds.sizes["ntracers"] == 4
        assert ds.sizes["ncdr"] == 1
        assert list(ds.tracer_release.values) == ["", "", "oae", "oae"]

    def test_roms_layout_cdr_lite(self, capsys):
        releases = [
            self._perturbation("oae_a", {"ALK": 1.0e6}),
            self._perturbation("dor_a", {"DIC": -5.0e5}, lon=-24.0),
        ]
        cdr = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=releases,
        )
        layout = cdr.roms_layout()
        out = capsys.readouterr().out
        assert list(layout.itrc.values) == [1, 2, 3, 4, 5]
        assert list(layout.tracer_name.values) == [
            "temp",
            "salt",
            "CDR_OAE_ALK1",
            "CDR_OAE_DIC1",
            "CDR_DOR_DIC1",
        ]
        assert list(layout.release.values) == ["", "", "oae_a", "oae_a", "dor_a"]
        assert layout.attrs["nt_cdr_oae"] == 1
        assert layout.attrs["nt_cdr_dor"] == 1
        assert layout.attrs["cdr_ncdr_parm"] == 2
        assert "itrc\ttracer_name\tunits\trelease" in out
        assert "nt_cdr_oae = 1" in out

    def test_roms_layout_pure_marbl(self, capsys):
        cdr = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=[self._perturbation("m", {"ALK": 1.0e6}, tracer_set="marbl")],
        )
        layout = cdr.roms_layout(print_table=False)
        assert capsys.readouterr().out == ""
        assert layout.sizes["itrc"] == NUM_TRACERS
        assert set(layout.release.values) == {""}
        assert layout.attrs["nt_cdr_oae"] == 0
        assert layout.attrs["cdr_ncdr_parm"] == 1


class TestPassiveForcing:
    """CDRForcing with tracer_set='passive' releases and three-model mixing."""

    def setup_method(self):
        self.start_time = datetime(2022, 1, 1)
        self.end_time = datetime(2022, 1, 31)

    def _dye(self, name, flux, lon=-25.0):
        return TracerPerturbation(
            name=name,
            lat=66.0,
            lon=lon,
            depth=50.0,
            tracer_set="passive",
            tracer_fluxes={"passive_tracer": flux},
        )

    def test_passive_only_forcing(self):
        cdr = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=[self._dye("dye1", 1.0e5), self._dye("dye2", 2.0e5, lon=-24.0)],
        )
        assert list(cdr.ds.tracer_name.values) == [
            "temp",
            "salt",
            "passive_tracer1",
            "passive_tracer2",
        ]
        assert cdr.tracer_schema.n_passive == 2
        assert cdr.release_tracers == {
            "dye1": ("passive_tracer1",),
            "dye2": ("passive_tracer2",),
        }
        names = list(cdr.ds.tracer_name.values)
        flx = cdr.ds.cdr_trcflx
        assert np.allclose(
            flx.isel(ntracers=names.index("passive_tracer1"), ncdr=0).values, 1.0e5
        )
        assert np.allclose(
            flx.isel(ntracers=names.index("passive_tracer1"), ncdr=1).values, 0.0
        )

    def test_three_model_perturbation_mix(self):
        releases = [
            TracerPerturbation(
                name="marbl_alk",
                lat=63.0,
                lon=-22.0,
                depth=50.0,
                tracer_fluxes={"ALK": 3.0e6},
            ),
            TracerPerturbation(
                name="oae1",
                lat=66.0,
                lon=-25.0,
                depth=50.0,
                tracer_set="cdr_lite",
                tracer_fluxes={"ALK": 1.0e6},
            ),
            self._dye("dye1", 1.0e5, lon=-24.0),
        ]
        cdr = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=releases,
        )
        names = list(cdr.ds.tracer_name.values)
        # generated blocks in ROMS order: passive, OAE pair, then MARBL BGC
        assert names[:5] == [
            "temp",
            "salt",
            "passive_tracer1",
            "CDR_OAE_ALK1",
            "CDR_OAE_DIC1",
        ]
        assert "ALK" in names  # MARBL block appended (auto include_marbl_bgc)
        tracer_release = list(cdr.ds.tracer_release.values)
        assert tracer_release[names.index("passive_tracer1")] == "dye1"
        assert tracer_release[names.index("CDR_OAE_ALK1")] == "oae1"
        assert tracer_release[names.index("ALK")] == ""
        flx = cdr.ds.cdr_trcflx
        assert np.allclose(
            flx.isel(ntracers=names.index("passive_tracer1"), ncdr=2).values, 1.0e5
        )
        assert np.allclose(flx.isel(ntracers=names.index("ALK"), ncdr=0).values, 3.0e6)

    def test_passive_volume_mixes_with_marbl_volume(self):
        releases = [
            VolumeRelease(
                name="mv",
                lat=66.0,
                lon=-25.0,
                depth=50.0,
                volume_fluxes=100.0,
                tracer_concentrations={"ALK": 2000.0},
            ),
            VolumeRelease(
                name="dye_v",
                lat=65.0,
                lon=-24.0,
                depth=50.0,
                tracer_set="passive",
                volume_fluxes=50.0,
                tracer_concentrations={"temp": 12.0, "passive_tracer": 10.0},
            ),
        ]
        cdr = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=releases,
        )
        names = list(cdr.ds.tracer_name.values)
        trc = cdr.ds.cdr_tracer
        assert np.allclose(
            trc.isel(ntracers=names.index("passive_tracer1"), ncdr=1).values, 10.0
        )
        assert np.allclose(trc.isel(ntracers=names.index("temp"), ncdr=1).values, 12.0)
        assert np.allclose(trc.isel(ntracers=names.index("ALK"), ncdr=0).values, 2000.0)

    def test_passive_volume_with_cdr_lite_perturbation_raises(self):
        # cdr_lite is perturbation-only, and mixed release types already raise.
        dye_volume = VolumeRelease(
            name="dye_v",
            lat=65.0,
            lon=-24.0,
            depth=50.0,
            tracer_set="passive",
            volume_fluxes=50.0,
            tracer_concentrations={"passive_tracer": 10.0},
        )
        oae = TracerPerturbation(
            name="oae",
            lat=66.0,
            lon=-25.0,
            depth=50.0,
            tracer_set="cdr_lite",
            tracer_fluxes={"ALK": 1.0e6},
        )
        with pytest.raises(ValidationError, match="same type"):
            CDRForcing(
                start_time=self.start_time,
                end_time=self.end_time,
                releases=[dye_volume, oae],
            )

    def test_roms_layout_includes_passive(self, capsys):
        cdr = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=[self._dye("dye1", 1.0e5)],
        )
        layout = cdr.roms_layout()
        out = capsys.readouterr().out
        assert list(layout.release.values) == ["", "", "dye1"]
        assert layout.attrs["nt_passive"] == 1
        assert "nt_passive = 1" in out

    def test_roundtrip_yaml_passive(self, tmp_path):
        grid = Grid(
            nx=10,
            ny=10,
            size_x=500,
            size_y=500,
            center_lon=-25,
            center_lat=66,
            rot=0,
            N=3,
        )
        cdr = CDRForcing(
            grid=grid,
            start_time=self.start_time,
            end_time=self.end_time,
            releases=[self._dye("dye1", 1.0e5)],
        )
        filepath = tmp_path / "passive.yaml"
        cdr.to_yaml(filepath)
        restored = CDRForcing.from_yaml(filepath)
        assert restored.releases[0].tracer_set == "passive"
        assert restored.ds.identical(cdr.ds)

    def test_passive_volume_marbl_rows_filled_not_diluting(self):
        """Regression: a passive VolumeRelease sharing a file with the MARBL
        block must not inject water with zero BGC concentrations (dilution).
        Its unfed MARBL rows follow fill_values, like marbl releases.
        """
        from roms_tools import BGCMarbl

        defaults = BGCMarbl.river_defaults()
        releases = [
            VolumeRelease(
                name="mv",
                lat=66.0,
                lon=-25.0,
                depth=50.0,
                volume_fluxes=100.0,
                tracer_concentrations={"ALK": 3000.0},
            ),
            VolumeRelease(
                name="dye_v",
                lat=65.0,
                lon=-24.0,
                depth=50.0,
                tracer_set="passive",
                volume_fluxes=50.0,
                tracer_concentrations={"passive_tracer": 10.0},
            ),
        ]
        cdr = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=releases,
        )
        names = list(cdr.ds.tracer_name.values)
        trc = cdr.ds.cdr_tracer
        for tracer in ("ALK", "DIC", "PO4"):
            assert np.allclose(
                trc.isel(ntracers=names.index(tracer), ncdr=1).values,
                defaults[tracer],
            )
        # the marbl release's own values are untouched, and anomaly/dye rows
        # of both releases stay zero except the dye's own slot
        assert np.allclose(trc.isel(ntracers=names.index("ALK"), ncdr=0).values, 3000.0)
        assert np.allclose(
            trc.isel(ntracers=names.index("passive_tracer1"), ncdr=0).values, 0.0
        )
        assert np.allclose(
            trc.isel(ntracers=names.index("passive_tracer1"), ncdr=1).values, 10.0
        )

    def test_passive_volume_fill_values_zero(self):
        releases = [
            VolumeRelease(
                name="mv",
                lat=66.0,
                lon=-25.0,
                depth=50.0,
                volume_fluxes=100.0,
                tracer_concentrations={"ALK": 3000.0},
            ),
            VolumeRelease(
                name="dye_v",
                lat=65.0,
                lon=-24.0,
                depth=50.0,
                tracer_set="passive",
                fill_values="zero",
                volume_fluxes=50.0,
                tracer_concentrations={"passive_tracer": 10.0},
            ),
        ]
        cdr = CDRForcing(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=releases,
        )
        names = list(cdr.ds.tracer_name.values)
        trc = cdr.ds.cdr_tracer
        # explicit user choice: added water carries zero BGC
        assert np.allclose(trc.isel(ntracers=names.index("ALK"), ncdr=1).values, 0.0)
        assert np.allclose(trc.isel(ntracers=names.index("DIC"), ncdr=1).values, 0.0)
