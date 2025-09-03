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
    def test_do_accounting(self, cdr_forcing, tracer_attr, request):
        dt = 30.0
        cdr_instance = getattr(self, cdr_forcing)

        df = cdr_instance.do_accounting(dt)

        # Check type
        assert isinstance(df, pd.DataFrame)

        # Check rows = number of releases
        assert df.shape[0] == len(cdr_instance.releases)

        # Columns = tracer names
        all_tracers = set()
        for r in cdr_instance.releases:
            all_tracers.update(getattr(r, tracer_attr).keys())
        assert set(df.columns) == all_tracers

        # Check index = release names
        expected_names = [r.name for r in cdr_instance.releases]
        assert list(df.index) == expected_names

        # All values finite
        assert np.all(np.isfinite(df.values))
