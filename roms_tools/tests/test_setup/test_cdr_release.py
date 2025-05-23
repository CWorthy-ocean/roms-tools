from unittest import mock

import pytest
from datetime import datetime
from pydantic import ValidationError

from roms_tools.setup.cdr_release import (
    Flux,
    Concentration,
    Release,
    VolumeRelease,
    TracerPerturbation,
)
from roms_tools.setup.utils import get_tracer_defaults


class TestValueArray:
    def test_check_length_passes_for_scalar(self):
        flux = Flux(name="flux", values=1.0)
        # Should not raise even if num_times doesn't match
        flux.check_length(num_times=5)

    def test_check_length_passes_for_matching_list(self):
        flux = Flux(name="flux", values=[1.0, 2.0, 3.0])
        flux.check_length(num_times=3)

    def test_check_length_raises_for_mismatched_list(self):
        conc = Concentration(name="NO3", values=[1.0, 2.0])
        with pytest.raises(ValueError, match="length of NO3"):
            conc.check_length(num_times=3)

        conc = Concentration(name="NO3", values=[1.0])
        with pytest.raises(ValueError, match="length of NO3"):
            conc.check_length(num_times=3)

    def test_flux_extend_scalar(self):

        # without times
        flux = Flux(name="flux", values=5.0)
        flux.extend_to_endpoints([], datetime(2020, 1, 1), datetime(2020, 1, 2))
        assert flux.values == [5.0, 5.0]

        # with times, no padding needed
        flux = Flux(name="flux", values=5.0)
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        flux.extend_to_endpoints(times, datetime(2020, 1, 1), datetime(2020, 1, 2))
        assert flux.values == [5.0, 5.0]

        # with times, padding needed
        flux = Flux(name="flux", values=5.0)
        times = [datetime(2020, 1, 2), datetime(2020, 1, 3)]
        flux.extend_to_endpoints(times, datetime(2020, 1, 1), datetime(2020, 1, 4))
        assert flux.values == [5.0, 5.0, 5.0, 5.0]

    def test_concentration_extend_scalar(self):

        # without times
        conc = Concentration(name="DIC", values=0.5)
        conc.extend_to_endpoints([], datetime(2020, 1, 1), datetime(2020, 1, 2))
        assert conc.values == [0.5, 0.5]

        # with times, no padding needed
        conc = Concentration(name="DIC", values=0.5)
        times = [datetime(2020, 1, 1)]
        conc.extend_to_endpoints(times, datetime(2020, 1, 1), datetime(2020, 1, 2))
        assert conc.values == [0.5, 0.5]

        # with times, padding needed
        conc = Concentration(name="DIC", values=0.5)
        times = [datetime(2020, 1, 2), datetime(2020, 1, 3)]
        conc.extend_to_endpoints(times, datetime(2020, 1, 1), datetime(2020, 1, 4))
        assert conc.values == [0.5, 0.5, 0.5, 0.5]

    def test_flux_extend_list_values(self):
        flux = Flux(name="flux", values=[3.0])
        times = [datetime(2020, 1, 2)]
        flux.extend_to_endpoints(
            times,
            start_time=datetime(2020, 1, 1),
            end_time=datetime(2020, 1, 3),
        )
        assert flux.values == [0.0, 3.0, 0.0]

        flux = Flux(name="flux", values=[3.0, 4.0])
        times = [datetime(2020, 1, 2), datetime(2020, 1, 3)]
        flux.extend_to_endpoints(
            times,
            start_time=datetime(2020, 1, 1),
            end_time=datetime(2020, 1, 4),
        )
        assert flux.values == [0.0, 3.0, 4.0, 0.0]

    def test_concentration_extend_list_values(self):
        conc = Concentration(name="DIC", values=[1.0])
        times = [datetime(2020, 1, 2)]
        conc.extend_to_endpoints(
            times,
            start_time=datetime(2020, 1, 1),
            end_time=datetime(2020, 1, 3),
        )
        assert conc.values == [1.0, 1.0, 1.0]

        conc = Concentration(name="DIC", values=[1.0, 2.0])
        times = [datetime(2020, 1, 2), datetime(2020, 1, 3)]
        conc.extend_to_endpoints(
            times,
            start_time=datetime(2020, 1, 1),
            end_time=datetime(2020, 1, 4),
        )
        assert conc.values == [1.0, 1.0, 2.0, 2.0]


class TestRelease:
    @pytest.fixture(scope="class", autouse=True)
    def mock_release_type(self):
        """This fixture fills in the release_type field for the base Release object, so
        we can test Release generally, without subclassing it."""
        original_fields = Release.model_fields
        modified = original_fields.copy()
        modified["release_type"].default = "testing_only"
        with mock.patch.object(Release, "model_fields", modified):
            Release.model_rebuild(force=True)
            yield

    def test_valid_release(self):
        times = [datetime(2022, 1, 1), datetime(2022, 1, 2)]
        r = Release(
            name="test", lat=0.0, lon=0.0, depth=100, hsc=10, vsc=5, times=times
        )

        assert r.name == "test"
        assert r.lat == 0.0
        assert r.lon == 0.0
        assert r.depth == 100.0
        assert r.hsc == 10.0
        assert r.vsc == 5.0
        assert r.times == times
        assert r.release_type == "testing_only"

    def test_lat_bounds_validation(self):
        with pytest.raises(ValidationError):
            Release(name="bad", lat=100.0, lon=0.0, depth=0, hsc=0, vsc=0, times=[])

    def test_depth_non_negative(self):
        with pytest.raises(ValidationError):
            Release(name="bad", lat=0.0, lon=0.0, depth=-1, hsc=0, vsc=0, times=[])

    def test_hsc_non_negative(self):
        with pytest.raises(ValidationError):
            Release(name="bad", lat=0.0, lon=0.0, depth=0, hsc=-1, vsc=0, times=[])

    def test_vsc_non_negative(self):
        with pytest.raises(ValidationError):
            Release(name="bad", lat=0.0, lon=0.0, depth=0, hsc=0, vsc=-1, times=[])

    def test_times_not_datetime_like(self):
        with pytest.raises(ValidationError):
            Release(
                name="bad",
                lat=0.0,
                lon=0.0,
                depth=0,
                hsc=0,
                vsc=0.0,
                times=["fake", "2023-02-01"],
            )

    def test_times_must_be_strictly_increasing(self):
        times = [datetime(2022, 1, 2), datetime(2022, 1, 1)]
        with pytest.raises(ValidationError):
            Release(name="bad", lat=0.0, lon=0.0, depth=0, hsc=0, vsc=0, times=times)

        times = [datetime(2022, 1, 2), datetime(2022, 1, 2)]
        with pytest.raises(ValidationError):
            Release(name="bad", lat=0.0, lon=0.0, depth=0, hsc=0, vsc=0, times=times)

    def test_extend_times_to_endpoints(self):
        r = Release(
            name="extend",
            lat=0.0,
            lon=0.0,
            depth=0,
            hsc=0,
            vsc=0,
            times=[datetime(2022, 1, 2)],
        )
        r._extend_times_to_endpoints(datetime(2022, 1, 1), datetime(2022, 1, 3))
        assert r.times == [
            datetime(2022, 1, 1),
            datetime(2022, 1, 2),
            datetime(2022, 1, 3),
        ]


class TestVolumeRelease:
    def setup_method(self):
        self.params = {"name": "vol", "lat": 0.0, "lon": 0.0, "depth": 0.0}

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            VolumeRelease(lat=0, lon=0, depth=0)  # name is missing

    def test_disallow_extra_fields(self):
        with pytest.raises(ValidationError):
            VolumeRelease(
                **self.params, volume_flux=1.0
            )  # correct field is volume_fluxes
        with pytest.raises(ValidationError):
            VolumeRelease(
                **self.params, tracer_conc={"ALK": 1.0}
            )  # correct field is tracer_concentration

    def test_volume_flux_not_valid(self):
        with pytest.raises(ValidationError):
            VolumeRelease(**self.params, volume_fluxes=["not_valid"])

    def test_volume_flux_non_negative(self):
        with pytest.raises(ValidationError):
            VolumeRelease(**self.params, volume_fluxes=-1)
        with pytest.raises(ValidationError):
            VolumeRelease(**self.params, volume_fluxes=[-1, 5])

    def test_tracer_concenctration_not_valid(self):
        with pytest.raises(ValidationError):
            VolumeRelease(
                **self.params, tracer_concentrations={"ALK": ["not", "valid"]}
            )

    def test_tracer_concentration_non_negative(self):
        with pytest.raises(ValidationError):
            VolumeRelease(**self.params, tracer_concentrations={"ALK": -1})
        with pytest.raises(ValidationError):
            VolumeRelease(**self.params, tracer_concentrations={"ALK": [-1, 15]})

    def test_auto_fill_strategy(self):

        alk_value = 100.0
        vr = VolumeRelease(**self.params, tracer_concentrations={"ALK": alk_value})
        defaults = get_tracer_defaults()
        for tracer in defaults:
            assert tracer in vr.tracer_concentrations
            assert isinstance(vr.tracer_concentrations[tracer], Concentration)
            if tracer == "ALK":
                assert vr.tracer_concentrations[tracer].values == alk_value
            else:
                assert vr.tracer_concentrations[tracer].values == defaults[tracer]

    def test_zero_fill_strategy(self):

        alk_value = 100.0
        vr = VolumeRelease(
            **self.params, tracer_concentrations={"ALK": alk_value}, fill_values="zero"
        )
        defaults = get_tracer_defaults()
        for tracer in defaults:
            assert tracer in vr.tracer_concentrations
            assert isinstance(vr.tracer_concentrations[tracer], Concentration)
            if tracer in ["temp", "salt"]:
                assert vr.tracer_concentrations[tracer].values == defaults[tracer]
            elif tracer == "ALK":
                assert vr.tracer_concentrations[tracer].values == alk_value
            else:
                assert vr.tracer_concentrations[tracer].values == 0.0

    def test_invalid_fill_strategy(self):

        with pytest.raises(ValidationError):
            VolumeRelease(**self.params, fill_values="zero_fill")

    def test_volume_flux_conversion(self):
        vr = VolumeRelease(**self.params, volume_fluxes=123)
        assert isinstance(vr.volume_fluxes, Flux)
        assert vr.volume_fluxes.values == 123

    def test_concentration_conversion(self):
        vr = VolumeRelease(**self.params, tracer_concentrations={"ALK": 2100})
        assert isinstance(vr.tracer_concentrations["ALK"], Concentration)
        assert vr.tracer_concentrations["ALK"].values == 2100

    def test_extend_to_endpoints(self):
        start = datetime(2022, 1, 1)
        mid = datetime(2022, 1, 2)
        end = datetime(2022, 1, 3)
        vr = VolumeRelease(
            **self.params,
            times=[mid],
            volume_fluxes=[1.0],
            tracer_concentrations={"ALK": [2000.0]},
        )
        vr._extend_to_endpoints(start, end)
        assert vr.times == [start, mid, end]
        assert vr.volume_fluxes.values == [0.0, 1.0, 0.0]
        assert vr.tracer_concentrations["ALK"].values == [2000.0, 2000.0, 2000.0]

    def test_mismatch_list_length(self):

        times = [datetime(2022, 1, 1), datetime(2022, 1, 2)]

        # Test mismatch between times and volume fluxes length
        volume_fluxes = [100]
        with pytest.raises(ValidationError):
            VolumeRelease(**self.params, times=times, volume_fluxes=volume_fluxes)

        # Test mismatch between times and tracer_concentrations length
        tracer_concentrations = {"ALK": [1]}
        with pytest.raises(ValidationError):
            VolumeRelease(
                **self.params, times=times, tracer_concentrations=tracer_concentrations
            )


class TestTracerPerturbation:
    def setup_method(self):
        self.params = {"name": "pert", "lat": 0.0, "lon": 0.0, "depth": 0.0}

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            TracerPerturbation(lat=0, lon=0, depth=0)  # name is missing

    def test_disallow_extra_fields(self):
        with pytest.raises(ValidationError):
            TracerPerturbation(
                **self.params, tracer_flux={"ALK": 1.0}
            )  # correct field is tracer_fluxes

    def test_tracer_fluxes_not_valid(self):
        with pytest.raises(ValidationError):
            TracerPerturbation(**self.params, tracer_fluxes={"ALK": ["not", "valid"]})

    def test_tracer_fluxes_non_negative(self):
        with pytest.raises(ValidationError):
            TracerPerturbation(**self.params, tracer_fluxes={"ALK": -1})
        with pytest.raises(ValidationError):
            TracerPerturbation(**self.params, tracer_fluxes={"ALK": [-1, 15]})

    def test_zero_fill_strategy(self):

        alk_value = 100.0
        tp = TracerPerturbation(
            **self.params,
            tracer_fluxes={"ALK": alk_value},
        )
        defaults = get_tracer_defaults()
        for tracer in defaults:
            assert tracer in tp.tracer_fluxes
            assert isinstance(tp.tracer_fluxes[tracer], Flux)
            if tracer == "ALK":
                assert tp.tracer_fluxes[tracer].values == alk_value
            else:
                assert tp.tracer_fluxes[tracer].values == 0.0

    def test_flux_conversion(self):
        tp = TracerPerturbation(**self.params, tracer_fluxes={"ALK": 2100})
        assert isinstance(tp.tracer_fluxes["ALK"], Flux)
        assert tp.tracer_fluxes["ALK"].values == 2100

    def test_extend_to_endpoints(self):
        start = datetime(2022, 1, 1)
        mid = datetime(2022, 1, 2)
        end = datetime(2022, 1, 3)
        tp = TracerPerturbation(
            **self.params,
            times=[mid],
            tracer_fluxes={"ALK": [2000.0]},
        )
        tp._extend_to_endpoints(start, end)
        assert tp.times == [start, mid, end]
        assert tp.tracer_fluxes["ALK"].values == [0.0, 2000.0, 0.0]

    def test_mismatch_list_length(self):

        times = [datetime(2022, 1, 1), datetime(2022, 1, 2)]

        # Test mismatch between times and tracer_fluxes length
        tracer_fluxes = {"ALK": [1]}
        with pytest.raises(ValidationError):
            TracerPerturbation(**self.params, times=times, tracer_fluxes=tracer_fluxes)
