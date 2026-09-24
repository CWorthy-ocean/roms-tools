import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, auto
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from annotated_types import Ge, Le
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core.core_schema import ValidationInfo
from scipy.interpolate import interp1d

from roms_tools.setup.bgc_model import (
    RELEASE_TRACER_MODELS,
    BGCCdrLite,
    BGCMarbl,
    BGCPassive,
    TracerSet,
)
from roms_tools.setup.utils import convert_to_relative_days

NonNegativeFloat = Annotated[float, Ge(0)]

# Show all columns when printing a DataFrame
pd.set_option("display.max_columns", None)


#: Flux keys a tracer_set="cdr_lite" release may specify. Physics tracers
#: (temp, salt) are deliberately excluded: CDR tracer experiments require the
#: physics to remain untouched by the release, so their rows in the forcing
#: file are always zero.
_CDR_LITE_KEYS = (BGCCdrLite.ROLE_ALK, BGCCdrLite.ROLE_DIC)

#: Concentration/flux key a tracer_set="passive" release specifies (volume
#: releases may additionally give temp/salt for the discharged water).
_PASSIVE_KEYS = (BGCPassive.ROLE_TRACER,)


def _raise_on_unknown_tracers(provided, allowed, tracer_set: TracerSet) -> None:
    """Reject tracer keys that are not part of the release's tracer set."""
    unknown = sorted(set(provided) - set(allowed))
    if unknown:
        raise ValueError(
            f"Unknown tracer name(s) {unknown} for tracer_set='{tracer_set}'. "
            f"Valid names: {sorted(allowed)}."
        )


@dataclass
class ValueArray(ABC):
    """Abstract base class representing a time series of values, either scalar or list.

    Attributes
    ----------
    name : str
        Name of the variable (e.g., flux or concentration).
    values : Union[float, List[float]]
        A constant value or a time-varying series of values.
    """

    name: str
    values: float | list[float]

    def check_length(self, num_times: int) -> None:
        """Checks that the number of values matches the number of time steps.

        Parameters
        ----------
        num_times : int
            Expected number of time steps.

        Raises
        ------
        ValueError
            If `values` is a list and its length does not match `num_times`.
        """
        if isinstance(self.values, list):
            if len(self.values) != num_times:
                raise ValueError(
                    f"The length of {self.name} ({len(self.values)}) does not match the number of times ({num_times})."
                )

    def _extend_scalar_series(
        self,
        times: list,
        start_time,
        end_time,
        start_pad: float,
        end_pad: float,
    ):
        """Extend self.values to align with times, including optional padding at
        start_time and end_time.

        Parameters
        ----------
        times : list
            List of datetime-like objects.
        start_time : datetime-like
            Start of the desired interval.
        end_time : datetime-like
            End of the desired interval.
        start_pad : float
            Value to prepend if `times[0] > start_time` (only if `self.values` is a list).
        end_pad : float
            Value to append if `times[-1] < end_time` (only if `self.values` is a list).

        Returns
        -------
        self : ValueArray
            The updated instance with extended `values`.
        """
        if isinstance(self.values, list):
            if times and times[0] > start_time:
                self.values.insert(0, start_pad)
            if times and times[-1] < end_time:
                self.values.append(end_pad)
        else:
            count = len(times)
            prepend = not times or times[0] > start_time
            append = not times or times[-1] < end_time
            count += int(prepend) + int(append)

            self.values = [self.values] * count

        return self

    @abstractmethod
    def extend_to_endpoints(self, times: list, start_time, end_time):
        """Abstract method to extend the value series to cover given time endpoints.

        Must be implemented in subclasses.
        """
        raise NotImplementedError()


@dataclass
class Flux(ValueArray):
    """Represents a time series of non-negative flux values.

    Attributes
    ----------
    name : str
        Name of the flux variable.
    values : Union[NonNegativeFloat, List[NonNegativeFloat]]
        A constant non-negative flux or a list of non-negative flux values.
    """

    values: NonNegativeFloat | list[NonNegativeFloat]

    def extend_to_endpoints(self, times: list, start_time, end_time):
        """Extends the flux series to ensure it covers the full time interval.

        - Pads with `0.0` before the first time if `start_time` is earlier.
        - Pads with `0.0` after the last time if `end_time` is later.
        - If no times are provided, assumes a constant flux throughout.

        Parameters
        ----------
        times : list
            List of datetime-like time points.
        start_time : datetime-like
        end_time : datetime-like
            End of the interval to cover.

        Returns
        -------
        self : Flux
            Updated instance with extended values.
        """
        return self._extend_scalar_series(
            times, start_time, end_time, start_pad=0.0, end_pad=0.0
        )


@dataclass
class Concentration(ValueArray):
    """Represents a time series of tracer concentrations.

    Attributes
    ----------
    name : str
        Name of the tracer (e.g., 'NO3', 'DIC').
    values : Union[float, List[float]]
        A constant concentration or a list of concentrations over time.
    """

    def extend_to_endpoints(self, times: list, start_time, end_time):
        """Extends the concentration series to ensure it covers the full time interval.

        - Pads with the first value before the first time if `start_time` is earlier.
        - Pads with the last value after the last time if `end_time` is later.
        - If no times are provided, assumes a constant concentration throughout.

        Parameters
        ----------
        times : list
            List of datetime-like time points.
        start_time : datetime-like
            Start of the interval to cover.
        end_time : datetime-like
            End of the interval to cover.

        Returns
        -------
        self : Concentration
            Updated instance with extended values.
        """
        if isinstance(self.values, list):
            start_pad = self.values[0]
            end_pad = self.values[-1]
        else:
            start_pad = self.values
            end_pad = self.values

        return self._extend_scalar_series(
            times, start_time, end_time, start_pad=start_pad, end_pad=end_pad
        )


class ReleaseType(StrEnum):
    volume = auto()
    tracer_perturbation = auto()


class Release(BaseModel):
    """Defines the basic properties and timing of a carbon dioxide removal (CDR)
    release.

    Attributes
    ----------
    name : str
        Unique identifier for the release.
    lat : float
        Latitude of the release location in degrees North. Must be between -90 and 90.
    lon : float
        Longitude of the release location in degrees East.
    depth : float
        Depth of the release in meters. Must be non-negative.
    hsc : float
        Horizontal scale (standard deviation) of the release in meters. Must be non-negative.
    vsc : float
        Vertical scale (standard deviation) of the release in meters. Must be non-negative.
    times : list of datetime
        Time points of the release events. Must be strictly increasing and within the simulation window.
    time_interpolation : bool, optional
        Whether to interpolate between tracer flux quantities. True to interpolate, False for step-like release. Defaults to False.
    tracer_set : {"marbl", "cdr_lite", "passive"}, optional
        Tracer schema. ``"marbl"`` (default) specifies tracer values by MARBL
        tracer name. ``"cdr_lite"`` targets the dedicated CDR tracers of a
        ROMS ``CDR_TRACER`` build: the release specifies ``"ALK"`` and/or
        ``"DIC"`` fluxes, and ``CDRForcing`` assigns tracers automatically —
        a release with ``"ALK"`` (an OAE or combined OAE+DOR intervention)
        gets the next ``CDR_OAE_ALK{k}``/``CDR_OAE_DIC{k}`` pair; a release
        with only ``"DIC"`` (a DOR intervention) gets the next
        ``CDR_DOR_DIC{j}`` tracer. cdr_lite is only supported on
        ``TracerPerturbation`` — CDR tracer experiments require the physics
        to remain untouched, so volume releases and ``temp`` / ``salt``
        forcing are not allowed. ``"passive"`` targets a generic passive
        (dye) tracer via a single ``"passive_tracer"`` flux (or, on
        ``VolumeRelease``, concentration together with the discharged
        water's ``temp``/``salt``), assigned the next ``passive_tracer{i}``
        slot.
    """

    name: str
    """Unique identifier for the release."""
    lat: Annotated[float, Ge(-90), Le(90)]
    """Latitude of the release location in degrees North."""
    lon: float
    """Longitude of the release location in degrees East."""
    depth: NonNegativeFloat
    """Depth of the release in meters."""
    hsc: NonNegativeFloat = 0.0
    """Horizontal scale (standard deviation) of the release in meters."""
    vsc: NonNegativeFloat = 0.0
    """Vertical scale (standard deviation) of the release in meters."""
    times: list[datetime]
    """Time points of the release events."""
    time_interpolation: bool = False
    """Whether to interpolate between prescribed tracer flux quantities. True interpolate, False step-like release."""
    tracer_set: TracerSet = "marbl"
    """Tracer schema: ``"marbl"`` (values keyed by MARBL tracer name),
    ``"cdr_lite"`` (values keyed by ``"ALK"``/``"DIC"``; ``CDRForcing``
    auto-assigns the release's own CDR tracer(s)), or ``"passive"`` (a single
    ``"passive_tracer"`` value; auto-assigned passive slot). ``"cdr_lite"``
    is only supported on :class:`TracerPerturbation`: CDR tracer experiments
    require the physics to remain untouched, which rules out volume releases
    (and temp/salt forcing)."""

    # this should be defined by subclasses
    release_type: ReleaseType
    """Type of the release."""

    model_config = ConfigDict(extra="forbid")

    @property
    def is_oae(self) -> bool:
        """Whether this cdr_lite release is an OAE (or combined OAE+DOR)
        intervention, i.e. specifies an ``"ALK"`` flux — it is assigned an
        OAE (ALK, DIC) tracer pair. False for marbl releases.
        """
        if self.tracer_set != "cdr_lite":
            return False
        return BGCCdrLite.ROLE_ALK in getattr(self, "tracer_fluxes", {})

    @property
    def is_dor(self) -> bool:
        """Whether this cdr_lite release is a DOR intervention, i.e. specifies
        only a ``"DIC"`` flux — it is assigned a standalone DOR tracer.
        False for marbl releases.
        """
        return self.tracer_set == "cdr_lite" and not self.is_oae

    @property
    def is_passive(self) -> bool:
        """Whether this release feeds a generic passive (dye) tracer — it is
        assigned the next ``passive_tracer{i}`` slot.
        """
        return self.tracer_set == "passive"

    @model_validator(mode="after")
    def _check_increasing_times(self) -> "Release":
        """Validates that `times` are strictly increasing and fall within the specified
        time window.

        Raises
        ------
        ValueError
            If times are not strictly increasing, or fall outside the [start_time, end_time] window.
        """
        if self.times and len(self.times) > 0:
            if not all(t1 < t2 for t1, t2 in zip(self.times, self.times[1:])):
                raise ValueError(
                    f"'times' must be strictly monotonically increasing. Got: {self.times}"
                )
        return self

    def _extend_times_to_endpoints(self, start_time, end_time) -> None:
        """Ensures that `times` includes both `start_time` and `end_time`.

        Modifies `self.times` in place by prepending or appending times as needed.
        If `times` is empty, it will be set to [`start_time`, `end_time`].
        """
        if not self.times:
            self.times = [start_time, end_time]
        else:
            self.times = list(self.times)  # Make mutable
            if self.times[0] > start_time:
                self.times.insert(0, start_time)
            if self.times[-1] < end_time:
                self.times.append(end_time)

    @classmethod
    def get_tracer_metadata(cls, tracer_set: TracerSet = "marbl"):
        return {}

    @classmethod
    def get_metadata(cls, tracer_set: TracerSet = "marbl"):
        return pd.DataFrame(cls.get_tracer_metadata(tracer_set))

    @property
    def metadata(self) -> pd.DataFrame:
        """Long names and expected units for this release's tracer inputs."""
        return pd.DataFrame(self.get_tracer_metadata(self.tracer_set))

    def _compute_integrated_tracers(
        self,
        roms_time_stamps: np.ndarray,
        model_reference_date: datetime,
        tracer_series_dict: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """
        Compute time-integrated tracer quantities over ROMS time steps using a left-hold rule.

        This method performs a left-hold (stepwise constant) integration of tracer fluxes
        over the intervals defined by the ROMS time stamps. It first interpolates the
        tracer time series from the release schedule onto the ROMS time stamps, then
        multiplies the value at the start of each interval by the duration of that interval.

        Parameters
        ----------
        roms_time_stamps : np.ndarray
            1D array of ROMS model time stamps in seconds since `model_reference_date`.
            Must be strictly increasing and contain at least two entries.
        model_reference_date : datetime
            Reference datetime of the ROMS model calendar, used to compute relative times
            for interpolation.
        tracer_series_dict : dict[str, np.ndarray]
            Dictionary mapping tracer names to 1D arrays of tracer flux values at the
            release schedule times (`self.times`). Each array must have the same length
            as `self.times`.

        Returns
        -------
        dict[str, float]
            Dictionary mapping each tracer name to its integrated quantity over the
            ROMS time period. Integration is performed using the left-hold rule,
            ignoring the last release point because it defines the end of the final interval.

        Raises
        ------
        ValueError
            If `roms_time_stamps` has fewer than two entries, since at least one interval
            is required for integration.
        """
        if len(roms_time_stamps) < 2:
            raise ValueError("Need at least two ROMS time stamps to define intervals.")

        dt = np.diff(roms_time_stamps)
        results = {}
        for tracer, series in tracer_series_dict.items():
            if self.time_interpolation:
                interp_values = np.interp(
                    roms_time_stamps,
                    convert_to_relative_days(self.times, model_reference_date)
                    * 3600
                    * 24,
                    series,
                )
            else:
                step_func = interp1d(
                    convert_to_relative_days(self.times, model_reference_date)
                    * 3600
                    * 24,
                    series,
                    kind="previous",
                )
                interp_values = step_func(roms_time_stamps)
            results[tracer] = np.sum(interp_values[:-1] * dt)
        return results


class VolumeRelease(Release):
    """Represents a CDR release with volume flux and tracer concentrations.

    Parameters
    ----------
    name : str
        Unique identifier for the release.
    lat : float
        Latitude of the release location in degrees North. Must be between -90 and 90.
    lon : float
        Longitude of the release location in degrees East.
    depth : float
        Depth of the release in meters. Must be non-negative.
    hsc : float
        Horizontal scale (standard deviation) of the release in meters. Must be non-negative. Defaults to 0.0.
    vsc : float
        Vertical scale (standard deviation) of the release in meters. Must be non-negative. Defaults to 0.0.
    times : list of datetime.datetime, optional
        Explicit time points for volume fluxes and tracer concentrations. Defaults to [self.start_time, self.end_time] if None.

        Example: `times=[datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)]`

    volume_fluxes : float or list of float, optional

        Volume flux(es) of the release in m³/s over time.

        - Constant: applies uniformly across the entire simulation period.
        - Time-varying: must match the length of `times`.

        Example:

        - Constant: `volume_fluxes=1000.0` (uniform across the entire simulation period).
        - Time-varying: `volume_fluxes=[1000.0, 1500.0, 2000.0]` (corresponds to each `times` entry).

    tracer_concentrations : dict, optional

        Dictionary of tracer names and their concentration values. The concentration values can be either
        a float (constant in time) or a list of float (time-varying).

        - Constant: applies uniformly across the entire simulation period.
        - Time-varying: must match the length of `times`.

        Default is an empty dictionary (`{}`) if not provided.

        Example:

        - Constant: `{"ALK": 2000.0, "DIC": 1900.0}`
        - Time-varying: `{"ALK": [2000.0, 2050.0, 2013.3], "DIC": [1900.0, 1920.0, 1910.2]}`
        - Mixed: `{"ALK": 2000.0, "DIC": [1900.0, 1920.0, 1910.2]}`

    fill_values : str, optional

        Strategy for filling missing tracer concentration values. Options:

        - "auto" (default): automatically set values to non-zero defaults
        - "zero": fill missing values with 0.0

    time_interpolation : bool, optional
        Whether to interpolate between tracer flux quantities. True to interpolate, False for step-like release. Defaults to False.
    """

    times: list[datetime] = Field([])
    fill_values: Literal["auto", "zero"] = "auto"
    """Strategy for filling missing tracer concentration values. For
    ``tracer_set="passive"`` releases this governs the MARBL BGC rows of the
    forcing file when the MARBL block is on the tracer axis (mixed with marbl
    releases or ``include_marbl_bgc=True``): "auto" fills them with river
    defaults, "zero" with 0 — a volume release adds water, so a zero
    concentration means the added water contains none of that tracer."""
    volume_fluxes: Flux | NonNegativeFloat | list[NonNegativeFloat] = Field(
        default=0.0, validate_default=True
    )
    """Volume flux(es) of the release in m³/s over time."""
    tracer_concentrations: dict[str, Concentration | float | list[float]] = Field({})
    """Dictionary of tracer names and their non-negative concentration values."""

    release_type: Literal[ReleaseType.volume] = ReleaseType.volume

    @field_validator("tracer_set", mode="after")
    @classmethod
    def _reject_cdr_lite_set(cls, tracer_set: TracerSet) -> TracerSet:
        """Volume releases perturb the physics (volume flux enters the continuity
        equation, and ROMS applies volume*concentration to temp/salt), which CDR
        tracer experiments must avoid — so only TracerPerturbation supports
        tracer_set="cdr_lite".
        """
        if tracer_set == "cdr_lite":
            raise ValueError(
                'tracer_set="cdr_lite" is not supported on VolumeRelease: '
                "CDR tracer experiments require the physics to remain untouched "
                "by the release. Use TracerPerturbation instead."
            )
        return tracer_set

    @field_validator("tracer_concentrations", mode="after")
    @classmethod
    def _create_concentrations(cls, tracer_concentrations, info: ValidationInfo):
        tracer_set: TracerSet = info.data.get("tracer_set", "marbl")
        defaults = BGCMarbl.river_defaults()

        if tracer_set == "passive":
            # Discharged water: physics tracers plus the dye. The dye is not
            # zero-filled — it must be provided (checked in a model validator).
            allowed = ("temp", "salt", *_PASSIVE_KEYS)
            _raise_on_unknown_tracers(tracer_concentrations, allowed, tracer_set)
            filled = {
                key: tracer_concentrations.get(key, defaults[key])
                for key in ("temp", "salt")
            }
            filled.update(
                {k: v for k, v in tracer_concentrations.items() if k in _PASSIVE_KEYS}
            )
            return {
                tracer: (
                    conc
                    if isinstance(conc, Concentration)
                    else Concentration(name=tracer, values=conc)
                )
                for tracer, conc in filled.items()
            }

        _raise_on_unknown_tracers(tracer_concentrations, defaults, "marbl")
        filled = {}
        for tracer_name in defaults:
            if tracer_name in tracer_concentrations:
                filled[tracer_name] = tracer_concentrations[tracer_name]
            elif tracer_name in ["temp", "salt"]:
                # Physics tracers always get river/physics defaults.
                filled[tracer_name] = defaults[tracer_name]
            else:
                fill_values = info.data["fill_values"]
                if fill_values == "auto":
                    filled[tracer_name] = defaults[tracer_name]
                elif fill_values == "zero":
                    filled[tracer_name] = 0.0

        return {
            tracer: (
                conc
                if isinstance(conc, Concentration)
                else Concentration(name=tracer, values=conc)
            )
            for tracer, conc in filled.items()
        }

    @field_validator("volume_fluxes", mode="after")
    @classmethod
    def _create_fluxes(cls, volume_fluxes) -> Flux:
        if not isinstance(volume_fluxes, Flux):
            volume_fluxes = Flux("volume", volume_fluxes)
        return volume_fluxes

    @model_validator(mode="after")
    def _check_passive_concentration_present(self) -> "VolumeRelease":
        """A passive volume release must specify the dye concentration."""
        if self.tracer_set == "passive" and not any(
            key in self.tracer_concentrations for key in _PASSIVE_KEYS
        ):
            raise ValueError(
                'Releases with tracer_set="passive" must specify a '
                "'passive_tracer' concentration."
            )
        return self

    @model_validator(mode="after")
    def _check_concentration_signs(self) -> "VolumeRelease":
        """Enforce non-negative tracer concentrations.

        Replaces the former ``NonNegativeFloat`` field typing: the rule is the
        same, but because values are heterogeneous (``Concentration`` | float
        | list) the check runs at model validation rather than field coercion,
        so the error surfaces at a different stage.
        """
        for tracer_name, conc in self.tracer_concentrations.items():
            values = conc.values if isinstance(conc, Concentration) else conc
            vals = values if isinstance(values, list) else [values]
            if any(v < 0 for v in vals):
                raise ValueError(
                    f"Tracer concentration for '{tracer_name}' must be non-negative. "
                    f"Got: {values}"
                )
        return self

    @model_validator(mode="after")
    def _check_lengths(self) -> "VolumeRelease":
        num_times = len(self.times)

        for tracer_concentrations in self.tracer_concentrations.values():
            if isinstance(tracer_concentrations, Concentration):
                tracer_concentrations.check_length(num_times)

        if isinstance(self.volume_fluxes, Flux):
            self.volume_fluxes.check_length(num_times)

        return self

    def _extend_to_endpoints(self, start_time, end_time):
        """Ensures that time series data includes endpoints at `start_time` and
        `end_time`.

        Pads `volume_fluxes` and each tracer concentration if needed to match the full time window.
        Also ensures `self.times` includes the endpoints.
        """
        self.volume_fluxes.extend_to_endpoints(self.times, start_time, end_time)
        for conc in self.tracer_concentrations.values():
            conc.extend_to_endpoints(self.times, start_time, end_time)
        self._extend_times_to_endpoints(start_time, end_time)

    @staticmethod
    def get_tracer_metadata(tracer_set: TracerSet = "marbl"):
        """Returns long names and expected units for the tracer concentrations."""
        if tracer_set == "cdr_lite":
            raise ValueError(
                'tracer_set="cdr_lite" is not supported on VolumeRelease. '
                "Use TracerPerturbation instead."
            )
        metadata = RELEASE_TRACER_MODELS[tracer_set].release_metadata(
            unit_type="concentration"
        )
        if tracer_set == "passive":
            # Volume releases also carry the physics of the discharged water.
            physics = RELEASE_TRACER_MODELS["marbl"].release_metadata(
                unit_type="concentration"
            )
            metadata = {
                "temp": physics["temp"],
                "salt": physics["salt"],
                **metadata,
            }
        return metadata

    def _do_accounting(
        self,
        roms_time_stamps: np.ndarray,
        model_reference_date: datetime,
    ) -> dict[str, float]:
        """
        Compute time-integrated tracer quantities over ROMS time steps.

        This method interpolates tracer flux time series from the CDR schedule
        onto the provided ROMS time stamps (in seconds since model reference date),
        then applies a "left-hold" rule: the interpolated value at t₀ is applied
        across the full interval [t₀, t₁).

        Parameters
        ----------
        roms_time_stamps : np.ndarray
            1D array of ROMS time stamps (seconds since `model_reference_date`).
            Must be strictly increasing.
        model_reference_date : datetime
            Reference date of the ROMS model calendar.

        Returns
        -------
        dict[str, float]
            Dictionary mapping tracer names to the total integrated quantity over
            the entire ROMS time period. Each value is the sum of the interpolated
            tracer fluxes multiplied by the corresponding ROMS time step durations.
        """
        tracer_series_dict = {}
        volume_array = (
            np.asarray(self.volume_fluxes.values)
            if isinstance(self.volume_fluxes, Flux)
            else np.asarray(self.volume_fluxes)
        )
        for tracer, conc in self.tracer_concentrations.items():
            tracer_array = (
                np.asarray(conc.values)
                if isinstance(conc, Concentration)
                else np.asarray(conc)
            )
            tracer_series_dict[tracer] = volume_array * tracer_array
        return self._compute_integrated_tracers(
            roms_time_stamps, model_reference_date, tracer_series_dict
        )

    @model_serializer(mode="wrap")
    def _simplified_dump(self, pydantic_serializer) -> dict:
        """Return a simplified dict representation with flattened values."""
        # pydantic_serializer is a function that runs pydantic's default conversion
        # of a model to a dict. then, we make some custom modifications after that
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pydantic_serializer(self)

        # Flatten volume_fluxes
        if "volume_fluxes" in data and isinstance(data["volume_fluxes"], dict):
            data["volume_fluxes"] = data["volume_fluxes"]["values"]

        # Flatten tracer_concentrations
        if "tracer_concentrations" in data:
            simplified = {}
            for tracer, contents in data["tracer_concentrations"].items():
                simplified[tracer] = contents["values"]
            data["tracer_concentrations"] = simplified

        return data


class TracerPerturbation(Release):
    """Represents a CDR release with tracer fluxes and without any volume.

    Parameters
    ----------
    name : str
        Unique identifier for the release.
    lat : float
        Latitude of the release location in degrees North. Must be between -90 and 90.
    lon : float
        Longitude of the release location in degrees East.
    depth : float
        Depth of the release in meters. Must be non-negative.
    hsc : float
        Horizontal scale (standard deviation) of the release in meters. Must be non-negative. Defaults to 0.0.
    vsc : float
        Vertical scale (standard deviation) of the release in meters. Must be non-negative. Defaults to 0.0.
    times : list of datetime.datetime, optional
        Explicit time points for volume fluxes and tracer concentrations. Defaults to [self.start_time, self.end_time] if None.

        Example: `times=[datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)]`

    tracer_fluxes : dict, optional

        Dictionary of tracer names and their flux values. The flux values can be either
        a float (constant in time) or a list of float (time-varying).

        - Constant: applies uniformly across the entire simulation period.
        - Time-varying: must match the length of `times`.

        Default is an empty dictionary (`{}`) if not provided.

        Example:

        - Constant: `{"ALK": 2000.0, "DIC": 1900.0}`
        - Time-varying: `{"ALK": [2000.0, 2050.0, 2013.3], "DIC": [1900.0, 1920.0, 1910.2]}`
        - Mixed: `{"ALK": 2000.0, "DIC": [1900.0, 1920.0, 1910.2]}`

        With ``tracer_set="marbl"`` keys are MARBL tracer names. With
        ``tracer_set="cdr_lite"`` the only keys are ``"ALK"`` and ``"DIC"``:
        a release providing ``"ALK"`` is an OAE (or, with negative ``"DIC"``,
        combined OAE+DOR) intervention assigned an OAE tracer pair; a release
        providing only ``"DIC"`` (negative = removal) is a DOR intervention
        assigned a standalone DOR tracer. With ``tracer_set="passive"`` the
        only key is ``"passive_tracer"``. ``temp`` / ``salt`` may not be
        forced by cdr_lite or passive perturbations.

    time_interpolation : bool, optional
        Whether to interpolate between tracer flux quantities. True to interpolate, False for step-like release. Defaults to False.
    """

    times: list[datetime] = Field([])
    tracer_fluxes: dict[str, Flux | float | list[float]] = Field({})
    """Dictionary of tracer names (or, for ``tracer_set="cdr_lite"``, role
    keys) and their flux values."""

    release_type: Literal[ReleaseType.tracer_perturbation] = (
        ReleaseType.tracer_perturbation
    )

    @field_validator("tracer_fluxes", mode="after")
    @classmethod
    def _create_fluxes(cls, tracer_fluxes, info: ValidationInfo):
        tracer_set: TracerSet = info.data.get("tracer_set", "marbl")

        if tracer_set == "cdr_lite":
            _raise_on_unknown_tracers(tracer_fluxes, _CDR_LITE_KEYS, tracer_set)
            # No zero-filling: the provided keys are the intervention-type
            # signal (ALK present -> OAE pair; DIC only -> DOR tracer).
            filled: dict[str, Flux | float | list[float]] = dict(tracer_fluxes)
        elif tracer_set == "passive":
            _raise_on_unknown_tracers(tracer_fluxes, _PASSIVE_KEYS, tracer_set)
            filled = dict(tracer_fluxes)
        else:
            allowed = list(BGCMarbl.river_defaults())
            _raise_on_unknown_tracers(tracer_fluxes, allowed, tracer_set)
            # Fill all tracer fluxes that are not provided with zero
            filled = {
                tracer_name: tracer_fluxes.get(tracer_name, 0.0)
                for tracer_name in allowed
            }

        return {
            tracer: (flux if isinstance(flux, Flux) else Flux(name=tracer, values=flux))
            for tracer, flux in filled.items()
        }

    @model_validator(mode="after")
    def _check_fluxes_present(self):
        """cdr_lite and passive releases must specify their flux key(s): for
        cdr_lite the provided keys determine whether it is an OAE-pair or DOR
        intervention; for passive the dye flux is the release's entire content.
        """
        if self.tracer_set == "cdr_lite" and not self.tracer_fluxes:
            raise ValueError(
                'Releases with tracer_set="cdr_lite" must specify tracer_fluxes '
                "for 'ALK' (OAE; optionally with 'DIC', which may be negative "
                "for combined OAE+DOR) and/or 'DIC' alone (DOR)."
            )
        if self.tracer_set == "passive" and not self.tracer_fluxes:
            raise ValueError(
                'Releases with tracer_set="passive" must specify a '
                "'passive_tracer' flux."
            )
        return self

    @model_validator(mode="after")
    def _check_tracer_flux_lengths(self):
        num_times = len(self.times)
        for flux in self.tracer_fluxes.values():
            if isinstance(flux, Flux):
                flux.check_length(num_times)
        return self

    def _extend_to_endpoints(self, start_time, end_time):
        """Ensures that time series data includes endpoints at `start_time` and
        `end_time`.

        Pads each tracer flux if needed to match the full time window.
        Also ensures `self.times` includes the endpoints.
        """
        for flux in self.tracer_fluxes.values():
            flux.extend_to_endpoints(self.times, start_time, end_time)
        self._extend_times_to_endpoints(start_time, end_time)

    @staticmethod
    def get_tracer_metadata(tracer_set: TracerSet = "marbl"):
        """Returns long names and expected units for the tracer fluxes."""
        return RELEASE_TRACER_MODELS[tracer_set].release_metadata(unit_type="flux")

    def _do_accounting(
        self,
        roms_time_stamps: np.ndarray,
        model_reference_date: datetime,
    ) -> dict[str, float]:
        """
        Compute time-integrated tracer quantities over ROMS time steps.

        This method interpolates tracer flux time series from the CDR schedule
        onto the provided ROMS time stamps (in days since model reference date),
        then applies a "left-hold" rule: the interpolated value at t₀ is applied
        across the full interval [t₀, t₁).

        Parameters
        ----------
        roms_time_stamps : np.ndarray
            1D array of ROMS time stamps (days since `model_reference_date`).
            Must be strictly increasing.
        model_reference_date : datetime
            Reference date of the ROMS model calendar.

        Returns
        -------
        dict[str, float]
            Dictionary mapping tracer names to the total integrated quantity over
            the entire ROMS time period. Each value is the sum of the interpolated
            tracer fluxes multiplied by the corresponding ROMS time step durations.
        """
        tracer_series_dict = {
            tracer: np.asarray(flux.values)
            if isinstance(flux, Flux)
            else np.asarray(flux)
            for tracer, flux in self.tracer_fluxes.items()
        }
        return self._compute_integrated_tracers(
            roms_time_stamps, model_reference_date, tracer_series_dict
        )

    @model_serializer(mode="wrap")
    def _simplified_dump(self, pydantic_serializer) -> dict:
        """Return a simplified dict representation with flattened values."""
        # pydantic_serializer is a function that runs pydantic's default conversion
        # of a model to a dict. then, we make some custom modifications after that
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pydantic_serializer(self)

        # Flatten tracer_fluxes
        if "tracer_fluxes" in data:
            simplified = {}
            for tracer, contents in data["tracer_fluxes"].items():
                simplified[tracer] = contents["values"]
            data["tracer_fluxes"] = simplified

        return data
