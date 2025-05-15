from dataclasses import dataclass
from abc import abstractmethod, ABC
from pydantic import BaseModel, model_validator, Field, ConfigDict
from typing import List, Dict, Union, Literal, Optional
from typing_extensions import Annotated
from annotated_types import Ge, Le
from datetime import datetime
import warnings
from roms_tools.setup.utils import get_tracer_defaults

NonNegativeFloat = Annotated[float, Ge(0)]


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
    values: Union[float, List[float]]

    def check_length(self, num_times: int):
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

    values: Union[NonNegativeFloat, List[NonNegativeFloat]]

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
            Start of the interval to cover.
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
        start_pad = self.values[0] if isinstance(self.values, list) else self.values
        end_pad = self.values[-1] if isinstance(self.values, list) else self.values
        return self._extend_scalar_series(
            times, start_time, end_time, start_pad=start_pad, end_pad=end_pad
        )


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
    """

    name: str
    lat: Annotated[float, Ge(-90), Le(90)]
    lon: float
    depth: NonNegativeFloat
    hsc: NonNegativeFloat = 0.0
    vsc: NonNegativeFloat = 0.0
    times: List[datetime]

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_increasing_times(self) -> "Release":
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

    def extend_times_to_endpoints(self, start_time, end_time) -> None:
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


class VolumeRelease(Release):
    """Represents a CDR release with volume flux and tracer concentrations.

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
    """

    times: Optional[List[datetime]] = None
    volume_fluxes: Union[NonNegativeFloat, List[NonNegativeFloat], Flux] = Field(
        default=0.0
    )
    tracer_concentrations: Optional[
        Dict[str, Union[NonNegativeFloat, List[NonNegativeFloat]]]
    ] = None
    fill_values: Literal["auto", "zero"] = "auto"

    @model_validator(mode="after")
    def _postprocess(self) -> "VolumeRelease":
        if self.times is None:
            self.times = []
        num_times = len(self.times)

        if self.tracer_concentrations is None:
            self.tracer_concentrations = {}

        defaults = get_tracer_defaults()
        for tracer_name in defaults.keys():
            if tracer_name not in self.tracer_concentrations:
                if tracer_name in ["temp", "salt"]:
                    self.tracer_concentrations[tracer_name] = defaults[tracer_name]
                else:
                    if self.fill_values == "auto":
                        self.tracer_concentrations[tracer_name] = defaults[tracer_name]
                    elif self.fill_values == "zero":
                        self.tracer_concentrations[tracer_name] = 0.0

        if not isinstance(self.volume_fluxes, Flux):
            self.volume_fluxes = Flux("volume", self.volume_fluxes)
        self.volume_fluxes.check_length(num_times)

        self.tracer_concentrations = {
            tracer: (
                conc
                if isinstance(conc, Concentration)
                else Concentration(name=tracer, values=conc)
            )
            for tracer, conc in self.tracer_concentrations.items()
        }
        for tracer_concentrations in self.tracer_concentrations.values():
            tracer_concentrations.check_length(num_times)

        return self

    def _extend_to_endpoints(self, start_time, end_time) -> "VolumeRelease":
        """Ensures that time series data includes endpoints at `start_time` and
        `end_time`.

        Pads `volume_fluxes` and each tracer concentration if needed to match the full time window.
        Also ensures `self.times` includes the endpoints.
        """
        self.volume_fluxes.extend_to_endpoints(self.times, start_time, end_time)
        for conc in self.tracer_concentrations.values():
            conc.extend_to_endpoints(self.times, start_time, end_time)
        self.extend_times_to_endpoints(start_time, end_time)

    def _simplified_dump(self) -> dict:
        """Return a simplified dict representation with flattened values."""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = self.model_dump()
        data["release_type"] = "VolumeRelease"

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

    tracer_fluxes : dict, optional

        Dictionary of tracer names and their non-negative flux values. The flux values can be either
        a float (constant in time) or a list of float (time-varying).

        - Constant: applies uniformly across the entire simulation period.
        - Time-varying: must match the length of `times`.

        Default is an empty dictionary (`{}`) if not provided.

        Example:

        - Constant: `{"ALK": 2000.0, "DIC": 1900.0}`
        - Time-varying: `{"ALK": [2000.0, 2050.0, 2013.3], "DIC": [1900.0, 1920.0, 1910.2]}`
        - Mixed: `{"ALK": 2000.0, "DIC": [1900.0, 1920.0, 1910.2]}`
    """

    times: Optional[List[datetime]] = None
    tracer_fluxes: Optional[
        Dict[str, Union[NonNegativeFloat, List[NonNegativeFloat]]]
    ] = None

    @model_validator(mode="after")
    def _postprocess(self) -> "TracerPerturbation":
        if self.times is None:
            self.times = []
        num_times = len(self.times)

        if self.tracer_fluxes is None:
            self.tracer_fluxes = {}

        # Fill all tracer fluxes that are not provided with zero
        defaults = get_tracer_defaults()
        for tracer_name in defaults.keys():
            if tracer_name not in self.tracer_fluxes:
                self.tracer_fluxes[tracer_name] = 0.0

        self.tracer_fluxes = {
            tracer: (flux if isinstance(flux, Flux) else Flux(name=tracer, values=flux))
            for tracer, flux in self.tracer_fluxes.items()
        }

        for flux in self.tracer_fluxes.values():
            flux.check_length(num_times)

        return self

    def _extend_to_endpoints(self, start_time, end_time) -> "TracerPerturbation":
        """Ensures that time series data includes endpoints at `start_time` and
        `end_time`.

        Pads each tracer flux if needed to match the full time window.
        Also ensures `self.times` includes the endpoints.
        """
        self.volume_fluxes.extend_to_endpoints(self.times, start_time, end_time)
        for flux in self.tracer_fluxes.values():
            flux.extend_to_endpoints(self.times, start_time, end_time)
        self.extend_times_to_endpoints(start_time, end_time)

    def _simplified_dump(self) -> dict:
        """Return a simplified dict representation with flattened values."""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = self.model_dump()
        data["release_type"] = "TracerPerturbation"

        # Flatten tracer_fluxes
        if "tracer_fluxes" in data:
            simplified = {}
            for tracer, contents in data["tracer_fluxes"].items():
                simplified[tracer] = contents["values"]
            data["tracer_fluxes"] = simplified

        return data
