from dataclasses import dataclass, field
from abc import abstractmethod, ABC
from pydantic import BaseModel, model_validator
from typing import List, Dict, Union
from typing_extensions import Annotated
from annotated_types import Ge, Le
from datetime import datetime

NonNegativeFloat = Annotated[float, Ge(0)]

@dataclass
class ValueArray(ABC):
    """
    Abstract base class representing a time series of values, either scalar or list.

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
        """
        Checks that the number of values matches the number of time steps.

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

    def _extend_scalar_series(self, times: list, start_time, end_time, start_pad, end_pad):
        """
        Internal utility to convert a scalar to a list and extend the series at endpoints.

        Parameters
        ----------
        times : list
            List of datetime-like objects.
        start_time : datetime-like
            Start of the interval.
        end_time : datetime-like
            End of the interval.
        start_pad : float
            Value to insert at the beginning if padding is needed.
        end_pad : float
            Value to append at the end if padding is needed.

        Returns
        -------
        self : ValueArray
            The updated instance with extended `values`.
        """
        if not times:
            self.values = [self.values, self.values]
        else:
            self.values = (
                list(self.values)
                if isinstance(self.values, list)
                else [self.values] * len(times)
            )

            if times[0] > start_time:
                self.values.insert(0, start_pad)
            if times[-1] < end_time:
                self.values.append(end_pad)

        return self

    @abstractmethod
    def extend_to_endpoints(self, times: list, start_time, end_time):
        """
        Abstract method to extend the value series to cover given time endpoints.

        Must be implemented in subclasses.
        """
        raise NotImplementedError()

@dataclass
class Flux(ValueArray):
    """
    Represents a time series of non-negative flux values.

    Attributes
    ----------
    name : str
        Name of the flux variable.
    values : Union[NonNegativeFloat, List[NonNegativeFloat]]
        A constant non-negative flux or a list of non-negative flux values.
    """

    values: Union[NonNegativeFloat, List[NonNegativeFloat]]
    
    def extend_to_endpoints(self, times: list, start_time, end_time):
        """
        Extends the flux series to ensure it covers the full time interval.

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
        return self._extend_scalar_series(times, start_time, end_time, start_pad=0.0, end_pad=0.0)

@dataclass
class Concentration(ValueArray):
    """
    Represents a time series of tracer concentrations.

    Attributes
    ----------
    name : str
        Name of the tracer (e.g., 'NO3', 'DIC').
    values : Union[float, List[float]]
        A constant concentration or a list of concentrations over time.
    """
    def extend_to_endpoints(self, times: list, start_time, end_time):
        """
        Extends the concentration series to ensure it covers the full time interval.

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
        start_pad = (
            self.values[0] if isinstance(self.values, list) else self.values
        )
        end_pad = (
            self.values[-1] if isinstance(self.values, list) else self.values
        )
        return self._extend_scalar_series(times, start_time, end_time, start_pad=start_pad, end_pad=end_pad)


class Release(BaseModel):
    """
    Defines the basic properties and timing of a carbon dioxide removal (CDR) release.

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
    start_time : datetime
        Start of the simulation.
    end_time : datetime
        End of the simulation.
    """

    name: str
    lat: Annotated[float, Ge(-90), Le(90)]
    lon: float
    depth: NonNegativeFloat
    hsc: NonNegativeFloat
    vsc: NonNegativeFloat
    times: List[datetime]

    start_time: datetime
    end_time: datetime

    @model_validator(mode="after")
    def check_times(self) -> "Release":
        """
        Validates that `times` are strictly increasing and fall within the specified time window.

        Raises
        ------
        ValueError
            If times are not strictly increasing, or fall outside the [start_time, end_time] window.
        """

        if len(self.times) > 0:

            # Ensure times are strictly increasing
            if not all(t1 < t2 for t1, t2 in zip(self.times, self.times[1:])):
                raise ValueError(
                    f"'times' must be strictly monotonically increasing. Got: {self.times}"
                )

            # First time must not be before start_time
            if self.times[0] < self.start_time:
                raise ValueError(
                    f"First time in 'times' cannot be before start_time ({self.start_time})."
                )

            # Last time must not be after end_time
            if self.times[-1] > self.end_time:
                raise ValueError(
                    f"Last time in 'times' cannot be after end_time ({self.end_time})."
                )
        return self

    def extend_times_to_endpoints(self) -> None:
        """
        Ensures that `times` includes both `start_time` and `end_time`.

        Modifies `self.times` in place by prepending or appending times as needed.
        If `times` is empty, it will be set to [`start_time`, `end_time`].
        """

        if not self.times:
            self.times = [self.start_time, self.end_time]
        else:
            self.times = list(self.times)  # Make mutable
            if self.times[0] > self.start_time:
                self.times.insert(0, self.start_time)
            if self.times[-1] < self.end_time:
                self.times.append(self.end_time)

class PointVolumeRelease(Release):
    """
    Represents a point-source CDR release with volume flux and optional tracer concentrations.

    Extends `Release` by specifying the volume flux and associated tracers released over time.

    Attributes
    ----------
    volume_fluxes : Flux
        Volume flux of the release in m³/s. May be a constant or a time series aligned with `times`.
    tracer_concentrations : dict of str -> Concentration
        Dictionary mapping tracer names to their concentrations. Each value may be a constant or time series.
    hsc : float, optional
        Horizontal scale of the release in meters. Defaults to 0.0.
    vsc : float, optional
        Vertical scale of the release in meters. Defaults to 0.0.
    """

    hsc: NonNegativeFloat = 0.0
    vsc: NonNegativeFloat = 0.0  
    volume_fluxes: Flux = field(default_factory=lambda: Flux(name="volume", values=0.0))
    tracer_concentrations: Dict[str, Concentration] = field(default_factory=dict)

    @model_validator(mode="after")
    def extend_to_endpoints(self) -> "PointVolumeRelease":
        """
        Ensures that time series data includes endpoints at `start_time` and `end_time`.

        Pads `volume_fluxes` and each tracer concentration if needed to match the full time window.
        Also ensures `self.times` includes the endpoints.

        Returns
        -------
        self : PointVolumeRelease
            The updated object with time and data series extended to include endpoints.
        """
        self.volume_fluxes.extend_to_endpoints(self.times, self.start_time, self.end_time)
        for conc in self.tracer_concentrations.values():
            conc.extend_to_endpoints(self.times, self.start_time, self.end_time)
        self.extend_times_to_endpoints()

        return self
