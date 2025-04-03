from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Union
import numpy as np
import xarray as xr
from roms_tools import Grid


@dataclass(kw_only=True)
class CDRPipeForcing:
    """Represents CDR pipe forcing data for ROMS, supporting both constant and time-
    varying tracers and volumes.

    Parameters
    ----------
    grid : Grid, optional
        Object representing the grid for spatial context.
    start_time : datetime
        Start time of the model simulation.
    end_time : datetime
        End time of the model simulation.
    model_reference_date : datetime, optional
        Reference date for converting absolute times to model-relative time. Defaults to Jan 1, 2000.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray dataset containing CDR release metadata and forcing variables.
    """

    grid: Optional["Grid"] = None
    start_time: datetime
    end_time: datetime
    model_reference_date: datetime = datetime(2000, 1, 1)

    def __post_init__(self):
        if self.start_time >= self.end_time:
            raise ValueError("`start_time` must be earlier than `end_time`.")

        self.ds = xr.Dataset(
            {
                "cdr_time": (["time"], np.empty(0)),
                "cdr_volume": (["time", "ncdr"], np.empty((0, 0))),
                # "cdr_tracer": (["time", "ntracers", "ncdr"], np.empty((0, 0))),
            },
            coords={
                "experiment_name": (["ncdr"], np.empty(0, dtype=str)),
            },
        )
        self._add_global_metadata()

    def _add_global_metadata(self):
        tracer_names = [
            "temp",
            "salt",
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
            "DOPr",
            "DONr",
            "DOCr",
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
        self.ds = self.ds.assign_coords({"tracer_name": (["ntracers"], tracer_names)})

    def add_release(
        self,
        *,
        name: str,
        lat: float,
        lon: float,
        depth: float,
        release_start_time: Optional[datetime] = None,
        release_end_time: Optional[datetime] = None,
        times: Optional[List[datetime]] = None,
        tracers: Optional[Dict[str, Union[float, List[float]]]] = None,
        volume: Union[float, List[float]] = 0.0,
        fill_values: Optional[str] = "auto_fill",
    ):
        """Adds a CDR pipe release to the forcing dataset.

        Parameters
        ----------
        name : str
            Unique identifier for the experiment.
        lat : float
            Latitude of the release location. Must be between -90 and 90.
        lon : float
            Longitude of the release location. No restrictions on bounds; longitude can be any value.
        depth : float
            Depth of the release.
        release_start_time : datetime, optional
            Start time of the release. Required if `times` is `None`.
        release_end_time : datetime, optional
            End time of the release. Required if `times` is `None`.
        times : List[datetime], optional
            Explicit time points for time-varying tracers and volumes.
        tracers : dict, optional
            A dictionary of tracer names and their corresponding values, which can be constant or time-varying.
            Example formats:
            - Constant tracers: {"temp": 20.0, "salt": 1.0, "ALK": 2000.0}
            - Time-varying tracers: {"temp": [19.5, 20, 20, 20], "salt": [1.1, 2, 1, 1], "ALK": [2000.0, 2014.3, 2001.0, 2004.2]} (with `times` set to four corresponding datetime entries)
        volume : float or list of float, optional
            Volume of release over time.
        fill_values : str, optional
            Strategy for filling missing tracer values. Options: "auto_fill", "zero_fill".
        """
        self._input_checks(
            name,
            lat,
            lon,
            depth,
            release_start_time,
            release_end_time,
            times,
            tracers,
            volume,
            fill_values,
        )

        # Check that the name is unique
        if name in self.ds["experiment_name"].values:
            raise ValueError(
                f"A release experiment with the name '{name}' already exists."
            )

        if release_start_time and release_end_time and not times:
            times = [release_start_time, release_end_time]
        elif times and not (release_start_time and release_end_time):
            release_start_time, release_end_time = times[0], times[-1]
        else:
            raise ValueError(
                "Specify either `times`, or both `release_start_time` and `release_end_time`."
            )

        # Convert times to model-relative days
        times = np.array(times)
        times = (times - self.model_reference_date).astype(
            "timedelta64[ns]"
        ) / np.timedelta64(1, "D")

        # Merge with existing time dimension
        existing_times = (
            self.ds["cdr_time"].values if len(self.ds["cdr_time"]) > 0 else []
        )
        union_time = np.union1d(existing_times, times)

        # Initialize updated dataset
        ds = xr.Dataset()
        ds["cdr_time"] = ("time", union_time)

        experiment_names = np.concatenate([self.ds.experiment_name.values, [name]])
        ds = ds.assign_coords({"experiment_name": (["ncdr"], experiment_names)})
        ds["cdr_volume"] = xr.zeros_like(ds.cdr_time * ds.ncdr)
        # ds["cdr_tracer"] = xr.zeros_like(ds.cdr_time * ds.ncdr)

        # Interpolate and retain previous experiment volumes and tracer concentrations
        if len(self.ds["ncdr"]) > 0:
            for i in range(len(self.ds.ncdr)):
                for key in ["volume"]:  # , "tracer"]:
                    interpolated = np.interp(
                        union_time.astype(np.float64),
                        self.ds["cdr_time"].values.astype(np.float64),
                        self.ds[f"cdr_{key}"].isel(ncdr=i).values,
                    )
                    ds[f"cdr_{key}"].loc[{"ncdr": i}] = interpolated

        # Handle new experiment volume and tracer concentrations
        if isinstance(volume, list):
            new_volume = np.interp(
                union_time.astype(np.float64), times.astype(np.float64), volume
            )
        else:
            new_volume = np.full(len(union_time), volume)

        ds["cdr_volume"].loc[{"ncdr": ds.sizes["ncdr"] - 1}] = new_volume

        self.ds = ds
        self._add_global_metadata()

    def _input_checks(
        self,
        name,
        lat,
        lon,
        depth,
        release_start_time,
        release_end_time,
        times,
        tracers,
        volume,
        fill_values,
    ):
        # Check that lat is valid
        if not (-90 <= lat <= 90):
            raise ValueError("Latitude must be between -90 and 90.")

        # Check that depth is non-negative
        if depth < 0:
            raise ValueError("Depth must be a non-negative number.")

        if release_start_time >= release_end_time:
            raise ValueError("`release_start_time` must be before `release_end_time`.")

        # Check that release_start_time is not before start_time
        if release_start_time:
            if release_start_time < self.start_time:
                raise ValueError(
                    "`release_start_time` cannot be before `self.start_time`."
                )

        # Check that release_end_time is not after end_time
        if release_end_time:
            if release_end_time > self.end_time:
                raise ValueError("`release_end_time` cannot be after `self.end_time`.")

        # Check that release_start_time is before release_end_time
        if release_start_time and release_end_time:
            if release_start_time >= release_end_time:
                raise ValueError("release_start_time must be before release_end_time.")

        # Ensure that times is either None (for constant tracers) or a list of datetimes
        if times is not None and not all(isinstance(t, datetime) for t in times):
            raise ValueError(
                "If 'times' is provided, all entries must be datetime objects."
            )

        if times is not None and times[0] < release_start_time:
            raise ValueError(
                "First entry in `times` cannot be before `release_start_time`."
            )

        if times is not None and times[-1] > release_end_time:
            raise ValueError(
                "Last entry in `times` cannot be after `release_end_time`."
            )

        # Ensure that tracers dictionary is not empty for time-varying forcing
        # if times is not None and not tracers:
        #    raise ValueError(
        #        "The 'tracers' dictionary cannot be empty when 'times' is provided."
        #    )
        #    raise ValueError("The 'tracers' dictionary cannot be empty.")

        # Check that volume is valid
        if isinstance(volume, float) and volume < 0:
            raise ValueError("Volume must be a non-negative number.")
        elif isinstance(volume, list) and not all(v >= 0 for v in volume):
            raise ValueError(
                "All entries in 'volume' list must be non-negative numbers."
            )

        # Ensure that time series for 'times', 'volume', and tracers are all the same length
        if times is not None:
            num_times = len(times)

        # Check that volume is either a constant or has the same length as 'times'
        if isinstance(volume, list) and len(volume) != num_times:
            raise ValueError(
                f"The length of 'volume' ({len(volume)}) does not match the length of 'times' ({num_times})."
            )

        # Check that each time-varying tracer has the same length as 'times'
        # for key, tracer_values in tracers.items():
        #    if isinstance(tracer_values, list) and len(tracer_values) != num_times:
        #        raise ValueError(
        #            f"The length of tracer '{key}' ({len(tracer_values)}) does not match the length of 'times' ({num_times})."
        #        )
