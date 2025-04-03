@dataclass(kw_only=True)
class CDRPipeForcing:
    """Represents CDR pipe forcing data for ROMS, supporting both constant and time-varying tracers and volumes.

    Parameters
    ----------
    grid : Grid, optional
        Object representing the grid information for spatial distribution.
    start_time : datetime
        Start time of the model simulation.
    end_time : datetime
        End time of the model simulation.
    model_reference_date : datetime, optional
        Reference date for the model. Default is January 1, 2000.

    Attributes
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the CDR pipe forcing data, which includes tracers, volumes, and associated time information.
    """
    
    grid: Optional['Grid'] = None
    start_time : datetime
    end_time : datetime
    model_reference_date: datetime = datetime(2000, 1, 1)

    def __post_init__(self):

        if self.start_time >= self.end_time:
            raise ValueError("start_time must be before end_time.")

        self.ds = xr.Dataset(
            {
                "cdr_time": (["time"], np.empty(0)),  # Empty time dimension initially
            },
            coords={
                "experiment_name": (["ncdr"], np.empty(0, dtype=str)),  # Empty initially
            }   
        )
        self._add_global_metadata()
        
    def _add_global_metadata(self):
        
        self.ds = self.ds.assign_coords(
                {
                "tracer_name": (["ntracers"], [
                    "temp", "salt", "PO4", "NO3", "SiO3", "NH4", "Fe", "Lig", "O2",
                    "DIC", "DIC_ALT_CO2", "ALK", "ALK_ALT_CO2", "DOC", "DON", "DOP",
                    "DOPr", "DONr", "DOCr", "zooC", "spChl", "spC", "spP", "spFe",
                    "spCaCO3", "diatChl", "diatC", "diatP", "diatFe", "diatSi", "diazChl",
                    "diazC", "diazP", "diazFe"
                ])
            }
        )

    def add_release(
        self,
        name: str,
        lat: float,
        lon: float,
        depth: float,
        release_start_time: Optional[datetime] = None,
        release_end_time: Optional[datetime] = None,
        times: Optional[List[datetime]] = None,  # Can be None for constant tracers
        fill_values: Optional[str] = "auto_fill",  # Default to "auto_fill" for missing tracer data
        tracers: Dict[str, Union[float, List[float]]],
        volume: Union[float, List[float]] = 0.0,  # Default volume to 0.0 if not provided
    ):
        """
        Parameters
        ----------
        name : str
            Name of the release experiment.
        lat : float
            Latitude of the CDR release. Must be between -90 and 90.
        lon : float
            Longitude of the CDR release. No restrictions on bounds; longitude can be any value.
        depth : float
            Depth of the CDR release.
        release_start_time : datetime, optional
            Start time of the CDR release. Required if `times` is `None`.
        release_end_time : datetime, optional
            End time of the CDR release. Required if `times` is `None`.
        times : List[datetime], optional
            List of times for time-varying tracers. Required if `release_start_time` and `release_end_time` are not provided.
        tracers : dict
            A dictionary of tracer names and their corresponding values, which can be constant or time-varying. 
            Example formats:
            - Constant tracers: {"temp": 20.0, "salt": 1.0, "ALK": 2000.0}
            - Time-varying tracers: {"temp": [19.5, 20, 20, 20], "salt": [1.1, 2, 1, 1], "ALK": [2000.0, 2014.3, 2001.0, 2004.2]} (with `times` set to four corresponding datetime entries)
        fill_values : Optional[str], default="auto_fill"
            Defines how missing tracer data should be handled:
            - "auto_fill": Use the model's default fill behavior.
            - "zero_fill": Fill missing values with zeros.
        volume : Union[float, List[float]], optional
            Volume of the CDR release, either constant (single float) or a time series (list of floats).
        """

        self._input_checks(name, lat, lon, depth, release_start_time, release_end_time, times, tracers, volume, fill_values)

        # append the experiment
        new_experiment_name = np.concatenate([ds1.experiment_name.values, [name]])
        self.ds = self.ds.update(
            {"experiment_name": (["ncdr"], new_experiment_name)}
        )

        if release_start_time and release_end_time and not times:
            # Use start and end times for constant tracers if no `times` provided
            times = [release_start_time, release_end_time]
        elif times and not (release_start_time and release_end_time):
            # Time-varying tracers: derive start and end time from `times`
            release_start_time, release_end_time = times[0], times[-1]
        else:
            raise ValueError("Either `times` must be provided, or both `release_start_time` and `release_end_time` must be specified.")

        # union with existing times
        if "cdr_time" in self.ds:
            existing_times = self.ds["cdr_time"].values
            union_time = np.union1d(existing_times, np.array(times))
        else:
            union_time = np.array(times)

        # interpolate existing volume and tracers onto union time
        if "cdr_volume" in self.ds:
            new_volume = np.interp(
                    union_time.astype(np.float64),
                    np.array(ds["time"]).astype(np.float64),
                    ds["cdr_volume"]
                    )
            ds["cdr_time"] = union_time
            ds["cdr_volume"] = new_volume


            if isinstance(volume, list):
                volume_data = np.interp(
                    union_time.astype(np.float64),
                    np.array(times).astype(np.float64),
                    volume
                )
            else:
                volume_data = np.full(len(union_time), volume)
        else:
            # If no existing time, just use the provided times and volume
            union_time = np.array(times)
            volume_data = np.array(volume if isinstance(volume, list) else [volume] * len(union_time))

        self.ds["time"] = union_time



    def _input_checks(name, lat, lon, depth, release_start_time, release_end_time, times, tracers, volume, fill_values):
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
                raise ValueError("`release_start_time` cannot be before `self.start_time`.")
        
        # Check that release_end_time is not after end_time
        if release_end_time:
            if release_end_time > self.end_time:
                raise ValueError("`release_end_time` cannot be after `self.end_time`.")

        # Check that release_start_time is before release_end_time
        if release_start_time and release_end_time:
            if release_start_time >= release_end_time:
                raise ValueError("release_start_time must be before release_end_time.")

        # Ensure that times is either None (for constant tracers) or a list of datetimes
        if self.times is not None and not all(isinstance(t, datetime) for t in self.times):
            raise ValueError("If 'times' is provided, all entries must be datetime objects.")

        if times is not None and times[0] < release_start_time:
            raise ValueError("First entry in `times` cannot be before `release_start_time`.")

        if times is not None and times[-1] > release_end_time:
            raise ValueError("Last entry in `times` cannot be after `release_end_time`.")

        # Ensure that tracers dictionary is not empty for time-varying forcing
        if self.times is not None and not self.tracers:
            raise ValueError("The 'tracers' dictionary cannot be empty when 'times' is provided.")
            raise ValueError("The 'tracers' dictionary cannot be empty.")

        # Check that volume is valid
        if isinstance(self.volume, float) and self.volume < 0:
            raise ValueError("Volume must be a non-negative number.")
        elif isinstance(self.volume, list) and not all(v >= 0 for v in self.volume):
            raise ValueError("All entries in 'volume' list must be non-negative numbers.")

    # Ensure that time series for 'times', 'volume', and tracers are all the same length
    if self.times is not None:
        num_times = len(self.times)
        
        # Check that volume is either a constant or has the same length as 'times'
        if isinstance(self.volume, list) and len(self.volume) != num_times:
            raise ValueError(f"The length of 'volume' ({len(self.volume)}) does not match the length of 'times' ({num_times}).")
        
        # Check that each time-varying tracer has the same length as 'times'
        for key, tracer_values in self.tracers.items():
            if isinstance(tracer_values, list) and len(tracer_values) != num_times:
                raise ValueError(f"The length of tracer '{key}' ({len(tracer_values)}) does not match the length of 'times' ({num_times}).")
