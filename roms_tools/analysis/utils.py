def _validate_plot_inputs(field, time, s, eta, xi, depth, lat, lon, include_boundary):
    """Validate input parameters for the plot method.

    Parameters
    ----------
    field : xr.DataArray
        Input data to be plotted.
    time : int
        Time index to select. Must be within bounds if the variable has a time dimension.
    s : int, float, or None
        Depth level index or value for the s-coordinate. Use None for surface plotting.
    eta : int or None
        Eta index for ROMS grid selection. Must be within bounds.
    xi : int or None
        Xi index for ROMS grid selection. Must be within bounds.
    depth : int, float, or None
        Depth value for slicing. Not yet implemented.
    lat : float or None
        Latitude value for slicing. Must be specified with `lon` if provided.
    lon : float or None
        Longitude value for slicing. Must be specified with `lat` if provided.
    include_boundary : bool
        Whether to include boundary points when selecting grid indices.

    Raises
    ------
    ValueError
        If time is out of bounds or not valid for a variable without a time dimension.
        If conflicting dimensions are specified.
        If eta or xi indices are out of bounds.
        If eta or xi lie on the boundary when `include_boundary=False`.
    NotImplementedError
        If only one of `lat` or `lon` is provided (feature not yet implemented)
        If `depth` is specified (feature not yet implemented).
    """

    # Check time dimension
    if "time" in field.dims:
        if time >= len(field.time):
            raise ValueError(
                f"Invalid time index: The specified time index ({time}) exceeds the maximum index "
                f"({len(field.time) - 1}) for the 'time' dimension."
            )
    else:
        if time > 0:
            raise ValueError(
                f"Invalid input: The field does not have a 'time' dimension, "
                f"but a time index ({time}) greater than 0 was provided."
            )

    # Check for unsupported lat/lon combinations
    if (lat is None) != (lon is None):
        raise NotImplementedError(
            "Both `lat` and `lon` must be specified together, or neither."
        )

    # Check if depth is specified, which is not yet supported
    if depth is not None:
        raise NotImplementedError(
            "Plotting at a specific depth is not implemented yet."
        )

    # Check conflicting dimension choices
    if s is not None and depth is not None:
        raise ValueError(
            "Conflicting input: You cannot specify both 's' and 'depth' at the same time."
        )
    if any([eta is not None, xi is not None]) and any(
        [lat is not None, lon is not None]
    ):
        raise ValueError(
            "Conflicting input: You cannot specify 'lat' or 'lon' simultaneously with 'eta' or 'xi'."
        )

    # 3D fields: Check for valid dimension specification
    if len(field.dims) == 3:
        if not any(
            [
                s is not None,
                eta is not None,
                xi is not None,
                depth is not None,
                lat is not None,
                lon is not None,
            ]
        ):
            raise ValueError(
                "Invalid input: For 3D fields, you must specify at least one of the dimensions 's', 'eta', 'xi', 'depth', 'lat', or 'lon'."
            )
        if sum([dim is not None for dim in [s, eta, xi, depth, lat, lon]]) > 2:
            raise ValueError(
                "Ambiguous input: For 3D fields, specify at most two of 's', 'eta', 'xi', 'depth', 'lat', or 'lon'. Specifying more than two is not allowed."
            )

    # 2D fields: Check for conflicts in dimension choices
    if len(field.dims) == 2:
        if all([eta is not None, xi is not None]):
            raise ValueError(
                "Conflicting input: For 2D fields, specify only one dimension, either 'eta' or 'xi', not both."
            )
        if all([lat is not None, lon is not None]):
            raise ValueError(
                "Conflicting input: For 2D fields, specify only one dimension, either 'lat' or 'lon', not both."
            )

    # Check that indices are within bounds for 2D fields
    if eta is not None:
        dim = "eta_rho" if "eta_rho" in field.dims else "eta_v"
        if not eta < len(field[dim]):
            raise ValueError(
                f"Invalid eta index: {eta} is out of bounds. Must be between 0 and {len(field[dim]) - 1}."
            )
        if not include_boundary:
            if eta == 0 or eta == len(field[dim]) - 1:
                raise ValueError(
                    f"Invalid eta index: {eta} lies on the boundary, which is excluded when `include_boundary = False`. "
                    "Either set `include_boundary = True`, or adjust eta to avoid boundary values."
                )

    if xi is not None:
        dim = "xi_rho" if "xi_rho" in field.dims else "xi_u"
        if not xi < len(field[dim]):
            raise ValueError(
                f"Invalid eta index: {xi} is out of bounds. Must be between 0 and {len(field[dim]) - 1}."
            )
        if not include_boundary:
            if xi == 0 or xi == len(field[dim]) - 1:
                raise ValueError(
                    f"Invalid xi index: {xi} lies on the boundary, which is excluded when `include_boundary = False`. "
                    "Either set `include_boundary = True`, or adjust eta to avoid boundary values."
                )
