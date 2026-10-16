"""Conversion of complex tidal harmonics to the Rutgers ROMS tidal forcing format.

ROMS-Tools natively produces UCLA-ROMS tidal forcing: complex harmonics
``Z = Re + i Im`` that UCLA-ROMS reconstructs as ``Re cos(wt) - Im sin(wt)``,
i.e. ``Real[Z exp(iwt)]``, with ``t`` measured from the model reference date.

Rutgers ROMS instead reads amplitude/phase for the free surface and tidal
current ellipse parameters (see ``Data/ROMS/CDL/frc_tides.cdl`` in the Rutgers
ROMS repository) and reconstructs ``A cos(wt - phi)``, with ``t`` measured from
``zero_phase_date``. The two are related by ``A = |Z|`` and ``phi = -arg(Z)``.
Current ellipses follow the Rutgers MATLAB ``ap2ep`` routine, with the
inclination measured counterclockwise from East (Rutgers ROMS rotates the
ellipses into grid directions itself).
"""

from datetime import datetime

import numpy as np
import xarray as xr

# Rutgers tidal variables checked for NaNs at wet points
RUTGERS_VALIDATED_VARS = ("tide_Eamp", "tide_Cmax")


def _rutgers_phase(z: xr.DataArray) -> xr.DataArray:
    """Phase lag in degrees, in [0, 360), such that ``Real[z exp(iwt)] = |z| cos(wt -
    phase)``.
    """
    return np.mod(-np.rad2deg(np.arctan2(z.imag, z.real)), 360.0)


def tidal_ellipse(
    u: xr.DataArray, v: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Convert complex east/north current harmonics to tidal ellipse parameters.

    This follows the Rutgers MATLAB ``ap2ep`` routine, applied to complex
    amplitudes ``u``, ``v`` for which the velocity is ``Real[u exp(iwt)]``,
    ``Real[v exp(iwt)]``.

    Parameters
    ----------
    u, v : xr.DataArray
        Complex harmonics of the eastward and northward velocity.

    Returns
    -------
    cmax : xr.DataArray
        Semi-major axis (same units as ``u``, ``v``).
    cmin : xr.DataArray
        Semi-minor axis; negative for clockwise rotation.
    cangle : xr.DataArray
        Inclination of the semi-major axis, degrees counterclockwise from East, in [0, 180).
    cphase : xr.DataArray
        Phase of maximum current along the semi-major axis, degrees in [0, 360).
    """
    # Counterclockwise and clockwise rotating components
    wp = (u + 1j * v) / 2
    wm = np.conj(u - 1j * v) / 2
    theta_p = np.arctan2(wp.imag, wp.real)
    theta_m = np.arctan2(wm.imag, wm.real)

    cmax = abs(wp) + abs(wm)
    cmin = abs(wp) - abs(wm)
    inclination = np.rad2deg(0.5 * (theta_m + theta_p))
    phase = np.rad2deg(0.5 * (theta_m - theta_p))

    # Flipping the semi-major axis by 180 degrees is compensated by a half-cycle phase shift
    flip = inclination < 0
    cangle = xr.where(flip, inclination + 180.0, inclination)
    cphase = np.mod(xr.where(flip, phase + 180.0, phase), 360.0)

    return cmax, cmin, cangle, cphase


def _zero_phase_date(date: datetime) -> float:
    """Encode a datetime as the Rutgers ``zero_phase_date`` value, ``YYYYMMDD.dddd``."""
    midnight = date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_fraction = (date - midnight).total_seconds() / 86400.0
    return float(date.strftime("%Y%m%d")) + day_fraction


def complex_to_rutgers(
    fields: dict[str, xr.DataArray],
    omega: xr.DataArray,
    zero_phase_date: datetime,
) -> xr.Dataset:
    """Build a Rutgers ROMS tidal forcing dataset from complex tidal harmonics.

    Parameters
    ----------
    fields : dict[str, xr.DataArray]
        Real and imaginary parts ``ssh_Re``, ``ssh_Im``, ``u_Re``, ``u_Im``, ``v_Re``
        and ``v_Im`` on rho-points, with dimension ``ntides``. Velocities must be
        eastward/northward barotropic velocities in m/s.
    omega : xr.DataArray
        Angular frequencies (rad/s) along ``ntides``.
    zero_phase_date : datetime
        Date at which the harmonics have zero phase (the date the phases were
        corrected to).

    Returns
    -------
    xr.Dataset
        Dataset with ``tide_period`` (hours), ``tide_Eamp``, ``tide_Ephase``,
        ``tide_Cmax``, ``tide_Cmin``, ``tide_Cangle``, ``tide_Cphase`` and
        ``zero_phase_date``, following the Rutgers ROMS ``frc_tides.cdl`` template.
    """
    labels = [
        label.decode() if isinstance(label, bytes) else str(label)
        for label in fields["ssh_Re"]["ntides"].values
    ]

    def _complex(name: str) -> xr.DataArray:
        z = fields[f"{name}_Re"] + 1j * fields[f"{name}_Im"]
        return z.reset_coords(drop=True).drop_vars("ntides", errors="ignore")

    zeta = _complex("ssh")
    cmax, cmin, cangle, cphase = tidal_ellipse(_complex("u"), _complex("v"))

    variables = {
        "tide_Eamp": (abs(zeta), "tidal elevation amplitude", "meter"),
        "tide_Ephase": (
            _rutgers_phase(zeta),
            "tidal elevation phase angle",
            "degrees, time of maximum elevation with respect to chosen time origin",
        ),
        "tide_Cmax": (
            cmax,
            "maximum tidal current, ellipse semi-major axis",
            "meter second-1",
        ),
        "tide_Cmin": (
            cmin,
            "minimum tidal current, ellipse semi-minor axis",
            "meter second-1",
        ),
        "tide_Cangle": (
            cangle,
            "tidal current inclination angle",
            "degrees between semi-major axis and East",
        ),
        "tide_Cphase": (
            cphase,
            "tidal current phase angle",
            "degrees, time of maximum velocity with respect chosen time origin",
        ),
    }

    ds = xr.Dataset()
    for name, (da, long_name, units) in variables.items():
        ds[name] = da.rename({"ntides": "tide_period"}).astype(np.float32)
        ds[name].attrs = {"long_name": long_name, "units": units}

    period = 2 * np.pi / omega.values / 3600.0
    ds = ds.assign_coords(
        tide_period=(
            "tide_period",
            period,
            {"long_name": "tide angular period", "units": "hours"},
        )
    )

    ds["zero_phase_date"] = xr.DataArray(
        _zero_phase_date(zero_phase_date),
        attrs={
            "long_name": "tidal reference date for zero phase",
            "units": "days as %Y%m%d.%f",
            "C_format": "%13.4f",
            "FORTRAN_format": "(f13.4)",
        },
    )

    ds.attrs["type"] = "ROMS Forcing File"
    ds.attrs["tidal_constituents"] = ", ".join(labels)

    return ds
