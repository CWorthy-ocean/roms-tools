from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from roms_tools import Grid, TidalForcing
from roms_tools.datasets.download import download_test_data
from roms_tools.setup.tides_rutgers import (
    _rutgers_phase,
    _zero_phase_date,
    tidal_ellipse,
)
from roms_tools.utils import interpolate_from_rho_to_u, rotate_velocities

MODEL_REFERENCE_DATE = datetime(2012, 1, 1, 12)


def _tpxo10v2a_files():
    return {
        key: Path(download_test_data(f"regional_{key}_tpxo10v2a.nc"))
        for key in ["grid", "h", "u"]
    }


def _tidal_forcing(convention, use_dask):
    grid = Grid(
        nx=3, ny=3, size_x=1500, size_y=1500, center_lon=235, center_lat=25, rot=-20
    )
    return TidalForcing(
        grid=grid,
        source={"name": "TPXO", "path": _tpxo10v2a_files()},
        ntides=3,
        model_reference_date=MODEL_REFERENCE_DATE,
        convention=convention,
        use_dask=use_dask,
    )


@pytest.fixture(scope="session")
def tidal_forcing_ucla(use_dask):
    return _tidal_forcing("ucla", use_dask)


@pytest.fixture(scope="session")
def tidal_forcing_rutgers(use_dask):
    return _tidal_forcing("rutgers", use_dask)


def _ucla_signal(re, im, omega_t):
    """UCLA-ROMS reconstruction, ``Re cos(wt) - Im sin(wt)`` (ucla-roms tides.F90)."""
    return re * np.cos(omega_t) - im * np.sin(omega_t)


def _rutgers_currents(cmax, cmin, cangle, cphase, omega_t):
    """Rutgers ROMS current reconstruction (set_tides.F) with a grid angle of zero."""
    angle = np.deg2rad(cangle)
    phase = omega_t - np.deg2rad(cphase)
    u = cmax * np.cos(angle) * np.cos(phase) - cmin * np.sin(angle) * np.sin(phase)
    v = cmax * np.sin(angle) * np.cos(phase) + cmin * np.cos(angle) * np.sin(phase)
    return u, v


@pytest.fixture
def random_harmonics():
    rng = np.random.default_rng(0)
    shape = (50,)

    def draw():
        return xr.DataArray(
            rng.normal(size=shape) + 1j * rng.normal(size=shape), dims="point"
        )

    return draw(), draw(), draw()


@pytest.fixture
def omega_t():
    return xr.DataArray(np.linspace(0, 2 * np.pi, 25), dims="time")


def test_rutgers_phase_reconstructs_elevation(random_harmonics, omega_t):
    zeta, _, _ = random_harmonics
    expected = _ucla_signal(zeta.real, zeta.imag, omega_t)
    actual = abs(zeta) * np.cos(omega_t - np.deg2rad(_rutgers_phase(zeta)))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_tidal_ellipse_reconstructs_currents(random_harmonics, omega_t):
    _, u, v = random_harmonics
    cmax, cmin, cangle, cphase = tidal_ellipse(u, v)

    assert (cmax >= abs(cmin)).all()
    assert ((cangle >= 0) & (cangle < 180)).all()
    assert ((cphase >= 0) & (cphase < 360)).all()

    u_rutgers, v_rutgers = _rutgers_currents(cmax, cmin, cangle, cphase, omega_t)
    np.testing.assert_allclose(
        u_rutgers, _ucla_signal(u.real, u.imag, omega_t), atol=1e-12
    )
    np.testing.assert_allclose(
        v_rutgers, _ucla_signal(v.real, v.imag, omega_t), atol=1e-12
    )


@pytest.mark.parametrize(
    "date, expected",
    [
        (datetime(2006, 1, 1), 20060101.0),
        (datetime(2005, 1, 1, 12), 20050101.5),
        (datetime(2000, 12, 31, 6), 20001231.25),
    ],
)
def test_zero_phase_date(date, expected):
    assert _zero_phase_date(date) == expected


def test_rutgers_output_variables(tidal_forcing_rutgers):
    ds = tidal_forcing_rutgers.ds

    for var_name in [
        "tide_Eamp",
        "tide_Ephase",
        "tide_Cmax",
        "tide_Cmin",
        "tide_Cangle",
        "tide_Cphase",
    ]:
        assert ds[var_name].dims == ("tide_period", "eta_rho", "xi_rho")
    assert ds["tide_period"].attrs["units"] == "hours"
    np.testing.assert_allclose(ds["tide_period"].values[0], 12.4206, atol=1e-3)
    assert ds["zero_phase_date"].item() == 20120101.5
    assert ds.attrs["tidal_constituents"] == "m2, s2, n2"
    for var_name in ["ssh_Re", "u_Re", "v_Re", "pot_Re", "omega"]:
        assert var_name not in ds


def test_rutgers_elevation_matches_ucla(tidal_forcing_ucla, tidal_forcing_rutgers):
    ucla = tidal_forcing_ucla.ds
    rutgers = tidal_forcing_rutgers.ds.rename({"tide_period": "ntides"})

    zeta = ucla.ssh_Re.astype(np.float64) + 1j * ucla.ssh_Im.astype(np.float64)
    np.testing.assert_allclose(rutgers.tide_Eamp.values, abs(zeta).values, rtol=1e-5)

    omega_t = xr.DataArray(np.linspace(0, 2 * np.pi, 13), dims="time")
    expected = _ucla_signal(ucla.ssh_Re, ucla.ssh_Im, omega_t)
    actual = rutgers.tide_Eamp * np.cos(omega_t - np.deg2rad(rutgers.tide_Ephase))
    np.testing.assert_allclose(
        actual.transpose(*expected.dims).values, expected.values, atol=1e-5
    )


def test_rutgers_currents_match_ucla(tidal_forcing_ucla, tidal_forcing_rutgers):
    """Rutgers ellipses, reconstructed as east/north velocity, rotated to the grid and
    staggered to u-points, reproduce the UCLA-ROMS u-velocity.
    """
    ucla = tidal_forcing_ucla.ds
    rutgers = tidal_forcing_rutgers.ds.rename({"tide_period": "ntides"})
    grid_ds = tidal_forcing_rutgers.grid.ds

    omega_t = xr.DataArray(np.linspace(0, 2 * np.pi, 13), dims="time")
    u_east, v_north = _rutgers_currents(
        rutgers.tide_Cmax,
        rutgers.tide_Cmin,
        rutgers.tide_Cangle,
        rutgers.tide_Cphase,
        omega_t,
    )
    u_grid, _ = rotate_velocities(u_east, v_north, grid_ds.angle)
    u_from_rutgers = interpolate_from_rho_to_u(u_grid)

    u_from_ucla = _ucla_signal(ucla.u_Re, ucla.u_Im, omega_t)

    wet = grid_ds.mask_u == 1
    np.testing.assert_allclose(
        u_from_rutgers.where(wet).transpose(*u_from_ucla.dims).values,
        u_from_ucla.where(wet).values,
        atol=1e-6,
    )


def test_rutgers_save(tidal_forcing_rutgers, tmp_path):
    (filepath,) = tidal_forcing_rutgers.save(tmp_path / "frc_tides.nc")
    with xr.open_dataset(filepath) as ds:
        assert "tide_Cmax" in ds
        assert ds["zero_phase_date"].item() == 20120101.5


def test_rutgers_plot(tidal_forcing_rutgers):
    tidal_forcing_rutgers.plot(var_name="tide_Eamp", ntides=0)


def test_rutgers_roundtrip_yaml(tidal_forcing_rutgers, tmp_path, use_dask):
    yaml_filepath = tmp_path / "test_yaml.yaml"
    tidal_forcing_rutgers.to_yaml(yaml_filepath)
    tidal_forcing_from_file = TidalForcing.from_yaml(yaml_filepath, use_dask=use_dask)

    assert tidal_forcing_from_file.convention == "rutgers"
    assert tidal_forcing_rutgers == tidal_forcing_from_file


def test_invalid_convention(use_dask):
    with pytest.raises(ValueError, match="`convention` must be"):
        _tidal_forcing("coawst", use_dask)
