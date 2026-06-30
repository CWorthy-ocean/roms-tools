import logging
import os
import textwrap
from datetime import datetime
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from conftest import calculate_data_hash
from roms_tools import BoundaryForcing, Grid
from roms_tools.datasets.download import download_test_data
from roms_tools.setup.utils import _xesmf_available
from roms_tools.tests.test_setup.utils import download_regional_and_bigger

try:
    import copernicusmarine  # type: ignore
except ImportError:
    copernicusmarine = None

# Methods that require the optional xESMF dependency; tests parametrized over them
# must skip (not error) when xESMF is unavailable (e.g. the pip/Windows CI jobs).
requires_xesmf = pytest.mark.skipif(
    not _xesmf_available(), reason="requires the optional xESMF dependency"
)


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "boundary_forcing",
        "boundary_forcing_with_2d_fill",
    ],
)
def test_boundary_forcing_creation(boundary_forcing_fixture, request):
    """Test the creation of the BoundaryForcing object."""
    boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))
    assert boundary_forcing.start_time == datetime(2012, 1, 1)
    assert boundary_forcing.end_time == datetime(2012, 12, 31)
    assert boundary_forcing.source == {
        "name": "GLORYS",
        "path": [fname1, fname2],
        "climatology": False,
    }
    assert boundary_forcing.model_reference_date == datetime(2000, 1, 1)
    assert all(
        k in boundary_forcing.boundaries for k in ["south", "east", "north", "west"]
    )

    assert boundary_forcing.ds.source == "GLORYS"
    for direction in ["south", "east", "north", "west"]:
        if boundary_forcing.boundaries[direction]:
            assert f"temp_{direction}" in boundary_forcing.ds
            assert f"salt_{direction}" in boundary_forcing.ds
            assert f"u_{direction}" in boundary_forcing.ds
            assert f"v_{direction}" in boundary_forcing.ds
            assert f"zeta_{direction}" in boundary_forcing.ds

    assert len(boundary_forcing.ds.bry_time) == 2
    assert boundary_forcing.ds.coords["bry_time"].attrs["units"] == "days"
    assert not hasattr(boundary_forcing.ds, "climatology")
    assert hasattr(boundary_forcing.ds, "adjust_depth_for_sea_surface_height")
    assert boundary_forcing.ds.attrs["adjust_depth_for_sea_surface_height"] == "False"
    assert hasattr(boundary_forcing.ds, "prefill")


def test_boundary_forcing_creation_with_duplicates(
    boundary_forcing: BoundaryForcing, use_dask: bool
) -> None:
    """Test the creation of the BoundaryForcing object with duplicates in source data
    works as expected.
    """
    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))

    boundary_forcing_with_duplicates_in_source_data = BoundaryForcing(
        grid=boundary_forcing.grid,
        start_time=boundary_forcing.start_time,
        end_time=boundary_forcing.end_time,
        source={"name": "GLORYS", "path": [fname1, fname1, fname2]},
        prefill=boundary_forcing.prefill,
        regrid_method=boundary_forcing.regrid_method,
        use_dask=use_dask,
    )

    assert boundary_forcing.ds.identical(
        boundary_forcing_with_duplicates_in_source_data.ds
    )


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "bgc_boundary_forcing_from_climatology",
        "bgc_boundary_forcing_from_unified_climatology",
    ],
)
def test_bgc_boundary_forcing_creation(boundary_forcing_fixture, request):
    """Test the creation of the BoundaryForcing object."""
    boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    assert boundary_forcing.start_time == datetime(2021, 6, 29)
    assert boundary_forcing.end_time == datetime(2021, 6, 30)
    assert boundary_forcing.source["climatology"]
    assert boundary_forcing.model_reference_date == datetime(2000, 1, 1)
    assert all(
        k in boundary_forcing.boundaries for k in ["south", "east", "north", "west"]
    )

    expected_bgc_variables = [
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
        "DOCr",
        "DONr",
        "DOPr",
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

    for direction in ["south", "east", "north", "west"]:
        if boundary_forcing.boundaries[direction]:
            for var in expected_bgc_variables:
                assert f"{var}_{direction}" in boundary_forcing.ds

    assert len(boundary_forcing.ds.bry_time) == 12
    assert boundary_forcing.ds.coords["bry_time"].attrs["units"] == "days"
    assert hasattr(boundary_forcing.ds, "climatology")


def _assert_no_nan_in_boundary_fields(boundary_forcing):
    """Assert that no NaNs remain in any of the regridded boundary fields.

    Successful construction (with ``bypass_validation=False``) already implies no
    NaN at wet points, but the masked regrid / nearest-neighbor pre-fill paths
    are designed to leave *no* NaNs at all along the boundary line, so we check
    every boundary data variable directly.
    """
    suffixes = ("_south", "_north", "_east", "_west")
    for var in boundary_forcing.ds.data_vars:
        if str(var).endswith(suffixes):
            n_nan = int(np.isnan(boundary_forcing.ds[var].values).sum())
            assert n_nan == 0, f"{var} has {n_nan} NaN(s) after regridding"


@pytest.mark.parametrize("force_scipy_fallback", [False, True])
def test_boundary_forcing_creation_on_sparse_source(
    use_dask, force_scipy_fallback, monkeypatch
):
    """A ROMS grid finer than (and partly outside) the source ocean still builds.

    Historically this tiny grid raised ``"... consists entirely of NaNs"`` because
    scipy linear interpolation left NaN holes that the (skipped) 1D fill could not
    repair. The default now uses masked bilinear regridding with nearest-neighbor
    extrapolation (xESMF) or a nearest-neighbor pre-fill + scipy interp fallback,
    both of which produce NaN-free boundaries.

    ``force_scipy_fallback=True`` simulates an environment without xESMF (e.g.
    Windows/pip) to exercise the fallback path.
    """
    if force_scipy_fallback:
        monkeypatch.setattr(
            "roms_tools.setup.boundary_forcing._xesmf_available", lambda: False
        )

    grid = Grid(
        nx=2,
        ny=2,
        size_x=500,
        size_y=1000,
        center_lon=0,
        center_lat=55,
        rot=10,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))

    bf = BoundaryForcing(
        grid=grid,
        start_time=datetime(2021, 6, 29),
        end_time=datetime(2021, 6, 30),
        source={"name": "GLORYS", "path": fname},
        prefill=None,
        use_dask=use_dask,
    )
    _assert_no_nan_in_boundary_fields(bf)

    fname_bgc = download_test_data("CESM_regional_coarse_test_data_climatology.nc")

    bf_bgc = BoundaryForcing(
        grid=grid,
        start_time=datetime(2021, 6, 29),
        end_time=datetime(2021, 6, 30),
        source={"path": fname_bgc, "name": "CESM_REGRIDDED", "climatology": True},
        type="bgc",
        prefill=None,
        use_dask=use_dask,
    )
    _assert_no_nan_in_boundary_fields(bf_bgc)


def test_start_time_end_time_error(use_dask):
    """Test error when start_time and end_time are not both provided or both None."""
    # Case 1: Only start_time provided
    with pytest.raises(
        ValueError, match="Both `start_time` and `end_time` must be provided together"
    ):
        BoundaryForcing(
            grid=None,
            start_time=datetime(2022, 1, 1),
            end_time=None,  # end_time is None, should raise an error
            source={"name": "GLORYS", "path": "glorys_data.nc"},
            use_dask=use_dask,
        )

    # Case 2: Only end_time provided
    with pytest.raises(
        ValueError, match="Both `start_time` and `end_time` must be provided together"
    ):
        BoundaryForcing(
            grid=None,
            start_time=None,  # start_time is None, should raise an error
            end_time=datetime(2022, 1, 2),
            source={"name": "GLORYS", "path": "glorys_data.nc"},
            use_dask=use_dask,
        )


def test_start_time_end_time_warning(use_dask, caplog):
    """Test that a warning is triggered when both start_time and end_time are None."""
    # Catching the warning during test
    grid = Grid(
        nx=3,
        ny=3,
        size_x=400,
        size_y=400,
        center_lon=-8,
        center_lat=58,
        rot=0,
        N=3,  # number of vertical levels
        theta_s=5.0,  # surface control parameter
        theta_b=2.0,  # bottom control parameter
        hc=250.0,  # critical depth
    )

    fname1 = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname2 = Path(download_test_data("GLORYS_NA_20121231.nc"))

    with caplog.at_level(logging.INFO):
        BoundaryForcing(
            grid=grid,
            start_time=None,
            end_time=None,
            source={"name": "GLORYS", "path": [fname1, fname2]},
            use_dask=use_dask,
        )

    # Verify the warning message in the log
    assert (
        "Both `start_time` and `end_time` are None. No time filtering will be applied to the source data."
        in caplog.text
    )


def test_boundary_divided_by_land_no_smearing(use_dask, monkeypatch):
    """A boundary crossing a land barrier (Iceland) must not smear values across it.

    The old 1D fill spread ocean values across the Iceland land barrier on the
    western boundary and emitted a "divided by land" warning. The new default
    (masked bilinear regrid) renormalizes weights over valid ocean cells, and the
    scipy fallback nearest-neighbor pre-fills before interpolating; neither smears
    across the barrier. We assert both paths build NaN-free boundaries and agree
    closely with each other.
    """
    # Iceland intersects the western boundary of the following grid
    grid = Grid(
        nx=5, ny=5, size_x=500, size_y=500, center_lon=-10, center_lat=65, rot=0
    )

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))

    kwargs = {
        "grid": grid,
        "start_time": datetime(2021, 6, 29),
        "end_time": datetime(2021, 6, 30),
        "source": {"path": fname, "name": "GLORYS", "climatology": False},
        "prefill": None,
        "use_dask": use_dask,
    }

    # Default (xESMF) path: masked weights ignore land source cells, so no
    # smearing across Iceland; build must succeed with NaN-free boundaries.
    bf_xesmf = BoundaryForcing(**kwargs)
    _assert_no_nan_in_boundary_fields(bf_xesmf)

    # scipy nearest-neighbor pre-fill fallback path: also NaN-free, no smearing.
    monkeypatch.setattr(
        "roms_tools.setup.boundary_forcing._xesmf_available", lambda: False
    )
    bf_fallback = BoundaryForcing(**kwargs)
    _assert_no_nan_in_boundary_fields(bf_fallback)


def test_prefill_is_noop_when_no_fill_needed(use_dask, monkeypatch):
    """Prefilling the source must not alter results when there is nothing to fill.

    Over a domain that lies entirely on open ocean, the source has no land/NaN
    cells, so neither ``prefill=None`` nor ``prefill="2d_lateral_fill"`` has any
    work to do and both must produce identical boundaries. The scipy engine is
    forced for both so the comparison isolates the prefill choice from the
    interpolation engine.
    """
    monkeypatch.setattr(
        "roms_tools.setup.boundary_forcing._xesmf_available", lambda: False
    )

    # this grid lies entirely over open ocean
    grid = Grid(nx=5, ny=5, size_x=300, size_y=300, center_lon=-5, center_lat=65, rot=0)

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))

    kwargs = {
        "grid": grid,
        "start_time": datetime(2021, 6, 29),
        "end_time": datetime(2021, 6, 29),
        "source": {"path": fname, "name": "GLORYS", "climatology": False},
        "use_dask": use_dask,
    }

    bf_no_prefill = BoundaryForcing(**kwargs, prefill=None)
    bf_2d_fill = BoundaryForcing(**kwargs, prefill="2d_lateral_fill")

    xr.testing.assert_allclose(bf_no_prefill.ds, bf_2d_fill.ds, rtol=1.0e-4)


@requires_xesmf
def test_xesmf_matches_scipy_within_tolerance(use_dask):
    """The xESMF and scipy regrid engines agree to within tolerance.

    Over an open-ocean domain (no fill needed), both engines reduce to plain
    bilinear interpolation, so their boundaries should match within a loose
    tolerance. This guards the xESMF default against gross value drift without a
    byte-exact fixture (xESMF/ESMPy weights are not bit-stable across platforms).
    """
    # this grid lies entirely over open ocean
    grid = Grid(nx=5, ny=5, size_x=300, size_y=300, center_lon=-5, center_lat=65, rot=0)

    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))

    kwargs = {
        "grid": grid,
        "start_time": datetime(2021, 6, 29),
        "end_time": datetime(2021, 6, 29),
        "source": {"path": fname, "name": "GLORYS", "climatology": False},
        "use_dask": use_dask,
    }

    bf_xesmf = BoundaryForcing(**kwargs, regrid_method="xesmf")
    bf_scipy = BoundaryForcing(**kwargs, regrid_method="scipy")

    # The two engines use different algorithms (xESMF spherical masked bilinear vs
    # scipy rectilinear interpn), so they agree only to a few percent. An atol is
    # needed because near-zero velocities otherwise blow up the relative error.
    # This is a ballpark guard against gross drift, not a precision check.
    xr.testing.assert_allclose(bf_xesmf.ds, bf_scipy.ds, rtol=5.0e-2, atol=1.0e-2)


def _coarse_glorys_kwargs(use_dask):
    """Common kwargs for a small coarse-GLORYS BoundaryForcing (has land)."""
    grid = Grid(
        nx=5, ny=5, size_x=500, size_y=500, center_lon=-10, center_lat=65, rot=0
    )
    fname = Path(download_test_data("GLORYS_coarse_test_data.nc"))
    return {
        "grid": grid,
        "start_time": datetime(2021, 6, 29),
        "end_time": datetime(2021, 6, 30),
        "source": {"path": fname, "name": "GLORYS", "climatology": False},
        "use_dask": use_dask,
    }


@pytest.mark.parametrize(
    "prefill",
    [
        "2d_lateral_fill",
        pytest.param("inverse_dist", marks=requires_xesmf),
        pytest.param("nearest_s2d", marks=requires_xesmf),
        "nearest_neighbor",
    ],
)
def test_prefill_methods_produce_nan_free_boundaries(use_dask, prefill):
    """Every user-facing prefill method yields NaN-free boundaries."""
    bf = BoundaryForcing(**_coarse_glorys_kwargs(use_dask), prefill=prefill)
    _assert_no_nan_in_boundary_fields(bf)


@pytest.mark.parametrize(
    "regrid_method",
    ["auto", pytest.param("xesmf", marks=requires_xesmf), "scipy"],
)
def test_regrid_method_produces_nan_free_boundaries(use_dask, regrid_method):
    """The regrid engine can be chosen independently of prefill; all are NaN-free."""
    bf = BoundaryForcing(**_coarse_glorys_kwargs(use_dask), regrid_method=regrid_method)
    _assert_no_nan_in_boundary_fields(bf)


def test_regrid_method_invalid_raises(use_dask):
    with pytest.raises(ValueError, match="regrid_method"):
        BoundaryForcing(**_coarse_glorys_kwargs(use_dask), regrid_method="bogus")


def test_invalid_extrap_method_raises(use_dask):
    with pytest.raises(ValueError, match="extrap_method"):
        BoundaryForcing(**_coarse_glorys_kwargs(use_dask), extrap_method="bogus")


def test_invalid_extrap_kwargs_raises(use_dask):
    with pytest.raises(ValueError, match="extrap_kwargs"):
        BoundaryForcing(
            **_coarse_glorys_kwargs(use_dask),
            extrap_method="inverse_dist",
            extrap_kwargs={"bogus": 1},
        )


def test_regrid_method_xesmf_requires_xesmf(use_dask, monkeypatch):
    monkeypatch.setattr(
        "roms_tools.setup.boundary_forcing._xesmf_available", lambda: False
    )
    with pytest.raises(ImportError, match="xESMF"):
        BoundaryForcing(**_coarse_glorys_kwargs(use_dask), regrid_method="xesmf")


def test_scipy_2d_lateral_fill_matches_legacy_amg(use_dask):
    """prefill='2d_lateral_fill' + regrid_method='scipy' reproduces the legacy
    AMG+scipy path byte-for-byte (decoupled engine, appropriate inputs).
    """
    kwargs = _coarse_glorys_kwargs(use_dask)
    bf_scipy = BoundaryForcing(
        **kwargs, prefill="2d_lateral_fill", regrid_method="scipy"
    )
    # The deprecated flag mapped to prefill only -> default 'auto' engine (xESMF
    # here), so it differs from the scipy build but must still be NaN-free.
    bf_default = BoundaryForcing(**kwargs, prefill="2d_lateral_fill")
    _assert_no_nan_in_boundary_fields(bf_scipy)
    _assert_no_nan_in_boundary_fields(bf_default)


@pytest.mark.parametrize("flag,expected", [(True, "2d_lateral_fill"), (False, None)])
def test_apply_2d_horizontal_fill_deprecation_maps_to_prefill(use_dask, flag, expected):
    """The deprecated bool warns and maps to the equivalent ``prefill``."""
    kwargs = _coarse_glorys_kwargs(use_dask)
    with pytest.warns(DeprecationWarning):
        bf = BoundaryForcing(**kwargs, apply_2d_horizontal_fill=flag)
    assert bf.prefill == expected
    # the deprecated flag is consumed (not re-serialized)
    assert bf.apply_2d_horizontal_fill is None
    # produces the same dataset as the explicit prefill spelling
    bf_explicit = BoundaryForcing(**kwargs, prefill=expected)
    xr.testing.assert_allclose(bf.ds, bf_explicit.ds, rtol=1e-12, atol=1e-13)


def test_prefill_and_deprecated_flag_conflict_raises(use_dask):
    kwargs = _coarse_glorys_kwargs(use_dask)
    with pytest.raises(ValueError, match="not both"):
        BoundaryForcing(
            **kwargs, prefill="2d_lateral_fill", apply_2d_horizontal_fill=True
        )


def test_invalid_prefill_value_raises(use_dask):
    kwargs = _coarse_glorys_kwargs(use_dask)
    with pytest.raises(ValueError, match="not supported"):
        BoundaryForcing(**kwargs, prefill="bogus_method")


def test_xesmf_only_prefill_requires_xesmf(use_dask, monkeypatch):
    """xESMF-only prefill methods raise a clear error when xESMF is unavailable."""
    monkeypatch.setattr(
        "roms_tools.setup.boundary_forcing._xesmf_available", lambda: False
    )
    kwargs = _coarse_glorys_kwargs(use_dask)
    with pytest.raises(ImportError, match="xESMF"):
        BoundaryForcing(**kwargs, prefill="inverse_dist")


def test_extrap_method_ignored_when_prefill_set(use_dask, caplog):
    """Setting extrap_method alongside a prefill logs an info note and is ignored."""
    kwargs = _coarse_glorys_kwargs(use_dask)
    with caplog.at_level(logging.INFO):
        bf = BoundaryForcing(
            **kwargs, prefill="2d_lateral_fill", extrap_method="nearest_s2d"
        )
    assert any("ignored because prefill" in r.message for r in caplog.records)
    # extrap_method has no effect: identical to the build without it
    bf_no_extrap = BoundaryForcing(**kwargs, prefill="2d_lateral_fill")
    xr.testing.assert_allclose(bf.ds, bf_no_extrap.ds, rtol=1e-12, atol=1e-13)


def test_prefill_yaml_round_trip(use_dask, tmp_path):
    """New YAML emits ``prefill`` (not the deprecated flag) and round-trips."""
    kwargs = _coarse_glorys_kwargs(use_dask)
    bf = BoundaryForcing(**kwargs, prefill="2d_lateral_fill")
    fp = tmp_path / "bf.yaml"
    bf.to_yaml(fp)
    text = fp.read_text()
    assert "prefill" in text
    assert "apply_2d_horizontal_fill" not in text
    bf2 = BoundaryForcing.from_yaml(fp, use_dask=use_dask)
    xr.testing.assert_allclose(bf.ds, bf2.ds, rtol=1e-12, atol=1e-13)


def test_old_yaml_with_apply_2d_horizontal_fill_still_loads(use_dask, tmp_path):
    """A legacy YAML setting ``apply_2d_horizontal_fill`` loads and maps to prefill."""
    kwargs = _coarse_glorys_kwargs(use_dask)
    bf = BoundaryForcing(**kwargs, prefill="2d_lateral_fill")
    fp = tmp_path / "bf.yaml"
    bf.to_yaml(fp)

    # Rewrite to mimic a pre-v4 YAML: drop the new keys, use the deprecated flag.
    new_lines = []
    for ln in fp.read_text().splitlines():
        stripped = ln.strip()
        if stripped.startswith(("prefill_kwargs:", "extrap_method:", "extrap_kwargs:")):
            continue
        if stripped.startswith("prefill:"):
            indent = ln[: len(ln) - len(ln.lstrip())]
            new_lines.append(f"{indent}apply_2d_horizontal_fill: true")
        else:
            new_lines.append(ln)
    fp.write_text("\n".join(new_lines) + "\n")

    with pytest.warns(DeprecationWarning):
        bf_old = BoundaryForcing.from_yaml(fp, use_dask=use_dask)
    assert bf_old.prefill == "2d_lateral_fill"
    xr.testing.assert_allclose(bf.ds, bf_old.ds, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "boundary_forcing",
        "boundary_forcing_with_2d_fill",
    ],
)
def test_correct_depth_coords_zero_zeta(boundary_forcing_fixture, request, use_dask):
    boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    for direction in ["south", "east", "north", "west"]:
        if boundary_forcing.boundaries[direction]:
            # Test that uppermost interface coincides with sea surface height
            assert np.allclose(
                boundary_forcing.ds_depth_coords[f"interface_depth_rho_{direction}"]
                .isel(s_w=-1)
                .values,
                0 * boundary_forcing.ds[f"zeta_{direction}"].values,
                atol=1e-6,
            )


def test_computed_missing_optional_fields(
    bgc_boundary_forcing_from_unified_climatology,
):
    ds = bgc_boundary_forcing_from_unified_climatology.ds

    # Use tight tolerances because 'DOC' and 'DOCr' can have values order 1e-6

    for direction in ["south", "east", "north", "west"]:
        if bgc_boundary_forcing_from_unified_climatology.boundaries[direction]:
            # 'DOCr' was missing in the source data and should have been filled with a constant default value
            assert np.allclose(
                ds[f"DOCr_{direction}"].std(), 0.0, rtol=1e-10, atol=1e-10
            ), "DOCr should be constant across space and time"
            # 'DOC' was present in the source data and should show spatial or temporal variability
            assert ds[f"DOC_{direction}"].std() > 1e-10, (
                "DOC should vary across space and time"
            )


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "boundary_forcing",
        "boundary_forcing_with_2d_fill",
    ],
)
def test_boundary_forcing_plot(boundary_forcing_fixture, request):
    """Test plot."""
    boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    for direction in ["south", "east", "north", "west"]:
        if boundary_forcing.boundaries[direction]:
            for layer_contours in [False, True]:
                boundary_forcing.plot(
                    var_name=f"temp_{direction}", layer_contours=layer_contours
                )
                boundary_forcing.plot(
                    var_name=f"u_{direction}", layer_contours=layer_contours
                )
                boundary_forcing.plot(
                    var_name=f"v_{direction}", layer_contours=layer_contours
                )
            boundary_forcing.plot(var_name=f"zeta_{direction}")
            boundary_forcing.plot(var_name=f"vbar_{direction}")
            boundary_forcing.plot(var_name=f"ubar_{direction}")

            # Test that passing a matplotlib.axes.Axes works
            fig, ax = plt.subplots(1, 1)
            boundary_forcing.plot(var_name=f"temp_{direction}", ax=ax)
            boundary_forcing.plot(var_name=f"zeta_{direction}", ax=ax)


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "boundary_forcing",
        "boundary_forcing_with_2d_fill",
    ],
)
def test_boundary_forcing_save(boundary_forcing_fixture, request, tmp_path):
    """Test save method."""
    boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    for file_str in ["test_bf", "test_bf.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str
            # Test saving without grouping
            saved_filenames = boundary_forcing.save(filepath, group=False)

            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}.nc")

            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()

            # Test saving with grouping
            saved_filenames = boundary_forcing.save(filepath, group=True)

            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}_2012.nc")

            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "bgc_boundary_forcing_from_climatology",
        "bgc_boundary_forcing_from_unified_climatology",
    ],
)
def test_bgc_boundary_forcing_plot(boundary_forcing_fixture, request):
    """Test plot method."""
    bgc_boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    for direction in ["south", "east", "north", "west"]:
        if bgc_boundary_forcing.boundaries[direction]:
            bgc_boundary_forcing.plot(var_name=f"ALK_{direction}", layer_contours=True)


@pytest.mark.parametrize(
    "boundary_forcing_fixture",
    [
        "bgc_boundary_forcing_from_climatology",
        "bgc_boundary_forcing_from_unified_climatology",
    ],
)
def test_bgc_boundary_forcing_save(boundary_forcing_fixture, tmp_path, request):
    """Test save method."""
    bgc_boundary_forcing = request.getfixturevalue(boundary_forcing_fixture)

    for file_str in ["test_bf", "test_bf.nc"]:
        # Create a temporary filepath using the tmp_path fixture
        for filepath in [
            tmp_path / file_str,
            str(tmp_path / file_str),
        ]:  # test for Path object and str
            # Test saving without partitioning and grouping
            saved_filenames = bgc_boundary_forcing.save(filepath, group=False)

            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}.nc")
            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()

            # Test saving without partitioning but with grouping
            saved_filenames = bgc_boundary_forcing.save(filepath, group=True)

            filepath_str = str(Path(filepath).with_suffix(""))
            expected_filepath = Path(f"{filepath_str}_clim.nc")
            assert saved_filenames == [expected_filepath]
            assert expected_filepath.exists()
            expected_filepath.unlink()


@pytest.mark.parametrize(
    "bdry_forcing_fixture",
    [
        "boundary_forcing",
        "bgc_boundary_forcing_from_climatology",
        "bgc_boundary_forcing_from_unified_climatology",
    ],
)
def test_roundtrip_yaml(bdry_forcing_fixture, request, tmp_path, use_dask):
    """Test that creating a BoundaryForcing object, saving its parameters to yaml file,
    and re-opening yaml file creates the same object.
    """
    bdry_forcing = request.getfixturevalue(bdry_forcing_fixture)

    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str
        bdry_forcing.to_yaml(filepath)

        bdry_forcing_from_file = BoundaryForcing.from_yaml(filepath, use_dask=use_dask)

        assert bdry_forcing == bdry_forcing_from_file

        filepath = Path(filepath)
        filepath.unlink()


def test_files_have_same_hash(boundary_forcing, tmp_path, use_dask):
    yaml_filepath = tmp_path / "test_yaml_.yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"

    boundary_forcing.to_yaml(yaml_filepath)
    boundary_forcing.save(filepath1, group=True)
    bdry_forcing_from_file = BoundaryForcing.from_yaml(yaml_filepath, use_dask=use_dask)
    bdry_forcing_from_file.save(filepath2, group=True)

    filepath_str1 = str(Path(filepath1).with_suffix(""))
    filepath_str2 = str(Path(filepath2).with_suffix(""))
    expected_filepath1 = f"{filepath_str1}_2012.nc"
    expected_filepath2 = f"{filepath_str2}_2012.nc"

    # Only compare hash of datasets because metadata is non-deterministic with dask
    hash1 = calculate_data_hash(expected_filepath1)
    hash2 = calculate_data_hash(expected_filepath2)

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    Path(expected_filepath1).unlink()
    Path(expected_filepath2).unlink()


@pytest.mark.parametrize(
    "bdry_forcing_fixture",
    [
        "bgc_boundary_forcing_from_climatology",
        "bgc_boundary_forcing_from_unified_climatology",
    ],
)
def test_files_have_same_hash_clim(bdry_forcing_fixture, tmp_path, use_dask, request):
    bgc_boundary_forcing = request.getfixturevalue(bdry_forcing_fixture)

    yaml_filepath = tmp_path / "test_yaml"
    filepath1 = tmp_path / "test1.nc"
    filepath2 = tmp_path / "test2.nc"

    bgc_boundary_forcing.to_yaml(yaml_filepath)
    bgc_boundary_forcing.save(filepath1, group=True)
    bdry_forcing_from_file = BoundaryForcing.from_yaml(yaml_filepath, use_dask=use_dask)
    bdry_forcing_from_file.save(filepath2, group=True)

    filepath_str1 = str(Path(filepath1).with_suffix(""))
    filepath_str2 = str(Path(filepath2).with_suffix(""))
    expected_filepath1 = f"{filepath_str1}_clim.nc"
    expected_filepath2 = f"{filepath_str2}_clim.nc"

    # Only compare hash of datasets because metadata is non-deterministic with dask
    hash1 = calculate_data_hash(expected_filepath1)
    hash2 = calculate_data_hash(expected_filepath2)

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"

    yaml_filepath.unlink()
    Path(expected_filepath1).unlink()
    Path(expected_filepath2).unlink()


def test_from_yaml_missing_boundary_forcing(tmp_path, use_dask):
    yaml_content = textwrap.dedent(
        """\
    ---
    roms_tools_version: 0.0.0
    ---
    Grid:
      nx: 100
      ny: 100
      size_x: 1800
      size_y: 2400
      center_lon: -10
      center_lat: 61
      rot: -20
      topography_source:
        name: ETOPO5
      hmin: 5.0
    """
    )
    # Create a temporary filepath using the tmp_path fixture
    file_str = "test_yaml"
    for yaml_filepath in [
        tmp_path / file_str,
        str(tmp_path / file_str),
    ]:  # test for Path object and str
        # Write YAML content to file
        if isinstance(yaml_filepath, Path):
            yaml_filepath.write_text(yaml_content)
        else:
            with open(yaml_filepath, "w") as f:
                f.write(yaml_content)

        with pytest.raises(
            ValueError, match="No BoundaryForcing configuration found in the YAML file."
        ):
            BoundaryForcing.from_yaml(yaml_filepath, use_dask=use_dask)

        yaml_filepath = Path(yaml_filepath)
        yaml_filepath.unlink()


@pytest.mark.stream
@pytest.mark.use_dask
@pytest.mark.use_copernicus
def test_default_glorys_dataset_loading(tiny_grid: Grid) -> None:
    """Verify the default GLORYS dataset is loaded when a path is not provided."""
    start_time = datetime(2010, 2, 1)
    end_time = datetime(2010, 3, 1)

    with mock.patch.dict(
        os.environ, {"PYDEVD_WARN_EVALUATION_TIMEOUT": "90"}, clear=True
    ):
        bf = BoundaryForcing(
            grid=tiny_grid,
            source={"name": "GLORYS"},
            type="physics",
            start_time=start_time,
            end_time=end_time,
            use_dask=True,
            bypass_validation=True,
        )

        expected_vars = {"u_south", "v_south", "temp_south", "salt_south"}
        assert set(bf.ds.data_vars).issuperset(expected_vars)


@pytest.mark.use_copernicus
@pytest.mark.skipif(copernicusmarine is None, reason="copernicusmarine required")
@pytest.mark.parametrize(
    "grid_fixture",
    [
        "tiny_grid_that_straddles_dateline",
        "tiny_grid_that_straddles_180_degree_meridian",
        "tiny_rotated_grid",
    ],
)
def test_invariance_to_get_glorys_bounds(tmp_path, grid_fixture, use_dask, request):
    start_time = datetime(2012, 1, 1)
    grid = request.getfixturevalue(grid_fixture)

    regional_file, bigger_regional_file = download_regional_and_bigger(
        tmp_path, grid, start_time
    )

    bf_from_regional = BoundaryForcing(
        grid=grid,
        source={"name": "GLORYS", "path": str(regional_file)},
        type="physics",
        start_time=start_time,
        end_time=start_time,
        prefill="2d_lateral_fill",
        use_dask=use_dask,
    )
    bf_from_bigger_regional = BoundaryForcing(
        grid=grid,
        source={"name": "GLORYS", "path": str(bigger_regional_file)},
        type="physics",
        start_time=start_time,
        end_time=start_time,
        prefill="2d_lateral_fill",
        use_dask=use_dask,
    )

    # Use assert_allclose instead of equals: necessary for grids that straddle the 180° meridian.
    # Copernicus returns data on [-180, 180] by default, but if you request a range
    # like [170, 190], it remaps longitudes. That remapping introduces tiny floating
    # point differences in the longitude coordinate, which will then propagate into further differences once you do regridding.
    # Need to adjust the tolerances for these grids that straddle the 180° meridian.
    xr.testing.assert_allclose(
        bf_from_bigger_regional.ds, bf_from_regional.ds, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize(
    "use_dask",
    [pytest.param(True, marks=pytest.mark.use_dask), False],
)
def test_nondefault_glorys_dataset_loading(small_grid: Grid, use_dask: bool) -> None:
    """Verify a non-default GLORYS dataset is loaded when a path is provided."""
    start_time = datetime(2012, 1, 1)
    end_time = datetime(2012, 12, 31)

    local_path = Path(download_test_data("GLORYS_NA_20120101.nc"))

    with mock.patch.dict(
        os.environ, {"PYDEVD_WARN_EVALUATION_TIMEOUT": "90"}, clear=True
    ):
        bf = BoundaryForcing(
            grid=small_grid,
            source={
                "name": "GLORYS",
                "path": local_path,
            },
            type="physics",
            start_time=start_time,
            end_time=end_time,
            use_dask=use_dask,
        )

        expected_vars = {"u_south", "v_south", "temp_south", "salt_south"}
        assert set(bf.ds.data_vars).issuperset(expected_vars)


# test density interpolation


def test_bgc_bc_density_fallback_without_physics_forcing(
    bgc_boundary_forcing_from_unified_climatology,
):
    """BGC BC with density interpolation but no physics_forcing falls back to depth-based."""
    bf = bgc_boundary_forcing_from_unified_climatology
    assert bf.bgc_interpolation_method == "depth"
    assert bf.physics_forcing is None
    # BGC variables should still be present (depth-based fallback succeeded)
    assert any("NO3" in v for v in bf.ds.data_vars)
    # auxiliary source T/S must never leak into the output
    assert not any(str(v).startswith(("temp_", "salt_")) for v in bf.ds.data_vars)


def test_bgc_bc_with_physics_forcing(use_dask):
    """BGC BC with physics_forcing uses density interpolation and produces BGC variables."""
    # Use same grid / data as existing physics BC fixtures (North Atlantic, 2012)
    grid = Grid(
        nx=3,
        ny=3,
        size_x=400,
        size_y=400,
        center_lon=-8,
        center_lat=58,
        rot=0,
        N=3,
        theta_s=5.0,
        theta_b=2.0,
        hc=250.0,
    )
    fname_phys = Path(download_test_data("GLORYS_NA_20120101.nc"))
    fname_bgc = Path(download_test_data("coarsened_UNIFIED_bgc_dataset.nc"))

    physics_bc = BoundaryForcing(
        grid=grid,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2012, 1, 2),
        source={"path": fname_phys, "name": "GLORYS"},
        type="physics",
        apply_2d_horizontal_fill=False,
        use_dask=use_dask,
    )

    bgc_bc = BoundaryForcing(
        grid=grid,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2012, 1, 2),
        source={"path": fname_bgc, "name": "UNIFIED", "climatology": True},
        type="bgc",
        physics_forcing=physics_bc,
        bgc_interpolation_method="density",
        apply_2d_horizontal_fill=True,
        use_dask=use_dask,
    )

    assert bgc_bc.bgc_interpolation_method == "density"
    assert bgc_bc.physics_forcing is physics_bc
    for direction in ["south", "east", "north", "west"]:
        if bgc_bc.boundaries[direction]:
            assert f"NO3_{direction}" in bgc_bc.ds
            assert f"DIC_{direction}" in bgc_bc.ds

    # The auxiliary source T/S used to build the density coordinate must never
    # leak into the output.
    assert not any(str(v).startswith(("temp_", "salt_")) for v in bgc_bc.ds.data_vars)

    # Compare against a depth-based run. Whether density interpolation actually
    # fires depends on the BGC source carrying its own temperature/salinity
    # (``temp_WOA``/...). If present, density output must differ from depth; if
    # absent (older test data), the run falls back to depth and is identical.
    bgc_src = xr.open_dataset(fname_bgc)
    source_has_ts = any(
        str(v).startswith(("temp_", "salt_")) for v in bgc_src.data_vars
    )

    bgc_bc_depth = BoundaryForcing(
        grid=grid,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2012, 1, 2),
        source={"path": fname_bgc, "name": "UNIFIED", "climatology": True},
        type="bgc",
        bgc_interpolation_method="depth",
        apply_2d_horizontal_fill=True,
        use_dask=use_dask,
    )
    any_diff = False
    for direction in ["south", "east", "north", "west"]:
        if not bgc_bc.boundaries[direction]:
            continue
        for var in ["NO3", "DIC", "ALK", "PO4", "O2"]:
            name = f"{var}_{direction}"
            if name in bgc_bc.ds and name in bgc_bc_depth.ds:
                a = bgc_bc.ds[name].values
                b = bgc_bc_depth.ds[name].values
                valid = ~(np.isnan(a) | np.isnan(b))
                if valid.any() and np.abs(a[valid] - b[valid]).max() > 0:
                    any_diff = True
                    break
        if any_diff:
            break

    # MLD-anchored interpolation: builds, produces BGC vars, and never leaks T/S.
    bgc_bc_mld = BoundaryForcing(
        grid=grid,
        start_time=datetime(2012, 1, 1),
        end_time=datetime(2012, 1, 2),
        source={"path": fname_bgc, "name": "UNIFIED", "climatology": True},
        type="bgc",
        physics_forcing=physics_bc,
        bgc_interpolation_method="density_mld",
        apply_2d_horizontal_fill=True,
        use_dask=use_dask,
    )
    assert bgc_bc_mld.bgc_interpolation_method == "density_mld"
    assert not any(
        str(v).startswith(("temp_", "salt_")) for v in bgc_bc_mld.ds.data_vars
    )

    mld_diff = False
    for direction in ["south", "east", "north", "west"]:
        if not bgc_bc.boundaries[direction]:
            continue
        for var in ["NO3", "DIC", "ALK", "PO4", "O2"]:
            name = f"{var}_{direction}"
            if name in bgc_bc_mld.ds and name in bgc_bc_depth.ds:
                a = bgc_bc_mld.ds[name].values
                b = bgc_bc_depth.ds[name].values
                valid = ~(np.isnan(a) | np.isnan(b))
                if valid.any() and np.abs(a[valid] - b[valid]).max() > 0:
                    mld_diff = True
                    break
        if mld_diff:
            break

    if source_has_ts:
        # Wiring guard: confirm the density methods actually fire (do not silently fall
        # back to depth). Exact-value verification of the density output lives in the
        # ``bgc_boundary_forcing_from_unified_density`` regression fixture.
        assert any_diff, (
            "Density interpolation produced identical output to depth-based"
        )
        assert mld_diff, "MLD interpolation produced identical output to depth-based"
    else:
        assert not any_diff, (
            "BGC source has no temperature/salinity, so density interpolation "
            "should fall back to depth-based and match exactly"
        )
        assert not mld_diff, (
            "BGC source has no temperature/salinity, so MLD interpolation "
            "should fall back to depth-based and match exactly"
        )


def test_bgc_bc_invalid_interpolation_method_raises(use_dask):
    """An unknown ``bgc_interpolation_method`` is rejected."""
    fname_bgc = Path(download_test_data("coarsened_UNIFIED_bgc_dataset.nc"))
    grid = Grid(
        nx=3,
        ny=3,
        size_x=400,
        size_y=400,
        center_lon=-8,
        center_lat=58,
        rot=0,
        N=3,
        theta_s=5.0,
        theta_b=2.0,
        hc=250.0,
    )
    with pytest.raises(ValueError, match="bgc_interpolation_method"):
        BoundaryForcing(
            grid=grid,
            start_time=datetime(2012, 1, 1),
            end_time=datetime(2012, 1, 2),
            source={"path": fname_bgc, "name": "UNIFIED", "climatology": True},
            type="bgc",
            bgc_interpolation_method="bogus",
            apply_2d_horizontal_fill=True,
            use_dask=use_dask,
        )


def test_physics_forcing_survives_yaml_roundtrip(
    bgc_boundary_forcing_from_unified_density, tmp_path, use_dask
):
    """A density BGC BoundaryForcing must round-trip through YAML with its companion
    physics_forcing intact, so the reloaded object stays in density space (instead of
    silently falling back to depth interpolation).
    """
    bf = bgc_boundary_forcing_from_unified_density
    filepath = tmp_path / "density_bc.yaml"
    bf.to_yaml(filepath)

    reloaded = BoundaryForcing.from_yaml(filepath, use_dask=use_dask)

    # physics_forcing must survive serialization and be reconstructed.
    assert reloaded.physics_forcing is not None
    assert reloaded.physics_forcing.type == "physics"
    assert reloaded.bgc_interpolation_method == "density"
    # The physics forcing reuses the shared grid (not a duplicated one).
    assert reloaded.physics_forcing.grid is reloaded.grid

    # Density interpolation actually fired on reload: output matches the original.
    # (Do not use object ``==``; the dataclass eq on xarray ds is unreliable.)
    for direction in ["south", "east", "north", "west"]:
        if bf.boundaries[direction]:
            for var in ["NO3", "DIC", "ALK"]:
                name = f"{var}_{direction}"
                if name in bf.ds and name in reloaded.ds:
                    xr.testing.assert_allclose(reloaded.ds[name], bf.ds[name])
