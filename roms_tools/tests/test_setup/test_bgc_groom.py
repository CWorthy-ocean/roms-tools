import dask
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from roms_tools.setup.bgc_groom import (
    FLAG_BOUNDS,
    FLAG_FLOOR_JUMP,
    FLAG_JUMP,
    FLAG_PHYSICAL,
    bgc_check,
    bgc_groom,
    summarize_flags,
)
from roms_tools.vertical_coordinate import compute_depth_coordinates

NS, NY, NX = 10, 20, 24
TRACERS = ["ALK", "DIC", "O2", "NO3", "PO4", "SiO3"]


def _grid() -> xr.Dataset:
    """Flat 1000 m basin with an abrupt 100 m shelf at x >= 18 and a land corner."""
    h = np.full((NY, NX), 1000.0)
    h[:, 18:] = 100.0
    mask = np.ones((NY, NX))
    mask[:3, :3] = 0
    sigma = (np.arange(1, NS + 1) - NS - 0.5) / NS
    return xr.Dataset(
        {
            "h": (("eta_rho", "xi_rho"), h),
            "mask_rho": (("eta_rho", "xi_rho"), mask),
            "Cs_r": ("s_rho", sigma.copy()),
            "sigma_r": ("s_rho", sigma),
        },
        attrs={"hc": 250.0},
    )


def _fields(grid: xr.Dataset) -> dict[str, np.ndarray]:
    """Smooth, plausible profiles on (1, s, y, x); land is 0 as in roms-tools output."""
    depth = (
        compute_depth_coordinates(grid).transpose("s_rho", "eta_rho", "xi_rho").values
    )
    salt = np.full(depth.shape, 34.5)
    alk = 770 + (2300 - 770) * salt / 35 + 0.05 * depth
    f = {
        "ALK": alk,
        "DIC": alk * (0.88 + 5e-5 * depth),
        "O2": 200 - 0.15 * depth,
        "NO3": 0.03 * depth,
        "PO4": 0.002 * depth,
        "SiO3": 0.1 * depth,
        "salt": salt,
        "temp": np.full(depth.shape, 10.0),
    }
    wet = grid.mask_rho.values.astype(bool)[None]
    return {k: np.where(wet, v, 0.0)[None] for k, v in f.items()}


def _ic(fields: dict[str, np.ndarray], chunked: bool = True) -> xr.Dataset:
    dims = ("ocean_time", "s_rho", "eta_rho", "xi_rho")
    ds = xr.Dataset({k: (dims, v.astype(np.float32)) for k, v in fields.items()})
    return ds.chunk() if chunked else ds


def _plant(f: dict[str, np.ndarray]) -> dict[str, list]:
    """Plant one defect per check type; return the planted cells per tracer."""
    f["ALK"][0, 0:3, 5, 5] = 0.0  # bottom stack of zeros: bounds
    f["DIC"][0, 5, 10, 8] = 1.2 * f["ALK"][0, 5, 10, 8]  # DIC/ALK ratio: physical
    f["ALK"][0, 4, 12, 3] += 220.0  # nALK jump (and above the mixing band)
    f["O2"][0, 9, 3, 15] = 590.0  # far above saturation: physical
    f["NO3"][0, :, 8, 12] = 0.0  # whole column of zeros beside nitrate: floor_jump
    return {
        "ALK": [(0, k, 5, 5) for k in range(3)] + [(0, 4, 12, 3)],
        "DIC": [(0, 5, 10, 8)],
        "O2": [(0, 9, 3, 15)],
        "NO3": [
            (0, k, 8, 12) for k in range(0, 8)
        ],  # levels whose neighbours hold >= 5
    }


@pytest.fixture
def grid():
    return _grid()


@pytest.fixture
def clean(grid):
    return _fields(grid)


def test_clean_fields_have_no_flags_including_the_shelf_step(grid, clean):
    flags = bgc_check(_ic(clean), grid)
    counts = summarize_flags(flags)
    assert all(c["total"] == 0 for c in counts.values()), counts


def test_plume_front_and_hypersaline_water_on_the_mixing_line_pass(grid, clean):
    """Water on the 770 mixing line is consistent, however sharp the salinity front."""
    salt = clean["salt"].copy()
    salt[..., :, :8] = 8.0  # plume, with an abrupt front at x = 8
    salt[..., :, 8:12] = 39.0  # hypersaline band next to it
    wet = clean["salt"] > 0
    for v, n in (("ALK", 2300.0), ("DIC", 2050.0)):
        clean[v] = np.where(wet, 770 + (n - 770) * salt / 35, 0.0)
    clean["salt"] = np.where(wet, salt, 0.0)
    counts = summarize_flags(bgc_check(_ic(clean), grid, variables=["ALK", "DIC"]))
    assert counts["ALK"]["total"] == 0 and counts["DIC"]["total"] == 0, counts


def _plume(clean: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fresh plume (S = 2) at x < 6, a brackish column (S = 6) at x = 6, ocean beyond;
    ALK and DIC on the 770 mixing line everywhere.
    """
    salt = clean["salt"].copy()
    salt[..., :, :6] = 2.0
    salt[..., :, 6] = 6.0
    wet = clean["salt"] > 0
    for v, n in (("ALK", 2300.0), ("DIC", 2050.0)):
        clean[v] = np.where(wet, 770 + (n - 770) * salt / 35, 0.0)
    clean["salt"] = np.where(wet, salt, 0.0)
    return clean


def test_fresh_hole_is_filled_from_fresh_water_in_raw_values(grid, clean):
    """Below S_MIN_NORMALIZE values are not normalized; a fresh hole beside brackish
    water must be filled from its fresh neighbours, not from normalized values.
    """
    clean = _plume(clean)
    orig = clean["ALK"].copy()
    clean["ALK"][0, 5, 10, 5] = 1.2
    out = bgc_groom(_ic(clean), grid, variables=["ALK", "DIC"]).compute()
    np.testing.assert_allclose(
        out["ALK"].values[0, 5, 10, 5], orig[0, 5, 10, 5], rtol=1e-3
    )
    counts = summarize_flags(bgc_check(out, grid, variables=["ALK", "DIC"]))
    assert counts["ALK"]["total"] == 0, counts


def test_brackish_hole_beside_fresh_water_is_filled_on_the_mixing_line(grid, clean):
    """A normalized (S >= 5) hole must not take raw fresh values as donors."""
    clean = _plume(clean)
    orig = clean["ALK"].copy()
    clean["ALK"][0, 5, 10, 6] = 2600.0
    out = bgc_groom(_ic(clean), grid, variables=["ALK", "DIC"]).compute()
    np.testing.assert_allclose(
        out["ALK"].values[0, 5, 10, 6], orig[0, 5, 10, 6], rtol=1e-3
    )
    counts = summarize_flags(bgc_check(out, grid, variables=["ALK", "DIC"]))
    assert counts["ALK"]["total"] == 0, counts


def test_check_flags_each_planted_defect(grid, clean):
    planted = _plant(clean)
    flags = bgc_check(_ic(clean), grid).compute()
    for var, cells in planted.items():
        for cell in cells:
            assert flags[var].values[cell] > 0, (var, cell)
    assert flags["ALK"].values[0, 0, 5, 5] & FLAG_BOUNDS
    assert flags["DIC"].values[0, 5, 10, 8] & FLAG_PHYSICAL
    assert flags["ALK"].values[0, 4, 12, 3] & FLAG_JUMP
    assert flags["O2"].values[0, 9, 3, 15] & FLAG_PHYSICAL
    assert flags["NO3"].values[0, 6, 8, 12] & FLAG_FLOOR_JUMP
    for var in ("PO4", "SiO3"):
        assert int(flags[var].sum()) == 0


def test_groom_removes_every_flag_and_keeps_good_water(grid, clean):
    orig = {k: v.copy() for k, v in clean.items()}
    _plant(clean)
    out = bgc_groom(_ic(clean), grid).compute()
    counts = summarize_flags(bgc_check(out, grid))
    assert all(c["total"] == 0 for c in counts.values()), counts
    # Cells well away from every defect are untouched, bit for bit.
    far = np.ones((1, NS, NY, NX), dtype=bool)
    for y, x in [(5, 5), (10, 8), (12, 3), (3, 15), (8, 12)]:
        far[:, :, max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = False
    for var in TRACERS:
        np.testing.assert_array_equal(
            out[var].values[far], orig[var].astype(np.float32)[far]
        )
    # The filled ALK stack lands close to the clean profile.
    np.testing.assert_allclose(
        out["ALK"].values[0, 0:3, 5, 5], orig["ALK"][0, 0:3, 5, 5], rtol=0.01
    )
    assert out["ALK"].dtype == np.float32
    assert "bgc_groom" in out.attrs and "bgc_groom" in out["ALK"].attrs


@pytest.mark.parametrize("tile", [3, 7, 16])
def test_result_is_identical_for_any_tiling(grid, clean, tile):
    _plant(clean)
    ref = bgc_groom(_ic(clean), grid, tile=1000).compute()
    got = bgc_groom(_ic(clean), grid, tile=tile).compute()
    for var in TRACERS:
        assert np.array_equal(ref[var].values, got[var].values), var


def test_eager_input_matches_dask_input(grid, clean):
    _plant(clean)
    a = bgc_groom(_ic(clean, chunked=False), grid, tile=7).compute()
    b = bgc_groom(_ic(clean, chunked=True), grid, tile=7).compute()
    for var in TRACERS:
        assert np.array_equal(a[var].values, b[var].values), var


def test_nothing_is_computed_until_asked(grid, clean):
    _plant(clean)
    ran = []

    class Count(Callback):
        def _pretask(self, key, dsk, state):
            ran.append(key)

    with Count():
        out = bgc_groom(_ic(clean), grid, tile=7)
        flags = bgc_check(out, grid, tile=7)
    assert ran == []
    assert isinstance(out["ALK"].data, dask.array.Array)
    assert isinstance(flags["ALK"].data, dask.array.Array)


def test_region_wider_than_reach_stays_flagged_and_reported(grid, clean):
    clean["NO3"][0, :, 6:16, 6:16] = 0.0  # 10x10 columns of zeros
    out = bgc_groom(_ic(clean), grid, passes=1, column_reach=1, relax_sweeps=0)
    counts = summarize_flags(bgc_check(out, grid))
    assert counts["NO3"]["total"] > 0
    assert counts["NO3"]["floor_jump"] > 0


def _boundary(fields, side="west"):
    sl = {"west": {"xi_rho": 0}, "south": {"eta_rho": 0}}[side]
    along = "eta_rho" if side == "west" else "xi_rho"
    dims = ("bry_time", "s_rho", along)
    t = np.arange(3.0)

    def sec(v):
        a = xr.DataArray(v[0], dims=("s_rho", "eta_rho", "xi_rho")).isel(**sl).values
        return np.repeat(a[None], 3, axis=0).astype(np.float32)

    bgc = xr.Dataset(
        {f"{v}_{side}": (dims, sec(fields[v])) for v in TRACERS}, coords={"bry_time": t}
    ).chunk()
    phys = xr.Dataset(
        {f"{v}_{side}": (dims, sec(fields[v])) for v in ("salt", "temp")},
        coords={"bry_time": t},
    ).chunk()
    return bgc, phys


def test_boundary_section_is_groomed(grid, clean):
    clean["ALK"][0, 0:3, 9, 0] = 0.0
    clean["DIC"][0, 6, 12, 0] = 1.3 * clean["ALK"][0, 6, 12, 0]
    bgc, phys = _boundary(clean, "west")
    before = summarize_flags(bgc_check(bgc, grid, phys))
    assert before["ALK_west"]["total"] == 3 * 3 and before["DIC_west"]["total"] >= 3
    out = bgc_groom(bgc, grid, phys)
    after = summarize_flags(bgc_check(out, grid, phys))
    assert all(c["total"] == 0 for c in after.values()), after


def test_boundary_needs_physics_and_matching_time(grid, clean):
    bgc, phys = _boundary(clean, "south")
    with pytest.raises(ValueError, match="physics"):
        bgc_check(bgc, grid)
    with pytest.raises(ValueError, match="bry_time"):
        bgc_check(bgc, grid, phys.assign_coords(bry_time=phys.bry_time + 1))


def test_ic_without_salinity_raises(grid, clean):
    ds = _ic(clean).drop_vars("salt")
    with pytest.raises(ValueError, match="salt"):
        bgc_check(ds, grid)


def test_steep_smooth_gradient_is_not_a_spike(grid, clean):
    """Each cell differs from its neighbours by more than max_jump, but lies between them."""
    ramp = 1000.0 + 110.0 * np.arange(NX)[None, None, None, :]  # 110 per cell along x
    clean["DIC"] = np.where(clean["salt"] > 0, ramp, 0.0)
    flags = bgc_check(_ic(clean), grid, variables=["DIC"])["DIC"].values
    assert not (flags & FLAG_JUMP).any()


def test_bad_column_is_flagged_at_every_level_and_fixed(grid, clean):
    """A whole column off its neighbours: its vertical neighbours agree with it."""
    clean["DIC"][0, :, 10, 10] -= 160.0
    flags = bgc_check(_ic(clean), grid, variables=["DIC"])["DIC"].values
    assert (flags[0, :, 10, 10] & FLAG_JUMP).all()
    out = bgc_groom(_ic(clean), grid, variables=["DIC"])
    assert summarize_flags(bgc_check(out, grid, variables=["DIC"]))["DIC"]["total"] == 0


def test_dic_alone_uses_alk_as_input_without_grooming_it(grid, clean):
    clean["DIC"][0, 5, 10, 8] = 1.2 * clean["ALK"][0, 5, 10, 8]
    clean["ALK"][0, 0:3, 5, 5] = 0.0  # bad ALK must stay untouched: not a target
    flags = bgc_check(_ic(clean), grid, variables=["DIC"])
    assert list(flags.data_vars) == ["DIC"]
    assert int((flags["DIC"] > 0).sum()) >= 1
    out = bgc_groom(_ic(clean), grid, variables=["DIC"]).compute()
    np.testing.assert_array_equal(out["ALK"].values, clean["ALK"].astype(np.float32))
    # DIC is fixed everywhere except where its required input (ALK) is itself bad.
    left = bgc_check(out, grid, variables=["DIC"])["DIC"].values > 0
    assert set(map(tuple, np.argwhere(left))) == {(0, k, 5, 5) for k in range(3)}


def test_deep_water_alkalinity_at_in_situ_density_passes(grid, clean):
    clean["ALK"] = np.where(clean["salt"] > 0, 2560.0, 0.0)  # deep water, mmol/m3
    clean["salt"] = np.where(clean["salt"] > 0, 34.68, 0.0)
    clean["DIC"] = np.where(clean["salt"] > 0, 2450.0, 0.0)
    assert (
        summarize_flags(bgc_check(_ic(clean), grid, variables=["ALK"]))["ALK"]["total"]
        == 0
    )


def test_unknown_variable_and_bad_parameters_raise(grid, clean):
    with pytest.raises(ValueError, match="No BgcCheck"):
        bgc_check(_ic(clean), grid, variables=["FOO"])
    with pytest.raises(ValueError, match="relax_sweeps"):
        bgc_groom(_ic(clean), grid, relax_sweeps=-1)
