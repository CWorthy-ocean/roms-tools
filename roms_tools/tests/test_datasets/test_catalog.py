"""Catalog-driven source datasets against the per-source subclasses they replace.

Every equivalence test builds the same source twice, once through the existing class
and once from the catalog entry, and requires the same processed data and the same
name maps. Nothing here needs network access beyond the pooch test files the rest of
the suite already downloads, except the tests marked ``use_copernicus``.
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("intake")
pytest.importorskip("cf_xarray")

from roms_tools import Grid, InitialConditions
from roms_tools.datasets import catalog
from roms_tools.datasets.contexts import CONTEXTS, qualifies
from roms_tools.datasets.download import download_test_data, download_topo
from roms_tools.datasets.lat_lon_datasets import (
    ETOPO5Dataset,
    GLORYSDataset,
    LatLonDataset,
    SRTM15Dataset,
)
from roms_tools.datasets.resolve import (
    RoleResolutionError,
    load_vocabulary,
    resolve_roles,
)
from roms_tools.datasets.roms_dataset import ROMSDataset
from roms_tools.datasets.transforms import Names, load_transform
from roms_tools.setup.bgc_model import BGCMarbl
from roms_tools.setup.initial_conditions import _set_required_vars
from roms_tools.setup.topography import _get_topography_data

PACKAGED = catalog.search_paths()[0] / "roms_tools.yaml"
OCEAN_SKILL_CATALOGS = Path("/Users/kthyng/packages/ocean-skill/ocean_skill/catalogs")


@pytest.fixture(scope="module")
def srtm15_file() -> str:
    return str(download_test_data("srtm15_coarsened.nc"))


@pytest.fixture(scope="module")
def glorys_file() -> str:
    return str(download_test_data("GLORYS_coarse_test_data.nc"))


@pytest.fixture(scope="module")
def restart_files() -> list[str]:
    return [
        str(download_test_data("eastpac25km_rst.19980106000000.nc")),
        str(download_test_data("eastpac25km_rst.19980126000000.nc")),
    ]


@pytest.fixture(scope="module")
def parent_grid() -> Grid:
    return Grid(
        center_lon=-120, center_lat=30, nx=8, ny=13, size_x=3000, size_y=4000, rot=32
    )


def assert_equivalent(a, b) -> None:
    """Same processed data and the same name maps; chunking and encoding may differ."""
    xr.testing.assert_identical(a.ds.compute(), b.ds.compute())
    for attr in ("var_names", "dim_names", "opt_var_names"):
        assert getattr(a, attr) == getattr(b, attr), attr
    for attr in ("needs_lateral_fill", "is_global", "resolution"):
        if hasattr(a, attr):
            assert getattr(a, attr) == getattr(b, attr), attr


# --- catalog integrity ------------------------------------------------------------


def test_packaged_catalog_loads_and_entries_are_well_formed():
    import intake

    cat = intake.from_yaml_file(str(PACKAGED))
    assert {"SRTM15", "ETOPO5", "GLORYS", "GLORYS_default", "ROMS"} <= set(cat.aliases)
    for key, desc in cat.entries.items():
        meta = desc.metadata
        assert "axes" in meta, key
        for stage, target in meta.get("transforms", {}).items():
            assert stage in {"clean_up", "post_process"}, (key, stage)
            assert callable(load_transform(target)), (key, target)
        for context in meta.get("contexts", []):
            assert context in CONTEXTS, (key, context)


def test_template_entries_take_a_path_and_others_do_not():
    templated = catalog._template_entries(catalog._open(PACKAGED))
    assert templated == {"SRTM15", "GLORYS", "ROMS"}


def test_contexts_follow_from_the_entry_not_from_a_list():
    assert catalog.has("SRTM15", "topography")
    assert catalog.has("ETOPO5", "topography")
    assert catalog.has("GLORYS", "ic_physics") and catalog.has("GLORYS", "bc_physics")
    # Eligibility is judged from metadata alone, so GLORYS is not excluded from
    # topography here; the missing role is reported when the data is opened.
    assert catalog.has("GLORYS", "topography")
    assert catalog.has("ROMS", "ic_physics") and catalog.has("ROMS", "ic_bgc")
    assert not catalog.has("ROMS", "bc_physics")  # boundary data from ROMS is nesting
    assert catalog.has("GLORYS", "ic_physics", "default")


def test_an_entry_can_restrict_but_never_widen_contexts():
    meta = {"axes": {"X": "lon", "Y": "lat", "Z": "depth"}}
    assert qualifies(meta, "ic_physics") and qualifies(meta, "bc_physics")
    assert not qualifies({**meta, "contexts": ["ic_physics"]}, "bc_physics")
    assert not qualifies({**meta, "contexts": ["topography"]}, "ic_physics")
    # a dataset without a vertical axis never qualifies for a 3D context
    assert not qualifies({"axes": {"X": "lon", "Y": "lat"}}, "ic_physics")


def test_a_source_that_lacks_a_required_role_fails_clearly_when_opened(glorys_file):
    with pytest.raises(RoleResolutionError, match="topo"):
        catalog.from_catalog(
            "GLORYS", {"name": "GLORYS", "path": glorys_file}, "topography"
        )


def test_unknown_source_suggests_the_close_name():
    with pytest.raises(KeyError, match="SRTM15"):
        catalog.resolve("SRTM5", "topography")


# --- role resolution --------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "roles", "expected"),
    [
        ("srtm15_coarsened.nc", ["topo"], {"topo": "z"}),
        ("etopo5.nc", ["topo"], {"topo": "topo"}),
        (
            "GLORYS_coarse_test_data.nc",
            ["temp", "salt", "u", "v", "zeta"],
            {"temp": "thetao", "salt": "so", "u": "uo", "v": "vo", "zeta": "zos"},
        ),
        (
            "eastpac25km_rst.19980106000000.nc",
            ["temp", "salt", "u", "v", "zeta"],
            {"temp": "temp", "salt": "salt", "u": "u", "v": "v", "zeta": "zeta"},
        ),
    ],
)
def test_roles_resolve_on_real_files_without_any_cf_attributes(
    fixture, roles, expected
):
    path = (
        download_topo(fixture)
        if fixture == "etopo5.nc"
        else download_test_data(fixture)
    )
    ds = xr.open_dataset(path, decode_times=False)
    assert resolve_roles(ds, roles) == expected


def test_a_role_matching_two_variables_is_refused():
    ds = xr.Dataset({"topo": ("x", np.zeros(2)), "elevation": ("x", np.zeros(2))})
    with pytest.raises(RoleResolutionError, match=r"elevation.*topo|topo.*elevation"):
        resolve_roles(ds, ["topo"])


def test_a_missing_required_role_is_an_error_and_a_missing_optional_one_is_not():
    ds = xr.Dataset({"q": ("x", np.zeros(2))})
    with pytest.raises(RoleResolutionError, match="topo"):
        resolve_roles(ds, ["topo"])
    assert resolve_roles(ds, ["topo"], required=False) == {}


def test_vocabulary_patterns_are_anchored():
    """cf-xarray matches from the start of a name, so an unanchored ``z`` matches ``zeta``."""
    for role, criteria in load_vocabulary().items():
        for attr, pattern in criteria.items():
            assert pattern.startswith("^") and pattern.endswith("$"), (role, attr)


# --- the engine seams -------------------------------------------------------------


def test_transforms_may_change_the_name_maps_by_return_value():
    ds = xr.Dataset({"a": ("x", np.arange(3.0)), "b": ("x", np.arange(3.0))})

    def drop_b_add_c(ds, names: Names):
        ds = ds.drop_vars("b").assign(c=ds["a"] * 2)
        var_names = {k: v for k, v in names.var_names.items() if v != "b"}
        return ds, Names({**var_names, "c": "c"}, names.dim_names, names.opt_var_names)

    data = LatLonDataset.from_dataset(
        ds,
        var_names={"a": "a", "b": "b"},
        dim_names={"longitude": "x", "latitude": "x"},
        needs_lateral_fill=False,
        post_process_fn=drop_b_add_c,
    )
    assert data.var_names == {"a": "a", "c": "c"}
    assert "c" in data.ds and "b" not in data.ds


# --- equivalence with the classes --------------------------------------------------


@pytest.mark.parametrize("use_dask", [False, True])
def test_srtm15(srtm15_file, use_dask):
    a = SRTM15Dataset(filename=srtm15_file, use_dask=use_dask)
    b = catalog.from_catalog(
        "SRTM15",
        {"name": "SRTM15", "path": srtm15_file},
        "topography",
        use_dask=use_dask,
    )
    assert type(b) is LatLonDataset
    assert_equivalent(a, b)


@pytest.mark.parametrize("use_dask", [False, True])
def test_etopo5_downloaded_through_the_reader_and_fixed_up_by_a_transform(use_dask):
    a = ETOPO5Dataset(use_dask=use_dask)
    b = catalog.from_catalog(
        "ETOPO5", {"name": "ETOPO5"}, "topography", use_dask=use_dask
    )
    assert b.clean_up_fn is not None
    assert_equivalent(a, b)


@pytest.mark.parametrize("use_dask", [False, True])
def test_glorys_files(glorys_file, use_dask):
    start = datetime(2021, 6, 29)
    a = GLORYSDataset(filename=glorys_file, start_time=start, use_dask=use_dask)
    b = catalog.from_catalog(
        "GLORYS",
        {"name": "GLORYS", "path": glorys_file},
        "ic_physics",
        start_time=start,
        use_dask=use_dask,
    )
    assert {"mask", "mask_vel"} <= set(b.ds.data_vars)
    assert_equivalent(a, b)


@pytest.mark.parametrize("use_dask", [False, True])
@pytest.mark.parametrize("kind", ["physics", "bgc"])
def test_roms_restart_one_file(restart_files, parent_grid, kind, use_dask):
    kwargs = {
        "var_names": _set_required_vars(kind),
        "start_time": datetime(1998, 1, 6),
        "use_dask": use_dask,
        "adjust_depth_for_sea_surface_height": True,
    }
    a = ROMSDataset(path=restart_files[0], grid=parent_grid, **kwargs)
    b = catalog.from_catalog(
        "ROMS",
        {"name": "ROMS", "path": restart_files[0], "grid": parent_grid},
        f"ic_{kind}",
        **kwargs,
    )
    assert_equivalent(a, b)
    assert a.model_reference_date == b.model_reference_date


def test_roms_restart_two_files(restart_files, parent_grid):
    kwargs = {
        "var_names": _set_required_vars("physics"),
        "use_dask": True,
        "adjust_depth_for_sea_surface_height": True,
    }
    a = ROMSDataset(path=restart_files, grid=parent_grid, **kwargs)
    b = catalog.from_catalog(
        "ROMS",
        {"name": "ROMS", "path": restart_files, "grid": parent_grid},
        "ic_physics",
        **kwargs,
    )
    assert b.ds.sizes["time"] == 4
    assert_equivalent(a, b)


# --- through the forcing classes --------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [{"name": "ETOPO5"}, "srtm15"],
    ids=["etopo5-default", "srtm15-path"],
)
def test_grid_topography_is_identical_through_the_catalog(source, srtm15_file):
    if source == "srtm15":
        source = {"name": "SRTM15", "path": srtm15_file}
    kwargs = {
        "nx": 5,
        "ny": 5,
        "size_x": 1000,
        "size_y": 1000,
        "center_lon": 0,
        "center_lat": 0,
        "rot": 20,
        "topography_source": source,
    }
    plain = Grid(**kwargs)
    with catalog.use_catalog():
        from_catalog = Grid(**kwargs)
    xr.testing.assert_identical(plain.ds, from_catalog.ds)


def _initial_conditions(source, **extra):
    grid = Grid(
        nx=2,
        ny=2,
        size_x=500,
        size_y=1000,
        center_lon=0,
        center_lat=55,
        rot=10,
        N=3,
        theta_s=5.0,
        theta_b=2.0,
        hc=250.0,
    )
    return InitialConditions(
        grid=grid,
        ini_time=datetime(2021, 6, 29),
        source=source,
        prefill="2d_lateral_fill",
        regrid_method="scipy",
        **extra,
    )


def test_initial_conditions_from_glorys_files_are_identical_through_the_catalog(
    glorys_file,
):
    source = {"path": glorys_file, "name": "GLORYS"}
    plain = _initial_conditions(source)
    with catalog.use_catalog():
        from_catalog = _initial_conditions(source)
    xr.testing.assert_identical(plain.ds, from_catalog.ds)


def test_initial_conditions_from_a_roms_restart_are_identical_through_the_catalog(
    restart_files, parent_grid
):
    def build():
        grid = Grid(
            nx=5, ny=5, center_lon=-120, center_lat=34, size_x=100, size_y=100, N=3
        )
        return InitialConditions(
            grid=grid,
            ini_time=datetime(1998, 1, 6),
            source={"name": "ROMS", "path": restart_files[0], "grid": parent_grid},
            bgc_source={"name": "ROMS", "path": restart_files[0], "grid": parent_grid},
            bgc_model=BGCMarbl,
            model_reference_date=datetime(2000, 1, 1),
        )

    plain = build()
    with catalog.use_catalog():
        from_catalog = build()
    xr.testing.assert_identical(plain.ds, from_catalog.ds)


# --- switching, shadowing, and ocean-skill compatibility ----------------------------


def test_the_catalog_path_is_off_by_default_and_scoped():
    assert not catalog.enabled()
    source = {"name": "SRTM15", "path": download_test_data("srtm15_coarsened.nc")}
    assert isinstance(_get_topography_data(source), SRTM15Dataset)
    with catalog.use_catalog():
        assert catalog.enabled()
        assert type(_get_topography_data(source)) is LatLonDataset
        with catalog.use_catalog(False):
            assert isinstance(_get_topography_data(source), SRTM15Dataset)
    assert not catalog.enabled()


def test_the_environment_variable_switches_the_catalog_on(monkeypatch):
    monkeypatch.setenv(catalog.ENABLE_ENV, "1")
    assert catalog.enabled()
    with catalog.use_catalog(False):
        assert not catalog.enabled()


def test_a_site_catalog_shadows_a_packaged_entry(tmp_path, monkeypatch, srtm15_file):
    (tmp_path / "site.yaml").write_text(
        PACKAGED.read_text().replace(
            "      featureType: grid\n      land_masked: false\n    user_parameters: {}\n  ETOPO5:",
            "      featureType: grid\n      land_masked: true\n    user_parameters: {}\n  ETOPO5:",
        )
    )
    source = {"name": "SRTM15", "path": srtm15_file}
    packaged = catalog.from_catalog("SRTM15", source, "topography")
    assert packaged.needs_lateral_fill is False
    monkeypatch.setenv(catalog.CATALOGS_ENV, str(tmp_path))
    assert catalog.resolve("SRTM15", "topography").path == tmp_path / "site.yaml"
    assert (
        catalog.from_catalog("SRTM15", source, "topography").needs_lateral_fill is True
    )


def test_a_roms_entry_can_carry_its_own_grid_and_reference_date(
    tmp_path, monkeypatch, restart_files, parent_grid
):
    grid_files = parent_grid.save(tmp_path / "parent_grid.nc")
    text = PACKAGED.read_text().replace(
        "      self_contained_grid: false\n",
        f"      self_contained_grid: false\n      grid: {grid_files[0]}\n      reference_date: '1995-01-01'\n",
    )
    (tmp_path / "site.yaml").write_text(text)
    monkeypatch.setenv(catalog.CATALOGS_ENV, str(tmp_path))
    data = catalog.from_catalog(
        "ROMS",
        {"name": "ROMS", "path": restart_files[0]},  # no grid object supplied
        "ic_physics",
        var_names=_set_required_vars("physics"),
        adjust_depth_for_sea_surface_height=True,
    )
    assert data.model_reference_date == datetime(1995, 1, 1)
    assert data.ds.sizes["eta_rho"] == parent_grid.ds.sizes["eta_rho"]


def test_a_roms_entry_without_a_grid_says_what_is_missing(restart_files):
    with pytest.raises(ValueError, match="needs a grid"):
        catalog.from_catalog(
            "ROMS",
            {"name": "ROMS", "path": restart_files[0]},
            "ic_physics",
            var_names=_set_required_vars("physics"),
        )


OCEAN_SKILL_ENTRY = textwrap.dedent(
    """\
    version: 2
    metadata: {title: written by ocean-skill}
    user_parameters: {}
    aliases:
      glorys_my_daily_geo: glorys_my_daily_geo
    data: {}
    entries:
      glorys_my_daily_geo:
        kwargs:
          dataset_id: cmems_mod_glo_phy_my_0.083deg_P1D-m
          service: arco-geo-series
        metadata:
          axes: {T: time, X: longitude, Y: latitude, Z: depth}
          featureType: grid
          standard_names: {thetao: sea_water_potential_temperature, so: sea_water_salinity}
          variables: [sea_water_potential_temperature, sea_water_salinity]
        output_instance: xarray:Dataset
        reader: ocean_skill.readers:CopernicusMarineReader
        user_parameters: {}
    """
)


def test_an_entry_written_by_ocean_skill_is_usable_as_it_stands(tmp_path, monkeypatch):
    """The point of one contract: nothing is appended to an ocean-skill entry."""
    (tmp_path / "copernicus.yaml").write_text(OCEAN_SKILL_ENTRY)
    monkeypatch.setenv(catalog.SHARED_CATALOGS_ENV, str(tmp_path))
    ref = catalog.resolve("glorys_my_daily_geo", "ic_physics")
    assert ref.path == tmp_path / "copernicus.yaml"
    assert ref.meta["axes"]["Z"] == "depth"
    if catalog.find_spec("ocean_skill") is None:
        reader = catalog._open(ref.path).entries["glorys_my_daily_geo"].reader
        assert reader == "roms_tools.datasets.readers:CopernicusMarineReader"


@pytest.mark.skipif(not OCEAN_SKILL_CATALOGS.is_dir(), reason="no ocean-skill checkout")
def test_every_shipped_ocean_skill_catalog_can_be_indexed(monkeypatch):
    monkeypatch.setenv(catalog.SHARED_CATALOGS_ENV, str(OCEAN_SKILL_CATALOGS))
    index = catalog.discover()
    assert len(index) > 100  # woa, modis, whots, copernicus, ...
    assert "glorys_my_daily_geo" in index


# --- streaming (opt-in, needs credentials) ------------------------------------------


@pytest.mark.stream
@pytest.mark.use_copernicus
@pytest.mark.use_dask
def test_glorys_streamed_from_copernicus_marine():
    start = datetime(2012, 1, 1)
    data = catalog.from_catalog(
        "GLORYS",
        {"name": "GLORYS"},
        "ic_physics",
        variant="default",
        start_time=start,
        use_dask=True,
    )
    assert set(data.var_names) == {"temp", "salt", "u", "v", "zeta"}
    assert "time" in data.ds.dims
    assert {"thetao", "so", "uo", "vo", "zos"} <= set(data.ds.data_vars)
