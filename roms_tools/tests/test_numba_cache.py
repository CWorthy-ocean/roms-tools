import importlib
import importlib.util
import inspect
import textwrap
from pathlib import Path

import numba as nb
import pytest
from numba.core import config
from numba.core.caching import CacheImpl

import roms_tools
from roms_tools import _numba_cache
from roms_tools._numba_cache import RomsToolsCacheLocator
from roms_tools.setup.utils import min_dist_to_land

PACKAGE_DIR = Path(roms_tools.__file__).resolve().parent


def test_package_kernels_cache_outside_package():
    """The import-time kernels were decorated after the locator was registered."""
    if config.CACHE_DIR:
        pytest.skip("NUMBA_CACHE_DIR is set; numba's own locator takes precedence")
    cache_path = Path(min_dist_to_land._cache.cache_path).resolve()
    assert cache_path.is_relative_to(_numba_cache._CACHE_ROOT.resolve())
    assert not cache_path.is_relative_to(PACKAGE_DIR)


def test_cache_files_written_under_cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", "")
    monkeypatch.setattr(_numba_cache, "_CACHE_ROOT", tmp_path)
    kernel = nb.njit(
        [nb.float64[:, :](nb.float64[:, :], nb.float64[:, :], nb.int32[:, :])],
        cache=True,
    )(min_dist_to_land.py_func)

    cache_path = Path(kernel._cache.cache_path)
    assert cache_path.is_relative_to(tmp_path)
    # Eager compilation saves the index/data files immediately.
    assert list(cache_path.glob("*.nbi"))
    assert list(cache_path.glob("*.nbc"))


def test_defers_to_numba_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    py_func = min_dist_to_land.py_func
    assert (
        RomsToolsCacheLocator.from_function(py_func, inspect.getfile(py_func)) is None
    )


def test_ignores_functions_from_other_packages(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", "")
    module_file = tmp_path / "other_pkg_module.py"
    module_file.write_text(
        textwrap.dedent(
            """
            import numba as nb

            @nb.njit("float64(float64)", cache=True)
            def double(x):
                return 2 * x
            """
        )
    )
    spec = importlib.util.spec_from_file_location("other_pkg_module", module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert Path(module.double._cache.cache_path) == tmp_path / "__pycache__"


def test_registered_once():
    importlib.reload(_numba_cache)
    names = [c.__name__ for c in CacheImpl._locator_classes]
    assert names.count("RomsToolsCacheLocator") == 1
    assert names[0] == "RomsToolsCacheLocator"
