"""Keep numba's on-disk cache for roms_tools kernels out of the package tree.

By default numba writes ``cache=True`` artifacts (``.nbi``/``.nbc``) into the
``__pycache__`` next to the source file, i.e. inside site-packages. pip and
conda only remove files listed in the install record, so uninstalling a
non-editable roms_tools leaves ``site-packages/roms_tools/setup/__pycache__``
behind. A later editable install then breaks: the leftover ``roms_tools``
directory (no ``__init__.py``) is found by the path finder before the editable
finder and is imported as an empty namespace package, so
``from roms_tools import Grid`` fails with "(unknown location)".

The locator below sends the cache for roms_tools source files to
``<user cache dir>/roms-tools-numba`` instead, in a per-source-path
subdirectory so separate checkouts never share entries. It is deliberately a
sibling of the pooch data cache (``<user cache dir>/roms-tools``), not inside
it: CI persists the pooch directory across runs, and compiled kernels must not
ride along (globals such as ``R_EARTH`` are frozen into them).
Everything else is left to numba's defaults:

* ``NUMBA_CACHE_DIR`` set: numba's own user-provided locator wins.
* ``NUMBA_CACHE_LOCATOR_CLASSES`` set: numba ignores the registered locator
  list entirely, so this locator is skipped.
* Cache dir not writable: ``from_function`` returns ``None`` and numba falls
  back to the in-tree ``__pycache__``.
* Functions from any other package are never matched.

Registration patches ``CacheImpl._locator_classes``, which is private numba
API (the same hook PyTensor uses). It must happen before any ``cache=True``
function is decorated, so modules that define such functions import this one
first.
"""

from pathlib import Path

import pooch
from numba.core import config
from numba.core.caching import CacheImpl, UserWideCacheLocator

_PACKAGE_DIR = Path(__file__).resolve().parent
_CACHE_ROOT = Path(pooch.os_cache("roms-tools-numba"))


class RomsToolsCacheLocator(UserWideCacheLocator):
    """Cache locator for functions defined in roms_tools source files."""

    def __init__(self, py_func, py_file):
        super().__init__(py_func, py_file)
        self._cache_path = str(_CACHE_ROOT / self.get_suitable_cache_subpath(py_file))

    @classmethod
    def from_function(cls, py_func, py_file):
        """Return a locator for ``py_func``, or ``None`` to defer to numba."""
        if config.CACHE_DIR:
            return None
        if not Path(py_file).resolve().is_relative_to(_PACKAGE_DIR):
            return None
        return super().from_function(py_func, py_file)


# Compare by name so a module reload doesn't register a second copy.
if not any(c.__name__ == "RomsToolsCacheLocator" for c in CacheImpl._locator_classes):
    CacheImpl._locator_classes.insert(0, RomsToolsCacheLocator)
