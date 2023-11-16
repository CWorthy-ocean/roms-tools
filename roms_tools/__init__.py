from importlib.metadata import version as _version

from roms_tools import setup


try:
    __version__ = _version("roms_tools")
except ImportError:  # pragma: no cover
    # Local copy or not installed with setuptools
    __version__ = "999"
