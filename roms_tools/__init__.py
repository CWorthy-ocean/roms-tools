from importlib.metadata import version as _version


try:
    __version__ = _version("roms_tools")
except ImportError:  # pragma: no cover
    # Local copy or not installed with setuptools
    __version__ = "999"


from roms_tools.setup.grid import Grid  # noqa: F401
from roms_tools.setup.tides import TidalForcing  # noqa: F401
from roms_tools.setup.atmospheric_forcing import AtmosphericForcing, SWRCorrection  # noqa: F401
