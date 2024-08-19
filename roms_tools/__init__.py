from importlib.metadata import version as _version


try:
    __version__ = _version("roms_tools")
except ImportError:  # pragma: no cover
    # Local copy or not installed with setuptools
    __version__ = "999"


from roms_tools.setup.grid import Grid  # noqa: F401
from roms_tools.setup.tides import TidalForcing  # noqa: F401
from roms_tools.setup.surface_forcing import SurfaceForcing  # noqa: F401
from roms_tools.setup.initial_conditions import InitialConditions  # noqa: F401
from roms_tools.setup.boundary_forcing import BoundaryForcing  # noqa: F401
