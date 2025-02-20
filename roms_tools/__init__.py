from importlib.metadata import version as _version
import logging  # noqa: F811

try:
    __version__ = _version("roms_tools")
except ImportError:  # pragma: no cover
    # Local copy or not installed with setuptools
    __version__ = "9999"


from roms_tools.setup.grid import Grid  # noqa: F401
from roms_tools.setup.tides import TidalForcing  # noqa: F401
from roms_tools.setup.surface_forcing import SurfaceForcing  # noqa: F401
from roms_tools.setup.initial_conditions import InitialConditions  # noqa: F401
from roms_tools.setup.boundary_forcing import BoundaryForcing  # noqa: F401
from roms_tools.setup.river_forcing import RiverForcing  # noqa: F401
from roms_tools.setup.nesting import ChildGrid  # noqa: F401
from roms_tools.tiling.partition import partition_netcdf  # noqa: F401
from roms_tools.analysis.roms_output import ROMSOutput  # noqa: F401

# Configure logging when the package is imported
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
