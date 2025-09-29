import logging
from importlib.metadata import version as _version

try:
    __version__ = _version("roms_tools")
except ImportError:  # pragma: no cover
    # Local copy or not installed with setuptools
    __version__ = "9999"

# grid must be imported first
from roms_tools.setup.grid import Grid  # noqa: I001, F401
from roms_tools.analysis.roms_output import ROMSOutput  # noqa: F401
from roms_tools.setup.boundary_forcing import BoundaryForcing  # noqa: F401
from roms_tools.setup.cdr_forcing import CDRForcing  # noqa: F401
from roms_tools.setup.cdr_release import TracerPerturbation, VolumeRelease  # noqa: F401
from roms_tools.setup.initial_conditions import InitialConditions  # noqa: F401
from roms_tools.setup.nesting import ChildGrid  # noqa: F401
from roms_tools.setup.river_forcing import RiverForcing  # noqa: F401
from roms_tools.setup.surface_forcing import SurfaceForcing  # noqa: F401
from roms_tools.setup.tides import TidalForcing  # noqa: F401
from roms_tools.tiling.partition import partition_netcdf  # noqa: F401
from roms_tools.setup.datasets import get_glorys_bounds  # noqa: F401
from roms_tools.tiling.join import open_partitions, join_netcdf  # noqa: F401


# Configure logging when the package is imported
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
