# from tripod import plot
from tripod.simulation import Simulation
from tripod import constants
from tripod import utils
from tripod.utils import hdf5writer

from simframe.io.dump import readdump
from simframe.io.writers import hdf5writer

__name__ = "TriPoD"
__version__ = "0.0.1"

Simulation.__version__ = __version__
utils.__version__ = __version__

__all__ = [
    "constants",
    "hdf5writer",
    "readdump",
    "Simulation"
]
