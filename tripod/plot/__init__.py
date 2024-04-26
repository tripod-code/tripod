from tripod.plot.plot import ipanel
from tripod.plot.plot import panel

from importlib import metadata as _md

__all__ = [
    "ipanel",
    "panel"
]
__version__ = _md.version("tripod")
