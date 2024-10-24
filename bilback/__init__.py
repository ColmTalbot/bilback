from importlib.metadata import entry_points

from ._version import __version__
from . import geometry, time

backends = entry_points(group="bilbackends")

BACKENDS = dict()

for backend in backends:
    try:
        backend.load()
        BACKENDS[backend.name] = backend
    except ImportError:
        pass


__all__ = ["geometry", "time", "__version__"]