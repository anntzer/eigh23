import importlib.metadata

try:
    __version__ = importlib.metadata.version("eigh23")
except ImportError:
    __version__ = "0+unknown"

from ._eigh23 import eigh22, eigh33

__all__ = ["eigh22", "eigh33"]
