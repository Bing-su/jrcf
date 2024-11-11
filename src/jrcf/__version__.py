from importlib.metadata import version

try:
    __version__ = version("jrcf")
except Exception:  # pragma: no cover
    __version__ = "unknown"
