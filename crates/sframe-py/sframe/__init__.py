"""SFrame — scalable, out-of-core columnar data store for Python.

Built on top of the Rust SFrame engine via PyO3.
"""

from _sframe import (
    SFrame,
    SArray,
    Sketch,
    SFrameStreamWriter,
    config,
    load,
)
from _sframe import aggregate

# C++ API compat alias
load_sframe = load

__all__ = [
    "SFrame",
    "SArray",
    "Sketch",
    "SFrameStreamWriter",
    "config",
    "load",
    "load_sframe",
    "aggregate",
]
