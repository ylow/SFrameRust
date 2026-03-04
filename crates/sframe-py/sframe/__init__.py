"""SFrame — scalable, out-of-core columnar data store for Python.

Built on top of the Rust SFrame engine via PyO3.
"""

from _sframe import (
    SFrame,
    SArray,
    SFrameStreamWriter,
    load,
)
from _sframe import aggregate

__all__ = [
    "SFrame",
    "SArray",
    "SFrameStreamWriter",
    "load",
    "aggregate",
]
