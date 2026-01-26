# imports can use EinopsError class
# ruff: noqa: E402

__author__ = "Alex Rogozhnikov"
__version__ = "0.9.0dev"


class EinopsError(RuntimeError):
    """Runtime error thrown by einops"""

    pass  # noqa: PIE790


__all__ = ["EinopsError", "asnumpy", "einsum", "pack", "parse_shape", "rearrange", "reduce", "repeat", "unpack"]

from .einops import asnumpy, einsum, parse_shape, rearrange, reduce, repeat
from .packing import pack, unpack
