# imports can use EinopsError class
# ruff: noqa: E402

__author__ = "Alex Rogozhnikov"
__version__ = "0.8.0"


class EinopsError(RuntimeError):
    """Runtime error thrown by einops"""

    pass


__all__ = ["rearrange", "reduce", "repeat", "einsum", "pack", "unpack", "parse_shape", "asnumpy", "EinopsError"]

from .einops import rearrange, reduce, repeat, einsum, parse_shape, asnumpy
from .packing import pack, unpack
