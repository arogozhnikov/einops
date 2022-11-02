__author__ = 'Alex Rogozhnikov'
__version__ = '0.5.0'


class EinopsError(RuntimeError):
    """ Runtime error thrown by einops """
    pass


__all__ = ['rearrange', 'reduce', 'repeat', 'einsum',
           'parse_shape', 'asnumpy', 'EinopsError']

from .einops import rearrange, reduce, repeat, einsum, parse_shape, asnumpy
