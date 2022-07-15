__author__ = 'Alex Rogozhnikov'
__version__ = '0.4.1'


class EinopsError(RuntimeError):
    """ Runtime error thrown by einops """
    pass



__all__ = ['einop', 'rearrange', 'reduce', 'repeat', 'parse_shape', 'asnumpy', 'EinopsError']

from .einops import einop, rearrange, reduce, repeat, parse_shape, asnumpy
