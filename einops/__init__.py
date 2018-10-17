__author__ = 'Alex Rogozhnikov'
__version__ = '0.1'

from .einops import rearrange, reduce, parse_shape, asnumpy, EinopsError
from .backends import get_backend
