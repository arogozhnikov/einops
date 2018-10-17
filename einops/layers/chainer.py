import chainer

from . import RearrangeMixin, ReduceMixin

__author__ = 'Alex Rogozhnikov'


class Rearrange(RearrangeMixin, chainer.Link):
    def __call__(self, x):
        return self._apply_recipe(x)


class Reduce(ReduceMixin, chainer.Link):
    def __call__(self, x):
        return self._apply_recipe(x)