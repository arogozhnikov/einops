import mxnet

from . import RearrangeMixin, ReduceMixin

__author__ = 'Alex Rogozhnikov'


# TODO symbolic is not working right now
class Rearrange(RearrangeMixin, mxnet.gluon.HybridBlock):
    def hybrid_forward(self, F, x):
        return self._apply_recipe(x)


class Reduce(ReduceMixin, mxnet.gluon.HybridBlock):
    def hybrid_forward(self, F, x):
        return self._apply_recipe(x)
