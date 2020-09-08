import mxnet

from . import RearrangeMixin, ReduceMixin
from ._weighted_einsum import WeightedEinsumMixin

__author__ = 'Alex Rogozhnikov'


class Rearrange(RearrangeMixin, mxnet.gluon.HybridBlock):
    def hybrid_forward(self, F, x):
        return self._apply_recipe(x)


class Reduce(ReduceMixin, mxnet.gluon.HybridBlock):
    def hybrid_forward(self, F, x):
        return self._apply_recipe(x)


class WeightedEinsum(WeightedEinsumMixin, mxnet.gluon.HybridBlock):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        with self.name_scope():

            self.weight = self.params.get(name='weight', shape=weight_shape,
                                          init=mxnet.initializer.Uniform(weight_bound),
                                          )
            if bias_shape is not None:
                self.bias = self.params.get(name='bias', shape=bias_shape,
                                            init=mxnet.initializer.Uniform(bias_bound),
                                            )
            else:
                self.bias = None

    def hybrid_forward(self, F, x, *args, **kwargs):
        result = mxnet.np.einsum(self.einsum_pattern, x, self.weight.data())
        if self.bias is not None:
            result += self.bias.data()
        return result
