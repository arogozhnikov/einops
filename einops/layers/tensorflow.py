import tensorflow as tf
from tensorflow.keras.layers import Layer

from .._backends import UnknownSize
from . import RearrangeMixin, ReduceMixin
from ._weighted_einsum import WeightedEinsumMixin

__author__ = 'Alex Rogozhnikov'


class Rearrange(RearrangeMixin, Layer):
    def compute_output_shape(self, input_shape):
        input_shape = tuple(UnknownSize() if d.value is None else int(d) for d in input_shape)
        init_shapes, reduced_axes, axes_reordering, final_shape = self.recipe().reconstruct_from_shape(input_shape)
        final_shape = tuple(None if isinstance(d, UnknownSize) else int(d) for d in final_shape)
        return final_shape

    def call(self, inputs):
        return self._apply_recipe(inputs)

    def get_config(self):
        return {'pattern': self.pattern, **self.axes_lengths}


class Reduce(ReduceMixin, Layer):
    def compute_output_shape(self, input_shape):
        input_shape = tuple(UnknownSize() if d.value is None else int(d) for d in input_shape)
        init_shapes, reduced_axes, axes_reordering, final_shape = self.recipe().reconstruct_from_shape(input_shape)
        final_shape = tuple(None if isinstance(d, UnknownSize) else int(d) for d in final_shape)
        return final_shape

    def call(self, inputs):
        return self._apply_recipe(inputs)

    def get_config(self):
        return {'pattern': self.pattern, 'reduction': self.reduction, **self.axes_lengths}


class WeightedEinsum(WeightedEinsumMixin, Layer):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = tf.Variable(tf.random_uniform_initializer(-weight_bound, weight_bound)(shape=weight_shape),
                                  trainable=True)
        if bias_shape is not None:
            self.bias = tf.Variable(tf.random_uniform_initializer(-bias_bound, bias_bound)(shape=bias_shape),
                                    trainable=True)
        else:
            self.bias = None

    def build(self, input_shape):
        pass

    def call(self, inputs):
        result = tf.einsum(self.einsum_pattern, inputs, self.weight)
        if self.bias is not None:
            result = result + self.bias
        return result

    def get_config(self):
        return {'pattern': self.pattern,
                'weight_shape': self.weight_shape,
                'bias_shape': self.bias_shape,
                **self.axes_lengths}
