"""
Comment about tensorflow layers:
unfortunately instructions on creation of TF layers change constantly,
and changed way too many times at this point to remember what-compatible-where.

Layers in einops==0.7.0 (and several prior versions)
 are compatible with TF 2.13

Layers in einops==0.8.0 were re-implemented
 according to official instructions for TF 2.16

"""

from typing import Optional, Dict, cast

import tensorflow as tf
from tensorflow.keras.layers import Layer


from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin


__author__ = "Alex Rogozhnikov"


class Rearrange(RearrangeMixin, Layer):
    def build(self, input_shape):
        pass  # layer does not have any parameters to be initialized

    def call(self, inputs):
        return self._apply_recipe(inputs)

    def get_config(self):
        return {"pattern": self.pattern, **self.axes_lengths}


class Reduce(ReduceMixin, Layer):
    def build(self, input_shape):
        pass  # layer does not have any parameters to be initialized

    def call(self, inputs):
        return self._apply_recipe(inputs)

    def get_config(self):
        return {"pattern": self.pattern, "reduction": self.reduction, **self.axes_lengths}


class EinMix(_EinmixMixin, Layer):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        # this method is called in __init__,
        #  but we postpone actual creation to build(), as TF instruction suggests
        self._params = [weight_shape, weight_bound, bias_shape, bias_bound]

    def _create_rearrange_layers(
        self,
        pre_reshape_pattern: Optional[str],
        pre_reshape_lengths: Optional[Dict],
        post_reshape_pattern: Optional[str],
        post_reshape_lengths: Optional[Dict],
    ):
        self.pre_rearrange = None
        if pre_reshape_pattern is not None:
            self.pre_rearrange = Rearrange(pre_reshape_pattern, **cast(dict, pre_reshape_lengths))

        self.post_rearrange = None
        if post_reshape_pattern is not None:
            self.post_rearrange = Rearrange(post_reshape_pattern, **cast(dict, post_reshape_lengths))

    def build(self, input_shape):
        [weight_shape, weight_bound, bias_shape, bias_bound] = self._params
        self.weight = self.add_weight(
            shape=weight_shape,
            initializer=tf.random_uniform_initializer(-weight_bound, weight_bound),
            trainable=True,
        )

        if bias_shape is not None:
            self.bias = self.add_weight(
                shape=bias_shape,
                initializer=tf.random_uniform_initializer(-bias_bound, bias_bound),
                trainable=True,
            )
        else:
            self.bias = None

    def call(self, inputs):
        if self.pre_rearrange is not None:
            inputs = self.pre_rearrange(inputs)
        result = tf.einsum(self.einsum_pattern, inputs, self.weight)
        if self.bias is not None:
            result = result + self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result

    def get_config(self):
        return {
            "pattern": self.pattern,
            "weight_shape": self.weight_shape,
            "bias_shape": self.bias_shape,
            **self.axes_lengths,
        }
