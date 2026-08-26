"""
Comment about tensorflow layers:
unfortunately instructions on creation of TF layers change constantly,
and changed way too many times at this point to remember what-compatible-where.

Layers in einops==0.7.0 (and several prior versions)
 are compatible with TF 2.13

Layers in einops==0.8.0 were re-implemented
 according to official instructions for TF 2.16

"""

from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import tensorflow as tf
from tensorflow.keras.layers import Layer

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin

__author__ = "Alex Rogozhnikov"

# matches the type parameters used by `tensorflow.keras.layers.Layer`
_InputT_contra = TypeVar("_InputT_contra", contravariant=True)
_OutputT_co = TypeVar("_OutputT_co", covariant=True)

# `tensorflow.keras.layers.Layer` is only generic in the stubs, not at runtime.
if TYPE_CHECKING:

    class _BaseLayer(Layer[_InputT_contra, _OutputT_co], Generic[_InputT_contra, _OutputT_co]): ...
else:

    class _BaseLayer(Layer, Generic[_InputT_contra, _OutputT_co]): ...


class Rearrange(RearrangeMixin, _BaseLayer[_InputT_contra, _OutputT_co]):
    def build(self, input_shape: Any) -> None:
        pass  # layer does not have any parameters to be initialized

    def call(self, inputs: _InputT_contra) -> _OutputT_co:
        return self._apply_recipe(inputs)

    def get_config(self) -> dict[str, Any]:
        return {"pattern": self.pattern, **self.axes_lengths}


class Reduce(ReduceMixin, _BaseLayer[_InputT_contra, _OutputT_co]):
    def build(self, input_shape: Any) -> None:
        pass  # layer does not have any parameters to be initialized

    def call(self, inputs: _InputT_contra) -> _OutputT_co:
        return self._apply_recipe(inputs)

    def get_config(self) -> dict[str, Any]:
        return {"pattern": self.pattern, "reduction": self.reduction, **self.axes_lengths}


class EinMix(_EinmixMixin, _BaseLayer[_InputT_contra, _OutputT_co]):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound) -> None:
        # this method is called in __init__,
        #  but we postpone actual creation to build(), as TF instruction suggests
        self._params = [weight_shape, weight_bound, bias_shape, bias_bound]

    def _create_rearrange_layers(
        self,
        pre_reshape_pattern: str | None,
        pre_reshape_lengths: dict | None,
        post_reshape_pattern: str | None,
        post_reshape_lengths: dict | None,
    ) -> None:
        self.pre_rearrange: Rearrange | None = None
        if pre_reshape_pattern is not None:
            self.pre_rearrange = Rearrange(pre_reshape_pattern, **cast(dict, pre_reshape_lengths))

        self.post_rearrange: Rearrange | None = None
        if post_reshape_pattern is not None:
            self.post_rearrange = Rearrange(post_reshape_pattern, **cast(dict, post_reshape_lengths))

    def build(self, input_shape: Any) -> None:
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

    def call(self, inputs: _InputT_contra) -> _OutputT_co:
        if self.pre_rearrange is not None:
            inputs = self.pre_rearrange(inputs)
        result = tf.einsum(self.einsum_pattern, inputs, self.weight)
        if self.bias is not None:
            result = result + self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "weight_shape": self.weight_shape,
            "bias_shape": self.bias_shape,
            **self.axes_lengths,
        }
