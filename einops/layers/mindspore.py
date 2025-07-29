from typing import Dict, Optional, cast

from mindspore import Parameter, mint, nn
from mindspore.common.initializer import Uniform, initializer

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin

__author__ = "Rustam Khadipash"


class Rearrange(RearrangeMixin, nn.Cell):
    def construct(self, input):
        return self._apply_recipe(input)


class Reduce(ReduceMixin, nn.Cell):
    def construct(self, input):
        return self._apply_recipe(input)


class EinMix(_EinmixMixin, nn.Cell):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = Parameter(initializer(Uniform(weight_bound), weight_shape))
        self.bias = None
        if bias_shape is not None:
            self.bias = Parameter(initializer(Uniform(bias_bound), bias_shape))

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

    def construct(self, input):
        if self.pre_rearrange is not None:
            input = self.pre_rearrange(input)
        result = mint.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result += self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result
