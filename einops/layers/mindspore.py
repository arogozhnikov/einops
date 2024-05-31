from typing import Optional, Dict, cast

import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Uniform

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin

__author__ = "Yufeng Lyu"


class Rearrange(RearrangeMixin, nn.Cell):
    def construct(self, input):
        return self._apply_recipe(input)


class Reduce(ReduceMixin, nn.Cell):
    def construct(self, input):
        return self._apply_recipe(input)


class EinMix(_EinmixMixin, nn.Cell):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = mindspore.Parameter(initializer(Uniform(weight_bound), weight_shape), requires_grad=True)
        if bias_shape is not None:
            self.bias = mindspore.Parameter(initializer(Uniform(weight_bound), bias_shape), requires_grad=True)
        else:
            self.bias = None

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
        result = ops.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result += self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result
