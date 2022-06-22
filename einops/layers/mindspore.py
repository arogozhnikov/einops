from typing import List, Optional, Dict

import mindspore as ms
from mindspore.common.initializer import initializer, Uniform

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin


__author__ = 'lvyufeng'


class Rearrange(RearrangeMixin, ms.nn.Cell):
    def construct(self, inputs):
        return self._apply_recipe(inputs)


class Reduce(ReduceMixin, ms.nn.Cell):
    def construct(self, inputs):
        return self._apply_recipe(inputs)


class EinMix(_EinmixMixin, ms.nn.Cell):
    def __init__(self, pattern, weight_shape, bias_shape=None, **axes_lengths):
        super().__init__(pattern, weight_shape, bias_shape, **axes_lengths)

        self.ignore_last = False
        split_pattern = self.einsum_pattern.split('->')
        if split_pattern[0].endswith(','):
            self.einsum_pattern = split_pattern[0][:-1] + '->' + split_pattern[1]
            self.ignore_last = True
        self.einsum = ms.ops.Einsum(self.einsum_pattern)

    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = ms.Parameter(initializer(Uniform(weight_bound), weight_shape), name='weight')
        if bias_shape is not None:
            self.bias = ms.Parameter(initializer(Uniform(bias_bound), bias_shape), name='bias')
        else:
            self.bias = None

    def _create_rearrange_layers(self,
                                 pre_reshape_pattern: Optional[str],
                                 pre_reshape_lengths: Optional[Dict],
                                 post_reshape_pattern: Optional[str],
                                 post_reshape_lengths: Optional[Dict],
                                 ):
        self.pre_rearrange = None
        if pre_reshape_pattern is not None:
            self.pre_rearrange = Rearrange(pre_reshape_pattern, **pre_reshape_lengths)

        self.post_rearrange = None
        if post_reshape_pattern is not None:
            self.post_rearrange = Rearrange(post_reshape_pattern, **post_reshape_lengths)

    def construct(self, inputs):
        if self.pre_rearrange is not None:
            inputs = self.pre_rearrange(inputs)
        if not self.ignore_last:
            result = self.einsum((inputs, self.weight))
        else:
            result = self.einsum((inputs,))
        if self.bias is not None:
            result = result + self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result
