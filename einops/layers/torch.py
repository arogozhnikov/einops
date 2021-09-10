import torch

from . import RearrangeMixin, ReduceMixin
from ._weighted_einsum import WeightedEinsumMixin
from .._torch_specific import apply_for_scriptable_torch

__author__ = 'Alex Rogozhnikov'


class Rearrange(RearrangeMixin, torch.nn.Module):
    def forward(self, input):
        return apply_for_scriptable_torch(self._recipe, input, reduction_type='rearrange')

    def _apply_recipe(self, x):
        # overriding parent method to prevent it's scripting
        pass


class Reduce(ReduceMixin, torch.nn.Module):
    def forward(self, input):
        return apply_for_scriptable_torch(self._recipe, input, reduction_type=self.reduction)

    def _apply_recipe(self, x):
        # overriding parent method to prevent it's scripting
        pass


class WeightedEinsum(WeightedEinsumMixin, torch.nn.Module):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = torch.nn.Parameter(torch.zeros(weight_shape).uniform_(-weight_bound, weight_bound),
                                         requires_grad=True)
        if bias_shape is not None:
            self.bias = torch.nn.Parameter(torch.zeros(bias_shape).uniform_(-bias_bound, bias_bound),
                                           requires_grad=True)
        else:
            self.bias = None

    def forward(self, input):
        result = torch.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result += self.bias
        return result
