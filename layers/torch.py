import torch

from layers import RearrangeMixin, ReduceMixin

__author__ = 'Alex Rogozhnikov'


class Rearrange(RearrangeMixin, torch.nn.Module):
    def forward(self, input):
        return self._apply_recipe(input)


class Reduce(ReduceMixin, torch.nn.Module):
    def forward(self, input):
        return self._apply_recipe(input)