__author__ = 'Alex Rogozhnikov'

from einops import TransformRecipe, _prepare_transformation_recipe
import functools


# TODO tests for serialization / deserialization inside the model
# TODO docstrings
# TODO make imports like from einops.torch import ...

class TransposeMixin:

    def __init__(self, pattern, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths
        self.recipe()  # checking parameters

    def __repr__(self):
        params = repr(self.pattern)
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)

    @functools.lru_cache(maxsize=1024)
    def recipe(self) -> TransformRecipe:
        hashable_lengths = tuple(sorted(self.axes_lengths.items()))
        return _prepare_transformation_recipe(self.pattern, reduction='none', axes_lengths=hashable_lengths)


class ReduceMixin:
    def __init__(self, pattern, reduction, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.reduction = reduction
        self.axes_lengths = axes_lengths
        self.recipe()  # checking parameters

    def __repr__(self):
        params = '{!r}, {!r}'.format(self.pattern, self.reduction)
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)

    @functools.lru_cache(maxsize=1024)
    def recipe(self) -> TransformRecipe:
        hashable_lengths = tuple(sorted(self.axes_lengths.items()))
        return _prepare_transformation_recipe(self.pattern, reduction=self.reduction, axes_lengths=hashable_lengths)


import torch


class TorchTranspose(TransposeMixin, torch.nn.Module):
    def forward(self, input):
        return self.recipe().apply(input)


class TorchReduce(ReduceMixin, torch.nn.Module):
    def forward(self, input):
        return self.recipe().apply(input)


import chainer


class ChainerTranspose(TransposeMixin, chainer.Link):
    def __call__(self, x):
        return self.recipe().apply(x)


class ChainerReduce(ReduceMixin, chainer.Link):
    def __call__(self, x):
        return self.recipe().apply(x)


import mxnet


class GluonTranspose(TransposeMixin, mxnet.gluon.Block):
    def forward(self, x):
        return self.recipe().apply(x)


class GluonReduce(ReduceMixin, mxnet.gluon.Block):
    def forward(self, x):
        return self.recipe().apply(x)
