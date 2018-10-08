__author__ = 'Alex Rogozhnikov'

import functools

from einops import transpose, TransformRecipe, _prepare_transformation_recipe, EinopsError


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
        try:
            hashable_lengths = tuple(sorted(self.axes_lengths.items()))
            return _prepare_transformation_recipe(self.pattern, reduction='none', axes_lengths=hashable_lengths)
        except EinopsError as e:
            raise EinopsError(' Error while preparing {!r}\n {}'.format(self, e))

    def _apply_recipe(self, x):
        try:
            return self.recipe().apply(x)
        except EinopsError as e:
            raise EinopsError(' Error while computing {!r}\n {}'.format(self, e))


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
        try:
            hashable_lengths = tuple(sorted(self.axes_lengths.items()))
            return _prepare_transformation_recipe(self.pattern, reduction=self.reduction, axes_lengths=hashable_lengths)
        except EinopsError as e:
            raise EinopsError(' Error while preparing {!r}\n {}'.format(self, e))

    def _apply_recipe(self, x):
        try:
            return self.recipe().apply(x)
        except EinopsError as e:
            raise EinopsError(' Error while computing {!r}\n {}'.format(self, e))


import torch


class TorchTranspose(TransposeMixin, torch.nn.Module):
    def forward(self, input):
        return self._apply_recipe(input)


class TorchReduce(ReduceMixin, torch.nn.Module):
    def forward(self, input):
        return self._apply_recipe(input)


import chainer


class ChainerTranspose(TransposeMixin, chainer.Link):
    def __call__(self, x):
        return self._apply_recipe(x)


class ChainerReduce(ReduceMixin, chainer.Link):
    def __call__(self, x):
        return self._apply_recipe(x)


import mxnet


# TODO symbolic is not working right now

class GluonTranspose(TransposeMixin, mxnet.gluon.HybridBlock):
    def hybrid_forward(self, F, x):
        return self._apply_recipe(x)


class GluonReduce(ReduceMixin, mxnet.gluon.HybridBlock):
    def hybrid_forward(self, F, x):
        return self._apply_recipe(x)


from keras.engine.topology import Layer


class KerasTranspose(TransposeMixin, Layer):
    def compute_output_shape(self, input_shape):
        input_shape = tuple(None if d is None else int(d) for d in input_shape)
        init_shapes, reduced_axes, axes_reordering, final_shapes = self.recipe().reconstruct_from_shape(input_shape)
        return final_shapes

    def call(self, inputs):
        return self._apply_recipe(inputs)

    def get_config(self):
        return {'pattern': self.pattern, **self.axes_lengths}


class KerasReduce(ReduceMixin, Layer):
    def compute_output_shape(self, input_shape):
        input_shape = tuple(None if d is None else int(d) for d in input_shape)
        init_shapes, reduced_axes, axes_reordering, final_shapes = self.recipe().reconstruct_from_shape(input_shape)
        return final_shapes

    def call(self, inputs):
        return self._apply_recipe(inputs)

    def get_config(self):
        return {'pattern': self.pattern, 'reduction': self.reduction, **self.axes_lengths}


keras_custom_objects = {'KerasTranspose': KerasTranspose, 'KerasReduce': KerasReduce}
