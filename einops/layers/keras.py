from tensorflow.keras.layers import Layer

from .._backends import UnknownSize
from . import RearrangeMixin, ReduceMixin
import warnings

__author__ = 'Alex Rogozhnikov'

warnings.warn(
    'Keras does not exist anymore as an independent framework, '
    'these layers (targeted at old multi-framework keras) will be removed, '
    'please use einops.layers.tensorflow.',
    DeprecationWarning,
)


def _compute_output_shape(self, input_shape):
    input_shape = tuple(UnknownSize() if d is None else int(d) for d in input_shape)
    init_shapes, reduced_axes, axes_reordering, added_axes, final_shape = \
        self.recipe().reconstruct_from_shape(input_shape)
    final_shape = tuple(None if isinstance(d, UnknownSize) else int(d) for d in final_shape)
    return final_shape


class Rearrange(RearrangeMixin, Layer):
    def compute_output_shape(self, input_shape):
        return _compute_output_shape(self, input_shape)

    def call(self, inputs):
        return self._apply_recipe(inputs)

    def get_config(self):
        return {'pattern': self.pattern, **self.axes_lengths}


class Reduce(ReduceMixin, Layer):
    def compute_output_shape(self, input_shape):
        return _compute_output_shape(self, input_shape)

    def call(self, inputs):
        return self._apply_recipe(inputs)

    def get_config(self):
        return {'pattern': self.pattern, 'reduction': self.reduction, **self.axes_lengths}


keras_custom_objects = {Rearrange.__name__: Rearrange, Reduce.__name__: Reduce}
