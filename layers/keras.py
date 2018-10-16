from keras.engine import Layer

from layers import RearrangeMixin, ReduceMixin

__author__ = 'Alex Rogozhnikov'


class Rearrange(RearrangeMixin, Layer):
    def compute_output_shape(self, input_shape):
        input_shape = tuple(None if d is None else int(d) for d in input_shape)
        init_shapes, reduced_axes, axes_reordering, final_shapes = self.recipe().reconstruct_from_shape(input_shape)
        return final_shapes

    def call(self, inputs):
        return self._apply_recipe(inputs)

    def get_config(self):
        return {'pattern': self.pattern, **self.axes_lengths}


class Reduce(ReduceMixin, Layer):
    def compute_output_shape(self, input_shape):
        input_shape = tuple(None if d is None else int(d) for d in input_shape)
        init_shapes, reduced_axes, axes_reordering, final_shapes = self.recipe().reconstruct_from_shape(input_shape)
        return final_shapes

    def call(self, inputs):
        return self._apply_recipe(inputs)

    def get_config(self):
        return {'pattern': self.pattern, 'reduction': self.reduction, **self.axes_lengths}


keras_custom_objects = {Rearrange.__name__: Rearrange, Reduce.__name__: Reduce}