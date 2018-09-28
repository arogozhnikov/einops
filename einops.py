import itertools
from typing import Tuple, List, Set
from collections import OrderedDict

import numpy
# TODO get rid of these mandatory imports
import torch
import mxnet
import tensorflow as tf
import cupy
import chainer

CompositeAxis = List[str]


def reduce_axes(tensor, reduction_type, reduced_axes):
    reduced_axes = tuple(reduced_axes)
    if reduction_type == 'none' or len(reduced_axes) == 0:
        return tensor
    assert reduction_type in ['min', 'max', 'sum', 'mean', 'logaddexp', 'prod']
    if reduction_type == 'mean':
        # TODO check that dtype is float or double.
        raise NotImplementedError()
    if isinstance(tensor, (numpy.ndarray, mxnet.ndarray.ndarray.NDArray, cupy.ndarray)):
        return getattr(tensor, reduction_type)(axis=reduced_axes)
    elif isinstance(tensor, tf.Tensor):
        return getattr(tf, 'reduce_' + reduction_type)(tensor, axis=reduced_axes)
    elif isinstance(tensor, chainer.Variable):
        return getattr(chainer.functions, reduction_type)(tensor, axis=reduced_axes)
    elif isinstance(tensor, torch.Tensor):
        for axis in sorted(reduced_axes, reverse=True):
            if reduction_type == 'min':
                tensor, _ = tensor.min(dim=axis)
            elif reduction_type == 'max':
                tensor, _ = tensor.max(dim=axis)
            elif reduction_type == 'sum':
                tensor = tensor.sum(dim=axis)
            else:
                raise NotImplementedError()
        return tensor
    else:
        raise NotImplementedError()


class TransformRecipe:
    def __init__(self,
                 elementary_axes_lengths: List,
                 # list of expressions (or just sizes) for elementary axes as they appear in left expression
                 input_composite_axes: List[Tuple[List[int], List[int]]],
                 # each dimension in input can help to reconstruct or verify one of dimensions
                 output_composite_axes: List[List[int]],  # ids of axes as they appear in result
                 reduction_type: str = 'none',
                 reduced_elementary_axes: List[int] = (),
                 ellipsis_positions: Tuple[int, int] = (1000, 1000),
                 ):
        self.axes_lengths = elementary_axes_lengths
        self.input_composite_axes = input_composite_axes
        self.output_composite_axes = output_composite_axes
        self.final_axes_grouping_flat = list(itertools.chain(*output_composite_axes))
        self.reduction_type = reduction_type
        self.reduced_elementary_axes = reduced_elementary_axes
        self.ellipsis_positions = ellipsis_positions

    def reconstruct_from_shape(self, shape):
        axes_lengths = list(self.axes_lengths)
        if self.ellipsis_positions != (1000, 1000):
            assert len(shape) >= len(self.input_composite_axes) - 1
        else:
            assert len(shape) == len(self.input_composite_axes)
        for input_axis, (known_axes, unknown_axes) in enumerate(self.input_composite_axes):
            before_ellipsis = input_axis
            after_ellipsis = input_axis + len(shape) - len(self.input_composite_axes)
            if input_axis == self.ellipsis_positions[0]:
                assert len(known_axes) == 0
                unknown_axis, = unknown_axes
                ellipsis_shape = shape[before_ellipsis:after_ellipsis + 1]
                axes_lengths[unknown_axis] = numpy.prod(ellipsis_shape, dtype=int)
            else:
                if input_axis < self.ellipsis_positions[0]:
                    length = shape[before_ellipsis]
                else:
                    length = shape[after_ellipsis]
                known_product = 1
                for axis in known_axes:
                    known_product *= axes_lengths[axis]

                if len(unknown_axes) == 0:
                    if isinstance(length, int) and isinstance(known_product, int):
                        assert length == known_product
                else:
                    if isinstance(length, int) and isinstance(known_product, int):
                        assert length % known_product == 0
                    unknown_axis, = unknown_axes
                    axes_lengths[unknown_axis] = length // known_product

        init_shapes = axes_lengths
        reduced_axes_lengths = [dim for i, dim in enumerate(axes_lengths) if i not in self.reduced_elementary_axes]
        final_shapes = []
        for output_axis, grouping in enumerate(self.output_composite_axes):
            if output_axis == self.ellipsis_positions[1]:
                final_shapes.extend(ellipsis_shape)
            else:
                group_length = 1
                for elementary_axis in grouping:
                    group_length = group_length * reduced_axes_lengths[elementary_axis]
                final_shapes.append(group_length)
        return init_shapes, final_shapes

    def reduce(self, tensor):
        return reduce_axes(tensor, reduction_type=self.reduction_type, reduced_axes=self.reduced_elementary_axes)

    def apply(self, tensor):
        if isinstance(tensor, (numpy.ndarray, mxnet.ndarray.ndarray.NDArray, cupy.ndarray, chainer.Variable)):
            init_shapes, final_shapes = self.reconstruct_from_shape(tensor.shape)
            return self.reduce(tensor.reshape(init_shapes)) \
                .transpose(self.final_axes_grouping_flat).reshape(final_shapes)
        elif isinstance(tensor, torch.Tensor):
            init_shapes, final_shapes = self.reconstruct_from_shape(tensor.shape)
            return self.reduce(tensor.reshape(init_shapes)) \
                .permute(self.final_axes_grouping_flat).reshape(final_shapes)
        elif isinstance(tensor, (tf.Tensor, tf.Variable)):
            init_shapes, final_shapes = self.reconstruct_from_shape(tf_get_shape(tensor))
            tensor = self.reduce(tf.reshape(tensor, init_shapes))
            tensor = tf.transpose(tensor, self.final_axes_grouping_flat)
            return tf.reshape(tensor, final_shapes)
        else:
            raise NotImplementedError('Type of tensor was not recognized')


def parse_expression(expression: str) -> Tuple[Set[str], List[CompositeAxis]]:
    """
    Parses an indexing expression (for a single tensor).
    Checks uniqueness of names, checks usage of '...'
    Returns set of all used identifiers and a list of axis groups
    """
    identifiers = set()
    composite_axes = []
    if '.' in expression:
        assert ('...' in expression) and (str.count(expression, '...') == 1) and (str.count(expression, '.') == 3)
        expression = expression.replace('...', '.')

    bracket_group = None

    def add_axis_name(x):
        if x is not None:
            if x in identifiers:
                raise ValueError('Indexing expression contains duplicate dimension "{x}"')
            identifiers.add(x)
            if bracket_group is None:
                composite_axes.append([x])
            else:
                bracket_group.append(x)

    current_identifier = None
    for char in expression:
        if char in '(). ':
            add_axis_name(current_identifier)
            current_identifier = None
            if char == '.':
                assert bracket_group is None
                add_axis_name('...')
            elif char == '(':
                assert bracket_group is None
                bracket_group = []
            elif char == ')':
                assert bracket_group is not None
                composite_axes.append(bracket_group)
                bracket_group = None
        elif '0' <= char <= '9':
            assert current_identifier is not None
            current_identifier += char
        elif 'a' <= char <= 'z':
            if current_identifier is None:
                current_identifier = char
            else:
                current_identifier += char
        else:
            raise RuntimeError("Unknown character '{}'".format(char))

    if bracket_group is not None:
        raise ValueError('Imbalanced parentheses in expression: "{}"'.format(expression))
    add_axis_name(current_identifier)
    return identifiers, composite_axes


def get_axes_names(composite_axis_name: str):
    axes_names = [axis for axis in composite_axis_name.split(' ') if len(axis) > 0]
    for axis in axes_names:
        if axis == '_':
            continue
        assert 'a' <= axis[0] <= 'z'
        for letter in axis:
            assert str.isdigit(letter) or 'a' <= letter <= 'z'
    return axes_names


def reduce(tensor, pattern, operation, **axes_lengths):
    assert operation in ['none', 'min', 'max', 'sum', 'mean', 'prod', 'logaddexp']
    left, right = pattern.split('->')
    # checking that both have similar letters
    identifiers_left, composite_axes_left = parse_expression(left)
    identifiers_rght, composite_axes_rght = parse_expression(right)

    # TODO add dots
    if operation == 'none':
        difference = set.symmetric_difference(identifiers_left, identifiers_rght)
        if len(difference) > 0:
            raise RuntimeError('Identifiers were only one side of expression: {}'.format(difference))
    else:
        difference = set.difference(identifiers_rght, identifiers_left)
        if len(difference) > 0:
            raise RuntimeError('Unexpected identifiers appeared on the right side of expression: {}'.format(difference))

    # parsing all dimensions to find out lengths
    known_lengths = OrderedDict()
    position_lookup = {}
    position_lookup_after_reduction = {}
    reduced_axes = []
    for composite_axis in composite_axes_left:
        for axis in composite_axis:
            position_lookup[axis] = len(position_lookup)
            if axis in identifiers_rght:
                position_lookup_after_reduction[axis] = len(position_lookup_after_reduction)
            else:
                reduced_axes.append(len(known_lengths))
            known_lengths[axis] = None

    def update_axis_length(axis_name, axis_length):
        if known_lengths[axis_name] is not None:
            # TODO add check for static graphs?
            if isinstance(axis_length, int) and isinstance(known_lengths[axis_name], int):
                assert axis_length == known_lengths[axis_name]
        else:
            known_lengths[axis_name] = axis_length

    for axis, axis_length in axes_lengths.items():
        # TODO better name validation
        elementary_axes = get_axes_names(axis)
        assert len(elementary_axes) == 1
        update_axis_length(elementary_axes[0], axis_length)

    input_axes_known_unknown = []
    # inferring rest of sizes from arguments
    for composite_axis in composite_axes_left:
        known = {axis for axis in composite_axis if known_lengths[axis] is not None}
        unknown = {axis for axis in composite_axis if known_lengths[axis] is None}
        lookup = dict(zip(list(known_lengths), range(len(known_lengths))))
        assert len(unknown) <= 1
        assert len(unknown) + len(known) == len(composite_axis)
        input_axes_known_unknown.append(([lookup[axis] for axis in known], [lookup[axis] for axis in unknown]))

    result_axes_grouping = [[position_lookup_after_reduction[axis] for axis in composite_axis]
                            for composite_axis in composite_axes_rght]

    ellipsis_left = 1000 if ['...'] not in composite_axes_left else composite_axes_left.index(['...'])
    ellipsis_rght = 1000 if ['...'] not in composite_axes_rght else composite_axes_rght.index(['...'])

    recipe = TransformRecipe(elementary_axes_lengths=list(known_lengths.values()),
                             input_composite_axes=input_axes_known_unknown,
                             output_composite_axes=result_axes_grouping,
                             reduction_type=operation,
                             reduced_elementary_axes=reduced_axes,
                             ellipsis_positions=(ellipsis_left, ellipsis_rght)
                             )

    return recipe.apply(tensor=tensor)


def transpose(tensor, pattern, **axes_lengths):
    return reduce(tensor, pattern, operation='none', **axes_lengths)


def check_shapes(*shapes: List[dict], **lengths):
    for shape in shapes:
        assert isinstance(shape, dict)
        for axis_name, axis_length in shape.items():
            assert isinstance(axis_length, int)
            if axis_name in lengths:
                # TODO static frameworks?
                assert lengths[axis_name] == axis_length
            else:
                lengths[axis_name] = axis_length


def tf_get_shape(x):
    if not tf.executing_eagerly():
        return tf.unstack(tf.shape(x))
    else:
        return x.shape


def parse_shape(x, names: str):
    names = [elementary_axis for elementary_axis in names.split(' ') if len(elementary_axis) > 0]
    shape = x.shape
    if isinstance(x, (tf.Variable, tf.Tensor)) and not tf.executing_eagerly():
        shape = tf.unstack(tf.shape(x))
    assert len(shape) == len(names)
    result = {}
    # TODO framework resolution?
    for axis_name, axis_length in zip(names, shape):
        if axis_name != '_':
            result[axis_name] = axis_length
    return result
